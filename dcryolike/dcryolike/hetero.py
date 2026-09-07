"""Heterogeneous reconstruction with a linear latent model and exact marginalisation.

Model
-----
Each particle i is a projection of  V(z) = V0 + sum_k z_k U_k  at pose R_i,
scaled by an unknown gain g_i, seen through its CTF c_i, plus noise that is
white in real space:

    y_i = c_i . S_{R_i}( g V0 + sum_k a_k U_k ) + n_i,      a = (g, a_1..a_K)

All coefficients a are REAL (the volumes are real). The Fourier slice operator
S is linear, so a slice of V(z) is a fixed linear combination of the K+1
slices T_0..T_K, and for a given pose the likelihood is a quadratic form in a:

    -log p(y | R, a)  =  tau/2 ( G_yy - 2 a.b + a^T A a ) + const
        A_jk = Re <T_j, T_k>_{w c^2},   b_j = Re <T_j, y>_{w c},   G_yy = ||y||^2_w

with tau = 1/(sigma_r^2 C) the real-space noise precision expressed in the
weighted polar metric (the same C measured in the iterative scripts). With a
Gaussian prior  a ~ N(0, Lambda^-1)  (flat on the gain, unit on each z_k), a
integrates out in closed form:

    log p(y | R)  =  -tau/2 G_yy + 1/2 m^T M m - 1/2 log det M + const
        M = tau A + Lambda,   m = M^-1 (tau b)    (posterior mean of a)

The log-det is what stops heterogeneity from explaining away a wrong pose: a
pose whose component slices align with the noise buys a larger quadratic term
but pays for it in volume.

Every Gram entry is a circular correlation along the in-plane angle, so ALL
in-plane rotations come out of one FFT pair -- the same trick the per-gamma
Isx uses in ill.py. Heterogeneity therefore enters AFTER the pose shortlist:
the shortlist is built from V0 alone (the recall-ceiling machinery), then the
marginal likelihood is evaluated on the shortlisted views over every gamma.

M-step
------
Given posterior weights over (particle, view, gamma) and the first two moments
of a at each, the volumes minimise the expected weighted residual. The exact
normal equations couple all K+1 volumes through S^T S; the standard gridding
approximation treats S^T S as diagonal in Fourier space, which turns them into
ONE (K+1)x(K+1) linear system PER VOXEL:

    [ sum  w c^2 E[a_j a_k] g_ins ] V_k(x)  =  sum  w E[a_j] c y g_ins

accumulated with the same trilinear splat the homogeneous reconstruction uses.
K=0 reduces exactly to num/den.

Gauge
-----
(V0, U, z) is only defined up to  z -> Q z + mu.  fix_gauge() centres z, then
rotates the basis so the posterior-mean latents are decorrelated with unit
variance and ordered by explained variance (the PCA gauge). The volume family
V(z) is unchanged by construction.

Not implemented (deliberately, first version): an offset regressor (CryoLike's
mask term); anisotropic-CTF Gram terms are supported but cost (K+1)^2 extra FFTs.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch


# --------------------------------------------------------------------------
# Gram matrices over all in-plane rotations
# --------------------------------------------------------------------------

def _corr(a, b):
    """corr(a,b)[gamma] = sum_theta conj(a(theta)) b(theta+gamma), over the last axis."""
    return torch.fft.ifft(torch.conj(torch.fft.fft(a, dim=-1)) * torch.fft.fft(b, dim=-1), dim=-1)


def hetero_gram(y, T, ctf, w):
    """Sufficient statistics of the linear model at every in-plane rotation.

    y   : (n, S, P) complex    images (particle frame)
    T   : (n, M, K+1, S, P)    component slices T_0..T_K at M candidate views,
                               gamma = 0, un-CTF'd
    ctf : (n, S, P) real       CTF of each image (may be anisotropic)
    w   : (S, P) real          quadrature weights
    returns A (n,M,P,K+1,K+1) real, b (n,M,P,K+1) real, Gyy (n,) real

    The convention matches the existing pose search: cc(gamma) =
    ifft(conj(fft T) . fft(w c y))[gamma] is the image correlated with the
    template rotated by gamma, and the best gamma maps to the angle
    -gamma_index * 2 pi / P.
    """
    n, M, K1, S, P = T.shape
    wc = (w[None] * ctf).to(y.dtype)                                    # (n,S,P)
    # Same convention as the pose search: index gi along the correlation axis is
    # the image rotated by gi steps relative to the template, i.e. template
    # in-plane angle gamma = -gi * 2 pi / P.
    #   b_j(gi) = Re sum_r sum_th conj(T_j(th)) (w c y)(th+gi) = Re corr(T_j, w c y)
    b = _corr(T, (wc * y)[:, None, None]).sum(dim=3).real               # (n,M,K+1,P)
    b = b.permute(0, 1, 3, 2).contiguous()                              # (n,M,P,K+1)
    #   A_jk(gi) = Re sum_r sum_th (w c^2)(th+gi) conj(T_j) T_k (th) = Re corr(conj(TT), w c^2)
    wc2 = (w[None] * ctf ** 2)
    iso = bool(torch.allclose(wc2, wc2[..., :1].expand_as(wc2)))
    TT = torch.conj(T[:, :, :, None]) * T[:, :, None, :]                # (n,M,K+1,K+1,S,P)
    if iso:
        A = (wc2[:, None, None, None, :, :1] * TT).sum(dim=(-2, -1)).real   # (n,M,K+1,K+1)
        A = A[:, :, None].expand(n, M, P, K1, K1).contiguous()
    else:
        A = _corr(torch.conj(TT), wc2.to(y.dtype)[:, None, None, None]).sum(dim=4).real
        A = A.permute(0, 1, 4, 2, 3).contiguous()                       # (n,M,P,K+1,K+1)
    Gyy = (y.abs() ** 2 * w[None]).sum(dim=(1, 2))
    return A, b, Gyy


def marginal_loglik(A, b, Gyy, tau, prior_prec):
    """Integrate the real coefficients out of the Gaussian likelihood.

    A          : (..., K+1, K+1)   b : (..., K+1)   Gyy : broadcastable to (...)
    tau        : (n,) or scalar    noise precision in the weighted polar metric
    prior_prec : (K+1,)            diagonal prior precision; 0 = flat (gain)
    returns    logml (...), mean (..., K+1), cov (..., K+1, K+1)

    Constants independent of pose are dropped EXCEPT the prior normalisation
    over the proper (non-flat) directions, so values are comparable across
    different K only up to that flat-direction constant.
    """
    nb = A.dim() - 2                                   # number of batch dims
    tau = torch.as_tensor(tau, dtype=A.dtype, device=A.device)
    while tau.dim() < nb:                              # (n,) -> (n,1,...,1): leading batch axis
        tau = tau[..., None]
    Mm = tau[..., None, None] * A + torch.diag(prior_prec.to(A.dtype))
    L, info = torch.linalg.cholesky_ex(Mm)
    tb = tau[..., None] * b
    mean = torch.cholesky_solve(tb[..., None], L)[..., 0]
    logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    Gyy = torch.as_tensor(Gyy, dtype=A.dtype, device=A.device)
    while Gyy.dim() < nb:
        Gyy = Gyy[..., None]
    pp = prior_prec[prior_prec > 0]
    prior_norm = 0.5 * torch.log(pp).sum()
    logml = -0.5 * tau * Gyy + 0.5 * (tb * mean).sum(-1) - 0.5 * logdet + prior_norm
    cov = torch.cholesky_inverse(L)
    logml = torch.where(info == 0, logml, torch.full_like(logml, -1e30))
    return logml, mean, cov


# --------------------------------------------------------------------------
# M-step: per-voxel (K+1)x(K+1) normal equations with trilinear splatting
# --------------------------------------------------------------------------

def insert_trilinear(pts, vals, D, out):
    """Splat ``vals`` (..., n) at ``pts`` (n, 3) into flat volumes out (..., D^3),
    trilinear weights, points outside the box dropped. Adjoint of trilinear
    interpolation; identical arithmetic to the homogeneous scripts' insert()."""
    x = pts + D // 2
    ok = ((x >= 0) & (x <= D - 1)).all(-1)
    x = x[ok]; vals = vals[..., ok]
    f = torch.floor(x).long(); fr = x - f
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                g = ((1 - fr[:, 0]) if dx == 0 else fr[:, 0]) * \
                    ((1 - fr[:, 1]) if dy == 0 else fr[:, 1]) * \
                    ((1 - fr[:, 2]) if dz == 0 else fr[:, 2])
                i0 = (f[:, 0] + dx).clamp(0, D - 1); i1 = (f[:, 1] + dy).clamp(0, D - 1)
                i2 = (f[:, 2] + dz).clamp(0, D - 1)
                fl = (i0 * D + i1) * D + i2
                out.index_add_(-1, fl, vals * g.to(vals.dtype))


@dataclass
class HeteroAccumulator:
    """Running numerator (K+1, D^3) and denominator (K+1, K+1, D^3) sums."""
    D: int
    K1: int
    num: torch.Tensor
    den: torch.Tensor

    @classmethod
    def new(cls, D, K1, device, cdtype=torch.complex128, rdtype=torch.float64):
        return cls(D, K1, torch.zeros(K1, D ** 3, dtype=cdtype, device=device),
                   torch.zeros(K1, K1, D ** 3, dtype=rdtype, device=device))

    def add(self, pts, y, c, w, Ea, Eaa):
        """A batch of (particle, pose) contributions, flattened to points.
        pts (n,3)      Fourier points, particle frame rotated to the pose
        y   (n,)       image values      c (n,) CTF      w (n,) quadrature x posterior x tau
        Ea  (n,K+1)    E[a] per point    Eaa (n,K+1,K+1) E[a a^T] per point
        (per-particle moments are expanded to per-point by the caller)
        """
        # the stack may be single precision while the accumulator is double: summing
        # millions of terms is where precision actually matters, so the values are cast
        # up here rather than the accumulator being cast down
        num_v = ((w * c * y)[None] * Ea.T.to(y.dtype)).to(self.num.dtype)
        den_v = ((w * c ** 2)[None, None] * Eaa.permute(1, 2, 0)).to(self.den.dtype)
        insert_trilinear(pts.to(self.den.dtype), num_v, self.D, self.num)
        insert_trilinear(pts.to(self.den.dtype), den_v, self.D, self.den)

    def solve(self, lam=1e-3):   # returns the accumulator's dtype (float64 by default)
        """Per-voxel solve, ridge lam x mean diagonal. Returns (K+1, D, D, D) complex."""
        den = self.den.permute(2, 0, 1).clone()                          # (D^3,K+1,K+1)
        diag = torch.diagonal(den, dim1=-2, dim2=-1)
        ridge = lam * diag.mean(dim=0)                                   # per component
        # voxels nothing touched have a zero row: give them a unit pivot so the
        # solve is well posed (their numerator is zero, so they come out zero)
        floor = (diag <= 0).to(den.dtype)
        den += torch.diag_embed(ridge[None].expand_as(diag) + floor)
        num = self.num.permute(1, 0)                                     # (D^3,K+1)
        sol = torch.linalg.solve(den.to(num.dtype), num[..., None])[..., 0]
        return sol.permute(1, 0).reshape(self.K1, self.D, self.D, self.D)


# --------------------------------------------------------------------------
# Gauge fixing and diagnostics
# --------------------------------------------------------------------------

def fix_gauge(V0, U, z_mean, z_cov=None):
    """PCA gauge. z_mean (n,K) posterior means, z_cov (n,K,K) posterior covariances
    (optional; adds the within-particle covariance so the fitted prior is unit).
    Returns V0', U', z', and the transform (Q, mu) with z' = Q (z - mu)."""
    K = U.shape[0]
    mu = z_mean.mean(0)
    V0n = V0 + torch.einsum('k,kxyz->xyz', mu.to(U.dtype), U)
    zc = z_mean - mu
    Sig = zc.T @ zc / zc.shape[0]
    if z_cov is not None:
        Sig = Sig + z_cov.mean(0)
    evals, evecs = torch.linalg.eigh(Sig)                                # ascending
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order].clamp(min=1e-12), evecs[:, order]
    # z' = D^-1/2 E^T (z - mu)  =>  U' = U E D^1/2  so that U z = U' z'
    Q = torch.diag(evals.rsqrt()) @ evecs.T
    Un = torch.einsum('jk,jxyz->kxyz', (evecs * evals.sqrt()[None]).to(U.dtype), U)
    zn = zc @ Q.T
    return V0n, Un, zn, (Q, mu)


def rigid_tangents(V):
    """Six volumes spanning first-order rigid motions of V: 3 translations
    (dV/dx_i) and 3 rotations ((r x grad V)_i). Central differences, real space."""
    D = V.shape[-1]
    gx, gy, gz = torch.gradient(V)
    ax = torch.arange(D, dtype=V.dtype, device=V.device) - D // 2
    X, Y, Z = torch.meshgrid(ax, ax, ax, indexing="ij")
    rot = [Y * gz - Z * gy, Z * gx - X * gz, X * gy - Y * gx]
    return torch.stack([gx, gy, gz] + rot, 0)


def rotation_leakage(V0, U):
    """Fraction of each component's energy lying in the span of rigid-body
    tangents of V0. A component that is mostly a rotation of V0 is a pose error
    wearing a heterogeneity costume. Returns (K,) fractions in [0,1]."""
    B = rigid_tangents(V0).reshape(6, -1).T                              # (D^3, 6)
    Uf = U.reshape(U.shape[0], -1).T                                     # (D^3, K)
    coef = torch.linalg.lstsq(B, Uf).solution
    proj = B @ coef
    return (proj ** 2).sum(0) / (Uf ** 2).sum(0).clamp(min=1e-30)


def rank_correlation(x, y):
    """Spearman rank correlation of two 1-D tensors."""
    rx = x.argsort().argsort().to(torch.float64)
    ry = y.argsort().argsort().to(torch.float64)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return ((rx * ry).sum() / (rx.norm() * ry.norm()).clamp(min=1e-30)).item()


def latent_r2(z, t):
    """R^2 of the best linear predictor of the true latent t (n,) from z (n,K)."""
    Zd = torch.cat([z, torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)], 1)
    coef = torch.linalg.lstsq(Zd, t[:, None].to(Zd.dtype)).solution
    resid = t.to(Zd.dtype) - (Zd @ coef)[:, 0]
    return (1 - resid.var() / t.var().clamp(min=1e-30)).item()


def align_bases(U_ref, U_other):
    """Orthogonal Procrustes: Q (K,K) with Q @ U_other ~ U_ref (flattened bases).
    Use it to compare two independently fitted half-set models."""
    A = U_ref.reshape(U_ref.shape[0], -1); B = U_other.reshape(U_other.shape[0], -1)
    W, _, Vh = torch.linalg.svd(A @ B.T)
    Q = W @ Vh
    return Q, torch.einsum('jk,kxyz->jxyz', Q, U_other)
