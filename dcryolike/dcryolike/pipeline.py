"""Reusable pieces of the iterative pipeline, lifted out of the analysis scripts.

Every function takes its inputs explicitly (the scripts close over globals).
Semantics are those of tests/iterative_halfset_physical.py, which remains the
reference: build_bank/search_bank reproduce pose_search(), reconstruct()
reproduces the trilinear num/den via HeteroAccumulator with K=0 (verified
identical on CPU in test_hetero TEST 3).
"""

from __future__ import annotations
import numpy as np
import torch

from .slicing import fourier_circles, slice_points, slice_nufft
from .hetero import HeteroAccumulator
from .generator import _slice


# ---- Fourier-shell correlation on a cubic grid --------------------------------

class ShellMasks:
    def __init__(self, D, device):
        ii = torch.arange(D, dtype=torch.float64, device=device) - D // 2
        X, Y, Z = torch.meshgrid(ii, ii, ii, indexing="ij")
        self.RR = (X ** 2 + Y ** 2 + Z ** 2).sqrt()
        self.masks = [(self.RR >= k - 0.5) & (self.RR < k + 0.5) for k in range(1, D // 2)]

    def fsc(self, F1, F2):
        return np.array([((F1[m] * F2[m].conj()).sum() /
                          (F1[m].abs().pow(2).sum().sqrt() * F2[m].abs().pow(2).sum().sqrt() + 1e-30)
                          ).real.item() for m in self.masks])

    def lowpass(self, F, kc, width=2.0):
        t = ((self.RR - kc) / width).clamp(0, 1)
        return F * (0.5 * (1 + torch.cos(np.pi * t)))


def res_at(f, t=0.5):
    for k, v in enumerate(f, start=1):
        if v < t:
            return k
    return len(f)


def to_real(F):
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(F), norm="ortho")).real.contiguous()


# ---- posterior temperature (measured Parseval constant) ------------------------

def temperature(p, w, D):
    """beta_i = ||y_i||^2_w / (2 sigma_i^2 C) and tau_i = 1/(sigma_i^2 C), with C
    measured from the realised noise exactly as the iterative scripts do."""
    In2 = (p.images_fourier.abs() ** 2 * w).sum(dim=(1, 2))
    n = p.images_fourier - p.clean_fourier
    nw = (n.abs() ** 2 * w).sum(dim=(1, 2))
    C = (nw / (D * D * p.sigma_noise ** 2)).mean()
    tau = 1.0 / (p.sigma_noise ** 2 * C)
    return In2 / (2 * p.sigma_noise ** 2 * C), tau, C


# ---- bank search: best normalised CC per view, over all in-plane rotations -----

class Bank:
    def __init__(self, dens, th, cpo, caz, rr, S, P, chunk=4096, backend="nufft"):
        V = caz.shape[0]
        cdt = torch.complex128 if dens.dtype == torch.float64 else torch.complex64
        self.Tf = torch.empty(V, S, P, dtype=cdt, device=dens.device)
        self.Tpow = torch.empty(V, S, dtype=dens.dtype, device=dens.device)
        for i in range(0, V, chunk):
            j = min(i + chunk, V)
            Tb = _slice(dens, slice_points(fourier_circles(th, cpo[i:j], caz[i:j],
                                                           torch.zeros_like(caz[i:j])), rr), backend).reshape(j - i, S, P)
            self.Tf[i:j] = torch.fft.fft(Tb, dim=-1).conj(); self.Tpow[i:j] = (Tb.abs() ** 2).sum(-1)
        self.V, self.P = V, P

    def search(self, images, cr, wr, In, batch=16):
        """images (n,S,P), cr (n,S) isotropic CTF, wr (S,) shell weights, In (n,) image norms.
        Returns best cc (n,V) and the in-plane angle (n,V) of the best gamma."""
        n = images.shape[0]; V, P = self.V, self.P
        Tn = torch.sqrt(torch.einsum('vr,br->bv', self.Tpow, wr[None, :] * cr ** 2))
        bc = torch.empty(n, V, device=images.device, dtype=self.Tpow.dtype)
        bg = torch.empty_like(bc)
        for i in range(0, n, batch):
            j = min(i + batch, n)
            If = torch.fft.fft(images[i:j], dim=-1); cw = (wr[None, :] * cr[i:j])[:, :, None]
            cc = torch.fft.ifft(torch.einsum('vrk,brk->bvk', self.Tf, If * cw), dim=-1).real
            cc = cc / (Tn[i:j, :, None] * In[i:j, None, None])
            m, gi = cc.max(-1); bc[i:j] = m; bg[i:j] = -(gi.to(bc.dtype)) * (2 * np.pi / P)
        return bc, bg


def neighbourhood(caz, cpo, deg):
    nd = torch.stack([torch.sin(cpo) * torch.cos(caz), torch.sin(cpo) * torch.sin(caz), torch.cos(cpo)], -1)
    return (nd @ nd.T) >= np.cos(np.radians(deg)), nd


def select_modes(bc, NB, modes, beta):
    """Local maxima of cc over the view sphere (within NB), top `modes` of them,
    softmax weights exp(beta (cc^2 - cc_best^2)). Returns indices (n,modes), weights."""
    V = bc.shape[1]
    nbmax = torch.stack([bc[:, NB[v]].max(1).values for v in range(V)], 1)
    masked = bc.masked_fill(~(bc >= nbmax - 1e-12), -2.0)
    tcc, tidx = masked.topk(modes, dim=1)
    lg = (beta[:, None] * (tcc ** 2 - tcc[:, :1] ** 2)).masked_fill(tcc < -1.0, -1e30)
    return tidx, torch.softmax(lg, dim=1), tcc


# ---- reconstruction ------------------------------------------------------------

def reconstruct(p, ids, sel_idx, sel_w, gam, th, cpo, caz, rr, w, D, lam=1e-3, B=250,
                K=0, Ea=None, Eaa=None, tau=None):
    """Mixture insertion. With K=0 this is the homogeneous num/den; with K>0 it
    is the per-voxel (K+1)x(K+1) solve, Ea/Eaa (n,modes,K+1[,K+1]) the posterior
    moments of the coefficients and tau (n,) the noise precision (weights the
    particles; None = 1). Returns (K+1,D,D,D) complex Fourier volumes."""
    dev = p.images_fourier.device
    acc = HeteroAccumulator.new(D, K + 1, dev)
    n = len(ids); SP = w.numel()
    for i in range(0, n, B):
        j = min(i + B, n); loc = torch.arange(i, j, device=dev); gid = ids[loc]
        for m in range(sel_idx.shape[1]):
            v = sel_idx[loc, m]; ww = sel_w[loc, m]; keep = ww > 1e-4
            if keep.sum() == 0:
                continue
            g_ = gid[keep]; l_ = loc[keep]; nk = len(g_)
            pts = slice_points(fourier_circles(th, cpo[v[keep]], caz[v[keep]], gam[l_, v[keep]]), rr).reshape(-1, 3)
            c = p.ctf[g_].reshape(-1)
            wt = ww[keep] * (tau[g_] if tau is not None else 1.0)
            wg = (w.reshape(1, -1) * wt[:, None]).reshape(-1)
            y = p.images_fourier[g_].reshape(-1)
            if K == 0:
                ea = torch.ones(nk * SP, 1, dtype=w.dtype, device=dev); eaa = ea[:, :, None]
            else:
                ea = Ea[l_, m].repeat_interleave(SP, 0); eaa = Eaa[l_, m].repeat_interleave(SP, 0)
            acc.add(pts, y, c, wg, ea, eaa)
    return acc.solve(lam)
