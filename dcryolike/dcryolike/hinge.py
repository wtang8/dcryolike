"""Conformation as a physical coordinate: a rigid domain on a hinge.

WHY NOT A LINEAR SUBSPACE. In 3DVA/RECOVAR the latent parametrises a linear
combination of DENSITIES. A swinging domain does not move in that model: density
fades out at the old position and appears at the new one. Mass is not conserved,
intermediate latents are structures no molecule could adopt, and a rotation needs
many components to approximate badly. Here the latent is the hinge ANGLE, the
density is generated from it, and mass moves.

THE MODEL. The volume splits into a static core and a domain that rotates rigidly
about an axis through the point p with direction n:

    V_theta(x) = V_core(x) + V_dom(Phi_theta^{-1}(x)),   Phi_theta(x) = p + R_theta (x - p)

THE FORWARD MODEL IS EXACT AND CHEAP. Writing q = R_theta^T k,

    F_theta(k) = F_core(k) + F_dom(q) * exp(2 pi i (q - k) . p / D)

so the domain's contribution to a central slice is a slice of the SAME domain
density at the COMPOSED orientation, times a phase ramp. No new operator: the
existing slice machinery evaluates it, and the conformation costs one extra slice
per latent grid point while the core costs none.

The phase convention matches slicing.slice_nufft: points are in voxel frequency
units and the transform carries exp(-2 pi i k . x / D), so a shift by t voxels
multiplies by exp(-2 pi i k . t / D). `test_hinge.py` checks the whole expression
against a volume deformed and sliced explicitly.
"""

from __future__ import annotations

import numpy as np
import torch

from .slicing import fourier_circles, slice_points, slice_nufft
from .generator import _slice


def rodrigues(axis, theta):
    """Rotation by ``theta`` (radians) about the unit vector ``axis``. (...,3,3)."""
    a = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    th = torch.as_tensor(theta, dtype=a.dtype, device=a.device)
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    z = torch.zeros_like(ax)
    K = torch.stack([torch.stack([z, -az, ay], -1),
                     torch.stack([az, z, -ax], -1),
                     torch.stack([-ay, ax, z], -1)], -2)
    I = torch.eye(3, dtype=a.dtype, device=a.device).expand(K.shape)
    s = torch.sin(th)[..., None, None]; c = torch.cos(th)[..., None, None]
    return I + s * K + (1 - c) * (K @ K)


def hinge_slice(core, dom, pts, axis_point, axis_dir, theta, D=None, backend="nufft"):
    """Central-slice samples of the hinged structure at one conformation.

    core, dom  : (D,D,D) real volumes, the static part and the moving domain
    pts        : (n_t, n_p, 3) slice coordinates in voxel frequency units, as
                 produced by slice_points(fourier_circles(...), radii)
    axis_point : (3,) a point on the hinge axis, in voxel coordinates measured from
                 the volume centre (the same origin the slice coordinates use)
    axis_dir   : (3,) the hinge axis direction (need not be normalised)
    theta      : scalar or (n_t,) rotation angle in radians
    returns    : (n_t, n_p) complex, F_core(k) + F_dom(q) exp(2 pi i (q-k).p / D)
    """
    if D is None:
        D = core.shape[-1]
    dt = core.dtype
    p = torch.as_tensor(axis_point, dtype=dt, device=core.device)
    a = torch.as_tensor(axis_dir, dtype=dt, device=core.device)
    th = torch.as_tensor(theta, dtype=dt, device=core.device)
    R = rodrigues(a, th)                                   # (3,3) or (n_t,3,3)
    if R.dim() == 2:
        q = pts @ R                                        # q = R^T k, since (R^T k)_i = k_j R_ji
    else:
        q = torch.einsum('tpj,tji->tpi', pts, R)
    phase = torch.exp(2j * np.pi * ((q - pts) @ p) / D)
    Fc = _slice(core, pts, backend)
    Fd = _slice(dom, q, backend)
    return Fc + Fd * phase.to(Fd.dtype)


def hinge_slice_bank(core, dom, thetas, th_grid, polars, azimus, gammas, radii,
                     axis_point, axis_dir, D=None, backend="nufft"):
    """Slices for every (pose, conformation) pair: (n_poses, n_theta, n_shells*n_inplane).

    The core term is computed once per pose and reused across conformations, which is
    the whole cost saving: a latent grid of size T costs T domain slices per pose, not
    T full slices.
    """
    if D is None:
        D = core.shape[-1]
    pts = slice_points(fourier_circles(th_grid, polars, azimus, gammas), radii)
    Fc = _slice(core, pts, backend)                        # (n_poses, n_pts)
    out = torch.empty(pts.shape[0], len(thetas), pts.shape[1], dtype=Fc.dtype,
                      device=core.device)
    p = torch.as_tensor(axis_point, dtype=core.dtype, device=core.device)
    a = torch.as_tensor(axis_dir, dtype=core.dtype, device=core.device)
    for t, theta in enumerate(thetas):
        R = rodrigues(a, torch.as_tensor(float(theta), dtype=core.dtype, device=core.device))
        q = pts @ R
        phase = torch.exp(2j * np.pi * ((q - pts) @ p) / D)
        out[:, t] = Fc + _slice(dom, q, backend) * phase.to(Fc.dtype)
    return out


def render_hinged(core, dom_atoms, axis_point, axis_dir, theta, D, sigma_vox=1.4,
                  device="cuda", dtype=torch.float64):
    """Reference renderer: rotate the domain's ATOMS and rasterise, for testing.

    This is the honest comparison for hinge_slice: it builds the deformed structure
    in real space from its atomic positions and never touches the Fourier identity
    the fast path relies on.
    """
    from .generator import _render_atoms
    p = torch.as_tensor(axis_point, dtype=dtype).cpu()
    a = torch.as_tensor(axis_dir, dtype=dtype).cpu()
    R = rodrigues(a, torch.as_tensor(float(theta), dtype=dtype))
    moved = (dom_atoms.cpu() - p) @ R.T + p
    return core + _render_atoms(moved, D, sigma_vox, device, dtype)


# ---------------------------------------------------------------------------
# Joint quadrature over (pose, hinge angle, in-plane angle, gain)
# ---------------------------------------------------------------------------

def hinge_logweights(images, ctf, w, beta, core, dom, pts, thetas, axis_point, axis_dir,
                     D=None, backend="nufft"):
    """Log posterior weight of every (view, hinge angle) for a batch of particles.

    images : (nb, S, P) complex      ctf : (nb, S, P) real      w : (S, P) real
    beta   : (nb,)   the temperature ||y||^2_w / (2 sigma^2 C)
    pts    : (nb, M, S*P, 3)  slice coordinates of each particle's M shortlisted views
    thetas : (T,)    the hinge-angle quadrature grid, radians

    returns  logw (nb, M, T)  the log weight with the in-plane angle MARGINALISED,
             gidx (nb, M, T)  the best in-plane index, for insertion

    The gain is MARGINALISED, not profiled. With a flat prior,

        log p(y | R, theta) = const + beta cc^2 - (1/2) log A,     A = <cT, cT>_w

    and the second term is not optional: A varies with the hinge angle (core and domain
    interfere differently as the domain swings) and between competing hypotheses, so
    dropping it would favour whichever template happens to carry more power. Profiling
    the gain instead of integrating it drops exactly that term, which is harmless for
    ranking angles within one model and wrong for comparing models.

    The in-plane angle is integrated by logsumexp over the ring, using the same FFT
    identity as the pose search. Two approximations, both deliberate:

      * the WEIGHT integrates over the in-plane angle but the INSERTION uses its
        argmax, because inserting at every in-plane angle would cost a factor P. The
        weight is therefore correct and the insertion position is a point estimate
        within one ring step.
      * a uniform prior is assumed over the hinge-angle grid, so `thetas` must be
        evenly spaced for the quadrature to be the plain sum used here.
    """
    if D is None:
        D = core.shape[-1]
    nb, M = pts.shape[0], pts.shape[1]
    S, P = images.shape[-2], images.shape[-1]
    T = len(thetas)
    cdt = torch.complex64 if core.dtype == torch.float32 else torch.complex128
    dev = core.device
    p = torch.as_tensor(axis_point, dtype=core.dtype, device=dev)
    a = torch.as_tensor(axis_dir, dtype=core.dtype, device=dev)

    wc = (w[None] * ctf).to(cdt)                                  # (nb,S,P)
    If = torch.fft.fft((wc * images).reshape(nb, S, P), dim=-1)   # (nb,S,P)
    In = torch.sqrt((images.abs() ** 2 * w[None]).sum(dim=(1, 2)))
    wc2 = (w[None] * ctf ** 2)                                    # (nb,S,P)

    logw = torch.empty(nb, M, T, dtype=core.dtype, device=dev)
    gidx = torch.empty(nb, M, T, dtype=torch.long, device=dev)
    flat = pts.reshape(nb * M, -1, 3)
    Fc = _slice(core, flat, backend).reshape(nb, M, S, P)          # core: independent of theta
    for t in range(T):
        R = rodrigues(a, torch.as_tensor(float(thetas[t]), dtype=core.dtype, device=dev))
        q = flat @ R
        phase = torch.exp(2j * np.pi * ((q - flat) @ p) / D).reshape(nb, M, S, P)
        Tb = Fc + _slice(dom, q, backend).reshape(nb, M, S, P) * phase.to(Fc.dtype)
        Tn = torch.sqrt(((Tb.abs() ** 2) * wc2[:, None]).sum(dim=(2, 3)))     # (nb,M)
        cc = torch.fft.ifft(torch.conj(torch.fft.fft(Tb, dim=-1)) * If[:, None], dim=-1)
        cc = cc.real.sum(2) / (Tn[:, :, None] * In[:, None, None])            # (nb,M,P)
        b = beta[:, None, None].to(cc.dtype)
        logw[:, :, t] = (torch.logsumexp(b * cc ** 2, dim=-1) - float(np.log(P))
                         - torch.log(Tn.clamp(min=1e-30)))
        gidx[:, :, t] = cc.argmax(-1)
        del Tb, cc, q, phase
    return logw, gidx


def hinge_mstep(acc_core, acc_dom, images, ctf, w, pts, gammas_ring, weights, gidx,
                thetas, axis_point, axis_dir, core, dom, tau, D, backend="nufft"):
    """Accumulate one batch into the core and domain volumes.

    The two volumes are coupled: a particle constrains their SUM. The exact normal
    equations have a dense cross term (a slice hits one voxel of the core and, rotated,
    a different voxel of the domain), so this uses the standard block scheme -- the
    same one RELION multi-body uses -- and subtracts the other volume's current
    prediction before inserting:

        core  <- insert  (y - g c F_dom(q) phase)          at k
        dom   <- insert  (y - g c F_core(k)) conj(phase)   at q

    with the gain g at its posterior mean. That is Gauss-Seidel on the coupled system,
    not a single exact solve, so it needs the outer iteration to converge.
    """
    nb, M = pts.shape[0], pts.shape[1]
    S, P = images.shape[-2], images.shape[-1]
    dev = core.device
    p = torch.as_tensor(axis_point, dtype=core.dtype, device=dev)
    a = torch.as_tensor(axis_dir, dtype=core.dtype, device=dev)
    flat = pts.reshape(nb * M, -1, 3)
    Fc_all = _slice(core, flat, backend).reshape(nb, M, S, P)
    for t in range(len(thetas)):
        wt = weights[:, :, t]                                       # (nb,M)
        keep = wt > 1e-4
        if not keep.any():
            continue
        R = rodrigues(a, torch.as_tensor(float(thetas[t]), dtype=core.dtype, device=dev))
        q = flat @ R
        phase = torch.exp(2j * np.pi * ((q - flat) @ p) / D).reshape(nb, M, S, P)
        Fd = _slice(dom, q, backend).reshape(nb, M, S, P)
        Fc = Fc_all
        # roll each image to the chosen in-plane angle rather than rotating the template
        g = gidx[:, :, t]                                           # (nb,M)
        idx = (torch.arange(P, device=dev)[None, None, :] + g[..., None]) % P
        y = torch.gather(images[:, None].expand(nb, M, S, P), 3,
                         idx[:, :, None, :].expand(nb, M, S, P))
        c = torch.gather(ctf[:, None].expand(nb, M, S, P), 3,
                         idx[:, :, None, :].expand(nb, M, S, P))
        wq = (w[None, None] * wt[..., None, None] * tau[:, None, None, None])
        gain = 1.0                                                  # gauge-pinned outside
        res_core = y - gain * c * Fd * phase.to(Fd.dtype)
        res_dom = (y - gain * c * Fc) * torch.conj(phase).to(Fc.dtype)
        sel = keep[..., None].expand(nb, M, S * P).reshape(-1)
        kk = flat.reshape(nb, M, S * P, 3).reshape(-1, 3)
        acc_core.add(kk[sel], (res_core.reshape(-1))[sel], (c.reshape(-1))[sel],
                     (wq.expand(nb, M, S, P).reshape(-1))[sel],
                     torch.ones(int(sel.sum()), 1, dtype=w.dtype, device=dev),
                     torch.ones(int(sel.sum()), 1, 1, dtype=w.dtype, device=dev))
        qq = q.reshape(nb, M, S * P, 3).reshape(-1, 3)
        acc_dom.add(qq[sel], (res_dom.reshape(-1))[sel], (c.reshape(-1))[sel],
                    (wq.expand(nb, M, S, P).reshape(-1))[sel],
                    torch.ones(int(sel.sum()), 1, dtype=w.dtype, device=dev),
                    torch.ones(int(sel.sum()), 1, 1, dtype=w.dtype, device=dev))
        del Fd, phase, q, y, c, res_core, res_dom
