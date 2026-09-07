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


# ---- GPU duty cycle -------------------------------------------------------------
# The heavy loops synchronise and sleep so the GPU averages GPU_DUTY of the wall
# clock (default 0.8: the machine is shared). Pausing the process from outside
# does not work -- torch queues seconds of kernels ahead of the CPU thread.
import os as _os, time as _time
GPU_DUTY = float(_os.environ.get("GPU_DUTY", "0.8"))


def duty_sleep(t0):
    """Call after a GPU-heavy block started at monotonic time t0; returns a new t0.
    (monotonic: the WSL wall clock steps backwards under NTP corrections)"""
    if GPU_DUTY < 1.0 and torch.cuda.is_available():
        torch.cuda.synchronize()
        busy = max(0.0, _time.monotonic() - t0)
        _time.sleep(busy * (1.0 / GPU_DUTY - 1.0))
    return _time.monotonic()


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
    nw = getattr(p, "noise_norm_w", None)
    if nw is None:                       # noiseless stack retained: recover the noise from it
        nw = ((p.images_fourier - p.clean_fourier).abs() ** 2 * w).sum(dim=(1, 2))
    C = (nw / (D * D * p.sigma_noise ** 2)).mean()
    tau = 1.0 / (p.sigma_noise ** 2 * C)
    return In2 / (2 * p.sigma_noise ** 2 * C), tau, C


# ---- bank search: best normalised CC per view, over all in-plane rotations -----

class Bank:
    """FFT-along-theta template bank. ``single=True`` stores it in complex64 and
    searches in float32 (what the D=96 recall scripts do; halves the memory)."""
    def __init__(self, dens, th, cpo, caz, rr, S, P, chunk=2048, backend="nufft", single=False):
        V = caz.shape[0]
        cdt = torch.complex64 if (single or dens.dtype != torch.float64) else torch.complex128
        rdt = torch.float32 if cdt == torch.complex64 else torch.float64
        self.Tf = torch.empty(V, S, P, dtype=cdt, device=dens.device)
        self.Tpow = torch.empty(V, S, dtype=rdt, device=dens.device)
        t0 = _time.monotonic()
        for i in range(0, V, chunk):
            j = min(i + chunk, V)
            Tb = _slice(dens, slice_points(fourier_circles(th, cpo[i:j], caz[i:j],
                                                           torch.zeros_like(caz[i:j])), rr), backend).reshape(j - i, S, P).to(cdt)
            self.Tf[i:j] = torch.fft.fft(Tb, dim=-1).conj(); self.Tpow[i:j] = (Tb.abs() ** 2).sum(-1)
            del Tb; t0 = duty_sleep(t0)
        self.V, self.P, self.cdt, self.rdt = V, P, cdt, rdt

    def search(self, images, cr, wr, In, batch=None):
        """images (n,S,P), cr (n,S) isotropic CTF, wr (S,) shell weights, In (n,) image norms.
        Returns best cc (n,V) and the in-plane angle (n,V) of the best gamma."""
        n = images.shape[0]; V, P = self.V, self.P
        if batch is None:   # keep the (b,V,P) correlation block near 1 GB
            batch = max(1, int(1e9 // (V * P * (8 if self.cdt == torch.complex64 else 16))))
        cr = cr.to(self.rdt); wr = wr.to(self.rdt); In = In.to(self.rdt)
        Tn = torch.sqrt(torch.einsum('vr,br->bv', self.Tpow, wr[None, :] * cr ** 2))
        t0 = _time.monotonic()
        bc = torch.empty(n, V, device=images.device, dtype=self.rdt)
        bg = torch.empty(n, V, device=images.device, dtype=self.rdt)   # (n,V): 2 GB in float64 at n=8000, V=30522
        for i in range(0, n, batch):
            j = min(i + batch, n)
            If = torch.fft.fft(images[i:j].to(self.cdt), dim=-1); cw = (wr[None, :] * cr[i:j])[:, :, None].to(self.cdt)
            cc = torch.fft.ifft(torch.einsum('vrk,brk->bvk', self.Tf, If * cw), dim=-1).real
            cc = cc / (Tn[i:j, :, None] * In[i:j, None, None])
            m, gi = cc.max(-1); bc[i:j] = m; bg[i:j] = -(gi.to(self.rdt)) * (2 * np.pi / P)
            del cc, If; t0 = duty_sleep(t0)
        return bc, bg


def neighbourhood(caz, cpo, deg, chunk=4096):
    """Padded neighbour index (V, max_nb) of views within `deg` of each view
    (a view is its own neighbour; padding repeats the view itself). Built in
    chunks so V ~ 30k never materialises a V x V matrix."""
    nd = torch.stack([torch.sin(cpo) * torch.cos(caz), torch.sin(cpo) * torch.sin(caz), torch.cos(cpo)], -1)
    V = nd.shape[0]; c = np.cos(np.radians(deg))
    width = 0                                          # pass 1: widest neighbourhood
    for i in range(0, V, chunk):
        j = min(i + chunk, V)
        width = max(width, int(((nd[i:j] @ nd.T) >= c).sum(1).max()))
    idx = torch.arange(V, device=nd.device)[:, None].expand(V, width).clone()   # padding = self
    for i in range(0, V, chunk):                       # pass 2: vectorised fill
        j = min(i + chunk, V)
        nb = (nd[i:j] @ nd.T) >= c
        r, col = torch.nonzero(nb, as_tuple=True)      # row-major, so rows are grouped
        start = torch.searchsorted(r, torch.arange(j - i, device=nd.device))
        pos = torch.arange(r.numel(), device=nd.device) - start[r]
        idx[i + r, pos] = col
        del nb
    return idx, nd


def select_modes(bc, NB, modes, beta, vchunk=2048):
    """Local maxima of cc over the view sphere (within the neighbour index NB),
    top `modes` of them, softmax weights exp(beta (cc^2 - cc_best^2)).
    Returns indices (n,modes), weights (n,modes), cc (n,modes)."""
    n, V = bc.shape
    nbmax = torch.empty_like(bc)
    for i in range(0, V, vchunk):                     # (n, vchunk, width) gathers
        j = min(i + vchunk, V)
        nbmax[:, i:j] = bc[:, NB[i:j]].max(-1).values
    masked = bc.masked_fill(~(bc >= nbmax - 1e-12), -2.0)
    tcc, tidx = masked.topk(modes, dim=1)
    tcc = tcc.to(beta.dtype)
    lg = (beta[:, None] * (tcc ** 2 - tcc[:, :1] ** 2)).masked_fill(tcc < -1.0, -1e30)
    return tidx, torch.softmax(lg, dim=1), tcc


# ---- gridding correction -----------------------------------------------------------
# The trilinear splat with per-voxel normalisation returns a volume attenuated
# per shell (measured 0.82 for the consensus and 0.75 for a component at D=96).
# Left uncorrected, the flat-prior gain absorbs the attenuation every E-step and
# the volume's scale drifts down each round (V0 norm 77 -> 38 over five
# iterations at D=96) while a component under a unit latent prior does not, so
# the fitted motion amplitude inflates relative to the consensus. The correction
# is a per-shell factor measured once: reconstruct a known volume from its own
# noiseless slices at the run's poses and take the ratio.

def gridding_correction(vol, ids_like, sel_idx, gam, th, cpo, caz, rr, w, D, ctf_like, B=250, floor=0.25, backend="nufft"):
    """Per-shell attenuation a(k) of reconstruct() at these poses: a Fourier volume
    reconstructed from noiseless slices of ``vol`` divided by the true one, real
    part of the cross-power over the shell. Returns (D,D,D) divisor (1/a, floored)."""
    from .slicing import fourier_volume
    from .generator import _slice
    dev = vol.device; n = len(ids_like); SP = w.numel()
    acc = HeteroAccumulator.new(D, 1, dev)
    for i in range(0, n, B):
        j = min(i + B, n); loc = torch.arange(i, j, device=dev)
        v = sel_idx[loc, 0]
        pts = slice_points(fourier_circles(th, cpo[v], caz[v], gam[loc, v]), rr)
        y = _slice(vol, pts, backend).reshape(-1)
        c = ctf_like[ids_like[loc]].reshape(-1)
        wg = w.reshape(1, -1).expand(j - i, -1).reshape(-1)
        ea = torch.ones((j - i) * SP, 1, dtype=w.dtype, device=dev)
        acc.add(pts.reshape(-1, 3), y * c, c, wg, ea, ea[:, :, None])
    Frec = acc.solve(1e-3)[0]; Ftrue = fourier_volume(vol)
    ii = torch.arange(D, dtype=torch.float64, device=dev) - D // 2
    X, Y, Z = torch.meshgrid(ii, ii, ii, indexing="ij"); RR = (X ** 2 + Y ** 2 + Z ** 2).sqrt()
    corr = torch.ones(D, D, D, dtype=torch.float64, device=dev)
    for k in range(1, D // 2):
        m = (RR >= k - 0.5) & (RR < k + 0.5)
        a = ((Frec[m] * Ftrue[m].conj()).real.sum() / (Ftrue[m].abs() ** 2).sum().clamp(min=1e-30)).item()
        corr[m] = 1.0 / max(a, floor)
    return corr


# ---- reconstruction ------------------------------------------------------------

def reconstruct(p, ids, sel_idx, sel_w, gam, th, cpo, caz, rr, w, D, lam=1e-3, B=250,
                K=0, Ea=None, Eaa=None, tau=None, grid_corr=None):
    """Mixture insertion. With K=0 this is the homogeneous num/den; with K>0 it
    is the per-voxel (K+1)x(K+1) solve, Ea/Eaa (n,modes,K+1[,K+1]) the posterior
    moments of the coefficients and tau (n,) the noise precision (weights the
    particles; None = 1). Returns (K+1,D,D,D) complex Fourier volumes."""
    dev = p.images_fourier.device
    acc = HeteroAccumulator.new(D, K + 1, dev)
    n = len(ids); SP = w.numel(); t0 = _time.monotonic()
    for i in range(0, n, B):
        t0 = duty_sleep(t0)
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
    F = acc.solve(lam)
    return F * grid_corr[None] if grid_corr is not None else F
