"""Gradient-based pose refinement against the exact CryoLike likelihood.

This is the stage cryoDRGN-AI performs with SGD after its hierarchical pose
search. The difference here is that the objective is the exact normalised
cross-correlation -- the matched filter, with the per-particle CTF applied
exactly -- and the gradient w.r.t. pose is ANALYTIC all the way down:

    angles -> slice coordinates      (fourier_circles, pure torch)
    coordinates -> Fourier samples   (NufftSlice, analytic d/dk)

so no autodiff through an interpolator and no grid-resolution floor.
"""

from __future__ import annotations
import torch

from .slicing import fourier_circles, slice_points, slice_nufft


def refine_poses(volume, images, ctf, thetas, radii, weights,
                 azimus, polars, gammas, steps=60, lr=0.02, backend="nufft"):
    """Refine per-particle (azimu, polar, gamma) by maximising the normalised
    cross-correlation. All inputs are batched; returns refined angles.

    Angles are optimised as free parameters -- no grid, so the result is not
    limited by the resolution of whatever search produced the starting point.
    """
    a = azimus.clone().requires_grad_(True)
    p = polars.clone().requires_grad_(True)
    g = gammas.clone().requires_grad_(True)
    opt = torch.optim.Adam([a, p, g], lr=lr)

    S, P = images.shape[-2], images.shape[-1]
    In = torch.sqrt((images.abs() ** 2 * weights).sum(dim=(1, 2)))

    for _ in range(steps):
        pts = slice_points(fourier_circles(thetas, p, a, g), radii)
        T = slice_nujit(volume, pts, backend).reshape(-1, S, P) * ctf
        num = (T.conj() * images * weights).sum(dim=(1, 2)).real
        Tn = torch.sqrt((T.abs() ** 2 * weights).sum(dim=(1, 2)))
        cc = num / (Tn * In)
        loss = -cc.sum()
        opt.zero_grad(); loss.backward(); opt.step()

    return a.detach(), p.detach(), g.detach()


def slice_nujit(volume, pts, backend="nufft"):
    if backend == "nufft":
        return slice_nufft(volume, pts)
    from .slicing import fourier_volume, slice_gridding
    return slice_gridding(fourier_volume(volume), pts)
