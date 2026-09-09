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
