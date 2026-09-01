"""Differentiable central-slice operators for cryo-EM.

The volume is parameterised in *real space* as a real-valued tensor. This is a
deliberate choice: it makes the density real and its Fourier transform Hermitian
by construction, so we never have to impose either as a constraint.

Two backends evaluate the central slice, both differentiable w.r.t. the density:

``gridding``
    FFT the volume once, then trilinearly interpolate at the slice coordinates
    via ``grid_sample``. Native autograd, no external dependency. This is the
    RELION/cryoSPARC convention, so it doubles as a fair baseline.
``nufft``
    Type-2 NUFFT evaluated directly from the real-space density. Accurate, but
    opaque to autograd, so we supply the adjoint analytically (see NufftSlice).

The backends are interchangeable; ``gridding`` is the reference the ``nufft``
path is validated against.

Conventions follow finufft:
    type 1 (nonuniform -> uniform):  f_k = sum_j c_j exp(i*isign*k.x_j)
    type 2 (uniform -> nonuniform):  c_j = sum_k f_k exp(i*isign*k.x_j)
Here the "modes" are voxel positions r and the nonuniform points are Fourier
coordinates k, i.e. c(k_j) = sum_r density_r exp(i*isign*r.k_j).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Slice geometry
# --------------------------------------------------------------------------

def fourier_circles(
    thetas: torch.Tensor,   # (n_inplane,)  in-plane angle of the polar grid
    polars: torch.Tensor,   # (n_templates,) beta
    azimus: torch.Tensor,   # (n_templates,) alpha
    gammas: torch.Tensor,   # (n_templates,) in-plane offset
) -> torch.Tensor:
    """Unit-radius central-slice directions for each viewing angle.

    Differentiable w.r.t. the angles, so pose gradients are available for free
    if we ever want them. Returns (n_templates, n_inplane, 3).

    This mirrors cryolike.stacks.template._fourier_circles; kept here so the
    differentiable path has no hidden coupling to the reference implementation.
    """
    th = thetas[None, :] + gammas[:, None]
    cos_th, sin_th = torch.cos(th), torch.sin(th)
    cos_b, sin_b = torch.cos(polars), torch.sin(polars)
    cos_a, sin_a = torch.cos(azimus), torch.sin(azimus)

    cos_th_cos_b = cos_th * cos_b[:, None]
    x = cos_a[:, None] * cos_th_cos_b - sin_th * sin_a[:, None]
    y = sin_a[:, None] * cos_th_cos_b + sin_th * cos_a[:, None]
    z = -sin_b[:, None] * cos_th
    return torch.stack((x, y, z), dim=2)


def slice_points(
    circles: torch.Tensor,        # (n_templates, n_inplane, 3) unit directions
    radius_shells: torch.Tensor,  # (n_shells,)
) -> torch.Tensor:
    """Scale unit directions out to every radial shell.

    Returns (n_templates, n_shells * n_inplane, 3), flattened shell-major to
    match cryolike's (n_shells, n_inplanes) point ordering.
    """
    pts = circles[:, None, :, :] * radius_shells[None, :, None, None]
    return pts.reshape(circles.shape[0], -1, 3)


# --------------------------------------------------------------------------
# Backend 1: gridding (native autograd)
# --------------------------------------------------------------------------

def fourier_volume(density: torch.Tensor) -> torch.Tensor:
    """Centred 3D DFT of a real density. Returns a complex (D, D, D) tensor."""
    return torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(density), norm="ortho")
    )


def slice_gridding(
    fvol: torch.Tensor,   # (D, D, D) complex, centred
    points: torch.Tensor, # (n_templates, n_points, 3) in *voxel* frequency units
) -> torch.Tensor:
    """Trilinear central-slice extraction. Returns (n_templates, n_points) complex.

    grid_sample has no complex support, so real and imaginary parts ride along
    as two channels. Coordinates are normalised to [-1, 1] with align_corners
    =True, which puts sample -1 exactly on voxel 0 and +1 on voxel D-1 -- the
    convention that makes the DC term land on the centre voxel of an even grid.
    """
    D = fvol.shape[-1]
    if fvol.shape != (D, D, D):
        raise ValueError(f"expected a cubic volume, got {tuple(fvol.shape)}")

    vol = torch.stack((fvol.real, fvol.imag), dim=0)[None]        # (1,2,D,D,D)

    # Voxel frequency index -> normalised [-1,1]. Frequency f lives at array
    # index (f + D//2), and align_corners=True puts index 0 at -1 and index
    # D-1 at +1, so the offset is not optional: omitting it costs half a voxel
    # and shows up as a phase ramp in the reconstruction.
    g = (points + D // 2) * (2.0 / (D - 1)) - 1.0
    # grid_sample's last grid axis indexes W (the fastest-varying volume axis),
    # so xyz must be reversed to map x->axis 0, y->axis 1, z->axis 2.
    g = torch.flip(g, dims=(-1,))
    n_t, n_p, _ = points.shape
    grid = g.reshape(1, n_t, n_p, 1, 3)

    out = F.grid_sample(
        vol, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )                                                             # (1,2,n_t,n_p,1)
    out = out[0, :, :, :, 0]                                      # (2,n_t,n_p)
    return torch.complex(out[0], out[1])


# --------------------------------------------------------------------------
# Backend 2: NUFFT with an analytic adjoint
# --------------------------------------------------------------------------

class NufftSlice(torch.autograd.Function):
    """Type-2 NUFFT from a real-space density to Fourier sample points.

    The forward map is *linear* in the density, so its vector-Jacobian product
    is exactly the adjoint operator -- a type-1 NUFFT with the sign of the
    exponent flipped. We supply that analytically rather than trying to
    differentiate through the C library.

    Gradients w.r.t. the sample points are also analytic:
        d c_j / d k_j^(d) = sum_r density_r * (i*isign*r_d) * exp(i*isign*r.k_j)
    i.e. three more type-2 transforms with the density premultiplied by the
    voxel coordinate along each axis. Disabled unless the points require grad,
    since it triples the forward cost.
    """

    @staticmethod
    def forward(ctx, density, points, isign: int, eps: float):
        from cufinufft import Plan

        D = density.shape[-1]
        n_t, n_p, _ = points.shape
        dtype = "complex64" if density.dtype in (torch.float32, torch.complex64) else "complex128"
        cdtype = torch.complex64 if dtype == "complex64" else torch.complex128

        # finufft wants nonuniform points in radians. Our points are in voxel
        # frequency index units, and the DFT convention validated by the
        # central-slice test is  exp(-2*pi*i * f.r / D)  with a D^1.5 scale.
        # With isign=-1 that means x_j = 2*pi*f/D.
        dens_c = density.to(cdtype).contiguous()
        ptype = torch.float32 if dtype == "complex64" else torch.float64
        pts = (points.reshape(-1, 3) * (2.0 * np.pi / D)).to(ptype).contiguous()

        plan = Plan(nufft_type=2, n_modes=(D, D, D), n_trans=1,
                    dtype=dtype, eps=eps, isign=isign)
        plan.setpts(pts[:, 0].contiguous(), pts[:, 1].contiguous(), pts[:, 2].contiguous())
        out = torch.zeros(pts.shape[0], dtype=cdtype, device=density.device)
        plan.execute(dens_c, out)

        ctx.save_for_backward(dens_c, pts)
        ctx.isign, ctx.eps, ctx.D, ctx.shape = isign, eps, D, (n_t, n_p)
        ctx.needs_pts = points.requires_grad
        ctx.scale = float(D) ** -1.5
        return out.reshape(n_t, n_p) * ctx.scale

    @staticmethod
    def backward(ctx, grad_out):
        from cufinufft import Plan

        dens_c, pts = ctx.saved_tensors
        D, isign, eps = ctx.D, ctx.isign, ctx.eps
        cdtype = dens_c.dtype
        dtype = "complex64" if cdtype == torch.complex64 else "complex128"
        g = (grad_out * ctx.scale).reshape(-1).contiguous().to(cdtype)

        grad_density = None
        if ctx.needs_input_grad[0]:
            # Adjoint of type-2 is type-1 with the exponent sign flipped.
            plan = Plan(nufft_type=1, n_modes=(D, D, D), n_trans=1,
                        dtype=dtype, eps=eps, isign=-isign)
            plan.setpts(pts[:, 0].contiguous(), pts[:, 1].contiguous(), pts[:, 2].contiguous())
            out = torch.zeros((D, D, D), dtype=cdtype, device=dens_c.device)
            plan.execute(g, out)
            # Density is real, so only the real part of the Wirtinger gradient
            # survives the projection back onto the real parameterisation.
            grad_density = out.real

        grad_points = None
        if ctx.needs_pts:
            r = torch.arange(D, dtype=torch.float64, device=dens_c.device) - D // 2
            plan = Plan(nufft_type=2, n_modes=(D, D, D), n_trans=1,
                        dtype=dtype, eps=eps, isign=isign)
            plan.setpts(pts[:, 0].contiguous(), pts[:, 1].contiguous(), pts[:, 2].contiguous())
            cols = []
            for axis in range(3):
                shape = [1, 1, 1]
                shape[axis] = D
                weighted = dens_c * (1j * isign * r.reshape(shape)).to(cdtype)
                o = torch.zeros(pts.shape[0], dtype=cdtype, device=dens_c.device)
                plan.execute(weighted.contiguous(), o)
                # VJP for a real parameter with complex output is Re(conj(g) * dc/dx);
                # no factor of 2 (validated against finite differences).
                cols.append(torch.real(torch.conj(g) * o))
            # chain rule through x_j = 2*pi*f/D
            grad_points = torch.stack(cols, dim=-1).reshape(*ctx.shape, 3)
            grad_points = grad_points * (2.0 * np.pi / D)

        return grad_density, grad_points, None, None


def slice_nufft(density, points, isign: int = -1, eps: float = 1e-6):
    """Differentiable type-2 NUFFT central slice. Returns (n_templates, n_points)."""
    return NufftSlice.apply(density, points, isign, eps)
