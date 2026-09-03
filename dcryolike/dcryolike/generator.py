"""Synthetic cryo-EM particle generation at controlled SNR.

SNR CONVENTION (stated explicitly, because CryoLike's two are not equivalent):

    snr = <|CTF * T|^2>_grid / sigma^2

where <.>_grid is the quadrature-weighted MEAN over polar grid points, i.e.
``integrate(f) / integrate(1)``, and sigma^2 is the TOTAL variance of the
complex noise at each grid point (real and imaginary each sigma^2/2).

This is deliberately normalised so it does not depend on grid size. For
reference, CryoLike's own two definitions differ from each other and from this:
  - ``add_noise_phys``    uses mean(|image|^2) over pixels, real noise
  - ``add_noise_fourier`` uses integrate(|F|^2), an un-normalised quadrature
    SUM (integrate(ones) ~ 800 on a 17x128 grid), with complex noise
so the same nominal ``snr`` means different things in each. Conversions:
    snr_here = snr_add_noise_fourier * integrate(ones)

Angles follow CryoLike exactly: R = Rz(azimu) Ry(polar) Rz(gamma), radians.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch

from .slicing import fourier_circles, slice_points, slice_nufft, slice_gridding


@dataclass
class Particles:
    """Generated stack, all tensors on the same device."""
    images_fourier: torch.Tensor   # (n, n_shells, n_inplanes) complex
    clean_fourier: torch.Tensor    # signal before noise, same shape
    azimus: torch.Tensor           # (n,) radians
    polars: torch.Tensor           # (n,)
    gammas: torch.Tensor           # (n,)
    ctf: torch.Tensor              # (n, n_shells, n_inplanes) real
    sigma_noise: torch.Tensor      # (n,) total complex noise std per grid point
    snr: float


def gaussian_atom_volume(D, n_atoms=400, radius_frac=0.30, sigma_vox=1.4,
                         seed=0, device="cuda", dtype=torch.float64):
    """A chiral 'protein': Gaussian atoms in a ball, with spectral content to Nyquist.

    Chirality matters -- a mirror-symmetric phantom cannot detect a handedness
    bug, which is the exact failure mode a wrong rotation convention produces.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    pos = torch.randn(n_atoms, 3, generator=g, dtype=dtype)
    pos = pos / pos.norm(dim=1, keepdim=True).clamp(min=1e-9)
    pos = pos * torch.rand(n_atoms, 1, generator=g, dtype=dtype).pow(1/3) * (D*radius_frac)
    # break mirror symmetry: a right-handed helical arm
    t = torch.linspace(0, 4*np.pi, 60, dtype=dtype)
    helix = torch.stack([0.20*D*torch.cos(t), 0.20*D*torch.sin(t),
                         0.055*D*t - 0.35*D], dim=1)
    pos = torch.cat([pos, helix], 0).to(device)

    ax = torch.arange(D, dtype=dtype, device=device) - D//2
    X, Y, Z = torch.meshgrid(ax, ax, ax, indexing="ij")
    vol = torch.zeros(D, D, D, dtype=dtype, device=device)
    for p in pos:
        vol += torch.exp(-((X-p[0])**2 + (Y-p[1])**2 + (Z-p[2])**2)/(2*sigma_vox**2))
    return vol.contiguous()


def uniform_so3(n, seed=0, device="cuda", dtype=torch.float64):
    """Uniform viewing angles in CryoLike's convention.

    polar = arccos(u), u ~ U[-1,1] gives uniform coverage of the sphere;
    sampling polar uniformly instead would over-weight the poles.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    azimus = torch.rand(n, generator=g, dtype=dtype) * 2*np.pi
    polars = torch.acos(1 - 2*torch.rand(n, generator=g, dtype=dtype))
    gammas = torch.rand(n, generator=g, dtype=dtype) * 2*np.pi
    return (azimus.to(device), polars.to(device), gammas.to(device))


def make_ctf(polar_grid, n, box_size, seed=0, device="cuda", dtype=torch.float64,
             defocus_range=(5000., 25000.), astig=100., voltage=300.,
             cs=2.7, amp_contrast=0.1, bfactor=None,
             defocusU=None, defocusV=None, defocusAngle=None):
    """CTF on the polar grid, following CryoLike's _ctf_relion formula.

    ``bfactor`` adds exp(-B k^2 / 4), which CryoLike parses from star files but
    does NOT apply. None reproduces CryoLike exactly.
    """
    h, e, c, m0 = 6.62607015e-34, 1.60217663e-19, 2.99792458e8, 9.1093837015e-31
    V = voltage * 1e3
    lam = h/np.sqrt(2*m0*e*V)/np.sqrt(1 + e*V/(2*m0*c*c))*1e10   # Angstrom
    Cs = cs * 1e7                                                 # mm -> Angstrom

    if defocusU is not None:
        # explicit values, so this can be checked against a reference implementation
        dfu = torch.as_tensor(defocusU, dtype=dtype).reshape(-1)
        dfv = torch.as_tensor(defocusV, dtype=dtype).reshape(-1)
        dfang = torch.as_tensor(defocusAngle, dtype=dtype).reshape(-1)   # radians
    else:
        g = torch.Generator(device="cpu").manual_seed(seed)
        dfu = defocus_range[0] + torch.rand(n, generator=g, dtype=dtype)*(defocus_range[1]-defocus_range[0])
        dfv = dfu + astig
        dfang = torch.rand(n, generator=g, dtype=dtype) * np.pi
    defocus = (0.5*(dfu+dfv)).to(device)
    astigm = (0.5*(dfv-dfu).abs()).to(device)
    dfang = dfang.to(device)

    r = torch.tensor(polar_grid.radius_shells, dtype=dtype, device=device) * 2.0/box_size
    th = torch.tensor(polar_grid.theta_shell, dtype=dtype, device=device)
    local_df = defocus[:,None,None] + astigm[:,None,None]*torch.cos(2*(th[None,None,:]-dfang[:,None,None]))
    r2 = (r**2)[None,:,None]
    gamma = -np.pi*lam*r2*local_df + (np.pi/2)*Cs*lam**3*(r2**2)
    ctf = -np.sqrt(1-amp_contrast**2)*torch.sin(gamma) + amp_contrast*torch.cos(gamma)
    if bfactor is not None:
        ctf = ctf * torch.exp(-bfactor * r2 / 4.0)
    return ctf


def generate(volume, polar_grid, n_particles, snr, box_size=None, seed=0,
             backend="nufft", ctf_kwargs=None, device="cuda"):
    """Generate a noisy particle stack from a volume. See module docstring for SNR.

    WARNING (review, 2026-09-02): the noise here is iid per polar grid point,
    which is UNPHYSICAL -- no real-space noise field produces it -- and reads
    far easier than real noise at the same nominal SNR (top-1 at "SNR 0.05":
    17.5% here vs 0% with physical noise). Kept only so earlier experiments
    remain reproducible. Use generate_physical() for anything new.
    """
    D = volume.shape[-1]
    box_size = float(D) if box_size is None else box_size
    dtype = volume.dtype

    azimus, polars, gammas = uniform_so3(n_particles, seed, device, dtype)
    thetas = torch.tensor(polar_grid.theta_shell, dtype=dtype, device=device)
    radii = torch.tensor(polar_grid.radius_shells, dtype=dtype, device=device)

    circles = fourier_circles(thetas, polars, azimus, gammas)
    pts = slice_points(circles, radii)
    if backend == "nufft":
        tmpl = slice_nufft(volume, pts)
    else:
        from .slicing import fourier_volume
        tmpl = slice_gridding(fourier_volume(volume), pts)
    tmpl = tmpl.reshape(n_particles, polar_grid.n_shells, polar_grid.n_inplanes)

    ctf = make_ctf(polar_grid, n_particles, box_size, seed=seed,
                   device=device, dtype=dtype, **(ctf_kwargs or {}))
    clean = tmpl * ctf

    # quadrature-weighted MEAN power -> grid-size independent SNR
    w = torch.tensor(polar_grid.integration_weight_points, dtype=dtype,
                     device=device).reshape(polar_grid.n_shells, polar_grid.n_inplanes)
    wsum = w.sum()
    power = (clean.abs().pow(2) * w).sum(dim=(1,2)) / wsum
    sigma = torch.sqrt(power / snr)

    gen = torch.Generator(device=device).manual_seed(seed + 99991)
    noise = torch.randn(clean.shape, generator=gen, dtype=clean.dtype, device=device)
    images = clean + noise * sigma[:,None,None]

    return Particles(images, clean, azimus, polars, gammas, ctf, sigma, snr)


# ---------------------------------------------------------------------------
# REVIEW FIX: physically correct noise.
#
# The iid-per-grid-point noise in generate() is UNPHYSICAL. Real cryo-EM noise
# is white in real space; its Fourier transform sampled on a polar grid has
# uniform variance but is strongly CORRELATED between nearby points (measured:
# 1.000 between adjacent inner-shell points, 0.48 mid-shell). No real-space
# field produces independent per-point polar noise. CryoLike's quadrature
# weighting is the matched filter for the physical case (verified: weighted CC
# beats unweighted under real-space white noise, loses under iid), so the
# generator, not the likelihood, was wrong.
#
# SNR here is defined in REAL SPACE, per pixel, over the whole box, on the
# CTF-free projection: snr = var(projection pixels) / var(noise pixels).
# This is comparable to literature figures; the polar per-point convention
# in generate() is not, and reads much easier than it is.
# ---------------------------------------------------------------------------

def _polar_dft_matrix(polar_grid, D, device, dtype=torch.complex128):
    """Exact DFT from a D x D real-space image to the polar Fourier points."""
    kx = torch.tensor(polar_grid.x_points, dtype=torch.float64, device=device)
    ky = torch.tensor(polar_grid.y_points, dtype=torch.float64, device=device)
    ax = torch.arange(D, dtype=torch.float64, device=device) - D // 2
    X, Y = torch.meshgrid(ax, ax, indexing="ij")
    ph = (-2j * np.pi / D) * (kx[:, None] * X.reshape(1, -1) + ky[:, None] * Y.reshape(1, -1))
    return torch.exp(ph.to(dtype)) / D


def real_space_projections(volume, azimus, polars, gammas, device="cuda"):
    """CTF-free real-space projections, via a Cartesian Fourier slice + ifft2."""
    D = volume.shape[-1]; n = azimus.shape[0]
    f = torch.arange(D, dtype=volume.dtype, device=device) - D // 2
    KX, KY = torch.meshgrid(f, f, indexing="ij")
    circ = torch.stack([KX.reshape(-1), KY.reshape(-1), torch.zeros(D * D, dtype=volume.dtype, device=device)], -1)
    # rotate the Cartesian slice plane with the same R = Rz(a) Ry(b) Rz(g) as fourier_circles
    ca, sa = torch.cos(azimus), torch.sin(azimus)
    cb, sb = torch.cos(polars), torch.sin(polars)
    cg, sg = torch.cos(gammas), torch.sin(gammas)
    Rz_a = torch.stack([torch.stack([ca, -sa, 0*ca], -1), torch.stack([sa, ca, 0*ca], -1), torch.stack([0*ca, 0*ca, 1+0*ca], -1)], -2)
    Ry_b = torch.stack([torch.stack([cb, 0*cb, sb], -1), torch.stack([0*cb, 1+0*cb, 0*cb], -1), torch.stack([-sb, 0*cb, cb], -1)], -2)
    Rz_g = torch.stack([torch.stack([cg, -sg, 0*cg], -1), torch.stack([sg, cg, 0*cg], -1), torch.stack([0*cg, 0*cg, 1+0*cg], -1)], -2)
    R = Rz_a @ Ry_b @ Rz_g
    pts = torch.einsum('bij,pj->bpi', R, circ)
    F = slice_nufft(volume, pts).reshape(n, D, D)
    img = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(F, dim=(-2, -1)), norm="ortho"), dim=(-2, -1)).real
    return img


def generate_physical(volume, polar_grid, n_particles, snr_real, box_size=None, seed=0,
                      ctf_kwargs=None, device="cuda"):
    """Like generate(), but with real-space white noise and a real-space SNR."""
    D = volume.shape[-1]
    box_size = float(D) if box_size is None else box_size
    dtype = volume.dtype

    azimus, polars, gammas = uniform_so3(n_particles, seed, device, dtype)
    thetas = torch.tensor(polar_grid.theta_shell, dtype=dtype, device=device)
    radii = torch.tensor(polar_grid.radius_shells, dtype=dtype, device=device)
    tmpl = slice_nufft(volume, slice_points(fourier_circles(thetas, polars, azimus, gammas), radii))
    tmpl = tmpl.reshape(n_particles, polar_grid.n_shells, polar_grid.n_inplanes)
    ctf = make_ctf(polar_grid, n_particles, box_size, seed=seed, device=device, dtype=dtype,
                   **(ctf_kwargs or {}))
    clean = tmpl * ctf

    # real-space signal variance per pixel, CTF-free projection
    proj = real_space_projections(volume, azimus, polars, gammas, device)
    sig_var = proj.var(dim=(1, 2))
    noise_std = torch.sqrt(sig_var / snr_real)                     # per particle

    gen = torch.Generator(device=device).manual_seed(seed + 99991)
    nreal = torch.randn(n_particles, D * D, generator=gen, dtype=dtype, device=device)
    nreal = nreal * noise_std[:, None]
    M = _polar_dft_matrix(polar_grid, D, device)
    npol = (nreal.to(torch.complex128) @ M.T).reshape(n_particles, polar_grid.n_shells,
                                                       polar_grid.n_inplanes).to(clean.dtype)
    # the polar DFT above and slice_nufft share the exp(-2 pi i k.x/D)/D convention
    # only up to a D-dependent scale; calibrate so a CTF-free clean image and its
    # own real-space projection agree in polar power
    ref = (M @ proj.reshape(n_particles, -1).to(torch.complex128).T).T.reshape_as(npol)
    scale = torch.sqrt((tmpl.abs() ** 2).sum(dim=(1, 2)) / (ref.abs() ** 2).sum(dim=(1, 2)))
    npol = npol * scale[:, None, None]

    images = clean + npol
    return Particles(images, clean, azimus, polars, gammas, ctf, noise_std, snr_real)
