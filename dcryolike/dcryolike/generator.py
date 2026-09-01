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
    """Generate a noisy particle stack from a volume. See module docstring for SNR."""
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
