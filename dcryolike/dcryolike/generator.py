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


def _phantom_atoms(D, n_atoms=400, radius_frac=0.30, seed=0, dtype=torch.float64):
    """Atom positions of the chiral phantom: (body, helical arm) on the CPU."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    pos = torch.randn(n_atoms, 3, generator=g, dtype=dtype)
    pos = pos / pos.norm(dim=1, keepdim=True).clamp(min=1e-9)
    pos = pos * torch.rand(n_atoms, 1, generator=g, dtype=dtype).pow(1/3) * (D*radius_frac)
    # break mirror symmetry: a right-handed helical arm
    t = torch.linspace(0, 4*np.pi, 60, dtype=dtype)
    helix = torch.stack([0.20*D*torch.cos(t), 0.20*D*torch.sin(t),
                         0.055*D*t - 0.35*D], dim=1)
    return pos, helix


def _render_atoms(pos, D, sigma_vox=1.4, device="cuda", dtype=torch.float64):
    """Sum of isotropic Gaussians on a D^3 grid, one atom at a time (bit-for-bit
    the original gaussian_atom_volume loop, so existing pins are unaffected)."""
    pos = pos.to(device)
    ax = torch.arange(D, dtype=dtype, device=device) - D//2
    X, Y, Z = torch.meshgrid(ax, ax, ax, indexing="ij")
    vol = torch.zeros(D, D, D, dtype=dtype, device=device)
    for p in pos:
        vol += torch.exp(-((X-p[0])**2 + (Y-p[1])**2 + (Z-p[2])**2)/(2*sigma_vox**2))
    return vol.contiguous()


def gaussian_atom_volume(D, n_atoms=400, radius_frac=0.30, sigma_vox=1.4,
                         seed=0, device="cuda", dtype=torch.float64):
    """A chiral 'protein': Gaussian atoms in a ball, with spectral content to Nyquist.

    Chirality matters -- a mirror-symmetric phantom cannot detect a handedness
    bug, which is the exact failure mode a wrong rotation convention produces.
    """
    body, helix = _phantom_atoms(D, n_atoms, radius_frac, seed, dtype)
    return _render_atoms(torch.cat([body, helix], 0), D, sigma_vox, device, dtype)


def hinge_phantom(D, t, n_atoms=400, radius_frac=0.30, sigma_vox=1.4,
                  seed=0, device="cuda", dtype=torch.float64, axis=(1.0, 0.0, 0.0)):
    """The same phantom with its helical arm swung by angle ``t`` (radians) about
    a hinge at the base of the arm. ``hinge_phantom(D, 0.0) == gaussian_atom_volume(D)``
    exactly, so t parameterises a one-dimensional continuous motion of a rigid
    domain while the body stays fixed -- the simplest heterogeneity that is
    NOT a linear function of t (a rotation), which is what a linear latent
    model has to approximate.

    Returns the volume and the arm's RMS atom displacement in voxels (a
    physical scale for how large the motion is).
    """
    body, helix = _phantom_atoms(D, n_atoms, radius_frac, seed, dtype)
    hinge = helix[0].clone()                       # base of the arm
    a = torch.tensor(axis, dtype=dtype); a = a / a.norm()
    K = torch.tensor([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=dtype)
    Rt = torch.eye(3, dtype=dtype) + np.sin(t)*K + (1-np.cos(t))*(K @ K)   # Rodrigues
    # t == 0 keeps the ORIGINAL positions (not a round trip through the hinge
    # frame), so the equality with gaussian_atom_volume is exact, not 1e-16
    arm = helix if t == 0 else (helix - hinge) @ Rt.T + hinge
    rms = (arm - helix).norm(dim=1).pow(2).mean().sqrt().item()
    vol = _render_atoms(torch.cat([body, arm], 0), D, sigma_vox, device, dtype)
    return vol, rms


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
    r2 = (r**2)[None,:,None]
    S, P = polar_grid.n_shells, polar_grid.n_inplanes
    n_out = defocus.shape[0]
    out = torch.empty(n_out, S, P, dtype=dtype, device=device)
    # chunked over particles: three (n,S,P) trig temporaries live at once, 1.6 GB
    # each at n=16000, D=96. Elementwise arithmetic, so the result is bit-for-bit
    # the one-shot version (asserted in test_ctf against cryolike's reference).
    step = max(1, int(2e8 // max(1, S * P)))
    for i in range(0, n_out, step):
        j = min(i + step, n_out)
        local_df = defocus[i:j,None,None] + astigm[i:j,None,None]*torch.cos(2*(th[None,None,:]-dfang[i:j,None,None]))
        gamma = -np.pi*lam*r2*local_df + (np.pi/2)*Cs*lam**3*(r2**2)
        c = -np.sqrt(1-amp_contrast**2)*torch.sin(gamma) + amp_contrast*torch.cos(gamma)
        if bfactor is not None:
            c = c * torch.exp(-bfactor * r2 / 4.0)
        out[i:j] = c
        del local_df, gamma, c
    return out


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

def _polar_dft_matrix(polar_grid, D, device, dtype=torch.complex128, rows=2048):
    """Exact DFT from a D x D real-space image to the polar Fourier points.

    Built in row chunks: the one-shot version holds three (n_points x D^2)
    float64/complex128 temporaries, ~5.5 GB at D=96 with 12.5k polar points.
    Element-wise arithmetic is identical, so the matrix is bit-for-bit the same.
    """
    kx = torch.tensor(polar_grid.x_points, dtype=torch.float64, device=device)
    ky = torch.tensor(polar_grid.y_points, dtype=torch.float64, device=device)
    ax = torch.arange(D, dtype=torch.float64, device=device) - D // 2
    X, Y = torch.meshgrid(ax, ax, indexing="ij")
    Xf, Yf = X.reshape(1, -1), Y.reshape(1, -1)
    M = torch.empty(kx.shape[0], D * D, dtype=dtype, device=device)
    for i in range(0, kx.shape[0], rows):
        j = min(i + rows, kx.shape[0])
        ph = (-2j * np.pi / D) * (kx[i:j, None] * Xf + ky[i:j, None] * Yf)
        M[i:j] = torch.exp(ph.to(dtype)) / D
        del ph
    return M


def real_space_projections(volume, azimus, polars, gammas, device="cuda", backend="nufft"):
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
    F = _slice(volume, pts, backend).reshape(n, D, D)
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


# ---------------------------------------------------------------------------
# Heterogeneous stacks: one conformation per particle.
# ---------------------------------------------------------------------------

@dataclass
class HeteroParticles(Particles):
    conf: torch.Tensor = None      # (n,) long, index into the conformation list
    t: torch.Tensor = None         # (n,) the latent value each conformation was built at
    noise_norm_w: torch.Tensor = None   # (n,) quadrature-weighted norm of the realised noise,
    # accumulated during generation. Its only consumer is pipeline.temperature(), which needed
    # the whole noiseless stack (3.2 GB at n=16000, D=96) purely to recover this one number per
    # particle. With it present, `clean_fourier` may be None and the noisy stack is built in
    # place over the signal buffer.


def _slice(volume, pts, backend="nufft"):
    """slice_nufft, or the CPU-capable trilinear gridding backend (smoke runs only:
    gridding is an approximation, not the operator the experiments use)."""
    if backend == "nufft":
        return slice_nufft(volume, pts)
    from .slicing import fourier_volume
    return slice_gridding(fourier_volume(volume), pts)


def generate_physical_hetero(volumes, t_values, conf, polar_grid, snr_real, box_size=None,
                             seed=0, ctf_kwargs=None, device="cuda", backend="nufft", chunk=1000):
    """generate_physical() for a stack whose particles come from different volumes.

    volumes  : list of (D,D,D) volumes, one per conformation
    t_values : (n_conf,) latent value of each conformation (recorded for scoring)
    conf     : (n,) long, which conformation each particle is drawn from

    Poses, CTFs and noise are drawn for the whole stack in one stream, exactly
    as generate_physical() draws them, so a homogeneous stack (all conf equal,
    volumes=[vol]) reproduces generate_physical(vol, ...): clean signal, CTF and
    poses bit for bit, images to ~1e-15 relative (the noise scale comes from a
    projection computed in a different NUFFT batch). The
    noise level is set per particle from ITS OWN conformation's projection
    variance, so the real-space SNR is the same for every conformation.
    """
    D = volumes[0].shape[-1]
    box_size = float(D) if box_size is None else box_size
    dtype = volumes[0].dtype
    conf = torch.as_tensor(conf, device=device).long()
    n_particles = conf.shape[0]

    azimus, polars, gammas = uniform_so3(n_particles, seed, device, dtype)
    thetas = torch.tensor(polar_grid.theta_shell, dtype=dtype, device=device)
    radii = torch.tensor(polar_grid.radius_shells, dtype=dtype, device=device)
    S, P = polar_grid.n_shells, polar_grid.n_inplanes
    tmpl = torch.empty(n_particles, S * P, dtype=torch.complex128 if dtype == torch.float64
                       else torch.complex64, device=device)
    proj = torch.empty(n_particles, D, D, dtype=dtype, device=device)
    # The slice coordinates are (n, S*P, 3) float64 = 4.8 GB at n=16000, D=96, and
    # they are only ever read one particle at a time, so they are built per chunk
    # rather than for the whole stack. Same for the projections' internal (n,D,D)
    # complex temporary. Values are unchanged: the same points, the same order.
    for b in range(len(volumes)):
        idx = torch.nonzero(conf == b).squeeze(1)
        if idx.numel() == 0:
            continue
        for i in range(0, idx.numel(), chunk):
            k = idx[i:i + chunk]
            pts_k = slice_points(fourier_circles(thetas, polars[k], azimus[k], gammas[k]), radii)
            tmpl[k] = _slice(volumes[b], pts_k, backend)
            del pts_k
            proj[k] = real_space_projections(volumes[b], azimus[k], polars[k], gammas[k], device, backend)
    tmpl = tmpl.reshape(n_particles, S, P)
    ctf = make_ctf(polar_grid, n_particles, box_size, seed=seed, device=device, dtype=dtype,
                   **(ctf_kwargs or {}))
    tmpl_pow = (tmpl.abs() ** 2).sum(dim=(1, 2))
    tmpl *= ctf                      # in place: no second (n,S,P) stack (3.2 GB at n=16000)
    clean = tmpl                     # same storage, renamed for clarity

    sig_var = proj.var(dim=(1, 2))
    noise_std = torch.sqrt(sig_var / snr_real)
    gen = torch.Generator(device=device).manual_seed(seed + 99991)
    nreal = torch.randn(n_particles, D * D, generator=gen, dtype=dtype, device=device)
    nreal = nreal * noise_std[:, None]
    M = _polar_dft_matrix(polar_grid, D, device)
    wq = torch.tensor(polar_grid.integration_weight_points, dtype=dtype,
                      device=device).reshape(S, P)
    noise_norm_w = torch.empty(n_particles, dtype=dtype, device=device)
    # chunked: the (n, S*P) complex128 products peak at ~10 GB for 8000 particles
    # at D=96 if done in one go; the per-particle result is identical.
    # The noise is added INTO the signal buffer and its weighted norm accumulated,
    # so no second (n,S,P) stack is allocated (3.2 GB at n=16000, D=96) and the
    # noiseless stack need not be retained.
    for i in range(0, n_particles, chunk):
        j = min(i + chunk, n_particles)
        npol = (nreal[i:j].to(torch.complex128) @ M.T).reshape(j - i, S, P).to(clean.dtype)
        ref = (M @ proj[i:j].reshape(j - i, -1).to(torch.complex128).T).T.reshape_as(npol)
        scale = torch.sqrt(tmpl_pow[i:j] / (ref.abs() ** 2).sum(dim=(1, 2)))
        npol *= scale[:, None, None]
        noise_norm_w[i:j] = (npol.abs() ** 2 * wq).sum(dim=(1, 2))
        clean[i:j] += npol                      # in place: `clean` becomes the noisy stack
        del npol, ref, scale
    del M, nreal, proj
    images = clean                              # same storage, renamed for clarity
    t_values = torch.as_tensor(t_values, dtype=dtype, device=device)
    # clean_fourier is None: the noiseless stack was overwritten in place. Everything
    # that consumed it (the temperature) now reads noise_norm_w instead.
    return HeteroParticles(images, None, azimus, polars, gammas, ctf, noise_std, snr_real,
                           conf=conf, t=t_values[conf], noise_norm_w=noise_norm_w)
