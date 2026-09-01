"""Integrated log likelihood with a per-gamma Isx correction.

CryoLike's ill_kernel marginalises gain, offset and noise scale for the model
    y = gain * x + offset * s + noise
and its ranking is then exactly invariant to any gain and offset applied to the
image -- that being the defining property of marginalising those parameters away.

It holds exactly only for a RADIALLY SYMMETRIC mask. The kernel forms
    Isx = sum over (radius, inplane) of  s * (CTF x)
which is one scalar per (image, template), then broadcasts it across in-plane
rotations. That assumes <s, R_gamma x> does not depend on gamma. The default
mask 4*sinc(2x)*sinc(2y) is separable in Cartesian coordinates, not radial, and
varies by up to 15% within a shell, so the assumption fails and invariance
degrades (rank correlation 0.92 at large offsets, with the argmax moving off the
true template).

The fix is to evaluate Isx at every in-plane rotation. Since
    Isx(gamma) = sum_r sum_theta s(r,theta) x(r, theta+gamma)
that is the same circular cross-correlation already used for Ixy, so it costs
one extra FFT pair rather than a redesign.
"""

from __future__ import annotations
from math import lgamma

import numpy as np
import torch

from cryolike.util import (absq, complex_mul_real, fourier_bessel_transform,
                           inverse_fourier_bessel_transform)


def isx_per_gamma(sqrt_weighted_mask_points, ctf_templates_msnw, n_inplanes):
    """<s, R_gamma x> for every in-plane rotation.

    sqrt_weighted_mask_points : (n_radius, n_inplane) real
    ctf_templates_msnw        : (M, S, n_radius, n_inplane) complex
    returns                   : (M, S, n_inplane)

    Real-valued by construction: for Hermitian templates and a centrosymmetric
    mask the theta-sum pairs x(theta) with conj(x(theta+pi)), so the imaginary
    parts cancel -- which is why the same irfft-based inverse used for Ixy
    applies here too.
    """
    m_hat = fourier_bessel_transform(sqrt_weighted_mask_points.to(ctf_templates_msnw.dtype))
    x_hat = fourier_bessel_transform(ctf_templates_msnw)
    prod = (m_hat.conj()[None, None] * x_hat).sum(dim=2)
    return inverse_fourier_bessel_transform(prod, n_inplanes=n_inplanes)


def ill_kernel_pergamma(Iss, n_pixels_phys, sqrt_weighted_mask_points,
                        CTF_sqrtweighted_fourier_templates_msnw,
                        images_fourier_sqrtweighted_mnw, Ixx, Iyy, Ixy,
                        per_gamma=True):
    """As cryolike.ill_kernel, but Isx may vary with the in-plane rotation.

    per_gamma=False reproduces the original exactly, for A/B comparison.
    """
    n_inplanes = Ixy.shape[-1]
    cdt = CTF_sqrtweighted_fourier_templates_msnw.dtype
    if per_gamma:
        # irfft returns a real tensor (Isx is real for Hermitian templates and a
        # centrosymmetric mask); cast so absq/complex_mul_real accept it
        Isx = isx_per_gamma(sqrt_weighted_mask_points,
                            CTF_sqrtweighted_fourier_templates_msnw,
                            n_inplanes).to(cdt).unsqueeze(2)       # (M,S,1,w)
    else:
        Isx = torch.sum(sqrt_weighted_mask_points
                        * CTF_sqrtweighted_fourier_templates_msnw,
                        dim=(2, 3)).unsqueeze(2).unsqueeze(3)      # (M,S,1,1)

    Isy = torch.sum(sqrt_weighted_mask_points * images_fourier_sqrtweighted_mnw,
                    dim=(1, 2))[:, None, None, None]

    A = -absq(Isx) + Ixx * Iss
    B = -complex_mul_real(Isx.to(Isy.dtype), Isy) + Ixy * Iss
    C = absq(Isy) - Iyy * Iss
    D = -(B ** 2 / A + C)
    p = n_pixels_phys / 2.0 - 2.0
    constant = ((3.0 - n_pixels_phys) / 2.0 * np.log(2 * np.pi)
                - np.log(2) - 0.5 * np.log(Iss)
                + lgamma(n_pixels_phys / 2.0 - 2.0)
                + p * np.log(2 * Iss))
    ll = -p * torch.log(D) - 0.5 * torch.log(A) + constant
    return torch.logsumexp(ll, dim=(2, 3))
