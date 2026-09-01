"""Adaptive concentric-shell grids for the Fourier slice theorem.

The 2D polar grid and the 3D spherical grid SHARE one radial frequency list.
Because a rotation preserves radius, a 2D shell at radius k maps exactly onto
the 3D shell at radius k -- so slicing and insertion never interpolate across
frequency. All interpolation is angular, within a single shell. That removes the
radial interpolation error which dominates Cartesian trilinear gridding.

Sampling is matched to each shell's angular bandwidth:
    2D shell k needs ~2*pi*k in-plane samples
    3D shell k needs ~(2k)^2 directions (bandlimit l_max ~ k)
so inner shells are cheap and only outer shells carry the cost.

Storage is flat with per-shell offsets, since shells have different sizes.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch

from .spheregrid import RingGrid


def _pow2_at_least(x, lo=8):
    return int(2 ** np.ceil(np.log2(max(lo, x))))


@dataclass
class AdaptiveShellGrid:
    """Shared-radius adaptive 2D polar + 3D spherical shell grids."""
    radii: torch.Tensor          # (S,) shell radii, shared by 2D and 3D
    n_theta: torch.Tensor        # (S,) in-plane samples per 2D shell
    n_polar: torch.Tensor        # (S,) polar samples per 3D shell
    n_azim: torch.Tensor         # (S,) azimuthal samples per 3D shell
    off2d: torch.Tensor          # (S+1,) offsets into the flat 2D array
    off3d: torch.Tensor          # (S+1,) offsets into the flat 3D array
    device: torch.device
    dtype: torch.dtype

    @classmethod
    def build(cls, radii, device="cuda", dtype=torch.float64,
              inplane_factor=1.0, sphere_factor=2.0):
        """Shared-radius adaptive grids. Every shell uses a SphereShell-backed
        RingGrid sized to that shell, so inner shells stay cheap."""
        r = torch.as_tensor(np.asarray(radii), dtype=dtype, device=device)
        nt, rings = [], []
        for k in r.tolist():
            nt.append(_pow2_at_least(inplane_factor * 2 * np.pi * k, 8))
            rings.append(RingGrid.for_shell(k, sphere_factor=sphere_factor,
                                            device=device, dtype=dtype))
        nt = torch.tensor(nt, device=device)
        npts = torch.tensor([g.n for g in rings], device=device)
        off2d = torch.cat([torch.zeros(1, dtype=torch.long, device=device), nt.cumsum(0)])
        off3d = torch.cat([torch.zeros(1, dtype=torch.long, device=device), npts.cumsum(0)])
        o = cls(r, nt, npts, npts, off2d, off3d, torch.device(device), dtype)
        o.rings = rings
        return o

    @property
    def n2d(self): return int(self.off2d[-1])
    @property
    def n3d(self): return int(self.off3d[-1])
    @property
    def n_shells(self): return self.radii.shape[0]

    def sphere_points(self):
        """(n3d, 3) unit directions of every 3D node, flat."""
        return torch.cat([g.points() for g in self.rings], 0)

    def sphere_weights(self):
        """(n3d,) quadrature weights (solid angle), per shell summing to 4*pi."""
        return torch.cat([g.weights() for g in self.rings], 0)

    def sphere_radii(self):
        return torch.cat([self.radii[s].repeat(self.rings[s].n)
                          for s in range(self.n_shells)])

    def slice_directions(self, R):
        """Great-circle directions of the 2D grid under rotation(s) R.

        R: (B,3,3). Returns (B, n2d, 3) unit vectors. Radius is NOT applied --
        it is the shell's own radius, unchanged by rotation, which is exactly
        why no radial interpolation is ever needed.
        """
        B = R.shape[0]
        out = torch.empty(B, self.n2d, 3, dtype=self.dtype, device=self.device)
        for s in range(self.n_shells):
            nt = int(self.n_theta[s])
            th = torch.arange(nt, dtype=self.dtype, device=self.device) * (2 * np.pi / nt)
            u = torch.stack([torch.cos(th), torch.sin(th),
                             torch.zeros_like(th)], -1)              # (nt,3)
            out[:, self.off2d[s]:self.off2d[s + 1]] = torch.einsum('bij,tj->bti', R, u)
        return out

    # ---- the two adjoint operators -------------------------------------

    def _bilinear_index(self, s, dirs):
        idx, w = self.rings[s].interp(dirs)
        return idx + self.off3d[s], w

    def extract(self, shells, R):
        """3D shell values -> 2D slice values. shells: (n3d,) complex. R: (B,3,3)."""
        dirs = self.slice_directions(R)
        B = R.shape[0]
        out = torch.zeros(B, self.n2d, dtype=shells.dtype, device=self.device)
        for s in range(self.n_shells):
            lo, hi = int(self.off2d[s]), int(self.off2d[s + 1])
            idx, w = self._bilinear_index(s, dirs[:, lo:hi])
            out[:, lo:hi] = (shells[idx] * w.to(shells.dtype)).sum(-1)
        return out

    def insert(self, vals, R, num=None, den=None, weights=None):
        """2D slice values -> accumulate into 3D shells. Exact adjoint of extract().

        num accumulates value*weight, den accumulates weight, so the caller can
        form num/den (the standard gridding reconstruction).
        """
        dirs = self.slice_directions(R)
        if num is None:
            num = torch.zeros(self.n3d, dtype=vals.dtype, device=self.device)
        if den is None:
            den = torch.zeros(self.n3d, dtype=self.dtype, device=self.device)
        for s in range(self.n_shells):
            lo, hi = int(self.off2d[s]), int(self.off2d[s + 1])
            idx, w = self._bilinear_index(s, dirs[:, lo:hi])
            v = vals[:, lo:hi]
            if weights is not None:
                w = w * weights[:, lo:hi, None]
            num.index_put_((idx.reshape(-1),),
                           (v[..., None] * w.to(v.dtype)).reshape(-1), accumulate=True)
            den.index_put_((idx.reshape(-1),), w.reshape(-1), accumulate=True)
        return num, den
