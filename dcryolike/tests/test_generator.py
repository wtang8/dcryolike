import sys, numpy as np, torch
import _paths  # noqa: F401  (sets sys.path relative to this file)
from cryolike.grids import PolarGrid
from dcryolike.generator import gaussian_atom_volume, generate, uniform_so3, make_ctf
import sys as _sys
_FAILS=[]
def _v(c,label=''):
    _FAILS.append(bool(c))
    return 'PASS' if c else 'FAIL'

dev = "cuda"; D = 48
vol = gaussian_atom_volume(D, device=dev)
pg = PolarGrid(radius_max=D/2, dist_radii=1.0, n_inplanes=128, uniform=True)
print(f"volume {tuple(vol.shape)}  grid: {pg.n_shells} shells x {pg.n_inplanes} inplanes")

print("\n" + "="*62); print("TEST 1: realised SNR matches requested SNR"); print("="*62)
ok = True
for snr in (1.0, 0.5, 0.1, 0.05, 0.01):
    p = generate(vol, pg, 256, snr, seed=1, device=dev)
    w = torch.tensor(pg.integration_weight_points, dtype=vol.dtype, device=dev).reshape(pg.n_shells, pg.n_inplanes)
    sig = ((p.clean_fourier.abs()**2 * w).sum(dim=(1,2))/w.sum())
    nz  = p.images_fourier - p.clean_fourier
    noi = ((nz.abs()**2 * w).sum(dim=(1,2))/w.sum())
    meas = (sig/noi).mean().item()
    good = abs(meas-snr)/snr < 0.10
    ok &= good
    print(f"  requested {snr:<6} realised {meas:8.4f}   rel dev {abs(meas-snr)/snr:6.2%}  {'ok' if good else 'BAD'}")
print(f"  verdict: {_v(ok)}")

print("\n" + "="*62); print("TEST 2: nufft vs gridding clean signal agree"); print("="*62)
a = generate(vol, pg, 32, 1.0, seed=2, device=dev, backend="nufft").clean_fourier
b = generate(vol, pg, 32, 1.0, seed=2, device=dev, backend="gridding").clean_fourier
cos = (torch.sum(a.conj()*b).real/(a.norm()*b.norm())).item()
print(f"  cosine similarity = {cos:.6f}   {_v(cos > 0.97)}")

print("\n" + "="*62); print("TEST 3: CTF is a real CTF"); print("="*62)
c = make_ctf(pg, 4, float(D), device=dev, dtype=vol.dtype)
zc = ((c[0,:-1,0]*c[0,1:,0]) < 0).sum().item()
print(f"  range [{c.min():.3f}, {c.max():.3f}]  (must lie within [-1,1])")
print(f"  zero crossings along radius = {zc}  {_v(zc >= 1 and c.abs().max() <= 1.0001)}")

print("\n" + "="*62); print("TEST 4: phantom is chiral (can detect a handedness bug)"); print("="*62)
mir = torch.flip(vol, dims=(2,))
best = 0.0
for k in range(4):   # cheap check over a few in-plane rotations of the mirror
    r = torch.rot90(mir, k, dims=(0,1))
    best = max(best, (torch.sum(vol*r)/(vol.norm()*r.norm())).item())
print(f"  best corr(volume, mirrored) = {best:.4f}   "
      f"{'PASS (chiral)' if best < 0.85 else 'FAIL (too symmetric)'}")

print("\n" + "="*62); print("TEST 5: ground-truth pose is the likelihood argmax at high SNR"); print("="*62)
p = generate(vol, pg, 1, 100.0, seed=7, device=dev)
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft
th = torch.tensor(pg.theta_shell, dtype=vol.dtype, device=dev)
rr = torch.tensor(pg.radius_shells, dtype=vol.dtype, device=dev)
az, po, ga = uniform_so3(2000, seed=11, device=dev, dtype=vol.dtype)
az = torch.cat([az, p.azimus]); po = torch.cat([po, p.polars]); ga = torch.cat([ga, p.gammas])
T = slice_nufft(vol, slice_points(fourier_circles(th, po, az, ga), rr))
T = T.reshape(-1, pg.n_shells, pg.n_inplanes) * p.ctf[0][None]
img = p.images_fourier[0][None]
cc = (T.conj()*img).sum(dim=(1,2)).real / (T.reshape(T.shape[0],-1).norm(dim=1)*img.norm())
best_i = cc.argmax().item()
print(f"  true pose index = {len(cc)-1}, argmax = {best_i}, "
      f"cc_true = {cc[-1]:.4f}, cc_max = {cc.max():.4f}")
print(f"  verdict: {_v(best_i == len(cc)-1)}")

_sys.exit(0 if all(_FAILS) else 1)   # exit code so a single run is meaningful
