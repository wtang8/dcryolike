"""Validate generate_physical: the polar-domain bridge must be exact, and the
real-space SNR must be what was requested."""
import _paths  # noqa
import sys, numpy as np, torch
_FAILS=[]
def _v(c): _FAILS.append(bool(c)); return 'PASS' if c else 'FAIL'
from cryolike.grids import PolarGrid
from dcryolike.generator import (gaussian_atom_volume, generate_physical,
    real_space_projections, _polar_dft_matrix, uniform_so3)
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft
dev="cuda"; D=48; N=48
vol=gaussian_atom_volume(D,device=dev)
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=128,uniform=True)
S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=vol.dtype,device=dev)
rr=torch.tensor(pg.radius_shells,dtype=vol.dtype,device=dev)

print("="*66); print("TEST 1: DFT-matrix bridge agrees with slice_nufft on the clean signal"); print("="*66)
az,po,ga=uniform_so3(N,seed=3,device=dev,dtype=vol.dtype)
tmpl=slice_nufft(vol,slice_points(fourier_circles(th,po,az,ga),rr)).reshape(N,S,P)
proj=real_space_projections(vol,az,po,ga,dev)
M=_polar_dft_matrix(pg,D,dev)
via=(M@proj.reshape(N,-1).to(torch.complex128).T).T.reshape(N,S,P)
sc=torch.sqrt((tmpl.abs()**2).sum(dim=(1,2))/(via.abs()**2).sum(dim=(1,2)))
via=via*sc[:,None,None]
cos=(torch.sum(tmpl.conj()*via).real/(tmpl.norm()*via.norm())).item()
print(f"  cosine(slice_nufft, DFT(projection)) = {cos:.8f}")
print(f"  scale factor spread across particles  = {sc.std().item()/sc.mean().item():.2e}  (should be ~constant)")
print(f"  verdict: {_v(cos>0.9999 and sc.std()/sc.mean()<1e-3)}")

print("\n"+"="*66); print("TEST 2: realised real-space SNR == requested"); print("="*66)
ok=True
for snr in (1.0,0.1,0.05,0.01):
    p=generate_physical(vol,pg,N,snr,seed=5,device=dev,ctf_kwargs=dict(astig=0.0))
    proj=real_space_projections(vol,p.azimus,p.polars,p.gammas,dev)
    real=(proj.var(dim=(1,2))/p.sigma_noise**2).mean().item()
    good=abs(real-snr)/snr<0.02; ok&=good
    print(f"  requested {snr:<5} realised {real:.4f}   {'ok' if good else 'BAD'}")
print(f"  verdict: {_v(ok)}")

print("\n"+"="*66); print("TEST 3: physical noise IS correlated (guards against regressing to iid)"); print("="*66)
p=generate_physical(vol,pg,N,0.1,seed=7,device=dev,ctf_kwargs=dict(astig=0.0))
nz=p.images_fourier-p.clean_fourier
a,b=nz[:,:,0],nz[:,:,1]
corr=((a*b.conj()).sum(0)/torch.sqrt((a.abs()**2).sum(0)*(b.abs()**2).sum(0))).real
print(f"  adjacent-point noise correlation: inner {corr[0]:.3f}  mid {corr[S//2]:.3f}  outer {corr[-1]:.3f}")
print(f"  verdict: {_v(corr[0]>0.9 and abs(corr[-1])<0.4)}   (iid would give ~0 everywhere)")
print(f"\noverall: {'PASS' if all(_FAILS) else 'FAIL'}")
sys.exit(0 if all(_FAILS) else 1)
