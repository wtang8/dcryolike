"""Same seed must give bitwise-identical results; different seeds must not.

The second half matters as much as the first: a determinism test that passes
because everything returns a constant is vacuous.
"""
import sys, numpy as np, torch
sys.path.insert(0,"/home/wtang/cryoem/dcryolike"); sys.path.insert(0,"/home/wtang/cryoem/cryolike/src")
_FAILS=[]
def _v(c):
    _FAILS.append(bool(c)); return 'PASS' if c else 'FAIL'
from cryolike.grids import PolarGrid
from dcryolike.generator import gaussian_atom_volume, generate, make_ctf, uniform_so3
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft

dev="cuda"; D=32
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=64,uniform=True)
S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=torch.float64,device=dev)
rr=torch.tensor(pg.radius_shells,dtype=torch.float64,device=dev)

def run(seed):
    v=gaussian_atom_volume(D,seed=seed,device=dev)
    p=generate(v,pg,16,0.1,seed=seed,device=dev,ctf_kwargs=dict(astig=0.0))
    az,po,ga=uniform_so3(8,seed=seed,device=dev,dtype=torch.float64)
    T=slice_nufft(v,slice_points(fourier_circles(th,po,az,ga),rr))
    return v,p.images_fourier,p.ctf,T

print("="*64); print("TEST 1: identical seed -> bitwise identical"); print("="*64)
a=run(3); b=run(3)
names=["volume","images","ctf","slices"]
same=[bool(torch.equal(x,y)) for x,y in zip(a,b)]
for n,s_ in zip(names,same): print(f"  {n:<10} identical: {s_}")
print(f"  verdict: {_v(all(same))}")

print("\n"+"="*64); print("TEST 2: different seed -> different (guards against vacuity)"); print("="*64)
c=run(4)
diff=[not bool(torch.equal(x,y)) for x,y in zip(a,c)]
for n,d_ in zip(names,diff): print(f"  {n:<10} differs:   {d_}")
print(f"  verdict: {_v(all(diff))}")

print("\n"+"="*64); print("TEST 3: NUFFT is deterministic on repeat calls"); print("="*64)
v=gaussian_atom_volume(D,seed=1,device=dev)
az,po,ga=uniform_so3(8,seed=1,device=dev,dtype=torch.float64)
pts=slice_points(fourier_circles(th,po,az,ga),rr)
r1=slice_nufft(v,pts); r2=slice_nufft(v,pts)
d=(r1-r2).abs().max().item()
print(f"  max |call1 - call2| = {d:.3e}")
print(f"  verdict: {_v(d==0.0)}   (nonzero would make every comparison unreproducible)")

print(f"\noverall: {'PASS' if all(_FAILS) else 'FAIL'}")
sys.exit(0 if all(_FAILS) else 1)
