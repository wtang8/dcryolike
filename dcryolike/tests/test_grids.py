"""Validate the SphereShell-backed adaptive shell grids (the only grid we keep)."""
import sys, numpy as np, torch
from scipy.special import sph_harm_y
import _paths  # noqa: F401  (sets sys.path relative to this file)
from dcryolike.spheregrid import RingGrid
from dcryolike.adaptive import AdaptiveShellGrid
from dcryolike.generator import gaussian_atom_volume
from dcryolike.slicing import slice_nufft
import sys as _sys
_FAILS=[]
def _v(c,label=''):
    _FAILS.append(bool(c))
    return 'PASS' if c else 'FAIL'
dev="cuda"; D=64; radii=np.arange(1,D//2)

def rand_R(B,seed):
    g=torch.Generator(device="cpu").manual_seed(seed)
    q=torch.randn(B,4,generator=g,dtype=torch.float64); q=q/q.norm(dim=1,keepdim=True)
    w,x,y,z=q.unbind(1)
    return torch.stack([
        torch.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],-1),
        torch.stack([2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],-1),
        torch.stack([2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1)],1).to(dev)

print("="*68); print("TEST 1: rings ascending in theta (the silent-failure guard)"); print("="*68)
G1=RingGrid.for_shell(16.0,device=dev)
d=(G1.theta[1:]-G1.theta[:-1])
print(f"  rings={len(G1.theta)}  min delta-theta={d.min():.4f}  "
      f"{'PASS (ascending)' if d.min()>0 else 'FAIL (not ascending)'}")

print("\n"+"="*68); print("TEST 2: quadrature (pose marginalisation)"); print("="*68)
p=G1.points().cpu().numpy(); w=G1.weights()
th=np.arccos(np.clip(p[:,2],-1,1)); ph=np.arctan2(p[:,1],p[:,0])%(2*np.pi)
worst=max(abs((torch.tensor(sph_harm_y(l,m,th,ph),device=dev)*w).sum().item())
          for l in range(1,17) for m in range(0,l+1))
print(f"  points={G1.n}  sum(w)/4pi={(w.sum()/(4*np.pi)).item():.10f}")
print(f"  worst |int Y_lm|, l<=16 : {worst:.3e}   {_v(worst<1e-4)}")

print("\n"+"="*68); print("TEST 3: radius preserved -> no cross-shell interpolation"); print("="*68)
G=AdaptiveShellGrid.build(radii,device=dev,sphere_factor=2.0)
R=rand_R(8,1); dirs=G.slice_directions(R)
shell_r=torch.cat([G.radii[s].repeat(int(G.n_theta[s])) for s in range(G.n_shells)])
err=((dirs*shell_r[None,:,None]).norm(dim=-1)-shell_r[None,:]).abs().max()
print(f"  max radius deviation: {err:.3e}   {_v(err<1e-12)}")

print("\n"+"="*68); print("TEST 4: insert is the exact adjoint of extract"); print("="*68)
x=torch.randn(G.n3d,dtype=torch.complex128,device=dev)
y=torch.randn(8,G.n2d,dtype=torch.complex128,device=dev)
lhs=(G.extract(x,R).conj()*y).sum(); num,_=G.insert(y,R); rhs=(x.conj()*num).sum()
rel=abs(lhs.item()-rhs.item())/abs(lhs.item())
print(f"  mismatch: {rel:.3e}   {_v(rel<1e-12)}")

print("\n"+"="*68); print("TEST 5: slice accuracy vs exact NUFFT"); print("="*68)
vol=gaussian_atom_volume(D,device=dev); R1=rand_R(16,7)
for f in (1.5,2.0,3.0):
    Gf=AdaptiveShellGrid.build(radii,device=dev,sphere_factor=f)
    sph=Gf.sphere_points()*Gf.sphere_radii()[:,None]
    shells=slice_nufft(vol,sph[None])[0]
    sr=torch.cat([Gf.radii[s].repeat(int(Gf.n_theta[s])) for s in range(Gf.n_shells)])
    dd=Gf.slice_directions(R1); exact=slice_nufft(vol,dd*sr[None,:,None])
    e=((Gf.extract(shells,R1)-exact).abs()**2).sum().sqrt()/((exact.abs()**2).sum().sqrt())
    print(f"  sphere_factor={f:.1f}  3D points={Gf.n3d:>8}  rel err {e:.4f}")

print("\n"+"="*68); print("TEST 6: hierarchical coarse->fine expansion"); print("="*68)
c=RingGrid.for_shell(4.0,device=dev); fn=RingGrid.for_shell(8.0,device=dev)
rmin=c.min_covering_radius_deg(fn)
print(f"  coarse {c.n} px -> fine {fn.n} px | minimum covering radius {rmin:.2f} deg")
print(f"  {'radius':>10}{'coverage':>11}{'missed':>9}{'children':>10}")
ok=True
for r,lab in ((rmin*0.9,"too small"),(rmin,"minimum"),(None,"auto"),(6.0,"6 deg")):
    ch=c.children_of(fn,radius_deg=r)
    reached=torch.unique(ch[ch>=0]).numel()          # unique FINE indices, not pad columns
    cov=100*reached/fn.n
    rr=rmin*1.10 if r is None else r
    print(f"  {rr:>9.2f}d{cov:>10.1f}%{fn.n-reached:>9}{(ch>=0).sum(1).float().mean():>10.1f}"
          f"   {lab}")
    if lab!="too small": ok &= (reached==fn.n)
    else: ok &= (reached<fn.n)                        # must actually under-cover
print(f"  verdict: {_v(ok)} "
      f"(auto and 6 deg cover fully; below the minimum does not)")

_sys.exit(0 if all(_FAILS) else 1)   # exit code so a single run is meaningful
