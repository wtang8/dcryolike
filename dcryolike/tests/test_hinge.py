"""Is the hinge forward model exact?

hinge.hinge_slice claims that rotating a domain about an axis is, in Fourier space,
a slice of the SAME domain density at a composed orientation times a phase ramp:

    F_theta(k) = F_core(k) + F_dom(R_theta^T k) exp(2 pi i (R_theta^T k - k) . p / D)

If that identity is right the conformation costs one extra slice per latent grid
point and nothing else. If the phase or the rotation sense is wrong the error is a
subtle shift that a casual look would miss, so it is checked against a structure
deformed in REAL SPACE by moving its atoms and rasterising, which shares no code
with the Fourier path.

Two different questions are at stake and they need different tests.

  IS THE CODE THE FORMULA?  Exact to machine precision, checked by evaluating the
  right-hand side independently (TESTS 1 and 3). Nothing here can hide a sign error
  in the phase or a transposed rotation.

  IS THE FORMULA THE RIGHT PHYSICS?  Checked against a structure deformed in REAL
  SPACE by moving its atoms and rasterising (TESTS 2 and 4). This can only ever
  agree to within rendering interpolation, because the reference re-rasterises
  rotated atoms while the fast path rotates the already-sampled density. The error
  therefore GROWS with angle, and the test asserts it stays at that level rather
  than at the level of a structurally wrong answer -- for which TEST 4 supplies the
  control, since the phantom's moving arm is only ~13% of its mass and a wrong sign
  costs only ~10% of peak, not O(1).
"""
import os, math, numpy as np, torch, sys
import _paths  # noqa: F401
from cryolike.grids import PolarGrid
from dcryolike.generator import _phantom_atoms, _render_atoms, uniform_so3
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft
from dcryolike.hinge import rodrigues, hinge_slice, render_hinged

_FAILS=[]
def _v(c): _FAILS.append(bool(c)); return 'PASS' if c else 'FAIL'
dev="cuda" if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES","x")!="" else "cpu"
BK="nufft" if dev=="cuda" else "gridding"
D=int(os.environ.get("D","48")); DT=torch.float64
torch.manual_seed(0)

# the phantom's own split: body = core, helical arm = the moving domain
body,helix=_phantom_atoms(D,dtype=DT)
core=_render_atoms(body,D,device=dev,dtype=DT)
dom =_render_atoms(helix,D,device=dev,dtype=DT)
AXIS_P=helix[0].clone()                       # hinge at the base of the arm
AXIS_D=torch.tensor([1.0,0.0,0.0],dtype=DT)

NT=12
az,po,ga=uniform_so3(NT,seed=3,device=dev,dtype=DT)
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=64,uniform=True)
th=torch.tensor(pg.theta_shell,dtype=DT,device=dev)
rr=torch.tensor(pg.radius_shells,dtype=DT,device=dev)
pts=slice_points(fourier_circles(th,po,az,ga),rr)

def relerr(a,b):
    return ((a-b).abs().max()/b.abs().max()).item()

print("="*66); print("TEST 1: theta = 0 reproduces the undeformed slice"); print("="*66)
ref=slice_nufft(core+dom,pts) if dev=="cuda" else None
if ref is None:
    from dcryolike.slicing import fourier_volume, slice_gridding
    ref=slice_gridding(fourier_volume(core+dom),pts)
got=hinge_slice(core,dom,pts,AXIS_P,AXIS_D,0.0,D,backend=BK)
e=relerr(got,ref)
print(f"  max rel err {e:.2e}   {_v(e<1e-10)}")

print("\n"+"="*66); print("TEST 2: identity vs a real-space rendered rotation"); print("="*66)
ok2=True
for deg in (5.0,15.0,30.0,-20.0):
    t=math.radians(deg)
    vol=render_hinged(core,helix,AXIS_P,AXIS_D,t,D,device=dev,dtype=DT)
    if dev=="cuda":
        ref=slice_nufft(vol,pts)
    else:
        from dcryolike.slicing import fourier_volume, slice_gridding
        ref=slice_gridding(fourier_volume(vol),pts)
    got=hinge_slice(core,dom,pts,AXIS_P,AXIS_D,t,D,backend=BK)
    e=relerr(got,ref)
    # the reference rasterises rotated atoms; the fast path rotates the sampled
    # density, so they differ by the interpolation error of the rendering grid
    print(f"  theta {deg:+6.1f} deg: max rel err {e:.2e}")
    ok2 &= e<0.05
print(f"  agree to better than 5% of peak at every angle: {_v(ok2)}")

print("\n"+"="*66); print("TEST 3: the code evaluates the stated formula, exactly"); print("="*66)
from dcryolike.generator import _slice as _sl
t=math.radians(17.0)
R=rodrigues(AXIS_D.to(dev),torch.tensor(t,dtype=DT,device=dev))
q=pts@R
# (a) axis through the origin: pure resampling, the phase must be identically 1
direct=hinge_slice(core*0,dom,pts,torch.zeros(3,dtype=DT),AXIS_D,t,D,backend=BK)
plain=_sl(dom,q,BK)
e0=relerr(direct,plain)
# (b) axis off the origin: the phase term itself, evaluated independently here
pref=AXIS_P.to(dev)
phase=torch.exp(2j*np.pi*((q-pts)@pref)/D)
indep=_sl(core,pts,BK)+_sl(dom,q,BK)*phase.to(plain.dtype)
got=hinge_slice(core,dom,pts,AXIS_P,AXIS_D,t,D,backend=BK)
e1=relerr(got,indep)
print(f"  axis at origin, phase must vanish : max rel err {e0:.2e}")
print(f"  axis off origin, phase term       : max rel err {e1:.2e}")
print(f"  code == formula to machine precision: {_v(e0<1e-12 and e1<1e-12)}")

print("\n"+"="*66); print("TEST 4: the sense of rotation (hand-flip hazard)"); print("="*66)
t=math.radians(25.0)
volp=render_hinged(core,helix,AXIS_P,AXIS_D,t,D,device=dev,dtype=DT)
if dev=="cuda":
    refp=slice_nufft(volp,pts)
else:
    from dcryolike.slicing import fourier_volume, slice_gridding
    refp=slice_gridding(fourier_volume(volp),pts)
gp=hinge_slice(core,dom,pts,AXIS_P,AXIS_D,t,D,backend=BK)
gm=hinge_slice(core,dom,pts,AXIS_P,AXIS_D,-t,D,backend=BK)
g0=hinge_slice(core,dom,pts,AXIS_P,AXIS_D,0.0,D,backend=BK)
ep, em, e0 = relerr(gp,refp), relerr(gm,refp), relerr(g0,refp)
# e0 and em are both "the arm is in the wrong place"; the arm is ~13% of the mass,
# so that is the scale a structurally wrong answer costs. ep must sit well below it.
print(f"  +theta {ep:.2e}   -theta {em:.2e}   theta=0 {e0:.2e}  (wrong-arm scale)")
print(f"  the correct sign is at the interpolation level, the others at the wrong-arm "
      f"level: {_v(ep < 0.4*min(em,e0))}")

print("\n"+("ALL PASS" if all(_FAILS) else "SOME FAIL"), f"({sum(_FAILS)}/{len(_FAILS)})")
sys.exit(0 if all(_FAILS) else 1)
