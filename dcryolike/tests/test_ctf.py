"""Check our CTF against CryoLike's reference _ctf_relion.

Both implement the same RELION formula, so they must agree to machine precision.
Any drift means one of them has a sign, unit, or convention error -- exactly the
class of bug that produces plausible-looking but wrong maps.
"""
import sys, numpy as np, torch
import _paths  # noqa: F401  (sets sys.path relative to this file)
from cryolike.grids import PolarGrid
from cryolike.metadata import LensDescriptor
from cryolike.microscopy.ctf import _ctf_relion
from dcryolike.generator import make_ctf

dev="cuda"; D=48; BOX=float(D)
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=128,uniform=True)
ok=True

CASES=[
    ("typical, astigmatic",  [12000.,  8000.], [12200.,  8300.], [ 30.,  75.], 300., 2.7, 0.10),
    ("isotropic (dfU=dfV)",  [15000., 20000.], [15000., 20000.], [  0.,   0.], 300., 2.7, 0.10),
    ("200 kV, low Cs",       [ 9000., 25000.], [ 9400., 25600.], [110., 165.], 200., 0.01, 0.07),
    ("high defocus",         [40000., 55000.], [40800., 55900.], [ 15., 140.], 300., 2.7, 0.10),
]

print(f"{'case':>24}{'max |ours - cryolike|':>24}{'rel':>12}")
print("-"*62)
for name,dfu,dfv,dfang_deg,volt,cs,amp in CASES:
    n=len(dfu)
    lens=LensDescriptor(
        defocusU=np.array(dfu), defocusV=np.array(dfv),
        defocusAngle=np.array(dfang_deg),          # degrees; LensDescriptor converts
        phaseShift=np.array([0.]*n),
        sphericalAberration=cs, voltage=volt, amplitudeContrast=amp,
        defocusAngle_degree=True, phaseShift_degree=True)
    ref=torch.tensor(np.asarray(_ctf_relion(lens,pg,box_size=BOX,anisotropy=True)),
                     dtype=torch.float64,device=dev)
    ours=make_ctf(pg,n,BOX,device=dev,dtype=torch.float64,
                  voltage=volt,cs=cs,amp_contrast=amp,
                  defocusU=dfu,defocusV=dfv,
                  defocusAngle=np.radians(dfang_deg))
    d=(ours-ref).abs().max().item(); rel=d/ref.abs().max().item()
    good=d<1e-10; ok&=good
    print(f"{name:>24}{d:>24.3e}{rel:>12.2e}   {'PASS' if good else 'FAIL'}")

print("\n"+"="*62)
print("range and structure checks")
print("="*62)
c=make_ctf(pg,4,BOX,device=dev,dtype=torch.float64,astig=0.0)
r_ok=bool((c.abs()<=1.0+1e-12).all()); ok&=r_ok
print(f"  |CTF| <= 1 everywhere            : {r_ok}")
zc=int(((c[0,:-1,0]*c[0,1:,0])<0).sum()); z_ok=zc>=1; ok&=z_ok
print(f"  zero crossings along radius      : {zc}  ({'PASS' if z_ok else 'FAIL'})")
iso=bool((c[0]-c[0,:,:1]).abs().max()<1e-12); ok&=iso
print(f"  astig=0 gives isotropic CTF      : {iso}")

# Exact analytic check of the envelope rather than "it got smaller".
# NOTE box_size is in Angstrom, so BOX=D means 1 A/pixel and the outer shell
# sits near 1 A^-1 -- B=200 there is exp(-46), i.e. total extinction. Use a
# small B so the comparison exercises the whole radial range meaningfully.
Bf=8.0
cb=make_ctf(pg,4,BOX,device=dev,dtype=torch.float64,astig=0.0,bfactor=Bf)
rr=torch.tensor(pg.radius_shells,dtype=torch.float64,device=dev)*2.0/BOX
expect=torch.exp(-Bf*rr**2/4.0)
got=(cb[0,:,0]/c[0,:,0])
d=(got-expect).abs().max().item()
b_ok=d<1e-12; ok&=b_ok
print(f"  B-factor envelope == exp(-B k^2/4) : max dev {d:.2e}  ({'PASS' if b_ok else 'FAIL'})")
print(f"    envelope at outer shell (B={Bf:.0f}) : x{expect[-1].item():.4f}")
print(f"    (CryoLike parses ctfBfactor but never applies it; ours is opt-in)")

print(f"\noverall: {'PASS' if ok else 'FAIL'}")
import sys as _sys; _sys.exit(0 if ok else 1)
