"""Does the joint quadrature recover the hinge angle?

The E-step integrates over the in-plane angle and the gain analytically and over the
hinge angle by quadrature on a grid. This checks it in ISOLATION -- true pose, true
core and domain volumes supplied -- so that a failure here is the estimator and not
the fit. It is the hinge counterpart of the linear model's E-step check, which found
rho 0.85 at the true pose and told us that model was sound before any EM ran.

TEST 1  the posterior over the hinge angle peaks at the true angle, and its mean
        tracks the truth across particles (rank correlation and slope).
TEST 2  the posterior sharpens as SNR rises -- an estimator that ignored the data
        would give the same width at every noise level.
TEST 3  a NEGATIVE CONTROL: with the domain volume replaced by noise of the same
        power, the correlation with the true angle collapses. Without this, a
        quadrature that merely reproduced the prior would look like success.
"""
import os, math, numpy as np, torch, sys
import _paths  # noqa: F401
from cryolike.grids import PolarGrid
from dcryolike.generator import _phantom_atoms, _render_atoms, generate_physical_hetero
from dcryolike.slicing import fourier_circles, slice_points
from dcryolike.hinge import hinge_logweights, render_hinged
from dcryolike.pipeline import temperature

_FAILS=[]
def _v(c): _FAILS.append(bool(c)); return 'PASS' if c else 'FAIL'
dev="cuda"; DT=torch.float32
D=int(os.environ.get("D","96")); NP=int(os.environ.get("NP","300"))
NCONF=16; SWING=math.radians(15.0); ISO=dict(astig=0.0)
NTHETA=int(os.environ.get("NTHETA","21"))

body,helix=_phantom_atoms(D,dtype=DT)
core=_render_atoms(body,D,device=dev,dtype=DT)
AXIS_P=helix[0].clone(); AXIS_D=torch.tensor([1.0,0.0,0.0],dtype=DT)
dom=_render_atoms(helix,D,device=dev,dtype=DT)
ts=list(np.linspace(-SWING,SWING,NCONF))
vols=[render_hinged(core,helix,AXIS_P,AXIS_D,float(t),D,device=dev,dtype=DT) for t in ts]

P_IN=min(256,int(2**np.ceil(np.log2(2*np.pi*(D//2)*0.85))))
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=P_IN,uniform=True); S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=DT,device=dev); rr=torch.tensor(pg.radius_shells,dtype=DT,device=dev)
w=torch.tensor(pg.integration_weight_points,dtype=DT,device=dev).reshape(S,P)
grid=torch.tensor(np.linspace(-SWING*1.3,SWING*1.3,NTHETA),dtype=DT,device=dev)
conf=torch.arange(NP,device=dev)%NCONF
print(f"D={D} grid {S}x{P}  {NP} particles  hinge grid {NTHETA} points over "
      f"+-{math.degrees(SWING*1.3):.0f} deg  (truth spans +-{math.degrees(SWING):.0f})")

def run(snr, dom_used):
    p=generate_physical_hetero(vols,ts,conf,pg,snr,seed=17,device=dev,ctf_kwargs=ISO)
    beta,tau,C=temperature(p,w,D)
    pts=slice_points(fourier_circles(th,p.polars,p.azimus,p.gammas),rr)[:,None]   # true pose, M=1
    lw=[];
    for i in range(0,NP,16):
        j=min(i+16,NP)
        l,_=hinge_logweights(p.images_fourier[i:j],p.ctf[i:j],w,beta[i:j],core,dom_used,
                             pts[i:j],grid,AXIS_P,AXIS_D,D)
        lw.append(l[:,0])
    lw=torch.cat(lw)                                     # (NP, NTHETA)
    pw=torch.softmax(lw.double(),dim=1)
    tmean=(pw*grid.double()[None]).sum(1)
    ttrue=p.t.double()
    var=(pw*(grid.double()[None]-tmean[:,None])**2).sum(1)
    del p; torch.cuda.empty_cache()
    return tmean,ttrue,var.sqrt(),pw

def spearman(a,b):
    ra=a.argsort().argsort().double(); rb=b.argsort().argsort().double()
    ra=ra-ra.mean(); rb=rb-rb.mean()
    return ((ra*rb).sum()/(ra.norm()*rb.norm())).item()

print("\n"+"="*70); print("TEST 1 & 2: recovery of the hinge angle, and its sharpening with SNR"); print("="*70)
print(f"{'SNR':>6}{'rho':>8}{'slope':>8}{'post sd':>10}{'true sd':>10}")
res={}
for snr in (1.0,0.3,0.1):
    tm,tt,sd,pw=run(snr,dom)
    r=spearman(tm,tt)
    slope=(((tm-tm.mean())*(tt-tt.mean())).sum()/((tt-tt.mean())**2).sum()).item()
    res[snr]=(r,slope,sd.median().item())
    print(f"{snr:>6.2f}{r:>8.3f}{slope:>8.3f}{math.degrees(sd.median().item()):>9.2f}°"
          f"{math.degrees(tt.std().item()):>9.2f}°")
r3,s3,_=res[0.3]
print(f"  recovers the angle at SNR 0.3 (rho > 0.8, slope within 30% of 1): "
      f"{_v(r3>0.8 and abs(s3-1)<0.3)}")
print(f"  posterior sharpens as SNR rises ({math.degrees(res[0.1][2]):.2f}° -> "
      f"{math.degrees(res[1.0][2]):.2f}°): {_v(res[1.0][2] < res[0.1][2])}")

print("\n"+"="*70); print("TEST 3: negative control, domain replaced by noise"); print("="*70)
g=torch.Generator(device=dev).manual_seed(7)
fake=torch.randn(D,D,D,generator=g,dtype=DT,device=dev)
fake=fake*(dom.norm()/fake.norm())
tmf,ttf,sdf,_=run(0.3,fake)
rf=spearman(tmf,ttf)
print(f"  rho with the true domain {res[0.3][0]:.3f}, with a noise domain {rf:.3f}")
print(f"  the signal is the domain, not the quadrature: {_v(abs(rf) < 0.3*res[0.3][0])}")

print("\n"+("ALL PASS" if all(_FAILS) else "SOME FAIL"), f"({sum(_FAILS)}/{len(_FAILS)})")
sys.exit(0 if all(_FAILS) else 1)
