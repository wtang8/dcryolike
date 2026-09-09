"""Can the physical likelihood tell a right hinge hypothesis from a wrong one?

THIS IS THE FOUNDATIONAL TEST OF THE AGENT-GUIDED PROGRAMME. The design is that an
agent proposes mechanical hypotheses from what it knows about the protein -- which
loop is a hinge, which domain swings -- and the physical posterior likelihood decides
between them, rather than a latent representation inventing density shifts. That is
only sound if the likelihood can REJECT a confident but wrong proposal. If a wrong
hinge scores as well as the right one, an agent's hallucination would be indis-
tinguishable from its insight and the whole scheme is circular.

WHAT IS COMPARED. Every hypothesis is scored by the same marginal log-likelihood, with
the gain integrated analytically, the in-plane angle integrated over the ring, and the
hinge angle integrated by quadrature over a uniform grid. Reported per particle, in
nats, relative to the true hypothesis. All hypotheses see identical particles.

    H1  the true axis                                (base of the arm, x direction)
    H0  rigid: no motion at all                      (the null)
    H2  axis displaced to the middle of the arm      (right idea, wrong place)
    H3  axis at the true point, direction rotated 90 deg
    H4  axis at the true point, direction rotated 30 deg   (a near miss)
    H5  axis through the static core                 (wrong part of the molecule)
    H6  the segmentation swapped: the body is treated as the moving domain

WHAT WOULD MAKE THIS UNCONVINCING, and is therefore controlled for:

  * Model complexity. Every hypothesis here has the same number of free parameters
    (one hinge angle per particle, marginalised) except H0, which has none, so a raw
    likelihood comparison is fair between H1-H6. H0 is quoted separately since it is
    nested and simpler.
  * Volume fitting. All hypotheses are given the SAME true core and domain volumes, so
    none can rescue itself by refitting density to match its own wrong geometry. That
    makes this the EASY version of the question; the harder version, where each
    hypothesis fits its own volumes, needs the M-step and is the natural follow-up.
  * Chance. The evidence gap is reported against its standard error across particles,
    so a gap that is merely noise is visible as such.
"""
import os, math, numpy as np, torch, sys
import _paths  # noqa: F401
from cryolike.grids import PolarGrid
from dcryolike.generator import _phantom_atoms, _render_atoms, generate_physical_hetero
from dcryolike.slicing import fourier_circles, slice_points
from dcryolike.hinge import hinge_logweights, render_hinged, rodrigues
from dcryolike.pipeline import temperature

_FAILS=[]
def _v(c): _FAILS.append(bool(c)); return 'PASS' if c else 'FAIL'
dev="cuda"; DT=torch.float32
D=int(os.environ.get("D","96")); NP=int(os.environ.get("NP","400"))
NCONF=16; SWING=math.radians(float(os.environ.get("SWING","15"))); ISO=dict(astig=0.0)
NTHETA=int(os.environ.get("NTHETA","21"))
SNRS=[float(x) for x in os.environ.get("SNRS","0.3,0.1").split(",")]

body,helix=_phantom_atoms(D,dtype=DT)
core=_render_atoms(body,D,device=dev,dtype=DT)
dom=_render_atoms(helix,D,device=dev,dtype=DT)
TRUE_P=helix[0].clone(); TRUE_D=torch.tensor([1.0,0.0,0.0],dtype=DT)
ts=list(np.linspace(-SWING,SWING,NCONF))
vols=[render_hinged(core,helix,TRUE_P,TRUE_D,float(t),D,device=dev,dtype=DT) for t in ts]

def rot(v,axis,deg):
    R=rodrigues(torch.as_tensor(axis,dtype=DT),torch.tensor(math.radians(deg),dtype=DT))
    return (R@torch.as_tensor(v,dtype=DT))

HYP=[
 ("H1 true axis",                 TRUE_P,               TRUE_D,                     dom,  core),
 ("H2 axis mid-arm",              helix[30].clone(),    TRUE_D,                     dom,  core),
 ("H3 direction rotated 90 deg",  TRUE_P,               torch.tensor([0.,1.,0.],dtype=DT), dom, core),
 ("H4 direction rotated 30 deg",  TRUE_P,               rot(TRUE_D,[0.,0.,1.],30.), dom,  core),
 ("H5 axis through the core",     body.mean(0).clone(), TRUE_D,                     dom,  core),
 ("H6 segmentation swapped",      TRUE_P,               TRUE_D,                     core, dom ),
]

P_IN=min(256,int(2**np.ceil(np.log2(2*np.pi*(D//2)*0.85))))
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=P_IN,uniform=True); S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=DT,device=dev); rr=torch.tensor(pg.radius_shells,dtype=DT,device=dev)
w=torch.tensor(pg.integration_weight_points,dtype=DT,device=dev).reshape(S,P)
grid=torch.tensor(np.linspace(-SWING*1.3,SWING*1.3,NTHETA),dtype=DT,device=dev)
zero=torch.zeros(1,dtype=DT,device=dev)
conf=torch.arange(NP,device=dev)%NCONF

def evidence(p,beta,pts,dm,cr,ap,ad,tgrid,B=16):
    """Marginal log-likelihood per particle, hinge angle integrated over tgrid."""
    out=[]
    for i in range(0,NP,B):
        j=min(i+B,NP)
        lw,_=hinge_logweights(p.images_fourier[i:j],p.ctf[i:j],w,beta[i:j],cr,dm,
                              pts[i:j],tgrid,ap,ad,D)
        out.append(torch.logsumexp(lw[:,0].double(),dim=1)-math.log(len(tgrid)))
    return torch.cat(out)

print(f"D={D}  {NP} particles  {NTHETA}-point hinge grid  true swing +-{math.degrees(SWING):.0f} deg")
print("Marginal log-likelihood per particle, nats, relative to the true hypothesis.")
print("Negative means the hypothesis is worse. +- is the standard error over particles.\n")
for snr in SNRS:
    p=generate_physical_hetero(vols,ts,conf,pg,snr,seed=17,device=dev,ctf_kwargs=ISO)
    beta,tau,C=temperature(p,w,D)
    pts=slice_points(fourier_circles(th,p.polars,p.azimus,p.gammas),rr)[:,None]  # true pose
    ev={}
    for name,ap,ad,dm,cr in HYP:
        ev[name]=evidence(p,beta,pts,dm,cr,ap,ad,grid)
    ev["H0 rigid, no motion"]=evidence(p,beta,pts,dom,core,TRUE_P,TRUE_D,zero)
    ref=ev["H1 true axis"]
    print(f"=== SNR {snr}")
    order=["H1 true axis","H4 direction rotated 30 deg","H2 axis mid-arm",
           "H3 direction rotated 90 deg","H5 axis through the core",
           "H6 segmentation swapped","H0 rigid, no motion"]
    for name in order:
        d=(ev[name]-ref)
        se=d.std().item()/math.sqrt(NP)
        print(f"  {name:32s} {d.mean().item():+10.2f}  +- {se:5.2f}")
    worst=min((ev[n]-ref).mean().item() for n in order if n!="H1 true axis")
    best_wrong=max((ev[n]-ref).mean().item() for n in order if n!="H1 true axis")
    se_bw=max((ev[n]-ref).std().item()/math.sqrt(NP) for n in order if n!="H1 true axis")
    print(f"  closest rival is {best_wrong:+.2f} nats behind ({abs(best_wrong)/se_bw:.0f} "
          f"standard errors): {_v(best_wrong < -3*se_bw)}")
    del p; torch.cuda.empty_cache()
    print()

print("A hypothesis behind by many nats per particle is rejected overwhelmingly once")
print("the evidence is summed over a real stack. The question this answers is whether")
print("the likelihood can reject a WRONG proposal, which is what makes an agent's")
print("hypothesis testable rather than merely plausible.")
print(("ALL PASS" if all(_FAILS) else "SOME FAIL"), f"({sum(_FAILS)}/{len(_FAILS)})")
sys.exit(0 if all(_FAILS) else 1)
