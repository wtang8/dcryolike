"""Full EM under each hinge hypothesis, scored on held-out particles.

THE HARD VERSION OF THE DISCRIMINATION TEST. The earlier version handed every
hypothesis the same true core and domain volumes, so a wrong hinge could not rescue
itself by reshaping density to match its own geometry. That is the adversary that
matters: given freedom to refit, can a wrong mechanical hypothesis explain the data as
well as the right one?

PROTOCOL, designed so a wrong hypothesis gets every fair advantage:
  * Each hypothesis runs its OWN EM: the hinge angle is marginalised per particle over
    a quadrature grid, and the core and domain volumes are re-fitted each iteration by
    the block scheme (subtract the other volume's prediction, insert the residual).
  * No hypothesis is given the true volumes. All start from the same low-passed
    CONSENSUS, split by that hypothesis's own mask. A hypothesis that claims the wrong
    part of the molecule moves therefore starts from a correspondingly wrong split,
    which is the honest consequence of its claim.
  * Fitting uses one half of the particles; scoring uses the other half, which the fit
    never saw. That is what makes the comparison fair across hypotheses that all fit
    far more parameters (two volumes) than the hinge itself.
  * Poses are fixed at the truth. This isolates the conformational model from the pose
    search; a wrong hypothesis cannot hide behind pose error, and neither can the right
    one. Adding the pose search is the next step, not this one.

The mask IS the hypothesis, in the sense that an agent proposing "this domain swings
about that loop" supplies exactly a mask and an axis. Deriving the mask from the true
atom sets means the agent identified the domain correctly; H6 is the case where it did
not.
"""
import os, math, numpy as np, torch, sys, time
import _paths  # noqa: F401
from cryolike.grids import PolarGrid
from dcryolike.generator import _phantom_atoms, _render_atoms, generate_physical_hetero
from dcryolike.slicing import fourier_circles, slice_points, fourier_volume
from dcryolike.hinge import hinge_logweights, hinge_mstep, render_hinged, rodrigues
from dcryolike.hetero import HeteroAccumulator
from dcryolike.pipeline import temperature, ShellMasks, to_real, duty_sleep, GPU_DUTY

_FAILS=[]
def _v(c): _FAILS.append(bool(c)); return 'PASS' if c else 'FAIL'
dev="cuda"; DT=torch.float32
D=int(os.environ.get("D","96")); NP=int(os.environ.get("NP","2000"))
NCONF=16; SWING=math.radians(float(os.environ.get("SWING","15"))); ISO=dict(astig=0.0)
NTHETA=int(os.environ.get("NTHETA","15")); ITERS=int(os.environ.get("ITERS","5"))
SNR=float(os.environ.get("SNR","0.3")); R_INIT=int(os.environ.get("R_INIT","10"))
EB=int(os.environ.get("EB","8"))

body,helix=_phantom_atoms(D,dtype=DT)
core_t=_render_atoms(body,D,device=dev,dtype=DT)
dom_t =_render_atoms(helix,D,device=dev,dtype=DT)
TRUE_P=helix[0].clone(); TRUE_D=torch.tensor([1.0,0.0,0.0],dtype=DT)
ts=list(np.linspace(-SWING,SWING,NCONF))
vols=[render_hinged(core_t,helix,TRUE_P,TRUE_D,float(t),D,device=dev,dtype=DT) for t in ts]
cons=torch.stack(vols).mean(0).contiguous()

def rot(v,axis,deg):
    R=rodrigues(torch.as_tensor(axis,dtype=DT),torch.tensor(math.radians(deg),dtype=DT))
    return (R@torch.as_tensor(v,dtype=DT))
def softmask(mob,sta):
    """Soft partition of the consensus: the fraction of local density owned by `mob`."""
    a=_render_atoms(mob,D,device=dev,dtype=DT); b=_render_atoms(sta,D,device=dev,dtype=DT)
    return (a/(a+b+1e-6)).clamp(0,1)
M_ARM=softmask(helix,body); M_BODY=softmask(body,helix)

HYP=[
 ("H1 true axis",                TRUE_P,            TRUE_D,                            M_ARM),
 ("H4 direction off 30 deg",     TRUE_P,            rot(TRUE_D,[0.,0.,1.],30.),        M_ARM),
 ("H2 axis mid-arm",             helix[30].clone(), TRUE_D,                            M_ARM),
 ("H3 direction off 90 deg",     TRUE_P,            torch.tensor([0.,1.,0.],dtype=DT), M_ARM),
 ("H5 axis through the core",    body.mean(0).clone(), TRUE_D,                         M_ARM),
 ("H6 segmentation swapped",     TRUE_P,            TRUE_D,                            M_BODY),
]

P_IN=min(256,int(2**np.ceil(np.log2(2*np.pi*(D//2)*0.85))))
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=P_IN,uniform=True); S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=DT,device=dev); rr=torch.tensor(pg.radius_shells,dtype=DT,device=dev)
w=torch.tensor(pg.integration_weight_points,dtype=DT,device=dev).reshape(S,P)
grid=torch.tensor(np.linspace(-SWING*1.3,SWING*1.3,NTHETA),dtype=DT,device=dev)
zero=torch.zeros(1,dtype=DT,device=dev)
shells=ShellMasks(D,dev)
conf=torch.arange(NP,device=dev)%NCONF
p=generate_physical_hetero(vols,ts,conf,pg,SNR,seed=17,device=dev,ctf_kwargs=ISO)
beta,tau,C=temperature(p,w,D)
pts_all=slice_points(fourier_circles(th,p.polars,p.azimus,p.gammas),rr)[:,None]
FIT=torch.arange(0,NP//2,device=dev); OUT=torch.arange(NP//2,NP,device=dev)
V0_init=to_real(shells.lowpass(fourier_volume(cons),R_INIT,width=1.0))

def em(ap,ad,mask,iters):
    core=(V0_init*(1-mask)).contiguous(); dom=(V0_init*mask).contiguous()
    t0=time.monotonic()
    for it in range(iters):
        acc_c=HeteroAccumulator.new(D,1,dev); acc_d=HeteroAccumulator.new(D,1,dev)
        for i in range(0,len(FIT),EB):
            j=min(i+EB,len(FIT)); g_=FIT[i:j]
            lw,gi,gm=hinge_logweights(p.images_fourier[g_],p.ctf[g_],w,beta[g_],core,dom,
                                      pts_all[g_],grid,ap,ad,D)
            wt=torch.softmax(lw.double(),dim=2).to(DT)
            hinge_mstep(acc_c,acc_d,p.images_fourier[g_],p.ctf[g_],w,pts_all[g_],
                        wt,gi,gm,grid,ap,ad,core,dom,tau[g_],D)
            t0=duty_sleep(t0)
        core=to_real(acc_c.solve(1e-3)[0]); dom=to_real(acc_d.solve(1e-3)[0])
        del acc_c,acc_d; torch.cuda.empty_cache()
    return core,dom

def heldout(core,dom,ap,ad,tgrid):
    out=[]
    t0=time.monotonic()
    for i in range(0,len(OUT),EB):
        j=min(i+EB,len(OUT)); g_=OUT[i:j]
        lw,_,_=hinge_logweights(p.images_fourier[g_],p.ctf[g_],w,beta[g_],core,dom,
                                pts_all[g_],tgrid,ap,ad,D)
        out.append(torch.logsumexp(lw[:,0].double(),dim=1)-math.log(len(tgrid)))
        t0=duty_sleep(t0)
    return torch.cat(out)

print(f"D={D}  SNR {SNR}  {NP} particles ({len(FIT)} fit / {len(OUT)} held out)  "
      f"{NTHETA}-point hinge grid  {ITERS} EM iterations  GPU duty {GPU_DUTY:g}")
print("Held-out marginal log-likelihood per particle, nats, relative to the true hypothesis.\n")
ev={}
for name,ap,ad,mask in HYP:
    t0=time.time()
    c,d=em(ap,ad,mask,ITERS)
    ev[name]=heldout(c,d,ap,ad,grid)
    fsc=shells.fsc(fourier_volume(c+d),fourier_volume(cons))
    print(f"  {name:28s} done in {time.time()-t0:5.0f}s   core+dom vs consensus, mean FSC "
          f"{fsc.mean():.4f}",flush=True)
# the rigid null: no motion, one volume, fitted the same way
c0,d0=em(TRUE_P,TRUE_D,M_ARM*0,1)   # mask 0 -> everything in core, domain empty
ev["H0 rigid, no motion"]=heldout(c0,d0,TRUE_P,TRUE_D,zero)
ref=ev["H1 true axis"]
print()
order=[h[0] for h in HYP]+["H0 rigid, no motion"]
for name in order:
    d_=(ev[name]-ref); se=d_.std().item()/math.sqrt(len(OUT))
    print(f"  {name:28s} {d_.mean().item():+10.2f}  +- {se:5.2f}")
rivals=[(ev[n]-ref).mean().item() for n in order if n!="H1 true axis"]
ses=[(ev[n]-ref).std().item()/math.sqrt(len(OUT)) for n in order if n!="H1 true axis"]
best=max(rivals); sb=ses[int(np.argmax(rivals))]
print(f"\n  closest rival {best:+.2f} nats behind ({abs(best)/max(sb,1e-9):.0f} standard "
      f"errors): {_v(best < -3*sb)}")
print(("ALL PASS" if all(_FAILS) else "SOME FAIL"), f"({sum(_FAILS)}/{len(_FAILS)})")
sys.exit(0 if all(_FAILS) else 1)
