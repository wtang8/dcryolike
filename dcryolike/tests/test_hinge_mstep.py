"""Does the two-volume M-step reconstruct the truth when given the truth?

The held-out hypothesis comparison put a RIGID model 47 nats per particle ahead of the
true hinge, on data that genuinely move, and the hypotheses that ended up moving least
fitted best. That is the signature of a reconstruction that degrades when motion is
modelled, not of a rigid molecule. So before reading anything into the comparison, the
M-step is given every true quantity and asked to return the truth.

  TEST 1  true poses, true hinge angle per particle (delta weight), true gain, noiseless
          images. The recovered core and domain must match the true ones. Anything less
          is a bug in the M-step, not a property of the data.
  TEST 2  the same with the domain held at truth and only the core solved, and vice
          versa, to localise a failure to one of the two blocks.
  TEST 3  the block iteration from a deliberately wrong split. Without the mask this
          diverges, because the split is NON-IDENTIFIABLE where the two volumes
          overlap: at zero hinge angle, moving density from one to the other leaves
          the model unchanged, so Gauss-Seidel trades density between them without
          bound. The mask is the hypothesis's own claim about which part moves, so it
          is enforced after each solve, as RELION multi-body enforces body masks.

Thresholds are set against what the geometry can deliver, not against 1.0: with a few
hundred particles the angular coverage is sparse, and the thin helical domain is harder
to reconstruct than the compact core, so neither reaches FSC 1 even noiselessly.
"""
import os, math, numpy as np, torch, sys
import _paths  # noqa: F401
from cryolike.grids import PolarGrid
from dcryolike.generator import _phantom_atoms, _render_atoms, generate_physical_hetero
from dcryolike.slicing import fourier_circles, slice_points, fourier_volume
from dcryolike.hinge import hinge_mstep, render_hinged
from dcryolike.hetero import HeteroAccumulator
from dcryolike.pipeline import temperature, ShellMasks, to_real

_FAILS=[]
def _v(c): _FAILS.append(bool(c)); return 'PASS' if c else 'FAIL'
dev="cuda"; DT=torch.float32
D=int(os.environ.get("D","64")); NP=int(os.environ.get("NP","1200"))
NCONF=8; SWING=math.radians(15.0); ISO=dict(astig=0.0); EB=8

body,helix=_phantom_atoms(D,dtype=DT)
core_t=_render_atoms(body,D,device=dev,dtype=DT)
dom_t =_render_atoms(helix,D,device=dev,dtype=DT)
AP=helix[0].clone(); AD=torch.tensor([1.0,0.0,0.0],dtype=DT)
ts=list(np.linspace(-SWING,SWING,NCONF))
vols=[render_hinged(core_t,helix,AP,AD,float(t),D,device=dev,dtype=DT) for t in ts]

P_IN=min(256,int(2**np.ceil(np.log2(2*np.pi*(D//2)*0.85))))
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=P_IN,uniform=True); S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=DT,device=dev); rr=torch.tensor(pg.radius_shells,dtype=DT,device=dev)
w=torch.tensor(pg.integration_weight_points,dtype=DT,device=dev).reshape(S,P)
shells=ShellMasks(D,dev)
conf=torch.arange(NP,device=dev)%NCONF
grid=torch.tensor(ts,dtype=DT,device=dev)          # the grid IS the true angle set
p=generate_physical_hetero(vols,ts,conf,pg,1e6,seed=17,device=dev,ctf_kwargs=ISO)  # noiseless
beta,tau,C=temperature(p,w,D)
pts=slice_points(fourier_circles(th,p.polars,p.azimus,p.gammas),rr)[:,None]
tau1=torch.ones_like(tau)

def scale_to(F,Fref,kmax):
    m=shells.RR<=kmax
    return ((F[m]*Fref[m].conj()).real.sum()/(Fref[m].abs()**2).sum().clamp(min=1e-30)).item()

CORR=[None,None]
def run(core_in,dom_in,solve_core=True,solve_dom=True,correct=True):
    """One M-step with delta weights at each particle's TRUE angle and unit gain."""
    ac=HeteroAccumulator.new(D,1,dev); ad=HeteroAccumulator.new(D,1,dev)
    T=len(grid)
    for i in range(0,NP,EB):
        j=min(i+EB,NP); g_=torch.arange(i,j,device=dev)
        wt=torch.zeros(j-i,1,T,dtype=DT,device=dev)
        wt[torch.arange(j-i),0,conf[g_]]=1.0                    # delta at the truth
        gi=torch.zeros(j-i,1,T,dtype=torch.long,device=dev)      # gamma already in the pose
        gm=torch.ones(j-i,1,T,dtype=DT,device=dev)               # unit gain
        hinge_mstep(ac,ad,p.images_fourier[g_],p.ctf[g_],w,pts[g_],wt,gi,gm,grid,
                    AP,AD,core_in,dom_in,tau1[g_],D)
    Fc=ac.solve(1e-3)[0] if solve_core else fourier_volume(core_in)
    Fd=ad.solve(1e-3)[0] if solve_dom else fourier_volume(dom_in)
    if correct:
        if CORR[0] is None:
            from dcryolike.hinge import hinge_gridding_correction
            wt=torch.zeros(NP,1,len(grid),dtype=DT,device=dev)
            wt[torch.arange(NP,device=dev),0,conf]=1.0
            gi=torch.zeros(NP,1,len(grid),dtype=torch.long,device=dev)
            gm=torch.ones(NP,1,len(grid),dtype=DT,device=dev)
            CORR[0],CORR[1]=hinge_gridding_correction(core_in,dom_in,pts,wt,gi,gm,grid,
                                                      AP,AD,p.ctf,w,tau1,D,EB=EB)
            print(f"    [gridding] core 1/a at k=4,12: "
                  f"{1/CORR[0][D//2+4,D//2,D//2].item():.3f} {1/CORR[0][D//2+12,D//2,D//2].item():.3f}"
                  f"   domain: {1/CORR[1][D//2+4,D//2,D//2].item():.3f} "
                  f"{1/CORR[1][D//2+12,D//2,D//2].item():.3f}",flush=True)
        if solve_core: Fc=Fc*CORR[0]
        if solve_dom:  Fd=Fd*CORR[1]
    return Fc,Fd

Fc_t=fourier_volume(core_t); Fd_t=fourier_volume(dom_t)
K=D//2-4
print(f"D={D}  {NP} noiseless particles  true poses, true angles, unit gain")
print(f"reported: scale of the recovered volume against the true one, and their FSC\n")

print("="*70); print("TEST 2: one block at a time, the other held at truth"); print("="*70)
Fc,_=run(core_t,dom_t,solve_core=True,solve_dom=False)
sc=scale_to(Fc,Fc_t,K); fc=shells.fsc(Fc,Fc_t).mean()
print(f"  core solved, domain at truth : scale {sc:.3f}  meanFSC {fc:.4f}   {_v(abs(sc-1)<0.10 and fc>0.90)}")
_,Fd=run(core_t,dom_t,solve_core=False,solve_dom=True)
sd=scale_to(Fd,Fd_t,K); fd=shells.fsc(Fd,Fd_t).mean()
print(f"  domain solved, core at truth : scale {sd:.3f}  meanFSC {fd:.4f}   {_v(abs(sd-1)<0.10 and fd>0.85)}")

print("\n"+"="*70); print("TEST 1: both blocks solved together from the truth"); print("="*70)
Fc,Fd=run(core_t,dom_t)
sc=scale_to(Fc,Fc_t,K); sd=scale_to(Fd,Fd_t,K)
fc=shells.fsc(Fc,Fc_t).mean(); fd=shells.fsc(Fd,Fd_t).mean()
print(f"  core   scale {sc:.3f}  meanFSC {fc:.4f}")
print(f"  domain scale {sd:.3f}  meanFSC {fd:.4f}")
print(f"  a fixed point at the truth: "
      f"{_v(abs(sc-1)<0.10 and abs(sd-1)<0.10 and fc>0.90 and fd>0.85)}")

print("\n"+"="*70); print("TEST 3: does the block iteration converge from a wrong split?"); print("="*70)
from dcryolike.generator import _render_atoms as _ra
_a=_ra(helix,D,device=dev,dtype=DT); _b=_ra(body,D,device=dev,dtype=DT)
MASK=(_a/(_a+_b+1e-6)).clamp(0,1)                 # the hypothesis: this part moves
c=(core_t+dom_t)*0.5; d=(core_t+dom_t)*0.5
for it in range(6):
    Fc,Fd=run(c,d)
    c=to_real(Fc)*(1-MASK); d=to_real(Fd)*MASK    # enforce the hypothesis's own split
    Fc=fourier_volume(c); Fd=fourier_volume(d)
    print(f"    iter {it+1}: core FSC {shells.fsc(Fc,Fc_t).mean():.4f}  "
          f"domain FSC {shells.fsc(Fd,Fd_t).mean():.4f}  "
          f"core scale {scale_to(Fc,Fc_t,K):+.3f}  dom scale {scale_to(Fd,Fd_t,K):+.3f}")
print(f"  converges towards the truth rather than diverging: "
      f"{_v(shells.fsc(Fc,Fc_t).mean()>0.80 and shells.fsc(Fd,Fd_t).mean()>0.70)}")

print("\n"+("ALL PASS" if all(_FAILS) else "SOME FAIL"), f"({sum(_FAILS)}/{len(_FAILS)})")
sys.exit(0 if all(_FAILS) else 1)
