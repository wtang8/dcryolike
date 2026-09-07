"""Is there anything to average over? Effective number of shortlist modes vs SNR.

THE QUESTION THIS DECIDES. The distinctive claim of this pipeline is that the
conformational latent should be averaged over the pose shortlist rather than
estimated at the single best pose. Every competing method (3DVA, RECOVAR,
cryoDRGN, 3DFlex) fixes the pose from a consensus refinement first. But the
claim can only be true if the shortlist posterior actually spreads: if one mode
carries essentially all the weight, then averaging over modes is arithmetically
identical to committing to the best one, and there is nothing to claim.

That was observed at SNR 0.3, where MODES=1 and MODES=10 gave trajectories
agreeing to three decimals. This measures the spread directly, across SNR,
without fitting anything.

WHAT IS MEASURED, per particle:
  n_eff  = exp(H(omega)), H the Shannon entropy of the mode weights.
           1.0 = all weight on one mode (averaging is a no-op).
           >2  = the posterior genuinely spans several poses.
  w1     = the weight of the top mode.
  hit@1  = is the top mode within tolerance of the true pose?
  hit@M  = is ANY shortlist mode within tolerance? (the recall ceiling)
  w_true = total weight on modes that ARE within tolerance. If n_eff is large
           but w_true is small, the spread is over WRONG poses and averaging
           would import noise rather than reduce bias.

Cost: one pose search per SNR, no EM. Minutes, not hours.
"""
import os, math, numpy as np, torch, sys
import _paths  # noqa: F401
from cryolike.grids import PolarGrid
from cryolike.metadata import ViewingAngles
from dcryolike.generator import hinge_phantom, generate_physical_hetero
from dcryolike.pipeline import (temperature, Bank, neighbourhood, select_modes, GPU_DUTY)

dev="cuda"; D=int(os.environ.get("D","96")); NP=int(os.environ.get("NP","1000"))
NCONF=16; SWING=math.radians(15.0); MODES=int(os.environ.get("MODES","10"))
SNRS=[float(x) for x in os.environ.get("SNRS","0.3,0.1,0.05,0.02").split(",")]
DT = torch.float64 if os.environ.get("DTYPE","single")=="double" else torch.float32
ISO=dict(astig=0.0); NEIGH=10.0*min(1.0,24.0/(D//2))
kmax=D//2; tol=np.degrees(1.25/kmax)

ts=list(np.linspace(-SWING,SWING,NCONF))
vols=[hinge_phantom(D,float(t),device=dev,dtype=DT)[0] for t in ts]
cons=torch.stack(vols).mean(0).contiguous()
P=min(256,int(2**np.ceil(np.log2(2*np.pi*kmax*0.85))))
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=P,uniform=True); S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=DT,device=dev); rr=torch.tensor(pg.radius_shells,dtype=DT,device=dev)
w=torch.tensor(pg.integration_weight_points,dtype=DT,device=dev).reshape(S,P); wr=w[:,0]
VIEW_DEG=np.degrees(1.0/kmax)
va=ViewingAngles.from_viewing_distance(np.radians(VIEW_DEG))
caz=va.azimus.to(dev).to(DT); cpo=va.polars.to(dev).to(DT); V=caz.shape[0]
NB,nd=neighbourhood(caz,cpo,NEIGH)
conf=torch.arange(NP,device=dev)%NCONF
bank=Bank(cons,th,cpo,caz,rr,S,P,single=True)
print(f"D={D} grid {S}x{P} {V} views @ {VIEW_DEG:.2f} deg, tol {tol:.2f} deg, {NP} particles/SNR, "
      f"shortlist {MODES}, consensus reference")
print(f"{'SNR':>6}{'beta':>9}{'n_eff':>8}{'w1':>7}{'hit@1':>8}{'hit@M':>8}{'w_true':>8}   reading")
print("-"*74)
for snr in SNRS:
    p=generate_physical_hetero(vols,ts,conf,pg,snr,seed=17,device=dev,ctf_kwargs=ISO)
    beta,tau,C=temperature(p,w,D)
    In=torch.sqrt((p.images_fourier.abs()**2*w).sum(dim=(1,2))); cr=p.ctf[:,:,0]
    ids=torch.arange(NP,device=dev)
    bc,bg=bank.search(p.images_fourier,cr,wr,In)
    tidx,wt,_=select_modes(bc,NB,MODES,beta)
    wt=wt.to(torch.float64)
    H=-(wt.clamp(min=1e-30)*wt.clamp(min=1e-30).log()).sum(1)
    n_eff=H.exp()
    # which shortlist entries are actually correct
    nt=torch.stack([torch.sin(p.polars)*torch.cos(p.azimus),
                    torch.sin(p.polars)*torch.sin(p.azimus),torch.cos(p.polars)],-1).to(nd.dtype)
    ang=torch.rad2deg(torch.arccos((nt@nd.T).clamp(-1,1)))          # (NP, V)
    ang_m=torch.gather(ang,1,tidx)                                   # (NP, MODES)
    good=ang_m<=tol
    hit1=good[:,0].double().mean().item()*100
    hitM=good.any(1).double().mean().item()*100
    w_true=(wt*good.double()).sum(1).mean().item()
    ne=n_eff.median().item()
    reading=("no spread: averaging == argmax" if ne<1.15 else
             "spread, but over WRONG poses" if w_true<0.5 else
             "spread over poses that include the truth")
    print(f"{snr:>6.2f}{beta.median().item():>9.0f}{ne:>8.2f}{wt[:,0].median().item():>7.2f}"
          f"{hit1:>7.1f}%{hitM:>7.1f}%{w_true:>8.2f}   {reading}")
    del p,bc,bg,tidx,wt,ang; torch.cuda.empty_cache()
print("\nn_eff is exp(entropy) of the mode weights: 1.0 means one mode holds all the weight.")
print("w_true is the weight sitting on modes within tolerance of the true pose.")
print("Averaging the latent over the shortlist can only beat committing to one pose")
print("where n_eff > 1 AND w_true is appreciable.")
