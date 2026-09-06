"""STAGE 2: discrete classes -- joint posterior over (class, pose), gold-standard halves.

Two conformations of the hinge phantom (t = -T_CLASS, +T_CLASS). Each half-set
keeps two class references; every particle is scored against both banks and
gets a joint posterior over (class, view) with the measured temperature beta.
Mixture insertion puts each particle into each class map with its posterior
weight; argmax puts it into one class at one pose. Halves are fully
independent (GOLD): each is posed against its own previous references.

Classes are unlabelled when INIT=consensus (both start from the same
low-passed consensus plus a small seeded perturbation, the standard symmetry
break). Labels are resolved per half by the permutation that maximises class
accuracy against the truth, then applied to every metric consistently.
INIT=random keeps the consensus start but assigns each particle a random class at
iteration 1 (RELION-style symmetry break). INIT=truth starts each class from its
own low-passed truth (upper bound on what the classification can hold).

Reported per iteration, argmax and mixture side by side:
  acc      class accuracy (mean of the two halves)
  vsTruth  FSC0.5 vs the class's true volume, mean over classes and halves (px)
  halfmap  FSC0.5 between the two halves' maps of the same class, mean (px)
Knobs: SNR, T_CLASS (deg), INIT, ITERS, NP, MODES, D, SMOKE=1 (tiny sizes),
DEV=cpu BACKEND=gridding (CPU smoke run through the trilinear backend).
"""
import os, math, numpy as np, torch
import _paths  # noqa: F401
from cryolike.grids import PolarGrid
from cryolike.metadata import ViewingAngles
from dcryolike.generator import hinge_phantom, generate_physical_hetero
from dcryolike.slicing import fourier_volume
from dcryolike.pipeline import (ShellMasks, res_at, to_real, temperature, Bank, neighbourhood,
                                select_modes, reconstruct)

dev=os.environ.get("DEV","cuda"); BACKEND=os.environ.get("BACKEND","nufft"); SMOKE=os.environ.get("SMOKE","0")=="1"
D=int(os.environ.get("D","32" if SMOKE else "48")); NP=int(os.environ.get("NP","400" if SMOKE else "8000"))
MODES=int(os.environ.get("MODES","10")); NEIGH=10.0; ITERS=int(os.environ.get("ITERS","2" if SMOKE else "6")); R_INIT=8
SNR=float(os.environ.get("SNR","0.1")); T_CLASS=math.radians(float(os.environ.get("T_CLASS","15")))
INIT=os.environ.get("INIT","consensus"); ISO=dict(astig=0.0); NCLS=2
torch.manual_seed(0)

vols=[hinge_phantom(D,-T_CLASS,device=dev)[0],hinge_phantom(D,+T_CLASS,device=dev)[0]]
Ftrue=[fourier_volume(v) for v in vols]
P=int(os.environ.get("P_INPLANE",str(64 if SMOKE else min(256,int(2**np.ceil(np.log2(2*np.pi*(D//2)*0.85)))))))
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=P,uniform=True)
S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=torch.float64,device=dev)
rr=torch.tensor(pg.radius_shells,dtype=torch.float64,device=dev)
w=torch.tensor(pg.integration_weight_points,dtype=torch.float64,device=dev).reshape(S,P); wr=w[:,0]
VIEW_DEG=float(os.environ.get("VIEW_DEG", "8.0" if SMOKE else ("4.0" if D<=48 else str(np.degrees(1.0/(D//2))))))
NEIGH=NEIGH*min(1.0,24.0/(D//2))
va=ViewingAngles.from_viewing_distance(np.radians(VIEW_DEG))
caz=va.azimus.to(dev).double(); cpo=va.polars.to(dev).double(); V=caz.shape[0]
NB,_=neighbourhood(caz,cpo,NEIGH)
SINGLE=D>48
conf=torch.arange(NP,device=dev)%NCLS
p=generate_physical_hetero(vols,[-T_CLASS,T_CLASS],conf,pg,SNR,seed=17,device=dev,ctf_kwargs=ISO,backend=BACKEND)
beta,tau,C=temperature(p,w,D); In=torch.sqrt((p.images_fourier.abs()**2*w).sum(dim=(1,2))); cr=p.ctf[:,:,0]
p.clean_fourier=None; torch.cuda.empty_cache()
HALF=[torch.arange(0,NP//2,device=dev),torch.arange(NP//2,NP,device=dev)]
shells=ShellMasks(D,dev)

def lowpassed(F,k): return to_real(shells.lowpass(F,k,width=1.0))
cons=fourier_volume(torch.stack(vols).mean(0))
if INIT=="truth":
    init=[lowpassed(Ftrue[k],R_INIT) for k in range(NCLS)]
else:
    g=torch.Generator(device=dev).manual_seed(5)
    pert=torch.randn(D,D,D,generator=g,dtype=torch.float64,device=dev)
    pert=lowpassed(fourier_volume(pert),R_INIT); pert=pert*(0.05*cons.abs().mean()/pert.abs().mean())
    base=lowpassed(cons,R_INIT); init=[base+pert,base-pert]
print(f"DISCRETE CLASSES | physical SNR {SNR} | classes t=+-{math.degrees(T_CLASS):.0f} deg | {NP} particles, "
      f"GOLD halves of {NP//2} | init={INIT} | C={C.item():.4f} beta median {beta.median():.1f}")
print(f"{'iter':>5}  {'---------- argmax ----------':^30}  {'---------- mixture ---------':^30}")
print(f"{'':5}  {'acc':>7}{'vsTruth':>9}{'halfmap':>9}{'meanFSC':>9}  {'acc':>7}{'vsTruth':>9}{'halfmap':>9}{'meanFSC':>9}")
print("-"*75)
ref={m:[[init[0].clone(),init[1].clone()] for _ in range(2)] for m in ("argmax","mixture")}
for it in range(1,ITERS+1):
    line=f"{it:>5}  "
    for mode in ("argmax","mixture"):
        newF=[[None]*NCLS for _ in range(2)]; accs=[]; perms=[]
        for h in (0,1):
            ids=HALF[h]; n=len(ids)
            bcs=[];bgs=[]
            for k in range(NCLS):
                bank=Bank(ref[mode][h][k],th,cpo,caz,rr,S,P,backend=BACKEND,single=SINGLE)
                bc,bg=bank.search(p.images_fourier[ids],cr[ids],wr,In[ids]); bcs.append(bc); bgs.append(bg); del bank
            # joint (class, view) modes: local maxima per class, softmax across both classes
            sel=[select_modes(bcs[k],NB,MODES,beta[ids]) for k in range(NCLS)]
            tcc=torch.cat([s[2] for s in sel],1)                        # (n, 2*MODES)
            best=tcc.max(1,keepdim=True).values
            lg=(beta[ids][:,None]*(tcc**2-best**2)).masked_fill(tcc<-1.0,-1e30)
            wt=torch.softmax(lg,dim=1)
            if INIT=="random" and it==1:
                # symmetry break: a RANDOM class per particle at iteration 1 (the
                # two references are identical then, so the posterior cannot split)
                gr=torch.Generator(device=dev).manual_seed(11+h)
                rc=torch.randint(0,NCLS,(n,),generator=gr,device=dev)
                keep=torch.zeros_like(wt); 
                for k in range(NCLS): keep[:,k*MODES:(k+1)*MODES]=(rc==k).double()[:,None]
                wt=wt*keep; wt=wt/wt.sum(1,keepdim=True).clamp(min=1e-30)
            if mode=="argmax":
                wt=torch.zeros_like(wt).scatter_(1,lg.argmax(1,keepdim=True),1.0)
            cls_post=torch.stack([wt[:,k*MODES:(k+1)*MODES].sum(1) for k in range(NCLS)],1)
            # resolve the label permutation against truth (classes are unlabelled)
            pred=cls_post.argmax(1); truth=conf[ids]
            acc_id=(pred==truth).double().mean().item(); acc_sw=(pred!=truth).double().mean().item()
            perm=[0,1] if acc_id>=acc_sw else [1,0]; perms.append(perm); accs.append(max(acc_id,acc_sw))
            for k in range(NCLS):
                F=reconstruct(p,ids,sel[k][0],wt[:,k*MODES:(k+1)*MODES],bgs[k],th,cpo,caz,rr,w,D)[0]
                newF[h][perm[k]]=F                                       # store under the TRUE label
            ref[mode][h]=[to_real(newF[h][perm[k]]) for k in range(NCLS)]   # the half keeps its OWN labelling
        vt=[];hm=[];mf=[]
        for k in range(NCLS):
            fa=shells.fsc(newF[0][k],Ftrue[k]); fb=shells.fsc(newF[1][k],Ftrue[k]); fh=shells.fsc(newF[0][k],newF[1][k])
            vt+=[res_at(fa),res_at(fb)]; hm.append(res_at(fh)); mf+=[fa.mean(),fb.mean()]
        line+=f"{np.mean(accs)*100:6.1f}%{np.mean(vt):8.1f}p{np.mean(hm):8.1f}p{np.mean(mf):9.4f}  "
    print(line)
