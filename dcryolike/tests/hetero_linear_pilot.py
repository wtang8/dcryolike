"""STAGE 3: linear latent model, exact z-marginalisation, gold-standard halves.

V(z) = V0 + sum_k z_k U_k fitted to a continuous hinge motion (NCONF
conformations, t uniform in +-SWING). Per iteration, per half (independent):

  1. shortlist   : MODES local maxima of the V0-only normalised CC (the recall
                   machinery), best in-plane angle per view
  2. E-step      : slices of [V0, U_1..U_K] at each shortlisted view; Gram
                   matrices over every in-plane rotation; the coefficients
                   a = (gain, z) integrated out in closed form (hetero.py); view
                   weights = softmax of the per-view log marginal (lse over gamma),
                   gamma = argmax; posterior moments of a at that gamma
  3. M-step      : per-voxel (K+1)x(K+1) solve with trilinear splatting
  4. gauge       : PCA gauge on the posterior-mean latents

K=0 runs the same code path with gain marginalised and no components -- the
homogeneous baseline for the FSC columns.

Reported per iteration:
  rho     Spearman rank correlation of z_1 with the true t (mean of halves)
  R2      R^2 of t from z (all K components)
  leak    fraction of U_1's energy that is a rigid rotation/translation of V0
  V0      FSC0.5 of V0 vs the truth at each conformation, mean over bins (px)
  V(z)    FSC0.5 of V(z_bar_c) vs the truth at conformation c, mean over bins
  half    FSC0.5 between the halves' V(z_bar_c), mean over bins (gold standard;
          each half uses its own z_bar, so no latent alignment is needed)
Knobs: SNR, K, SWING (deg), NCONF, ITERS, NP, MODES, D, U_INIT (truth|random), U_KMAX, SMOKE=1,
DEV=cpu BACKEND=gridding (CPU smoke run through the trilinear backend).
"""
import os, math, numpy as np, torch
import _paths  # noqa: F401
from cryolike.grids import PolarGrid
from cryolike.metadata import ViewingAngles
from dcryolike.generator import hinge_phantom, generate_physical_hetero
from dcryolike.slicing import fourier_volume, fourier_circles, slice_points
from dcryolike.generator import _slice
from dcryolike.hetero import hetero_gram, marginal_loglik, fix_gauge, rotation_leakage, rank_correlation, latent_r2, align_bases
from dcryolike.pipeline import (ShellMasks, res_at, to_real, temperature, Bank, neighbourhood,
                                select_modes, reconstruct)

dev=os.environ.get("DEV","cuda"); BACKEND=os.environ.get("BACKEND","nufft"); SMOKE=os.environ.get("SMOKE","0")=="1"
D=int(os.environ.get("D","32" if SMOKE else "48")); NP=int(os.environ.get("NP","400" if SMOKE else "8000"))
MODES=int(os.environ.get("MODES","10")); NEIGH=10.0; ITERS=int(os.environ.get("ITERS","2" if SMOKE else "6")); R_INIT=8
SNR=float(os.environ.get("SNR","0.1")); K=int(os.environ.get("K","1"))
SWING=math.radians(float(os.environ.get("SWING","15"))); NCONF=int(os.environ.get("NCONF","16"))
U_INIT=os.environ.get("U_INIT","random"); ISO=dict(astig=0.0); EB=32
U_KMAX=float(os.environ.get("U_KMAX","0"))   # >0: band-limit the components to |k|<=U_KMAX after each M-step
torch.manual_seed(0)

ts=list(np.linspace(-SWING,SWING,NCONF)); vols=[hinge_phantom(D,float(t),device=dev)[0] for t in ts]
Ftrue=[fourier_volume(v) for v in vols]
P=int(os.environ.get("P_INPLANE",str(64 if SMOKE else min(256,int(2**np.ceil(np.log2(2*np.pi*(D//2)*0.85)))))))
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=P,uniform=True)
S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=torch.float64,device=dev)
rr=torch.tensor(pg.radius_shells,dtype=torch.float64,device=dev)
w=torch.tensor(pg.integration_weight_points,dtype=torch.float64,device=dev).reshape(S,P); wr=w[:,0]
VIEW_DEG=float(os.environ.get("VIEW_DEG", "8.0" if SMOKE else ("4.0" if D<=48 else str(np.degrees(1.0/(D//2))))))
NEIGH=NEIGH*min(1.0,24.0/(D//2))                     # local-max radius scales with 1/k_max
va=ViewingAngles.from_viewing_distance(np.radians(VIEW_DEG))
caz=va.azimus.to(dev).double(); cpo=va.polars.to(dev).double(); V=caz.shape[0]
NB,_=neighbourhood(caz,cpo,NEIGH)
SINGLE=D>48                                           # complex64 bank at D=96 (3 GB, not 6)
conf=torch.arange(NP,device=dev)%NCONF
p=generate_physical_hetero(vols,ts,conf,pg,SNR,seed=17,device=dev,ctf_kwargs=ISO,backend=BACKEND)
beta,tau,C=temperature(p,w,D); In=torch.sqrt((p.images_fourier.abs()**2*w).sum(dim=(1,2))); cr=p.ctf[:,:,0]
p.clean_fourier=None; torch.cuda.empty_cache()        # not needed past the temperature; 1.6 GB at D=96
HALF=[torch.arange(0,NP//2,device=dev),torch.arange(NP//2,NP,device=dev)]
shells=ShellMasks(D,dev); prior_prec=torch.tensor([0.0]+[1.0]*K,dtype=torch.float64,device=dev)

def lowpassed(F,k): return to_real(shells.lowpass(F,k,width=1.0))
cons=torch.stack(vols).mean(0); V0_init=lowpassed(fourier_volume(cons),R_INIT)
if K>0 and U_INIT=="truth":
    # first principal direction of the true family, low-passed (upper-bound init)
    X=torch.stack([v-cons for v in vols]).reshape(NCONF,-1)
    _,_,Vh=torch.linalg.svd(X,full_matrices=False)
    U_init=torch.stack([lowpassed(fourier_volume(Vh[k].reshape(D,D,D)*X.norm()/math.sqrt(NCONF)),R_INIT) for k in range(K)])
elif K>0:
    g=torch.Generator(device=dev).manual_seed(5)
    U_init=torch.stack([lowpassed(fourier_volume(torch.randn(D,D,D,generator=g,dtype=torch.float64,device=dev)),R_INIT) for _ in range(K)])
    U_init=U_init*(0.1*V0_init.norm()/U_init.reshape(K,-1).norm(dim=1)[:,None,None,None])
else:
    U_init=torch.zeros(0,D,D,D,dtype=torch.float64,device=dev)

def estep(V0,U,ids,tidx,bg):
    """Marginal likelihood on the shortlist over all gamma. Returns view weights,
    gamma matrix (n,V) updated at the shortlisted views, Ea (n,M,K+1), Eaa (n,M,K+1,K+1)."""
    n=len(ids); M=tidx.shape[1]; K1=K+1
    vol_list=[V0]+[U[k] for k in range(K)]
    lse=torch.empty(n,M,dtype=torch.float64,device=dev); Ea=torch.empty(n,M,K1,dtype=torch.float64,device=dev)
    Eaa=torch.empty(n,M,K1,K1,dtype=torch.float64,device=dev); bg=bg.clone()
    for i in range(0,n,EB):
        j=min(i+EB,n); nb=j-i; g_=ids[i:j]; v=tidx[i:j].reshape(-1)
        pts=slice_points(fourier_circles(th,cpo[v],caz[v],torch.zeros_like(caz[v])),rr)
        T=torch.stack([_slice(vk,pts,BACKEND) for vk in vol_list],1).reshape(nb,M,K1,S,P)
        A,b,Gyy=hetero_gram(p.images_fourier[g_],T,p.ctf[g_],w)
        logml,mean,cov=marginal_loglik(A,b,Gyy,tau[g_],prior_prec)   # (nb,M,P)
        lse[i:j]=torch.logsumexp(logml,dim=2)
        gi=logml.argmax(2)                                             # (nb,M)
        bg[torch.arange(i,j,device=dev)[:,None],tidx[i:j]]=-(gi.double())*(2*np.pi/P)
        mu=torch.gather(mean,2,gi[...,None,None].expand(nb,M,1,K1)).squeeze(2)
        cv=torch.gather(cov,2,gi[...,None,None,None].expand(nb,M,1,K1,K1)).squeeze(2)
        Ea[i:j]=mu; Eaa[i:j]=cv+mu[...,:,None]*mu[...,None,:]
        del T,A,b,logml,mean,cov
    wt=torch.softmax(lse-lse.max(1,keepdim=True).values,dim=1)
    return wt,bg,Ea,Eaa

def family_metrics(V0,U,zbar):
    """FSC0.5 of V0 and of V(zbar_c) against the truth at each conformation."""
    r0=[];rz=[];Fz=[]
    F0=fourier_volume(V0)
    for c in range(NCONF):
        Vc=V0+(torch.einsum('k,kxyz->xyz',zbar[c],U) if K>0 else 0)
        Fc=fourier_volume(Vc); Fz.append(Fc)
        r0.append(res_at(shells.fsc(F0,Ftrue[c]))); rz.append(res_at(shells.fsc(Fc,Ftrue[c])))
    return np.mean(r0),np.mean(rz),Fz

print(f"LINEAR LATENT | D={D} grid {S}x{P} {V} views @ {VIEW_DEG:.2f} deg | physical SNR {SNR} | K={K} | U_KMAX={U_KMAX:g} | swing +-{math.degrees(SWING):.0f} deg, {NCONF} conformations | "
      f"{NP} particles, GOLD halves | U_init={U_INIT} | C={C.item():.4f} beta median {beta.median():.1f}")
print(f"{'iter':>5}{'rho':>7}{'R2':>7}{'leak':>7}  {'V0':>7}{'V(z)':>7}{'half':>7}   (px at FSC 0.5, mean over conformations)")
print("-"*70)
state=[(V0_init.clone(),U_init.clone()) for _ in range(2)]
for it in range(1,ITERS+1):
    rho=[];r2=[];leak=[];Fz_h=[];m0=[];mz=[]
    for h in (0,1):
        ids=HALF[h]; V0,U=state[h]
        bank=Bank(V0,th,cpo,caz,rr,S,P,backend=BACKEND,single=SINGLE); bc,bg=bank.search(p.images_fourier[ids],cr[ids],wr,In[ids]); del bank; torch.cuda.empty_cache()
        tidx,_,_=select_modes(bc,NB,MODES,beta[ids])
        wt,bg,Ea,Eaa=estep(V0,U,ids,tidx,bg)
        F=reconstruct(p,ids,tidx,wt,bg,th,cpo,caz,rr,w,D,K=K,Ea=Ea,Eaa=Eaa,tau=tau)
        V0=to_real(F[0])
        if K>0:
            # a band-limited component has far fewer noise degrees of freedom
            # (p shrinks ~k^3) while the motion energy is mostly low-k: the
            # 3DVA "filter resolution" trick, here as a hard cosine edge
            U=torch.stack([to_real(shells.lowpass(F[k+1],U_KMAX,width=1.0)) if U_KMAX>0 else to_real(F[k+1]) for k in range(K)])
        if K>0:
            z_mean=(wt[...,None]*Ea[...,1:]).sum(1)                       # posterior mean z per particle
            z_cov=(wt[...,None,None]*(Eaa[...,1:,1:]-Ea[...,1:,None]*Ea[...,None,1:])).sum(1)
            # GAUGE: centre only inside the loop. Re-whitening z every iteration
            # rescales U by the posterior std of z, which is < 1 while U is still
            # uninformative -- U then shrinks each round and the component dies
            # (observed: rho ~0.02 flat, random init). PPCA EM lets the M-step
            # set U's scale; the PCA (whitened) gauge is used for REPORTING only.
            mu=z_mean.mean(0); V0=V0+torch.einsum('k,kxyz->xyz',mu,U); z_mean=z_mean-mu
            V0r,Ur,z_rep,_=fix_gauge(V0,U,z_mean,z_cov)
            t_ids=p.t[ids]
            rho.append(abs(rank_correlation(z_rep[:,0],t_ids))); r2.append(latent_r2(z_rep,t_ids))
            leak.append(rotation_leakage(V0r,Ur)[0].item())
            zbar=torch.stack([z_mean[conf[ids]==c].mean(0) for c in range(NCONF)])
            unorm=U.reshape(K,-1).norm(dim=1)/V0.norm()
        else:
            zbar=torch.zeros(NCONF,0,dtype=torch.float64,device=dev)
        a0,az,Fz=family_metrics(V0,U,zbar); m0.append(a0); mz.append(az); Fz_h.append(Fz)
        state[h]=(V0,U)
    half=np.mean([res_at(shells.fsc(Fz_h[0][c],Fz_h[1][c])) for c in range(NCONF)])
    print(f"{it:>5}{np.mean(rho) if rho else float('nan'):7.3f}{np.mean(r2) if r2 else float('nan'):7.3f}"
          f"{np.mean(leak) if leak else float('nan'):7.3f}  {np.mean(m0):6.1f}p{np.mean(mz):6.1f}p{half:6.1f}p"
          +(f"   |U|/|V0| {unorm[0].item():.3f}  z std {z_mean[:,0].std().item():.3f}" if K>0 else ""))
if K>1:
    Q,_=align_bases(state[0][1],state[1][1])
    print(f"basis agreement between halves (Procrustes): {torch.diagonal(Q).abs().mean().item():.3f} "
          f"(1 = the halves found the same components in the same order)")
