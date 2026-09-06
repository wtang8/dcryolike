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
Knobs: SNR, K, SWING (deg), NCONF, ITERS, NP, MODES, D, U_INIT (truth|random), U_KMAX, U_SHRINK,
TAU_MODE (residual|known), POSE_FROM (v0|marginal), SMOKE=1,
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
                                select_modes, reconstruct, duty_sleep, GPU_DUTY, gridding_correction)
import time

dev=os.environ.get("DEV","cuda"); BACKEND=os.environ.get("BACKEND","nufft"); SMOKE=os.environ.get("SMOKE","0")=="1"
D=int(os.environ.get("D","32" if SMOKE else "48")); NP=int(os.environ.get("NP","400" if SMOKE else "8000"))
MODES=int(os.environ.get("MODES","10")); NEIGH=10.0; ITERS=int(os.environ.get("ITERS","2" if SMOKE else "6")); R_INIT=8
SNR=float(os.environ.get("SNR","0.1")); K=int(os.environ.get("K","1"))
SWING=math.radians(float(os.environ.get("SWING","15"))); NCONF=int(os.environ.get("NCONF","16"))
U_INIT=os.environ.get("U_INIT","random"); ISO=dict(astig=0.0); EB=32
U_KMAX=float(os.environ.get("U_KMAX","0"))   # >0: band-limit the components to |k|<=U_KMAX after each M-step
U_SHRINK=os.environ.get("U_SHRINK","1")=="1"   # shrink each component per shell by the half-set FSC of the component
POSE_FROM=os.environ.get("POSE_FROM","v0")     # v0: in-plane angle and mode weights from the V0-only correlation (the pose
# search), the latent estimated at that fixed pose. "oracle": each particle's TRUE pose and in-plane angle, no
# search at all (isolates the pilot's post-steps from the grid pose error). "marginal": see below.
# search), the latent estimated at that fixed pose. "marginal": pick gamma and weight modes by the K-component
# marginal likelihood. The marginal choice is a selection bias: out of ~2,560 (mode, gamma) candidates it
# prefers those where the noise projects onto the component, so E[z] is inflated in magnitude, and correlated
# with U, and the M-step grows U each round (5x over 10 iterations at SNR 0.3). The gate showed the shortlist
# does not need heterogeneity, so the pose is taken without it.
SAVE=os.environ.get("SAVE","")                  # directory: dump V0/U of both halves and U_TRUE per iteration (offline per-shell analysis)
GAUGE_G=os.environ.get("GAUGE_G","1")=="1"      # pin the scale gauge the flat gain prior leaves free: after each M-step multiply
# (V0, U) by the mean posterior gain of the E-step that produced them, so the next E-step sees a mean gain of 1.
# Without it any scale error a of the reconstruction operator compounds (s_{t+1} = a s_t: 0.81/round uncorrected,
# ~1.04/round with the measured correction); with it the scale stays at a constant a instead.
GRID_CORR=os.environ.get("GRID_CORR","1")=="1"  # per-shell gridding correction of the M-step (see pipeline.gridding_correction);
# without it the consensus scale drifts down every iteration (V0 norm 77 -> 38 over 5 rounds at D=96) and the
# fitted motion amplitude inflates relative to it. This was the "amplitude bias" of drivers 5-13.
TAU_MODE=os.environ.get("TAU_MODE","residual")  # residual: fit the noise precision per particle from the model residual (EM for
# sigma^2). "known": the generator's sigma. With "known" the E-step treats every un-modelled part of the
# image (the motion the component does not capture, the band-limited part, the pose smear) as absent, so the
# z posterior is overconfident: var(E[z]) came out ~5.7 against a unit prior, and the M-step inflated the
# component 5x. The bias is common to both halves, so nothing half-set based can see it; fitting the noise
# level is the PPCA-consistent step. Diagnostic printed: var(E[z]) + mean posterior var, which should be ~1.
# (Wiener: an estimate whose two independent halves agree with FSC f has SNR f/(1-f); the
# optimal amplitude factor is SNR/(1+SNR) = f. Same per-shell scalar for both halves, the
# coupling RELION's SSNR regularisation also has; the maps themselves stay independent.)
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
ORACLE=POSE_FROM=="oracle"
conf=torch.arange(NP,device=dev)%NCONF
p=generate_physical_hetero(vols,ts,conf,pg,SNR,seed=17,device=dev,ctf_kwargs=ISO,backend=BACKEND)
if ORACLE:
    # the "view grid" becomes the particles' own true directions: view index == particle index
    cpo=p.polars.clone(); caz=p.azimus.clone(); V=NP
    G_ORACLE=torch.zeros(NP,NP,dtype=torch.float64,device=dev); G_ORACLE[torch.arange(NP,device=dev),torch.arange(NP,device=dev)]=p.gammas
beta,tau,C=temperature(p,w,D); In=torch.sqrt((p.images_fourier.abs()**2*w).sum(dim=(1,2))); cr=p.ctf[:,:,0]
p.clean_fourier=None; torch.cuda.empty_cache()        # not needed past the temperature; 1.6 GB at D=96
HALF=[torch.arange(0,NP//2,device=dev),torch.arange(NP//2,NP,device=dev)]
shells=ShellMasks(D,dev); prior_prec=torch.tensor([0.0]+[1.0]*K,dtype=torch.float64,device=dev)

def lowpassed(F,k): return to_real(shells.lowpass(F,k,width=1.0))
cons=torch.stack(vols).mean(0); V0_init=lowpassed(fourier_volume(cons),R_INIT)
# true first principal component, scaled so its coefficient has unit variance over
# the conformations: the amplitude the fitted U should converge to (diagnostic only)
_X=torch.stack([v-cons for v in vols]).reshape(NCONF,-1); _,_Sv,_Vh=torch.linalg.svd(_X,full_matrices=False)
_zt=_X@_Vh[0]; U_TRUE=(_Vh[0]*_zt.std()).reshape(D,D,D); Z_TRUE=(_zt/_zt.std())     # (NCONF,)
F_UTRUE=fourier_volume(U_TRUE); del _X
if K>0 and U_INIT=="truth":
    # first principal direction of the true family, low-passed (upper-bound init)
    X=torch.stack([v-cons for v in vols]).reshape(NCONF,-1)
    _,_,Vh=torch.linalg.svd(X,full_matrices=False)
    U_init=torch.stack([lowpassed(fourier_volume(Vh[k].reshape(D,D,D)*X.norm()/math.sqrt(NCONF)),R_INIT) for k in range(K)])
elif K>0:
    def rand_init(seed):
        g=torch.Generator(device=dev).manual_seed(seed)
        Ui=torch.stack([lowpassed(fourier_volume(torch.randn(D,D,D,generator=g,dtype=torch.float64,device=dev)),R_INIT) for _ in range(K)])
        return Ui*(0.1*V0_init.norm()/Ui.reshape(K,-1).norm(dim=1)[:,None,None,None])
    U_init=rand_init(5)
else:
    U_init=torch.zeros(0,D,D,D,dtype=torch.float64,device=dev)
# INIT_SPLIT=1: each half starts from its OWN random component. With a shared start, the
# part of the start that survives EM is common to both halves, so the half-set FSC
# shrinkage keeps it as if it were signal (observed: the shrunk component's norm doubles
# between iterations 3 and 8 with no growth in its projection on the true component).
INIT_SPLIT=os.environ.get("INIT_SPLIT","1")=="1"

def estep(V0,U,ids,tidx,bg,wt0=None):
    """Coefficient posterior on the shortlist. POSE_FROM=v0: gamma from bg (the V0
    correlation's best in-plane angle per view) and mode weights wt0; the marginal
    is still evaluated over all gamma (it is free) but only used for reporting.
    POSE_FROM=marginal: gamma = argmax of the marginal, weights = softmax of its
    lse over gamma. Returns weights, gamma matrix (n,V), Ea (n,M,K+1),
    Eaa (n,M,K+1,K+1), expected residual and z posterior variance at the top mode."""
    n=len(ids); M=tidx.shape[1]; K1=K+1
    gi0=torch.remainder(torch.round(-torch.gather(bg,1,tidx)/(2*np.pi/P)).long(),P)   # (n,M) V0 gamma index
    vol_list=[V0]+[U[k] for k in range(K)]
    lse=torch.empty(n,M,dtype=torch.float64,device=dev); Ea=torch.empty(n,M,K1,dtype=torch.float64,device=dev)
    Eaa=torch.empty(n,M,K1,K1,dtype=torch.float64,device=dev); bg=bg.clone(); t0=time.monotonic()
    resid=torch.empty(n,M,dtype=torch.float64,device=dev); pvar=torch.empty(n,M,dtype=torch.float64,device=dev)
    for i in range(0,n,EB):
        t0=duty_sleep(t0)
        j=min(i+EB,n); nb=j-i; g_=ids[i:j]; v=tidx[i:j].reshape(-1)
        pts=slice_points(fourier_circles(th,cpo[v],caz[v],torch.zeros_like(caz[v])),rr)
        T=torch.stack([_slice(vk,pts,BACKEND) for vk in vol_list],1).reshape(nb,M,K1,S,P)
        A,b,Gyy=hetero_gram(p.images_fourier[g_],T,p.ctf[g_],w)
        logml,mean,cov=marginal_loglik(A,b,Gyy,tau_cur[g_],prior_prec)   # (nb,M,P)
        lse[i:j]=torch.logsumexp(logml,dim=2)
        gi=logml.argmax(2) if POSE_FROM=="marginal" else gi0[i:j]       # (nb,M)
        bg[torch.arange(i,j,device=dev)[:,None],tidx[i:j]]=-(gi.double())*(2*np.pi/P)
        mu=torch.gather(mean,2,gi[...,None,None].expand(nb,M,1,K1)).squeeze(2)
        cv=torch.gather(cov,2,gi[...,None,None,None].expand(nb,M,1,K1,K1)).squeeze(2)
        Ag=torch.gather(A,2,gi[...,None,None,None].expand(nb,M,1,K1,K1)).squeeze(2)
        bgm=torch.gather(b,2,gi[...,None,None].expand(nb,M,1,K1)).squeeze(2)
        # expected weighted residual  E||y - T a||^2_w  under the posterior of a
        resid[i:j]=(Gyy[:,None]-2*(mu*bgm).sum(-1)+torch.einsum('nmi,nmij,nmj->nm',mu,Ag,mu)
                    +torch.einsum('nmij,nmji->nm',Ag,cv))
        pvar[i:j]=cv[...,-1,-1] if K>0 else 0.0
        Ea[i:j]=mu; Eaa[i:j]=cv+mu[...,:,None]*mu[...,None,:]
        del T,A,b,logml,mean,cov
    wt=torch.softmax(lse-lse.max(1,keepdim=True).values,dim=1) if POSE_FROM=="marginal" else wt0
    top=wt.argmax(1)
    r_sel=resid[torch.arange(n,device=dev),top]; pv_sel=pvar[torch.arange(n,device=dev),top]
    return wt,bg,Ea,Eaa,r_sel,pv_sel

def family_metrics(V0,U,zbar):
    """FSC0.5 crossing AND mean FSC of V0 and of V(zbar_c) against the truth at
    each conformation. The crossing is blind to a correct low-resolution
    component (FSC is ~1 there already); the mean FSC is not."""
    r0=[];rz=[];Fz=[];m0=[];mz=[]
    F0=fourier_volume(V0)
    for c in range(NCONF):
        Vc=V0+(torch.einsum('k,kxyz->xyz',zbar[c],U) if K>0 else 0)
        Fc=fourier_volume(Vc); Fz.append(Fc)
        f0=shells.fsc(F0,Ftrue[c]); fz=shells.fsc(Fc,Ftrue[c])
        r0.append(res_at(f0)); rz.append(res_at(fz)); m0.append(f0.mean()); mz.append(fz.mean())
    return np.mean(r0),np.mean(rz),Fz,np.mean(m0),np.mean(mz)

print(f"LINEAR LATENT | GPU duty {GPU_DUTY:g} | U_SHRINK={int(U_SHRINK)} TAU={TAU_MODE} POSE={POSE_FROM} INIT_SPLIT={int(INIT_SPLIT)} GRID_CORR={int(GRID_CORR)} GAUGE_G={int(GAUGE_G)} | D={D} grid {S}x{P} {V} views @ {VIEW_DEG:.2f} deg | physical SNR {SNR} | K={K} | U_KMAX={U_KMAX:g} | swing +-{math.degrees(SWING):.0f} deg, {NCONF} conformations | "
      f"{NP} particles, GOLD halves | U_init={U_INIT} | C={C.item():.4f} beta median {beta.median():.1f}")
print(f"{'iter':>5}{'rho':>7}{'R2':>7}{'leak':>7}  {'V0':>7}{'V(z)':>7}{'half':>7}   (px at FSC 0.5, mean over conformations)")
print("-"*70)
state=[(V0_init.clone(),(rand_init(5+h) if (K>0 and U_INIT!="truth" and INIT_SPLIT) else U_init).clone()) for h in range(2)]
GCORR=[None,None]                                       # per-half gridding correction, measured at the first iteration's poses
tau_cur=tau.clone()                                    # per-particle noise precision, refitted each E-step
for it in range(1,ITERS+1):
    tau_ratio=[];zcons=[]
    rho=[];r2=[];leak=[];Fz_h=[];m0=[];mz=[];zbars=[None,None];mf0=[];mfz=[];uscale=[];zslope=[];gbars=[]
    for h in (0,1):
        ids=HALF[h]; V0,U=state[h]
        if ORACLE:
            tidx=ids[:,None]; wt0=torch.ones(len(ids),1,dtype=torch.float64,device=dev); bg=G_ORACLE[ids]
        else:
            bank=Bank(V0,th,cpo,caz,rr,S,P,backend=BACKEND,single=SINGLE); bc,bg=bank.search(p.images_fourier[ids],cr[ids],wr,In[ids]); del bank; torch.cuda.empty_cache()
            tidx,wt0,_=select_modes(bc,NB,MODES,beta[ids])
        wt,bg,Ea,Eaa,r_sel,pv_sel=estep(V0,U,ids,tidx,bg,wt0)
        if TAU_MODE=="residual":
            # ML noise precision from the expected residual: tau = D^2 / E||y - T a||^2_w
            # (the same metric that defined C: E||n||^2_w = D^2 sigma^2 C)
            tau_cur[ids]=(D*D)/r_sel.clamp(min=1e-30)
        tau_ratio.append((tau_cur[ids]/tau[ids]).median().item())
        if K>0:
            zm=(wt[...,None]*Ea[...,1:]).sum(1)[:,0]
            zcons.append((zm.var()+pv_sel.mean()).item())
        if GRID_CORR and GCORR[h] is None:
            GCORR[h]=gridding_correction(cons,ids,tidx,bg,th,cpo,caz,rr,w,D,p.ctf,backend=BACKEND)
            _ks=[k for k in (4,12,24) if k<D//2]
            print(f"      half {h}: gridding attenuation a(k) at k={_ks}: "
                  +" ".join(f"{1/GCORR[h][D//2+k,D//2,D//2].item():.3f}" for k in _ks))
        F=reconstruct(p,ids,tidx,wt,bg,th,cpo,caz,rr,w,D,K=K,Ea=Ea,Eaa=Eaa,tau=tau_cur,grid_corr=GCORR[h])
        if GAUGE_G:
            gbar=(wt*Ea[...,0]).sum(1).mean().item()          # mean posterior gain over particles (mode-weighted)
            F=F*gbar; gbars.append(gbar)
        V0=to_real(F[0])
        if K>0:
            # a band-limited component has far fewer noise degrees of freedom
            # (p shrinks ~k^3) while the motion energy is mostly low-k: the
            # 3DVA "filter resolution" trick, here as a hard cosine edge
            U=torch.stack([to_real(shells.lowpass(F[k+1],U_KMAX,width=1.0)) if U_KMAX>0 else to_real(F[k+1]) for k in range(K)])
        if K>0:
            # scale of the raw M-step component against the true PC, |k|<=14 (where both are resolved)
            _Fu=fourier_volume(U[0]); _m=shells.RR<=14
            _s=((_Fu[_m]*F_UTRUE[_m].conj()).sum()/(F_UTRUE[_m].abs()**2).sum()).real.item()
            z_mean=(wt[...,None]*Ea[...,1:]).sum(1)                       # posterior mean z per particle
            # slope of E[z] on the true unit-variance latent: alpha; with U at scale s the
            # product alpha*s should be 1. Both halves' values are averaged in the print.
            _zt_i=Z_TRUE[conf[ids]]; _al=((z_mean[:,0]-z_mean[:,0].mean())*(_zt_i-_zt_i.mean())).sum()/((_zt_i-_zt_i.mean())**2).sum()
            uscale.append(_s); zslope.append(_al.item())
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
        a0,az,Fz,f0,fz=family_metrics(V0,U,zbar); m0.append(a0); mz.append(az); Fz_h.append(Fz); mf0.append(f0); mfz.append(fz)
        state[h]=(V0,U); zbars[h]=zbar
    if K>0 and U_SHRINK:
        # align the halves' components (sign/rotation), shrink by per-shell FSC, redo the metrics
        Ua,Ub=state[0][1],state[1][1]
        Q,Ub_al=align_bases(Ua,Ub)
        Fa=[fourier_volume(Ua[k]) for k in range(K)]; Fb=[fourier_volume(Ub_al[k]) for k in range(K)]
        ushr=[]
        for k in range(K):
            f=np.clip(shells.fsc(Fa[k],Fb[k]),0,1)
            filt=torch.zeros_like(shells.RR)
            for i,m in enumerate(shells.masks): filt[m]=float(f[i])
            Fa[k]=Fa[k]*filt; Fb[k]=Fb[k]*filt; ushr.append(res_at(f))
        Ua=torch.stack([to_real(F) for F in Fa]); Ub_al=torch.stack([to_real(F) for F in Fb])
        Ub=torch.einsum('jk,jxyz->kxyz',Q,Ub_al)          # back to half B's own gauge
        state[0]=(state[0][0],Ua); state[1]=(state[1][0],Ub)
        m0=[];mz=[];Fz_h=[];mf0=[];mfz=[]
        for h in (0,1):
            V0,U=state[h]; zb=zbars[h]
            a0,az,Fz,f0,fz=family_metrics(V0,U,zb); m0.append(a0); mz.append(az); Fz_h.append(Fz); mf0.append(f0); mfz.append(fz)
    half=np.mean([res_at(shells.fsc(Fz_h[0][c],Fz_h[1][c])) for c in range(NCONF)])
    if SAVE:
        os.makedirs(SAVE,exist_ok=True)
        torch.save({"it":it,"V0":[st[0].cpu() for st in state],"U":[st[1].cpu() for st in state],
                    "U_TRUE":U_TRUE.cpu(),"cons":cons.cpu(),"zbars":[zb.cpu() for zb in zbars]},
                   os.path.join(SAVE,f"state_it{it:02d}.pt"))
    print(f"{it:>5}{np.mean(rho) if rho else float('nan'):7.3f}{np.mean(r2) if r2 else float('nan'):7.3f}"
          f"{np.mean(leak) if leak else float('nan'):7.3f}  {np.mean(m0):6.1f}p{np.mean(mz):6.1f}p{half:6.1f}p  meanFSC V0 {np.mean(mf0):.4f} V(z) {np.mean(mfz):.4f}"
          +(f"   |U|/|V0| {unorm[0].item():.3f}  z std {z_mean[:,0].std().item():.3f}" if K>0 else "")
          +(f"  U halfFSC0.5 {ushr[0]}p |U|shr {(state[0][1].reshape(K,-1).norm(dim=1)[0]/state[0][0].norm()).item():.3f}" if K>0 and U_SHRINK else "")
          +f"  |V0| {state[0][0].norm().item():.1f}"+(f"  gbar {np.mean(gbars):.3f}" if gbars else "")+f"  tau/tau0 {np.mean(tau_ratio):.3f}"+(f"  z-consistency {np.mean(zcons):.2f}" if zcons else "")
          +(f"  rawU/Utrue(k<=14) {np.mean(uscale):.3f}  dE[z]/dz_true {np.mean(zslope):.3f}  product {np.mean(uscale)*np.mean(zslope):.3f}" if uscale else ""))
if K>1:
    Q,_=align_bases(state[0][1],state[1][1])
    print(f"basis agreement between halves (Procrustes): {torch.diagonal(Q).abs().mean().item():.3f} "
          f"(1 = the halves found the same components in the same order)")
