"""E-step isolation check: latent recovery at the TRUE pose with the TRUE component.

The pilots never reached rho > 0.15 even from a truth-initialised component.
Theory (data precision on z = |T_U|^2 / (sigma^2 C) ~ SNR_box D^2 |U|^2/|V|^2)
says a per-particle latent estimate should correlate strongly with the true t
at SNR 0.1 when U is the true full-resolution component. This runs the exact
E-step (hetero_gram + marginal_loglik) with M=1 candidate = the true view,
every gamma, and compares z_hat with t. Also checks that the argmax gamma
recovers the true in-plane angle, and cross-checks z_hat against a plain
least-squares fit at the true gamma (no marginalisation).

If rho is high here, the E-step is right and the pilot's failure is in the
component's resolution/scale or the shortlist. If rho is low here, the E-step
has a convention bug that the random-tensor unit tests could not see.
"""
import os, math, numpy as np, torch
import _paths  # noqa: F401
from cryolike.grids import PolarGrid
from dcryolike.generator import hinge_phantom, generate_physical_hetero
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft
from dcryolike.hetero import hetero_gram, marginal_loglik, rank_correlation
from dcryolike.pipeline import temperature

_FAILS=[]
def _v(c): _FAILS.append(bool(c)); return 'PASS' if c else 'FAIL'
dev="cuda"; D=int(os.environ.get("D","32")); NP=int(os.environ.get("NP","800")); NCONF=16
SWING=math.radians(float(os.environ.get("SWING","15"))); ISO=dict(astig=0.0)
ts=list(np.linspace(-SWING,SWING,NCONF)); vols=[hinge_phantom(D,float(t),device=dev)[0] for t in ts]
cons=torch.stack(vols).mean(0)
X=torch.stack([v-cons for v in vols]).reshape(NCONF,-1)
_,Sv,Vh=torch.linalg.svd(X,full_matrices=False)
U=Vh[0].reshape(D,D,D); ztrue_conf=(X@Vh[0])                      # true z of each conformation (U unit norm)
U=U*ztrue_conf.std(); ztrue_conf=ztrue_conf/ztrue_conf.std()       # scale so z has unit variance
print(f"D={D} swing +-{math.degrees(SWING):.0f} deg: first PC explains {(Sv[0]**2/(Sv**2).sum()).item()*100:.1f}% "
      f"of the family variance; |U|/|V0| = {(U.norm()/cons.norm()).item():.3f}")
P=int(os.environ.get("P_INPLANE",str(min(256,int(2**np.ceil(np.log2(2*np.pi*(D//2)*0.85)))))))
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=P,uniform=True); S,P=pg.n_shells,pg.n_inplanes
print(f"polar grid {S}x{P}")
th=torch.tensor(pg.theta_shell,dtype=torch.float64,device=dev); rr=torch.tensor(pg.radius_shells,dtype=torch.float64,device=dev)
w=torch.tensor(pg.integration_weight_points,dtype=torch.float64,device=dev).reshape(S,P)
conf=torch.arange(NP,device=dev)%NCONF
for snr in (1.0,0.3,0.1,0.05):
    p=generate_physical_hetero(vols,ts,conf,pg,snr,seed=17,device=dev,ctf_kwargs=ISO)
    beta,tau,C=temperature(p,w,D)
    # slices of V0 and U at the true view, gamma=0
    pts=slice_points(fourier_circles(th,p.polars,p.azimus,torch.zeros_like(p.azimus)),rr)
    T=torch.stack([slice_nufft(cons,pts),slice_nufft(U,pts)],1).reshape(NP,1,2,S,P)
    A,b,Gyy=hetero_gram(p.images_fourier,T,p.ctf,w)
    logml,mean,cov=marginal_loglik(A,b,Gyy,tau,torch.tensor([0.0,1.0],dtype=torch.float64,device=dev))
    ar=torch.arange(NP,device=dev)
    def rec(gi):
        dg=torch.remainder(-(gi.double())*(2*np.pi/P)-p.gammas+np.pi,2*np.pi)-np.pi
        return (dg.abs()<2.5*np.pi/P).double().mean().item()*100
    gi_cc=b[:,0,:,0].argmax(1)                                       # plain correlation with V0 (sign-aware)
    gi_ml=logml[:,0].argmax(1)                                       # marginal, flat gain (sign-blind)
    gi_pos=logml[:,0].masked_fill(mean[:,0,:,0]<=0,-1e30).argmax(1)  # marginal, gain constrained > 0
    A0,b0=A[...,:1,:1],b[...,:1]
    lm0,_,_=marginal_loglik(A0,b0,Gyy,tau,torch.tensor([0.0],dtype=torch.float64,device=dev))
    gi_k0=lm0[:,0].argmax(1)                                         # K=0 marginal
    print(f"SNR {snr:4.2f}: gamma within 1 step -- CC {rec(gi_cc):5.1f}%  K=0 marginal {rec(gi_k0):5.1f}%  "
          f"K=1 marginal {rec(gi_ml):5.1f}%  K=1 gain>0 {rec(gi_pos):5.1f}%")
    gi=gi_pos; dg=torch.remainder(-(gi.double())*(2*np.pi/P)-p.gammas+np.pi,2*np.pi)-np.pi
    zh=mean[ar,0,gi,1]; gh=mean[ar,0,gi,0]
    zh_cc=mean[ar,0,gi_cc,1]
    zt=ztrue_conf[conf]
    # plain least squares at the TRUE gamma, no marginalisation, as a cross-check
    pts_t=slice_points(fourier_circles(th,p.polars,p.azimus,p.gammas),rr)
    T0=slice_nufft(cons,pts_t).reshape(NP,S,P)*p.ctf; TU=slice_nufft(U,pts_t).reshape(NP,S,P)*p.ctf
    def ip(a,b_): return (a.conj()*b_*w).sum(dim=(1,2)).real
    G=torch.stack([torch.stack([ip(T0,T0),ip(T0,TU)],-1),torch.stack([ip(TU,T0),ip(TU,TU)],-1)],-2)
    rhs=torch.stack([ip(T0,p.images_fourier),ip(TU,p.images_fourier)],-1)
    ls=torch.linalg.solve(G,rhs[...,None])[...,0]
    prec=(tau*ip(TU,TU)).median().item()
    rho=rank_correlation(zh,zt); rho_ls=rank_correlation(ls[:,1],zt)
    slope=((zh-zh.mean())*(zt-zt.mean())).sum()/((zt-zt.mean())**2).sum()
    print(f"          gain median {gh.median().item():.3f} | z data precision median {prec:.1f} (expect rho ~ {math.sqrt(prec/(prec+1)):.2f}) | "
          f"rho(z_hat,t): gain>0 gamma {rho:.3f}, CC gamma {rank_correlation(zh_cc,zt):.3f}, LS at true gamma {rho_ls:.3f} | slope {slope.item():.2f}")
    if snr==0.1: ok=rho>0.3 and rho_ls>0.3 and rec(gi_pos)>0.9*rec(gi_cc)
print(f"\nAt SNR 0.1 the E-step matches the LS bound and the gain>0 angle matches CC: {_v(ok)}")
import sys; sys.exit(0 if all(_FAILS) else 1)
