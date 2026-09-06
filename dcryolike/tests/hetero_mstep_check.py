"""M-step isolation check: does the per-voxel (K+1)x(K+1) solve return the
component at the TRUE amplitude when given the true poses and true latents?

The D=96 linear pilots inflate the component as EM runs (product of component
norm and latent spread reaches 2-5x the true motion). Refitting the noise
precision and taking the pose from V0 alone both changed nothing, so the
suspect is the M-step. This feeds reconstruct() exact posterior moments
(E[a] = (1, z_true), E[a a^T] = E[a]E[a]^T + diag(0, v)) at the true pose and
in-plane angle and compares the returned U with the true first principal
component, per shell, with and without noise, with and without a posterior
variance v, and with E[z] carrying the noise-correlated error the real E-step
produces (z_ls = least-squares z at the true pose).

If the scale is ~1 in every case, the M-step is unbiased and the inflation is
an E-M interaction; if the noisy-E[z] case inflates, that is the mechanism.
"""
import os, math, numpy as np, torch
import _paths  # noqa: F401
from cryolike.grids import PolarGrid
from dcryolike.generator import hinge_phantom, generate_physical_hetero
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft, fourier_volume
from dcryolike.pipeline import temperature, reconstruct, ShellMasks, to_real

_FAILS=[]
def _v(c): _FAILS.append(bool(c)); return 'PASS' if c else 'FAIL'
dev="cuda"; D=int(os.environ.get("D","96")); NP=int(os.environ.get("NP","4000")); NCONF=16
SWING=math.radians(15.0); ISO=dict(astig=0.0); SNR=float(os.environ.get("SNR","0.3"))
ts=list(np.linspace(-SWING,SWING,NCONF)); vols=[hinge_phantom(D,float(t),device=dev)[0] for t in ts]
cons=torch.stack(vols).mean(0)
X=torch.stack([v-cons for v in vols]).reshape(NCONF,-1); _,Sv,Vh=torch.linalg.svd(X,full_matrices=False)
zt=X@Vh[0]; U_TRUE=(Vh[0]*zt.std()).reshape(D,D,D); Z_TRUE=zt/zt.std(); del X
F_UT=fourier_volume(U_TRUE); F_C=fourier_volume(cons); shells=ShellMasks(D,dev)
P=min(256,int(2**np.ceil(np.log2(2*np.pi*(D//2)*0.85))))
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=P,uniform=True); S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=torch.float64,device=dev); rr=torch.tensor(pg.radius_shells,dtype=torch.float64,device=dev)
w=torch.tensor(pg.integration_weight_points,dtype=torch.float64,device=dev).reshape(S,P)
conf=torch.arange(NP,device=dev)%NCONF
p=generate_physical_hetero(vols,ts,conf,pg,SNR,seed=17,device=dev,ctf_kwargs=ISO)
beta,tau,C=temperature(p,w,D); p.clean_fourier=None; torch.cuda.empty_cache()
ids=torch.arange(NP,device=dev); z=Z_TRUE[conf]
# "views" = each particle's own true (polar, azimu); gamma from its true gamma
cpo=p.polars; caz=p.azimus; tidx=ids[:,None]; wt=torch.ones(NP,1,dtype=torch.float64,device=dev)
bg=torch.zeros(NP,NP,dtype=torch.float64,device=dev)   # gam[l, v] lookup: only the diagonal is used
bg[ids,ids]=p.gammas
def scale(Fu,kmax):
    m=shells.RR<=kmax
    return ((Fu[m]*F_UT[m].conj()).sum()/(F_UT[m].abs()**2).sum()).real.item()
def run(label,Ez,v):
    Ea=torch.stack([torch.ones_like(Ez),Ez],1)[:,None,:]                              # (NP,1,2)
    Eaa=Ea[...,:,None]*Ea[...,None,:]; Eaa=Eaa.clone(); Eaa[...,1,1]+=v
    F=reconstruct(p,ids,tidx,wt,bg,th,cpo,caz,rr,w,D,K=1,Ea=Ea,Eaa=Eaa,tau=tau)
    s14,s24,s40=scale(F[1],14),scale(F[1],24),scale(F[1],40)
    v0=((F[0][shells.RR<=24]*F_C[shells.RR<=24].conj()).sum()/(F_C[shells.RR<=24].abs()**2).sum()).real.item()
    print(f"  {label:44s} U scale k<=14 {s14:.3f}  k<=24 {s24:.3f}  k<=40 {s40:.3f} | V0 scale {v0:.3f}")
    return s14,s24
print(f"D={D} SNR {SNR} {NP} particles, true poses; |U_true|/|V0| = {(U_TRUE.norm()/cons.norm()).item():.3f}")
# 1. exact latents, no posterior variance
s_a=run("E[z]=z_true, v=0",z,0.0)
# 2. exact latents plus the posterior variance the E-step would carry (d ~ tau |T_U|^2 ~ 70 -> v ~ 0.014)
s_b=run("E[z]=z_true, v=1/(1+70)",z,1.0/71)
# 3. least-squares z at the true pose (noise-correlated error, as the real E-step)
pts=slice_points(fourier_circles(th,p.polars,p.azimus,p.gammas),rr)
zls=torch.empty(NP,dtype=torch.float64,device=dev)
for i in range(0,NP,256):
    j=min(i+256,NP)
    T0=slice_nufft(cons,pts[i:j]).reshape(j-i,S,P)*p.ctf[i:j]; TU=slice_nufft(U_TRUE,pts[i:j]).reshape(j-i,S,P)*p.ctf[i:j]
    def ip(a,b_): return (a.conj()*b_*w).sum(dim=(1,2)).real
    G=torch.stack([torch.stack([ip(T0,T0),ip(T0,TU)],-1),torch.stack([ip(TU,T0),ip(TU,TU)],-1)],-2)
    rhs=torch.stack([ip(T0,p.images_fourier[i:j]),ip(TU,p.images_fourier[i:j])],-1)
    zls[i:j]=torch.linalg.solve(G,rhs[...,None])[...,1,0]
print(f"  LS latent: slope on z_true {((zls-zls.mean())*(z-z.mean())).sum().item()/((z-z.mean())**2).sum().item():.3f}, "
      f"var {zls.var().item():.3f} (true 1 + 1/d)")
s_c=run("E[z]=z_LS (noise-correlated), v=0",zls,0.0)
# 4. shrunk-prior posterior mean, as the E-step: E[z] = d/(d+1) z_LS, v = 1/(d+1) with d = tau |T_U|^2 per particle
d=tau*ip(TU,TU).mean()   # crude single d for all (median-ish)
dd=d.median() if d.dim()>0 else d
zpm=zls*(dd/(dd+1)); s_d=run(f"E[z]=posterior mean (d={dd.item():.0f}), v=1/(d+1)",zpm,1.0/(dd.item()+1))
ok=abs(s_a[1]-1)<0.1 and abs(s_b[1]-1)<0.1
print(f"\nM-step returns the true amplitude from exact latents (k<=24 within 10%): {_v(ok)}")
print(f"noise-correlated latents inflate it: k<=24 scale {s_c[1]:.3f} (LS), {s_d[1]:.3f} (posterior mean)")
import sys; sys.exit(0 if all(_FAILS) else 1)
