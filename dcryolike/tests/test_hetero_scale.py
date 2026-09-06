"""Regression test for the reconstruction scale bug (found 6 Sep 2026).

THE BUG. The trilinear splat with per-voxel normalisation returns a volume
scaled DOWN (measured 0.82 for the consensus, 0.75 for a component at D=96).
The gain has a flat prior, so each E-step silently absorbs that factor and the
next M-step shrinks the volume again: over five iterations the consensus norm
fell 77 -> 38 while the component's norm, held by the unit prior on the latent,
stayed at 4.1-4.3. FSC is scale-invariant, so no resolution number showed it;
what showed was V(z) drifting below V0 after iteration 5, which was misread for
six runs as the COMPONENT inflating.

THE FIXES, both under test here:
  1. pipeline.gridding_correction  - per-shell measurement of the attenuation,
     applied to the M-step output (reconstruct(..., grid_corr=...)).
  2. the gauge pin in the pilot     - multiply (V0, U) by the mean posterior
     gain after each M-step, so a residual constant scale error cannot compound.

TEST 1 measures the attenuation and asserts the correction removes it.
TEST 2 runs the E-M loop at true poses in all four on/off combinations of the
two fixes. It asserts the shipping configuration is stationary, that with
NEITHER fix the scale runs away (the negative control: a no-op fix cannot pass),
and it records which of the two is load-bearing. Writing this test corrected the
first account of the fix: the gauge pin alone holds the scale, and the
correction only sets the constant it holds at.
TEST 3 asserts the scale-invariance of FSC that made the bug invisible, which
is why the norm has to be asserted directly.

Small box (D=48, 600 particles) so it belongs in the suite: the mechanism is
box-size independent, only the size of the attenuation changes.
"""
import os, math, numpy as np, torch, sys
import _paths  # noqa: F401
from cryolike.grids import PolarGrid
from dcryolike.generator import hinge_phantom, generate_physical_hetero
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft, fourier_volume
from dcryolike.hetero import hetero_gram, marginal_loglik
from dcryolike.pipeline import (temperature, reconstruct, ShellMasks, to_real,
                                gridding_correction)

_FAILS=[]
def _v(c):
    _FAILS.append(bool(c)); return 'PASS' if c else 'FAIL'

dev="cuda"; D=int(os.environ.get("D","48")); NP=int(os.environ.get("NP","600"))
NCONF=8; SWING=math.radians(15.0); SNR=float(os.environ.get("SNR","0.3")); ISO=dict(astig=0.0)
ts=list(np.linspace(-SWING,SWING,NCONF))
vols=[hinge_phantom(D,float(t),device=dev)[0] for t in ts]
cons=torch.stack(vols).mean(0)
X=torch.stack([v-cons for v in vols]).reshape(NCONF,-1)
_,_,Vh=torch.linalg.svd(X,full_matrices=False)
zt=X@Vh[0]; U_TRUE=(Vh[0]*zt.std()).reshape(D,D,D); Z_TRUE=zt/zt.std(); del X
P=min(128,int(2**np.ceil(np.log2(2*np.pi*(D//2)*0.85))))
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=P,uniform=True); S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=torch.float64,device=dev)
rr=torch.tensor(pg.radius_shells,dtype=torch.float64,device=dev)
w=torch.tensor(pg.integration_weight_points,dtype=torch.float64,device=dev).reshape(S,P)
conf=torch.arange(NP,device=dev)%NCONF
p=generate_physical_hetero(vols,ts,conf,pg,SNR,seed=17,device=dev,ctf_kwargs=ISO)
beta,tau,C=temperature(p,w,D); p.clean_fourier=None; torch.cuda.empty_cache()
shells=ShellMasks(D,dev); F_C=fourier_volume(cons)
ids=torch.arange(NP,device=dev); tidx=ids[:,None]
wt=torch.ones(NP,1,dtype=torch.float64,device=dev)
gi_true=torch.remainder(torch.round(-p.gammas/(2*np.pi/P)).long(),P)
bg=torch.zeros(NP,NP,dtype=torch.float64,device=dev)
bg[ids,ids]=-(gi_true.double())*(2*np.pi/P)
cpo=p.polars; caz=p.azimus
def vol_scale(F,Fref,kmax):
    m=shells.RR<=kmax
    return ((F[m]*Fref[m].conj()).real.sum()/(Fref[m].abs()**2).sum()).item()

print("="*64); print("TEST 1: the splat attenuates, and the correction removes it"); print("="*64)
GC=gridding_correction(cons,ids,tidx,bg,th,cpo,caz,rr,w,D,p.ctf)
att=[1/GC[D//2+k,D//2,D//2].item() for k in (4,8,16)]
Ea=torch.stack([torch.ones(NP,dtype=torch.float64,device=dev),Z_TRUE[conf]],1)[:,None,:]
Eaa=Ea[...,:,None]*Ea[...,None,:]
F_raw=reconstruct(p,ids,tidx,wt,bg,th,cpo,caz,rr,w,D,K=1,Ea=Ea,Eaa=Eaa,tau=tau)
F_cor=reconstruct(p,ids,tidx,wt,bg,th,cpo,caz,rr,w,D,K=1,Ea=Ea,Eaa=Eaa,tau=tau,grid_corr=GC)
s_raw=vol_scale(F_raw[0],F_C,D//2-2); s_cor=vol_scale(F_cor[0],F_C,D//2-2)
print(f"  measured attenuation a(k) at k=4,8,16: "+" ".join(f"{a:.3f}" for a in att))
print(f"  V0 scale from the M-step: uncorrected {s_raw:.3f}, corrected {s_cor:.3f}")
print(f"  attenuation is real (uncorrected < 0.95) and the correction fixes it "
      f"(corrected within 10% of 1): {_v(s_raw<0.95 and abs(s_cor-1)<0.10)}")

print("\n"+"="*64); print("TEST 2: the E-M loop holds the consensus scale; which fix is load-bearing"); print("="*64)
prior=torch.tensor([0.0,1.0],dtype=torch.float64,device=dev)
pts0=slice_points(fourier_circles(th,cpo,caz,torch.zeros_like(caz)),rr)
def estep(V0,U):
    Ea=torch.empty(NP,1,2,dtype=torch.float64,device=dev)
    Eaa=torch.empty(NP,1,2,2,dtype=torch.float64,device=dev)
    for i in range(0,NP,64):
        j=min(i+64,NP)
        T=torch.stack([slice_nufft(V0,pts0[i:j]),slice_nufft(U,pts0[i:j])],1).reshape(j-i,1,2,S,P)
        A,b,Gyy=hetero_gram(p.images_fourier[i:j],T,p.ctf[i:j],w)
        _,mean,cov=marginal_loglik(A,b,Gyy,tau[i:j],prior)
        g=gi_true[i:j]; ar=torch.arange(j-i,device=dev)
        mu=mean[ar,0,g]; cv=cov[ar,0,g]
        Ea[i:j,0]=mu; Eaa[i:j,0]=cv+mu[:,:,None]*mu[:,None,:]
    return Ea,Eaa
def loop(gc,gauge,iters=4):
    V0=cons.clone(); U=0.5*U_TRUE.clone(); norms=[]
    for _ in range(iters):
        Ea,Eaa=estep(V0,U)
        F=reconstruct(p,ids,tidx,wt,bg,th,cpo,caz,rr,w,D,K=1,Ea=Ea,Eaa=Eaa,tau=tau,grid_corr=gc)
        if gauge: F=F*Ea[:,0,0].mean().item()
        V0=to_real(F[0]); U=to_real(F[1]); norms.append(V0.norm().item())
    return norms
cases={"correction + gauge (shipping)":(GC,True), "gauge only":(None,True),
       "correction only":(GC,False), "neither (the bug)":(None,False)}
res={}
for name,(gc,gauge) in cases.items():
    n=loop(gc,gauge); d=n[-1]/n[0]-1; res[name]=d
    print(f"  |V0| per iteration, {name:30s}: "+" ".join(f"{x:.1f}" for x in n)+f"   drift {d*100:+.1f}%")
print(f"  the shipping configuration is stationary (|drift| < 5%): {_v(abs(res['correction + gauge (shipping)'])<0.05)}")
print(f"  NEGATIVE CONTROL -- with neither fix the scale runs away (drift < -10%): "
      f"{_v(res['neither (the bug)']<-0.10)}")
print(f"  the gauge pin is the load-bearing fix: it is stationary on its own "
      f"({res['gauge only']*100:+.1f}%) while the correction alone is not "
      f"({res['correction only']*100:+.1f}%): "
      f"{_v(abs(res['gauge only'])<0.05 and abs(res['correction only'])>abs(res['gauge only']))}")

print("\n"+"="*64); print("TEST 3: FSC cannot see a scale error (why the norm is asserted directly)"); print("="*64)
f_ref=shells.fsc(F_C,fourier_volume(vols[0]))
f_scaled=shells.fsc(F_C*0.4,fourier_volume(vols[0]))
same=float(np.abs(f_ref-f_scaled).max())
print(f"  FSC of V0 vs a conformation, and of 0.4*V0 vs the same: max difference {same:.2e}")
print(f"  scale-invariant (difference < 1e-9): {_v(same<1e-9)}")

print("\n"+("ALL PASS" if all(_FAILS) else "SOME FAIL"), f"({sum(_FAILS)}/{len(_FAILS)})")
sys.exit(0 if all(_FAILS) else 1)
