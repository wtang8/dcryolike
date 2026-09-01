"""BASIN-TAIL ANALYSIS.

The 38x grid reduction rests on one assumption: that a coarse search lands in the
correct basin, after which refinement supplies precision. Refinement is local, so
it cannot escape a wrong basin -- it converges confidently to the wrong answer.
All previous numbers were medians, which hide exactly that.

Two measurements:
  A. CAPTURE RADIUS -- perturb the TRUE pose by a known amount and refine. How
     far off can you start and still come back? This sets how coarse a grid may be.
  B. END-TO-END TAIL -- grid search + refine at each grid spacing, reporting the
     full error distribution, not the median.
"""
import sys, time, numpy as np, torch
sys.path.insert(0,"/home/wtang/cryoem/dcryolike"); sys.path.insert(0,"/home/wtang/cryoem/cryolike/src")
from cryolike.grids import PolarGrid
from cryolike.metadata import ViewingAngles
from dcryolike.generator import gaussian_atom_volume, generate
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft
from dcryolike.refine import refine_poses

dev="cuda"; D=48; N=256; ISO=dict(astig=0.0)
vol=gaussian_atom_volume(D,device=dev)
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=128,uniform=True)
S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=vol.dtype,device=dev)
rr=torch.tensor(pg.radius_shells,dtype=vol.dtype,device=dev)
w=torch.tensor(pg.integration_weight_points,dtype=vol.dtype,device=dev).reshape(S,P); wr=w[:,0]

def ang_err(a,p,at,pt):
    n1=torch.stack([torch.sin(p)*torch.cos(a),torch.sin(p)*torch.sin(a),torch.cos(p)],-1)
    n2=torch.stack([torch.sin(pt)*torch.cos(at),torch.sin(pt)*torch.sin(at),torch.cos(pt)],-1)
    return torch.rad2deg(torch.arccos((n1*n2).sum(-1).clamp(-1,1)))

g=torch.Generator(device=dev).manual_seed(11)

print("="*74)
print("A. CAPTURE RADIUS - how far off can refinement start and still recover?")
print("="*74)
print("   (binned by ACTUAL starting error; 'recovered' = final error < 5 deg)\n")
BINS=[(0,3),(3,6),(6,12),(12,25),(25,50),(50,180)]
for snr in (1.0,0.1,0.05,0.02):
    p=generate(vol,pg,N,snr,seed=17,device=dev,ctf_kwargs=ISO)
    scale=torch.exp(torch.empty(N,device=dev,dtype=vol.dtype).uniform_(np.log(0.01),np.log(1.2)))
    a0=p.azimus+torch.randn(N,generator=g,device=dev,dtype=vol.dtype)*scale
    p0=p.polars+torch.randn(N,generator=g,device=dev,dtype=vol.dtype)*scale
    g0=p.gammas+torch.randn(N,generator=g,device=dev,dtype=vol.dtype)*scale
    e0=ang_err(a0,p0,p.azimus,p.polars)
    a1,p1,_=refine_poses(vol,p.images_fourier,p.ctf,th,rr,w,a0,p0,g0,steps=300,lr=0.02)
    e1=ang_err(a1,p1,p.azimus,p.polars)
    print(f"  SNR {snr}")
    print(f"    {'start error':>14}{'n':>6}{'median final':>15}{'recovered':>12}")
    for lo,hi in BINS:
        m=(e0>=lo)&(e0<hi)
        if m.sum()<3: continue
        print(f"    {f'{lo}-{hi} deg':>14}{int(m.sum()):>6}{e1[m].median().item():>13.2f}d"
              f"{100*(e1[m]<5).float().mean().item():>11.0f}%")
    print()

print("="*74)
print("B. END-TO-END TAIL - grid search + refinement, full distribution")
print("="*74)
print("   catastrophic = final error > 20 deg (converged in the wrong basin)\n")
for spacing in (2.39,5.0,10.0,20.0):
    va=ViewingAngles.from_viewing_distance(np.radians(spacing))
    caz=va.azimus.to(dev).to(vol.dtype); cpo=va.polars.to(dev).to(vol.dtype); V=caz.shape[0]
    Tb=slice_nufft(vol,slice_points(fourier_circles(th,cpo,caz,torch.zeros_like(caz)),rr)).reshape(V,S,P)
    Tpow=(Tb.abs()**2).sum(-1); Tf=torch.fft.fft(Tb,dim=-1).conj()
    print(f"  grid {spacing:.2f} deg -> {V} references")
    print(f"    {'SNR':>6}{'median':>10}{'p90':>9}{'p99':>9}{'<5deg':>9}{'catastrophic':>14}")
    for snr in (1.0,0.1,0.05,0.02):
        p=generate(vol,pg,N,snr,seed=17,device=dev,ctf_kwargs=ISO)
        cr=p.ctf[:,:,0]; If=torch.fft.fft(p.images_fourier,dim=-1)
        Tn=torch.sqrt(torch.einsum('vr,br->bv',Tpow,wr[None,:]*cr**2))
        In=torch.sqrt((p.images_fourier.abs()**2*w).sum(dim=(1,2)))
        bi=torch.empty(N,dtype=torch.long,device=dev); bg=torch.empty(N,dtype=vol.dtype,device=dev)
        for i in range(0,N,8):
            j=min(i+8,N); cw=(wr[None,:]*cr[i:j])[:,:,None]
            cc=torch.fft.ifft(torch.einsum('vrk,brk->bvk',Tf,If[i:j]*cw),dim=-1).real
            cc=cc/(Tn[i:j,:,None]*In[i:j,None,None])
            best,gi=cc.amax(-1),cc.argmax(-1)
            bi[i:j]=best.argmax(1)
            bg[i:j]=-(gi.gather(1,best.argmax(1,keepdim=True)).squeeze(1).to(vol.dtype))*(2*np.pi/P)
        a1,p1,_=refine_poses(vol,p.images_fourier,p.ctf,th,rr,w,
                             caz[bi].clone(),cpo[bi].clone(),bg.clone(),steps=200,lr=0.02)
        e=ang_err(a1,p1,p.azimus,p.polars)
        print(f"    {snr:>6.2f}{e.median():>9.2f}d{e.quantile(0.9):>8.1f}d{e.quantile(0.99):>8.1f}d"
              f"{100*(e<5).float().mean():>8.0f}%{100*(e>20).float().mean():>13.0f}%")
    print()
    del Tb,Tf,Tpow; torch.cuda.empty_cache()
