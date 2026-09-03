"""CORRECTED recall ceiling: physical (real-space white) noise, real-space SNR.

Supersedes recall_sweep.py, whose iid polar noise is unphysical and reads
far easier than real noise at the same nominal number. Same grid, same
tolerance, same statistic (CryoLike's weighted CC, which IS the matched
filter for this noise).
"""
import _paths  # noqa
import time, numpy as np, torch
from cryolike.grids import PolarGrid
from cryolike.metadata import ViewingAngles
from dcryolike.generator import gaussian_atom_volume, generate_physical
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft

dev="cuda"; D=48; TOL_DEG=3.0; N_TEST=200; ISO=dict(astig=0.0)
SNRS=[1.0,0.3,0.1,0.05,0.02,0.01]; KS=[1,5,10,25,50,100,250,500,1000,2000]
vol=gaussian_atom_volume(D,device=dev)
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=128,uniform=True)
S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=vol.dtype,device=dev)
rr=torch.tensor(pg.radius_shells,dtype=vol.dtype,device=dev)
w=torch.tensor(pg.integration_weight_points,dtype=vol.dtype,device=dev).reshape(S,P); wr=w[:,0]
va=ViewingAngles.from_viewing_distance(1.0/(D/2))
caz=va.azimus.to(dev).to(vol.dtype); cpo=va.polars.to(dev).to(vol.dtype); V=caz.shape[0]
Tb=slice_nufft(vol,slice_points(fourier_circles(th,cpo,caz,torch.zeros_like(caz)),rr)).reshape(V,S,P)
nd=torch.stack([torch.sin(cpo)*torch.cos(caz),torch.sin(cpo)*torch.sin(caz),torch.cos(cpo)],-1)
Tf=torch.fft.fft(Tb,dim=-1).conj(); Tpow=(Tb.abs()**2).sum(-1)

def score_all(imgs,ctf):
    cr=ctf[:,:,0]
    Tn=torch.sqrt(torch.einsum('vr,br->bv',Tpow,wr[None,:]*cr**2))
    In=torch.sqrt((imgs.abs()**2*w).sum(dim=(1,2)))
    out=torch.empty(imgs.shape[0],V,device=dev,dtype=vol.dtype)
    for i in range(0,imgs.shape[0],8):
        j=min(i+8,imgs.shape[0]); cw=(wr[None,:]*cr[i:j])[:,:,None]
        If=torch.fft.fft(imgs[i:j],dim=-1)
        cc=torch.fft.ifft(torch.einsum('vrk,brk->bvk',Tf,If*cw),dim=-1).real
        out[i:j]=cc.amax(-1)/(Tn[i:j]*In[i:j,None])
    return out

print(f"PHYSICAL noise, REAL-SPACE per-pixel SNR (whole box, CTF-free projection)")
print(f"{V} views @ 2.39 deg, {N_TEST} particles/SNR, tol {TOL_DEG} deg, D={D}\n")
hdr="  SNR  "+"".join(f"{('@'+str(k)):>8}" for k in KS)+"  medrank  top1err"
print(hdr); print("-"*len(hdr))
for snr in SNRS:
    p=generate_physical(vol,pg,N_TEST,snr,seed=17,device=dev,ctf_kwargs=ISO)
    sc=score_all(p.images_fourier,p.ctf)
    nt=torch.stack([torch.sin(p.polars)*torch.cos(p.azimus),torch.sin(p.polars)*torch.sin(p.azimus),torch.cos(p.polars)],-1)
    ang=torch.rad2deg(torch.arccos((nt@nd.T).clamp(-1,1)))
    o=sc.argsort(dim=1,descending=True); hit=torch.gather(ang,1,o)<=TOL_DEG
    first=torch.where(hit.any(1),hit.float().argmax(1),torch.full((N_TEST,),V,device=dev))
    top1=torch.gather(ang,1,o[:,:1]).squeeze(1)
    print(f"  {snr:<5.2f}"+"".join(f"{(first<k).float().mean().item()*100:7.1f}%" for k in KS)
          +f"{first.median().item():8.0f}{top1.median().item():9.1f}")
