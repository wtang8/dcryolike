"""Recall ceiling under PHYSICAL noise vs box size.

The D=48 corrected ceiling is too hard to be representative (real boxes are
larger and pose information grows ~k_max^2.7). This runs the same experiment
at D=96 on the identical code path, with D=48 alongside for comparison.

SNR is real-space per-pixel over the WHOLE BOX on the CTF-free projection
(what generate_physical uses). The equivalent IN-MASK SNR is reported per row
so the numbers can be read against literature figures, which are closer to
the in-mask convention.

Angular grid spacing and success tolerance both scale with 1/k_max.
"""
import _paths  # noqa
import os, time, numpy as np, torch
from cryolike.grids import PolarGrid
from cryolike.metadata import ViewingAngles
from dcryolike.generator import gaussian_atom_volume, generate_physical, real_space_projections
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft

dev="cuda"; N_TEST=int(os.environ.get("N_TEST","200")); ISO=dict(astig=0.0)
SNRS=[1.0,0.3,0.1,0.05,0.02]; KS=[1,5,25,100,500,2000]
CDT=torch.complex64

def run(D):
    kmax=D//2; P=int(2**np.ceil(np.log2(2*np.pi*kmax*0.85)))   # ~1.18 vox/sample at k_max, as at D=48
    tol=np.degrees(1.25/kmax)
    vol=gaussian_atom_volume(D,device=dev)
    pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=P,uniform=True)
    S=pg.n_shells
    th=torch.tensor(pg.theta_shell,dtype=vol.dtype,device=dev)
    rr=torch.tensor(pg.radius_shells,dtype=vol.dtype,device=dev)
    w=torch.tensor(pg.integration_weight_points,dtype=torch.float32,device=dev).reshape(S,P); wr=w[:,0]
    va=ViewingAngles.from_viewing_distance(1.0/kmax)
    caz=va.azimus.to(dev).to(vol.dtype); cpo=va.polars.to(dev).to(vol.dtype); V=caz.shape[0]
    nd=torch.stack([torch.sin(cpo)*torch.cos(caz),torch.sin(cpo)*torch.sin(caz),torch.cos(cpo)],-1).float()

    t0=time.time()
    Tf=torch.empty(V,S,P,dtype=CDT,device=dev); Tpow=torch.empty(V,S,dtype=torch.float32,device=dev)
    CH=2048
    for i in range(0,V,CH):
        j=min(i+CH,V); z=torch.zeros(j-i,device=dev,dtype=vol.dtype)
        Tb=slice_nufft(vol,slice_points(fourier_circles(th,cpo[i:j],caz[i:j],z),rr)).reshape(j-i,S,P).to(CDT)
        Tpow[i:j]=(Tb.abs()**2).sum(-1); Tf[i:j]=torch.fft.fft(Tb,dim=-1).conj()
        del Tb
    torch.cuda.synchronize(); t_bank=time.time()-t0

    def score_all(imgs,ctf):
        imgs=imgs.to(CDT); cr=ctf[:,:,0].float()
        Tn=torch.sqrt(torch.einsum('vr,br->bv',Tpow,wr[None,:]*cr**2))
        In=torch.sqrt((imgs.abs()**2*w).sum(dim=(1,2)))
        out=torch.empty(imgs.shape[0],V,device=dev,dtype=torch.float32)
        B=4
        for i in range(0,imgs.shape[0],B):
            j=min(i+B,imgs.shape[0]); cw=(wr[None,:]*cr[i:j])[:,:,None].to(CDT)
            If=torch.fft.fft(imgs[i:j],dim=-1)*cw
            cc=torch.fft.ifft(torch.einsum('vrk,brk->bvk',Tf,If),dim=-1).real
            out[i:j]=cc.amax(-1)/(Tn[i:j]*In[i:j,None]); del cc
        return out

    print(f"\n=== D={D}  k_max={kmax}  grid {S}x{P}  {V} views @ {np.degrees(1.0/kmax):.2f} deg  "
          f"tol {tol:.2f} deg  N={N_TEST}  (bank {t_bank:.0f}s, {Tf.numel()*8/1e9:.1f} GB) ===")
    hdr=f"{'SNR box':>8}{'SNR mask':>10}"+"".join(f"{('@'+str(k)):>8}" for k in KS)+f"{'medrank':>9}{'top1err':>9}"
    print(hdr); print("-"*len(hdr))
    for snr in SNRS:
        t1=time.time()
        p=generate_physical(vol,pg,N_TEST,snr,seed=17,device=dev,ctf_kwargs=ISO)
        proj=real_space_projections(vol,p.azimus,p.polars,p.gammas,dev)
        mask=proj.abs()>0.05*proj.abs().amax(dim=(1,2),keepdim=True)
        vbox=proj.var(dim=(1,2)); vmask=torch.stack([proj[i][mask[i]].var() for i in range(N_TEST)])
        snr_mask=(snr*vmask/vbox).mean().item()
        sc=score_all(p.images_fourier,p.ctf)
        nt=torch.stack([torch.sin(p.polars)*torch.cos(p.azimus),torch.sin(p.polars)*torch.sin(p.azimus),torch.cos(p.polars)],-1).float()
        ang=torch.rad2deg(torch.arccos((nt@nd.T).clamp(-1,1)))
        o=sc.argsort(dim=1,descending=True); hit=torch.gather(ang,1,o)<=tol
        first=torch.where(hit.any(1),hit.float().argmax(1),torch.full((N_TEST,),V,device=dev))
        top1=torch.gather(ang,1,o[:,:1]).squeeze(1)
        print(f"{snr:>8.2f}{snr_mask:>10.2f}"+"".join(f"{(first<k).float().mean().item()*100:7.1f}%" for k in KS)
              +f"{first.median().item():9.0f}{top1.median().item():8.1f}°   ({time.time()-t1:.0f}s)")
        del p,sc,ang,o,hit; torch.cuda.empty_cache()
    del Tf,Tpow,vol; torch.cuda.empty_cache()

for D in (48,96):
    run(D)
