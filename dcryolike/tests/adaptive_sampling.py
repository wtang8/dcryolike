"""Is hierarchical (resolution-matched) angular sampling safe?

At cutoff k, the angular grid only needs spacing ~1/k, so cost ~ k^4
(views ~ k^2 x points-per-view ~ k^2). The question is whether the TRUE basin
survives pruning when scored on low-frequency shells alone -- because pose
information is concentrated in the outer shells that this defers.

Success = the true viewing direction lies within one grid spacing of at least
one of the top-K candidates at that level (i.e. it survives into the next stage).
"""
import sys, numpy as np, torch
import _paths  # noqa: F401  (sets sys.path relative to this file)
from cryolike.grids import PolarGrid
from cryolike.metadata import ViewingAngles
from dcryolike.generator import gaussian_atom_volume, generate
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft

dev="cuda"; D=48; N=200; ISO=dict(astig=0.0)
vol=gaussian_atom_volume(D,device=dev)
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=128,uniform=True)
S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=vol.dtype,device=dev)
rr=torch.tensor(pg.radius_shells,dtype=vol.dtype,device=dev)
w=torch.tensor(pg.integration_weight_points,dtype=vol.dtype,device=dev).reshape(S,P)
radii=pg.radius_shells
KCUTS=[4,6,8,12,16,23]
full_cost=None

for snr in (0.5,0.1,0.05):
    p=generate(vol,pg,N,snr,seed=17,device=dev,ctf_kwargs=ISO)
    nt=torch.stack([torch.sin(p.polars)*torch.cos(p.azimus),
                    torch.sin(p.polars)*torch.sin(p.azimus),torch.cos(p.polars)],-1)
    print(f"\n=== SNR {snr} ===")
    print(f"{'k_cut':>6}{'spacing':>9}{'views':>8}{'shells':>8}{'rel cost':>10}"
          f"{'survive@1':>11}{'@5':>7}{'@20':>7}{'@100':>7}")
    for kc in KCUTS:
        sp=1.0/kc                                   # radians, matched to cutoff
        va=ViewingAngles.from_viewing_distance(sp)
        caz=va.azimus.to(dev).to(vol.dtype); cpo=va.polars.to(dev).to(vol.dtype); V=caz.shape[0]
        keep=torch.tensor(radii<=kc,device=dev)
        ns=int(keep.sum()); nip=int(2**np.ceil(np.log2(max(8,2*np.pi*kc))))  # adaptive in-plane
        Tb=slice_nufft(vol,slice_points(fourier_circles(th,cpo,caz,torch.zeros_like(caz)),rr)).reshape(V,S,P)
        Tb=Tb[:,keep]; ww=w[keep]; wr=ww[:,0]
        cr=p.ctf[:,keep,0]
        Tf=torch.fft.fft(Tb,dim=-1).conj(); If=torch.fft.fft(p.images_fourier[:,keep],dim=-1)
        Tn=torch.sqrt(torch.einsum('vr,br->bv',(Tb.abs()**2).sum(-1),wr[None,:]*cr**2))
        In=torch.sqrt((p.images_fourier[:,keep].abs()**2*ww).sum(dim=(1,2)))
        sc=torch.empty(N,V,device=dev,dtype=vol.dtype)
        for i in range(0,N,8):
            j=min(i+8,N); cw=(wr[None,:]*cr[i:j])[:,:,None]
            cc=torch.fft.ifft(torch.einsum('vrk,brk->bvk',Tf,If[i:j]*cw),dim=-1).real
            sc[i:j]=cc.amax(-1)/(Tn[i:j]*In[i:j,None])
        nd=torch.stack([torch.sin(cpo)*torch.cos(caz),torch.sin(cpo)*torch.sin(caz),torch.cos(cpo)],-1)
        ang=torch.rad2deg(torch.arccos((nt@nd.T).clamp(-1,1)))
        tol=np.degrees(sp)*1.2                       # one grid cell
        o=sc.argsort(dim=1,descending=True); A=torch.gather(ang,1,o)
        hit=A<=tol
        first=torch.where(hit.any(1),hit.float().argmax(1),torch.full((N,),V,device=dev))
        cost=V*ns*nip
        if kc==KCUTS[-1] and full_cost is None: full_cost=cost
        rc=cost/(full_cost if full_cost else cost)
        print(f"{kc:>6}{np.degrees(sp):>8.1f}d{V:>8}{ns:>8}{rc:>9.3f}"
              f"{100*(first<1).float().mean():>10.0f}%{100*(first<5).float().mean():>6.0f}%"
              f"{100*(first<20).float().mean():>6.0f}%{100*(first<100).float().mean():>6.0f}%")
        del Tb,Tf,If; torch.cuda.empty_cache()
