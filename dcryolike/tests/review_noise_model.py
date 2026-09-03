"""REVIEW CHECK: is the weighted CC actually the matched filter for our noise?

The generator adds iid noise (same variance at every polar grid point). The
likelihood weights each point by the quadrature weight w, which spans 111x
across shells. Weighted CC is optimal for noise with variance ~ 1/w; unweighted
CC is optimal for iid noise. If unweighted beats weighted on our data, the
"ceiling" was measured with a suboptimal statistic and is too low.
"""
import _paths  # noqa
import numpy as np, torch
from cryolike.grids import PolarGrid
from cryolike.metadata import ViewingAngles
from dcryolike.generator import gaussian_atom_volume, uniform_so3, make_ctf
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft

dev="cuda"; D=48; N=300; TOL=3.0; KS=[1,25,100,500]; ISO=dict(astig=0.0)
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

def scores(imgs,ctf,wvec):
    """wvec: per-shell weights used by the statistic (wr = CryoLike, ones = unweighted)."""
    cr=ctf[:,:,0]
    Tn=torch.sqrt(torch.einsum('vr,br->bv',Tpow,wvec[None,:]*cr**2))
    In=torch.sqrt((imgs.abs()**2*wvec[None,:,None]).sum(dim=(1,2)))
    o=torch.empty(imgs.shape[0],V,device=dev,dtype=vol.dtype)
    for i in range(0,imgs.shape[0],8):
        j=min(i+8,imgs.shape[0]); cw=(wvec[None,:]*cr[i:j])[:,:,None]
        If=torch.fft.fft(imgs[i:j],dim=-1)
        cc=torch.fft.ifft(torch.einsum('vrk,brk->bvk',Tf,If*cw),dim=-1).real
        o[i:j]=cc.amax(-1)/(Tn[i:j]*In[i:j,None])
    return o

def recall(sc,ang):
    o=sc.argsort(dim=1,descending=True); hit=torch.gather(ang,1,o)<=TOL
    f=torch.where(hit.any(1),hit.float().argmax(1),torch.full((sc.shape[0],),V,device=dev))
    return [(f<k).float().mean().item()*100 for k in KS]

def make(snr,noise_model,seed=17):
    az,po,ga=uniform_so3(N,seed=seed,device=dev,dtype=vol.dtype)
    ctf=make_ctf(pg,N,float(D),seed=seed,device=dev,dtype=vol.dtype,**ISO)
    clean=slice_nufft(vol,slice_points(fourier_circles(th,po,az,ga),rr)).reshape(N,S,P)*ctf
    power=(clean.abs()**2*w).sum(dim=(1,2))/w.sum()
    sig=torch.sqrt(power/snr)
    g=torch.Generator(device=dev).manual_seed(seed+7)
    n=torch.randn(clean.shape,generator=g,dtype=clean.dtype,device=dev)
    if noise_model=="iid":
        imgs=clean+n*sig[:,None,None]
    else:  # variance ~ 1/w, scaled so the w-weighted mean noise power is still sig^2
        scale=torch.sqrt(w.mean()/w)[None]
        imgs=clean+n*scale*sig[:,None,None]
    nt=torch.stack([torch.sin(po)*torch.cos(az),torch.sin(po)*torch.sin(az),torch.cos(po)],-1)
    ang=torch.rad2deg(torch.arccos((nt@nd.T).clamp(-1,1)))
    return imgs,ctf,ang

ones=torch.ones_like(wr)
hdr="".join(f"{('@'+str(k)):>8}" for k in KS)
for nm in ("iid","1/w"):
    print(f"\n=== generated noise: {nm:<4} {'(what generator.py does)' if nm=='iid' else '(CryoLike-consistent)'} ===")
    print(f"{'SNR':>6}{'statistic':>13}{hdr}")
    for snr in (0.1,0.05):
        imgs,ctf,ang=make(snr,nm)
        rw=recall(scores(imgs,ctf,wr),ang); ru=recall(scores(imgs,ctf,ones),ang)
        print(f"{snr:>6.2f}{'weighted':>13}"+"".join(f"{x:7.1f}%" for x in rw)+"   <- CryoLike")
        print(f"{'':6}{'unweighted':>13}"+"".join(f"{x:7.1f}%" for x in ru))
        print(f"{'':6}{'diff':>13}"+"".join(f"{u-x:+7.1f} " for x,u in zip(rw,ru)))
