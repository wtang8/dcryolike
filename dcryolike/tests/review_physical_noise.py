"""REVIEW CHECK, part 2: which noise model is PHYSICAL?

Real cryo-EM noise is white in real space. Its Fourier transform sampled on a
polar grid has UNIFORM variance per point but is CORRELATED between nearby
points (the transform of a finite image is smooth). Inner shells pack many
points into little area -> heavily correlated; outer shells are sparse ->
nearly independent. CryoLike's quadrature weighting is the standard way to
approximate the integral over that correlated field.

Here the noise is generated exactly: white real-space noise, exact DFT to the
polar points. If the WEIGHTED CC now wins, the likelihood was right all along
and the generator's iid noise was the unphysical part.
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

# exact DFT matrix: real-space pixels -> polar Fourier points
kx=torch.tensor(pg.x_points,dtype=vol.dtype,device=dev); ky=torch.tensor(pg.y_points,dtype=vol.dtype,device=dev)
ax=torch.arange(D,dtype=vol.dtype,device=dev)-D//2
X,Y=torch.meshgrid(ax,ax,indexing="ij")
phase=(-2j*np.pi/D)*(kx[:,None]*X.reshape(1,-1)+ky[:,None]*Y.reshape(1,-1))
DFT=torch.exp(phase.to(torch.complex128))/D          # (S*P, D*D), ortho-ish scale
print(f"DFT matrix {tuple(DFT.shape)} = {DFT.numel()*16/1e6:.0f} MB")

def scores(imgs,ctf,wvec):
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

ones=torch.ones_like(wr); hdr="".join(f"{('@'+str(k)):>8}" for k in KS)
print(f"\n=== generated noise: WHITE IN REAL SPACE, exact DFT to polar (physical) ===")
print(f"{'SNR':>6}{'statistic':>13}{hdr}")
for snr in (0.1,0.05):
    az,po,ga=uniform_so3(N,seed=17,device=dev,dtype=vol.dtype)
    ctf=make_ctf(pg,N,float(D),seed=17,device=dev,dtype=vol.dtype,**ISO)
    clean=slice_nufft(vol,slice_points(fourier_circles(th,po,az,ga),rr)).reshape(N,S,P)*ctf
    power=(clean.abs()**2*w).sum(dim=(1,2))/w.sum()
    g=torch.Generator(device=dev).manual_seed(24)
    nreal=torch.randn(N,D*D,generator=g,dtype=vol.dtype,device=dev)
    npol=(nreal.to(torch.complex128)@DFT.T).reshape(N,S,P)
    # rescale so the w-weighted mean noise power equals power/snr (same SNR convention)
    npow=(npol.abs()**2*w).sum(dim=(1,2))/w.sum()
    npol=npol*torch.sqrt((power/snr)/npow)[:,None,None]
    imgs=clean+npol
    nt=torch.stack([torch.sin(po)*torch.cos(az),torch.sin(po)*torch.sin(az),torch.cos(po)],-1)
    ang=torch.rad2deg(torch.arccos((nt@nd.T).clamp(-1,1)))
    rw=recall(scores(imgs,ctf,wr),ang); ru=recall(scores(imgs,ctf,ones),ang)
    print(f"{snr:>6.2f}{'weighted':>13}"+"".join(f"{x:7.1f}%" for x in rw)+"   <- CryoLike")
    print(f"{'':6}{'unweighted':>13}"+"".join(f"{x:7.1f}%" for x in ru))
    print(f"{'':6}{'diff':>13}"+"".join(f"{u-x:+7.1f} " for x,u in zip(rw,ru)))
# how correlated IS the physical noise between adjacent polar points?
n1=npol[:,:,0]; n2=npol[:,:,1]
corr=((n1*n2.conj()).sum(0)/torch.sqrt((n1.abs()**2).sum(0)*(n2.abs()**2).sum(0))).real
print(f"\nnoise correlation between adjacent in-plane points:")
print(f"  inner shell r={rr[0]:.1f}: {corr[0]:.3f}   mid r={rr[S//2]:.1f}: {corr[S//2]:.3f}   outer r={rr[-1]:.1f}: {corr[-1]:.3f}")
print("  (iid noise would give ~0 everywhere; physical noise is strongly correlated inside)")
