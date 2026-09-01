"""How much does gradient refinement against the exact likelihood buy,
starting from the best pose the 2.39-degree search grid can offer?"""
import sys, numpy as np, torch
sys.path.insert(0,"/home/wtang/cryoem/dcryolike"); sys.path.insert(0,"/home/wtang/cryoem/cryolike/src")
from cryolike.grids import PolarGrid
from cryolike.metadata import ViewingAngles
from dcryolike.generator import gaussian_atom_volume, generate
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft
from dcryolike.refine import refine_poses

dev="cuda"; D=48; N=96; ISO=dict(astig=0.0)
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

def grid_search(imgs,ctf):
    """Best grid view + its best in-plane angle -- the coarse stage."""
    cr=ctf[:,:,0]; Tf=torch.fft.fft(Tb,dim=-1).conj(); If=torch.fft.fft(imgs,dim=-1)
    Tn=torch.sqrt(torch.einsum('vr,br->bv',(Tb.abs()**2).sum(-1),wr[None,:]*cr**2))
    In=torch.sqrt((imgs.abs()**2*w).sum(dim=(1,2)))
    bi=torch.empty(imgs.shape[0],dtype=torch.long,device=dev)
    bg=torch.empty(imgs.shape[0],dtype=vol.dtype,device=dev)
    for i in range(0,imgs.shape[0],8):
        j=min(i+8,imgs.shape[0]); cw=(wr[None,:]*cr[i:j])[:,:,None]
        cc=torch.fft.ifft(torch.einsum('vrk,brk->bvk',Tf,If[i:j]*cw),dim=-1).real
        cc=cc/(Tn[i:j,:,None]*In[i:j,None,None])
        best,gi=cc.amax(-1),cc.argmax(-1)
        bi[i:j]=best.argmax(1)
        bg[i:j]=-(gi.gather(1,best.argmax(1,keepdim=True)).squeeze(1).to(vol.dtype))*(2*np.pi/P)
    return bi,bg

def ang_err(a,p,at,pt):
    n1=torch.stack([torch.sin(p)*torch.cos(a),torch.sin(p)*torch.sin(a),torch.cos(p)],-1)
    n2=torch.stack([torch.sin(pt)*torch.cos(at),torch.sin(pt)*torch.sin(at),torch.cos(pt)],-1)
    return torch.rad2deg(torch.arccos((n1*n2).sum(-1).clamp(-1,1)))

# Assertions, so this file verifies rather than merely reporting:
#  A. refinement must not make the objective worse (it maximises cc)
#  B. refinement must not make the angular error worse than the grid start
#  C. at SNR 1.0 it must reach the information limit (~1.2 deg for D=48)
ok = True

def cc_of(a,p,g,imgs,ctf):
    T=slice_nufft(vol,slice_points(fourier_circles(th,p,a,g),rr)).reshape(-1,S,P)*ctf
    num=(T.conj()*imgs*w).sum(dim=(1,2)).real
    return num/(torch.sqrt((T.abs()**2*w).sum(dim=(1,2)))
                *torch.sqrt((imgs.abs()**2*w).sum(dim=(1,2))))

print("angular error, median / 90th percentile, degrees. grid spacing 2.39 deg.\n")
print(f"{'SNR':>6}{'grid search':>16}{'+ refinement':>16}{'improvement':>14}")
print("-"*52)
for snr in (1.0,0.5,0.1,0.05,0.02):
    p=generate(vol,pg,N,snr,seed=17,device=dev,ctf_kwargs=ISO)
    bi,bg=grid_search(p.images_fourier,p.ctf)
    e0=ang_err(caz[bi],cpo[bi],p.azimus,p.polars)
    a1,p1,g1=refine_poses(vol,p.images_fourier,p.ctf,th,rr,w,
                          caz[bi].clone(),cpo[bi].clone(),bg.clone(),steps=80,lr=0.015)
    e1=ang_err(a1,p1,p.azimus,p.polars)
    c0=cc_of(caz[bi],cpo[bi],bg,p.images_fourier,p.ctf)
    c1=cc_of(a1,p1,g1,p.images_fourier,p.ctf)
    a_ok=(c1>=c0-1e-9).float().mean().item()>0.99          # A: objective improves
    b_ok=e1.median()<=e0.median()*1.02                     # B: error not worse
    c_ok=(e1.median()<2.0) if snr>=1.0 else True           # C: hits the limit
    ok &= a_ok and b_ok and c_ok
    print(f"{snr:>6.2f}{e0.median():>9.2f} /{e0.quantile(0.9):>5.1f}"
          f"{e1.median():>9.2f} /{e1.quantile(0.9):>5.1f}"
          f"{(e0.median()/e1.median()):>13.2f}x"
          f"   {'PASS' if (a_ok and b_ok and c_ok) else 'FAIL'}"
          f"{'' if a_ok else ' [objective worsened]'}"
          f"{'' if b_ok else ' [error worsened]'}"
          f"{'' if c_ok else ' [missed limit]'}")
print(f"\nverdict: {'PASS' if ok else 'FAIL'}")
import sys as _sys; _sys.exit(0 if ok else 1)
