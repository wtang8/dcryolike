"""Where does a reconstruction iteration spend its time? Per-particle cost of
each stage of the pipeline the measurements support, at D=48:
  coarse search   208-view bank (20 deg grid -- refinement makes this sufficient)
  refinement      gradient pose refinement, 100 steps
  mixture insert  top-10 posterior-weighted trilinear insertion
  wiener          num/den division (once per iteration, amortised)
"""
import _paths  # noqa
import time, numpy as np, torch
from cryolike.grids import PolarGrid
from cryolike.metadata import ViewingAngles
from dcryolike.generator import gaussian_atom_volume, generate_physical
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft
from dcryolike.refine import refine_poses
dev="cuda"; D=48; N=256
vol=gaussian_atom_volume(D,device=dev)
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=128,uniform=True); S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=vol.dtype,device=dev); rr=torch.tensor(pg.radius_shells,dtype=vol.dtype,device=dev)
w=torch.tensor(pg.integration_weight_points,dtype=vol.dtype,device=dev).reshape(S,P); wr=w[:,0]
p=generate_physical(vol,pg,N,0.3,seed=1,device=dev,ctf_kwargs=dict(astig=0.0))
def T(f,reps=3):
    f(); torch.cuda.synchronize(); t=time.time()
    for _ in range(reps): f()
    torch.cuda.synchronize(); return (time.time()-t)/reps
# coarse bank, 20 deg
va=ViewingAngles.from_viewing_distance(np.radians(20.0)); V=va.azimus.shape[0]
caz=va.azimus.to(dev).to(vol.dtype); cpo=va.polars.to(dev).to(vol.dtype)
Tb=slice_nufft(vol,slice_points(fourier_circles(th,cpo,caz,torch.zeros_like(caz)),rr)).reshape(V,S,P)
Tf=torch.fft.fft(Tb,dim=-1).conj(); Tpow=(Tb.abs()**2).sum(-1); cr=p.ctf[:,:,0]
def coarse():
    Tn=torch.sqrt(torch.einsum('vr,br->bv',Tpow,wr[None,:]*cr**2)); In=torch.sqrt((p.images_fourier.abs()**2*w).sum(dim=(1,2)))
    If=torch.fft.fft(p.images_fourier,dim=-1)*(wr[None,:]*cr)[:,:,None]
    cc=torch.fft.ifft(torch.einsum('vrk,brk->bvk',Tf,If),dim=-1).real
    return (cc.amax(-1)/(Tn*In[:,None])).argmax(1)
bi=coarse(); bg=torch.zeros(N,device=dev,dtype=vol.dtype)
def refine(): return refine_poses(vol,p.images_fourier,p.ctf,th,rr,w,caz[bi].clone(),cpo[bi].clone(),bg.clone(),steps=100,lr=0.02)
def insert_mix():
    num=torch.zeros(D**3,dtype=torch.complex128,device=dev); den=torch.zeros(D**3,dtype=torch.float64,device=dev)
    for m in range(10):
        pts=slice_points(fourier_circles(th,cpo[bi],caz[bi],bg),rr).reshape(N,-1,3)
        x=(pts.reshape(-1,3)+D//2); f=torch.floor(x).long().clamp(0,D-1); fl=(f[:,0]*D+f[:,1])*D+f[:,2]
        v=(p.images_fourier.reshape(N,-1)*p.ctf.reshape(N,-1)*w.reshape(1,-1)).reshape(-1)
        num.index_put_((fl,),v,accumulate=True); den.index_put_((fl,),(p.ctf.reshape(N,-1)**2*w.reshape(1,-1)).reshape(-1),accumulate=True)
    return num,den
num,den=insert_mix()
def wiener(): return (num/(den+1e-3*den.mean())).reshape(D,D,D)
rows=[("coarse search (208 views)",T(coarse)),("refinement (100 steps)",T(refine,1)),
      ("mixture insert (10 poses)",T(insert_mix)),("wiener divide (per iter)",T(wiener,10))]
tot=sum(t for _,t in rows[:3])
print(f"D={D}, N={N} particles, physical noise SNR 0.3\n")
print(f"{'stage':>28}{'total':>9}{'per particle':>15}{'share':>8}")
for n,t in rows:
    print(f"{n:>28}{t*1e3:>8.0f}ms{t/N*1e6:>13.0f} us{(t/tot*100 if n!='wiener divide (per iter)' else 0):>7.0f}%")
print(f"\nextrapolated per iteration at 1e5 particles: {tot/N*1e5/60:.1f} min  (+ wiener, negligible)")
