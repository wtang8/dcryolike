"""Iterative refinement: does posterior-weighted insertion resist model bias?

Now the reference is rebuilt from the SAME particles it is used to pose --
the real pipeline, and exactly where model bias / Einstein-from-noise arises.
Both branches start from the same deliberately poor low-pass reference and run
the same loop; only the insertion rule differs.

Scored against GROUND TRUTH every iteration. Half-map FSC is useless here: both
halves would share the same reference and inflate together.
"""
import sys, os, time, numpy as np, torch
sys.path.insert(0,"/home/wtang/cryoem/dcryolike"); sys.path.insert(0,"/home/wtang/cryoem/cryolike/src")
from cryolike.grids import PolarGrid
from cryolike.metadata import ViewingAngles
from dcryolike.generator import gaussian_atom_volume, generate
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft, fourier_volume

dev="cuda"; D=48; NP=8000; MODES=10; NEIGH=10.0; ITERS=6; R_INIT=8
SNR=float(os.environ.get("SNR","0.05")); ISO=dict(astig=0.0)
vol=gaussian_atom_volume(D,device=dev); Ftrue=fourier_volume(vol)
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=128,uniform=True)
S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=vol.dtype,device=dev)
rr=torch.tensor(pg.radius_shells,dtype=vol.dtype,device=dev)
w=torch.tensor(pg.integration_weight_points,dtype=vol.dtype,device=dev).reshape(S,P); wr=w[:,0]
va=ViewingAngles.from_viewing_distance(np.radians(4.0))
caz=va.azimus.to(dev).to(vol.dtype); cpo=va.polars.to(dev).to(vol.dtype); V=caz.shape[0]
nd=torch.stack([torch.sin(cpo)*torch.cos(caz),torch.sin(cpo)*torch.sin(caz),torch.cos(cpo)],-1)
NB=(nd@nd.T)>=np.cos(np.radians(NEIGH))
p=generate(vol,pg,NP,SNR,seed=17,device=dev,ctf_kwargs=ISO)
In=torch.sqrt((p.images_fourier.abs()**2*w).sum(dim=(1,2)))
beta=(In**2/(2*p.sigma_noise**2)); cr=p.ctf[:,:,0]

ii=torch.arange(D,dtype=torch.float64,device=dev)-D//2
X,Y,Z=torch.meshgrid(ii,ii,ii,indexing="ij"); RR=(X**2+Y**2+Z**2).sqrt()
def fsc(F1,F2):
    return np.array([((F1[m]*F2[m].conj()).sum()/
        (F1[m].abs().pow(2).sum().sqrt()*F2[m].abs().pow(2).sum().sqrt()+1e-30)).real.item()
        for m in [(RR>=k-0.5)&(RR<k+0.5) for k in range(1,D//2)]])
def res_at(f,t=0.5):
    for k,v in enumerate(f,start=1):
        if v<t: return k
    return len(f)
def to_real(F):
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(F),norm="ortho")).real.contiguous()

def insert(pts,vals,c2,num,den):
    x=pts+D//2; ok=((x>=0)&(x<=D-1)).all(-1)
    x=x[ok]; v=vals[ok]; c=c2[ok]; f=torch.floor(x).long(); fr=x-f
    for dx in (0,1):
        for dy in (0,1):
            for dz in (0,1):
                g=((1-fr[:,0]) if dx==0 else fr[:,0])*((1-fr[:,1]) if dy==0 else fr[:,1])*\
                  ((1-fr[:,2]) if dz==0 else fr[:,2])
                i0=(f[:,0]+dx).clamp(0,D-1);i1=(f[:,1]+dy).clamp(0,D-1);i2=(f[:,2]+dz).clamp(0,D-1)
                num.index_put_((((i0*D+i1)*D+i2),),v*g.to(v.dtype),accumulate=True)
                den.index_put_((((i0*D+i1)*D+i2),),c*g,accumulate=True)

def reconstruct(sel_idx,sel_w,gam,lam=1e-3,B=250):
    num=torch.zeros(D**3,dtype=torch.complex128,device=dev)
    den=torch.zeros(D**3,dtype=torch.float64,device=dev)
    for i in range(0,NP,B):
        j=min(i+B,NP); ids=torch.arange(i,j,device=dev)
        for m in range(sel_idx.shape[1]):
            v=sel_idx[i:j,m]; ww=sel_w[i:j,m]; keep=ww>1e-4
            if keep.sum()==0: continue
            idx=ids[keep]
            pts=slice_points(fourier_circles(th,cpo[v[keep]],caz[v[keep]],
                                             gam[idx,v[keep]]),rr).reshape(len(idx),-1,3)
            c=p.ctf[idx].reshape(len(idx),-1); wgt=w.reshape(1,-1)*ww[keep][:,None]
            insert(pts.reshape(-1,3),
                   (p.images_fourier[idx].reshape(len(idx),-1)*c*wgt).reshape(-1),
                   (c**2*wgt).reshape(-1),num,den)
    return (num/(den+lam*den.mean())).reshape(D,D,D)

def pose_search(dens_ref):
    Tb=slice_nufft(dens_ref,slice_points(fourier_circles(th,cpo,caz,torch.zeros_like(caz)),rr)).reshape(V,S,P)
    Tpow=(Tb.abs()**2).sum(-1); Tf=torch.fft.fft(Tb,dim=-1).conj()
    Tn=torch.sqrt(torch.einsum('vr,br->bv',Tpow,wr[None,:]*cr**2))
    bc=torch.empty(NP,V,device=dev,dtype=vol.dtype); bg=torch.empty(NP,V,device=dev,dtype=vol.dtype)
    for i in range(0,NP,16):
        j=min(i+16,NP)
        If=torch.fft.fft(p.images_fourier[i:j],dim=-1); cw=(wr[None,:]*cr[i:j])[:,:,None]
        cc=torch.fft.ifft(torch.einsum('vrk,brk->bvk',Tf,If*cw),dim=-1).real
        cc=cc/(Tn[i:j,:,None]*In[i:j,None,None])
        m,gi=cc.max(-1); bc[i:j]=m; bg[i:j]=-(gi.to(vol.dtype))*(2*np.pi/P)
    nbmax=torch.stack([bc[:,NB[v]].max(1).values for v in range(V)],1)
    masked=bc.masked_fill(~(bc>=nbmax-1e-12),-2.0)
    tcc,tidx=masked.topk(MODES,dim=1)
    lg=(beta[:,None]*(tcc**2-tcc[:,:1]**2)).masked_fill(tcc<-1.0,-1e30)
    return tidx,torch.softmax(lg,dim=1),bg

# ---- shared starting reference: deliberately poor low-pass ----
F0=Ftrue.clone(); F0[RR>R_INIT]=0
dens0=to_real(F0)
print(f"SNR {SNR} | {NP} particles | start = truth low-passed at r={R_INIT} "
      f"(FSC0.5 {res_at(fsc(fourier_volume(dens0),Ftrue))} px)\n")
print(f"{'iter':>5}{'argmax FSC0.5':>16}{'mixture FSC0.5':>17}"
      f"{'argmax meanFSC':>17}{'mixture meanFSC':>18}{'top-mode wt':>13}")
print("-"*88)
ref={"argmax":dens0.clone(),"mixture":dens0.clone()}
for it in range(1,ITERS+1):
    out={}
    for mode in ("argmax","mixture"):
        tidx,wt,bg=pose_search(ref[mode])
        if mode=="argmax":
            sel_i,sel_w=tidx[:,:1],torch.ones(NP,1,device=dev,dtype=vol.dtype); tw=wt[:,0].mean()
        else:
            sel_i,sel_w=tidx,wt
        F=reconstruct(sel_i,sel_w,bg)
        ref[mode]=to_real(F); out[mode]=fsc(F,Ftrue)
    print(f"{it:>5}{res_at(out['argmax']):>13} px{res_at(out['mixture']):>14} px"
          f"{out['argmax'].mean():>17.4f}{out['mixture'].mean():>18.4f}{tw:>13.3f}")
