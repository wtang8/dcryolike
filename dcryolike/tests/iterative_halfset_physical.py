"""Iterative refinement with a half-set split, CROSS-referenced -- PHYSICAL NOISE.

Port of iterative_halfset.py to generate_physical(). The one substantive change
is the posterior temperature. The old beta = ||y||^2_w / (2 sigma^2) assumed iid
per-point polar noise with sigma a per-point std. Under physical noise the
correct whitened log-likelihood is the REAL-SPACE one, -||y-cT||^2 / (2 sigma_r^2),
which by Parseval equals -||Y-cT||^2_w / (2 sigma_r^2 C) in the weighted polar
metric, for a constant C fixed by the grid and DFT normalisation. C is MEASURED
here from pure noise (C = E||n||^2_w / (D^2 sigma_r^2)), not assumed, so

    beta_i = ||y_i||^2_w / (2 sigma_r,i^2 C)

is the principled temperature and reduces to the profiled-scale Gaussian
likelihood in real space.

The previous loop let a particle's own contribution come back as evidence for
itself: the mixture smears one particle over 10 poses, all 10 land in the
reference, and next round the posterior is more confident about a reference it
partly wrote. Here half A is posed against the reference built from half B and
vice versa, so no particle can ever influence the map it is scored against.

Reported per iteration:
  vs truth   - FSC of each half-map against the ground truth (the honest number)
  half-map   - FSC between the two half-maps (the gold-standard ESTIMATE)
If half-map >> vs truth, the estimate is inflated and the refinement is
overfitting despite the split.
"""
import sys, os, numpy as np, torch
import _paths  # noqa: F401  (sets sys.path relative to this file)
from cryolike.grids import PolarGrid
from cryolike.metadata import ViewingAngles
from dcryolike.generator import gaussian_atom_volume, generate_physical
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
p=generate_physical(vol,pg,NP,SNR,seed=17,device=dev,ctf_kwargs=ISO)
In=torch.sqrt((p.images_fourier.abs()**2*w).sum(dim=(1,2)))
# ---- posterior temperature from MEASURED noise statistics ----
_n=p.images_fourier-p.clean_fourier                    # pure noise, per particle
_nw=(_n.abs()**2*w).sum(dim=(1,2))                     # weighted polar noise norm
C=(_nw/(D*D*p.sigma_noise**2)).mean()                  # Parseval constant, grid+DFT dependent
beta=In**2/(2*p.sigma_noise**2*C)
_beta_old=In**2/(2*(_nw/w.sum()))                      # what the OLD formula would give here
print(f"posterior temperature: C={C.item():.4f}  beta median {beta.median():.1f}  "
      f"(old iid formula would give {_beta_old.median():.1f}, ratio {(beta/_beta_old).median():.2f})")
cr=p.ctf[:,:,0]
HALF=[torch.arange(0,NP//2,device=dev), torch.arange(NP//2,NP,device=dev)]

ii=torch.arange(D,dtype=torch.float64,device=dev)-D//2
X,Y,Z=torch.meshgrid(ii,ii,ii,indexing="ij"); RR=(X**2+Y**2+Z**2).sqrt()
MASKS=[(RR>=k-0.5)&(RR<k+0.5) for k in range(1,D//2)]
def fsc(F1,F2):
    return np.array([((F1[m]*F2[m].conj()).sum()/
        (F1[m].abs().pow(2).sum().sqrt()*F2[m].abs().pow(2).sum().sqrt()+1e-30)).real.item()
        for m in MASKS])
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
                fl=(i0*D+i1)*D+i2
                num.index_put_((fl,),v*g.to(v.dtype),accumulate=True)
                den.index_put_((fl,),c*g,accumulate=True)

def reconstruct(ids,sel_idx,sel_w,gam,lam=1e-3,B=250):
    num=torch.zeros(D**3,dtype=torch.complex128,device=dev)
    den=torch.zeros(D**3,dtype=torch.float64,device=dev)
    n=len(ids)
    for i in range(0,n,B):
        j=min(i+B,n); loc=torch.arange(i,j,device=dev); gid=ids[loc]
        for m in range(sel_idx.shape[1]):
            v=sel_idx[loc,m]; ww=sel_w[loc,m]; keep=ww>1e-4
            if keep.sum()==0: continue
            g_=gid[keep]; l_=loc[keep]
            pts=slice_points(fourier_circles(th,cpo[v[keep]],caz[v[keep]],
                                             gam[l_,v[keep]]),rr).reshape(len(g_),-1,3)
            c=p.ctf[g_].reshape(len(g_),-1); wg=w.reshape(1,-1)*ww[keep][:,None]
            insert(pts.reshape(-1,3),
                   (p.images_fourier[g_].reshape(len(g_),-1)*c*wg).reshape(-1),
                   (c**2*wg).reshape(-1),num,den)
    return (num/(den+lam*den.mean())).reshape(D,D,D)

def pose_search(dens_ref,ids):
    n=len(ids)
    Tb=slice_nufft(dens_ref,slice_points(fourier_circles(th,cpo,caz,torch.zeros_like(caz)),rr)).reshape(V,S,P)
    Tf=torch.fft.fft(Tb,dim=-1).conj(); Tpow=(Tb.abs()**2).sum(-1)
    crs=cr[ids]; Ins=In[ids]
    Tn=torch.sqrt(torch.einsum('vr,br->bv',Tpow,wr[None,:]*crs**2))
    bc=torch.empty(n,V,device=dev,dtype=vol.dtype); bg=torch.empty(n,V,device=dev,dtype=vol.dtype)
    for i in range(0,n,16):
        j=min(i+16,n); g_=ids[i:j]
        If=torch.fft.fft(p.images_fourier[g_],dim=-1); cw=(wr[None,:]*crs[i:j])[:,:,None]
        cc=torch.fft.ifft(torch.einsum('vrk,brk->bvk',Tf,If*cw),dim=-1).real
        cc=cc/(Tn[i:j,:,None]*Ins[i:j,None,None])
        m,gi=cc.max(-1); bc[i:j]=m; bg[i:j]=-(gi.to(vol.dtype))*(2*np.pi/P)
    nbmax=torch.stack([bc[:,NB[v]].max(1).values for v in range(V)],1)
    masked=bc.masked_fill(~(bc>=nbmax-1e-12),-2.0)
    tcc,tidx=masked.topk(MODES,dim=1)
    lg=(beta[ids][:,None]*(tcc**2-tcc[:,:1]**2)).masked_fill(tcc<-1.0,-1e30)
    return tidx,torch.softmax(lg,dim=1),bg

F0=Ftrue.clone(); F0[RR>R_INIT]=0; dens0=to_real(F0)
print(f"PHYSICAL noise, real-space SNR {SNR} | {NP} particles, halves of {NP//2} | cross-referenced")
print(f"start = truth low-passed at r={R_INIT} (FSC0.5 {res_at(fsc(fourier_volume(dens0),Ftrue))} px)\n")
print(f"{'iter':>5}  {'--------- argmax ---------':^28}  {'--------- mixture --------':^28}")
print(f"{'':5}  {'vsTruth':>9}{'halfmap':>9}{'meanFSC':>10}  {'vsTruth':>9}{'halfmap':>9}{'meanFSC':>10}")
print("-"*70)
ref={m:[dens0.clone(),dens0.clone()] for m in ("argmax","mixture")}
for it in range(1,ITERS+1):
    line=f"{it:>5}  "
    for mode in ("argmax","mixture"):
        newF=[None,None]
        for h in (0,1):
            ids=HALF[h]
            tidx,wt,bg=pose_search(ref[mode][1-h],ids)        # CROSS reference
            si,sw=(tidx[:,:1],torch.ones(len(ids),1,device=dev,dtype=vol.dtype)) \
                  if mode=="argmax" else (tidx,wt)
            newF[h]=reconstruct(ids,si,sw,bg)
        ref[mode]=[to_real(newF[0]),to_real(newF[1])]
        fa,fb=fsc(newF[0],Ftrue),fsc(newF[1],Ftrue); fh=fsc(newF[0],newF[1])
        line+=f"{(res_at(fa)+res_at(fb))/2:>8.1f}p{res_at(fh):>8}p{(fa.mean()+fb.mean())/2:>10.4f}  "
    print(line)
