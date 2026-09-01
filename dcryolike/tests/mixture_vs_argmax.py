"""Does posterior-weighted insertion beat argmax insertion?

At SNR 0.05 the top pose beats its runner-up by a score ratio of ~0.997 -- the
likelihood cannot tell them apart. Committing to the argmax turns a coin flip
into a coherent contribution at the wrong orientation, i.e. BIAS. Spreading the
same weight across the tied basins turns it into NOISE, which averages away over
many particles while bias does not.

Four reconstructions, all from the SAME particles, scored against ground truth
(not half-map FSC, which cannot see systematic error shared by both halves):
  true      poses handed over free -- upper bound
  argmax    single best pose, full weight
  mixture   top-M basins, weights from the actual log-likelihood
  mix+sharp mixture, additionally down-weighting flat-posterior particles
"""
import sys, time, numpy as np, torch
sys.path.insert(0,"/home/wtang/cryoem/dcryolike"); sys.path.insert(0,"/home/wtang/cryoem/cryolike/src")
from cryolike.grids import PolarGrid
from cryolike.metadata import ViewingAngles
from dcryolike.generator import gaussian_atom_volume, generate
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft, fourier_volume

import os
dev="cuda"; D=48; NP=8000; MODES=10; NEIGH=10.0; ISO=dict(astig=0.0)
SNR=float(os.environ.get("SNR","0.05"))
vol=gaussian_atom_volume(D,device=dev); Ftrue=fourier_volume(vol)
# DECONTAMINATION: the pose-search reference must NOT be the ground truth.
# Searching against truth-derived templates preferentially selects noise that
# agrees with the truth, inflating exactly the reconstructions being tested --
# measured as a +17% mean-FSC advantage over using the true poses at SNR 0.05,
# growing as SNR falls. The reference here is instead built from a DISJOINT set
# of particles, so its noise is independent of the particles being posed.
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=128,uniform=True)
S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=vol.dtype,device=dev)
rr=torch.tensor(pg.radius_shells,dtype=vol.dtype,device=dev)
w=torch.tensor(pg.integration_weight_points,dtype=vol.dtype,device=dev).reshape(S,P); wr=w[:,0]
va=ViewingAngles.from_viewing_distance(np.radians(4.0))
caz=va.azimus.to(dev).to(vol.dtype); cpo=va.polars.to(dev).to(vol.dtype); V=caz.shape[0]
nd=torch.stack([torch.sin(cpo)*torch.cos(caz),torch.sin(cpo)*torch.sin(caz),torch.cos(cpo)],-1)
NB=(nd@nd.T)>=np.cos(np.radians(NEIGH))
print(f"{V} views | {NP} particles | SNR {SNR} | top-{MODES} modes")

p_ref=generate(vol,pg,NP,SNR,seed=101,device=dev,ctf_kwargs=ISO)   # disjoint
p    =generate(vol,pg,NP,SNR,seed=17, device=dev,ctf_kwargs=ISO)

# ---- score every particle against every view, keep best in-plane per view ----
best_cc=torch.empty(NP,V,device=dev,dtype=vol.dtype)
best_g =torch.empty(NP,V,device=dev,dtype=vol.dtype)
In=torch.sqrt((p.images_fourier.abs()**2*w).sum(dim=(1,2)))
beta=(In**2/(2*p.sigma_noise**2))

def score_all_particles():
  global wt, sharp, top_idx, top_cc
  t0=time.time()
  cr=p.ctf[:,:,0]
  Tn=torch.sqrt(torch.einsum('vr,br->bv',Tpow,wr[None,:]*cr**2))
  for i in range(0,NP,16):
    j=min(i+16,NP)
    If=torch.fft.fft(p.images_fourier[i:j],dim=-1)
    cw=(wr[None,:]*cr[i:j])[:,:,None]
    cc=torch.fft.ifft(torch.einsum('vrk,brk->bvk',Tf,If*cw),dim=-1).real
    cc=cc/(Tn[i:j,:,None]*In[i:j,None,None])
    m,gi=cc.max(-1)
    best_cc[i:j]=m; best_g[i:j]=-(gi.to(vol.dtype))*(2*np.pi/P)
  print(f"scored in {time.time()-t0:.0f}s")
  nbmax=torch.stack([best_cc[:,NB[v]].max(1).values for v in range(V)],1)
  ismax=best_cc>=nbmax-1e-12
  masked=best_cc.masked_fill(~ismax,-2.0)
  top_cc,top_idx=masked.topk(MODES,dim=1)
  logits=beta[:,None]*(top_cc**2-top_cc[:,:1]**2)
  logits=logits.masked_fill(top_cc<-1.0,-1e30)
  wt=torch.softmax(logits,dim=1); sharp=wt[:,0]
  print(f"posterior: mean top-mode weight {sharp.mean():.3f}, "
        f"median {sharp.median():.3f}, frac<0.5 {100*(sharp<0.5).float().mean():.0f}%")

# ---- reconstruction ----
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

def reconstruct(pose_iter,lam=1e-3):
    num=torch.zeros(D**3,dtype=torch.complex128,device=dev)
    den=torch.zeros(D**3,dtype=torch.float64,device=dev)
    for az,po,ga,idx,pw in pose_iter:
        pts=slice_points(fourier_circles(th,po,az,ga),rr).reshape(len(idx),-1,3)
        c=p.ctf[idx].reshape(len(idx),-1)
        ww=(w.reshape(1,-1)*pw[:,None])
        insert(pts.reshape(-1,3),
               (p.images_fourier[idx].reshape(len(idx),-1)*c*ww).reshape(-1),
               (c**2*ww).reshape(-1), num, den)
    return (num/(den+lam*den.mean())).reshape(D,D,D)

def batches(sel_idx, sel_w, B=250):
    """sel_idx: (NP,M) view indices; sel_w: (NP,M) weights."""
    M=sel_idx.shape[1]
    for i in range(0,NP,B):
        j=min(i+B,NP); ids=torch.arange(i,j,device=dev)
        for m in range(M):
            v=sel_idx[i:j,m]; ww=sel_w[i:j,m]
            keep=ww>1e-4
            if keep.sum()==0: continue
            yield (caz[v[keep]],cpo[v[keep]],best_g[ids[keep],v[keep]],
                   ids[keep],ww[keep])

def fsc(F1,F2):
    i=torch.arange(D,dtype=torch.float64,device=dev)-D//2
    X,Y,Z=torch.meshgrid(i,i,i,indexing="ij"); R=(X**2+Y**2+Z**2).sqrt()
    out=[]
    for k in range(1,D//2):
        m=(R>=k-0.5)&(R<k+0.5); a,b=F1[m],F2[m]
        out.append(((a*b.conj()).sum()/(a.abs().pow(2).sum().sqrt()*b.abs().pow(2).sum().sqrt()+1e-30)).real.item())
    return np.array(out)
def res_at(f,t=0.5):
    for k,v in enumerate(f,start=1):
        if v<t: return k
    return len(f)

# ---- reference volume from the DISJOINT particles, then the template bank ----
_saved=p
p=p_ref
F_ref=reconstruct([(p_ref.azimus,p_ref.polars,p_ref.gammas,
                    torch.arange(NP,device=dev),
                    torch.ones(NP,device=dev,dtype=vol.dtype))])
p=_saved
dens_ref=torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(F_ref),norm="ortho")).real.contiguous()
f_ref=fsc(fourier_volume(dens_ref),Ftrue)
print(f"reference volume (disjoint {NP} particles, true poses): FSC0.5 = {res_at(f_ref)} px")
Tb=slice_nufft(dens_ref,slice_points(fourier_circles(th,cpo,caz,torch.zeros_like(caz)),rr)).reshape(V,S,P)
Tpow=(Tb.abs()**2).sum(-1); Tf=torch.fft.fft(Tb,dim=-1).conj()
score_all_particles()

runs={}
runs["true"]=reconstruct([(p.azimus,p.polars,p.gammas,torch.arange(NP,device=dev),
                           torch.ones(NP,device=dev,dtype=vol.dtype))])
one=torch.ones(NP,1,device=dev,dtype=vol.dtype)
runs["argmax"]=reconstruct(batches(top_idx[:,:1],one))
runs["mixture"]=reconstruct(batches(top_idx,wt))
runs["mix+sharp"]=reconstruct(batches(top_idx,wt*sharp[:,None]))

print(f"\n{'reconstruction':>14}{'FSC0.5':>9}{'FSC@0.143':>11}{'mean FSC':>11}")
print("-"*46)
base=None
for k in ("true","argmax","mixture","mix+sharp"):
    f=fsc(runs[k],Ftrue); m=f.mean()
    if k=="argmax": base=m
    tag="" if base is None or k=="argmax" else f"   {100*(m-base)/abs(base):+.1f}% vs argmax"
    print(f"{k:>14}{res_at(f):>7} px{res_at(f,0.143):>9} px{m:>11.4f}{tag}")
