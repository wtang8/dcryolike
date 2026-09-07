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

dev="cuda"; D=int(os.environ.get("D","48")); NP=int(os.environ.get("NP","8000"))
MODES=int(os.environ.get("MODES","10")); NEIGH=10.0*min(1.0,24.0/(D//2)); ITERS=int(os.environ.get("ITERS","6")); R_INIT=8
# Single precision by default (see the note in hetero_linear_pilot.py); DTYPE=double
# restores the original path. At D=96 the complex128 stack alone is 3.2 GB.
DT=torch.float64 if os.environ.get("DTYPE","single")=="double" else torch.float32
SNR=float(os.environ.get("SNR","0.05")); ISO=dict(astig=0.0)
# LOWPASS=1: before a half-map becomes the next pose-search bank, low-pass it at
# the HALF-MAP FSC 0.5 crossing (soft cosine edge, 2 shells). Half-map FSC is
# what a real pipeline has; truth FSC is used only for SCORING, never for the
# cutoff, so no ground truth leaks into the loop. Scoring is on the raw
# reconstruction -- the filter only limits what the next iteration may chase.
LOWPASS=os.environ.get("LOWPASS","0")=="1"
# GOLD=1: TRUE gold standard. Each half is posed against its OWN previous
# reference, so the two halves are fully independent -- no cross-coupling.
# Default (GOLD=0) is cross-referenced: A posed against B's map and vice versa,
# which breaks self-reinforcement but lets the halves converge on shared error.
GOLD=os.environ.get("GOLD","0")=="1"
vol=gaussian_atom_volume(D,device=dev,dtype=DT); Ftrue=fourier_volume(vol)
P_IN=int(os.environ.get("P_INPLANE",str(min(256,int(2**np.ceil(np.log2(2*np.pi*(D//2)*0.85)))))))
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=P_IN,uniform=True)
S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=vol.dtype,device=dev)
rr=torch.tensor(pg.radius_shells,dtype=vol.dtype,device=dev)
w=torch.tensor(pg.integration_weight_points,dtype=vol.dtype,device=dev).reshape(S,P); wr=w[:,0]
VIEW_DEG=float(os.environ.get("VIEW_DEG", "4.0" if D<=48 else str(np.degrees(1.0/(D//2)))))
va=ViewingAngles.from_viewing_distance(np.radians(VIEW_DEG))
caz=va.azimus.to(dev).to(vol.dtype); cpo=va.polars.to(dev).to(vol.dtype); V=caz.shape[0]
# neighbour INDEX, not a V x V matrix: 30,522^2 booleans is 931 MB at D=96
from dcryolike.pipeline import neighbourhood as _nbh
NB,nd=_nbh(caz,cpo,NEIGH)
p=generate_physical(vol,pg,NP,SNR,seed=17,device=dev,ctf_kwargs=ISO,keep_clean=False)
_dummy=None
In=torch.sqrt((p.images_fourier.abs()**2*w).sum(dim=(1,2)))
# ---- posterior temperature from MEASURED noise statistics ----
_nw=p.noise_norm_w.to(w.dtype)                         # weighted polar noise norm,
# accumulated during generation (the noiseless stack is 0.8 GB at D=96 and was kept
# only for this one number per particle)
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
def lowpass(F,kc,width=2.0):
    t=((RR-kc)/width).clamp(0,1)
    return F*(0.5*(1+torch.cos(np.pi*t)))
def to_real(F):
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(F),norm="ortho")).real.contiguous()

def insert(pts,vals,c2,num,den):
    pts=pts.to(torch.float64); x=pts+D//2; ok=((x>=0)&(x<=D-1)).all(-1)
    x=x[ok]; v=vals[ok]; c=c2[ok]; f=torch.floor(x).long(); fr=x-f
    for dx in (0,1):
        for dy in (0,1):
            for dz in (0,1):
                g=((1-fr[:,0]) if dx==0 else fr[:,0])*((1-fr[:,1]) if dy==0 else fr[:,1])*\
                  ((1-fr[:,2]) if dz==0 else fr[:,2])
                i0=(f[:,0]+dx).clamp(0,D-1);i1=(f[:,1]+dy).clamp(0,D-1);i2=(f[:,2]+dz).clamp(0,D-1)
                fl=(i0*D+i1)*D+i2
                num.index_put_((fl,),(v*g.to(v.dtype)).to(num.dtype),accumulate=True)
                den.index_put_((fl,),(c*g).to(den.dtype),accumulate=True)

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

def _mem(tag):
    if torch.cuda.is_available() and os.environ.get("MEMLOG","1")=="1":
        a=torch.cuda.memory_allocated()/1e9; m=torch.cuda.max_memory_allocated()/1e9
        f=torch.cuda.mem_get_info()[0]/1e9
        print(f"      [mem] {tag:22s} allocated {a:5.2f} GB  peak {m:5.2f} GB  device free {f:5.2f} GB",flush=True)

def pose_search(dens_ref,ids):
    n=len(ids)
    # chunked bank: 30,522 views at D=96 is 3.1 GB in complex64 and cannot be built
    # in one slice call
    CDTB=torch.complex64 if DT==torch.float32 else torch.complex128
    Tf=torch.empty(V,S,P,dtype=CDTB,device=dev); Tpow=torch.empty(V,S,dtype=DT,device=dev)
    for i in range(0,V,2048):
        j=min(i+2048,V)
        Tb=slice_nufft(dens_ref,slice_points(fourier_circles(th,cpo[i:j],caz[i:j],torch.zeros_like(caz[i:j])),rr)).reshape(j-i,S,P).to(CDTB)
        Tf[i:j]=torch.fft.fft(Tb,dim=-1).conj(); Tpow[i:j]=(Tb.abs()**2).sum(-1).to(DT)
        del Tb
    crs=cr[ids]; Ins=In[ids]
    Tn=torch.sqrt(torch.einsum('vr,br->bv',Tpow,(wr[None,:]*crs**2).to(Tpow.dtype)))
    bc=torch.empty(n,V,device=dev,dtype=vol.dtype); bg=torch.empty(n,V,device=dev,dtype=vol.dtype)
    # the (b,V,P) correlation block is 1 GB per 16 images at D=96; keep it near 0.5 GB
    CB=max(1,int(5e8//(V*P*8)))
    for i in range(0,n,CB):
        j=min(i+CB,n); g_=ids[i:j]
        If=torch.fft.fft(p.images_fourier[g_].to(Tf.dtype),dim=-1); cw=(wr[None,:]*crs[i:j])[:,:,None].to(Tf.dtype)
        cc=torch.fft.ifft(torch.einsum('vrk,brk->bvk',Tf,If*cw),dim=-1).real
        cc=cc/(Tn[i:j,:,None]*Ins[i:j,None,None])
        m,gi=cc.max(-1); bc[i:j]=m.to(bc.dtype); bg[i:j]=-(gi.to(bg.dtype))*(2*np.pi/P)
        del cc,If,cw
    _mem("after search loop")
    del Tf,Tpow,Tn; torch.cuda.empty_cache()   # 3.1 GB, not needed past the search
    # the neighbourhood gather is (n, chunk, width) and width is ~55 views at D=96,
    # so the chunk has to be small or this is the peak of the whole run
    nbmax=torch.empty_like(bc)
    GC=max(64,int(2e8//(n*max(1,NB.shape[1])*4)))
    for i in range(0,V,GC):
        j=min(i+GC,V); nbmax[:,i:j]=bc[:,NB[i:j]].max(-1).values
    bc.masked_fill_(~(bc>=nbmax-1e-12),-2.0)   # in place: bc is not needed afterwards
    del nbmax
    _mem("after neighbourhood")
    tcc,tidx=bc.topk(MODES,dim=1)
    lg=(beta[ids][:,None]*(tcc**2-tcc[:,:1]**2)).masked_fill(tcc<-1.0,-1e30)
    return tidx,torch.softmax(lg,dim=1),bg

F0=Ftrue.clone(); F0[RR>R_INIT]=0; dens0=to_real(F0)
print(f"PHYSICAL noise, real-space SNR {SNR} | {NP} particles, halves of {NP//2}"
      f"{' | LOW-PASS CONTROL ON (half-map FSC 0.5 cutoff)' if LOWPASS else ''}"
      f"{' | GOLD STANDARD (independent halves, own reference)' if GOLD else ' | cross-referenced'}")
print(f"start = truth low-passed at r={R_INIT} (FSC0.5 {res_at(fsc(fourier_volume(dens0),Ftrue))} px)\n")
print(f"{'iter':>5}  {'--------- argmax ---------':^28}  {'--------- mixture --------':^28}")
print(f"{'':5}  {'vsTruth':>9}{'halfmap':>9}{'meanFSC':>10}  {'vsTruth':>9}{'halfmap':>9}{'meanFSC':>10}")
print("-"*70)
ref={m:[dens0.clone(),dens0.clone()] for m in ("argmax","mixture")}; cut={}
for it in range(1,ITERS+1):
    line=f"{it:>5}  "
    for mode in ("argmax","mixture"):
        newF=[None,None]
        for h in (0,1):
            ids=HALF[h]
            tidx,wt,bg=pose_search(ref[mode][h if GOLD else 1-h],ids)   # own ref if GOLD, else cross
            si,sw=(tidx[:,:1],torch.ones(len(ids),1,device=dev,dtype=vol.dtype)) \
                  if mode=="argmax" else (tidx,wt)
            newF[h]=reconstruct(ids,si,sw,bg)
        fa,fb=fsc(newF[0],Ftrue),fsc(newF[1],Ftrue); fh=fsc(newF[0],newF[1])
        if LOWPASS:
            kc=res_at(fh,0.5)                       # from HALF-MAP FSC only
            ref[mode]=[to_real(lowpass(newF[0],kc)),to_real(lowpass(newF[1],kc))]
            cut[mode]=kc
        else:
            ref[mode]=[to_real(newF[0]),to_real(newF[1])]
        line+=f"{(res_at(fa)+res_at(fb))/2:>8.1f}p{res_at(fh):>8}p{(fa.mean()+fb.mean())/2:>10.4f}  "
    if LOWPASS: line+=f"  cut a/m {cut['argmax']}/{cut['mixture']}p"
    print(line)
