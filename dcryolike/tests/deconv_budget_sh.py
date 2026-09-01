"""DECONVOLUTION BUDGET, done in the spherical-harmonic domain.

Rotational blur is a convolution on SO(3): it attenuates angular band l within
each radial shell by ~exp(-l(l+1)sigma^2/2). A per-SHELL scalar cannot undo that
(and cannot change FSC at all, since FSC is scale-invariant). The filter must be
per (shell, l).

We measure the optimal per-(k,l) linear filter using the perfect-pose
reconstruction as the target -- knowledge unavailable in practice, which makes
the answer an UPPER BOUND on what any deconvolution could recover.
"""
import sys, numpy as np, torch
from scipy.special import sph_harm_y
import _paths  # noqa: F401  (sets sys.path relative to this file)
from cryolike.grids import PolarGrid
from dcryolike.generator import gaussian_atom_volume, uniform_so3, make_ctf
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft, fourier_volume, slice_gridding

dev="cuda"; D=48; NP=8000; BATCH=500; ISO=dict(astig=0.0); LMAX=24; M=1200
vol=gaussian_atom_volume(D,device=dev)
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=128,uniform=True)
S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=vol.dtype,device=dev)
rr=torch.tensor(pg.radius_shells,dtype=vol.dtype,device=dev)
w=torch.tensor(pg.integration_weight_points,dtype=vol.dtype,device=dev).reshape(S,P)

def insert(pts,vals,c2,num,den):
    x=pts+D//2; ok=((x>=0)&(x<=D-1)).all(-1)
    x=x[ok]; v=vals[ok]; c=c2[ok]; f=torch.floor(x).long(); fr=x-f
    for dx in (0,1):
        for dy in (0,1):
            for dz in (0,1):
                wg=((1-fr[:,0]) if dx==0 else fr[:,0])*((1-fr[:,1]) if dy==0 else fr[:,1])*\
                   ((1-fr[:,2]) if dz==0 else fr[:,2])
                i0=(f[:,0]+dx).clamp(0,D-1);i1=(f[:,1]+dy).clamp(0,D-1);i2=(f[:,2]+dz).clamp(0,D-1)
                fl=(i0*D+i1)*D+i2
                num.index_put_((fl,),v*wg.to(v.dtype),accumulate=True)
                den.index_put_((fl,),c*wg,accumulate=True)

def reconstruct(az,po,ga,imgs,ctf,lam=1e-3):
    num=torch.zeros(D**3,dtype=torch.complex128,device=dev)
    den=torch.zeros(D**3,dtype=torch.float64,device=dev)
    for i in range(0,az.shape[0],BATCH):
        j=min(i+BATCH,az.shape[0])
        pts=slice_points(fourier_circles(th,po[i:j],az[i:j],ga[i:j]),rr).reshape(j-i,-1,3)
        c=ctf[i:j].reshape(j-i,-1); v=(imgs[i:j].reshape(j-i,-1)*c*w.reshape(1,-1)).reshape(-1)
        insert(pts.reshape(-1,3),v,(c**2*w.reshape(1,-1)).reshape(-1),num,den)
    return (num/(den+lam*den.mean())).reshape(D,D,D)

# ---- spherical sampling grid + real SH basis (shared across shells) ----
i=np.arange(M)+0.5
phi=np.arccos(1-2*i/M); tht=np.pi*(1+5**0.5)*i
dirs=np.stack([np.sin(phi)*np.cos(tht),np.sin(phi)*np.sin(tht),np.cos(phi)],-1)
cols=[];lidx=[]
for l in range(LMAX+1):
    for m in range(-l,l+1):
        Y=sph_harm_y(l,abs(m),phi,tht)
        cols.append(np.real(Y) if m>=0 else np.imag(Y)); lidx.append(l)
Ymat=np.stack(cols,-1); Ypinv=np.linalg.pinv(Ymat)
Ypinv_t=torch.tensor(Ypinv,dtype=torch.complex128,device=dev)
Ymat_t=torch.tensor(Ymat,dtype=torch.complex128,device=dev)
lidx_t=torch.tensor(lidx,device=dev)
dirs_t=torch.tensor(dirs,dtype=torch.float64,device=dev)
KS=list(range(2,D//2))
pts_all=torch.cat([k*dirs_t for k in KS],0)[None]

def shell_coeffs(F):
    v=slice_gridding(F,pts_all)[0].reshape(len(KS),M)
    return (v.to(torch.complex128)@Ypinv_t.T)          # (nshell, ncoef)

def fsc_from(a,b):
    num=(a*b.conj()).sum(-1).real
    return num/(a.abs().pow(2).sum(-1).sqrt()*b.abs().pow(2).sum(-1).sqrt()+1e-30)

def res_at(f,thr=0.5):
    for k,v in zip(KS,f):
        if v<thr: return k
    return KS[-1]

g=torch.Generator(device=dev).manual_seed(5)
az,po,ga=uniform_so3(NP,seed=42,device=dev,dtype=vol.dtype)
ctf=make_ctf(pg,NP,float(D),seed=42,device=dev,dtype=vol.dtype,**ISO)
T=slice_nufft(vol,slice_points(fourier_circles(th,po,az,ga),rr)).reshape(NP,S,P)
Ctrue=shell_coeffs(fourier_volume(vol))

for snr in (1.0,0.1):
    clean=T*ctf; pw=(clean.abs()**2*w).sum(dim=(1,2))/w.sum(); sg=torch.sqrt(pw/snr)
    imgs=clean+torch.randn(clean.shape,generator=g,dtype=clean.dtype,device=dev)*sg[:,None,None]
    Cref=shell_coeffs(reconstruct(az,po,ga,imgs,ctf))
    print(f"\n=== SNR {snr} | {NP} particles | perfect-pose FSC0.5 = "
          f"{res_at(fsc_from(Cref,Ctrue))} px ===")
    print(f"{'pose err':>9}{'blurred':>11}{'l-deconv':>11}{'gain':>8}   attenuation by l (sigma=blur)")
    for sd in (2.0,5.0,10.0):
        s=np.radians(sd)
        Fb=reconstruct(az+torch.randn(NP,generator=g,device=dev,dtype=vol.dtype)*s,
                       po+torch.randn(NP,generator=g,device=dev,dtype=vol.dtype)*s,
                       ga+torch.randn(NP,generator=g,device=dev,dtype=vol.dtype)*s, imgs, ctf)
        Cb=shell_coeffs(Fb)
        f_b=fsc_from(Cb,Ctrue)
        Cd=Cb.clone(); att=[]
        for l in range(LMAX+1):
            m=(lidx_t==l)
            a=Cb[:,m]; b=Cref[:,m]
            t=((b*a.conj()).sum(-1)/(a.abs().pow(2).sum(-1)+1e-30))
            Cd[:,m]=a*t[:,None]
            att.append(t.abs().median().item())
        f_d=fsc_from(Cd,Ctrue)
        rb,rd=res_at(f_b),res_at(f_d)
        a_s="".join(f"{att[l]:5.2f}" for l in (0,4,8,12,16,20,24))
        print(f"{sd:>7.0f}d{rb:>9} px{rd:>9} px{rd-rb:>+7} {a_s}")
    print(f"{'':38}  l=  0    4    8   12   16   20   24")
