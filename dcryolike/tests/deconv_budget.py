"""DECONVOLUTION BUDGET.

Pose error blurs the reconstruction. Rotational smearing is a convolution on
SO(3), so in principle it is invertible. This measures how much is actually
recoverable, and at what noise cost.

Method: reconstruct the same particles three ways --
  (a) perfect poses            -> reference
  (b) poses perturbed by sigma -> blurred
  (c) (b) Wiener-deconvolved using the transfer function measured against (a)
and score each against the GROUND-TRUTH volume, not against half-map FSC.
Scoring against truth avoids the trap where applying the same deconvolution to
both half-maps correlates them and inflates FSC in exactly the shells at issue.

(c) uses knowledge of (a), which you would not have in practice. That is
deliberate: it makes the result an UPPER BOUND on what deconvolution can buy.
"""
import sys, numpy as np, torch
import _paths  # noqa: F401  (sets sys.path relative to this file)
from cryolike.grids import PolarGrid
from dcryolike.generator import gaussian_atom_volume, generate, uniform_so3, make_ctf
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft, fourier_volume

dev="cuda"; D=48; NP=12000; BATCH=500; ISO=dict(astig=0.0)
vol=gaussian_atom_volume(D,device=dev)
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=128,uniform=True)
S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=vol.dtype,device=dev)
rr=torch.tensor(pg.radius_shells,dtype=vol.dtype,device=dev)
w=torch.tensor(pg.integration_weight_points,dtype=vol.dtype,device=dev).reshape(S,P)
Ftrue=fourier_volume(vol)

def insert(pts, vals, ctf2, num, den):
    """Trilinear scatter of slice values into the 3D Fourier grid (RELION-style)."""
    x=(pts+D//2)
    ok=((x>=0)&(x<=D-1)).all(-1)
    x=x[ok]; v=vals[ok]; c=ctf2[ok]
    f=torch.floor(x).long(); fr=x-f
    for dx in (0,1):
        for dy in (0,1):
            for dz in (0,1):
                wgt=((1-fr[:,0]) if dx==0 else fr[:,0]) * \
                    ((1-fr[:,1]) if dy==0 else fr[:,1]) * \
                    ((1-fr[:,2]) if dz==0 else fr[:,2])
                i0=(f[:,0]+dx).clamp(0,D-1); i1=(f[:,1]+dy).clamp(0,D-1); i2=(f[:,2]+dz).clamp(0,D-1)
                flat=(i0*D+i1)*D+i2
                num.index_put_((flat,), v*wgt.to(v.dtype), accumulate=True)
                den.index_put_((flat,), c*wgt, accumulate=True)

def reconstruct(az,po,ga,imgs,ctf,lam=1e-3):
    num=torch.zeros(D**3,dtype=torch.complex128,device=dev)
    den=torch.zeros(D**3,dtype=torch.float64,device=dev)
    for i in range(0,az.shape[0],BATCH):
        j=min(i+BATCH,az.shape[0])
        pts=slice_points(fourier_circles(th,po[i:j],az[i:j],ga[i:j]),rr).reshape(j-i,-1,3)
        c=ctf[i:j].reshape(j-i,-1)
        v=(imgs[i:j].reshape(j-i,-1)*c*w.reshape(1,-1)).reshape(-1)
        insert(pts.reshape(-1,3), v, (c**2*w.reshape(1,-1)).reshape(-1),
               num, den)
    F=(num/(den+lam*den.mean())).reshape(D,D,D)
    return F

def fsc(F1,F2):
    i=torch.arange(D,dtype=torch.float64,device=dev)-D//2
    X,Y,Z=torch.meshgrid(i,i,i,indexing="ij"); R=(X**2+Y**2+Z**2).sqrt()
    out=[]
    for k in range(1,D//2):
        m=(R>=k-0.5)&(R<k+0.5)
        a,b=F1[m],F2[m]
        c=(a*b.conj()).sum()/ (a.abs().pow(2).sum().sqrt()*b.abs().pow(2).sum().sqrt()+1e-30)
        out.append(c.real.item())
    return np.array(out)

def res_at(f,thr=0.5):
    for k,v in enumerate(f,start=1):
        if v<thr: return k
    return len(f)

g=torch.Generator(device=dev).manual_seed(5)
az,po,ga=uniform_so3(NP,seed=42,device=dev,dtype=vol.dtype)
ctf=make_ctf(pg,NP,float(D),seed=42,device=dev,dtype=vol.dtype,**ISO)
T=slice_nufft(vol,slice_points(fourier_circles(th,po,az,ga),rr)).reshape(NP,S,P)

for snr in (1.0, 0.1):
    clean=T*ctf
    pw=(clean.abs()**2*w).sum(dim=(1,2))/w.sum()
    sig=torch.sqrt(pw/snr)
    imgs=clean+torch.randn(clean.shape,generator=g,dtype=clean.dtype,device=dev)*sig[:,None,None]
    Fref=reconstruct(az,po,ga,imgs,ctf)
    f_ref=fsc(Fref,Ftrue)
    print(f"\n=== SNR {snr} | {NP} particles ===")
    print(f"{'pose err':>10}{'FSC0.5 blurred':>17}{'FSC0.5 deconv':>16}{'recovered':>12}")
    print("-"*57)
    print(f"{'0 (perfect)':>10}{res_at(f_ref):>14} px{'-':>16}{'-':>12}")
    for sd in (2.0,5.0,10.0):
        s=np.radians(sd)
        a2=az+torch.randn(NP,generator=g,device=dev,dtype=vol.dtype)*s
        p2=po+torch.randn(NP,generator=g,device=dev,dtype=vol.dtype)*s
        g2=ga+torch.randn(NP,generator=g,device=dev,dtype=vol.dtype)*s
        Fb=reconstruct(a2,p2,g2,imgs,ctf)
        f_b=fsc(Fb,Ftrue)
        # measured shell transfer of the blur, then Wiener inverse (upper bound)
        i=torch.arange(D,dtype=torch.float64,device=dev)-D//2
        X,Y,Z=torch.meshgrid(i,i,i,indexing="ij"); R=(X**2+Y**2+Z**2).sqrt()
        Fd=Fb.clone()
        for k in range(1,D//2):
            m=(R>=k-0.5)&(R<k+0.5)
            a,b=Fb[m],Fref[m]
            t=((b*a.conj()).sum()/(a.abs().pow(2).sum()+1e-30))
            snr_k=(b.abs().pow(2).sum()/(1e-30+(a*t.conj()-b).abs().pow(2).sum())).real
            Fd[m]=a*t*(snr_k/(snr_k+1.0))
        f_d=fsc(Fd,Ftrue)
        rb,rd,r0=res_at(f_b),res_at(f_d),res_at(f_ref)
        frac = 0.0 if r0==rb else 100*(rd-rb)/(r0-rb)
        print(f"{sd:>8.0f}d{rb:>14} px{rd:>13} px{frac:>11.0f}%")
