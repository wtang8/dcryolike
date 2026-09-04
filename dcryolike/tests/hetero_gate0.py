"""GATE ZERO for heterogeneous reconstruction: does motion destroy the pose shortlist?

The whole design (pose shortlist from a matched filter, heterogeneity handled
by marginalisation AFTER the shortlist) rests on one assumption: a consensus
volume still finds the true pose in its shortlist when the particle is a
different conformation. This measures that directly, on the identical code
path as the recall ceiling (recall_ceiling_physical_boxsize_lowmem.py).

Phantom: the helical arm of the standard phantom swung about a hinge by t,
t uniform in [-SWING, +SWING] degrees over NCONF conformations. The bank is
built from the CONSENSUS (mean over conformations), which is what a
homogeneous refinement converges to. SWING=0 reproduces the homogeneous
D=96 table (a consistency check on the run itself).

Read the verdict off recall@100 vs SWING at each SNR:
  - flat            : shortlist survives; heterogeneity enters after it (plan stands)
  - collapses       : the search itself must be class-aware
  - extremes only   : (see the |t| split) a wider shortlist or a two-class bank fixes it

Memory (D=96, P=256): bank 6 GB + generator DFT matrix 1.8 GB, freed between
swings. Run it in a cgroup so a surprise cannot take WSL down:
  systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 \\
      venv/bin/python dcryolike/tests/hetero_gate0.py
Knobs: D, P_INPLANE, N_TEST, NCONF, SWINGS ("0,10,30"), SNRS ("0.3,0.1,0.05").
"""
import _paths  # noqa
import os, time, math, numpy as np, torch
from cryolike.grids import PolarGrid
from cryolike.metadata import ViewingAngles
from dcryolike.generator import hinge_phantom, generate_physical_hetero, real_space_projections
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft, fourier_volume
from dcryolike.pipeline import ShellMasks, res_at

dev="cuda"; D=int(os.environ.get("D","96")); P=int(os.environ.get("P_INPLANE","256"))
N_TEST=int(os.environ.get("N_TEST","200")); NCONF=int(os.environ.get("NCONF","32"))
SWINGS=[float(x) for x in os.environ.get("SWINGS","0,10,30").split(",")]
SNRS=[float(x) for x in os.environ.get("SNRS","0.3,0.1,0.05").split(",")]
KS=[1,5,25,100,500,2000]; ISO=dict(astig=0.0); CDT=torch.complex64
kmax=D//2; tol=np.degrees(1.25/kmax)

pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=P,uniform=True); S=pg.n_shells
th=torch.tensor(pg.theta_shell,dtype=torch.float64,device=dev)
rr=torch.tensor(pg.radius_shells,dtype=torch.float64,device=dev)
w=torch.tensor(pg.integration_weight_points,dtype=torch.float32,device=dev).reshape(S,P); wr=w[:,0]
va=ViewingAngles.from_viewing_distance(1.0/kmax)
caz=va.azimus.to(dev).double(); cpo=va.polars.to(dev).double(); V=caz.shape[0]
nd=torch.stack([torch.sin(cpo)*torch.cos(caz),torch.sin(cpo)*torch.sin(caz),torch.cos(cpo)],-1).float()
shells=ShellMasks(D,dev)
print(f"D={D} k_max={kmax} grid {S}x{P} {V} views @ {np.degrees(1.0/kmax):.2f} deg, tol {tol:.2f} deg, "
      f"N={N_TEST}/SNR, {NCONF} conformations")

def build_bank(vol):
    Tf=torch.empty(V,S,P,dtype=CDT,device=dev); Tpow=torch.empty(V,S,dtype=torch.float32,device=dev)
    for i in range(0,V,1024):
        j=min(i+1024,V); z=torch.zeros(j-i,device=dev,dtype=vol.dtype)
        Tb=slice_nufft(vol,slice_points(fourier_circles(th,cpo[i:j],caz[i:j],z),rr)).reshape(j-i,S,P).to(CDT)
        Tpow[i:j]=(Tb.abs()**2).sum(-1); Tf[i:j]=torch.fft.fft(Tb,dim=-1).conj(); del Tb
    return Tf,Tpow

def score_all(Tf,Tpow,imgs,ctf):
    imgs=imgs.to(CDT); cr=ctf[:,:,0].float()
    Tn=torch.sqrt(torch.einsum('vr,br->bv',Tpow,wr[None,:]*cr**2))
    In=torch.sqrt((imgs.abs()**2*w).sum(dim=(1,2)))
    out=torch.empty(imgs.shape[0],V,device=dev,dtype=torch.float32)
    for i in range(0,imgs.shape[0],2):
        j=min(i+2,imgs.shape[0]); cw=(wr[None,:]*cr[i:j])[:,:,None].to(CDT)
        If=torch.fft.fft(imgs[i:j],dim=-1)*cw
        cc=torch.fft.ifft(torch.einsum('vrk,brk->bvk',Tf,If),dim=-1).real
        out[i:j]=cc.amax(-1)/(Tn[i:j]*In[i:j,None]); del cc
    return out

summary={}
for swing in SWINGS:
    t0=time.time()
    nconf=1 if swing==0 else NCONF
    ts=[0.0] if swing==0 else list(np.linspace(-math.radians(swing),math.radians(swing),nconf))
    vols=[]; rms=0.0
    for t in ts:
        v,r=hinge_phantom(D,float(t),device=dev); vols.append(v); rms=max(rms,r)
    cons=torch.stack(vols).mean(0).contiguous()
    Fc=fourier_volume(cons); Fe=fourier_volume(vols[-1])
    moving=((vols[-1]-cons).pow(2).sum()/cons.pow(2).sum()).sqrt().item()
    fsc_ce=res_at(shells.fsc(Fc,Fe)); del Fc,Fe
    Tf,Tpow=build_bank(cons); torch.cuda.synchronize()
    print(f"\n=== SWING +-{swing:.0f} deg: arm rms displacement {rms:.1f} vox at the extreme, "
          f"|extreme-consensus|/|consensus| = {moving:.3f}, consensus-vs-extreme FSC0.5 at {fsc_ce} px "
          f"(bank {time.time()-t0:.0f}s, {Tf.numel()*8/1e9:.1f} GB) ===")
    hdr=f"{'SNR':>6}"+"".join(f"{('@'+str(k)):>8}" for k in KS)+f"{'medrank':>9}{'top1err':>9}{'@100 |t|<half':>15}{'@100 |t|>half':>15}"
    print(hdr); print("-"*len(hdr))
    for snr in SNRS:
        t1=time.time()
        conf=torch.arange(N_TEST,device=dev)%nconf
        p=generate_physical_hetero(vols,ts,conf,pg,snr,seed=17,device=dev,ctf_kwargs=ISO)
        sc=score_all(Tf,Tpow,p.images_fourier,p.ctf)
        nt=torch.stack([torch.sin(p.polars)*torch.cos(p.azimus),torch.sin(p.polars)*torch.sin(p.azimus),torch.cos(p.polars)],-1).float()
        ang=torch.rad2deg(torch.arccos((nt@nd.T).clamp(-1,1)))
        o=sc.argsort(dim=1,descending=True); hit=torch.gather(ang,1,o)<=tol
        first=torch.where(hit.any(1),hit.float().argmax(1),torch.full((N_TEST,),V,device=dev))
        top1=torch.gather(ang,1,o[:,:1]).squeeze(1)
        inner=p.t.abs()<=(math.radians(swing)/2+1e-9)
        r100=lambda m: (first[m]<100).float().mean().item()*100 if m.any() else float('nan')
        summary[(swing,snr)]=(first<100).float().mean().item()*100
        print(f"{snr:>6.2f}"+"".join(f"{(first<k).float().mean().item()*100:7.1f}%" for k in KS)
              +f"{first.median().item():9.0f}{top1.median().item():8.1f}°{r100(inner):14.1f}%{r100(~inner):14.1f}%   ({time.time()-t1:.0f}s)")
        del p,sc,ang,o,hit; torch.cuda.empty_cache()
    del Tf,Tpow,vols,cons; torch.cuda.empty_cache()

print("\n=== GATE: recall@100 (consensus bank) vs swing ===")
print(f"{'SNR':>6}"+"".join(f"{('+-'+str(int(s))+'deg'):>10}" for s in SWINGS))
for snr in SNRS:
    print(f"{snr:>6.2f}"+"".join(f"{summary[(s,snr)]:9.1f}%" for s in SWINGS))
print("flat across swing -> shortlist survives motion; heterogeneity enters after the shortlist.")
