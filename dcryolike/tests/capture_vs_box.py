"""Does the basin of attraction shrink with box size?

Capture radius should track the width of the correlation peak, ~1/k_max. If so,
the "a 20 deg grid is free" result from D=48 will NOT survive to realistic box
sizes, and the coarse search comes back as a real cost.

Perturb the TRUE pose by a controlled amount, refine, and measure the recovery
rate as a function of starting error. In-plane sampling is scaled with the box
so the outermost shell stays properly sampled.
"""
import sys, time, numpy as np, torch
sys.path.insert(0,"/home/wtang/cryoem/dcryolike"); sys.path.insert(0,"/home/wtang/cryoem/cryolike/src")
from cryolike.grids import PolarGrid
from dcryolike.generator import gaussian_atom_volume, generate
from dcryolike.refine import refine_poses

dev="cuda"; N=128; ISO=dict(astig=0.0)
BINS=[(0,2),(2,5),(5,10),(10,20),(20,35),(35,60),(60,180)]

def ang_err(a,p,at,pt):
    n1=torch.stack([torch.sin(p)*torch.cos(a),torch.sin(p)*torch.sin(a),torch.cos(p)],-1)
    n2=torch.stack([torch.sin(pt)*torch.cos(at),torch.sin(pt)*torch.sin(at),torch.cos(pt)],-1)
    return torch.rad2deg(torch.arccos((n1*n2).sum(-1).clamp(-1,1)))

g=torch.Generator(device=dev).manual_seed(11)
print(f"{'D':>5}{'k_max':>7}{'n_inpl':>8}{'SNR':>7}   recovery rate by starting error")
print(f"{'':27}" + "".join(f"{f'{lo}-{hi}':>9}" for lo,hi in BINS) + f"{'capture':>10}")
print("-"*(27+9*len(BINS)+10))

for D in (32,48,64,96,128):
    P=int(2**np.ceil(np.log2(max(128, np.pi*D))))
    vol=gaussian_atom_volume(D,device=dev)
    pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=P,uniform=True)
    S=pg.n_shells
    th=torch.tensor(pg.theta_shell,dtype=vol.dtype,device=dev)
    rr=torch.tensor(pg.radius_shells,dtype=vol.dtype,device=dev)
    w=torch.tensor(pg.integration_weight_points,dtype=vol.dtype,device=dev).reshape(S,P)
    tol = 5.0 if D<=64 else 3.0     # success threshold scaled loosely with precision
    for snr in (1.0,0.1):
        p=generate(vol,pg,N,snr,seed=17,device=dev,ctf_kwargs=ISO)
        scale=torch.exp(torch.empty(N,device=dev,dtype=vol.dtype).uniform_(np.log(0.005),np.log(1.3)))
        a0=p.azimus+torch.randn(N,generator=g,device=dev,dtype=vol.dtype)*scale
        p0=p.polars+torch.randn(N,generator=g,device=dev,dtype=vol.dtype)*scale
        g0=p.gammas+torch.randn(N,generator=g,device=dev,dtype=vol.dtype)*scale
        e0=ang_err(a0,p0,p.azimus,p.polars)
        a1,p1,_=refine_poses(vol,p.images_fourier,p.ctf,th,rr,w,a0,p0,g0,steps=250,lr=0.02)
        e1=ang_err(a1,p1,p.azimus,p.polars)
        rates=[]; row=""
        for lo,hi in BINS:
            m=(e0>=lo)&(e0<hi)
            if m.sum()<4: row+=f"{'-':>9}"; rates.append(None); continue
            r=(e1[m]<tol).float().mean().item()
            rates.append(r); row+=f"{100*r:>8.0f}%"
        # capture radius: last bin whose rate is >= half the first valid bin's rate
        valid=[(i,r) for i,r in enumerate(rates) if r is not None]
        base=valid[0][1] if valid else 0
        cap=0
        for i,r in valid:
            if base>0 and r>=0.5*base: cap=BINS[i][1]
        print(f"{D:>5}{D//2:>7}{P:>8}{snr:>7.1f}   {row}{cap:>9.0f}d")
    del vol; torch.cuda.empty_cache()
print("\nIf capture radius tracks 1/k_max it should halve as D doubles.")
