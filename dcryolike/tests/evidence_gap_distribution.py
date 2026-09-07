"""Distribution of the per-particle evidence gap, and what an adaptive rule could win.

WHY. Every marginalisation number in this project is a MEDIAN over particles at one
nominal SNR. Real datasets are not like that: ice thickness, defocus and particle
orientation all vary within a single collection, so the per-particle evidence gap --
the log-likelihood advantage of the best pose over its nearest rival -- has a
DISTRIBUTION. The anatomy measurement showed averaging over the pose shortlist only
helps where that gap is of order a nat. So the question that decides whether the
effect generalises is not "what is the gap at SNR 0.1" but "what fraction of a real
stack sits in the low-gap tail".

If that fraction is usable, an adaptive rule -- marginalise the ambiguous particles,
take the argmax for the rest -- captures the benefit where it exists at no cost
elsewhere, and the claim generalises. If it is negligible, it does not.

WHAT IS MEASURED, at D=96 with per-particle SNR spread:
  the gap distribution by decile; the fraction below 1, 3 and 10 nats; the effective
  number of competing basins and the top-1 accuracy within each gap bin; and how much
  of the variation the particle's own SNR explains (versus its orientation, which also
  matters because some views are more distinguishable than others).

SPREAD is the geometric factor between the 10th and 90th percentile of per-particle
SNR: 1.0 is the uniform case used everywhere else, 3.0 a realistic collection.
"""
import os, math, numpy as np, torch, sys
import _paths  # noqa: F401
from cryolike.grids import PolarGrid
from cryolike.metadata import ViewingAngles
from dcryolike.generator import gaussian_atom_volume, generate_physical
from dcryolike.pipeline import Bank, neighbourhood, temperature

dev="cuda"; DT=torch.float32; ISO=dict(astig=0.0)
D=int(os.environ.get("D","96")); NP=int(os.environ.get("NP","2000"))
SNR_MED=float(os.environ.get("SNR_MED","0.1"))
SPREADS=[float(x) for x in os.environ.get("SPREADS","1.0,3.0").split(",")]
MODES=int(os.environ.get("MODES","10"))
kmax=D//2; tol=np.degrees(1.25/kmax); NEIGH=10.0*min(1.0,24.0/kmax)

vol=gaussian_atom_volume(D,device=dev,dtype=DT)
P=min(256,int(2**np.ceil(np.log2(2*np.pi*kmax*0.85))))
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=P,uniform=True); S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=DT,device=dev); rr=torch.tensor(pg.radius_shells,dtype=DT,device=dev)
w=torch.tensor(pg.integration_weight_points,dtype=DT,device=dev).reshape(S,P); wr=w[:,0]
va=ViewingAngles.from_viewing_distance(1.0/kmax)
caz=va.azimus.to(dev).to(DT); cpo=va.polars.to(dev).to(DT); V=caz.shape[0]
NB,nd=neighbourhood(caz,cpo,NEIGH)
bank=Bank(vol,th,cpo,caz,rr,S,P,single=True)
print(f"D={D}  {V} views @ {np.degrees(1.0/kmax):.2f} deg  tol {tol:.2f} deg  "
      f"{NP} particles  median SNR {SNR_MED}")

for spread in SPREADS:
    g=torch.Generator(device="cpu").manual_seed(11)
    if spread<=1.0:
        snr=torch.full((NP,),SNR_MED,dtype=torch.float64)
    else:
        # lognormal with the stated 10-90 geometric range
        sig=math.log(spread)/(2*1.2816)
        snr=SNR_MED*torch.exp(sig*torch.randn(NP,generator=g,dtype=torch.float64))
    snr_d=snr.to(dev).to(DT)
    p=generate_physical(vol,pg,NP,snr_d,seed=17,device=dev,ctf_kwargs=ISO,keep_clean=False)
    beta,tau,C=temperature(p,w,D)
    In=torch.sqrt((p.images_fourier.abs()**2*w).sum(dim=(1,2))); cr=p.ctf[:,:,0]
    bc,bg=bank.search(p.images_fourier,cr,wr,In)
    nbmax=torch.empty_like(bc)
    for i in range(0,V,1024):
        j=min(i+1024,V); nbmax[:,i:j]=bc[:,NB[i:j]].max(-1).values
    bc.masked_fill_(~(bc>=nbmax-1e-12),-2.0); del nbmax
    tcc,tidx=bc.topk(MODES,dim=1)
    gap=(beta.to(DT)*(tcc[:,0]**2-tcc[:,1]**2)).double()
    lg=(beta[:,None].to(DT)*(tcc**2-tcc[:,:1]**2)).masked_fill(tcc<-1.0,-1e30)
    wt=torch.softmax(lg.double(),dim=1)
    neff=(-(wt.clamp(min=1e-30)*wt.clamp(min=1e-30).log()).sum(1)).exp()
    nt=torch.stack([torch.sin(p.polars)*torch.cos(p.azimus),
                    torch.sin(p.polars)*torch.sin(p.azimus),torch.cos(p.polars)],-1).to(nd.dtype)
    ang=torch.rad2deg(torch.arccos((nt@nd.T).clamp(-1,1)))
    ok=(torch.gather(ang,1,tidx[:,:1]).squeeze(1)<=tol)
    print(f"\n=== SNR spread {spread:.1f}x  (10-90 percentile {np.percentile(snr.numpy(),10):.3f}"
          f"-{np.percentile(snr.numpy(),90):.3f})")
    q=np.percentile(gap.cpu().numpy(),[10,25,50,75,90])
    print("  evidence gap, nats, by percentile:  "
          +"  ".join(f"p{p_}={v:.1f}" for p_,v in zip([10,25,50,75,90],q)))
    for thr in (1.0,3.0,10.0):
        m=gap<thr
        f=m.double().mean().item()*100
        print(f"  below {thr:5.1f} nats: {f:5.1f}% of particles"
              +(f"   n_eff {neff[m].median().item():.2f}   top-1 {ok[m].double().mean().item()*100:5.1f}%"
                if m.any() else ""))
    # how much of the gap does the particle's own SNR explain?
    ls=torch.log(snr_d.double()); lgp=torch.log(gap.clamp(min=1e-6))
    r=((ls-ls.mean())*(lgp-lgp.mean())).sum()/((ls-ls.mean()).norm()*(lgp-lgp.mean()).norm()+1e-30)
    print(f"  corr(log SNR, log gap) = {r.item():+.3f}  "
          f"-> {'SNR spread drives the tail' if abs(r)>0.6 else 'orientation matters as much as SNR'}")
    del p,bc,bg,tcc,tidx,ang; torch.cuda.empty_cache()
print("\nAn adaptive rule (marginalise the low-gap particles, argmax the rest) can only")
print("matter in proportion to the fraction below a few nats. Compare that fraction with")
print("the +1-4% margin measured where the whole stack is in that regime.")
