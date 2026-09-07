"""Anatomy of the pose posterior: between-basin vs within-basin, as a function of
the per-image evidence. Replaces grid-searching (D, SNR).

WHY THIS EXISTS. Earlier work here reported mixture-vs-argmax margins at four
(D, SNR) points and concluded the advantage "collapses at D=96". Two things were
wrong with that.

  1. D and SNR are not independent knobs. The posterior temperature is
     beta = ||y||^2_w / (2 sigma^2 C), and it converges to D^2/2 -- the pixel
     count -- almost independently of SNR (measured: 4653-5301 at D=96,
     1171-1443 at D=48). What SNR changes is the EVIDENCE GAP between competing
     poses. The controlling quantity is that gap in nats, not the pair (D, SNR).

  2. The earlier n_eff counted local maxima separated by >= NEIGH degrees, so it
     collapsed each basin to a single mode. n_eff = 1.00 meant ONE BASIN, not
     zero pose uncertainty. At D=96 the success tolerance is 1.49 deg and a basin
     is several degrees wide, so the within-basin spread -- the part that
     actually produces pose-error blur in the reconstruction -- was invisible to
     the statistic used to declare there was nothing to average.

WHAT THIS MEASURES, per condition, reported against beta and the evidence gap:

  BETWEEN-BASIN (global search on the standard view grid)
    n_eff_basin  exp(entropy) of the posterior over local maxima >= NEIGH apart
    gap_nats     beta * (cc_1^2 - cc_2^2), the log-evidence gap to the runner-up
    top1         fraction whose best basin is within tolerance

  WITHIN-BASIN (dense local scan around each particle's TRUE pose, so the basin
                is resolved regardless of the global grid spacing)
    n_eff_local  exp(entropy) of the posterior over the local cone
    sd_deg       posterior standard deviation of the viewing direction, degrees
    frac_tol     posterior mass within the success tolerance

INTERPRETATION. Marginalising over pose can only help where the posterior has
width. n_eff_basin says whether there are competing BASINS; sd_deg says whether
there is width WITHIN the winning basin. If sd_deg is comparable to or larger
than the tolerance, dense local marginalisation is the operation worth testing,
and a 10-mode shortlist is the wrong instrument for it.
"""
import os, math, numpy as np, torch, sys
import _paths  # noqa: F401
from cryolike.grids import PolarGrid
from cryolike.metadata import ViewingAngles
from dcryolike.generator import gaussian_atom_volume, generate_physical
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft
from dcryolike.pipeline import Bank, neighbourhood, temperature

dev="cuda"
DS=[int(x) for x in os.environ.get("DS","48,64,96").split(",")]
SNRS=[float(x) for x in os.environ.get("SNRS","0.3,0.1,0.05").split(",")]
NP=int(os.environ.get("NP","200"))          # particles for the global statistics
NPROBE=int(os.environ.get("NPROBE","48"))   # particles for the dense local scan
MODES=int(os.environ.get("MODES","10"))
CONE=float(os.environ.get("CONE","4.0"))    # local scan half-angle, degrees
STEP=float(os.environ.get("STEP","0.25"))   # local scan spacing, degrees
DT=torch.float32; ISO=dict(astig=0.0)

def local_directions(pol0, azi0, cone_deg, step_deg, dtype, device):
    """A dense grid of viewing directions inside a cone about (pol0, azi0).
    Returns (m,) polars, (m,) azimus and (m,) angular offsets in degrees."""
    r = torch.arange(0.0, cone_deg + 1e-9, step_deg, dtype=dtype, device=device)
    dirs = [(torch.zeros(1, dtype=dtype, device=device), torch.zeros(1, dtype=dtype, device=device))]
    for rad in r[1:]:
        n_phi = max(6, int(round(2 * np.pi * float(rad) / step_deg)))
        phi = torch.arange(n_phi, dtype=dtype, device=device) * (2 * np.pi / n_phi)
        dirs.append((torch.full((n_phi,), float(rad), dtype=dtype, device=device), phi))
    rad = torch.cat([d[0] for d in dirs]); phi = torch.cat([d[1] for d in dirs])
    # offset (rad, phi) in the tangent plane at the reference direction
    t = torch.deg2rad(rad)
    n0 = torch.stack([torch.sin(pol0)*torch.cos(azi0), torch.sin(pol0)*torch.sin(azi0), torch.cos(pol0)])
    a = torch.tensor([0.0,0.0,1.0],dtype=dtype,device=device)
    if torch.abs(torch.dot(n0,a)) > 0.9: a = torch.tensor([1.0,0.0,0.0],dtype=dtype,device=device)
    e1 = torch.cross(n0,a,dim=0); e1 = e1/e1.norm(); e2 = torch.cross(n0,e1,dim=0)
    nd = (torch.cos(t)[:,None]*n0[None,:]
          + (torch.sin(t)*torch.cos(phi))[:,None]*e1[None,:]
          + (torch.sin(t)*torch.sin(phi))[:,None]*e2[None,:])
    pol = torch.arccos(nd[:,2].clamp(-1,1)); azi = torch.atan2(nd[:,1], nd[:,0])
    return pol, azi, rad

print(f"{'D':>4}{'SNR':>6}{'beta':>8}{'gap_nats':>10}{'n_eff_bas':>10}{'top1':>7}"
      f"{'n_eff_loc':>10}{'sd_deg':>8}{'tol_deg':>8}{'frac_tol':>9}")
print("-"*88)
for D in DS:
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
    for snr in SNRS:
        p=generate_physical(vol,pg,NP,snr,seed=17,device=dev,ctf_kwargs=ISO,keep_clean=False)
        beta,tau,C=temperature(p,w,D)
        In=torch.sqrt((p.images_fourier.abs()**2*w).sum(dim=(1,2))); cr=p.ctf[:,:,0]
        bc,bg=bank.search(p.images_fourier,cr,wr,In)
        # ---- between-basin: posterior over local maxima
        nbmax=torch.empty_like(bc)
        for i in range(0,V,1024):
            j=min(i+1024,V); nbmax[:,i:j]=bc[:,NB[i:j]].max(-1).values
        masked=bc.masked_fill(~(bc>=nbmax-1e-12),-2.0)
        tcc,tidx=masked.topk(MODES,dim=1)
        lg=(beta[:,None].to(DT)*(tcc**2-tcc[:,:1]**2)).masked_fill(tcc<-1.0,-1e30)
        wt=torch.softmax(lg.double(),dim=1)
        n_eff_b=(-(wt.clamp(min=1e-30)*wt.clamp(min=1e-30).log()).sum(1)).exp().median().item()
        gap=(beta.to(DT)*(tcc[:,0]**2-tcc[:,1]**2)).double().median().item()
        nt=torch.stack([torch.sin(p.polars)*torch.cos(p.azimus),
                        torch.sin(p.polars)*torch.sin(p.azimus),torch.cos(p.polars)],-1).to(nd.dtype)
        ang=torch.rad2deg(torch.arccos((nt@nd.T).clamp(-1,1)))
        top1=(torch.gather(ang,1,tidx[:,:1]).squeeze(1)<=tol).double().mean().item()*100
        del bc,bg,nbmax,masked,ang; torch.cuda.empty_cache()
        # ---- within-basin: dense scan about each probe particle's TRUE pose
        ne_l=[]; sd_l=[]; ft_l=[]
        for q in range(min(NPROBE,NP)):
            pol,azi,rad = local_directions(p.polars[q],p.azimus[q],CONE,STEP,DT,dev)
            m=pol.shape[0]
            cc=torch.empty(m,dtype=DT,device=dev)
            for i in range(0,m,512):
                j=min(i+512,m)
                Tb=slice_nufft(vol,slice_points(fourier_circles(th,pol[i:j],azi[i:j],
                        torch.zeros(j-i,dtype=DT,device=dev)),rr)).reshape(j-i,S,P)
                Tf=torch.fft.fft(Tb,dim=-1).conj()
                Tn=torch.sqrt((Tb.abs()**2*(w*cr[q][:,None]**2)).sum(dim=(1,2)))
                If=torch.fft.fft(p.images_fourier[q][None]*(w*cr[q][:,None]),dim=-1)
                c=torch.fft.ifft(Tf*If,dim=-1).real.sum(1).amax(-1)
                cc[i:j]=c/(Tn*In[q]); del Tb,Tf,If,c
            lgl=(beta[q]*(cc**2-cc.max()**2)).double()
            wl=torch.softmax(lgl,0)
            ne_l.append((-(wl.clamp(min=1e-30)*wl.clamp(min=1e-30).log()).sum()).exp().item())
            sd_l.append(torch.sqrt((wl*rad.double()**2).sum()-(wl*rad.double()).sum()**2).item())
            ft_l.append(wl[rad<=tol].sum().item())
            del cc,wl
        print(f"{D:>4}{snr:>6.2f}{beta.median().item():>8.0f}{gap:>10.1f}{n_eff_b:>10.2f}{top1:>6.1f}%"
              f"{np.median(ne_l):>10.1f}{np.median(sd_l):>8.2f}{tol:>8.2f}{np.median(ft_l):>9.2f}",flush=True)
        del p; torch.cuda.empty_cache()
    del bank,vol; torch.cuda.empty_cache()
print("\nn_eff_bas: competing BASINS (local maxima >= NEIGH apart). 1.0 = one basin wins outright.")
print("n_eff_loc / sd_deg: width WITHIN the winning basin, from a dense scan about the true pose.")
print("Marginalising helps where the posterior has width. Compare sd_deg with tol_deg:")
print("sd_deg >= tol_deg means the pose is uncertain at the scale that matters for the map,")
print("even when n_eff_bas = 1.0 and a 10-mode shortlist reports 'nothing to average'.")
