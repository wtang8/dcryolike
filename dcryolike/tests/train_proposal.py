"""Train a proposal network, then score it against the exact-likelihood ceiling
on IDENTICAL test particles, with the identical recall@K protocol.
"""
import sys, time, numpy as np, torch, torch.nn.functional as F
import _paths  # noqa: F401  (sets sys.path relative to this file)
from cryolike.grids import PolarGrid
from cryolike.metadata import ViewingAngles
from dcryolike.generator import gaussian_atom_volume, generate, uniform_so3, make_ctf
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft
from dcryolike.proposal import ProposalNet, crossshell_phase

dev="cuda"; D=48; TOL=3.0; N_TEST=200; NFREQ=24
KS=[1,5,10,25,50,100,250,500,1000,2000]
SNRS=[1.0,0.5,0.1,0.05,0.02,0.01]
ISO=dict(astig=0.0)
torch.manual_seed(0)

vol = gaussian_atom_volume(D, device=dev)
pg  = PolarGrid(radius_max=D/2, dist_radii=1.0, n_inplanes=128, uniform=True)
S,P = pg.n_shells, pg.n_inplanes
th  = torch.tensor(pg.theta_shell, dtype=vol.dtype, device=dev)
rr  = torch.tensor(pg.radius_shells, dtype=vol.dtype, device=dev)
w   = torch.tensor(pg.integration_weight_points, dtype=vol.dtype, device=dev).reshape(S,P)
wr  = w[:,0]

va  = ViewingAngles.from_viewing_distance(1.0/(D/2))
caz = va.azimus.to(dev).to(vol.dtype); cpo = va.polars.to(dev).to(vol.dtype)
V   = caz.shape[0]
Tb  = slice_nufft(vol, slice_points(fourier_circles(th, cpo, caz, torch.zeros_like(caz)), rr)).reshape(V,S,P)
ndir= torch.stack([torch.sin(cpo)*torch.cos(caz), torch.sin(cpo)*torch.sin(caz), torch.cos(cpo)],-1)
print(f"bank: {V} views | grid {S}x{P} | D={D}")

# ---------------- exact likelihood (the ceiling) ----------------
def exact_scores(imgs, ctf):
    ctf_r = ctf[:,:,0]
    Tf = torch.fft.fft(Tb, dim=-1).conj()
    If = torch.fft.fft(imgs, dim=-1)
    Tpow = (Tb.abs()**2).sum(-1)
    Tn = torch.sqrt(torch.einsum('vr,br->bv', Tpow, wr[None,:]*ctf_r**2))
    In = torch.sqrt((imgs.abs()**2*w).sum(dim=(1,2)))
    out = torch.empty(imgs.shape[0], V, device=dev, dtype=vol.dtype)
    for i in range(0, imgs.shape[0], 8):
        j=min(i+8, imgs.shape[0])
        cw=(wr[None,:]*ctf_r[i:j])[:,:,None]
        cc=torch.fft.ifft(torch.einsum('vrk,brk->bvk', Tf, If[i:j]*cw), dim=-1).real
        out[i:j]=cc.amax(-1)/(Tn[i:j]*In[i:j,None])
    return out

# ---------------- features ----------------
tpl_feat = crossshell_phase(Tb, None, NFREQ).float()
_probe = crossshell_phase(Tb[:2], make_ctf(pg,2,float(D),device=dev,dtype=vol.dtype,**ISO), NFREQ)
net = ProposalNet(_probe.shape[1], tpl_feat.shape[1]).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=6e-4, weight_decay=1e-4)
print(f"features: img {_probe.shape[1]}  tpl {tpl_feat.shape[1]}  | params "
      f"{sum(p.numel() for p in net.parameters())/1e6:.2f}M")

# ---------------- training ----------------
# Particles at grid views are just template * CTF + noise: features are in-plane
# invariant, so gamma is irrelevant and no NUFFT is needed in the loop.
def batch(B, gen):
    v = torch.randint(0, V, (B,), device=dev, generator=gen)
    snr = torch.exp(torch.empty(B, device=dev).uniform_(np.log(0.01), np.log(1.0)))
    ctf = make_ctf(pg, B, float(D), seed=int(torch.randint(0,10**6,(1,),device=dev,generator=gen).item()),
                   device=dev, dtype=vol.dtype, **ISO)
    clean = Tb[v]*ctf
    pw = (clean.abs()**2*w).sum(dim=(1,2))/w.sum()
    sig = torch.sqrt(pw/snr.to(vol.dtype))
    noise = torch.randn(clean.shape, dtype=clean.dtype, device=dev)
    return crossshell_phase(clean+noise*sig[:,None,None], ctf, NFREQ).float(), v

STEPS, BS = 30000, 256
gen = torch.Generator(device=dev).manual_seed(1)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=6e-4, total_steps=STEPS, pct_start=0.05)
t0=time.time(); net.train()
for s in range(STEPS):
    f, v = batch(BS, gen)
    loss = F.cross_entropy(net.scores(f, net.tpl(tpl_feat)), v)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if (s+1) % 5000 == 0:
        print(f"  step {s+1:5d}  loss {loss.item():.3f}  ({time.time()-t0:.0f}s)")
net.eval()

# ---------------- evaluation: identical particles, two rankers ----------------
def recall(sc, ang):
    o = sc.argsort(dim=1, descending=True)
    hit = torch.gather(ang, 1, o) <= TOL
    first = torch.where(hit.any(1), hit.float().argmax(1),
                        torch.full((sc.shape[0],), V, device=dev))
    return [ (first<k).float().mean().item()*100 for k in KS ]

print(f"\n{'':6}{'':4}" + "".join(f"{('@'+str(k)):>8}" for k in KS))
print("-"*(10+8*len(KS)))
with torch.no_grad():
    for snr in SNRS:
        p = generate(vol, pg, N_TEST, snr, seed=17, device=dev, ctf_kwargs=ISO)
        nt = torch.stack([torch.sin(p.polars)*torch.cos(p.azimus),
                          torch.sin(p.polars)*torch.sin(p.azimus),
                          torch.cos(p.polars)],-1)
        ang = torch.rad2deg(torch.arccos((nt @ ndir.T).clamp(-1,1)))
        ceil = recall(exact_scores(p.images_fourier, p.ctf), ang)
        f = crossshell_phase(p.images_fourier, p.ctf, NFREQ).float()
        netr = recall(net.scores(f, net.tpl(tpl_feat)).to(vol.dtype), ang)
        print(f"{snr:<6.2f}{'exact':>4}" + "".join(f"{x:7.1f}%" for x in ceil))
        print(f"{'':6}{'net':>4}" + "".join(f"{x:7.1f}%" for x in netr))
        print(f"{'':6}{'gap':>4}" + "".join(f"{n-c:+7.1f} " for c,n in zip(ceil,netr)))
print(f"\ntotal {time.time()-t0:.0f}s")
torch.save(net.state_dict(), str(__import__('pathlib').Path(__file__).resolve().parents[1]/'proposal_net.pt'))
