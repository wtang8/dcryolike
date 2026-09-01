"""Tests for the proposal features and network.

The load-bearing property is TEST 2/3: an in-plane-invariant feature must still
distinguish a view from its ANTIPODE. The obvious choice, |FFT_theta|, is
provably blind to that (for a real density F(-k)=conj(F(k)), so the antipodal
slice has an identical magnitude spectrum) and its nearest neighbour lands at
180 deg for every view. TEST 3 pins that failure in place so a regression back
to magnitude features cannot pass silently.
"""
import sys, numpy as np, torch
sys.path.insert(0,"/home/wtang/cryoem/dcryolike"); sys.path.insert(0,"/home/wtang/cryoem/cryolike/src")
from cryolike.grids import PolarGrid
from cryolike.metadata import ViewingAngles
from dcryolike.generator import gaussian_atom_volume, make_ctf
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft
from dcryolike.proposal import crossshell_phase, inplane_invariant, ProposalNet

dev="cuda"; D=48; NF=24; torch.manual_seed(0)
vol=gaussian_atom_volume(D,device=dev)
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=128,uniform=True)
S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=vol.dtype,device=dev)
rr=torch.tensor(pg.radius_shells,dtype=vol.dtype,device=dev)
va=ViewingAngles.from_viewing_distance(np.radians(4.0))
caz=va.azimus.to(dev).to(vol.dtype); cpo=va.polars.to(dev).to(vol.dtype); V=caz.shape[0]
Tb=slice_nufft(vol,slice_points(fourier_circles(th,cpo,caz,torch.zeros_like(caz)),rr)).reshape(V,S,P)
nd=torch.stack([torch.sin(cpo)*torch.cos(caz),torch.sin(cpo)*torch.sin(caz),torch.cos(cpo)],-1)
sub=torch.randperm(V,device=dev)[:600]
angs=torch.rad2deg(torch.arccos((nd[sub]@nd.T).clamp(-1,1)))
ok=True

def nn_angle(feat):
    f=torch.nn.functional.normalize(feat.float(),dim=-1)
    sim=f[sub]@f.T
    sim[torch.arange(len(sub),device=dev),sub]=-2          # exclude self
    a=angs[torch.arange(len(sub),device=dev),sim.argmax(1)]
    return a.median().item(), (a<5).float().mean().item()*100, (a>150).float().mean().item()*100

print("="*70); print("TEST 1: in-plane invariance (a roll along theta IS a rotation)"); print("="*70)
f0=crossshell_phase(Tb[:64],None,NF)
worst=0.0
for k in (1,7,31,64):
    fk=crossshell_phase(torch.roll(Tb[:64],k,dims=-1),None,NF)
    worst=max(worst,((fk-f0).abs().max()/f0.abs().max()).item())
print(f"  max relative change over rolls of 1,7,31,64: {worst:.3e}")
# Invariance is exact in exact arithmetic (the roll's phase ramp cancels in
# C_ij = f_i conj(f_j)). It is not exact numerically: where |C| ~ eps, the
# normalisation C/(|C|+eps) amplifies 1e-16 FFT rounding to ~1e-8. A genuine
# break in invariance would be O(1), so 1e-5 still discriminates sharply.
t1=worst<1e-5; ok&=t1
print(f"  verdict: {'PASS' if t1 else 'FAIL'}  (tolerance 1e-5: eps-limited, not exact;")
print(f"           a real invariance break would be O(1))")

print("\n"+"="*70); print("TEST 2: cross-shell phase features resolve the antipode"); print("="*70)
med,near,anti=nn_angle(crossshell_phase(Tb,None,NF))
print(f"  nearest-neighbour angle: median {med:.2f} deg | <5deg {near:.0f}% | at 180deg {anti:.0f}%")
t2=(med<6.0) and (near>95) and (anti<1); ok&=t2
print(f"  verdict: {'PASS' if t2 else 'FAIL'}  (grid spacing is 4 deg)")

print("\n"+"="*70); print("TEST 3: |FFT_theta| magnitude MUST fail (regression guard)"); print("="*70)
m=inplane_invariant(Tb,NF); m=m/(m.norm(dim=(-2,-1),keepdim=True)+1e-8)
med3,near3,anti3=nn_angle(m.flatten(1))
print(f"  nearest-neighbour angle: median {med3:.2f} deg | <5deg {near3:.0f}% | at 180deg {anti3:.0f}%")
t3=(anti3>90); ok&=t3
print(f"  verdict: {'PASS' if t3 else 'FAIL'}  (expected to collapse onto the antipode;")
print(f"           if this ever passes as 'good', the guard has stopped working)")

print("\n"+"="*70); print("TEST 4: CTF robustness - retrieval survives per-particle defocus"); print("="*70)
ctf=make_ctf(pg,V,float(D),seed=5,device=dev,dtype=vol.dtype,astig=0.0)
bank=torch.nn.functional.normalize(crossshell_phase(Tb,None,NF).float(),dim=-1)
qry=torch.nn.functional.normalize(crossshell_phase(Tb*ctf,ctf,NF).float(),dim=-1)
hit=(qry[sub]@bank.T).argmax(1)
a=angs[torch.arange(len(sub),device=dev),hit]
# Query and bank come from the same templates, so exact CTF invariance shows
# up as an exact self-match. This tests the sign(ctf_i*ctf_j) correction, not
# retrieval under noise -- that is what the recall sweeps measure.
print(f"  CTF-modulated query vs clean bank: median {a.median():.2f} deg | "
      f"<5deg {100*(a<5).float().mean():.0f}%")
t4=(a.median()<6.0) and ((a<5).float().mean()>0.90); ok&=t4
print(f"  verdict: {'PASS' if t4 else 'FAIL'}")

print("\n"+"="*70); print("TEST 5: ProposalNet forward contract"); print("="*70)
tpl=crossshell_phase(Tb[:256],None,NF).float()
net=ProposalNet(tpl.shape[1],tpl.shape[1]).to(dev)
with torch.no_grad():
    e=net.tpl(tpl); sc=net.scores(tpl,e)
n_ok=torch.allclose(e.norm(dim=-1),torch.ones(e.shape[0],device=dev),atol=1e-5)
s_ok=torch.isfinite(sc).all().item() and sc.shape==(256,256)
print(f"  embeddings L2-normalised : {n_ok}")
print(f"  scores finite, shape {tuple(sc.shape)} : {s_ok}")
t5=n_ok and s_ok; ok&=t5; print(f"  verdict: {'PASS' if t5 else 'FAIL'}")

print(f"\n{'='*70}\noverall: {'PASS' if ok else 'FAIL'}")
import sys as _sys; _sys.exit(0 if ok else 1)
