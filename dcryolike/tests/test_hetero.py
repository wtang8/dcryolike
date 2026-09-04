"""Verification of the heterogeneous-reconstruction module (dcryolike/hetero.py).

Tests 1-6 are CPU-only (no NUFFT), so they can run while the GPU is busy:
    CUDA_VISIBLE_DEVICES= python dcryolike/tests/test_hetero.py
Test 7 (generator bridge) needs cufinufft and is skipped without a GPU.

Each test has a negative control or an independent reference, so a green run
means the code was compared to something, not merely that it ran.
"""
import sys, os, math, numpy as np, torch
import _paths  # noqa: F401
from dcryolike.hetero import (hetero_gram, marginal_loglik, insert_trilinear, HeteroAccumulator,
                              fix_gauge, rigid_tangents, rotation_leakage, rank_correlation,
                              latent_r2, align_bases)
from dcryolike.generator import _phantom_atoms, _render_atoms, gaussian_atom_volume, hinge_phantom

_FAILS=[]
def _v(c,label=''):
    _FAILS.append(bool(c))
    return 'PASS' if c else 'FAIL'
GPU=torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES","x")!=""
dev="cuda" if GPU else "cpu"
torch.manual_seed(0)

# ------------------------------------------------------------------------
print("="*64); print("TEST 1: Gram over all in-plane rotations == explicit rotation"); print("="*64)
n,M,K1,S,P=2,3,3,5,16
y=torch.randn(n,S,P,dtype=torch.complex128)
T=torch.randn(n,M,K1,S,P,dtype=torch.complex128)
w=torch.rand(S,1,dtype=torch.float64).expand(S,P).contiguous()
for name,ctf in (("isotropic",torch.rand(n,S,1,dtype=torch.float64).expand(n,S,P).contiguous()),
                 ("anisotropic",torch.rand(n,S,P,dtype=torch.float64))):
    A,b,Gyy=hetero_gram(y,T,ctf,w)
    # explicit: rotate the IMAGE by gi steps (theta -> theta+gi), template fixed
    errA=errb=0.0
    for gi in range(P):
        yr=torch.roll(y,-gi,dims=-1); cr=torch.roll(ctf,-gi,dims=-1)
        wc=w[None]*cr
        b_ref=torch.einsum('nmksp,nsp->nmk',T.conj(),(wc*yr).to(T.dtype)).real
        A_ref=torch.einsum('nmjsp,nmksp,nsp->nmjk',T.conj(),T,(wc*cr).to(T.dtype)).real
        errb=max(errb,(b[:,:,gi]-b_ref).abs().max().item()/b_ref.abs().max().item())
        errA=max(errA,(A[:,:,gi]-A_ref).abs().max().item()/A_ref.abs().max().item())
    ok=errA<1e-10 and errb<1e-10 and abs(Gyy[0]-(y[0].abs()**2*w).sum())<1e-9
    print(f"  {name:11s} CTF: max rel err A {errA:.1e}  b {errb:.1e}   {_v(ok)}")

# ------------------------------------------------------------------------
print("\n"+"="*64); print("TEST 2: marginal log-likelihood == brute-force integral over (g, z)"); print("="*64)
# K=1: coefficients a=(g,z). Use a PROPER wide prior on g in both so the
# integral is finite; the code's flat-gain case is the limit.
K1=2; tau=torch.tensor([0.7]); pp=torch.tensor([0.05,1.0])
Am=torch.randn(K1,K1,dtype=torch.float64); Am=Am@Am.T+K1*torch.eye(K1,dtype=torch.float64)
bm=torch.randn(K1,dtype=torch.float64); Gyy=torch.tensor(5.0,dtype=torch.float64)
lm,mean,cov=marginal_loglik(Am[None,None,None],bm[None,None,None],Gyy,tau,pp)
g=torch.linspace(-12,12,1201,dtype=torch.float64); z=torch.linspace(-8,8,801,dtype=torch.float64)
G,Z=torch.meshgrid(g,z,indexing="ij"); a=torch.stack([G,Z],-1)
quad=(a@Am*a).sum(-1)-2*(a@bm)
logp=-0.5*tau*(Gyy+quad)-0.5*(pp[0]*G**2+pp[1]*Z**2)+0.5*torch.log(pp).sum()-math.log(2*math.pi)*K1/2
# the code drops the (2pi)^{-(K+1)/2}: restore it here, and the Gaussian
# integral's own (2pi)^{(K+1)/2} from the analytic form, so the two match
ref=torch.logsumexp(logp.reshape(-1),0)+math.log((g[1]-g[0])*(z[1]-z[0]))
pm=torch.exp(logp-logp.max()); pm=pm/pm.sum()
mean_ref=torch.stack([(pm*G).sum(),(pm*Z).sum()])
ok=abs(lm.item()-ref.item())<1e-6 and (mean[0,0,0]-mean_ref).abs().max()<1e-5
print(f"  analytic {lm.item():.6f} vs quadrature {ref.item():.6f}; posterior mean err "
      f"{(mean[0,0,0]-mean_ref).abs().max():.1e}   {_v(ok)}")
# negative control: perturbing A must change the answer (guards a constant function)
lm2,_,_=marginal_loglik(1.3*Am[None,None,None],bm[None,None,None],Gyy,tau,pp)
print(f"  responds to A: {_v(abs(lm2.item()-lm.item())>1e-3)}")
# log-det penalty: a bigger design (more template power) with the same fit must score lower
Ab=Am.clone(); Ab[1,1]*=4; bb=bm.clone(); bb[1]*=2   # same fitted z-contribution, 4x column norm
lm3,_,_=marginal_loglik(Ab[None,None,None],bb[None,None,None],Gyy,tau,pp)
print(f"  Occam term present (logml drops when the component is scaled up without "
      f"improving the fit): {lm3.item():.4f} < {lm.item():.4f}  {_v(lm3.item()<lm.item())}")

# ------------------------------------------------------------------------
print("\n"+"="*64); print("TEST 3: trilinear splat == explicit loop; K=0 accumulator == num/den"); print("="*64)
D=12; npts=500
pts=(torch.rand(npts,3,dtype=torch.float64)-0.5)*(D+2)   # some fall outside
vals=torch.randn(2,npts,dtype=torch.complex128)
out=torch.zeros(2,D**3,dtype=torch.complex128); insert_trilinear(pts,vals,D,out)
ref=torch.zeros(2,D,D,D,dtype=torch.complex128)
for q in range(npts):
    x=pts[q]+D//2
    if not ((x>=0)&(x<=D-1)).all(): continue
    f=torch.floor(x).long(); fr=x-f
    for dx in (0,1):
        for dy in (0,1):
            for dz in (0,1):
                wgt=(fr[0] if dx else 1-fr[0])*(fr[1] if dy else 1-fr[1])*(fr[2] if dz else 1-fr[2])
                i=(f+torch.tensor([dx,dy,dz])).clamp(0,D-1)
                ref[:,i[0],i[1],i[2]]+=vals[:,q]*wgt
err=(out.reshape(2,D,D,D)-ref).abs().max().item()
print(f"  splat vs loop: max err {err:.1e}   {_v(err<1e-12)}")
acc=HeteroAccumulator.new(D,1,"cpu")
y=torch.randn(npts,dtype=torch.complex128); c=torch.rand(npts,dtype=torch.float64); wq=torch.rand(npts,dtype=torch.float64)
acc.add(pts,y,c,wq,torch.ones(npts,1,dtype=torch.float64),torch.ones(npts,1,1,dtype=torch.float64))
num=torch.zeros(1,D**3,dtype=torch.complex128); den=torch.zeros(1,D**3,dtype=torch.float64)
insert_trilinear(pts,(wq*c*y)[None],D,num); insert_trilinear(pts,(wq*c**2)[None],D,den)
vol=acc.solve(lam=1e-3)[0].reshape(-1); refv=num[0]/(den[0]+1e-3*den[0].mean())
err=(vol-refv).abs().max().item()/refv.abs().max().item()
print(f"  K=0 solve vs num/(den+ridge): rel err {err:.1e}   {_v(err<1e-12)}")

# ------------------------------------------------------------------------
print("\n"+"="*64); print("TEST 4: K=1 with z=+-1 == two-class reconstruction, exactly"); print("="*64)
# Two classes with delta posteriors are a reparametrisation (V0 = (A+B)/2,
# U = (A-B)/2) of two independent homogeneous problems, so the per-voxel
# solve must reproduce the class-wise num/den on every voxel both classes touch.
npts=4000; D=12
pts=(torch.rand(npts,3,dtype=torch.float64)-0.5)*(D-1)
cls=torch.randint(0,2,(npts,)); zval=torch.where(cls==0,1.0,-1.0).to(torch.float64)
y=torch.randn(npts,dtype=torch.complex128); c=torch.rand(npts,dtype=torch.float64)+0.5; wq=torch.rand(npts,dtype=torch.float64)+0.5
Ea=torch.stack([torch.ones(npts,dtype=torch.float64),zval],1); Eaa=Ea[:,:,None]*Ea[:,None,:]
acc=HeteroAccumulator.new(D,2,"cpu"); acc.add(pts,y,c,wq,Ea,Eaa)
sol=acc.solve(lam=1e-12)
recon=[]
for k in (0,1):
    m=cls==k
    num=torch.zeros(1,D**3,dtype=torch.complex128); den=torch.zeros(1,D**3,dtype=torch.float64)
    insert_trilinear(pts[m],(wq*c*y)[m][None],D,num); insert_trilinear(pts[m],(wq*c**2)[m][None],D,den)
    recon.append((num[0],den[0]))
both=(recon[0][1]>1e-2)&(recon[1][1]>1e-2)
Va=recon[0][0][both]/recon[0][1][both]; Vb=recon[1][0][both]/recon[1][1][both]
V0=sol[0].reshape(-1)[both]; U=sol[1].reshape(-1)[both]
err=max((V0-(Va+Vb)/2).abs().max().item(),(U-(Va-Vb)/2).abs().max().item())/Va.abs().max().item()
print(f"  {both.sum().item()} voxels seen by both classes; rel err {err:.1e}   {_v(err<1e-9)}")

# ------------------------------------------------------------------------
print("\n"+"="*64); print("TEST 5: gauge fix leaves the volume family unchanged, whitens z"); print("="*64)
D=8; K=2; n=300
V0=torch.randn(D,D,D,dtype=torch.float64); U=torch.randn(K,D,D,D,dtype=torch.float64)
L=torch.tensor([[2.0,0.0],[1.5,0.5]],dtype=torch.float64)
z=torch.randn(n,K,dtype=torch.float64)@L.T+torch.tensor([1.0,-2.0])
V0n,Un,zn,(Q,mu)=fix_gauge(V0,U,z)
before=V0[None]+torch.einsum('nk,kxyz->nxyz',z,U)
after=V0n[None]+torch.einsum('nk,kxyz->nxyz',zn,Un)
err=(before-after).abs().max().item()/before.abs().max().item()
cov=zn.T@zn/n; white=(cov-torch.eye(K,dtype=torch.float64)).abs().max().item()
print(f"  V(z) unchanged: rel err {err:.1e};  cov(z') - I: {white:.1e};  mean(z') "
      f"{zn.mean(0).abs().max():.1e}   {_v(err<1e-10 and white<1e-10 and zn.mean(0).abs().max()<1e-10)}")
Qp,Ual=align_bases(Un,torch.einsum('jk,kxyz->jxyz',torch.tensor([[0.,1.],[-1.,0.]],dtype=torch.float64),Un))
print(f"  Procrustes recovers a basis rotation: err {(Ual-Un).abs().max():.1e}   {_v((Ual-Un).abs().max()<1e-10)}")

# ------------------------------------------------------------------------
print("\n"+"="*64); print("TEST 6: rotation-leakage diagnostic and the hinge phantom"); print("="*64)
D=24
vol=gaussian_atom_volume(D,device="cpu")
tang=rigid_tangents(vol)
# a genuine small rotation of the phantom, rendered exactly, minus the original
body,helix=_phantom_atoms(D); pos=torch.cat([body,helix],0)
eps=0.01; Rz=torch.tensor([[math.cos(eps),-math.sin(eps),0],[math.sin(eps),math.cos(eps),0],[0,0,1]],dtype=torch.float64)
dv=(_render_atoms(pos@Rz.T,D,device="cpu")-vol)/eps
leak_rot=rotation_leakage(vol,dv[None])[0].item()
leak_rand=rotation_leakage(vol,torch.randn(1,D,D,D,dtype=torch.float64))[0].item()
print(f"  leakage of a true rigid rotation: {leak_rot:.3f} (expect ~1);  of random noise: "
      f"{leak_rand:.4f} (expect ~0)   {_v(leak_rot>0.95 and leak_rand<0.05)}")
h0,r0=hinge_phantom(D,0.0,device="cpu"); h1,r1=hinge_phantom(D,math.radians(30),device="cpu")
same=torch.equal(h0,vol); moved=(h1-vol).abs().max()>0.1
body_still=(h1-vol).abs()[:, :, :D//2-int(0.35*D)-1].max()<1e-6 if D//2-int(0.35*D)-1>0 else True
print(f"  hinge(t=0) == gaussian_atom_volume bit-for-bit: {_v(same)};  t=30deg moves the arm "
      f"(rms {r1:.2f} vox, t=0 rms {r0:.2f}): {_v(moved and r0==0.0 and r1>1.0)}")
t=torch.linspace(-1,1,200,dtype=torch.float64); zz=torch.stack([t**3+0.02*torch.randn(200,dtype=torch.float64),torch.randn(200,dtype=torch.float64)],1)
print(f"  rank corr(t^3+small noise, t) {rank_correlation(zz[:,0],t):.3f}, R2 {latent_r2(zz,t):.3f}   "
      f"{_v(rank_correlation(zz[:,0],t)>0.9 and 0.5<latent_r2(zz,t)<1.0)}")

# ------------------------------------------------------------------------
print("\n"+"="*64); print("TEST 7: heterogeneous generator reproduces generate_physical (GPU)"); print("="*64)
if not GPU:
    print("  SKIPPED: no GPU (set CUDA_VISIBLE_DEVICES to run)")
else:
    from cryolike.grids import PolarGrid
    from dcryolike.generator import generate_physical, generate_physical_hetero
    D=32; vol=gaussian_atom_volume(D,device=dev)
    pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=64,uniform=True)
    p0=generate_physical(vol,pg,24,0.1,seed=3,device=dev,ctf_kwargs=dict(astig=0.0))
    p1=generate_physical_hetero([vol],[0.0],torch.zeros(24,dtype=torch.long),pg,0.1,seed=3,device=dev,ctf_kwargs=dict(astig=0.0))
    # clean signal, CTF and poses are bit-identical; the noise scale differs at
    # ~1e-15 relative because the projection variance is computed on a subset
    # copy (different NUFFT batch), so the images are compared to 1e-12
    same=(torch.equal(p0.clean_fourier,p1.clean_fourier) and torch.equal(p0.ctf,p1.ctf)
          and torch.allclose(p0.images_fourier,p1.images_fourier,rtol=1e-12,atol=0))
    print(f"  homogeneous limit reproduces generate_physical (clean/CTF exact, images to 1e-12): {_v(same)}")
    v1,_=hinge_phantom(D,math.radians(30),device=dev)
    conf=torch.arange(24,device=dev)%2
    p2=generate_physical_hetero([vol,v1],[0.0,math.radians(30)],conf,pg,0.1,seed=3,device=dev,ctf_kwargs=dict(astig=0.0))
    # same poses, same noise draws; particles from conformation 0 identical, from 1 different
    e0=(p2.clean_fourier[conf==0]-p0.clean_fourier[conf==0]).abs().max().item()
    e1=(p2.clean_fourier[conf==1]-p0.clean_fourier[conf==1]).abs().max().item()
    print(f"  conf 0 clean identical ({e0:.1e}), conf 1 differs ({e1:.2e}): {_v(e0==0.0 and e1>1e-3)}")
    # SNR realised per conformation (noise set from each particle's own projection)
    print(f"  t recorded per particle: {_v(torch.allclose(p2.t[conf==1],torch.tensor(math.radians(30),dtype=p2.t.dtype,device=dev)))}")

print("\n"+("ALL PASS" if all(_FAILS) else "SOME FAIL"), f"({sum(_FAILS)}/{len(_FAILS)})")
sys.exit(0 if all(_FAILS) else 1)
