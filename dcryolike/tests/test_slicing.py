"""Correctness tests for the differentiable slicing operators.

The central-slice-theorem test is evaluated at *integer* grid frequencies,
where trilinear interpolation lands exactly on volume nodes. Any discrepancy
there is therefore a convention bug (shift, axis order, normalisation), not
interpolation error -- which is what we actually want to catch.
"""
import sys, torch, numpy as np
import _paths  # noqa: F401
from dcryolike.slicing import fourier_volume, slice_gridding, fourier_circles
import sys as _sys
_FAILS=[]
def _v(c,label=''):
    _FAILS.append(bool(c))
    return 'PASS' if c else 'FAIL'

torch.manual_seed(0)
D = 32
dev = "cuda"


def smooth_density(D, dev):
    v = torch.randn(D, D, D, dtype=torch.float64, device=dev)
    k = torch.fft.fftfreq(D, device=dev, dtype=torch.float64)
    kk = (k[:,None,None]**2 + k[None,:,None]**2 + k[None,None,:]**2).sqrt()
    v = torch.fft.ifftn(torch.fft.fftn(v) * torch.exp(-30.0 * kk**2)).real
    return v.contiguous()


print("=" * 62)
print("TEST 1: central slice theorem (z=0 plane, integer frequencies)")
print("=" * 62)
dens = smooth_density(D, dev)
fvol = fourier_volume(dens)

# z-projection -> its 2D transform must equal the z=0 central slice
proj = dens.sum(dim=2)
ref2d = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(proj), norm="ortho"))
ref2d = ref2d / np.sqrt(D)          # 3D ortho vs 2D ortho normalisation

f = torch.arange(D, dtype=torch.float64, device=dev) - D // 2
fx, fy = torch.meshgrid(f, f, indexing="ij")
pts = torch.stack((fx.reshape(-1), fy.reshape(-1),
                   torch.zeros_like(fx.reshape(-1))), dim=-1)[None]

got = slice_gridding(fvol, pts).reshape(D, D)
err = (got - ref2d).abs().max().item() / ref2d.abs().max().item()
print(f"  max relative error : {err:.3e}")
print(f"  verdict            : {_v(err < 1e-10)}")

# localise the failure if there is one
if err >= 1e-10:
    direct = fvol[:, :, D // 2]
    e2 = (direct - ref2d).abs().max().item() / ref2d.abs().max().item()
    print(f"  -> direct fvol[:,:,D//2] vs ref2d  : {e2:.3e}"
          f"  ({'fvol/normalisation OK' if e2 < 1e-10 else 'fvol or norm WRONG'})")
    e3 = (got - direct).abs().max().item() / direct.abs().max().item()
    print(f"  -> grid_sample vs direct indexing  : {e3:.3e}"
          f"  ({'indexing OK' if e3 < 1e-10 else 'COORDINATE MAPPING WRONG'})")
    # is it a pure integer-voxel shift?
    for s in (-1, 1):
        for ax in (0, 1):
            r = (got - torch.roll(direct, s, dims=ax)).abs().max().item()
            if r / direct.abs().max().item() < 1e-10:
                print(f"  -> exactly a roll of {s:+d} along axis {ax}")


print()
print("=" * 62)
print("TEST 2: fourier_circles geometry (polars=0 must give the z=0 plane)")
print("=" * 62)
th = torch.linspace(0, 2 * np.pi, 8, dtype=torch.float64, device=dev)[:-1]
z = torch.zeros(1, dtype=torch.float64, device=dev)
c = fourier_circles(th, z, z, z)
print(f"  max |z| on circle  : {c[...,2].abs().max().item():.3e}")
print(f"  radius deviation   : {(c.norm(dim=-1) - 1).abs().max().item():.3e}")
print(f"  verdict            : {_v(c[...,2].abs().max() < 1e-12)}")


print()
print("=" * 62)
print("TEST 3: adjoint identity  <A x, y> == <x, A^H y>   (A^H via autograd)")
print("=" * 62)
x = smooth_density(D, dev).requires_grad_(True)
npts = 500
p = (torch.rand(1, npts, 3, dtype=torch.float64, device=dev) - 0.5) * (D - 2)
y = torch.randn(1, npts, dtype=torch.complex128, device=dev)

# A maps R^n -> C^m, so the adjoint in the *real* inner product is Re(A^H y),
# which is exactly what autograd returns for a real-valued input.
Ax = slice_gridding(fourier_volume(x), p)
lhs = (Ax.conj() * y).sum().real
(AHy,) = torch.autograd.grad(Ax, x, grad_outputs=y, create_graph=False)
rhs = (x * AHy).sum()
rel = abs(lhs.item() - rhs.item()) / abs(lhs.item())
print(f"  Re<Ax,y>  = {lhs.item():.10e}")
print(f"  <x,A^T y> = {rhs.item():.10e}")
print(f"  relative mismatch  : {rel:.3e}")
print(f"  verdict            : {_v(rel < 1e-10)}")


print()
print("=" * 62)
print("TEST 4: gradcheck of density -> slice")
print("=" * 62)
Ds = 12
xs = smooth_density(Ds, dev).requires_grad_(True)
ps = (torch.rand(1, 24, 3, dtype=torch.float64, device=dev) - 0.5) * (Ds - 3)
def fn(v):
    return slice_gridding(fourier_volume(v), ps)
ok = torch.autograd.gradcheck(fn, (xs,), eps=1e-6, atol=1e-7, rtol=1e-4)
print(f"  verdict            : {_v(ok)}")


print()
print("=" * 62)
print("TEST 5: nufft backend vs exact DFT and vs gridding backend")
print("=" * 62)
from dcryolike.slicing import slice_nufft
Dn = 24
dn = smooth_density(Dn, dev).to(torch.float64)
pn = (torch.rand(2, 300, 3, dtype=torch.float64, device=dev) - 0.5) * (Dn - 4)

ii = torch.arange(Dn, dtype=torch.float64, device=dev) - Dn // 2
GX, GY, GZ = torch.meshgrid(ii, ii, ii, indexing="ij")
rr = torch.stack([GX.ravel(), GY.ravel(), GZ.ravel()], -1)
exact = (torch.exp((-2j*np.pi/Dn) * (pn.reshape(-1,3) @ rr.T).to(torch.complex128))
         @ dn.ravel().to(torch.complex128)).reshape(2,-1) / Dn**1.5

nu = slice_nufft(dn, pn)
gr = slice_gridding(fourier_volume(dn), pn)
e_nu = ((nu-exact).abs().max()/exact.abs().max()).item()
e_gr = ((gr-exact).abs().max()/exact.abs().max()).item()
print(f"  nufft    vs exact DFT : {e_nu:.3e}   {_v(e_nu < 1e-5)}")
print(f"  gridding vs exact DFT : {e_gr:.3e}   (interpolation error, for scale)")


print()
print("=" * 62)
print("TEST 6: NufftSlice.backward vs the EXACT adjoint (explicit matrix)")
print("=" * 62)
# Compare against the true adjoint built as an explicit matrix. Comparing to
# the gridding backend instead would only measure gridding's interpolation
# error (cosine ~0.85), which says nothing about whether backward is correct.
Dg, Mg = 12, 100
dg = torch.randn(Dg, Dg, Dg, dtype=torch.float64, device=dev)
pg = (torch.rand(1, Mg, 3, dtype=torch.float64, device=dev) - 0.5) * (Dg - 4)
cot = torch.randn(1, Mg, dtype=torch.complex128, device=dev)
ig = torch.arange(Dg, dtype=torch.float64, device=dev) - Dg // 2
AX, AY, AZ = torch.meshgrid(ig, ig, ig, indexing="ij")
rg = torch.stack([AX.ravel(), AY.ravel(), AZ.ravel()], -1)
Amat = torch.exp((-2j*np.pi/Dg) * (pg[0] @ rg.T).to(torch.complex128)) / Dg**1.5
exact_grad = torch.real(Amat.conj().T @ cot[0]).reshape(Dg, Dg, Dg)

xa = dg.clone().requires_grad_(True)
(ga,) = torch.autograd.grad(slice_nufft(xa, pg), xa, grad_outputs=cot)
rel = ((ga - exact_grad).norm()/exact_grad.norm()).item()
cos = (ga.ravel()@exact_grad.ravel()/(ga.norm()*exact_grad.norm())).item()
print(f"  relative difference : {rel:.3e}")
print(f"  cosine similarity   : {cos:.10f}")
print(f"  verdict             : {_v(rel < 1e-5)}")

_sys.exit(0 if all(_FAILS) else 1)   # exit code so a single run is meaningful
