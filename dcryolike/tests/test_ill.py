"""Does CryoLike's integrated (noise/gain/offset-marginalised) log likelihood work?

The model is  y = gain*x + offset*s + noise(sigma),  with gain, offset and sigma
all integrated out. The defining property is that the RANKING of templates must
be exactly invariant to an arbitrary gain and offset applied to the image --
because those are precisely the parameters being marginalised away.

Normalised cross-correlation is the control: it is gain-invariant but NOT
offset-invariant, so it should break where the ILL does not.
"""
import sys, os, numpy as np, torch
import _paths  # noqa: F401  (sets sys.path relative to this file)
from cryolike.grids import PolarGrid
from cryolike.metadata import ViewingAngles
from cryolike.util import fourier_bessel_transform
from cryolike.likelihoods.kernels.cross_correlation_likelihood_kernel import (
    compute_cross_correlation, compute_template_norms, compute_image_norms)
from cryolike.likelihoods.kernels.integrated_log_likelihood_kernel import ill_kernel
from dcryolike.ill import ill_kernel_pergamma
from dcryolike.generator import gaussian_atom_volume, make_ctf
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft

dev="cuda"; D=48; NT=150; SNR=0.5
vol=gaussian_atom_volume(D,device=dev)
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=128,uniform=True)
S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=vol.dtype,device=dev)
rr=torch.tensor(pg.radius_shells,dtype=vol.dtype,device=dev)
va=ViewingAngles.from_viewing_distance(np.radians(14.0))
caz=va.azimus.to(dev).to(vol.dtype)[:NT]; cpo=va.polars.to(dev).to(vol.dtype)[:NT]
NT=caz.shape[0]
T=slice_nufft(vol,slice_points(fourier_circles(th,cpo,caz,torch.zeros_like(caz)),rr)).reshape(NT,S,P)

MASK=os.environ.get("MASK","default")
if MASK=="radial":
    # radially symmetric analogue of the default: depends on |k| only, so
    # <s, R_gamma x> is genuinely independent of the in-plane rotation
    pg.set_mask_points(lambda x,y: 4.0*np.sinc(2.0*np.sqrt(x**2+y**2)))
iw=torch.sqrt(torch.tensor(pg.integration_weight_points,dtype=torch.float64,device=dev)
              ).reshape(S,P)
mask=torch.tensor(pg.mask_points,dtype=torch.float64,device=dev).reshape(S,P)
sqrt_mask=mask*iw
Iss=float(pg.mask_integral)
npix=D*D                          # n_pixels_phys is REAL-SPACE pixels
ctf=make_ctf(pg,1,float(D),device=dev,dtype=torch.float64,astig=0.0)

TRUE=7
g=torch.Generator(device=dev).manual_seed(4)
clean=(T[TRUE]*ctf[0])[None]
pw=(clean.abs()**2*iw**2).sum(dim=(1,2))/ (iw**2).sum()
img0=clean+torch.randn(clean.shape,generator=g,dtype=clean.dtype,device=dev)*torch.sqrt(pw/SNR)

def ill_and_cc(img):
    t_snw=T*iw[None]
    i_mnw=img*iw[None]
    Ixx=compute_template_norms(P,t_snw,ctf).unsqueeze(2)
    Iyy=compute_image_norms(i_mnw)
    # weights applied ONCE, matching template_first_comparator
    t_b=fourier_bessel_transform(t_snw[:,None])                     # (S,1,n,q)
    i_b=fourier_bessel_transform(i_mnw*ctf).conj()
    Ixy=compute_cross_correlation(P,i_b,t_b)
    args=(Iss,npix,sqrt_mask,(t_snw*ctf[:,None]),i_mnw,Ixx,Iyy,Ixy)
    orig=ill_kernel(*args)
    corr=ill_kernel_pergamma(*args,per_gamma=True)
    cc=(Ixy/torch.sqrt(Ixx*Iyy)).amax(dim=(2,3))
    return orig[0], corr[0], cc[0]

o0,c0,cc0=ill_and_cc(img0)
print(f"{NT} templates, true index {TRUE}, SNR {SNR}, n_pixels={npix}, mask={MASK}")
print(f"  argmax -- original {o0.argmax().item()} | corrected {c0.argmax().item()} "
      f"| CC {cc0.argmax().item()}  (true {TRUE})")
ok = torch.isfinite(c0).all().item() and c0.argmax().item()==TRUE
print(f"  identifies the true template: {'PASS' if ok else 'FAIL'}\n")

print(f"{'gain':>7}{'offset':>9}{'ORIGINAL corr':>15}{'argmax':>8}"
      f"{'CORRECTED corr':>16}{'argmax':>8}{'CC corr':>10}")
print("-"*76)
def spearman(a,b):
    ra=a.argsort().argsort().double(); rb=b.argsort().argsort().double()
    ra=ra-ra.mean(); rb=rb-rb.mean()
    return (ra@rb/(ra.norm()*rb.norm())).item()
for gain,off in [(1.0,0.0),(3.7,0.0),(0.21,0.0),(1.0,5.0),(1.0,-12.0),(2.5,8.0),(0.4,-30.0)]:
    img=gain*img0+off*mask[None].to(img0.dtype)
    o,c,cc=ill_and_cc(img)
    so,sc_,scc=spearman(o0,o),spearman(c0,c),spearman(cc0,cc)
    ok &= (sc_>0.99999) and (c.argmax().item()==TRUE)
    print(f"{gain:>7.2f}{off:>9.1f}{so:>15.6f}{o.argmax().item():>8}"
          f"{sc_:>16.6f}{c.argmax().item():>8}{scc:>10.4f}")
print(f"\n  CORRECTED ranking invariant to gain+offset AND still correct: "
      f"{'PASS' if ok else 'FAIL'}")
import sys as _sys; _sys.exit(0 if ok else 1)
