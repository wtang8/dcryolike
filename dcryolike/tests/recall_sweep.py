"""GO/NO-GO: recall@K vs SNR, using the EXACT likelihood (matched filter).

This establishes the CEILING. No network is involved. If the optimal estimator
cannot put a pose within tol of truth in its top K, no learned proposal can.

Success@K = at least one of the top-K scoring viewing directions lies within
`tol` of the true viewing direction. Score for a direction is the max over
in-plane rotation, computed for all rotations at once by FFT along theta
(requires isotropic CTF so ||T|| is gamma-independent).
"""
import sys, time, numpy as np, torch
sys.path.insert(0, "/home/wtang/cryoem/dcryolike"); sys.path.insert(0, "/home/wtang/cryoem/cryolike/src")
from cryolike.grids import PolarGrid
from dcryolike.generator import gaussian_atom_volume, generate, uniform_so3
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft

dev = "cuda"; D = 48
# Angular sampling must resolve the outermost shell: a rotation of dtheta moves
# the sample at radius k_max by k_max*dtheta voxels. With k_max = D/2 = 24,
# dtheta ~ 1/24 rad ~ 2.4 deg. A coarser grid measures grid coverage, not SNR.
TOL_DEG = 3.0
VIEW_SPACING = 1.0 / (D / 2)      # radians
N_TEST = 200
SNRS = [1.0, 0.5, 0.1, 0.05, 0.02, 0.01]
KS = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000]
ISO_CTF = dict(astig=0.0)          # isotropic: template norm independent of gamma

vol = gaussian_atom_volume(D, device=dev)
pg = PolarGrid(radius_max=D/2, dist_radii=1.0, n_inplanes=128, uniform=True)
S, P = pg.n_shells, pg.n_inplanes
th = torch.tensor(pg.theta_shell, dtype=vol.dtype, device=dev)
rr = torch.tensor(pg.radius_shells, dtype=vol.dtype, device=dev)
w  = torch.tensor(pg.integration_weight_points, dtype=vol.dtype, device=dev).reshape(S, P)
wr = w[:, :1]                      # uniform grid: weights depend only on shell

# ---- candidate bank: regular sphere grid, gamma = 0 ------------------------
from cryolike.metadata import ViewingAngles
va = ViewingAngles.from_viewing_distance(VIEW_SPACING)
caz = va.azimus.to(dev).to(vol.dtype); cpo = va.polars.to(dev).to(vol.dtype)
N_VIEW = caz.shape[0]
print(f"candidate bank: {N_VIEW} views at {np.degrees(VIEW_SPACING):.2f} deg spacing")
zero = torch.zeros_like(caz)
Tb = slice_nufft(vol, slice_points(fourier_circles(th, cpo, caz, zero), rr)).reshape(N_VIEW, S, P)
ndir = torch.stack([torch.sin(cpo)*torch.cos(caz), torch.sin(cpo)*torch.sin(caz), torch.cos(cpo)], -1)

def score_all(imgs, ctf):
    """-> (B, N_VIEW) best normalised cross-correlation over in-plane rotation.

    Each particle has its OWN defocus, so templates cannot be CTF-modulated
    once and shared. An isotropic CTF depends only on radius, so it folds into
    the per-shell contraction instead -- exact, and avoids materialising a
    (B, V, S, P) template bank.
    """
    ctf_r = ctf[:, :, 0]                                  # (B,S) isotropic
    wr_ = w[:, 0]                                         # (S,)
    Tf = torch.fft.fft(Tb, dim=-1).conj()                 # (V,S,P)
    If = torch.fft.fft(imgs, dim=-1)                      # (B,S,P)
    Tpow = (Tb.abs()**2).sum(dim=-1)                      # (V,S), gamma-independent
    Tn = torch.sqrt(torch.einsum('vr,br->bv', Tpow, wr_[None,:]*ctf_r**2))
    In = torch.sqrt((imgs.abs()**2 * w).sum(dim=(1,2)))
    out = torch.empty(imgs.shape[0], N_VIEW, device=dev, dtype=vol.dtype)
    for i in range(0, imgs.shape[0], 8):
        j = min(i+8, imgs.shape[0])
        cw = (wr_[None,:] * ctf_r[i:j])[:, :, None]       # (b,S,1)
        pk = torch.einsum('vrk,brk->bvk', Tf, If[i:j] * cw)
        cc = torch.fft.ifft(pk, dim=-1).real
        out[i:j] = cc.amax(dim=-1) / (Tn[i:j] * In[i:j,None])
    return out

# ---- sanity: FFT in-plane search agrees with brute force -------------------
p0 = generate(vol, pg, 2, 100.0, seed=3, device=dev, ctf_kwargs=ISO_CTF)
sc = score_all(p0.images_fourier, p0.ctf)
NG = 64   # brute-force gamma samples; must be dense enough to find the peak
bf = torch.empty(2, 12, NG, device=dev, dtype=vol.dtype)   # (image, view, gamma)
In0 = torch.sqrt((p0.images_fourier.abs()**2*w).sum(dim=(1,2)))
for gi, g in enumerate(torch.linspace(0, 2*np.pi, NG+1, device=dev, dtype=vol.dtype)[:-1]):
    Tg = slice_nufft(vol, slice_points(fourier_circles(th, cpo[:12], caz[:12],
            g.repeat(12)), rr)).reshape(12, S, P)
    Tc = Tg[None] * p0.ctf[:, None]                      # (img, view, S, P)
    num = (Tc.conj() * p0.images_fourier[:,None] * w).sum(dim=(2,3)).real
    bf[:, :, gi] = num / (torch.sqrt((Tc.abs()**2*w).sum(dim=(2,3))) * In0[:,None])
err = (sc[:, :12] - bf.amax(dim=2)).abs().max()
print(f"FFT in-plane search vs brute force: max dev {err:.2e} "
      f"{'(consistent)' if err < 5e-3 else '(MISMATCH)'}\n")

# ---- the sweep -------------------------------------------------------------
print(f"exact-likelihood ceiling | {N_VIEW} views, {N_TEST} particles/SNR, "
      f"tol={TOL_DEG} deg, D={D}")
print(f"(a {TOL_DEG} deg cap covers "
      f"{100*(1-np.cos(np.radians(TOL_DEG)))/2:.3f}% of the sphere)")
# grid coverage ceiling: fraction of random true directions with ANY candidate
# within tol. If this is not ~100%, the sweep measures the grid, not the SNR.
_az,_po,_ = uniform_so3(4000, seed=999, device=dev, dtype=vol.dtype)
_n = torch.stack([torch.sin(_po)*torch.cos(_az), torch.sin(_po)*torch.sin(_az), torch.cos(_po)],-1)
_cov = (torch.rad2deg(torch.arccos((_n @ ndir.T).clamp(-1,1))).min(1).values <= TOL_DEG)
print(f"grid coverage ceiling: {_cov.float().mean().item()*100:.1f}% "
      f"(must be ~100% or the experiment is grid-limited)\n")
hdr = "  SNR  " + "".join(f"{('@'+str(k)):>8}" for k in KS) + "  medrank  top1err"
print(hdr); print("-"*len(hdr))
t0 = time.time()
for snr in SNRS:
    p = generate(vol, pg, N_TEST, snr, seed=17, device=dev, ctf_kwargs=ISO_CTF)
    sc = score_all(p.images_fourier, p.ctf)
    ntrue = torch.stack([torch.sin(p.polars)*torch.cos(p.azimus),
                         torch.sin(p.polars)*torch.sin(p.azimus),
                         torch.cos(p.polars)], -1)
    ang = torch.rad2deg(torch.arccos((ntrue @ ndir.T).clamp(-1, 1)))   # (B, N_VIEW)
    order = sc.argsort(dim=1, descending=True)
    hit = (torch.gather(ang, 1, order) <= TOL_DEG)
    # rank of the first hit (N_VIEW if never)
    first = torch.where(hit.any(1), hit.float().argmax(1), torch.full((N_TEST,), N_VIEW, device=dev))
    top1 = torch.gather(ang, 1, order[:, :1]).squeeze(1)
    row = f"  {snr:<5.2f}" + "".join(f"{(first < k).float().mean().item()*100:7.1f}%" for k in KS)
    print(row + f"{first.median().item():8.0f}{top1.median().item():9.1f}")
print(f"\nelapsed {time.time()-t0:.1f}s")
