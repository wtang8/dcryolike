"""Per-shell anatomy of a fitted component, from the SAVE= dumps of hetero_linear_pilot.py.

For each saved iteration and each shell k:
  S_true   power of the true first principal component (unit-variance latent)
  S_hat    cross-power of the two halves' components  Re<U_a,U_b>  (the part both halves agree on)
  N_hat    (|U_a|^2+|U_b|^2)/2 - S_hat                  (the part they do not)
  corr     correlation of (U_a+U_b)/2 with the true PC
  amp      cross-power of the halves' V(zbar_c)-V0 difference (family amplitude) over S_true, at the extreme conformation
The question: as iterations go on, where does the fitted family gain energy relative to the truth?
Usage: python hetero_shell_analysis.py <dumpdir>
"""
import sys, glob, os, numpy as np, torch
import _paths  # noqa: F401
from dcryolike.slicing import fourier_volume
from dcryolike.pipeline import ShellMasks

d=sys.argv[1]; files=sorted(glob.glob(os.path.join(d,"state_it*.pt")))
dev="cuda" if torch.cuda.is_available() else "cpu"
first=torch.load(files[0]); D=first["U_TRUE"].shape[-1]; sh=ShellMasks(D,dev)
Ft=fourier_volume(first["U_TRUE"].to(dev)); St=np.array([(Ft[m].abs()**2).mean().item() for m in sh.masks])
bands=[(1,8),(9,14),(15,20),(21,24),(25,32)]
def band(v,lo,hi): return float(np.mean(v[lo-1:hi]))
print(f"{'it':>3} {'band':>7} {'S_hat/S_true':>13} {'N_hat/S_true':>13} {'corr':>6} {'amp_a/true':>11} {'amp_b/true':>11}")
for f in files:
    st=torch.load(f); it=st["it"]
    Ua,Ub=st["U"][0][0].to(dev),st["U"][1][0].to(dev)
    Fa,Fb=fourier_volume(Ua),fourier_volume(Ub)
    sgn=torch.sign((Fa*Fb.conj()).sum().real).item() or 1.0; Fb=Fb*sgn
    S=np.array([(Fa[m]*Fb[m].conj()).real.mean().item() for m in sh.masks])
    Pm=np.array([((Fa[m].abs()**2).mean().item()+(Fb[m].abs()**2).mean().item())/2 for m in sh.masks])
    Fm=(Fa+Fb)/2
    corr=np.array([((Fm[m]*Ft[m].conj()).sum()/(Fm[m].abs().pow(2).sum().sqrt()*Ft[m].abs().pow(2).sum().sqrt()+1e-30)).real.item() for m in sh.masks])
    # family amplitude at the extreme conformation: (V(zbar_last)-V0) per half vs true (U_TRUE * z_true_last)
    amps=[]
    for h in (0,1):
        zb=st["zbars"][h][-1,0].item(); Fd=fourier_volume((zb*st["U"][h][0]).to(dev))
        amps.append(np.array([((Fd[m]*Ft[m].conj()).real.sum()/(Ft[m].abs()**2).sum()).item() for m in sh.masks]))
    ztrue_last=1.65   # unit-variance latent at the end of a uniform +-swing over 16 bins
    for lo,hi in bands:
        print(f"{it:>3} {f'{lo}-{hi}':>7} {band(S,lo,hi)/band(St,lo,hi):13.3f} {band(Pm-S,lo,hi)/band(St,lo,hi):13.3f} {band(corr,lo,hi):6.3f} "
              f"{band(amps[0],lo,hi)/ztrue_last:11.3f} {band(amps[1],lo,hi)/ztrue_last:11.3f}")
    print()
