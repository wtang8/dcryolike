"""Project the fitted component (per half, per band) onto the first three TRUE
principal components of the conformational family, from SAVE= dumps.

A hinge is nonlinear in its angle, so the family has PC2 (even in the angle,
uncorrelated with a symmetric latent) and PC3 (odd, correlated with it). A
single linear component fitted by regression on z legitimately absorbs PC3;
if the fitted component's norm grows in a direction orthogonal to PC1 that
both halves agree on, this tells whether that direction is PC2/PC3 (model
capacity) or something else (an artefact).
Usage: python hetero_pc_analysis.py <dumpdir>
"""
import sys, glob, os, math, numpy as np, torch
import _paths  # noqa: F401
from dcryolike.generator import hinge_phantom
from dcryolike.slicing import fourier_volume
from dcryolike.pipeline import ShellMasks
d=sys.argv[1]; files=sorted(glob.glob(os.path.join(d,"state_it*.pt"))); dev="cuda"
D=96; NCONF=16; SW=math.radians(15); ts=np.linspace(-SW,SW,NCONF)
vols=[hinge_phantom(D,float(t),device=dev)[0] for t in ts]; cons=torch.stack(vols).mean(0)
X=torch.stack([v-cons for v in vols]).reshape(NCONF,-1); _,Sv,Vh=torch.linalg.svd(X,full_matrices=False)
print("family variance explained by PC1..4:", " ".join(f"{v:.3f}" for v in (Sv[:4]**2/(Sv**2).sum()).tolist()))
zt=X@Vh[0]; PCs=[]
for k in range(3):
    coef=X@Vh[k]; PCs.append(fourier_volume((Vh[k]*coef.std()).reshape(D,D,D)))   # unit-variance coefficient scaling
sh=ShellMasks(D,dev); bands=[(1,8),(9,14),(15,20),(21,24)]
def bandsum(F,G,lo,hi):
    m=(sh.RR>=lo-0.5)&(sh.RR<hi+0.5); return (F[m]*G[m].conj()).real.sum().item()
print(f"{'it':>3} {'half':>4} {'band':>6} {'|U|^2/|PC1|^2':>13} {'frac PC1':>9} {'frac PC2':>9} {'frac PC3':>9} {'frac rest':>9}")
for f in files:
    st=torch.load(f); it=st["it"]
    if it not in (2,3,5,8): continue
    for h in (0,1):
        FU=fourier_volume(st["U"][h][0].to(dev))
        for lo,hi in bands:
            pu=bandsum(FU,FU,lo,hi); fr=[]
            for k in range(3):
                pk=bandsum(PCs[k],PCs[k],lo,hi); c=bandsum(FU,PCs[k],lo,hi)
                fr.append(c*c/(pk*pu+1e-30))                 # fraction of U's band power along PC_k
            print(f"{it:>3} {h:>4} {f'{lo}-{hi}':>6} {pu/bandsum(PCs[0],PCs[0],lo,hi):13.3f} {fr[0]:9.3f} {fr[1]:9.3f} {fr[2]:9.3f} {max(0.0,1-sum(fr)):9.3f}")
    print()
