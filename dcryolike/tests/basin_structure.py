"""Are the competing pose basins (a) few, (b) comparable in score, and
(c) structured or random? This decides whether a multi-basin mixture is worth it.

A basin = a local maximum of the score over the viewing-direction grid
(higher than every view within NEIGH degrees).
"""
import sys, numpy as np, torch
import _paths  # noqa: F401  (sets sys.path relative to this file)
from cryolike.grids import PolarGrid
from cryolike.metadata import ViewingAngles
from dcryolike.generator import gaussian_atom_volume, generate
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft

dev="cuda"; D=48; N=300; ISO=dict(astig=0.0); NEIGH=10.0; TOL=5.0
vol=gaussian_atom_volume(D,device=dev)
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=128,uniform=True)
S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=vol.dtype,device=dev)
rr=torch.tensor(pg.radius_shells,dtype=vol.dtype,device=dev)
w=torch.tensor(pg.integration_weight_points,dtype=vol.dtype,device=dev).reshape(S,P); wr=w[:,0]
va=ViewingAngles.from_viewing_distance(np.radians(4.0))
caz=va.azimus.to(dev).to(vol.dtype); cpo=va.polars.to(dev).to(vol.dtype); V=caz.shape[0]
Tb=slice_nufft(vol,slice_points(fourier_circles(th,cpo,caz,torch.zeros_like(caz)),rr)).reshape(V,S,P)
nd=torch.stack([torch.sin(cpo)*torch.cos(caz),torch.sin(cpo)*torch.sin(caz),torch.cos(cpo)],-1)
print(f"{V} views at 4 deg spacing")
cosN=np.cos(np.radians(NEIGH))
NB=(nd@nd.T)>=cosN                       # neighbour mask on the sphere

def scores(imgs,ctf):
    cr=ctf[:,:,0]; Tf=torch.fft.fft(Tb,dim=-1).conj(); If=torch.fft.fft(imgs,dim=-1)
    Tn=torch.sqrt(torch.einsum('vr,br->bv',(Tb.abs()**2).sum(-1),wr[None,:]*cr**2))
    In=torch.sqrt((imgs.abs()**2*w).sum(dim=(1,2)))
    o=torch.empty(imgs.shape[0],V,device=dev,dtype=vol.dtype)
    for i in range(0,imgs.shape[0],8):
        j=min(i+8,imgs.shape[0]); cw=(wr[None,:]*cr[i:j])[:,:,None]
        cc=torch.fft.ifft(torch.einsum('vrk,brk->bvk',Tf,If[i:j]*cw),dim=-1).real
        o[i:j]=cc.amax(-1)/(Tn[i:j]*In[i:j,None])
    return o

for snr in (0.1,0.05,0.02):
    p=generate(vol,pg,N,snr,seed=17,device=dev,ctf_kwargs=ISO)
    nt=torch.stack([torch.sin(p.polars)*torch.cos(p.azimus),
                    torch.sin(p.polars)*torch.sin(p.azimus),torch.cos(p.polars)],-1)
    ang=torch.rad2deg(torch.arccos((nt@nd.T).clamp(-1,1)))
    sc=scores(p.images_fourier,p.ctf)
    # local maxima: score >= max over neighbours
    nbmax=torch.stack([sc[:,NB[v]].max(1).values for v in range(V)],1)
    ismax=sc>=nbmax-1e-12
    rank_of_true=[];ratio=[];nmodes=[];top1_wrong_ang=[];modes_to_truth=[]
    for b in range(N):
        idx=torch.nonzero(ismax[b]).squeeze(1)
        s=sc[b,idx]; order=idx[s.argsort(descending=True)]
        nmodes.append(len(order))
        a=ang[b,order]
        hit=(a<=TOL).nonzero()
        top=sc[b,order[0]]
        if len(hit):
            r=hit[0,0].item(); rank_of_true.append(r)
            ratio.append((sc[b,order[r]]/top).item())
            modes_to_truth.append(r+1)
        else:
            rank_of_true.append(-1); ratio.append(np.nan); modes_to_truth.append(-1)
        if ang[b,order[0]]>20: top1_wrong_ang.append(ang[b,order[0]].item())
    rk=np.array(rank_of_true); rt=np.array(ratio); nm=np.array(nmodes)
    found=rk>=0
    print(f"\n=== SNR {snr} ===")
    print(f"  local maxima per particle (median)      : {np.median(nm):.0f}")
    print(f"  top-1 mode is correct                   : {100*np.mean(rk==0):.0f}%")
    print(f"  truth is SOME mode (within {TOL:.0f} deg)      : {100*found.mean():.0f}%")
    for m in (2,3,5,10):
        print(f"  truth within top-{m:<2d} modes                : "
              f"{100*np.mean((rk>=0)&(rk<m)):.0f}%")
    sub=rt[(rk>0)&~np.isnan(rt)]
    if len(sub):
        print(f"  when truth is NOT top-1, its score/top   : "
              f"median {np.median(sub):.3f}, 10th pct {np.percentile(sub,10):.3f}")
    twa=np.array(top1_wrong_ang)
    if len(twa):
        print(f"  wrong top-1 (>20 deg): n={len(twa)}, angle to truth "
              f"median {np.median(twa):.0f} deg, frac >150 deg (antipodal) "
              f"{100*np.mean(twa>150):.0f}%, uniform-sphere expectation 3%")
