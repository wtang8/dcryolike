"""REVIEW CHECK: does refine.py degrade near the coordinate poles?

refine_poses optimises (azimu, polar, gamma) as free Euler angles with Adam.
At polar ~ 0 or pi the azimuth is degenerate (a coordinate singularity), so
gradients w.r.t. azimu vanish and the parameterisation can stall or spin.
The capture-radius and precision tests drew poses uniformly, so ~2% of
particles were near a pole -- not enough to show up in a median.
"""
import _paths  # noqa
import numpy as np, torch
from cryolike.grids import PolarGrid
from dcryolike.generator import gaussian_atom_volume, make_ctf
from dcryolike.slicing import fourier_circles, slice_points, slice_nufft
from dcryolike.refine import refine_poses

dev="cuda"; D=48; N=96
vol=gaussian_atom_volume(D,device=dev)
pg=PolarGrid(radius_max=D/2,dist_radii=1.0,n_inplanes=128,uniform=True)
S,P=pg.n_shells,pg.n_inplanes
th=torch.tensor(pg.theta_shell,dtype=vol.dtype,device=dev)
rr=torch.tensor(pg.radius_shells,dtype=vol.dtype,device=dev)
w=torch.tensor(pg.integration_weight_points,dtype=vol.dtype,device=dev).reshape(S,P)
ctf=make_ctf(pg,N,float(D),device=dev,dtype=vol.dtype,astig=0.0)

def ang_err(a,p,at,pt):
    n1=torch.stack([torch.sin(p)*torch.cos(a),torch.sin(p)*torch.sin(a),torch.cos(p)],-1)
    n2=torch.stack([torch.sin(pt)*torch.cos(at),torch.sin(pt)*torch.sin(at),torch.cos(pt)],-1)
    return torch.rad2deg(torch.arccos((n1*n2).sum(-1).clamp(-1,1)))

g=torch.Generator(device=dev).manual_seed(5)
print(f"{'true polar':>12}{'start err':>11}{'final err (median)':>20}{'p90':>8}{'recovered<2deg':>16}")
print("-"*68)
for pol_deg in (1.0, 3.0, 10.0, 45.0, 90.0):
    az=torch.rand(N,generator=g,device=dev,dtype=vol.dtype)*2*np.pi
    po=torch.full((N,),np.radians(pol_deg),device=dev,dtype=vol.dtype)
    ga=torch.rand(N,generator=g,device=dev,dtype=vol.dtype)*2*np.pi
    clean=slice_nufft(vol,slice_points(fourier_circles(th,po,az,ga),rr)).reshape(N,S,P)*ctf
    pw=(clean.abs()**2*w).sum(dim=(1,2))/w.sum()
    imgs=clean+torch.randn(clean.shape,generator=g,dtype=clean.dtype,device=dev)*torch.sqrt(pw/1.0)[:,None,None]
    d=np.radians(2.0)
    a0=az+torch.randn(N,generator=g,device=dev,dtype=vol.dtype)*d
    p0=po+torch.randn(N,generator=g,device=dev,dtype=vol.dtype)*d
    g0=ga+torch.randn(N,generator=g,device=dev,dtype=vol.dtype)*d
    e0=ang_err(a0,p0,az,po)
    a1,p1,_=refine_poses(vol,imgs,ctf,th,rr,w,a0,p0,g0,steps=300,lr=0.02)
    e1=ang_err(a1,p1,az,po)
    print(f"{pol_deg:>11.0f}°{e0.median():>10.2f}°{e1.median():>19.2f}°{e1.quantile(0.9):>7.1f}°"
          f"{100*(e1<2.0).float().mean():>15.0f}%")
print("\nSNR 1.0, 2 deg initial perturbation. At 45-90 deg polar the earlier tests")
print("gave ~1.1 deg. A much worse number at 1-3 deg polar = pole singularity.")
