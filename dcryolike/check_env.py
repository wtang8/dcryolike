"""Verify the environment actually works, not merely that imports succeed.

`torch.cuda.is_available()` returns True on a GPU whose architecture torch has
no kernels for; the failure only appears when a kernel launches. Every check
here therefore EXECUTES something.
"""
import sys
_F = []
def check(label, fn, want=None):
    try:
        got = fn()
        ok = True if want is None else (got == want)
    except Exception as e:
        got, ok = f"{type(e).__name__}: {e}", False
    _F.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<38} {got}")

import numpy, scipy, torch
check("numpy",  lambda: numpy.__version__,  "2.5.2")
check("scipy",  lambda: scipy.__version__,  "1.18.1")
check("torch",  lambda: torch.__version__,  "2.11.0+cu128")
check("cuda available", lambda: torch.cuda.is_available(), True)
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    check("device", lambda: torch.cuda.get_device_name(0))
    check("compute capability", lambda: f"sm_{cap[0]}{cap[1]}")
    check("arch in torch build", lambda: f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list(), True)
    # execute, do not just query -- this is what catches a missing kernel image
    check("matmul executes", lambda: bool(torch.randn(256,256,device="cuda").mm(
        torch.randn(256,256,device="cuda")).isfinite().all()), True)
    check("cuFFT executes", lambda: tuple(torch.fft.fftn(
        torch.randn(32,32,32,device="cuda")).shape), (32,32,32))
    check("grid_sample 3D executes", lambda: tuple(torch.nn.functional.grid_sample(
        torch.randn(1,2,16,16,16,device="cuda"),
        torch.rand(1,4,4,1,3,device="cuda")*2-1, align_corners=True).shape), (1,2,4,4,1))
try:
    import cufinufft
    check("cufinufft", lambda: cufinufft.__version__, "2.5.1")
    def _nufft():
        import numpy as np
        N, M = 8, 32
        f = torch.randn(N,N,N,device="cuda",dtype=torch.complex64)
        p = ((torch.rand(3,M,device="cuda",dtype=torch.float32)-0.5)*2*np.pi).contiguous()
        out = torch.as_tensor(cufinufft.nufft3d2(p[0].contiguous(),p[1].contiguous(),
                                                 p[2].contiguous(), f, isign=-1, eps=1e-6))
        return bool(out.isfinite().all())
    check("cufinufft executes on this GPU", _nufft, True)
except Exception as e:
    check("cufinufft", lambda: (_ for _ in ()).throw(e))
try:
    sys.path.insert(0, "/home/wtang/cryoem/cryolike/src")
    from cryolike.grids import PolarGrid
    check("cryolike reference importable", lambda: PolarGrid(
        radius_max=8, dist_radii=1.0, n_inplanes=32, uniform=True).n_shells > 0, True)
except Exception as e:
    check("cryolike reference importable", lambda: (_ for _ in ()).throw(e))
print(f"\n{'ENVIRONMENT OK' if all(_F) else 'ENVIRONMENT PROBLEM'}  ({sum(_F)}/{len(_F)})")
sys.exit(0 if all(_F) else 1)
