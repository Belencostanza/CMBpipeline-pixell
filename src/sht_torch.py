"""
Differentiable spin-2 spherical-harmonic transforms on a CAR cut-out, for the
curved-sky training loss (losses_car.py).

pixell/ducc has no autograd, but gives the exact adjoint of each transform
(map2alm/alm2map with adjoint=True). Each operator is a torch.autograd.Function
that runs pixell on the CPU in its forward pass and the adjoint in its backward
pass; tensors are moved between the device and the CPU.

  A : map (..., 2, ny, nx) real  ->  a_lm (..., 2, nalm) complex   (E, B)
  Y : a_lm (..., 2, nalm) complex ->  map (..., 2, ny, nx) real     (Q, U)

Gradients follow torch's convention for a real loss L and a complex tensor z:
z.grad = dL/dRe z + i dL/dIm z. pixell's adjoints are defined with the inner
product <a, b> = sum_lm w_m Re(conj(a) b), w_0 = 1 and w_{m>0} = 2 (each m > 0
coefficient stands for +m and -m), so:

  grad_map = A^T (g / w)          for map -> alm
  grad_alm = w * Y^T (h)          for alm -> map

method = 'cyl' (ring by ring on the cut-out rows, quadrature weights of the
full fejer1 grid) is the default: for band-limited content it gives the same
result as the exact-quadrature '2d' to 1e-4 and is ~2x faster than '2d'
(notebooks/03_loss_check.ipynb).
"""

import time

import numpy as np
import torch
from pixell import enmap, curvedsky


class CarSHT:
    """Spin-2 analysis/synthesis on a fixed CAR geometry, up to lmax."""

    def __init__(self, shape, wcs, lmax, method="cyl", nthread=None):
        self.shape = tuple(shape[-2:])
        self.wcs = wcs
        self.lmax = int(lmax)
        self.method = method
        self.nthread = nthread
        self.ainfo = curvedsky.alm_info(self.lmax)
        self.nalm = self.ainfo.nelem
        # l and m of each a_lm element (pixell layout: index = mstart[m] + l * stride)
        self.ell = np.zeros(self.nalm, int)
        self.m = np.zeros(self.nalm, int)
        for m in range(self.ainfo.mmax + 1):
            ls = np.arange(m, self.lmax + 1)
            idx = (self.ainfo.mstart[m] + ls * self.ainfo.stride).astype(int)
            self.ell[idx], self.m[idx] = ls, m
        # weight of each a_lm in the real inner product (m = 0: 1, m > 0: 2)
        self.w = np.where(self.m == 0, 1.0, 2.0)
        self.reset_timer()

    def reset_timer(self):
        """Accumulated wall time of the transforms (t_total) and number of calls (n_calls)."""
        self.t_total, self.n_calls = 0.0, 0

    def _timed(self, fn, x):
        t0 = time.perf_counter()
        out = fn(x)
        self.t_total += time.perf_counter() - t0
        self.n_calls += 1
        return out

    # numpy operators ------------------------------------------------------- #
    def A(self, f):
        """map (..., 2, ny, nx) -> alm (..., 2, nalm)."""
        f = enmap.enmap(np.ascontiguousarray(f, dtype=np.float64), self.wcs)
        return curvedsky.map2alm(f, ainfo=self.ainfo, spin=2, method=self.method, nthread=self.nthread)

    def AT(self, a):
        """adjoint of A: alm (..., 2, nalm) -> map (..., 2, ny, nx)."""
        a = np.ascontiguousarray(a, dtype=np.complex128)
        out = enmap.zeros(a.shape[:-1] + self.shape, self.wcs)
        return np.asarray(curvedsky.map2alm(out, alm=a.copy(), ainfo=self.ainfo, spin=2, method=self.method,
                                            adjoint=True, nthread=self.nthread))

    def Y(self, a):
        """alm (..., 2, nalm) -> map (..., 2, ny, nx)."""
        a = np.ascontiguousarray(a, dtype=np.complex128)
        out = enmap.zeros(a.shape[:-1] + self.shape, self.wcs)
        return np.asarray(curvedsky.alm2map(a, out, ainfo=self.ainfo, spin=2, method=self.method, nthread=self.nthread))

    def YT(self, f):
        """adjoint of Y: map (..., 2, ny, nx) -> alm (..., 2, nalm)."""
        f = enmap.enmap(np.array(f, dtype=np.float64), self.wcs)
        a = np.zeros(f.shape[:-2] + (self.nalm,), np.complex128)
        return curvedsky.alm2map(a, f, ainfo=self.ainfo, spin=2, method=self.method, adjoint=True, nthread=self.nthread)

    # torch operators ------------------------------------------------------- #
    def analysis(self, f):
        """torch: map -> alm (differentiable)."""
        return _Analysis.apply(f, self)

    def synthesis(self, a):
        """torch: alm -> map (differentiable)."""
        return _Synthesis.apply(a, self)


def _to_np(t):
    return t.detach().cpu().numpy()


def _cplx_dtype(dtype):
    return torch.complex128 if dtype == torch.float64 else torch.complex64


class _Analysis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, f, sht):
        ctx.sht, ctx.dtype, ctx.device = sht, f.dtype, f.device
        a = sht._timed(sht.A, _to_np(f))
        return torch.from_numpy(np.asarray(a)).to(device=f.device, dtype=_cplx_dtype(f.dtype))

    @staticmethod
    def backward(ctx, g):
        sht = ctx.sht
        grad = sht._timed(sht.AT, _to_np(g) / sht.w)
        return torch.from_numpy(grad).to(device=ctx.device, dtype=ctx.dtype), None


class _Synthesis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, sht):
        ctx.sht, ctx.dtype, ctx.device = sht, a.dtype, a.device
        f = sht._timed(sht.Y, _to_np(a))
        rdtype = torch.float64 if a.dtype == torch.complex128 else torch.float32
        return torch.from_numpy(f).to(device=a.device, dtype=rdtype)

    @staticmethod
    def backward(ctx, h):
        sht = ctx.sht
        grad = sht.w * np.asarray(sht._timed(sht.YT, _to_np(h)))
        return torch.from_numpy(grad).to(device=ctx.device, dtype=ctx.dtype), None
