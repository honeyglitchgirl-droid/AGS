"""Physics-guided spectral refinement inspired by Spectral-Refiner.

Includes a solver-agnostic spectral refinement objective and an optional compact
three-axis PyTorch spectral operator. The optional model is *not* a full
trajectory-to-trajectory ST-FNO implementation; the numerical solver/referee
remains the source of physical authority.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from .spectral_estimators import negative_sobolev_norm

@dataclass(frozen=True)
class RefinementResult:
    field: np.ndarray
    initial_loss: float
    final_loss: float
    iterations: int
    converged: bool


def pde_residual_loss(prediction, residual_fn: Callable[[np.ndarray], np.ndarray], domain_lengths=None, order=1):
    pred = np.asarray(prediction, dtype=float)
    residual = np.asarray(residual_fn(pred), dtype=float)
    if not np.all(np.isfinite(residual)):
        return float("inf")
    return negative_sobolev_norm(residual, domain_lengths, order=order)


def spectral_gradient(field, domain_lengths=None):
    a = np.asarray(field, dtype=float)
    lengths = tuple(domain_lengths or (2*np.pi,) * a.ndim)
    if len(lengths) != a.ndim:
        raise ValueError("domain_lengths mismatch")
    grids = np.meshgrid(*[2*np.pi*np.fft.fftfreq(n, d=L/n) for n, L in zip(a.shape, lengths)], indexing="ij")
    F = np.fft.fftn(a)
    return tuple(np.real(np.fft.ifftn(1j*k*F)) for k in grids)


def refine_gradient_descent(initial, residual_fn, domain_lengths=None, lr=1e-2, steps=20, order=1, eps=1e-5):
    """Derivative-free spectral residual refinement for small experiments."""
    x = np.array(initial, dtype=float, copy=True)
    if not np.all(np.isfinite(x)) or steps < 1 or lr <= 0:
        raise ValueError("invalid refinement input")
    def loss(z): return pde_residual_loss(z, residual_fn, domain_lengths, order)
    start = loss(x); best = start
    rng = np.random.default_rng(0)
    for _ in range(int(steps)):
        # SPSA-like normalized perturbation keeps memory bounded while using
        # an independent deterministic Rademacher direction per iteration.
        delta = rng.choice((-1.0, 1.0), size=x.shape)
        lp = loss(x + eps * delta); lm = loss(x - eps * delta)
        g = ((lp - lm) / (2 * eps)) * delta
        scale = max(float(np.linalg.norm(g.ravel())), 1.0)
        trial = x - lr * g / scale
        lt = loss(trial)
        if lt <= best:
            x, best = trial, lt
        else:
            lr *= 0.5
    return RefinementResult(x, float(start), float(best), int(steps), bool(best <= start))

try:
    import torch
    import torch.nn as nn
    class SpectralConv3d(nn.Module):
        def __init__(self, in_channels, out_channels, modes):
            super().__init__(); self.in_channels=in_channels; self.out_channels=out_channels; self.modes=tuple(int(x) for x in modes)
            scale = 1.0 / max(1, in_channels*out_channels)
            self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, *self.modes, dtype=torch.cfloat))
        def forward(self, x):
            n = x.shape[-3:]
            fx = torch.fft.rfftn(x, dim=(-3,-2,-1))
            out = torch.zeros(x.shape[0], self.out_channels, n[0], n[1], n[2]//2+1, dtype=fx.dtype, device=x.device)
            m0,m1,m2 = [min(m,s) for m,s in zip(self.modes, out.shape[-3:])]
            out[..., :m0, :m1, :m2] = torch.einsum('bixyz,ioxyz->boxyz', fx[..., :m0,:m1,:m2], self.weight[..., :m0,:m1,:m2])
            return torch.fft.irfftn(out, s=n, dim=(-3,-2,-1))
    class STFNO3D(nn.Module):
        """Compact optional 3-axis spectral operator; not a full ST-FNO trajectory model."""
        def __init__(self, channels=1, width=16, modes=(8,8,8)):
            super().__init__(); self.lift=nn.Conv3d(channels,width,1); self.spec=SpectralConv3d(width,width,modes); self.mix=nn.Conv3d(width,width,1); self.out=nn.Conv3d(width,channels,1)
        def forward(self,x):
            y=self.lift(x); y=torch.nn.functional.gelu(self.spec(y)+self.mix(y)); return self.out(y)
except Exception:  # optional dependency
    torch = None
    class STFNO3D:
        def __init__(self,*args,**kwargs): raise RuntimeError("STFNO3D requires PyTorch")
