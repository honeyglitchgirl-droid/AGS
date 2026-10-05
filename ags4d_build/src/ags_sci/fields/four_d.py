"""Experimental 4D scientific field engine.

This module promotes the existing dimension-generic spectral core into an explicit
4D backend.  It is intentionally Euclidean/spatial: a 4D FFT is not claimed to be
Lorentz-covariant.  Lorentzian tensor operations remain in ``fields.geometry``.

Phase 1 provides a production-quality scalar PDE layer:
- 4D gradient, divergence and Laplacian
- zero-mean Poisson inversion
- configurable spectral filtering/dealiasing
- diffusion and screened-Poisson/Helmholtz RHS operators
- scalar L2/gradient energy diagnostics

Time is not one of the four FFT coordinates.  A future 3+1 spacetime engine can
reuse the same spatial operators while evolving an explicit time coordinate.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .contracts import FieldCapabilities
from .operators import SpectralEngineND


@dataclass(frozen=True)
class FourDScalarDiagnostics:
    l2: float
    gradient_energy: float
    mean: float
    max_abs: float


class FourDScalarFieldEngine:
    """Explicit 4D periodic scalar-field backend built on ``SpectralEngineND``."""

    capabilities = FieldCapabilities(
        dimension=4,
        geometry="cartesian-periodic-4d",
        method="fft",
        supports_projection=False,
        supports_adaptive_dt=False,
        supports_spectrum=True,
        experimental=True,
    )

    def __init__(self, shape: tuple[int, int, int, int], domain_lengths=None,
                 dealias: str = "spherical", filter_alpha: float = 36.0,
                 filter_order: int = 36):
        if len(shape) != 4:
            raise ValueError("FourDScalarFieldEngine requires a 4D shape")
        self.shape = tuple(int(n) for n in shape)
        if any(n < 4 for n in self.shape):
            raise ValueError("4D grid dimensions must be >= 4")
        self.lengths = tuple(float(x) for x in (
            domain_lengths if domain_lengths is not None else (2*np.pi,) * 4
        ))
        self.spectral = SpectralEngineND(
            self.shape, self.lengths, dealias=dealias,
            filter_alpha=filter_alpha, filter_order=filter_order,
        )

    def _check(self, field):
        a = np.asarray(field, dtype=float)
        if a.shape != self.shape:
            raise ValueError(f"field shape {a.shape} != {self.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError("field contains non-finite values")
        return a

    def gradient(self, field):
        return tuple(self.spectral.gradient(self._check(field)))

    def laplacian(self, field):
        return self.spectral.laplacian(self._check(field))

    def solve_poisson(self, source):
        """Solve ∇²u=source with the zero Fourier mode fixed to zero."""
        return self.spectral.solve_poisson(self._check(source))

    def filter_field(self, field):
        return self.spectral.filter_field(self._check(field))

    def diagnostics(self, field) -> FourDScalarDiagnostics:
        a = self._check(field)
        grad = self.gradient(a)
        volume = float(np.prod(self.lengths))
        cell = volume / float(np.prod(self.shape))
        l2 = float(np.sum(a * a) * cell)
        ge = float(0.5 * sum(np.sum(g * g) * cell for g in grad))
        return FourDScalarDiagnostics(
            l2=l2,
            gradient_energy=ge,
            mean=float(np.mean(a)),
            max_abs=float(np.max(np.abs(a))),
        )

    def diffusion_rhs(self, field, diffusivity: float) -> np.ndarray:
        if not np.isfinite(diffusivity) or diffusivity < 0:
            raise ValueError("diffusivity must be finite and non-negative")
        return float(diffusivity) * self.laplacian(field)

    def helmholtz_rhs(self, field, diffusivity: float, mass2: float = 0.0) -> np.ndarray:
        """Return κ∇²u - m²u for diffusion/screened-wave experiments."""
        if not np.isfinite(diffusivity) or diffusivity < 0:
            raise ValueError("diffusivity must be finite and non-negative")
        if not np.isfinite(mass2) or mass2 < 0:
            raise ValueError("mass2 must be finite and non-negative")
        u = self._check(field)
        return float(diffusivity) * self.laplacian(u) - float(mass2) * u

    def spectral_tail_ratio(self, field, inner_fraction: float = 0.8):
        u = self._check(field)
        if not 0.0 <= inner_fraction <= 1.0:
            raise ValueError("inner_fraction must be in [0,1]")
        return self.spectral.spectral_tail_ratio(u, inner_fraction)
