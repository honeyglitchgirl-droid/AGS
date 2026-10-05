"""5D scientific field engines.

This module extends the dimension-generic spectral core to five spatial
dimensions, following the same explicit pattern as the 4D backend.  Two engines
are provided so that "all working fields" are covered:

``FiveDScalarFieldEngine``
    Scalar PDE layer: gradient, Laplacian, zero-mode-fixed Poisson inversion,
    spectral filtering/dealiasing, diffusion and screened-Poisson/Helmholtz
    RHS operators, and scalar diagnostics.

``FiveDVectorFieldEngine``
    Vector layer: Jacobian, divergence, curl (a rank-2 antisymmetric tensor in
    5D, with 10 independent components), componentwise Laplacian, Leray
    projection for incompressibility, and energy/enstrophy diagnostics.

As with the 4D engine, the discretization is Euclidean.  Nothing here claims
Lorentz covariance or extra spatial dimensions of physics; these are numerical
backends for experiments that happen to be five-dimensional.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .contracts import FieldCapabilities, validate_state_arrays
from .operators import SpectralEngineND


@dataclass(frozen=True)
class FiveDScalarDiagnostics:
    l2: float
    gradient_energy: float
    mean: float
    max_abs: float


class FiveDScalarFieldEngine:
    """Explicit 5D periodic scalar-field backend built on ``SpectralEngineND``."""

    capabilities = FieldCapabilities(
        dimension=5,
        geometry="cartesian-periodic-5d",
        method="fft",
        supports_projection=False,
        supports_adaptive_dt=False,
        supports_spectrum=True,
        experimental=True,
    )

    def __init__(self, shape, domain_lengths=None, dealias: str = "spherical",
                 filter_alpha: float = 36.0, filter_order: int = 36):
        if len(shape) != 5:
            raise ValueError("FiveDScalarFieldEngine requires a 5D shape")
        self.shape = tuple(int(n) for n in shape)
        if any(n < 4 for n in self.shape):
            raise ValueError("5D grid dimensions must be >= 4")
        self.lengths = tuple(float(x) for x in (
            domain_lengths if domain_lengths is not None else (2 * np.pi,) * 5
        ))
        if len(self.lengths) != 5 or any(x <= 0 for x in self.lengths):
            raise ValueError("domain_lengths must be 5 positive values")
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

    def gradient_denoised(self, field, **kwargs):
        """Noise-robust gradient; see :meth:`SpectralEngineND.gradient_denoised`.

        Delegated so the 5D engines get the same evidence-carrying result rather
        than a silently different variant.
        """
        return self.spectral.gradient_denoised(self._check(field), **kwargs)

    def laplacian(self, field):
        return self.spectral.laplacian(self._check(field))

    def solve_poisson(self, source):
        """Solve Laplacian(u)=source with the zero Fourier mode fixed to zero."""
        return self.spectral.solve_poisson(self._check(source))

    def filter_field(self, field):
        return self.spectral.filter_field(self._check(field))

    def diagnostics(self, field) -> FiveDScalarDiagnostics:
        a = self._check(field)
        grad = self.gradient(a)
        volume = float(np.prod(self.lengths))
        cell = volume / float(np.prod(self.shape))
        l2 = float(np.sum(a * a) * cell)
        ge = float(0.5 * sum(np.sum(g * g) * cell for g in grad))
        return FiveDScalarDiagnostics(
            l2=l2, gradient_energy=ge, mean=float(np.mean(a)),
            max_abs=float(np.max(np.abs(a))),
        )

    def diffusion_rhs(self, field, diffusivity: float) -> np.ndarray:
        if not np.isfinite(diffusivity) or diffusivity < 0:
            raise ValueError("diffusivity must be finite and non-negative")
        return float(diffusivity) * self.laplacian(field)

    def helmholtz_rhs(self, field, diffusivity: float, mass2: float = 0.0) -> np.ndarray:
        """Return kappa*Laplacian(u) - m^2*u for diffusion/screened-wave runs."""
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

    def step_diffusion(self, field, diffusivity: float, dt: float) -> np.ndarray:
        """One exact-in-time linear diffusion step (integrating factor).

        Uses the exact integrating factor exp(-kappa*k^2*dt) rather than an
        explicit update, so arbitrarily large dt remains stable for the linear
        part.  Nonlinear advection is not included here.
        """
        if not np.isfinite(diffusivity) or diffusivity < 0:
            raise ValueError("diffusivity must be finite and non-negative")
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and non-negative")
        u = self._check(field)
        if diffusivity == 0.0 or dt == 0.0:
            return u.copy()
        uhat = self.spectral._raw_hat(u)
        decay = np.exp(-float(diffusivity) * self.spectral.K_sq * float(dt))
        return np.real(np.fft.ifftn(uhat * decay))


@dataclass(frozen=True)
class FiveDVectorDiagnostics:
    divergence_max_abs: float
    kinetic_energy: float
    enstrophy: float
    max_abs: float


class FiveDVectorFieldEngine:
    """5D vector-field backend: divergence, curl-as-2-form, projection, energies."""

    capabilities = FieldCapabilities(
        dimension=5,
        geometry="cartesian-periodic-5d",
        method="fft",
        supports_projection=True,
        supports_adaptive_dt=False,
        supports_spectrum=True,
        experimental=True,
    )

    #: Number of independent components of a rank-2 antisymmetric tensor in 5D.
    CURL_COMPONENTS = 10

    def __init__(self, shape, domain_lengths=None, dealias: str = "spherical",
                 filter_alpha: float = 36.0, filter_order: int = 36):
        if len(shape) != 5:
            raise ValueError("FiveDVectorFieldEngine requires a 5D shape")
        self.shape = tuple(int(n) for n in shape)
        if any(n < 4 for n in self.shape):
            raise ValueError("5D grid dimensions must be >= 4")
        self.lengths = tuple(float(x) for x in (
            domain_lengths if domain_lengths is not None else (2 * np.pi,) * 5
        ))
        if len(self.lengths) != 5 or any(x <= 0 for x in self.lengths):
            raise ValueError("domain_lengths must be 5 positive values")
        self.spectral = SpectralEngineND(
            self.shape, self.lengths, dealias=dealias,
            filter_alpha=filter_alpha, filter_order=filter_order,
        )

    def _check_state(self, state):
        return validate_state_arrays(state, self.shape, 5)

    def jacobian(self, state):
        """Return the 5x5 Jacobian d(v_i)/d(x_j) as a tuple of tuples."""
        v = self._check_state(state)
        grads = [self.spectral.gradient(comp) for comp in v]
        return tuple(tuple(grads[j][i] for j in range(5)) for i in range(5))

    def divergence(self, state):
        v = self._check_state(state)
        total = np.zeros(self.shape, dtype=float)
        for i in range(5):
            total = total + self.spectral.gradient(v[i])[i]
        return total

    def curl(self, state):
        """Curl as the rank-2 antisymmetric tensor (dv_i/dx_j - dv_j/dx_i)/2.

        In five dimensions the curl of a vector is not a vector but a 2-form
        with C(5,2)=10 independent components.  Returned as a tuple of
        (i, j, field) triples with i < j.
        """
        v = self._check_state(state)
        grads = [self.spectral.gradient(comp) for comp in v]
        out = []
        for i in range(5):
            for j in range(i + 1, 5):
                out.append((i, j, 0.5 * (grads[j][i] - grads[i][j])))
        return tuple(out)

    def laplacian(self, state):
        v = self._check_state(state)
        return tuple(self.spectral.laplacian(comp) for comp in v)

    def leray_project(self, state):
        """Project onto the divergence-free subspace."""
        v = self._check_state(state)
        hats = [self.spectral._raw_hat(comp) for comp in v]
        K = self.spectral.K_grids
        Ksq = self.spectral.K_sq
        dot = sum(K[i] * hats[i] for i in range(5))
        nz = Ksq > 0
        out = []
        for i in range(5):
            ph = hats[i].copy()
            ph[nz] -= K[i][nz] * dot[nz] / Ksq[nz]
            ph *= self.spectral.dealias_mask * self.spectral.spectral_filter
            out.append(np.real(np.fft.ifftn(ph)))
        return tuple(out)

    def diagnostics(self, state) -> FiveDVectorDiagnostics:
        v = self._check_state(state)
        div = self.divergence(v)
        volume = float(np.prod(self.lengths))
        cell = volume / float(np.prod(self.shape))
        ke = float(0.5 * sum(np.sum(c * c) * cell for c in v))
        curl = self.curl(v)
        ens = float(0.5 * sum(np.sum(f * f) * cell for _, _, f in curl))
        return FiveDVectorDiagnostics(
            divergence_max_abs=float(np.max(np.abs(div))),
            kinetic_energy=ke,
            enstrophy=ens,
            max_abs=float(np.max(np.abs(np.asarray(v)))),
        )

    def advection_rhs(self, state):
        """Return -(v . grad) v for each component."""
        v = self._check_state(state)
        grads = [self.spectral.gradient(comp) for comp in v]
        out = []
        for i in range(5):
            acc = np.zeros(self.shape, dtype=float)
            for j in range(5):
                acc = acc + v[j] * grads[i][j]
            out.append(-acc)
        return tuple(out)

    def diffusion_rhs(self, state, diffusivity: float):
        if not np.isfinite(diffusivity) or diffusivity < 0:
            raise ValueError("diffusivity must be finite and non-negative")
        return tuple(float(diffusivity) * l for l in self.laplacian(state))
