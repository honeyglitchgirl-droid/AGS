"""Dimension/backend registry for AGS-Sci field experiments.

The registry is deliberately declarative: AGS can discover what a field backend
supports without importing dimension-specific implementation details into its
planning layer.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any

@dataclass(frozen=True)
class BackendSpec:
    name: str
    dimension: int
    geometry: str
    method: str
    factory: Callable[..., Any]
    experimental: bool = True

class FieldBackendRegistry:
    def __init__(self):
        self._items: dict[str, BackendSpec] = {}

    def register(self, spec: BackendSpec) -> None:
        if spec.dimension not in (1, 2, 3, 4, 5):
            raise ValueError("field backend dimension must be 1..5")
        if not spec.name or spec.name in self._items:
            raise ValueError("backend name must be non-empty and unique")
        self._items[spec.name] = spec

    def get(self, name: str) -> BackendSpec:
        try:
            return self._items[name]
        except KeyError as exc:
            raise KeyError(f"unknown field backend: {name}") from exc

    def by_dimension(self, dimension: int) -> tuple[BackendSpec, ...]:
        return tuple(v for v in self._items.values() if v.dimension == int(dimension))

    def describe(self) -> tuple[dict[str, object], ...]:
        return tuple({"name": v.name, "dimension": v.dimension, "geometry": v.geometry,
                      "method": v.method, "experimental": v.experimental} for v in self._items.values())


def default_field_registry() -> FieldBackendRegistry:
    from .turbulence2d import FastRFFTTurbulence2D
    from .turbulence3d import PseudoSpectral3D
    from .hybrid_spectral3d import FourierChebyshev3D
    from .spherical_shell3d import SphericalShellToroidalDiffusion
    from .four_d import FourDScalarFieldEngine
    from .five_d import FiveDScalarFieldEngine, FiveDVectorFieldEngine
    r = FieldBackendRegistry()
    r.register(BackendSpec("turbulence2d", 2, "cartesian-periodic", "vorticity-rfft", FastRFFTTurbulence2D, False))
    r.register(BackendSpec("pseudo_spectral3d", 3, "cartesian-periodic", "fft", PseudoSpectral3D, True))
    r.register(BackendSpec("fourier_chebyshev3d", 3, "slab", "fourier-chebyshev", FourierChebyshev3D, True))
    r.register(BackendSpec("spherical_shell3d", 3, "spherical-shell", "chebyshev-radial", SphericalShellToroidalDiffusion, True))
    r.register(BackendSpec("scalar4d", 4, "cartesian-periodic-4d", "fft-scalar-pde", FourDScalarFieldEngine, True))
    r.register(BackendSpec("scalar5d", 5, "cartesian-periodic-5d", "fft-scalar-pde", FiveDScalarFieldEngine, True))
    r.register(BackendSpec("vector5d", 5, "cartesian-periodic-5d", "fft-vector-pde", FiveDVectorFieldEngine, True))
    return r
