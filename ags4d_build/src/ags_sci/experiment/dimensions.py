"""Dimension-aware, resource-bounded experiment admission for AGS-Sci.

The sandbox treats experiment state as ephemeral.  Only the declared experiment
metadata and returned value/artifacts cross the sandbox boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math


SUPPORTED_DIMENSIONS: Tuple[int, ...] = (1, 2, 3, 4)


@dataclass(frozen=True)
class DimensionLimits:
    max_cells: int
    max_fields: int = 16
    max_steps: int = 100_000
    max_memory_mb: int = 512
    max_runtime_s: float = 30.0


DEFAULT_DIMENSION_LIMITS = {
    1: DimensionLimits(max_cells=10_000_000),
    2: DimensionLimits(max_cells=4_000_000),
    3: DimensionLimits(max_cells=262_144),
    4: DimensionLimits(max_cells=65_536),
}


@dataclass(frozen=True)
class ExperimentSpec:
    dimension: int
    shape: Tuple[int, ...]
    fields: int = 1
    dtype_bytes: int = 8
    steps: int = 1
    runtime_s: float = 1.0

    @property
    def cells(self) -> int:
        if len(self.shape) != self.dimension:
            raise ValueError("shape dimensionality does not match experiment dimension")
        if any(int(x) <= 0 for x in self.shape):
            raise ValueError("shape entries must be positive")
        return int(__import__("math").prod(self.shape))

    @property
    def estimated_memory_bytes(self) -> int:
        # Conservative working-set estimate: state + two temporaries + field data.
        return self.cells * self.fields * self.dtype_bytes * 3


@dataclass(frozen=True)
class Admission:
    allowed: bool
    reason: str = ""
    estimated_memory_mb: float = 0.0


def validate_spec(spec: ExperimentSpec) -> None:
    if spec.dimension not in SUPPORTED_DIMENSIONS:
        raise ValueError(f"unsupported experiment dimension: {spec.dimension}")
    if not isinstance(spec.shape, tuple):
        raise ValueError("shape must be a tuple")
    if isinstance(spec.fields, bool) or not isinstance(spec.fields, int) or spec.fields <= 0 or spec.fields > 1024:
        raise ValueError("fields must be between 1 and 1024")
    if isinstance(spec.dtype_bytes, bool) or not isinstance(spec.dtype_bytes, int) or spec.dtype_bytes not in (1, 2, 4, 8, 16):
        raise ValueError("dtype_bytes must be one of 1, 2, 4, 8, 16")
    if isinstance(spec.steps, bool) or not isinstance(spec.steps, int) or spec.steps <= 0:
        raise ValueError("steps must be positive")
    if isinstance(spec.runtime_s, bool) or not isinstance(spec.runtime_s, (int, float)) or not math.isfinite(float(spec.runtime_s)) or spec.runtime_s <= 0:
        raise ValueError("runtime_s must be finite and positive")
    if len(spec.shape) != spec.dimension:
        raise ValueError("shape dimensionality does not match experiment dimension")
    for n in spec.shape:
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError("shape entries must be positive integers")
    _ = spec.cells


def admit(spec: ExperimentSpec, limits: dict[int, DimensionLimits] | None = None) -> Admission:
    validate_spec(spec)
    table = limits or DEFAULT_DIMENSION_LIMITS
    lim = table[spec.dimension]
    memory_mb = spec.estimated_memory_bytes / (1024 * 1024)
    if spec.cells > lim.max_cells:
        return Admission(False, f"cell budget exceeded: {spec.cells} > {lim.max_cells}", memory_mb)
    if spec.fields > lim.max_fields:
        return Admission(False, f"field budget exceeded: {spec.fields} > {lim.max_fields}", memory_mb)
    if spec.steps > lim.max_steps:
        return Admission(False, f"step budget exceeded: {spec.steps} > {lim.max_steps}", memory_mb)
    if memory_mb > lim.max_memory_mb:
        return Admission(False, f"estimated memory exceeded: {memory_mb:.1f} MB > {lim.max_memory_mb} MB", memory_mb)
    if spec.runtime_s > lim.max_runtime_s:
        return Admission(False, f"runtime budget exceeded: {spec.runtime_s:.2f}s > {lim.max_runtime_s:.2f}s", memory_mb)
    return Admission(True, "admitted", memory_mb)
