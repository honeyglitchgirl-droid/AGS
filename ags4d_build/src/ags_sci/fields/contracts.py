"""Shared contracts for AGS-Sci 2D/3D field engines.

The scientific solvers remain independent implementations, but they expose the
same bounded capability and state interfaces so AGS can plan experiments
without hard-coding a particular dimension or numerical method.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence
import numpy as np

@dataclass(frozen=True)
class FieldCapabilities:
    dimension: int
    geometry: str
    method: str
    supports_projection: bool
    supports_adaptive_dt: bool
    supports_spectrum: bool
    experimental: bool = True

@dataclass(frozen=True)
class ExperimentBudget:
    max_steps: int = 100_000
    max_wall_seconds: float = 120.0
    max_memory_mb: float = 512.0
    max_output_bytes: int = 8_000_000

    def validate(self) -> None:
        if self.max_steps < 1 or self.max_wall_seconds <= 0 or self.max_memory_mb <= 0 or self.max_output_bytes < 1:
            raise ValueError("invalid experiment budget")

@dataclass(frozen=True)
class FieldSnapshot:
    time: float
    step: int
    fields: tuple[np.ndarray, ...]
    diagnostics: Mapping[str, float]

class FieldEngine(Protocol):
    capabilities: FieldCapabilities
    shape: tuple[int, ...]
    lengths: tuple[float, ...]

    def diagnostics(self, state: Any) -> Any: ...
    def rhs(self, state: Any, **kwargs: Any) -> Any: ...


def validate_state_arrays(state: Sequence[np.ndarray], shape: tuple[int, ...], components: int) -> tuple[np.ndarray, ...]:
    if len(state) != components:
        raise ValueError(f"expected {components} field components")
    out = tuple(np.asarray(x, dtype=float) for x in state)
    if any(x.shape != shape for x in out):
        raise ValueError("state component shape mismatch")
    if any(not np.all(np.isfinite(x)) for x in out):
        raise ValueError("state contains non-finite values")
    return out
