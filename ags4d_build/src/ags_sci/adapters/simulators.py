"""Pure simulation adapter contracts. Actual execution belongs behind sandbox boundaries."""
from __future__ import annotations
from typing import Any, Callable, Protocol
from dataclasses import dataclass
from ags_sci.experiment.sandbox import SandboxedRunner, SandboxResult

class Simulator(Protocol):
    def simulate(self, intervention: dict[str,float], controls: dict[str,float]) -> Any: ...

@dataclass
class DeterministicSimulator:
    function: Callable[[dict[str,float],dict[str,float]],Any]
    runner: SandboxedRunner
    def simulate(self, intervention, controls):
        result: SandboxResult=self.runner.run(self.function, intervention, controls)
        if not result.ok: raise RuntimeError(result.error)
        return result.value
