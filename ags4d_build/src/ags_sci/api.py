"""The AGS-Sci service facade.

This is the single, stable entry point an external AI uses to drive AGS. It
exists so that a client never has to import internal modules or know which
dimension a particular engine lives in: it asks the service for a capability and
the service resolves it.

Design rules this facade enforces:

* **Discovery before use.** :meth:`AGSService.capabilities` describes what this
  build can do, so a client adapts to AGS instead of assuming.
* **No silent authority.** Results carry explicit provenance and epistemic
  status. A numerical fit is never presented as a discovered law.
* **Fail closed.** Invalid identifiers, oversized payloads and unknown
  capabilities raise :class:`~ags_sci.core.security.SecurityViolation` rather
  than degrading to a permissive default.
* **Plugins extend, never replace.** A plugin receives this service handle; it
  cannot reach into protected core state.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from .core.evolution import PassiveEvolutionEngine

import numpy as np

from . import __version__
from .core.security import bounded_text
from .plugins import PluginRegistry, ServiceCapabilities
from .complex_systems import DOMAINS, resolve as resolve_domain
from .quantum import (QuantumCircuit, SimulatorBackend, expectation, pauli_z,
                      quantum_walk_mixing, vqe_ground_energy)


@dataclass(frozen=True)
class DiscoveredLaw:
    """A discovered equation set with its evidence, explicitly separated.

    ``train_rmse`` and ``holdout_rmse`` are statements about the data.
    ``evidence_tier`` is a statement about how much support AGS is willing to
    claim. They are deliberately different fields so that a good fit can never
    be mistaken for a validated law.
    """
    equations: tuple[str, ...]
    train_rmse: float
    holdout_rmse: float | None
    bic: float
    n_active_terms: int
    evidence_tier: str
    library: tuple[str, ...]
    note: str

    def describe(self) -> dict[str, Any]:
        return {
            "equations": list(self.equations),
            "train_rmse": self.train_rmse,
            "holdout_rmse": self.holdout_rmse,
            "bic": self.bic,
            "n_active_terms": self.n_active_terms,
            "evidence_tier": self.evidence_tier,
            "library_size": len(self.library),
            "note": self.note,
        }


class AGSService:
    """Stable facade over AGS-Sci's engines, sandbox, discovery and quantum."""

    def __init__(self, *, sandbox_timeout: float = 10.0, sandbox_memory_mb: int = 512,
                 max_qubits: int = 16):
        from .fields.registry import default_field_registry
        self._registry = default_field_registry()
        self._plugins = PluginRegistry()
        self._sandbox_timeout = float(sandbox_timeout)
        self._sandbox_memory_mb = int(sandbox_memory_mb)
        self._quantum = SimulatorBackend(max_qubits=int(max_qubits))

    # -- discovery ------------------------------------------------------
    def capabilities(self) -> dict[str, Any]:
        return ServiceCapabilities(
            version=__version__,
            field_dimensions=tuple(sorted({b.dimension for b in self._registry._items.values()})),
            field_backends=tuple(sorted(self._registry._items)),
            domain_primitives=tuple(DOMAINS),
            sandbox_available=True,
            quantum_available=True,
            plugins=self._plugins.names(),
            invention_available=True,
        ).describe()

    def plugins(self) -> PluginRegistry:
        return self._plugins

    # -- field engines --------------------------------------------------
    def field(self, backend: str, shape: Sequence[int], **kwargs: Any):
        """Construct a field engine by backend name.

        The backend's own dimension/shape contract is enforced by the engine, so
        a 5D shape handed to a 2D engine fails there rather than being silently
        reinterpreted here.
        """
        name = str(backend)
        spec = self._registry.get(name)
        return spec.factory(tuple(int(n) for n in shape), **kwargs)

    def field_backends(self, dimension: int | None = None) -> tuple[str, ...]:
        if dimension is None:
            return tuple(sorted(self._registry._items))
        return tuple(sorted(b.name for b in self._registry.by_dimension(int(dimension))))

    # -- sandboxed execution -------------------------------------------
    def run_code(self, code: str, *, timeout: float | None = None,
                 internet_enabled: bool = False,
                 internet_allowed_domains: Sequence[str] = ()) -> dict[str, Any]:
        """Execute untrusted code in the hardened sandbox.

        Off by default; network egress requires an explicit domain allow-list.
        """
        from .experiment.sandbox import ExecutionSandbox
        bounded_text(code, field="code")
        with ExecutionSandbox(
            timeout_seconds=float(timeout if timeout is not None else self._sandbox_timeout),
            max_memory_mb=self._sandbox_memory_mb,
            internet_enabled=bool(internet_enabled),
            internet_allowed_domains=tuple(str(d) for d in internet_allowed_domains),
        ) as sandbox:
            result = sandbox.execute(code)
        return {
            "status": result.status,
            "stdout": result.stdout,
            "error": result.error,
            "value": result.value,
            "artifacts": sorted((result.artifacts or {}).keys()),
            "execution_time_ms": result.execution_time_ms,
        }

    # -- equation / law discovery --------------------------------------
    def discover_law(self, t, X, names: Sequence[str] | None = None,
                     degree: int = 2, holdout_fraction: float = 0.25,
                     include_transcendentals: bool = False) -> "DiscoveredLaw":
        """Identify sparse ODEs from a trajectory.

        Uses :class:`~ags_sci.discovery.law.SparseLawDiscovery`, which selects the
        sparsity threshold by BIC and scores the result on held-out data. The
        returned candidate carries an explicit ``evidence_tier``; AGS never
        promotes a numerical fit to a scientific law on its own.
        """
        from .discovery.law import SparseLawDiscovery
        bounded_text(str(names or ""), field="names")
        discovery = SparseLawDiscovery(
            degree=int(degree),
            holdout_fraction=float(holdout_fraction),
            include_transcendentals=bool(include_transcendentals),
        )
        result = discovery.fit(np.asarray(t, float), np.asarray(X, float),
                               tuple(names) if names else None)
        return DiscoveredLaw(
            equations=result.equations,
            train_rmse=result.train_rmse,
            holdout_rmse=result.holdout_rmse,
            bic=result.bic,
            n_active_terms=result.n_active_terms,
            evidence_tier=result.evidence_tier,
            library=result.library,
            note=result.note,
        )

    # -- domain primitives ---------------------------------------------
    def domain(self, name: str):
        """Resolve one of the 25 complex-systems primitives by domain name."""
        return resolve_domain(str(name))

    def domain_names(self) -> tuple[str, ...]:
        return tuple(DOMAINS)

    # -- quantum --------------------------------------------------------
    def quantum_backend(self) -> SimulatorBackend:
        return self._quantum

    def quantum_circuit(self, n_qubits: int) -> QuantumCircuit:
        return QuantumCircuit(int(n_qubits))

    def vqe(self, hamiltonian: np.ndarray, n_qubits: int, **kwargs) -> dict[str, Any]:
        """Bounded variational ground-state estimate (simulated)."""
        return vqe_ground_energy(np.asarray(hamiltonian, complex), int(n_qubits), **kwargs)

    def quantum_walk(self, n_sites: int, steps: int) -> np.ndarray:
        return quantum_walk_mixing(int(n_sites), int(steps))

    def z_expectation(self, circuit: QuantumCircuit, qubit: int) -> float:
        return expectation(circuit, pauli_z(circuit.n_qubits, int(qubit)))

    # -- gradient operators -----------------------------------------------
    def _spectral_engine(self, backend: str, shape, **kwargs):
        """Resolve the ``SpectralEngineND`` behind a backend, however it is held.

        Backends wrap the engine under different attribute names (``engine`` for
        the 3D turbulence family, ``spectral`` for the 4D/5D engines), so this
        looks for it rather than assuming one spelling.
        """
        from .fields.operators import SpectralEngineND
        obj = self.field(backend, shape, **kwargs)
        for attr in ("engine", "spectral"):
            inner = getattr(obj, attr, None)
            if isinstance(inner, SpectralEngineND):
                return inner
        if isinstance(obj, SpectralEngineND):
            return obj
        raise TypeError(f"backend {backend!r} is not backed by a SpectralEngineND")

    def gradient(self, backend: str, shape, field, **kwargs):
        """Plain spectral gradient: exact to machine precision below Nyquist.

        This is the default because for a band-limited field nothing can beat it.
        Use :meth:`denoised_gradient` when the data is noisy.
        """
        return self._spectral_engine(backend, shape, **kwargs).gradient(field)

    def denoised_gradient(self, backend: str, shape, field, **kwargs):
        """Noise-robust gradient with the evidence for its own choice attached.

        Returns a :class:`~ags_sci.fields.operators.DenoisedGradient`. Read
        ``regime`` first: ``"shrunk"`` means a Wiener multiplier was applied and
        the error is roughly half the plain gradient's; ``"plain"`` means the
        operator declined, because the spectral tail still decays and carries
        real unresolved signal that shrinking would discard. When it declines,
        the result is *identical* to the plain spectral gradient.
        """
        return self._spectral_engine(backend, shape, **kwargs).gradient_denoised(field)

    # -- invention ---------------------------------------------------------
    def invent(self, t, target, *, max_depth: int = 3, population_size: int = 120,
               generations: int = 40, seed: int = 0, top_k: int = 5) -> dict[str, Any]:
        """Invent candidate laws by searching a symbolic grammar.

        Unlike :meth:`discover_law`, which fits a fixed library of terms, this
        *generates* candidates, assesses each against a declared corpus of known
        laws, and formulates open problems from the residuals of the best fit.
        Novelty is corpus-relative and reported as such.
        """
        from .invention import InventionEngine
        t_arr = np.asarray(t, float).reshape(-1)
        target_arr = np.asarray(target, dtype=float)
        if target_arr.ndim == 1:
            target_arr = target_arr.reshape(-1, 1)
        # A single series regressed on the sample axis is naturally a function
        # of ``t``; several series are several state variables.
        if target_arr.shape[1] == 1:
            names = ["t"]
        else:
            names = [f"x{i}" for i in range(target_arr.shape[1])]
        results: dict[str, Any] = {"status": "CANDIDATES_GENERATED", "per_target": []}
        for j, name in enumerate(names):
            eng = InventionEngine(max_depth=int(max_depth),
                                  population_size=int(population_size),
                                  generations=int(generations), seed=int(seed))
            res = eng.invent(target_arr[:, j], {name: t_arr})
            res["target"] = name
            results["per_target"].append(res)
        if len(results["per_target"]) == 1:
            results.update(results["per_target"][0])
            results.pop("per_target", None)
            results["target"] = names[0]
        return results

    def conjecture_sequence(self, sequence, *, holdout: int = 5,
                            start_index: int = 0) -> Any:
        """Invent a closed form for a numeric sequence and try to falsify it.

        This is the one invention capability with an objective pass/fail: a
        finite sequence is a finite object, so a conjecture that survives the
        held-out tail has genuinely predicted something.
        """
        from .invention import InventionEngine
        eng = InventionEngine()
        return eng.conjecture_sequence(sequence, holdout=int(holdout),
                                       start_index=int(start_index))

    def propose_problems(self, residuals, x=None, t=None, driver=None) -> tuple:
        """Formulate open problems from unexplained residual structure."""
        from .invention import ProblemGenerator
        return ProblemGenerator().generate(residuals, x=x, t=t, driver=driver)

    # -- passive self-evolution -----------------------------------------
    def evolution_engine(self, parameter_space: dict[str, tuple[float, float]],
                         min_observations: int = 5) -> "PassiveEvolutionEngine":
        """Create a passive self-evolution engine over a declared parameter space.

        The engine may only *observe* and *propose*; it never edits source and
        never touches a protected path. The host accepts a proposal explicitly.
        """
        from .core.evolution import PassiveEvolutionEngine
        return PassiveEvolutionEngine(parameter_space, min_observations=min_observations)

    # -- plugin management ---------------------------------------------
    def install_plugin(self, plugin: Any, *, allow_privileged: bool = False):
        """Validate and install a plugin. Privileged plugins need opt-in."""
        record = self._plugins.register(plugin, allow_privileged=allow_privileged)
        self._plugins.install(self, record.name)
        return record.describe()
