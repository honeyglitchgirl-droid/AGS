"""Tests for the v102 capability upgrade: 5D engines, hardened sandbox, plugin
architecture, improved law discovery, passive self-evolution, and the quantum
simulator.

Every numerical claim here is checked against a closed form or an independently
computed reference, not against the implementation's own output.
"""
import numpy as np
import pytest

import ags_sci
from ags_sci import AGSService
from ags_sci.fields.five_d import FiveDScalarFieldEngine, FiveDVectorFieldEngine
from ags_sci.fields.registry import default_field_registry
from ags_sci.discovery.law import SparseLawDiscovery, build_library, _stlsq
from ags_sci.core.evolution import (EvolutionObservation, ParameterProposal,
                                    PassiveEvolutionEngine)
from ags_sci.core.security import SecurityViolation
from ags_sci.quantum import (QuantumCircuit, SimulatorBackend, ansatz_ry,
                             expectation, pauli_z, pauli_string,
                             quantum_walk_mixing, vqe_ground_energy)
from ags_sci.plugins import ALLOWED_CAPABILITIES, PluginRegistry


# ==========================================================================
# 5D engines
# ==========================================================================
# N=12 keeps modes 1..5 below the Nyquist limit so nothing aliases.
N5 = 12
SHAPE5 = (N5,) * 5


def _grid(shape, L=2 * np.pi):
    x = [L * np.arange(n) / n for n in shape]
    return np.meshgrid(*x, indexing="ij")


@pytest.fixture(scope="module")
def engines5():
    return FiveDScalarFieldEngine(SHAPE5), FiveDVectorFieldEngine(SHAPE5)


def test_5d_scalar_laplacian_matches_per_mode_eigenvalues(engines5):
    s, _ = engines5
    X = _grid(SHAPE5)
    modes = [1, 2, 3, 4, 5]
    phi = sum(np.sin(m * X[k]) for k, m in enumerate(modes))
    # Each term has its own eigenvalue -m^2; they do not share one.
    exact = sum(-(m * m) * np.sin(m * X[k]) for k, m in enumerate(modes))
    assert np.max(np.abs(s.laplacian(phi) - exact)) < 1e-10


def test_5d_poisson_round_trip(engines5):
    s, _ = engines5
    X = _grid(SHAPE5)
    src = np.prod([np.sin(X[k]) for k in range(5)], axis=0)
    assert np.max(np.abs(s.laplacian(s.solve_poisson(src)) - src)) < 1e-10


def test_5d_exact_diffusion_step_matches_analytic_decay(engines5):
    s, _ = engines5
    X = _grid(SHAPE5)
    modes = [1, 2, 3, 4, 5]
    phi = sum(np.sin(m * X[k]) for k, m in enumerate(modes))
    kappa, T = 0.01, 0.3
    exact = sum(np.exp(-kappa * m * m * T) * np.sin(m * X[k]) for k, m in enumerate(modes))
    assert np.max(np.abs(s.step_diffusion(phi, kappa, T) - exact)) < 1e-10


def test_5d_helmholtz_rhs(engines5):
    s, _ = engines5
    X = _grid(SHAPE5)
    modes = [1, 2, 3, 4, 5]
    phi = sum(np.sin(m * X[k]) for k, m in enumerate(modes))
    lap = sum(-(m * m) * np.sin(m * X[k]) for k, m in enumerate(modes))
    got = s.helmholtz_rhs(phi, 0.01, mass2=2.0)
    assert np.max(np.abs(got - (0.01 * lap - 2.0 * phi))) < 1e-10


def test_5d_vector_divergence_and_curl(engines5):
    _, v = engines5
    X = _grid(SHAPE5)
    comp = tuple(np.sin(X[k]) for k in range(5))
    # d(sin(x_k))/d(x_k) = cos(x_k)
    exact_div = sum(np.cos(X[k]) for k in range(5))
    assert np.max(np.abs(v.divergence(comp) - exact_div)) < 1e-10
    # In 5D the curl is a 2-form with C(5,2) = 10 independent components.
    curl = v.curl(comp)
    assert len(curl) == 10 == v.CURL_COMPONENTS
    assert all(i < j for i, j, _ in curl)


def test_5d_leray_projection_is_divergence_free(engines5):
    _, v = engines5
    X = _grid(SHAPE5)
    comp = tuple(np.sin(X[k]) * np.cos(X[(k + 1) % 5]) for k in range(5))
    projected = v.leray_project(comp)
    assert np.max(np.abs(v.divergence(projected))) < 1e-10


def test_5d_advection_rhs_matches_analytic(engines5):
    _, v = engines5
    X = _grid(SHAPE5)
    comp = tuple(np.sin(X[k]) for k in range(5))
    got = v.advection_rhs(comp)
    # -(v . grad) v_k = -sin(x_k) * cos(x_k)
    exact = tuple(-np.sin(X[k]) * np.cos(X[k]) for k in range(5))
    assert max(np.max(np.abs(a - b)) for a, b in zip(got, exact)) < 1e-10


def test_5d_engines_reject_wrong_dimensionality():
    with pytest.raises(ValueError):
        FiveDScalarFieldEngine((8, 8, 8, 8))
    with pytest.raises(ValueError):
        FiveDVectorFieldEngine((8,) * 6)


def test_registry_exposes_5d_backends():
    reg = default_field_registry()
    assert sorted({b.dimension for b in reg._items.values()}) == [2, 3, 4, 5]
    assert [b.name for b in reg.by_dimension(5)] == ["scalar5d", "vector5d"]
    assert len(reg.describe()) == 7


# ==========================================================================
# Hardened sandbox
# ==========================================================================
def test_sandbox_blocks_reflection_builtins():
    from ags_sci.experiment.sandbox import ExecutionSandbox
    for payload in ["result = getattr(1, 'real')", "result = type(1)",
                    "result = vars()", "result = object()"]:
        with ExecutionSandbox(timeout_seconds=10) as sb:
            assert sb.execute(payload).status == "BLOCKED", payload


def test_sandbox_blocks_loop_fuel_tampering():
    from ags_sci.experiment.sandbox import ExecutionSandbox
    for payload in ["_ags_fuel = 10**12\nresult = 1",
                    "_ags_fuel += 10**12\nresult = 1",
                    "global _ags_fuel\nresult = 1"]:
        with ExecutionSandbox(timeout_seconds=10) as sb:
            assert sb.execute(payload).status == "BLOCKED", payload


def test_sandbox_address_space_cap_engages_when_budget_allows():
    """The RLIMIT_AS cap must actually bite, not silently no-op."""
    from ags_sci.experiment import sandbox as sb
    vm = sb._current_address_space_bytes()
    assert vm is not None, "expected a Linux /proc VmSize reading"
    # A 20 GiB allocation under a 4 GiB cap must be refused in-process.
    with sb.ExecutionSandbox(timeout_seconds=25, max_memory_mb=4096) as s:
        r = s.execute("import numpy as np\nresult = float(np.zeros(20*1024**3//8).sum())")
    assert r.status == "ERROR"
    assert "MemoryError" in (r.error or "")


def test_sandbox_still_runs_legitimate_code():
    from ags_sci.experiment.sandbox import ExecutionSandbox
    with ExecutionSandbox(timeout_seconds=10) as sb:
        r = sb.execute("import numpy as np\nresult = float(np.sum(np.arange(10)))")
    assert r.status == "SUCCESS" and r.value == 45.0


# ==========================================================================
# Plugin architecture
# ==========================================================================
class _ReadOnlyPlugin:
    name = "demo.readonly"
    version = "1.0"
    description = "read-only analysis demo"
    capabilities = frozenset({"analysis"})

    def install(self, service):
        self.service = service


class _PrivilegedPlugin:
    name = "demo.privileged"
    version = "1.0"
    description = "needs sandbox"
    capabilities = frozenset({"sandbox"})

    def install(self, service):
        self.service = service


def test_plugin_registry_accepts_valid_plugin():
    reg = PluginRegistry()
    rec = reg.register(_ReadOnlyPlugin())
    assert rec.name == "demo.readonly" and rec.privileged is False
    assert reg.by_capability("analysis") == ("demo.readonly",)


def test_plugin_registry_rejects_unknown_capability():
    class Bad:
        name = "bad"; version = "1"; description = ""
        capabilities = frozenset({"take_over"})
        def install(self, s): pass
    with pytest.raises(SecurityViolation):
        PluginRegistry().register(Bad())


def test_privileged_plugin_requires_explicit_opt_in():
    reg = PluginRegistry()
    with pytest.raises(SecurityViolation):
        reg.register(_PrivilegedPlugin())
    rec = reg.register(_PrivilegedPlugin(), allow_privileged=True)
    assert rec.privileged is True


def test_plugin_registry_rejects_duplicate_names():
    reg = PluginRegistry()
    reg.register(_ReadOnlyPlugin())
    with pytest.raises(ValueError):
        reg.register(_ReadOnlyPlugin())


def test_service_install_plugin_and_capability_discovery():
    svc = AGSService()
    rec = svc.install_plugin(_ReadOnlyPlugin())
    assert rec["name"] == "demo.readonly"
    caps = svc.capabilities()
    assert "demo.readonly" in caps["plugins"]
    assert caps["field_dimensions"] == [2, 3, 4, 5]
    assert len(caps["domain_primitives"]) == 25


# ==========================================================================
# Improved law discovery
# ==========================================================================
def test_library_construction_labels_and_columns():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Theta, labels = build_library(X, ("a", "b"), degree=2)
    assert labels == ("1", "a", "b", "a*a", "a*b", "b*b")
    assert Theta.shape == (2, 6)


def test_harmonic_oscillator_recovered_without_spurious_terms():
    t = np.linspace(0, 10, 600)
    X = np.column_stack([np.cos(t), -np.sin(t)])
    law = SparseLawDiscovery(degree=2).fit(t, X, ["x", "v"])
    assert law.equations[0] == "d(x)/dt = (1)*(v)"
    assert law.equations[1] == "d(v)/dt = (-1)*(x)"
    assert law.n_active_terms == 2
    assert law.evidence_tier == "PARSIMONIOUS"
    assert law.holdout_rmse < 1e-4


def test_logistic_growth_recovered():
    t = np.linspace(0, 6, 500)
    x = 1.0 / (1.0 + np.exp(-t))
    law = SparseLawDiscovery(degree=2).fit(t, x[:, None], ["x"])
    assert law.equations[0] == "d(x)/dt = (1)*(x) + (-1)*(x*x)"


def test_lorenz_system_recovered():
    def f(s):
        return np.array([10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1],
                         s[0] * s[1] - 8 / 3 * s[2]])

    def lorenz(n=1500, dt=0.005):
        # Midpoint (RK2): plain Euler is too inaccurate on this stiff chaotic
        # system, and the resulting trajectory is no longer the Lorenz system,
        # so the recovered law would not match the analytic one.
        s = np.array([1.0, 1.0, 1.0]); out = []
        for _ in range(n):
            out.append(s.copy())
            s = s + dt * f(s + 0.5 * dt * f(s))
        return np.array(out)
    L = lorenz(); tl = np.arange(len(L)) * 0.005
    law = SparseLawDiscovery(degree=2, holdout_fraction=0.25).fit(tl, L, ["x", "y", "z"])

    def coef(terms, label):
        return next(c for t, c in terms if t == label)

    # x' = 10*(y - x): the coefficients on (y) and (x) are opposite and ~10.
    cx, cy = coef(law.coefficients[0], "x"), coef(law.coefficients[0], "y")
    assert cy == pytest.approx(-cx, rel=0.05) and abs(cy) == pytest.approx(10.0, rel=0.05)
    # y' = 28*x - y - x*z
    assert coef(law.coefficients[1], "x") == pytest.approx(28.0, rel=0.05)
    assert coef(law.coefficients[1], "y") == pytest.approx(-1.0, rel=0.05)
    assert coef(law.coefficients[1], "x*z") == pytest.approx(-1.0, rel=0.05)
    # z' = x*y - (8/3)*z
    assert coef(law.coefficients[2], "x*y") == pytest.approx(1.0, rel=0.05)
    assert coef(law.coefficients[2], "z") == pytest.approx(-8 / 3, rel=0.05)
    assert law.holdout_rmse is not None


def test_law_discovery_rejects_short_trajectory():
    with pytest.raises(ValueError):
        SparseLawDiscovery().fit(np.arange(4), np.zeros((4, 1)), ["x"])


def test_law_discovery_is_deterministic():
    t = np.linspace(0, 8, 400)
    X = np.column_stack([np.cos(t), -np.sin(t)])
    a = SparseLawDiscovery().fit(t, X, ["x", "v"])
    b = SparseLawDiscovery().fit(t, X, ["x", "v"])
    assert a.equations == b.equations and a.bic == b.bic


def test_stlsq_relative_threshold_prunes_noise():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(200, 3)); A[:, 1] *= 1e-6   # a numerically dead column
    b = 2.0 * A[:, 0] + 0.5 * A[:, 2]
    coef, active = _stlsq(A, b, threshold=0.01)
    assert active[0] and active[2]
    assert not active[1], "a term 1e-6 of the dominant scale must be pruned"


# ==========================================================================
# Passive self-evolution
# ==========================================================================
def _trained_engine():
    e = PassiveEvolutionEngine({"threshold": (0.001, 0.5)}, min_observations=5)
    e.declare_default("threshold", 0.01)
    for v, score in [(0.01, 5e-2), (0.01, 6e-2), (0.05, 1e-3), (0.05, 2e-3),
                     (0.05, 1.5e-3), (0.02, 2e-2), (0.2, 4e-1), (0.5, 9e-1)]:
        e.observe(EvolutionObservation("threshold", v, "holdout_rmse", score))
    return e


def test_evolution_proposes_best_observed_parameter():
    prop = _trained_engine().propose("threshold")
    assert isinstance(prop, ParameterProposal)
    assert (prop.current_value, prop.proposed_value) == (0.01, 0.05)
    assert prop.supporting_observations == 8
    assert len(prop.fingerprint) == 64


def test_evolution_requires_sufficient_evidence():
    e = PassiveEvolutionEngine({"threshold": (0.001, 0.5)}, min_observations=5)
    e.declare_default("threshold", 0.01)
    e.observe(EvolutionObservation("threshold", 0.05, "holdout_rmse", 1e-9))
    assert e.propose("threshold") is None


def test_evolution_defaults_are_not_shared_between_instances():
    a = PassiveEvolutionEngine({"threshold": (0.001, 0.5)})
    b = PassiveEvolutionEngine({"threshold": (0.001, 0.5)})
    a.declare_default("threshold", 0.01)
    b.declare_default("threshold", 0.2)
    assert a._defaults == {"threshold": 0.01}
    assert b._defaults == {"threshold": 0.2}


def test_evolution_rejects_unknown_parameter_and_out_of_bounds():
    e = PassiveEvolutionEngine({"threshold": (0.001, 0.5)})
    with pytest.raises(SecurityViolation):
        e.observe(EvolutionObservation("nope", 0.1, "m", 1.0))
    with pytest.raises(SecurityViolation):
        e.observe(EvolutionObservation("threshold", 99.0, "m", 1.0))
    with pytest.raises(SecurityViolation):
        e.declare_default("threshold", 99.0)


def test_evolution_protected_paths_still_blocked():
    from ags_sci.core.evolution import EvolutionProposal
    p = EvolutionProposal(
        proposal_id="p1",
        changed_paths=("src/ags_sci/experiment/sandbox.py",),
        rationale="weaken the sandbox",
        baseline_sha256="0" * 64,
    )
    with pytest.raises(SecurityViolation):
        p.validate()


# ==========================================================================
# Quantum layer
# ==========================================================================
def test_quantum_bell_state_correlations():
    c = QuantumCircuit(2); c.h(0); c.cnot(0, 1)
    be = SimulatorBackend()
    p = be.probabilities(c)
    assert p[0b00] == pytest.approx(0.5) and p[0b11] == pytest.approx(0.5)
    assert p[0b01] == pytest.approx(0.0) and p[0b10] == pytest.approx(0.0)


def test_quantum_ghz_state():
    c = QuantumCircuit(3); c.h(0); c.cnot(0, 1); c.cnot(1, 2)
    p = SimulatorBackend().probabilities(c)
    assert p[0b000] == pytest.approx(0.5) and p[0b111] == pytest.approx(0.5)
    assert float(p.sum()) == pytest.approx(1.0)


def test_quantum_expectation_values():
    be = SimulatorBackend()
    c = QuantumCircuit(1); c.x(0)
    assert expectation(c, pauli_z(1, 0)) == pytest.approx(-1.0)
    c2 = QuantumCircuit(1)                     # |0>
    assert expectation(c2, pauli_z(1, 0)) == pytest.approx(1.0)
    c3 = QuantumCircuit(1); c3.h(0)            # |+>
    assert expectation(c3, pauli_z(1, 0)) == pytest.approx(0.0)


def test_quantum_pauli_string_tensor_product():
    P = pauli_string(2, {0: "X", 1: "Z"})
    assert np.allclose(P, np.kron(np.array([[0, 1], [1, 0]]), np.diag([1, -1])))


def test_quantum_vqe_recovers_known_ground_state():
    # H = -Z on qubit 0, tensor identity elsewhere -> ground energy -1.
    H = -pauli_z(3, 0)
    res = vqe_ground_energy(H, 3, restarts=3)
    assert res["energy"] == pytest.approx(-1.0, abs=1e-6)


def test_quantum_walk_is_normalised_and_starts_as_delta():
    p0 = quantum_walk_mixing(8, 0)
    assert np.allclose(p0, np.eye(8)[0])
    for n, k in [(4, 1), (8, 20), (16, 50)]:
        p = quantum_walk_mixing(n, k)
        assert p.sum() == pytest.approx(1.0)
        assert p.min() >= 0.0


def test_quantum_backend_reports_that_it_is_a_simulation():
    meta = SimulatorBackend().metadata()
    assert meta.simulated is True
    assert "no quantum speedup" in meta.note.lower() or "2^n" in meta.note


def test_quantum_circuit_validates_targets():
    with pytest.raises(ValueError):
        QuantumCircuit(2).cnot(0, 0)
    with pytest.raises(ValueError):
        QuantumCircuit(2).h(5)


# ==========================================================================
# Service facade
# ==========================================================================
def test_service_capabilities_are_complete():
    caps = AGSService().capabilities()
    assert caps["version"] == ags_sci.__version__
    assert caps["sandbox_available"] and caps["quantum_available"]
    assert set(caps["field_dimensions"]) == {2, 3, 4, 5}


def test_capabilities_advertise_invention_with_an_epistemic_note():
    caps = AGSService().capabilities()
    assert caps["invention_available"] is True
    # The note must state the boundary, not just that the feature exists.
    note = caps["epistemic_note"].lower()
    assert "novel only relative" in note
    assert "law of nature" in note


def test_service_field_factory_rejects_dimension_mismatch():
    svc = AGSService()
    with pytest.raises(Exception):
        svc.field("turbulence2d", (8, 8, 8))


def test_service_discover_law_uses_improved_pipeline():
    svc = AGSService()
    t = np.linspace(0, 10, 600)
    X = np.column_stack([np.cos(t), -np.sin(t)])
    law = svc.discover_law(t, X, ["x", "v"])
    d = law.describe()
    assert d["evidence_tier"] == "PARSIMONIOUS"
    assert d["n_active_terms"] == 2
    assert d["equations"][0] == "d(x)/dt = (1)*(v)"


def test_service_run_code_is_fail_closed():
    svc = AGSService()
    assert svc.run_code("result = 1 + 1")["status"] == "SUCCESS"
    assert svc.run_code("import os\nresult = 1")["status"] == "BLOCKED"


def test_service_evolution_engine_round_trip():
    svc = AGSService()
    e = svc.evolution_engine({"threshold": (0.001, 0.5)}, min_observations=3)
    e.declare_default("threshold", 0.01)
    for v, s_ in [(0.01, 1e-1), (0.01, 1e-1), (0.05, 1e-4), (0.05, 2e-4)]:
        e.observe(EvolutionObservation("threshold", v, "holdout_rmse", s_))
    prop = e.propose("threshold")
    assert prop is not None and prop.proposed_value == 0.05
