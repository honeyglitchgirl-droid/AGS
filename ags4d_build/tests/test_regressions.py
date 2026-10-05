"""Regression tests for bugs found in the v102 audit.

Each test pins a defect that previously passed silently (or crashed) while the
existing 144-test suite still reported green.
"""
import importlib.util
import math
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


_LEGACY = None


def _load_legacy():
    """Import the legacy master once; it runs import-time diagnostics."""
    global _LEGACY
    if _LEGACY is None:
        spec = importlib.util.spec_from_file_location("ags_v100_regressions", ROOT / "legacy_master_v100.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _LEGACY = mod
    return _LEGACY


# --------------------------------------------------------------------------
# 1. RobustDifferentiator.derivative raised NameError on its default path
#    because `savgol_filter` was used but never imported.
# --------------------------------------------------------------------------
def test_robust_differentiator_default_path_does_not_raise():
    from ags_sci.dynamics.identification import RobustDifferentiator

    t = np.linspace(0, 1, 200)
    y = np.sin(2 * np.pi * t)
    out = RobustDifferentiator().derivative(t, y)
    assert np.all(np.isfinite(out))
    assert float(np.max(np.abs(out - 2 * np.pi * np.cos(2 * np.pi * t)))) < 1e-3


def test_robust_differentiator_falls_back_without_scipy(monkeypatch):
    import ags_sci.dynamics.identification as ident

    monkeypatch.setattr(ident, "savgol_filter", None)
    t = np.linspace(0, 1, 200)
    y = np.sin(2 * np.pi * t)
    out = ident.RobustDifferentiator().derivative(t, y)
    assert np.all(np.isfinite(out))
    assert float(np.max(np.abs(out - 2 * np.pi * np.cos(2 * np.pi * t)))) < 1e-3


def test_robust_differentiator_nonuniform_sampling():
    from ags_sci.dynamics.identification import RobustDifferentiator

    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 1, 200))
    out = RobustDifferentiator().derivative(t, np.sin(2 * np.pi * t))
    assert np.all(np.isfinite(out))


# --------------------------------------------------------------------------
# 2. `navier_stokes_control` was referenced but never defined, which made
#    v61_open_problem_benchmark() raise NameError and silently forced two
#    V62_DIAGNOSTIC flags (domain_run, failure_isolated) to False.
# --------------------------------------------------------------------------
def test_navier_stokes_control_is_defined_and_reproduces_analytic_decay():
    m = _load_legacy()
    d = m.navier_stokes_control(T=1.0, steps=50).out()
    assert d["status"] == "CONTROL_ONLY"
    # Taylor-Green vorticity decays exactly as exp(-2*nu*t); the spectral solver
    # must reproduce it to near machine precision.
    assert d["details"]["rel_l2_error"] < 1e-10


def test_navier_stokes_control_is_stable_at_coarse_budgets():
    m = _load_legacy()
    d = m.navier_stokes_control(T=0.5, steps=10).out()
    assert math.isfinite(d["details"]["rel_l2_error"])
    assert d["details"]["rel_l2_error"] < 1e-10


def test_v61_open_problem_benchmark_does_not_raise():
    m = _load_legacy()
    results = m.v61_open_problem_benchmark()
    assert len(results) == 7
    assert all(hasattr(r, "status") for r in results)


def test_v62_diagnostic_domain_flags_all_true():
    m = _load_legacy()
    # These two were False purely because navier_stokes_control was missing.
    assert m.V62_DIAGNOSTIC["domain_run"] is True
    assert m.V62_DIAGNOSTIC["failure_isolated"] is True


def test_navier_stokes_control_rejects_invalid_budget():
    m = _load_legacy()
    with pytest.raises(ValueError):
        m.navier_stokes_control(T=0.0, steps=10)


# --------------------------------------------------------------------------
# 3. complex_systems.REGISTRY advertised 24 of 25 nonexistent callables.
# --------------------------------------------------------------------------
def test_registry_maps_every_domain_to_a_real_callable():
    from ags_sci import complex_systems as cs

    assert len(cs.REGISTRY) == len(cs.DOMAINS) == 25
    assert tuple(d.name for d in cs.REGISTRY) == cs.DOMAINS
    for entry in cs.REGISTRY:
        assert callable(cs.resolve(entry.name)), entry.name


def test_registry_unknown_domain_raises():
    from ags_sci import complex_systems as cs

    with pytest.raises(KeyError):
        cs.resolve("definitely_not_a_domain")


# --------------------------------------------------------------------------
# 4. scaled_dot_product_attention returned NaN for a fully-masked query row.
# --------------------------------------------------------------------------
def test_attention_rejects_fully_masked_row():
    from ags_sci.complex_systems import scaled_dot_product_attention

    with pytest.raises(ValueError):
        scaled_dot_product_attention(np.eye(2), np.eye(2), np.eye(2), mask=np.zeros((2, 2), bool))


def test_attention_rejects_bad_mask_shape():
    from ags_sci.complex_systems import scaled_dot_product_attention

    with pytest.raises(ValueError):
        scaled_dot_product_attention(np.eye(2), np.eye(2), np.eye(2), mask=np.ones((3, 2), bool))


def test_attention_valid_mask_still_works():
    from ags_sci.complex_systems import scaled_dot_product_attention

    _, w = scaled_dot_product_attention(
        np.eye(2), np.eye(2), np.eye(2), mask=np.array([[True, False], [False, True]])
    )
    assert np.allclose(w.sum(axis=1), 1.0)
    assert not np.any(np.isnan(w))


# --------------------------------------------------------------------------
# 5. christoffel_symbols was only tested against the trivial zero-derivative
#    case, which cannot detect an index-convention error.
# --------------------------------------------------------------------------
def test_christoffel_symbols_matches_known_frw_metric():
    from ags_sci.complex_systems import christoffel_symbols

    # FLRW metric g = diag(-1, a(t)^2, a(t)^2, a(t)^2), a(t)=1+t^2, at t=0.5.
    # Analytic: Gamma^0_ii = a*a', Gamma^i_0i = a'/a.
    a, ap = 1.25, 1.0
    g = np.diag([-1.0, a * a, a * a, a * a])
    dg = np.zeros((4, 4, 4))
    for i in range(1, 4):
        dg[0, i, i] = 2 * a * ap  # d(a^2)/dt

    G = christoffel_symbols(g, dg)
    for i in range(1, 4):
        assert G[0, i, i] == pytest.approx(a * ap, abs=1e-12)
        assert G[i, 0, i] == pytest.approx(ap / a, abs=1e-12)
        assert G[i, i, 0] == pytest.approx(ap / a, abs=1e-12)
    # Nothing else should be populated.
    expected = np.zeros((4, 4, 4))
    for i in range(1, 4):
        expected[0, i, i] = a * ap
        expected[i, 0, i] = ap / a
        expected[i, i, 0] = ap / a
    assert np.allclose(G, expected, atol=1e-12)


# --------------------------------------------------------------------------
# 6. gdg_singularity_diagnostic.py used `np` in main() without importing it.
# --------------------------------------------------------------------------
def test_gdg_diagnostic_script_imports_numpy_at_module_scope():
    src = (ROOT / "scripts" / "gdg_singularity_diagnostic.py").read_text(encoding="utf-8")
    # Only inspect the module-level part, i.e. before the PAYLOAD string.
    module_part = src.split("PAYLOAD=", 1)[0]
    assert re.search(r"^import numpy as np$", module_part, re.M), module_part


# --------------------------------------------------------------------------
# 7. The release identity was spread across 25+ inconsistent version strings
#    and audit_v102.py hard-coded one of them.
# --------------------------------------------------------------------------
def test_release_identity_is_consistent_everywhere():
    import ags_sci

    version_file = (ROOT / "VERSION_v102.txt").read_text(encoding="utf-8").strip()
    assert version_file == "AGS-Sci-v102"
    assert ags_sci.__version__ == "102"

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "102"$', pyproject, re.M)


def test_audit_v102_does_not_hardcode_a_patch_version():
    src = (ROOT / "audit_v102.py").read_text(encoding="utf-8")
    assert not re.search(r"AGS-Sci-v102\.\d", src)
    assert not re.search(r"== '102\.\d", src)


def test_no_stray_patch_level_version_strings_in_active_sources():
    """No active source/doc may still declare a v102.x.y release identity."""
    skip = ("legacy_master_v100.py",)
    offenders = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in (".py", ".md", ".toml", ".txt"):
            continue
        rel = str(path.relative_to(ROOT))
        if any(part in (".venv", "__pycache__", ".pytest_cache") for part in path.parts):
            continue
        if path.name.startswith("SHA256_MANIFEST") or path.name in skip:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"\bv102\.\d+(\.\d+)?\b", text):
            offenders.append(rel)
    assert offenders == [], f"stale version strings remain in: {offenders}"
