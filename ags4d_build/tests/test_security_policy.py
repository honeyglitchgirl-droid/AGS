import math
import pytest

from ags_sci.core.security import (
    SecurityViolation, bounded_text, validate_payload, validate_simulations,
    sha256_json, finite_number, safe_identifier,
)


def test_payload_is_json_like_and_finite():
    assert validate_payload({"x": [1, 2.5, "ok"], "b": True})["x"][1] == 2.5
    with pytest.raises(SecurityViolation):
        validate_payload(float("nan"))
    with pytest.raises(SecurityViolation):
        validate_payload(object())


def test_payload_bounds():
    with pytest.raises(SecurityViolation):
        bounded_text("x" * 20_001)
    with pytest.raises(SecurityViolation):
        validate_payload({str(i): i for i in range(129)})


def test_search_budget_fail_closed():
    with pytest.raises(SecurityViolation):
        validate_simulations(0)
    with pytest.raises(SecurityViolation):
        validate_simulations(10_001)
    assert validate_simulations(100) == 100


def test_integrity_hash_is_canonical():
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})


def test_finite_and_identifiers():
    assert finite_number(1.25, field="x") == 1.25
    with pytest.raises(SecurityViolation):
        finite_number(math.inf, field="x")
    assert safe_identifier("exp-01:v2") == "exp-01:v2"
    with pytest.raises(SecurityViolation):
        safe_identifier("../secret")


def test_evolution_gate_protects_security_boundaries():
    from ags_sci.core.evolution import EvolutionProposal
    from ags_sci.core.security import SecurityViolation
    ok = EvolutionProposal("exp-1", ("src/ags_sci/fields/new_plugin.py",), "improve solver", "0" * 64)
    ok.validate()
    with pytest.raises(SecurityViolation):
        EvolutionProposal("exp-2", ("src/ags_sci/experiment/sandbox.py",), "weaken limits", "0" * 64).validate()


def test_evolution_gate_rejects_path_traversal():
    from ags_sci.core.evolution import EvolutionProposal
    from ags_sci.core.security import SecurityViolation
    with pytest.raises(SecurityViolation):
        EvolutionProposal("exp-3", ("../escape.py",), "escape", "0" * 64).validate()
