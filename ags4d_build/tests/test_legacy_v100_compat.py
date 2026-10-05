import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "legacy_master_v100.py"

def load():
    spec = importlib.util.spec_from_file_location("ags_v100_test", MASTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Import the cumulative master once; legacy modules include deterministic import-time
# diagnostics, so repeated imports only add latency without adding coverage.
M = load()

OLD = [
    "v56_self_test","v58_self_test","v59_self_test","session1_self_test","session1_extended_tests",
    "session2_self_test","session2_hardening_tests","session3_self_test","session3_hardening_tests",
    "ai_hypothesis_generation_tests","run_all_tests","v60_integration_self_test","v61_self_test",
    "v61_1_benchmark_self_test","v64_self_test","v64_hardening_tests","v65_self_test","v65_hardening_tests",
    "v66_self_test","v67_adversarial_tests","v68_identifiability_tests","v69_breakthrough_tests",
    "v69_stress_tests","v70_independent_challenge_tests","v71_regime_tests","v75_self_evolution_tests",
]
NEW = [f"v{i}_self_test" for i in range(76,100)] + ["v100_self_test","v100_edge_case_tests"]

def assert_result(result):
    ignored={"version","finding_count","module_count","domain_count","claim_boundary"}
    assert all(bool(v) for k,v in result.items() if k not in ignored), result

def test_all_version_suites():
    m=M
    for name in OLD+NEW:
        assert_result(getattr(m,name)())

def test_final_audit():
    m=M; a=m.v100_final_audit()
    assert a["version"] == m.AGS_V100_VERSION
    assert a["eval_calls"] == 0
    assert a["exec_calls"] == 0
    assert a["top_level_main_blocks"] == 1
    assert a["all_cumulative_tests_pass"] is True
    assert a["safety_pass"] is True
    assert a["reproducibility_pass"] is True
    assert a["evolution_gate_pass"] is True
    assert m.VERSION == m.AGS_V100_VERSION

def test_random_attention_and_derivative_controls():
    m=M; rng=np.random.default_rng(123)
    for _ in range(100):
        qn=int(rng.integers(1,7)); kn=int(rng.integers(1,7)); d=int(rng.choice([2,4,8])); od=int(rng.choice([2,5]))
        q=rng.normal(size=(2,qn,d)); k=rng.normal(size=(2,kn,d)); v=rng.normal(size=(2,kn,od))
        out=m.V86DeepLearningLab.scaled_dot_attention(q,k,v)
        assert out.shape==(2,qn,od) and np.all(np.isfinite(out))
    for x in np.linspace(-3,3,17):
        got=m.V82NumericalLab.finite_difference(lambda z:z**3+2*z,x)
        assert abs(got-(3*x*x+2)) < 1e-4

def test_hardening_cases():
    m=M; rng=np.random.default_rng(7)
    q=rng.normal(size=(1,2,4)); k=rng.normal(size=(1,3,4)); v=rng.normal(size=(1,3,5))
    try:
        m.V86DeepLearningLab.scaled_dot_attention(q,k,v,mask=np.array([[[True,False,False],[False,False,False]]]))
    except ValueError:
        pass
    else:
        raise AssertionError("fully masked attention was not rejected")
    try:
        m.V81TimeSeriesLab.dominant_frequency(np.ones(32))
    except ValueError:
        pass
    else:
        raise AssertionError("constant spectrum was not rejected")
