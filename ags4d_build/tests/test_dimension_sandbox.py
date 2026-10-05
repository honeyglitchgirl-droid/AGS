import numpy as np
import pytest

from ags_sci.experiment import ExecutionSandbox, ExperimentSpec, admit


def test_all_dimensions_are_admitted_at_safe_sizes():
    shapes = {1: (128,), 2: (32, 32), 3: (16, 16, 16), 4: (8, 8, 8, 8)}
    for dim, shape in shapes.items():
        a = admit(ExperimentSpec(dim, shape, fields=2, dtype_bytes=4, steps=10, runtime_s=1))
        assert a.allowed, (dim, a.reason)


def test_dimension_mismatch_rejected():
    with pytest.raises(ValueError):
        admit(ExperimentSpec(3, (8, 8)))


def test_four_d_cell_budget_rejected():
    a = admit(ExperimentSpec(4, (32, 32, 32, 32), fields=1, dtype_bytes=4, steps=1, runtime_s=1))
    assert not a.allowed
    assert "cell budget" in a.reason


def test_memory_estimate_is_conservative():
    a = admit(ExperimentSpec(3, (16, 16, 16), fields=4, dtype_bytes=8, steps=1, runtime_s=1))
    assert a.allowed
    assert a.estimated_memory_mb > 0


def test_sandbox_executes_dimension_agnostic_numpy_code():
    code = """
import numpy as np
for dim, shape in [(1,(8,)), (2,(4,4)), (3,(3,3,3)), (4,(2,2,2,2))]:
    x = np.ones(shape, dtype=np.float32)
    assert x.ndim == dim
result = True
"""
    with ExecutionSandbox(timeout_seconds=5, loop_fuel=10000) as sb:
        r = sb.execute_experiment(ExperimentSpec(4, (2, 2, 2, 2), steps=10, runtime_s=1), code)
    assert r.status == "SUCCESS", r.error
    assert r.value is True


def test_admission_rejects_before_running_code():
    code = "result = 123"
    with ExecutionSandbox(timeout_seconds=2) as sb:
        r = sb.execute_experiment(ExperimentSpec(4, (64, 64, 64, 64), steps=1, runtime_s=1), code)
    assert r.status == "REJECTED"
    assert "cell budget" in r.error
