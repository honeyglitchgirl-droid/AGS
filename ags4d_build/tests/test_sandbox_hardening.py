import os
from pathlib import Path

from ags_sci.experiment.sandbox import SandboxLimits, SandboxedRunner


def test_callable_worker_sanitizes_environment_and_cwd():
    original = os.environ.get("AGS_TEST_SECRET")
    os.environ["AGS_TEST_SECRET"] = "should-not-leak"
    try:
        result = SandboxedRunner().run(lambda: (os.environ.get("AGS_TEST_SECRET"), os.getcwd()))
    finally:
        if original is None:
            os.environ.pop("AGS_TEST_SECRET", None)
        else:
            os.environ["AGS_TEST_SECRET"] = original
    assert result.ok
    secret, cwd = result.value
    assert secret is None
    assert "ags_sandbox_" in cwd


def test_source_blocks_dangerous_imports():
    result = SandboxedRunner().run_source("import os\nresult = os.getcwd()")
    assert not result.ok
    assert "blocked import: os" in result.error


def test_source_blocks_open():
    result = SandboxedRunner().run_source("result = open('/tmp/x', 'w')")
    assert not result.ok
    assert "blocked builtin: open" in result.error


def test_source_allows_numeric_compute():
    result = SandboxedRunner().run_source("import math\nresult = math.sqrt(16)")
    assert result.ok
    assert result.value == 4.0


def test_source_loop_fuel():
    result = SandboxedRunner(SandboxLimits(timeout_s=2, max_loop_iterations=100)).run_source(
        "x = 0\nwhile True:\n    x += 1\nresult = x"
    )
    assert not result.ok
    assert "fuel exhausted" in result.error


def test_timeout_still_terminates():
    result = SandboxedRunner(SandboxLimits(timeout_s=0.2)).run_source(
        "while True:\n    pass\nresult = 1"
    )
    assert not result.ok
    assert result.error in {"timeout", "SandboxViolation: ..."} or "fuel exhausted" in (result.error or "")
