from ags_sci.experiment.sandbox import ExecutionSandbox


def test_warm_worker_numeric_execution_and_imports():
    with ExecutionSandbox(timeout_seconds=2) as s:
        r = s.execute("import math\nresult = math.sqrt(16)")
        assert r.status == "SUCCESS"


def test_warm_worker_blocks_host_access():
    with ExecutionSandbox(timeout_seconds=2) as s:
        assert s.execute("import os\nresult = os.environ").status == "BLOCKED"
        assert s.execute("result = open('/tmp/x', 'w')").status == "BLOCKED"


def test_warm_worker_repeated_execution():
    with ExecutionSandbox(timeout_seconds=2) as s:
        results = [s.execute("result = 40 + 2") for _ in range(5)]
        assert all(r.status == "SUCCESS" and r.error is None for r in results)
