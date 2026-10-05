from ags_sci.experiment.sandbox import ExecutionSandbox

def test_controlled_internet_fetch():
    code = '''result = internet_fetch("https://example.com")\n'''
    with ExecutionSandbox(timeout_seconds=12, internet_enabled=True,
                          internet_allowed_domains=("example.com",)) as sandbox:
        r = sandbox.execute(code)
    # CI/containers may intentionally have no DNS or egress.  In that case
    # the capability must fail closed, not make the deterministic suite flaky.
    if r.status == "SUCCESS":
        assert r.value["status"] == 200
        assert "Example Domain" in r.value["text"]
        assert len(r.value["sha256"]) == 64
    else:
        assert r.error and ("gaierror" in r.error or "URLError" in r.error or "timed out" in r.error)

def test_internet_disabled_by_default():
    code = '''result = internet_fetch("https://example.com")\n'''
    with ExecutionSandbox(timeout_seconds=2) as sandbox:
        r = sandbox.execute(code)
    assert r.status in {"ERROR", "TIMEOUT"}
