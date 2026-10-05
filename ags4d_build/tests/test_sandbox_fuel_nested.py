from ags_sci.experiment.sandbox import ExecutionSandbox

def test_nested_function_loop_fuel_no_unboundlocal():
    code = '''\ndef inner(n):\n    x = 0\n    for i in range(n):\n        x += i\n    return x\nresult = inner(10)\n'''
    with ExecutionSandbox(timeout_seconds=2, loop_fuel=100) as sandbox:
        r = sandbox.execute(code)
    assert r.status == 'SUCCESS', r.error
    assert r.value == 45
