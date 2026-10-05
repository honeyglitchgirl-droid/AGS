import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('legacy_master_v100', ROOT/'legacy_master_v100.py')
M=importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
def test_legacy_bridge_compatibility():
    r=M.v60_integration_self_test()
    assert all(r.values()), r
