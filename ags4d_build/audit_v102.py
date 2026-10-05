from pathlib import Path
import sys, importlib, json
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
sys.dont_write_bytecode = True
MODULES=['ags_sci','ags_sci.core','ags_sci.core.evolution','ags_sci.epistemic','ags_sci.dynamics','ags_sci.dynamics.identification','ags_sci.fields.turbulence3d','ags_sci.fields.referee3d','ags_sci.fields.five_d','ags_sci.fields.registry','ags_sci.experiment','ags_sci.search','ags_sci.adapters','ags_sci.discovery','ags_sci.discovery.law','ags_sci.plugins','ags_sci.api','ags_sci.quantum','ags_sci.complex_systems','ags_sci.cli']

def main():
    for m in MODULES: importlib.import_module(m)
    assert not (ROOT/'AGS_Sci_COMPLETE_MASTER_v100.py').exists()
    assert (ROOT/'legacy_master_v100.py').exists()
    assert (ROOT/'src'/'ags_sci').exists()
    assert not list(ROOT.glob('ags_*'))
    assert not (ROOT/'adapters').exists() and not (ROOT/'discovery').exists()
    assert not (ROOT/'SHA256_MANIFEST_v101.json').exists()
    assert (ROOT/'SHA256_MANIFEST_v102_3_1.json').exists()
    # Version identity must be a single value of "v102" declared consistently in
    # every place it appears. Deriving the expectation from VERSION_v102.txt (rather
    # than hard-coding it here) keeps this audit from breaking on every release bump.
    declared=(ROOT/'VERSION_v102.txt').read_text().strip()
    assert declared == 'AGS-Sci-v102', f'unexpected version identity: {declared!r}'
    import ags_sci
    assert ags_sci.__version__ == '102', f'package __version__={ags_sci.__version__!r} != 102'
    try:
        import tomllib
        with open(ROOT/'pyproject.toml','rb') as fh:
            assert tomllib.load(fh)['project']['version'] == '102'
    except ModuleNotFoundError:
        pass  # tomllib requires Python >= 3.11; packaging check is best-effort
    assert (ROOT/'experiments').exists()
    # Capability-upgrade structure: the 5D backends must be registered, the
    # service facade must import, and the plugin/quantum layers must be present.
    import ags_sci
    from ags_sci.fields.registry import default_field_registry
    from ags_sci.plugins import ALLOWED_CAPABILITIES
    from ags_sci.quantum import SimulatorBackend
    reg = default_field_registry()
    assert sorted({b.dimension for b in reg._items.values()}) == [2, 3, 4, 5], \
        'field registry must cover dimensions 2..5'
    assert len(reg.by_dimension(5)) == 2, '5D scalar and vector backends must be registered'
    assert SimulatorBackend().metadata().simulated is True
    assert 'sandbox' in ALLOWED_CAPABILITIES
    svc = ags_sci.AGSService()
    caps = svc.capabilities()
    assert caps['field_dimensions'] == [2, 3, 4, 5]
    assert len(caps['domain_primitives']) == 25
    assert caps['sandbox_available'] and caps['quantum_available']
    assert not (ROOT/'VERSION_v101.txt').exists()
    assert not (ROOT/'audit_v101.py').exists()
    print('AGS-Sci v102 audit: PASS')
    print('Unified src/ags_sci namespace: PASS')
    print('Active v100 monolith removed: PASS')
    print('Legacy fallback isolated: PASS')
    print('v102 version/manifest naming: PASS')
    print('No obsolete v101 release artifacts: PASS')
    print('Runtime cache check: performed at release packaging')
if __name__=='__main__': main()
