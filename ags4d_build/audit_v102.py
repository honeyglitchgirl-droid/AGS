from pathlib import Path
import sys, importlib, json
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
sys.dont_write_bytecode = True
MODULES=['ags_sci','ags_sci.core','ags_sci.epistemic','ags_sci.dynamics','ags_sci.dynamics.identification','ags_sci.fields.turbulence3d','ags_sci.fields.referee3d','ags_sci.experiment','ags_sci.search','ags_sci.adapters','ags_sci.discovery','ags_sci.cli']

def main():
    for m in MODULES: importlib.import_module(m)
    assert not (ROOT/'AGS_Sci_COMPLETE_MASTER_v100.py').exists()
    assert (ROOT/'legacy_master_v100.py').exists()
    assert (ROOT/'src'/'ags_sci').exists()
    assert not list(ROOT.glob('ags_*'))
    assert not (ROOT/'adapters').exists() and not (ROOT/'discovery').exists()
    assert not (ROOT/'SHA256_MANIFEST_v101.json').exists()
    assert (ROOT/'SHA256_MANIFEST_v102_3_1.json').exists()
    assert (ROOT/'VERSION_v102.txt').read_text().strip() == 'AGS-Sci-v102.3.1'
    assert (ROOT/'experiments').exists()
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
