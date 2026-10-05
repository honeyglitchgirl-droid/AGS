#!/usr/bin/env python3
"""Run the AGS-Sci v102 repository audit and regression suite."""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main() -> int:
    sys.path.insert(0, str(ROOT / 'src'))
    import audit_v102
    try:
        audit_v102.main()
    except Exception as exc:
        print(f'v102 structural audit failed: {type(exc).__name__}: {exc}')
        return 1
    env=dict(os.environ)
    env['PYTHONDONTWRITEBYTECODE']='1'
    test=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,env=env)
    if test.returncode != 0:
        return test.returncode
    print('AGS-Sci v102.3.1 full audit: PASS')
    return 0
if __name__=='__main__': raise SystemExit(main())
