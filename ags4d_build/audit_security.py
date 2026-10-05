"""Application-level AGS-Sci security audit. Fail closed."""
from __future__ import annotations
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'src'/'ags_sci'

# These are permitted only in the sandbox implementation itself.
DANGEROUS={'os.system','subprocess.Popen','subprocess.run','eval','exec','pickle.loads','marshal.loads'}
FAILURES=[]
for p in SRC.rglob('*.py'):
    text=p.read_text(errors='replace')
    try: tree=ast.parse(text)
    except SyntaxError as e:
        FAILURES.append(f'{p}: syntax error: {e}'); continue
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {'eval','exec'} and p.name!='sandbox.py':
            FAILURES.append(f'{p}: dynamic {n.func.id} outside sandbox')
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            q=[]; cur=n.func
            while isinstance(cur, ast.Attribute): q.append(cur.attr); cur=cur.value
            if isinstance(cur, ast.Name): q.append(cur.id)
            dotted='.'.join(reversed(q))
            if dotted in DANGEROUS and p.name!='sandbox.py': FAILURES.append(f'{p}: {dotted}')
if FAILURES:
    print('SECURITY AUDIT: FAIL')
    print('\n'.join(FAILURES))
    raise SystemExit(1)
print('SECURITY AUDIT: PASS')
