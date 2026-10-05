"""Minimal CLI for the active-inference research stack."""
from __future__ import annotations
import argparse

def main(argv=None):
    p=argparse.ArgumentParser(prog='ags-sci')
    p.add_argument('--self-check',action='store_true',help='run the active-inference component checks')
    p.add_argument('--forced-referee', action='store_true', help='run the integrated EXP-2D-TURB-003-B forced 2-D referee')
    p.add_argument('--forced-tmax', type=float, default=0.1, help='short referee duration; use the library API for production windows')
    p.add_argument('--forced-sigma', type=float, default=5615.0)
    p.add_argument('--forced-N', type=int, default=32)
    args=p.parse_args(argv)
    if args.forced_referee:
        from ags_sci.fields.forced_turbulence2d import ForcedRFFTTurbulence2D
        solver = ForcedRFFTTurbulence2D(N=args.forced_N, sigma_f=args.forced_sigma, workers=1)
        solver.begin_audit_window()
        solver.advance(args.forced_tmax)
        audit = solver.audit()
        print(f'EXP={audit.experiment}')
        print(f'J_budget={audit.J_budget:.8e} [{"PASS" if audit.budget_pass else "FAIL"}]')
        print(f'J_stat={audit.J_stat:.8f} [{"PASS" if audit.stationarity_pass else "FAIL"}]')
        print(f'R_tail={audit.max_r_tail:.8e} [{"PASS" if audit.admissibility_pass else "FAIL"}]')
        return 0 if audit.certified else 2
    if args.self_check:
        from ags_sci.discovery.loop import ResearchLoop
        from ags_sci.epistemic import PolicyEvaluation, BOED
        policies=BOED.rank([PolicyEvaluation('baseline',1.0,0.0,0.0),PolicyEvaluation('explore',-0.5,0.7,-0.2)])
        print(policies[0].policy)
        return 0
    p.print_help(); return 0

if __name__=='__main__':
    raise SystemExit(main())
