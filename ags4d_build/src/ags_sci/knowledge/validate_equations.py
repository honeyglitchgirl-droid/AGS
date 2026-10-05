"""Command-line validation for the packaged equation corpus.

Run with::

    python -m ags_sci.knowledge.validate_equations
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .equations.validation import validate_corpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate AGS-Sci equation reference data")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="alternate corpus reference root (defaults to packaged data)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the deterministic report as JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate_corpus(args.root)
    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(f"Equation corpus: {'PASS' if report.valid else 'FAIL'}")
        print(f"Records: {report.record_count}")
        print(f"Mathematics: {report.domain_counts.get('mathematics', 0)}")
        print(f"Physics: {report.domain_counts.get('physics', 0)}")
        for level in ("k12", "undergraduate", "graduate"):
            counts = report.level_domain_counts.get(level, {})
            print(
                f"{level}: mathematics={counts.get('mathematics', 0)}, "
                f"physics={counts.get('physics', 0)}"
            )
        print(f"Data files: {report.file_count}")
        print(f"Corpus bytes: {report.total_bytes}")
        print(f"Duplicate IDs: {report.duplicate_ids}")
        print(f"Duplicate formulas: {report.duplicate_formulas}")
        print(
            "Whitespace-normalized duplicate formulas: "
            f"{report.normalized_duplicate_formulas}"
        )
        print(f"Duplicate records: {report.duplicate_records}")
        print(f"Malformed records: {report.malformed_records}")
        for issue in report.issues:
            print(f"{issue.code}: {issue.location}: {issue.message}")
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
