#!/usr/bin/env python3
"""Benchmark repeated AGS-Sci v102 warm-worker sandbox executions."""
from __future__ import annotations
import statistics
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ags_sci.experiment.sandbox import ExecutionSandbox


def main() -> int:
    iterations = 50
    samples = []
    startup_t0 = time.perf_counter()
    sandbox = ExecutionSandbox(timeout_seconds=2.0)
    startup_ms = (time.perf_counter() - startup_t0) * 1000.0
    try:
        warm = sandbox.execute("result = 2 + 2")
        if warm.status != "SUCCESS":
            raise SystemExit(f"warm-up failed: {warm}")
        for _ in range(iterations):
            t0 = time.perf_counter()
            result = sandbox.execute("result = 2 + 2")
            dt = (time.perf_counter() - t0) * 1000.0
            if result.status != "SUCCESS":
                raise SystemExit(f"benchmark execution failed: {result}")
            samples.append(dt)
    finally:
        sandbox.close()
    print("AGS-Sci v102 warm-worker benchmark")
    print(f"iterations: {iterations}")
    print(f"startup_ms: {startup_ms:.3f}")
    print(f"mean_ms: {statistics.mean(samples):.3f}")
    print(f"median_ms: {statistics.median(samples):.3f}")
    print(f"p95_ms: {statistics.quantiles(samples, n=20)[18]:.3f}")
    print(f"min_ms: {min(samples):.3f}")
    print(f"max_ms: {max(samples):.3f}")
    print(f"under_50ms: {sum(x < 50 for x in samples)}/{iterations}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
