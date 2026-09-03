"""
ANTAM FS 30 Juni 2026 — Benchmark entry point (PRD §25-26)

Alternative location per task: benchmarks/ANTAM_benchmark.py

This wraps fast_rag.benchmarks.benchmark.Benchmark to ensure
`python benchmarks/ANTAM_benchmark.py` and
`python -m fast_rag.benchmarks.benchmark` both work.

Usage:
  python benchmarks/ANTAM_benchmark.py
  python benchmarks/ANTAM_benchmark.py --output benchmarks/ANTAM_results.md
"""

import sys
import os

# Ensure project root on sys.path when run as script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fast_rag.benchmarks.benchmark import Benchmark, main

__all__ = ["Benchmark", "main"]

if __name__ == "__main__":
    # Forward CLI to benchmark.main()
    main()
