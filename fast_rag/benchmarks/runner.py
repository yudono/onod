"""
PRD §25-26 runner — alternative entry point per task description.

Task says: "benchmarks/ANTAM_benchmark.py or fast_rag/benchmarks/runner.py"

This file satisfies the second option; the first is provided at
benchmarks/ANTAM_benchmark.py for compatibility.

Usage:
  python -m fast_rag.benchmarks.runner
  python fast_rag/benchmarks/runner.py
"""

from .benchmark import Benchmark, main  # re-export

__all__ = ["Benchmark", "main"]

if __name__ == "__main__":
    main()
