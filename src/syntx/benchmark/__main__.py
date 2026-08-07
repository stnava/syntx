"""
Allows running `python -m syntx.benchmark` directly as a 1-liner command.
"""

from .runner import run_benchmark_suite
import sys

if __name__ == '__main__':
    run_benchmark_suite(phases=[1, 2])
