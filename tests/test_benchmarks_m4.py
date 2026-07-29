#!/usr/bin/env python3
"""
Milestone 4 Benchmark Verification and Test Coverage Suite (tests/test_benchmarks_m4.py).

Tests:
1. 2D benchmark on r16 & r64: PyTorch vs JAX parity (MSE discrepancy <= 0.001) for TVF and Geodesic Shooting.
2. 3D benchmark on Mindboggle Pair 08: Cortical DKT Dice, min det J(x), folding %, inverse identity error, and runtime.
3. Verification of generated HTML reports in docs/reports/ (benchmark_2d_r16.html, benchmark_2d_r64.html, benchmark_3d_pair08.html).
"""

import os
import sys
import pytest
import numpy as np
import torch
import ants

# Ensure local syntx and examples are in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import syntx
from examples.run_benchmark_2d import run_2d_benchmarks
from examples.run_benchmark_3d_pair08 import run_3d_benchmark_pair08


def test_2d_benchmarks_r16_r64_parity_and_report():
    """
    Test 2D registration on r16 -> r64:
    - Verifies PyTorch vs JAX TVF MSE discrepancy <= 0.001
    - Verifies PyTorch vs JAX Geodesic Shooting MSE discrepancy <= 0.001
    - Verifies syn baseline MSE improvement
    - Verifies HTML report generation
    """
    html_report = os.path.join("docs", "reports", "benchmark_2d_r16.html")
    res = run_2d_benchmarks(fixed_name='r16', moving_name='r64', output_html=html_report)

    tvf_disc = res['tvf_discrepancy']
    shoot_disc = res['shoot_discrepancy']
    syn_disc = res['syn_discrepancy']

    assert tvf_disc <= 0.001, f"TVF PyTorch vs JAX MSE discrepancy failed: {tvf_disc}"
    assert shoot_disc <= 0.001, f"Shooting PyTorch vs JAX MSE discrepancy failed: {shoot_disc}"
    assert syn_disc <= 0.001, f"SyN PyTorch vs JAX MSE discrepancy failed: {syn_disc}"
    assert res['syn']['mse_pt'] < 0.05, f"syn baseline MSE unexpectedly high: {res['syn']['mse_pt']}"

    assert os.path.exists(html_report), f"HTML report file missing: {html_report}"
    assert os.path.getsize(html_report) > 1000, f"HTML report file empty or too small: {html_report}"


def test_2d_benchmarks_r64_r16_parity_and_report():
    """
    Test 2D registration on r64 -> r16:
    - Verifies PyTorch vs JAX TVF MSE discrepancy <= 0.001
    - Verifies PyTorch vs JAX Geodesic Shooting MSE discrepancy <= 0.001
    - Verifies HTML report generation
    """
    html_report = os.path.join("docs", "reports", "benchmark_2d_r64.html")
    res = run_2d_benchmarks(fixed_name='r64', moving_name='r16', output_html=html_report)

    tvf_disc = res['tvf_discrepancy']
    shoot_disc = res['shoot_discrepancy']
    syn_disc = res['syn_discrepancy']

    assert tvf_disc <= 0.001, f"TVF PyTorch vs JAX MSE discrepancy failed: {tvf_disc}"
    assert shoot_disc <= 0.001, f"Shooting PyTorch vs JAX MSE discrepancy failed: {shoot_disc}"
    assert syn_disc <= 0.001, f"SyN PyTorch vs JAX MSE discrepancy failed: {syn_disc}"

    assert os.path.exists(html_report), f"HTML report file missing: {html_report}"
    assert os.path.getsize(html_report) > 1000, f"HTML report file empty or too small: {html_report}"


def test_3d_benchmark_pair08_metrics_and_report():
    """
    Test 3D registration on Mindboggle Pair 08:
    - Verifies Cortical DKT Dice > 0.50
    - Verifies min det J > 0 and folding % < 1%
    - Verifies mean inverse identity error < 1.0 mm
    - Verifies HTML report generation at docs/reports/benchmark_3d_pair08.html
    """
    fixed_path = '/Users/stnava/data/mindboggle/volumes/NKI-TRT-20_volumes/NKI-TRT-20-2/t1weighted_brain.nii.gz'
    moving_path = '/Users/stnava/data/mindboggle/volumes/MMRR-21_volumes/MMRR-21-2/t1weighted_brain.nii.gz'

    if not (os.path.exists(fixed_path) and os.path.exists(moving_path)):
        pytest.skip("Mindboggle Pair 08 volume files not found on local path.")

    html_report = os.path.join("docs", "reports", "benchmark_3d_pair08.html")
    res = run_3d_benchmark_pair08(output_html=html_report)

    assert res['cortical_dice'] > 0.50, f"Cortical DKT Dice score too low: {res['cortical_dice']}"
    assert res['folding_pct'] < 1.0, f"Excessive Jacobian grid folding: {res['folding_pct']}%"
    assert res['mean_inv_err_mm'] < 1.0, f"Inverse identity error too high: {res['mean_inv_err_mm']} mm"
    assert res['runtime_sec'] > 0.0, f"Runtime measurement invalid: {res['runtime_sec']}"

    assert os.path.exists(html_report), f"HTML report file missing: {html_report}"
    assert os.path.getsize(html_report) > 1000, f"HTML report file empty or too small: {html_report}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
