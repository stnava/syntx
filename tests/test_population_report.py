import os
import json
import tempfile
import pytest
import syntx
from syntx.viz import create_population_benchmark_report

def test_population_benchmark_report_generator():
    """Verify that create_population_benchmark_report produces a valid, interactive HTML report with Plotly scatterplots."""
    # 1. Create simulated benchmark records
    sim_records = {}
    for i in range(10):
        c_type = "intra" if i < 5 else "inter"
        sob_dice = 0.62 + 0.05 * (i % 3)
        ants_dice = 0.59 + 0.04 * (i % 3)
        rec = {
            "pair_idx": i,
            "cohort_type": c_type,
            "syntx_dice_sym": sob_dice,
            "syntx_dice_fixed": sob_dice - 0.01,
            "syntx_dice_moving": sob_dice + 0.01,
            "syntx_fold": 0.0001 * (i % 2),
            "syntx_min_jac": 0.05,
            "syntx_time": 45.0 + i * 2.0,
            "ants_baseline": {
                "dice_sym": ants_dice,
                "dice_fixed": ants_dice - 0.01,
                "dice_moving": ants_dice + 0.01,
                "folding_pct": 0.0,
                "min_jacobian": 0.10,
                "runtime_seconds": 180.0 + i * 5.0
            }
        }
        sim_records[i] = rec

    with tempfile.TemporaryDirectory() as tmpdir:
        out_html = os.path.join(tmpdir, "test_report.html")
        
        # Test generation from dict of records
        res_path = create_population_benchmark_report(
            results_source=sim_records,
            output_html=out_html,
            title="Test Population Benchmark"
        )
        
        assert os.path.exists(res_path)
        assert os.path.getsize(res_path) > 2000
        
        with open(res_path, "r") as f:
            content = f.read()
            
        # Verify required HTML and Plotly elements
        assert "plotly-2.27.0.min.js" in content
        assert "diceScatterPlot" in content
        assert "timeScatterPlot" in content
        assert "Intra-Cohort Pairs" in content
        assert "Inter-Cohort Pairs" in content
        assert "Test Population Benchmark" in content
        assert "Plotly.newPlot" in content

def test_population_benchmark_report_from_summary_json():
    """Verify report generation from a master summary JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        summary_file = os.path.join(tmpdir, "summary.json")
        out_html = os.path.join(tmpdir, "test_summary_report.html")
        
        master_data = {
            "sobolev_results": {
                "0": {
                    "pair_idx": 0, "cohort_type": "intra",
                    "syntx_dice_sym": 0.65, "syntx_dice_fixed": 0.64, "syntx_dice_moving": 0.66,
                    "syntx_fold": 0.0, "syntx_time": 50.0,
                    "ants_baseline": {"dice_sym": 0.61, "runtime_seconds": 200.0}
                },
                "1": {
                    "pair_idx": 1, "cohort_type": "intra",
                    "syntx_dice_sym": 0.68, "syntx_dice_fixed": 0.67, "syntx_dice_moving": 0.69,
                    "syntx_fold": 0.0, "syntx_time": 52.0,
                    "ants_baseline": {"dice_sym": 0.63, "runtime_seconds": 210.0}
                }
            },
            "gaussian_results": {
                "0": {"pair_idx": 0, "syntx_dice_sym": 0.63, "syntx_fold": 0.01}
            }
        }
        with open(summary_file, "w") as f:
            json.dump(master_data, f)
            
        res_path = create_population_benchmark_report(
            results_source=summary_file,
            output_html=out_html
        )
        assert os.path.exists(res_path)
        with open(res_path, "r") as f:
            content = f.read()
        assert "Ablation Study: Sobolev Smoothing vs. Standard Gaussian Regularization" in content
