"""
Unit tests for syntx.cli command-line interface.
"""

import os
import sys
import json
import pytest
import ants
from syntx.cli import main, cmd_info, cmd_register, parse_iterations


def test_parse_iterations():
    assert parse_iterations("100x100x20") == [100, 100, 20]
    assert parse_iterations("100,100,20") == [100, 100, 20]
    assert parse_iterations("[40, 20, 10]") == [40, 20, 10]
    assert parse_iterations("40 20 10") == [40, 20, 10]
    assert parse_iterations("") == [100, 100, 20]


def test_cli_info(capsys):
    test_args = ["syntx", "info"]
    sys.argv = test_args
    ret = main()
    assert ret == 0
    captured = capsys.readouterr()
    assert "SYNTX ENVIRONMENT & SYSTEM INTROSPECTION" in captured.out
    assert "Syntx Version" in captured.out


def test_cli_register_missing_files(tmp_path):
    test_args = [
        "syntx", "register",
        "-f", str(tmp_path / "non_existent_fixed.nii.gz"),
        "-m", str(tmp_path / "non_existent_moving.nii.gz"),
        "-o", str(tmp_path / "out")
    ]
    sys.argv = test_args
    ret = main()
    assert ret == 1


def test_cli_register_syn_2d(tmp_path):
    r16_path = str(tmp_path / "r16.nii.gz")
    r64_path = str(tmp_path / "r64.nii.gz")
    out_dir = str(tmp_path / "syn_out")

    r16 = ants.image_read(ants.get_ants_data('r16'))
    r64 = ants.image_read(ants.get_ants_data('r64'))
    ants.image_write(r16, r16_path)
    ants.image_write(r64, r64_path)

    test_args = [
        "syntx", "register",
        "-f", r16_path,
        "-m", r64_path,
        "-o", out_dir,
        "-p", "test_syn_",
        "--model", "syn",
        "-i", "20x10",
        "--report"
    ]
    sys.argv = test_args
    ret = main()
    assert ret == 0

    # Verify generated outputs
    assert os.path.exists(os.path.join(out_dir, "test_syn_Warped.nii.gz"))
    assert os.path.exists(os.path.join(out_dir, "test_syn_InverseWarped.nii.gz"))
    assert os.path.exists(os.path.join(out_dir, "test_syn_Jacobian.nii.gz"))
    assert os.path.exists(os.path.join(out_dir, "test_syn_metrics.json"))
    assert os.path.exists(os.path.join(out_dir, "registration_report.html"))

    with open(os.path.join(out_dir, "test_syn_metrics.json")) as f:
        metrics = json.load(f)
    assert metrics["model"] == "syn"
    assert "harmonic_energy" in metrics
    assert "bending_energy" in metrics
    assert "jacobian_folding_pct" in metrics
    assert metrics["jacobian_folding_pct"] == 0.0


def test_cli_register_tvf_2d(tmp_path):
    r16_path = str(tmp_path / "r16.nii.gz")
    r64_path = str(tmp_path / "r64.nii.gz")
    out_dir = str(tmp_path / "tvf_out")

    r16 = ants.image_read(ants.get_ants_data('r16'))
    r64 = ants.image_read(ants.get_ants_data('r64'))
    ants.image_write(r16, r16_path)
    ants.image_write(r64, r64_path)

    test_args = [
        "syntx", "register",
        "-f", r16_path,
        "-m", r64_path,
        "-o", out_dir,
        "-p", "test_tvf_",
        "--model", "tvf",
        "-i", "20x10",
        "--no-report"
    ]
    sys.argv = test_args
    ret = main()
    assert ret == 0

    assert os.path.exists(os.path.join(out_dir, "test_tvf_Warped.nii.gz"))
    assert os.path.exists(os.path.join(out_dir, "test_tvf_metrics.json"))
