"""
tests/test_benchmark_core.py — Unit and Correctness Tests for syntx.benchmark
=============================================================================

Validates dataset checking, download instructions, pair loading, single-pair
evaluation, and CLI integration.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import ants
import torch

import syntx
from syntx.benchmark import (
    check_mindboggle_data,
    load_mindboggle_pair,
    resolve_data_dir,
    evaluate_mindboggle_pair,
    compute_bidirectional_dice,
    compute_jacobian_metrics,
    normalize_intensity,
    MINDBOGGLE_SETUP_INSTRUCTIONS
)
from syntx.benchmark.cli import main as cli_main


def test_resolve_data_dir_valid(tmp_path):
    """Test resolving an existing data directory."""
    d = tmp_path / "test_volumes"
    d.mkdir()
    resolved = resolve_data_dir(str(d))
    assert os.path.isabs(resolved)
    assert resolved == str(d.resolve())


def test_resolve_data_dir_missing(tmp_path, capsys):
    """Test resolving a non-existent directory raises FileNotFoundError and prints instructions."""
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_data_dir(str(non_existent))
    assert "Mindboggle data directory not found" in str(exc_info.value)
    
    captured = capsys.readouterr()
    assert "MINDBOGGLE-101 DATASET SETUP GUIDE" in captured.err


def test_check_mindboggle_data_missing_csv(tmp_path, capsys):
    """Test check_mindboggle_data when pairs.csv does not exist."""
    is_valid, report = check_mindboggle_data(pairs_csv=str(tmp_path / "no_such_file.csv"), verbose=True)
    assert not is_valid
    assert not report["pairs_csv_exists"]
    captured = capsys.readouterr()
    assert "Pairs CSV not found" in captured.err


def test_check_mindboggle_data_mock_valid(tmp_path):
    """Test check_mindboggle_data with a mocked directory structure."""
    # Create mock volumes
    data_dir = tmp_path / "data"
    c1_dir = data_dir / "COHORT1_volumes" / "SUBJ1"
    c2_dir = data_dir / "COHORT2_volumes" / "SUBJ2"
    c1_dir.mkdir(parents=True)
    c2_dir.mkdir(parents=True)

    img = ants.from_numpy(np.ones((10, 10, 10), dtype=np.float32))
    lab = ants.from_numpy(np.ones((10, 10, 10), dtype=np.uint32))

    ants.image_write(img, str(c1_dir / "t1weighted_brain.nii.gz"))
    ants.image_write(lab, str(c1_dir / "labels.DKT31.manual.nii.gz"))
    ants.image_write(img, str(c2_dir / "t1weighted_brain.nii.gz"))
    ants.image_write(lab, str(c2_dir / "labels.DKT31.manual.nii.gz"))

    # Create mock CSV
    csv_file = tmp_path / "pairs.csv"
    df = pd.DataFrame([
        {"cohort1": "COHORT1", "subject1": "SUBJ1", "cohort2": "COHORT2", "subject2": "SUBJ2", "type": "inter"}
    ])
    df.to_csv(csv_file, index=False)

    is_valid, rep = check_mindboggle_data(pairs_csv=str(csv_file), data_dir=str(data_dir), verbose=False)
    assert is_valid
    assert rep["total_pairs_in_csv"] == 1
    assert rep["available_pairs"] == 1
    assert len(rep["missing_pairs"]) == 0


def test_load_mindboggle_pair_out_of_bounds(tmp_path):
    """Test out-of-range pair index raises IndexError."""
    csv_file = tmp_path / "pairs.csv"
    df = pd.DataFrame([
        {"cohort1": "C1", "subject1": "S1", "cohort2": "C2", "subject2": "S2", "type": "intra"}
    ])
    df.to_csv(csv_file, index=False)

    with pytest.raises(IndexError):
        load_mindboggle_pair(pair_idx=5, pairs_csv=str(csv_file), data_dir=str(tmp_path))


def test_intensity_normalization():
    """Test percentile intensity normalization."""
    arr = np.linspace(0, 1000, 1000).reshape((10, 10, 10)).astype(np.float32)
    img = ants.from_numpy(arr)
    norm_img = normalize_intensity(img)
    norm_arr = norm_img.numpy()
    assert norm_arr.min() >= 0.0
    assert norm_arr.max() <= 1.0
    assert norm_arr.dtype == np.float32


def test_compute_bidirectional_dice_precision():
    """Test compute_bidirectional_dice returns finite floats and handles pandas/numpy correctly."""
    fl = ants.from_numpy(np.random.randint(0, 5, (16, 16, 16)).astype(np.uint32))
    ml = ants.from_numpy(np.random.randint(0, 5, (16, 16, 16)).astype(np.uint32))
    fi = ants.from_numpy(np.ones((16, 16, 16), dtype=np.float32))
    mi = ants.from_numpy(np.ones((16, 16, 16), dtype=np.float32))

    df_f, df_m, sym = compute_bidirectional_dice(fl, ml, fi, mi, [], [])
    assert np.isfinite(df_f)
    assert np.isfinite(df_m)
    assert np.isfinite(sym)
    assert isinstance(sym, float)


def test_evaluate_mindboggle_pair_mock(tmp_path):
    """Test evaluate_mindboggle_pair end-to-end with a fast small volume."""
    data_dir = tmp_path / "data"
    c1_dir = data_dir / "OASIS_volumes" / "S1"
    c2_dir = data_dir / "OASIS_volumes" / "S2"
    c1_dir.mkdir(parents=True)
    c2_dir.mkdir(parents=True)

    np.random.seed(42)
    img1_arr = (np.random.rand(48, 48, 48) * 100).astype(np.float32)
    img2_arr = (np.random.rand(48, 48, 48) * 100).astype(np.float32)
    lab1_arr = np.random.randint(1, 4, (48, 48, 48)).astype(np.uint32)
    lab2_arr = np.random.randint(1, 4, (48, 48, 48)).astype(np.uint32)

    ants.image_write(ants.from_numpy(img1_arr), str(c1_dir / "t1weighted_brain.nii.gz"))
    ants.image_write(ants.from_numpy(lab1_arr), str(c1_dir / "labels.DKT31.manual.nii.gz"))
    ants.image_write(ants.from_numpy(img2_arr), str(c2_dir / "t1weighted_brain.nii.gz"))
    ants.image_write(ants.from_numpy(lab2_arr), str(c2_dir / "labels.DKT31.manual.nii.gz"))

    csv_file = tmp_path / "pairs.csv"
    df = pd.DataFrame([
        {"cohort1": "OASIS", "subject1": "S1", "cohort2": "OASIS", "subject2": "S2", "type": "intra"}
    ])
    df.to_csv(csv_file, index=False)

    rec = evaluate_mindboggle_pair(
        pair_idx=0,
        model="sobolev",
        device="cpu",
        pairs_csv=str(csv_file),
        data_dir=str(data_dir),
        ants_baseline_dir=str(tmp_path),
        generate_report=False,
        verbose=False
    )

    assert rec["status"] == "SUCCESS"
    assert rec["model_type"] == "sobolev"
    assert np.isfinite(rec["syntx_dice_sym"])
    assert np.isfinite(rec["syntx_fold"])
    assert "fwdtransforms" in rec["transforms"]


def test_organize_mindboggle_data(tmp_path):
    """Test organizing unzipped/nested Mindboggle files into standard structure."""
    from syntx.benchmark.data import organize_mindboggle_data

    raw_dir = tmp_path / "raw_downloads"
    target_dir = tmp_path / "organized_volumes"

    # Create mock raw downloads with nested directory
    s1_raw = raw_dir / "Mindboggle101" / "OASIS-TRT-20-1"
    s2_raw = raw_dir / "Mindboggle101" / "OASIS-TRT-20-2"
    s1_raw.mkdir(parents=True)
    s2_raw.mkdir(parents=True)

    img = ants.from_numpy(np.ones((10, 10, 10), dtype=np.float32))
    lab = ants.from_numpy(np.ones((10, 10, 10), dtype=np.uint32))

    ants.image_write(img, str(s1_raw / "t1weighted_brain.nii.gz"))
    ants.image_write(lab, str(s1_raw / "labels.DKT31.manual.nii.gz"))
    ants.image_write(img, str(s2_raw / "t1weighted_brain.nii.gz"))
    ants.image_write(lab, str(s2_raw / "labels.DKT31.manual.nii.gz"))

    csv_file = tmp_path / "pairs.csv"
    df = pd.DataFrame([
        {"cohort1": "OASIS-TRT-20", "subject1": "OASIS-TRT-20-1", "cohort2": "OASIS-TRT-20", "subject2": "OASIS-TRT-20-2", "type": "intra"}
    ])
    df.to_csv(csv_file, index=False)

    is_valid, rep = organize_mindboggle_data(
        source_path=str(raw_dir),
        target_dir=str(target_dir),
        pairs_csv=str(csv_file),
        verbose=False
    )

    assert is_valid
    assert rep["organized_subjects"] == 2
    assert os.path.exists(target_dir / "OASIS-TRT-20_volumes" / "OASIS-TRT-20-1" / "t1weighted_brain.nii.gz")
    assert os.path.exists(target_dir / "OASIS-TRT-20_volumes" / "OASIS-TRT-20-1" / "labels.DKT31.manual.nii.gz")

