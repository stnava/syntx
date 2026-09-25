"""Tests for syntx.tabulate -- generic, atlas-agnostic label-based tabulation."""

import itertools

import numpy as np
import pandas as pd
import pytest

from syntx.tabulate import (
    correlation_matrix_wide_from_image,
    correlation_matrix_wide_from_timeseries,
    widen_summary_dataframe,
)


def test_widen_summary_dataframe_basic():
    long_df = pd.DataFrame({"Label": [1, 2, 3], "VolumeInMillimeters": [100.0, 200.0, 300.0]})
    wide = widen_summary_dataframe(long_df, description="Label", value="VolumeInMillimeters", prefix="smri.")
    assert list(wide.columns) == [
        "smri.Label1.VolumeInMillimeters",
        "smri.Label2.VolumeInMillimeters",
        "smri.Label3.VolumeInMillimeters",
    ]
    assert wide.iloc[0].tolist() == [100.0, 200.0, 300.0]


def test_widen_summary_dataframe_skip_first():
    long_df = pd.DataFrame({"Label": [0, 1, 2], "Mean": [-1.0, 10.0, 20.0]})
    wide = widen_summary_dataframe(long_df, description="Label", value="Mean", skip_first=True)
    assert list(wide.columns) == ["Label1.Mean", "Label2.Mean"]


def test_widen_summary_dataframe_missing_column_raises():
    long_df = pd.DataFrame({"Label": [1], "Other": [1.0]})
    with pytest.raises(ValueError):
        widen_summary_dataframe(long_df, description="Label", value="Mean")


def test_correlation_matrix_wide_matches_numpy_corrcoef():
    rng = np.random.default_rng(0)
    ts = rng.normal(size=(500, 4))
    df = correlation_matrix_wide_from_timeseries(ts, roi_ids=[1, 2, 3, 4])
    np_corr = np.corrcoef(ts, rowvar=False)
    for a, b in itertools.combinations(range(4), 2):
        col = f"Corr_Label{a + 1}_Label{b + 1}"
        assert np.isclose(df[col].iloc[0], np_corr[a, b], atol=1e-8)


def test_correlation_matrix_wide_perfect_and_anti_correlation():
    rng = np.random.default_rng(1)
    sig = rng.normal(size=(200,))
    ts = np.stack([sig, sig, -sig], axis=1)
    df = correlation_matrix_wide_from_timeseries(ts, roi_ids=[10, 20, 30], prefix="rsf.")
    assert np.isclose(df["rsf.Label10_Label20"].iloc[0], 1.0, atol=1e-6)
    assert np.isclose(df["rsf.Label10_Label30"].iloc[0], -1.0, atol=1e-6)
    assert np.isclose(df["rsf.Label20_Label30"].iloc[0], -1.0, atol=1e-6)


def test_correlation_matrix_wide_from_image():
    ants = pytest.importorskip("ants")
    shape = (8, 8, 8, 50)
    rng = np.random.default_rng(2)
    sig = rng.normal(size=(50,)).astype(np.float32)
    arr = np.zeros(shape, dtype=np.float32)
    arr[0:4, :, :, :] = sig[None, None, None, :]
    arr[4:8, :, :, :] = -sig[None, None, None, :]
    ts_img = ants.from_numpy(arr)

    label_arr = np.zeros(shape[:3], dtype=np.float32)
    label_arr[0:4, :, :] = 1
    label_arr[4:8, :, :] = 2
    label_img = ants.from_numpy(label_arr)

    df = correlation_matrix_wide_from_image(ts_img, label_img)
    assert np.isclose(df["Corr_Label1_Label2"].iloc[0], -1.0, atol=1e-6)
