"""Tests for syntx.tabulate -- generic, atlas-agnostic label-based tabulation."""

import itertools

import numpy as np
import pandas as pd
import pytest

from syntx.tabulate import (
    correlation_matrix_wide_from_image,
    correlation_matrix_wide_from_timeseries,
    map_intensity_to_dataframe,
    widen_summary_dataframe,
)


def test_map_intensity_to_dataframe_distinct_region_means():
    ants = pytest.importorskip("ants")
    shape = (8, 8, 8)
    arr = np.zeros(shape, dtype=np.float32)
    arr[0:4, :, :] = 10.0
    arr[4:8, :, :] = 20.0
    scalar_img = ants.from_numpy(arr)

    label_arr = np.zeros(shape, dtype=np.float32)
    label_arr[0:4, :, :] = 1
    label_arr[4:8, :, :] = 2
    label_img = ants.from_numpy(label_arr)

    label_df = pd.DataFrame({"label": [1, 2, 3], "label_name": ["region_a", "region_b", "region_c"]})
    df = map_intensity_to_dataframe(scalar_img, label_img, label_df)

    assert list(df["label"]) == [1, 2, 3]
    assert list(df["label_name"]) == ["region_a", "region_b", "region_c"]
    row_a = df.loc[df["label"] == 1, "value"].iloc[0]
    row_b = df.loc[df["label"] == 2, "value"].iloc[0]
    assert np.isclose(row_a, 10.0)
    assert np.isclose(row_b, 20.0)


def test_map_intensity_to_dataframe_zero_voxel_label_is_nan():
    ants = pytest.importorskip("ants")
    shape = (4, 4, 4)
    arr = np.full(shape, 5.0, dtype=np.float32)
    scalar_img = ants.from_numpy(arr)
    label_img = ants.from_numpy(np.ones(shape, dtype=np.float32))

    label_df = pd.DataFrame({"label": [1, 99], "label_name": ["present", "absent"]})
    df = map_intensity_to_dataframe(scalar_img, label_img, label_df)

    present_val = df.loc[df["label"] == 1, "value"].iloc[0]
    absent_val = df.loc[df["label"] == 99, "value"].iloc[0]
    assert np.isclose(present_val, 5.0)
    assert np.isnan(absent_val)


def test_map_intensity_to_dataframe_median_stat():
    ants = pytest.importorskip("ants")
    shape = (4, 4, 4)
    arr = np.ones(shape, dtype=np.float32)
    # Inject one outlier voxel so mean != median within the labeled region.
    arr[0, 0, 0] = 1000.0
    scalar_img = ants.from_numpy(arr)
    label_img = ants.from_numpy(np.ones(shape, dtype=np.float32))

    label_df = pd.DataFrame({"label": [1], "label_name": ["whole"]})
    df_mean = map_intensity_to_dataframe(scalar_img, label_img, label_df, stat="mean")
    df_median = map_intensity_to_dataframe(scalar_img, label_img, label_df, stat="median")

    assert df_median["value"].iloc[0] == pytest.approx(1.0)
    assert df_mean["value"].iloc[0] > df_median["value"].iloc[0]


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


def test_correlation_matrix_matches_wide_form():
    from syntx.tabulate import correlation_matrix

    rng = np.random.default_rng(3)
    ts = rng.normal(size=(50, 4))
    mat = correlation_matrix(ts)
    wide = correlation_matrix_wide_from_timeseries(ts, roi_ids=[10, 20, 30, 40])
    assert np.isclose(wide["Corr_Label10_Label20"].iloc[0], mat[0, 1], atol=1e-6)
    assert np.isclose(wide["Corr_Label30_Label40"].iloc[0], mat[2, 3], atol=1e-6)
    assert np.allclose(np.diag(mat), 1.0, atol=1e-6)


def test_roi_mean_timeseries_matches_correlation_matrix_wide_from_image():
    from syntx.tabulate import correlation_matrix, roi_mean_timeseries

    ants = pytest.importorskip("ants")
    shape = (8, 8, 8, 50)
    rng = np.random.default_rng(4)
    sig = rng.normal(size=(50,)).astype(np.float32)
    arr = np.zeros(shape, dtype=np.float32)
    arr[0:4, :, :, :] = sig[None, None, None, :]
    arr[4:8, :, :, :] = -sig[None, None, None, :]
    ts_img = ants.from_numpy(arr)

    label_arr = np.zeros(shape[:3], dtype=np.float32)
    label_arr[0:4, :, :] = 1
    label_arr[4:8, :, :] = 2
    label_img = ants.from_numpy(label_arr)

    mean_roi, roi_ids = roi_mean_timeseries(ts_img, label_img)
    assert roi_ids == [1, 2]
    mat = correlation_matrix(mean_roi)
    assert np.isclose(mat[0, 1], -1.0, atol=1e-6)


def test_connectivity_summary_stats_basic():
    from syntx.tabulate import connectivity_summary_stats

    mat = np.array([[1.0, 0.5, -0.5], [0.5, 1.0, 0.2], [-0.5, 0.2, 1.0]])
    stats = connectivity_summary_stats(mat)
    assert stats["positive_fraction"] == pytest.approx(4 / 6)
    assert stats["negative_fraction"] == pytest.approx(2 / 6)
    assert stats["mean_abs_connectivity"] == pytest.approx(0.4)


def test_connectivity_summary_stats_single_roi_returns_nan():
    from syntx.tabulate import connectivity_summary_stats

    stats = connectivity_summary_stats(np.array([[1.0]]))
    assert stats["mean_abs_connectivity"] != stats["mean_abs_connectivity"]  # NaN
