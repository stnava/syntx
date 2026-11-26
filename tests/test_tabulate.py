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


def test_correlation_matrix_wide_with_label_names():
    ants = pytest.importorskip("ants")
    sig = np.linspace(-1.0, 1.0, 40, dtype=np.float32)
    ts = np.stack([sig, -sig, sig], axis=1)
    names = {1002: "left_caudal_anterior_cingulate", 2029: "right_superior_parietal"}
    df = correlation_matrix_wide_from_timeseries(
        ts, roi_ids=[1002, 2029, 9999], prefix="rsf_conn_", label_names=names
    )
    assert list(df.columns) == [
        "rsf_conn_left_caudal_anterior_cingulate_right_superior_parietal",
        "rsf_conn_left_caudal_anterior_cingulate_Label9999",
        "rsf_conn_right_superior_parietal_Label9999",
    ]
    assert np.isclose(
        df["rsf_conn_left_caudal_anterior_cingulate_right_superior_parietal"].iloc[0], -1.0, atol=1e-6
    )

    shape = (4, 4, 4, 40)
    arr = np.zeros(shape, dtype=np.float32)
    arr[0:2, :, :, :] = sig[None, None, None, :]
    arr[2:4, :, :, :] = -sig[None, None, None, :]
    ts_img = ants.from_numpy(arr)
    label_arr = np.zeros(shape[:3], dtype=np.float32)
    label_arr[0:2, :, :] = 1002
    label_arr[2:4, :, :] = 2029
    label_img = ants.from_numpy(label_arr)
    df_img = correlation_matrix_wide_from_image(
        ts_img, label_img, prefix="rsf_conn_", label_names=names
    )
    assert list(df_img.columns) == [
        "rsf_conn_left_caudal_anterior_cingulate_right_superior_parietal"
    ]
    assert np.isclose(
        df_img["rsf_conn_left_caudal_anterior_cingulate_right_superior_parietal"].iloc[0], -1.0, atol=1e-6
    )


def test_centralized_label_lookup_and_roi_summary_wide():
    ants = pytest.importorskip("ants")
    from syntx.tabulate import label_df_to_lookup, normalize_label_name, region_wide_row, roi_summary_wide

    assert normalize_label_name("  left cerebellem exterior  ") == "left_cerebellum_exterior"

    label_df = pd.DataFrame({
        "label": [0, 1006, 2006],
        "label_name": ["background", "left entorhinal", "right entorhinal"],
    })
    lookup = label_df_to_lookup(label_df)
    assert lookup == {1006: "left_entorhinal", 2006: "right_entorhinal"}

    shape = (6, 6, 6)
    lab_arr = np.zeros(shape, dtype=np.float32)
    lab_arr[:2, :, :] = 1006
    lab_arr[2:4, :, :] = 2006
    lab_arr[4:, :, :] = 99
    val_arr = np.zeros(shape, dtype=np.float32)
    val_arr[:2, :, :] = 10.0
    val_arr[2:4, :, :] = 20.0
    val_arr[4:, :, :] = 30.0
    lab_img = ants.from_numpy(lab_arr)
    val_img = ants.from_numpy(val_arr)

    wide = roi_summary_wide(val_img, lab_img, prefix="pet_", label_df=label_df)
    assert list(wide.columns) == [
        "pet_Label99.Mean",
        "pet_left_entorhinal.Mean",
        "pet_right_entorhinal.Mean",
    ]
    assert wide["pet_left_entorhinal.Mean"].iloc[0] == pytest.approx(10.0)
    assert wide["pet_right_entorhinal.Mean"].iloc[0] == pytest.approx(20.0)
    assert wide["pet_Label99.Mean"].iloc[0] == pytest.approx(30.0)

    r_row = region_wide_row(val_img, lab_img, label_df, column_prefix="FA_DKT")
    assert "FA_DKT_background" not in r_row
    assert r_row["FA_DKT_left_entorhinal"] == pytest.approx(10.0)
    assert r_row["FA_DKT_right_entorhinal"] == pytest.approx(20.0)


def test_roi_mean_timeseries_sparse_labels_matches_boolean_loop():
    ants = pytest.importorskip("ants")
    from syntx.tabulate import roi_mean_timeseries

    rng = np.random.default_rng(7)
    shape = (12, 14, 10, 64)
    ts_arr = rng.normal(loc=100.0, scale=15.0, size=shape).astype(np.float32)
    lbl_arr = np.zeros(shape[:3], dtype=np.float32)
    sparse_ids = [1002, 1005, 1024, 2002, 2035]
    flat_lbl = lbl_arr.ravel()
    chosen = rng.choice([0] + sparse_ids, size=flat_lbl.size, p=[0.6, 0.08, 0.08, 0.08, 0.08, 0.08])
    lbl_arr = chosen.reshape(shape[:3]).astype(np.float32)

    ts_img = ants.from_numpy(ts_arr, spacing=(2.0, 2.0, 2.0, 0.8))
    lbl_img = ants.from_numpy(lbl_arr, spacing=(2.0, 2.0, 2.0))

    mean_roi, roi_ids = roi_mean_timeseries(ts_img, lbl_img)
    assert roi_ids == sparse_ids
    ref = np.column_stack([ts_arr[lbl_arr == lid, :].mean(axis=0) for lid in sparse_ids])
    assert np.allclose(mean_roi, ref, atol=1e-5, rtol=1e-5)

    from syntx.tabulate import roi_mean_from_masked_timeseries

    brain_mask = (lbl_arr > 0) | (rng.random(shape[:3]) < 0.2)
    flat_masked = np.zeros_like(ts_arr)
    flat_masked[brain_mask, :] = ts_arr[brain_mask, :]
    ts_masked_img = ants.from_numpy(flat_masked, spacing=(2.0, 2.0, 2.0, 0.8))
    ref_m, rids_m = roi_mean_timeseries(ts_masked_img, lbl_img)
    fast_m, rids_fast = roi_mean_from_masked_timeseries(ts_arr[brain_mask, :].T, brain_mask, lbl_img)
    assert rids_fast == rids_m
    assert np.allclose(fast_m, ref_m, atol=1e-5, rtol=1e-5)




