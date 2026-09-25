"""
Tests for syntx.motion — Motion Correction, Parameter Recovery, and FD Metrics.
"""

import os
import tempfile
import numpy as np
import pytest
import ants

import syntx
from syntx.motion import (
    motion_correction,
    MotionParameters,
    TransformCollection,
    MotionCorrectionResult,
    calculate_framewise_displacement,
    calculate_dvars,
    _extract_rigid_parameters,
    _create_identity_transform,
)


def _create_2d_phantom(num_frames: int = 3, nx: int = 32, ny: int = 32, shifts=None):
    """Create a synthetic 2D+t time series with a Gaussian blob and known shifts."""
    if shifts is None:
        shifts = [(0.0, 0.0), (1.0, -0.8), (2.0, -1.6)]

    data = np.zeros((nx, ny, num_frames), dtype=np.float32)
    gx, gy = np.ogrid[-nx // 2 : nx // 2, -ny // 2 : ny // 2]

    for t in range(num_frames):
        sx, sy = shifts[t]
        data[..., t] = np.exp(-((gx - sx) ** 2 + (gy - sy) ** 2) / (2.0 * 4.0**2))

    img = ants.from_numpy(data, spacing=(1.0, 1.0, 1.0))
    return img, shifts


def _create_3d_phantom(num_frames: int = 3, nx: int = 16, ny: int = 16, nz: int = 16, shifts=None):
    """Create a synthetic 3D+t time series with a 3D Gaussian blob and known shifts."""
    if shifts is None:
        shifts = [(0.0, 0.0, 0.0), (0.8, -0.6, 0.5), (1.5, -1.2, 1.0)]

    data = np.zeros((nx, ny, nz, num_frames), dtype=np.float32)
    gx, gy, gz = np.ogrid[-nx // 2 : nx // 2, -ny // 2 : ny // 2, -nz // 2 : nz // 2]

    for t in range(num_frames):
        sx, sy, sz = shifts[t]
        data[..., t] = np.exp(
            -((gx - sx) ** 2 + (gy - sy) ** 2 + (gz - sz) ** 2) / (2.0 * 3.5**2)
        )

    img = ants.from_numpy(data, spacing=(1.0, 1.0, 1.0, 1.0))
    return img, shifts


def test_motion_parameters_class():
    """Test the MotionParameters ndarray subclass for both array and dict access."""
    data = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.2, -0.5, 0.8, 0.05, -0.02, 0.1],
    ])
    cols = ["tx", "ty", "tz", "rx", "ry", "rz"]
    rad_cols = ["rx", "ry", "rz"]
    mp = MotionParameters(data, columns=cols, rad_to_deg_cols=rad_cols, spatial_dim=3)

    assert isinstance(mp, np.ndarray)
    assert mp.shape == (2, 6)
    assert mp.spatial_dimension == 3
    assert np.allclose(mp.translations, data[:, :3])
    assert np.allclose(mp.rotations, data[:, 3:])
    assert np.allclose(mp.rotations_deg, np.degrees(data[:, 3:]))

    # Test indexing
    assert np.allclose(mp[:, 0], data[:, 0])
    assert np.allclose(mp["tx"], data[:, 0])
    assert np.allclose(mp["ty"], data[:, 1])
    assert np.allclose(mp["tz"], data[:, 2])
    assert np.allclose(mp["rx"], data[:, 3])
    assert np.allclose(mp["rx_deg"], np.degrees(data[:, 3]))

    # Test aliases
    assert np.allclose(mp["trans_x"], data[:, 0])
    assert np.allclose(mp["rot_x"], data[:, 3])

    # Test attribute access
    assert np.allclose(mp.tx, data[:, 0])
    assert np.allclose(mp.ty, data[:, 1])
    assert np.allclose(mp.rx, data[:, 3])

    # Test dict-like methods
    assert "tx" in mp
    assert "trans_x" in mp
    assert "rx_deg" in mp
    assert "unknown" not in mp
    assert 123 not in mp
    assert "tx" in mp.keys()
    assert len(mp.values()) == len(mp.keys())
    assert len(mp.items()) == len(mp.keys())
    assert mp.get("tz") is not None
    assert mp.get("nonexistent", 999) == 999

    with pytest.raises(KeyError):
        _ = mp["nonexistent_column"]

    with pytest.raises(AttributeError):
        _ = mp.nonexistent_attr

    # Test conversion to DataFrame and dict
    df = mp.to_dataframe()
    assert df.shape == (2, 9)  # 6 base + 3 deg columns
    assert "tx" in df.columns
    assert "rx_deg" in df.columns

    d = mp.to_dict(include_degrees=True)
    assert isinstance(d, dict)
    assert "tx" in d
    assert "rx_deg" in d


def test_motion_parameters_2d():
    """Test MotionParameters with 2D spatial dimensions."""
    data = np.array([
        [0.0, 0.0, 0.0],
        [1.0, -0.5, 0.1],
    ])
    cols = ["tx", "ty", "rz"]
    mp = MotionParameters(data, columns=cols, rad_to_deg_cols=["rz"], spatial_dim=2)
    assert mp.spatial_dimension == 2
    assert "theta" in mp
    assert np.allclose(mp["theta"], data[:, 2])
    assert np.allclose(mp.rotations_deg, np.degrees(data[:, 2:]))


def test_transform_collection():
    """Test TransformCollection dual list/dict interface."""
    fwd = [["vol0.mat"], ["vol1.mat"]]
    inv = [["vol0_inv.mat"], ["vol1_inv.mat"]]
    tc = TransformCollection(fwd, inv)

    assert len(tc) == 2
    assert tc[0] == ["vol0.mat"]
    assert tc[1] == ["vol1.mat"]

    # Dict access
    assert tc["fwdtransforms"] == fwd
    assert tc["fwd"] == fwd
    assert tc["invtransforms"] == inv
    assert tc["inv"] == inv
    assert tc.fwdtransforms == fwd
    assert tc.invtransforms == inv
    assert "fwdtransforms" in tc.keys()
    assert len(tc.values()) == 2
    assert len(tc.items()) == 2
    assert tc.get("fwd") == fwd
    assert tc.get("nonexistent", 42) == 42

    with pytest.raises(KeyError):
        _ = tc["unknown_key"]


def test_motion_correction_result():
    """Test MotionCorrectionResult class and repr."""
    res = MotionCorrectionResult({"summary": {"fd_mean": 0.123, "temporal_variance_reduction_percent": 99.1}})
    res.custom_attr = "hello"
    assert res.custom_attr == "hello"
    assert res["custom_attr"] == "hello"
    assert "0.123" in repr(res)

    with pytest.raises(AttributeError):
        _ = res.non_existent_attribute


def test_calculate_framewise_displacement_edge_cases():
    """Test FD with 0 or 1 frames."""
    data = np.zeros((1, 6))
    mp = MotionParameters(data, columns=["tx", "ty", "tz", "rx", "ry", "rz"], spatial_dim=3)
    fd_p, fd_j = calculate_framewise_displacement(mp, [np.eye(4)])
    assert len(fd_p) == 1 and fd_p[0] == 0.0
    assert len(fd_j) == 1 and fd_j[0] == 0.0


def test_calculate_dvars():
    """Test DVARS calculation on synthetic time series."""
    img, _ = _create_2d_phantom(num_frames=3)
    dvars = calculate_dvars(img)

    assert len(dvars) == 3
    assert dvars[0] == 0.0
    assert dvars[1] > 0.0
    assert dvars[2] > 0.0

    # Test with 1 frame
    dvars_single = calculate_dvars(img.numpy()[..., :1])
    assert len(dvars_single) == 1
    assert dvars_single[0] == 0.0

    # Test with custom empty mask triggering fallback
    empty_mask = np.zeros((32, 32), dtype=bool)
    dvars_empty = calculate_dvars(img, mask=empty_mask)
    assert len(dvars_empty) == 3


def test_extract_rigid_parameters_gimbal_lock_and_errors():
    """Test gimbal lock branches and error handling."""
    # Test invalid dim
    tx2d = ants.new_ants_transform(precision="float", dimension=2)
    with pytest.raises(ValueError, match="Unsupported spatial dimensionality"):
        _extract_rigid_parameters(tx2d, dim=4)

    # Test _create_identity_transform with invalid dim
    with pytest.raises(ValueError, match="Unsupported dimension"):
        _create_identity_transform(dim=4, filename="invalid.mat")

    # Test gimbal lock sy > 0.999999
    tx3d = ants.new_ants_transform(precision="float", dimension=3)
    # Rotation with sy = -R[2, 0] = 1.0 -> R[2, 0] = -1.0
    R_gimbal = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ], dtype=np.float64)
    tx3d.set_parameters(np.concatenate([R_gimbal.ravel(), np.zeros(3)]))
    tx3d.set_fixed_parameters(np.zeros(3))
    trans, rot, T_h = _extract_rigid_parameters(tx3d, dim=3)
    assert np.isclose(rot[1], np.pi / 2.0)


def test_motion_correction_2d_mean_reference():
    """Test 2D+t motion correction with temporal mean reference."""
    img, shifts = _create_2d_phantom(num_frames=3)

    result = motion_correction(
        img,
        reference="mean",
        type_of_transform="Rigid",
        aff_metric="meansquares",
        verbose=False,
    )

    expected_keys = [
        "motion_corrected",
        "transforms",
        "fwdtransforms",
        "invtransforms",
        "fd",
        "fd_power",
        "fd_jenkinson",
        "motion_parameters",
        "dvars",
        "dvars_pre",
        "dvars_post",
        "reference",
        "summary",
    ]
    for key in expected_keys:
        assert key in result, f"Key '{key}' missing from motion_correction result."

    corrected = result["motion_corrected"]
    assert corrected.dimension == 3
    assert corrected.shape == img.shape

    summary = result["summary"]
    assert summary["temporal_variance_post"] < summary["temporal_variance_pre"]
    assert summary["temporal_variance_reduction_percent"] > 95.0

    fd = result["fd"]
    assert len(fd) == 3
    assert fd[0] == 0.0
    assert fd[1] > 0.0
    assert fd[2] > 0.0

    mp = result["motion_parameters"]
    assert len(mp) == 3
    assert mp.spatial_dimension == 2
    assert "tx" in mp
    assert "ty" in mp
    assert "rz" in mp


def test_motion_correction_2d_frame_reference():
    """Test 2D+t motion correction using frame 0 as reference."""
    img, shifts = _create_2d_phantom(num_frames=3)

    result = motion_correction(
        img,
        reference=0,
        type_of_transform="Rigid",
        aff_metric="meansquares",
    )

    mp = result["motion_parameters"]
    assert np.isclose(mp[0, 0], 0.0, atol=1e-5)
    assert np.isclose(mp[0, 1], 0.0, atol=1e-5)

    assert np.isclose(mp[1, 0], shifts[1][0], atol=0.2)
    assert np.isclose(mp[1, 1], shifts[1][1], atol=0.2)

    summary = result["summary"]
    assert summary["temporal_variance_reduction_percent"] > 95.0

    assert result["fd_power"][0] == 0.0
    assert result["fd_power"][1] > 0.0
    assert result["fd_jenkinson"][0] == 0.0
    assert result["fd_jenkinson"][1] > 0.0


def test_motion_correction_3d_phantom():
    """Test 3D+t motion correction with 6-DOF parameter extraction."""
    img, shifts = _create_3d_phantom(num_frames=3)

    result = syntx.motion_correction(
        img,
        reference=0,
        type_of_transform="Rigid",
        aff_metric="meansquares",
        fd_method="jenkinson",
    )

    assert result.motion_corrected.dimension == 4
    assert result.motion_corrected.shape == img.shape

    mp = result.motion_parameters
    assert mp.shape == (3, 6)
    assert mp.spatial_dimension == 3

    assert np.allclose(mp[0], 0.0, atol=1e-4)
    assert np.isclose(mp[1, 0], shifts[1][0], atol=0.25)
    assert np.isclose(mp[1, 1], shifts[1][1], atol=0.25)
    assert np.isclose(mp[1, 2], shifts[1][2], atol=0.25)

    assert result.summary["temporal_variance_post"] < result.summary["temporal_variance_pre"]
    assert result.summary["temporal_variance_reduction_percent"] > 90.0

    assert result.fd[0] == 0.0
    assert result.fd[1] > 0.0


def test_motion_correction_explicit_reference():
    """Test providing an explicit ANTsImage reference volume."""
    img, _ = _create_2d_phantom(num_frames=3)
    ref_slice = ants.slice_image(img, axis=2, idx=0)

    result = motion_correction(
        img,
        reference=ref_slice,
        type_of_transform="QuickRigid",
        aff_metric="meansquares",
    )

    assert result["reference"] is ref_slice
    assert result["motion_corrected"].shape == img.shape
    assert result["summary"]["temporal_variance_post"] < result["summary"]["temporal_variance_pre"]


def test_motion_correction_two_pass():
    """Test two-pass motion correction refinement."""
    img, _ = _create_2d_phantom(num_frames=3)

    result = motion_correction(
        img,
        reference="mean",
        two_pass=True,
        type_of_transform="Rigid",
        aff_metric="meansquares",
    )

    assert result["motion_corrected"].shape == img.shape
    assert result["summary"]["temporal_variance_reduction_percent"] > 98.0


def test_motion_correction_with_numpy_and_filepath_and_options(tmp_path):
    """Test motion correction with numpy input, file path, outprefix, mask, and interpolator."""
    img, _ = _create_2d_phantom(num_frames=2)

    # 1. Test with numpy array input directly
    arr = img.numpy()
    res_np = motion_correction(arr, reference=0, type_of_transform="QuickRigid", verbose=True)
    assert res_np.motion_corrected.shape == arr.shape

    # 2. Test saving to disk and reading from file path string
    nii_path = str(tmp_path / "test_series.nii.gz")
    ants.image_write(img, nii_path)

    out_prefix = str(tmp_path / "moco_out")
    mask_img = ants.from_numpy((img.numpy().mean(axis=-1) > 0.05).astype(np.float32))

    res_file = motion_correction(
        nii_path,
        reference="mean",
        type_of_transform="QuickRigid",
        mask=mask_img,
        outprefix=out_prefix,
        interpolator="nearestNeighbor",
        backend="ants",
    )
    assert res_file.motion_corrected.shape == img.shape
    assert len(res_file.transforms) == 2
    assert os.path.exists(res_file.transforms[0][0])


def test_motion_correction_invalid_inputs():
    """Test proper exception handling for invalid inputs."""
    # 2D static image passed instead of time series
    static_2d = ants.from_numpy(np.zeros((10, 10), dtype=np.float32))
    with pytest.raises(ValueError, match="requires a time-series image"):
        motion_correction(static_2d)

    # Valid 2D+t image
    img, _ = _create_2d_phantom(num_frames=3)

    # Invalid string reference
    with pytest.raises(ValueError, match="Unknown reference string"):
        motion_correction(img, reference="invalid_ref")

    # Frame index out of bounds
    with pytest.raises(IndexError, match="out of valid range"):
        motion_correction(img, reference=99)

    # Explicit reference of wrong dimension
    ref_3d = ants.from_numpy(np.zeros((10, 10, 10), dtype=np.float32))
    with pytest.raises(ValueError, match="Reference image dimension"):
        motion_correction(img, reference=ref_3d)

    # Unsupported reference object type
    with pytest.raises(TypeError, match="Unsupported reference type"):
        motion_correction(img, reference=[1, 2, 3])
