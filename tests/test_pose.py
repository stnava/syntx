"""Unit tests for syntx.pose: probe_points, pose_error, pose_error_report."""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest
import ants

import syntx
from syntx.pose import probe_points, pose_error, pose_error_report, PoseErrorReport


def _create_synthetic_fixed(shape=(32, 32, 11), spacing=(1.0, 1.0, 3.5), origin=(0.0, 0.0, 0.0)):
    """Create a synthetic 3D ANTsImage."""
    arr = np.ones(shape, dtype=np.float32)
    img = ants.from_numpy(arr, origin=origin, spacing=spacing)
    return img


def test_probe_points_default_geometry():
    """Default probe points comprise 8 corners at (+-15, +-15, +-5) mm plus the center."""
    fixed = _create_synthetic_fixed()
    c = np.asarray(ants.get_center_of_mass(fixed * 0 + 1), dtype=np.float64)
    probes = probe_points(fixed, kind="default")

    assert isinstance(probes, np.ndarray)
    assert probes.shape == (9, 3)

    # 9th point is the center
    np.testing.assert_allclose(probes[8], c, atol=1e-5)

    # First 8 are the corner permutations
    expected_corners = [
        c + np.array([x, y, z], dtype=np.float64)
        for x in (-15.0, 15.0)
        for y in (-15.0, 15.0)
        for z in (-5.0, 5.0)
    ]
    np.testing.assert_allclose(probes[:8], expected_corners, atol=1e-5)


def test_probe_points_slab_thick():
    """slab_thick expands the z lever arm up to 1.5 slices from the outer boundary."""
    fixed = _create_synthetic_fixed(shape=(64, 64, 11), spacing=(0.5, 0.5, 3.5))
    c = np.asarray(ants.get_center_of_mass(fixed * 0 + 1), dtype=np.float64)
    probes = probe_points(fixed, kind="slab_thick")

    assert probes.shape == (9, 3)
    np.testing.assert_allclose(probes[8], c, atol=1e-5)

    # Expected dz = max(5.0, (11 / 2 - 1.5) * 3.5) = 4.0 * 3.5 = 14.0 mm
    expected_dz = 14.0
    for i, p in enumerate(probes[:8]):
        dz = abs(p[2] - c[2])
        np.testing.assert_allclose(dz, expected_dz, atol=1e-5)


def test_probe_points_custom_kind():
    """Custom kind supports float (dz) or (dx, dy, dz) tuple."""
    c = np.array([10.0, 20.0, 30.0])
    p_float = probe_points(c, kind=12.5)
    assert p_float.shape == (9, 3)
    assert abs(p_float[0][2] - c[2]) == 12.5

    p_tuple = probe_points(c, kind=(10.0, 10.0, 8.0))
    assert abs(p_tuple[0][0] - c[0]) == 10.0
    assert abs(p_tuple[0][1] - c[1]) == 10.0
    assert abs(p_tuple[0][2] - c[2]) == 8.0

    with pytest.raises(ValueError, match="Unknown probe kind"):
        probe_points(c, kind="unsupported_kind")


def test_identical_transforms_zero_error():
    """Two identical transforms produce zero error across all probe points."""
    fixed = _create_synthetic_fixed()

    # Identity matrix
    eye = np.eye(4)
    assert pose_error(fixed, eye, eye) == 0.0

    # Non-trivial affine transform matrix
    mat = np.eye(4)
    mat[:3, :3] = [[0.96, -0.28, 0.0], [0.28, 0.96, 0.0], [0.0, 0.0, 1.0]]
    mat[:3, 3] = [1.5, -2.3, 4.1]
    assert pose_error(fixed, mat, mat) == 0.0

    # ANTs transform object
    tx = ants.create_ants_transform(transform_type="Euler3DTransform", dimension=3, translation=(1.2, 3.4, -5.6))
    assert pose_error(fixed, tx, tx) == 0.0


def test_known_3mm_translation():
    """A known 3.0 mm pure translation yields exactly 3.0 mm pose error."""
    fixed = _create_synthetic_fixed()
    eye = np.eye(4)

    for axis in range(3):
        shift = [0.0, 0.0, 0.0]
        shift[axis] = 3.0
        mat_trans = np.eye(4)
        mat_trans[:3, 3] = shift

        err = pose_error(fixed, eye, mat_trans)
        np.testing.assert_allclose(err, 3.0, atol=1e-6)

        # Reverse order produces the same scalar magnitude
        err_rev = pose_error(fixed, mat_trans, eye)
        np.testing.assert_allclose(err_rev, 3.0, atol=1e-6)

    # ANTs transform with 3 mm translation
    tx_trans = ants.create_ants_transform(transform_type="Euler3DTransform", dimension=3, translation=(0.0, 3.0, 0.0))
    tx_eye = ants.create_ants_transform(transform_type="Euler3DTransform", dimension=3)
    np.testing.assert_allclose(pose_error(fixed, tx_eye, tx_trans), 3.0, atol=1e-6)


def test_known_rotation():
    """
    A 90-degree rotation about the z-axis centered at the fixed center of mass
    produces a predictable maximum probe displacement.
    """
    fixed = _create_synthetic_fixed()
    c = np.asarray(ants.get_center_of_mass(fixed * 0 + 1), dtype=np.float64)

    # 90 deg rotation about z: (x', y') = (-y, x)
    # T(q) = R (q - c) + c = R q + (I - R) c
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    t = (np.eye(3) - R) @ c
    mat_rot = np.eye(4)
    mat_rot[:3, :3] = R
    mat_rot[:3, 3] = t

    eye = np.eye(4)
    rep = pose_error_report(fixed, eye, mat_rot)

    # At corner q = c + (15, 15, z):
    # R (q - c) = (-15, 15, z)
    # Difference (T(q) - q) = (-30, 0, 0), Euclidean norm = 30.0 mm
    # At corner q = c + (15, -15, z):
    # R (q - c) = (15, 15, z)
    # Difference (T(q) - q) = (0, 30, 0), Euclidean norm = 30.0 mm
    np.testing.assert_allclose(rep.max_error_mm, 30.0, atol=1e-6)

    # At probe index 8 (the center point c), displacement is exactly 0.0
    np.testing.assert_allclose(rep.probe_errors_mm[8], 0.0, atol=1e-6)


def test_itk_direction_convention():
    """
    ITK fixed->moving convention test:
    A forward transform maps physical fixed points to physical moving points: p_mov = T(p_fix).
    """
    fixed = _create_synthetic_fixed()
    shift = np.array([4.0, -2.0, 1.0], dtype=np.float64)
    expected_dist = float(np.linalg.norm(shift))

    mat_a = np.eye(4)
    mat_b = np.eye(4)
    mat_b[:3, 3] = shift

    rep = pose_error_report(fixed, mat_a, mat_b)
    np.testing.assert_allclose(rep.max_error_mm, expected_dist, atol=1e-6)
    np.testing.assert_allclose(rep.mean_error_mm, expected_dist, atol=1e-6)

    # Displacements
    # disp_A = 0
    # disp_B = shift
    # diff = disp_A - disp_B = -shift
    np.testing.assert_allclose(rep.displacements_a, np.zeros((9, 3)), atol=1e-6)
    np.testing.assert_allclose(rep.displacements_b, np.tile(shift, (9, 1)), atol=1e-6)


def test_pose_error_report_structure():
    """PoseErrorReport exposes dictionary keys, properties, and summary string."""
    fixed = _create_synthetic_fixed()
    mat_a = np.eye(4)
    mat_b = np.eye(4)
    mat_b[:3, 3] = [3.0, 4.0, 0.0]  # norm = 5.0

    rep = pose_error_report(fixed, mat_a, mat_b)

    assert isinstance(rep, PoseErrorReport)
    assert isinstance(rep, dict)
    assert rep["max_error_mm"] == 5.0
    assert rep.max_error_mm == 5.0
    assert rep.mean_error_mm == 5.0
    assert rep.z_mean_error_mm == 0.0
    assert rep.probe_errors_mm.shape == (9,)
    assert rep.probes.shape == (9, 3)
    assert rep.displacements_a.shape == (9, 3)
    assert rep.displacements_b.shape == (9, 3)
    assert rep.worst_probe_index in range(9)
    assert "PoseErrorReport" in rep.summary()
    assert "5.000 mm" in rep.summary()


def test_transform_file_and_list_support(tmp_path):
    """pose_error accepts transform file paths and single-element lists."""
    fixed = _create_synthetic_fixed()

    # Write a real ANTs transform to file
    tx = ants.create_ants_transform(transform_type="Euler3DTransform", dimension=3, translation=(0.0, 0.0, 2.5))
    mat_file = str(tmp_path / "test_tx.mat")
    ants.write_transform(tx, mat_file)

    # Compare file path with in-memory transform
    err = pose_error(fixed, mat_file, tx)
    np.testing.assert_allclose(err, 0.0, atol=1e-5)

    # Compare file path in list (like reg['fwdtransforms'])
    err_list = pose_error(fixed, [mat_file], tx)
    np.testing.assert_allclose(err_list, 0.0, atol=1e-5)

    # 4x4 matrix equivalent
    mat4 = np.eye(4)
    mat4[:3, 3] = [0.0, 0.0, 2.5]
    err_mat = pose_error(fixed, [mat_file], mat4)
    np.testing.assert_allclose(err_mat, 0.0, atol=1e-5)


def test_error_handling():
    """Verify named exceptions on invalid inputs."""
    fixed = _create_synthetic_fixed()

    # None transform
    with pytest.raises(ValueError, match="Transform cannot be None"):
        pose_error(fixed, None, np.eye(4))

    # Missing file
    with pytest.raises(FileNotFoundError):
        pose_error(fixed, "/nonexistent/transform.mat", np.eye(4))

    # Invalid matrix shape
    with pytest.raises(ValueError, match="Transform matrix must have shape"):
        pose_error(fixed, np.ones((5, 5)), np.eye(4))

    # Unsupported type
    with pytest.raises(TypeError, match="Unsupported transform type"):
        pose_error(fixed, 12345, np.eye(4))

    # NaN / Inf transform matrix
    nan_mat = np.eye(4)
    nan_mat[0, 3] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        pose_error(fixed, nan_mat, np.eye(4))
