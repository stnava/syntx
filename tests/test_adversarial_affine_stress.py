"""Adversarial stress test suite for affine transformation roundtrips in syntx.spatial."""

import numpy as np
import pytest
import torch
import ants
import syntx.spatial as sp


def random_rotation_matrix_3d(rng):
    """Generate a random 3D rotation matrix (SO(3)) using QR decomposition."""
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det = 1) rather than reflection
    Q = Q @ np.diag([1.0, 1.0, np.linalg.det(Q)])
    return Q.astype(np.float32)


def random_rotation_matrix_2d(rng):
    """Generate a random 2D rotation matrix (SO(2))."""
    theta = rng.uniform(-np.pi, np.pi)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def generate_random_affine_3d(rng):
    """Generate a 4x4 affine matrix with asymmetric scale, shear, rotation, and translation."""
    R = random_rotation_matrix_3d(rng)
    # Asymmetric scales in [0.5, 2.0]
    scales = rng.uniform(0.5, 2.0, size=3).astype(np.float32)
    S = np.diag(scales)
    # Shears in [-0.3, 0.3]
    shear = np.eye(3, dtype=np.float32)
    shear[0, 1] = rng.uniform(-0.3, 0.3)
    shear[0, 2] = rng.uniform(-0.3, 0.3)
    shear[1, 0] = rng.uniform(-0.3, 0.3)
    shear[1, 2] = rng.uniform(-0.3, 0.3)
    shear[2, 0] = rng.uniform(-0.3, 0.3)
    shear[2, 1] = rng.uniform(-0.3, 0.3)

    linear_part = R @ S @ shear
    translation = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = linear_part
    T[:3, 3] = translation
    return T


def generate_random_geometry_3d(rng):
    """Generate random 3D ANTsImage with anisotropic spacing, arbitrary origin, and rotation direction."""
    shape = tuple(rng.integers(16, 48, size=3).tolist())
    spacing = tuple(rng.uniform(0.5, 3.5, size=3).tolist())
    origin = tuple(rng.uniform(-100.0, 100.0, size=3).tolist())
    direction = random_rotation_matrix_3d(rng)
    arr = np.zeros(shape, dtype=np.float32)
    return ants.from_numpy(arr, spacing=spacing, origin=origin, direction=direction)


def generate_random_affine_2d(rng):
    """Generate a 3x3 affine matrix with asymmetric scale, shear, rotation, and translation."""
    R = random_rotation_matrix_2d(rng)
    scales = rng.uniform(0.6, 1.8, size=2).astype(np.float32)
    S = np.diag(scales)
    shear = np.eye(2, dtype=np.float32)
    shear[0, 1] = rng.uniform(-0.3, 0.3)
    shear[1, 0] = rng.uniform(-0.3, 0.3)

    linear_part = R @ S @ shear
    translation = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)

    T = np.eye(3, dtype=np.float32)
    T[:2, :2] = linear_part
    T[:2, 2] = translation
    return T


def generate_random_geometry_2d(rng):
    """Generate random 2D ANTsImage with anisotropic spacing, arbitrary origin, and rotation direction."""
    shape = tuple(rng.integers(20, 64, size=2).tolist())
    spacing = tuple(rng.uniform(0.5, 2.5, size=2).tolist())
    origin = tuple(rng.uniform(-80.0, 80.0, size=2).tolist())
    direction = random_rotation_matrix_2d(rng)
    arr = np.zeros(shape, dtype=np.float32)
    return ants.from_numpy(arr, spacing=spacing, origin=origin, direction=direction)


class TestAdversarialAffineStress50Trials:
    """Stress test grid_to_physical_affine <-> physical_to_grid_affine across 50+ random configurations."""

    def test_50_trials_forward_roundtrip_3d_distinct_geometries(self):
        """Forward roundtrip: T_grid -> (M_phys, t_phys) -> T_grid' across 50 random configurations.
        Verifies that L_infinity < 10^-5 on every single trial.
        """
        rng = np.random.default_rng(seed=42)
        errors = []

        for trial in range(50):
            fixed = generate_random_geometry_3d(rng)
            moving = generate_random_geometry_3d(rng)
            T_grid = generate_random_affine_3d(rng)

            M_phys, t_phys = sp.grid_to_physical_affine(T_grid, fixed, moving)
            T_grid_rec = sp.physical_to_grid_affine(M_phys, t_phys, fixed, moving)

            max_err = float(np.max(np.abs(T_grid - T_grid_rec)))
            errors.append(max_err)

            assert max_err < 1e-5, (
                f"Trial {trial} failed: L_inf error {max_err:.2e} >= 1e-5.\n"
                f"Original T_grid:\n{T_grid}\n"
                f"Recovered T_grid:\n{T_grid_rec}\n"
                f"Diff:\n{np.abs(T_grid - T_grid_rec)}"
            )

        mean_err = np.mean(errors)
        max_err_all = np.max(errors)
        print(f"\n[3D Forward 50 trials] Max L_inf error: {max_err_all:.3e}, Mean L_inf error: {mean_err:.3e}")
        assert max_err_all < 1e-5

    def test_50_trials_reverse_roundtrip_3d_distinct_geometries(self):
        """Reverse roundtrip: (M_phys, t_phys) -> T_grid -> (M_phys', t_phys') across 50 configurations.
        Verifies that L_infinity < 10^-5 on M_phys and relative error < 10^-5 on t_phys.
        """
        rng = np.random.default_rng(seed=123)
        m_errors = []
        t_errors = []

        for trial in range(50):
            fixed = generate_random_geometry_3d(rng)
            moving = generate_random_geometry_3d(rng)

            # Generate random physical affine
            R = random_rotation_matrix_3d(rng)
            S = np.diag(rng.uniform(0.7, 1.5, size=3).astype(np.float32))
            M_phys = (R @ S).astype(np.float32)
            t_phys = rng.uniform(-50.0, 50.0, size=3).astype(np.float32)

            T_grid = sp.physical_to_grid_affine(M_phys, t_phys, fixed, moving)
            M_phys_rec, t_phys_rec = sp.grid_to_physical_affine(T_grid, fixed, moving)

            err_M = float(np.max(np.abs(M_phys - M_phys_rec)))
            err_t = float(np.max(np.abs(t_phys - t_phys_rec)))
            m_errors.append(err_M)
            t_errors.append(err_t)

            assert err_M < 1e-5, f"Trial {trial} failed on M_phys: err_M={err_M:.2e} >= 1e-5"
            assert err_t < 1e-4, f"Trial {trial} failed on t_phys: err_t={err_t:.2e} >= 1e-4"

        print(f"\n[3D Reverse 50 trials] Max M error: {np.max(m_errors):.3e}, Max t error: {np.max(t_errors):.3e}")

    def test_50_trials_forward_roundtrip_2d_distinct_geometries(self):
        """Forward roundtrip 2D: T_grid -> (M_phys, t_phys) -> T_grid' across 50 random configurations.
        Verifies that L_infinity < 10^-5 on every single trial.
        """
        rng = np.random.default_rng(seed=999)
        errors = []

        for trial in range(50):
            fixed = generate_random_geometry_2d(rng)
            moving = generate_random_geometry_2d(rng)
            T_grid = generate_random_affine_2d(rng)

            M_phys, t_phys = sp.grid_to_physical_affine(T_grid, fixed, moving)
            T_grid_rec = sp.physical_to_grid_affine(M_phys, t_phys, fixed, moving)

            max_err = float(np.max(np.abs(T_grid - T_grid_rec)))
            errors.append(max_err)

            assert max_err < 1e-5, (
                f"2D Trial {trial} failed: L_inf error {max_err:.2e} >= 1e-5.\n"
                f"Original:\n{T_grid}\nRecovered:\n{T_grid_rec}"
            )

        mean_err = np.mean(errors)
        max_err_all = np.max(errors)
        print(f"\n[2D Forward 50 trials] Max L_inf error: {max_err_all:.3e}, Mean L_inf error: {mean_err:.3e}")
        assert max_err_all < 1e-5

    def test_point_mapping_invariance_under_affine_bridge(self):
        """Point transformation consistency test:
        Transforming points via normalized grid affine matrix T_grid MUST yield identical
        physical coordinates as transforming points via (M_phys, t_phys).
        """
        rng = np.random.default_rng(seed=777)

        for _ in range(20):
            fixed = generate_random_geometry_3d(rng)
            moving = generate_random_geometry_3d(rng)
            T_grid = generate_random_affine_3d(rng)

            M_phys, t_phys = sp.grid_to_physical_affine(T_grid, fixed, moving)

            # Sample random points in normalized grid space of fixed image: g_f in [-1, 1]^3
            g_f = rng.uniform(-1.0, 1.0, size=(10, 3)).astype(np.float32)

            # Route 1: Map g_f to g_m via T_grid, then convert g_m to physical coords p_m1
            # In T_grid homogeneous coordinates:
            g_f_homo = np.hstack([g_f, np.ones((10, 1), dtype=np.float32)])  # (10, 4)
            g_m = (T_grid @ g_f_homo.T).T[:, :3]  # (10, 3)

            # Moving grid to physical mapping:
            # p = Dy @ (Sy * (Ky * g + Cy)) + Oy
            Ny = np.array(moving.shape, dtype=np.float32)
            Sy = np.array(moving.spacing, dtype=np.float32)
            Oy = np.array(moving.origin, dtype=np.float32)
            Dy = np.array(moving.direction, dtype=np.float32)
            Ky = (Ny - 1) / 2.0
            Cy = (Ny - 1) / 2.0
            # Voxel indices in moving:
            idx_m = g_m * Ky + Cy
            p_m1 = (Dy @ (Sy * idx_m).T).T + Oy

            # Route 2: Convert g_f to fixed physical coords p_f, then map via (M_phys, t_phys)
            Nx = np.array(fixed.shape, dtype=np.float32)
            Sx = np.array(fixed.spacing, dtype=np.float32)
            Ox = np.array(fixed.origin, dtype=np.float32)
            Dx = np.array(fixed.direction, dtype=np.float32)
            Kx = (Nx - 1) / 2.0
            Cx = (Nx - 1) / 2.0
            idx_f = g_f * Kx + Cx
            p_f = (Dx @ (Sx * idx_f).T).T + Ox

            p_m2 = (M_phys @ p_f.T).T + t_phys

            # Parity check
            pt_diff = np.max(np.abs(p_m1 - p_m2))
            assert pt_diff < 1e-4, f"Point mapping mismatch between grid and physical paths: {pt_diff:.2e} mm"

    def test_pytorch_tensor_input_compatibility(self):
        """Verify grid_to_physical_affine and physical_to_grid_affine seamlessly accept PyTorch tensors."""
        rng = np.random.default_rng(seed=555)
        fixed = generate_random_geometry_3d(rng)
        moving = generate_random_geometry_3d(rng)

        T_grid_np = generate_random_affine_3d(rng)
        T_grid_torch = torch.from_numpy(T_grid_np)

        M_phys, t_phys = sp.grid_to_physical_affine(T_grid_torch, fixed, moving)
        assert isinstance(M_phys, np.ndarray)
        assert isinstance(t_phys, np.ndarray)

        M_torch = torch.from_numpy(M_phys)
        t_torch = torch.from_numpy(t_phys)
        T_rec = sp.physical_to_grid_affine(M_torch, t_torch, fixed, moving)

        assert isinstance(T_rec, np.ndarray)
        assert np.max(np.abs(T_grid_np - T_rec)) < 1e-5

    def test_extreme_anisotropy_stress(self):
        """Stress test with 10:1 voxel spacing ratio and large dimensions."""
        shape_f = (128, 128, 16)
        spacing_f = (0.3, 0.3, 3.0)
        origin_f = (-150.0, 200.0, -50.0)
        direction_f = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
        fixed = ants.from_numpy(np.zeros(shape_f, dtype=np.float32), spacing=spacing_f, origin=origin_f, direction=direction_f)

        shape_m = (64, 64, 32)
        spacing_m = (0.5, 0.5, 2.5)
        origin_m = (50.0, -100.0, 150.0)
        direction_m = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        moving = ants.from_numpy(np.zeros(shape_m, dtype=np.float32), spacing=spacing_m, origin=origin_m, direction=direction_m)

        rng = np.random.default_rng(seed=321)
        for _ in range(10):
            T_grid = generate_random_affine_3d(rng)
            M_phys, t_phys = sp.grid_to_physical_affine(T_grid, fixed, moving)
            T_rec = sp.physical_to_grid_affine(M_phys, t_phys, fixed, moving)
            assert np.max(np.abs(T_grid - T_rec)) < 1e-5
