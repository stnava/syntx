import pytest
import torch
import numpy as np
import ants

from syntx.core.affine import (
    get_rotation_matrix,
    HierarchicalAffine,
    get_rotation_matrix,
    grid_to_physical_affine_torch,
    physical_to_grid_affine,
    grid_to_physical_affine,
    parse_ants_affine,
    compute_initial_grid,
)


def test_rotation_matrix_2d_and_3d():
    # 2D identity
    omega_2d_zero = torch.zeros(1)
    R_2d_zero = get_rotation_matrix(omega_2d_zero, dim=2)
    assert torch.allclose(R_2d_zero, torch.eye(2), atol=1e-6)

    # 2D 90 deg
    omega_2d_90 = torch.tensor([np.pi / 2.0], dtype=torch.float32)
    R_2d_90 = get_rotation_matrix(omega_2d_90, dim=2)
    expected_2d = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(R_2d_90, expected_2d, atol=1e-5)

    # 3D identity
    omega_3d_zero = torch.zeros(3)
    R_3d_zero = get_rotation_matrix(omega_3d_zero, dim=3)
    assert torch.allclose(R_3d_zero, torch.eye(3), atol=1e-6)

    # 3D rotation orthogonality: R @ R.T == I
    omega_3d = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    R_3d = get_rotation_matrix(omega_3d, dim=3)
    assert torch.allclose(R_3d @ R_3d.t(), torch.eye(3), atol=1e-5)
    assert torch.allclose(torch.det(R_3d), torch.tensor(1.0), atol=1e-5)


def test_hierarchical_affine_modes():
    for mode in ['Translation', 'Rigid', 'Similarity', 'Affine']:
        ha_2d = HierarchicalAffine(dim=2, transform_type=mode)
        T_2d = ha_2d.get_matrix()
        assert T_2d.shape == (3, 3)
        assert torch.allclose(T_2d, torch.eye(3))

        ha_3d = HierarchicalAffine(dim=3, transform_type=mode)
        T_3d = ha_3d.get_matrix()
        assert T_3d.shape == (4, 4)
        assert torch.allclose(T_3d, torch.eye(4))

        ha_3d.clamp_parameters()
        grid_mat = ha_3d.get_affine_grid_matrix()
        assert grid_mat.shape == (3, 4)


def test_parse_ants_affine_empty_or_none():
    M, t = parse_ants_affine([], dim=3)
    assert M is None and t is None
    M, t = parse_ants_affine(None, dim=3)
    assert M is None and t is None


def test_grid_to_physical_roundtrip_2d():
    fi = ants.from_numpy(np.zeros((32, 32), dtype=np.float32), spacing=(1.0, 1.0), origin=(0.0, 0.0))
    mi = ants.from_numpy(np.zeros((32, 32), dtype=np.float32), spacing=(1.0, 1.0), origin=(0.0, 0.0))

    T_grid = np.eye(3, dtype=np.float32)
    M_phys, t_phys = grid_to_physical_affine(T_grid, fi, mi)
    assert np.allclose(M_phys, np.eye(2), atol=1e-5)
    assert np.allclose(t_phys, np.zeros(2), atol=1e-5)
