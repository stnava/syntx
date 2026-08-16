import pytest
import torch
import numpy as np

from syntx.core.jacobian import (
    _spatial_jacobian_nd,
    compute_jacobian_determinant_nd,
    compute_physical_jacobian_determinant,
)


def test_jacobian_determinant_zero_displacement():
    # Zero displacement field = identity map, det(J) should be 1.0 everywhere
    disp_2d = torch.zeros(1, 16, 16, 2)
    detJ_2d = compute_jacobian_determinant_nd(disp_2d)
    assert torch.allclose(detJ_2d, torch.ones_like(detJ_2d), atol=1e-5)

    disp_3d = torch.zeros(1, 8, 8, 8, 3)
    detJ_3d = compute_jacobian_determinant_nd(disp_3d)
    assert torch.allclose(detJ_3d, torch.ones_like(detJ_3d), atol=1e-5)


def test_physical_jacobian_determinant_zero_displacement():
    disp_2d = torch.zeros(1, 16, 16, 2)
    direction = torch.eye(2)
    spacing = torch.tensor([1.0, 1.0])

    detJ_phys_2d = compute_physical_jacobian_determinant(disp_2d, direction, spacing)
    assert torch.allclose(detJ_phys_2d, torch.ones_like(detJ_phys_2d), atol=1e-5)

    disp_3d = torch.zeros(1, 8, 8, 8, 3)
    direction_3d = torch.eye(3)
    spacing_3d = torch.tensor([1.0, 1.0, 1.0])

    detJ_phys_3d = compute_physical_jacobian_determinant(disp_3d, direction_3d, spacing_3d)
    assert torch.allclose(detJ_phys_3d, torch.ones_like(detJ_phys_3d), atol=1e-5)
