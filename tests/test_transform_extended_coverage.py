"""
Unit tests targeting high code coverage for syntx.transform.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import pytest
import ants

from syntx.transform import (
    SyNToTransform,
    export_ants_displacement_field,
    export_ants_affine_transform
)


def test_synto_transform_creation_and_device_transfer():
    aff_grid = torch.eye(3).unsqueeze(0).repeat(1, 10, 10, 1)
    warp = torch.zeros(1, 10, 10, 2)
    meta = {
        'origin': [0.0, 0.0],
        'spacing': [1.0, 1.0],
        'direction': np.eye(2),
        'shape': (10, 10)
    }

    st = SyNToTransform(aff_grid, warp, meta, device='cpu')
    assert st.dim == 2
    assert st.spatial == (10, 10)

    # Move device
    st.to('cpu')
    assert st.device == 'cpu'


def test_synto_transform_apply_and_jacobian():
    aff_grid = F.affine_grid(torch.eye(2, 3).unsqueeze(0), size=[1, 1, 16, 16], align_corners=True)
    warp = torch.zeros(1, 16, 16, 2)
    meta = {
        'origin': [0.0, 0.0],
        'spacing': [1.0, 1.0],
        'direction': np.eye(2),
        'shape': (16, 16)
    }

    st = SyNToTransform(aff_grid, warp, meta, device='cpu', is_physical=True)

    img_tensor = torch.randn(1, 1, 16, 16)
    warped_tensor = st.apply(img_tensor, mode='bilinear')
    assert warped_tensor.shape == (1, 1, 16, 16)

    jac_det = st.get_jacobian_determinant()
    assert jac_det.shape == (16, 16)


def test_export_ants_transforms_and_displacement_field(tmp_path):
    disp_np = np.zeros((16, 16, 2), dtype=np.float32)
    disp_img = export_ants_displacement_field(disp_np, origin=(0.0, 0.0), spacing=(1.0, 1.0), direction=np.eye(2))
    assert isinstance(disp_img, ants.ANTsImage)

    tx_path = str(tmp_path / "test_affine.mat")
    res_tx = export_ants_affine_transform(np.eye(2), np.zeros(2), dim=2, filename=tx_path)
    assert res_tx is not None
