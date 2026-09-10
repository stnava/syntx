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
    export_ants_affine_transform,
    create_ants_affine,
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


def test_synto_transform_export_in_memory():
    warp = torch.zeros(1, 12, 12, 2)
    meta = {
        'origin': [0.0, 0.0],
        'spacing': [1.0, 1.0],
        'direction': np.eye(2),
        'shape': (12, 12)
    }
    tx = SyNToTransform(warp_field=warp, metadata=meta, device='cpu')
    out = tx.export(outprefix=None)
    assert isinstance(out, dict)
    assert 'warp_field' in out
    assert 'fwd_warp' in out
    assert 'inv_warp' in out
    assert 'affine_matrix' in out
    assert 'metadata' in out
    assert out['metadata'] == meta


def test_synto_transform_export_files_with_affine_tuple(tmp_path):
    shape = (10, 12, 14)
    warp = torch.zeros(1, *shape, 3)
    meta = {
        'origin': [1.0, 2.0, 3.0],
        'spacing': [0.8, 1.2, 1.5],
        'direction': np.eye(3),
        'shape': shape
    }
    M_phys = np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    t_phys = np.array([2.5, -1.0, 3.0], dtype=np.float32)

    tx = SyNToTransform(
        warp_field=warp,
        metadata=meta,
        device='cpu',
        affine_matrix=(M_phys, t_phys)
    )

    outprefix = str(tmp_path / "exp_tuple_")
    res = tx.export(outprefix=outprefix)

    assert os.path.exists(res['warp'])
    assert os.path.exists(res['inverse_warp'])
    assert os.path.exists(res['affine'])
    assert len(res['fwd_transforms']) == 2
    assert len(res['inv_transforms']) == 2
    assert res['fwd_transforms'][0] == res['warp']
    assert res['fwd_transforms'][1] == res['affine']
    assert res['inv_transforms'][0] == res['affine']
    assert res['inv_transforms'][1] == res['inverse_warp']

    tx_read = ants.read_transform(res['affine'])
    assert isinstance(tx_read, ants.ANTsTransform)
    warp_read = ants.image_read(res['warp'])
    assert warp_read.components == 3
    inv_warp_read = ants.image_read(res['inverse_warp'])
    assert inv_warp_read.components == 3


def test_synto_transform_export_files_with_homogeneous_and_matrix(tmp_path):
    shape = (12, 16)
    warp = torch.zeros(1, *shape, 2)
    meta = {
        'origin': [0.0, 0.0],
        'spacing': [1.0, 1.0],
        'direction': np.eye(2),
        'shape': shape
    }
    homo = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    tx = SyNToTransform(warp_field=warp, metadata=meta, device='cpu', affine_matrix=homo)
    res = tx.export(outprefix=str(tmp_path / "homo_"))
    assert os.path.exists(res['affine'])
    assert os.path.exists(res['warp'])

    aug = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]], dtype=np.float32)
    tx2 = SyNToTransform(warp_field=warp, metadata=meta, device='cpu', affine_matrix=aug)
    res2 = tx2.export(outprefix=str(tmp_path / "aug_"))
    assert os.path.exists(res2['affine'])


def test_synto_transform_export_with_ants_transform(tmp_path):
    shape = (10, 10)
    warp = torch.zeros(1, *shape, 2)
    meta = {
        'origin': [0.0, 0.0],
        'spacing': [1.0, 1.0],
        'direction': np.eye(2),
        'shape': shape
    }
    ants_tx = create_ants_affine(np.eye(2), np.array([3.0, 4.0]), dim=2)
    tx = SyNToTransform(warp_field=warp, metadata=meta, device='cpu', affine_matrix=ants_tx)
    res = tx.export(outprefix=str(tmp_path / "ants_tx_"))
    assert os.path.exists(res['affine'])


def test_synto_transform_invert():
    shape = (12, 12)
    warp = torch.zeros(1, *shape, 2)
    warp[..., 0] = 0.05
    meta = {
        'origin': [0.0, 0.0],
        'spacing': [1.0, 1.0],
        'direction': np.eye(2),
        'shape': shape
    }
    M = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    t = np.array([1.0, -1.0], dtype=np.float32)

    tx = SyNToTransform(warp_field=warp, metadata=meta, device='cpu', affine_matrix=(M, t))
    inv_tx = tx.invert()

    assert isinstance(inv_tx, SyNToTransform)
    assert inv_tx.warp_field is not None
    assert inv_tx.warp_inv_field is tx.warp_field
    assert inv_tx.affine_matrix is not None
    M_inv, t_inv = inv_tx.affine_matrix
    assert np.allclose(M_inv @ M, np.eye(2), atol=1e-4)
    assert np.allclose(M_inv @ t + t_inv, np.zeros(2), atol=1e-4)

    re_inv = inv_tx.invert()
    assert re_inv.warp_field is tx.warp_field


def test_synto_transform_jacobian_alias_and_image():
    shape = (16, 16)
    warp = torch.zeros(1, *shape, 2)
    meta = {
        'origin': [0.0, 0.0],
        'spacing': [1.0, 1.0],
        'direction': np.eye(2),
        'shape': shape
    }
    tx = SyNToTransform(warp_field=warp, metadata=meta, device='cpu')
    jac1 = tx.get_jacobian_determinant()
    jac2 = tx.jacobian()
    assert np.allclose(jac1, jac2)

    jac_t = tx.jacobian(method='bspline', as_numpy=False)
    assert isinstance(jac_t, torch.Tensor)

    jac_img = tx.jacobian_determinant_image()
    assert isinstance(jac_img, ants.ANTsImage)
    assert jac_img.shape == (16, 16)
    assert np.allclose(jac_img.numpy(), 1.0, atol=1e-3)


def test_synto_transform_to_ants(tmp_path):
    shape = (10, 10)
    warp = torch.zeros(1, *shape, 2)
    warp_inv = torch.zeros(1, *shape, 2)
    meta = {
        'origin': [0.0, 0.0],
        'spacing': [1.0, 1.0],
        'direction': np.eye(2),
        'shape': shape
    }
    M = np.eye(2, dtype=np.float32)
    t = np.array([1.0, 2.0], dtype=np.float32)

    tx = SyNToTransform(
        warp_field=warp,
        warp_inv_field=warp_inv,
        metadata=meta,
        device='cpu',
        affine_matrix=(M, t)
    )

    ants_objs = tx.to_ants()
    assert isinstance(ants_objs, dict)
    assert isinstance(ants_objs['warp'], ants.ANTsImage)
    assert isinstance(ants_objs['affine'], ants.ANTsTransform)
    assert isinstance(ants_objs['inverse_warp'], ants.ANTsImage)

    res = tx.to_ants(outprefix=str(tmp_path / "to_ants_disk_"))
    assert os.path.exists(res['warp'])
    assert os.path.exists(res['affine'])


def test_synto_transform_t_grid_export(tmp_path):
    shape = (8, 8)
    warp = torch.zeros(1, *shape, 2)
    meta = {
        'origin': [0.0, 0.0],
        'spacing': [1.0, 1.0],
        'direction': np.eye(2),
        'shape': shape
    }
    T_grid = torch.eye(3)[:2, :].unsqueeze(0)
    tx = SyNToTransform(
        warp_field=warp,
        metadata=meta,
        device='cpu',
        T_grid=T_grid
    )
    res = tx.export(outprefix=str(tmp_path / "t_grid_exp_"))
    assert os.path.exists(res['affine'])
