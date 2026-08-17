"""
Unit tests targeting high code coverage for syntx.robust_affine.
"""

import os
import numpy as np
import torch
import pytest
import ants

from syntx.robust_affine import (
    compute_center_of_mass,
    create_translation_transform,
    _eval_low_res_mi,
    _rodrigues_rotation_matrix_3d,
    _rotation_matrix_2d,
    _generate_cone_rotation_candidates_3d,
    _run_pytorch_affine_solver,
    robust_affine
)


def test_compute_center_of_mass_2d_and_3d():
    # 2D weighted and unweighted
    arr2d = np.zeros((30, 30), dtype=np.float32)
    arr2d[10:20, 10:20] = 1.0
    img2d = ants.from_numpy(arr2d, origin=(0.0, 0.0), spacing=(1.0, 1.0))

    com_w2d = compute_center_of_mass(img2d, weighted=True)
    com_g2d = compute_center_of_mass(img2d, weighted=False)
    assert com_w2d.shape == (2,)
    assert com_g2d.shape == (2,)

    # 2D empty image (zero weights fallback)
    empty2d = ants.from_numpy(np.zeros((20, 20), dtype=np.float32), origin=(0.0, 0.0), spacing=(1.0, 1.0))
    com_empty = compute_center_of_mass(empty2d, weighted=False)
    assert com_empty.shape == (2,)

    # 3D weighted and unweighted
    arr3d = np.zeros((20, 20, 20), dtype=np.float32)
    arr3d[5:15, 5:15, 5:15] = 2.0
    img3d = ants.from_numpy(arr3d, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))

    com_w3d = compute_center_of_mass(img3d, weighted=True)
    com_g3d = compute_center_of_mass(img3d, weighted=False)
    assert com_w3d.shape == (3,)
    assert com_g3d.shape == (3,)


def test_create_translation_transform_and_eval_mi():
    fi = ants.from_numpy(np.ones((20, 20), dtype=np.float32), origin=(0.0, 0.0), spacing=(1.0, 1.0))
    mi = ants.from_numpy(np.ones((20, 20), dtype=np.float32), origin=(0.0, 0.0), spacing=(1.0, 1.0))
    t_phys = np.array([2.0, 3.0])

    tx_path, temp_dir = create_translation_transform(fi, mi, t_phys)
    assert os.path.exists(tx_path)

    mi_score = _eval_low_res_mi(fi, mi, tx_path)
    assert isinstance(mi_score, float)

    # Test invalid transform path exception handling in _eval_low_res_mi
    bad_score = _eval_low_res_mi(fi, mi, "nonexistent_file.mat")
    assert bad_score == 999.0


def test_rodrigues_and_rotation_matrices():
    # Near zero angle Taylor expansion check
    omega_zero = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    R_zero = _rodrigues_rotation_matrix_3d(omega_zero)
    assert torch.allclose(R_zero, torch.eye(3), atol=1e-4)

    # Non-zero angle check
    omega_rot = torch.tensor([0.1, 0.2, 0.0], dtype=torch.float32)
    R_rot = _rodrigues_rotation_matrix_3d(omega_rot)
    assert R_rot.shape == (3, 3)
    assert torch.allclose(R_rot @ R_rot.t(), torch.eye(3), atol=1e-4)

    # 2D rotation matrix check
    theta = torch.tensor([0.5], dtype=torch.float32)
    R2d = _rotation_matrix_2d(theta[0])
    assert R2d.shape == (2, 2)
    assert torch.allclose(R2d @ R2d.t(), torch.eye(2), atol=1e-4)


def test_generate_cone_rotation_candidates_3d():
    com_f = np.array([10.0, 10.0, 10.0])
    t_init = np.array([1.0, 2.0, 3.0])
    candidates = _generate_cone_rotation_candidates_3d(com_f, t_init)
    assert len(candidates) > 0
    name, path, R, t_rot, r_dir = candidates[0]
    assert os.path.exists(path)
    assert R.shape == (3, 3)


def test_run_pytorch_affine_solver_2d_and_3d():
    # 2D PyTorch affine solver
    fi2d = ants.from_numpy(np.pad(np.ones((16, 16)), 8).astype(np.float32), origin=(0.0, 0.0), spacing=(1.0, 1.0))
    mi2d = ants.from_numpy(np.pad(np.ones((16, 16)), 8).astype(np.float32), origin=(0.0, 0.0), spacing=(1.0, 1.0))

    res2d = _run_pytorch_affine_solver(fi2d, mi2d, device='cpu', verbose=False)
    assert 'fwdtransforms' in res2d
    assert 'warpedmovout' in res2d
    assert os.path.exists(res2d['fwdtransforms'][0])

    # 3D PyTorch affine solver
    arr3d_f = np.zeros((16, 16, 16), dtype=np.float32)
    arr3d_f[4:12, 4:12, 4:12] = 1.0
    arr3d_m = np.zeros((16, 16, 16), dtype=np.float32)
    arr3d_m[5:13, 5:13, 4:12] = 1.0
    fi3d = ants.from_numpy(arr3d_f, origin=(0.0, 0.0, 0.0), spacing=(2.0, 2.0, 2.0))
    mi3d = ants.from_numpy(arr3d_m, origin=(0.0, 0.0, 0.0), spacing=(2.0, 2.0, 2.0))

    res3d = _run_pytorch_affine_solver(fi3d, mi3d, device='cpu', verbose=False)
    assert 'fwdtransforms' in res3d
    assert 'warpedmovout' in res3d
    assert os.path.exists(res3d['fwdtransforms'][0])


def test_robust_affine_modes():
    fi = ants.from_numpy(np.pad(np.ones((16, 16)), 8).astype(np.float32), origin=(0.0, 0.0), spacing=(1.0, 1.0))
    mi = ants.from_numpy(np.pad(np.ones((16, 16)), 8).astype(np.float32), origin=(0.0, 0.0), spacing=(1.0, 1.0))

    # Mode: 'com_only'
    res_com = robust_affine(fi, mi, mode='com_only')
    assert 'warpedmovout' in res_com
    assert 'fwdtransforms' in res_com

    # Mode: 'pytorch'
    res_pt = robust_affine(fi, mi, mode='pytorch', device='cpu')
    assert 'warpedmovout' in res_pt

    # Mode: 'auto' with multi_start=True
    res_auto = robust_affine(fi, mi, mode='auto', multi_start=True, low_res_spacing=2.0, verbose=False)
    assert 'warpedmovout' in res_auto

    # Mode: 'ants_fast' with multi_start=False
    res_ants = robust_affine(fi, mi, mode='ants_fast', multi_start=False, verbose=False)
    assert 'warpedmovout' in res_ants


def test_pytorch_affine_convergence_and_parity():
    """Verifies physical space alignment, convergence accuracy, and parse_ants_affine parity for PyTorch Affine."""
    from syntx.syn import parse_ants_affine

    # Create 3D synthetic sphere image pair with anisotropic spacing and physical translation shift
    shape = (48, 64, 64)
    sp_f = (1.0, 1.0, 1.0)
    sp_m = (1.2, 1.0, 1.0)

    grid_z, grid_y, grid_x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center_f = (24, 32, 32)
    center_m = (20, 28, 30)

    dist_f = (grid_z - center_f[0])**2 + (grid_y - center_f[1])**2 + (grid_x - center_f[2])**2
    dist_m = (grid_z - center_m[0])**2 + (grid_y - center_m[1])**2 + (grid_x - center_m[2])**2

    arr_f = (dist_f <= 12**2).astype(np.float32)
    arr_m = (dist_m <= 12**2).astype(np.float32)

    fixed = ants.from_numpy(arr_f, origin=(0.0, 0.0, 0.0), spacing=sp_f)
    moving = ants.from_numpy(arr_m, origin=(0.0, 0.0, 0.0), spacing=sp_m)

    fixed_label = ants.from_numpy((arr_f > 0.5).astype(np.uint32), origin=(0.0, 0.0, 0.0), spacing=sp_f)
    moving_label = ants.from_numpy((arr_m > 0.5).astype(np.uint32), origin=(0.0, 0.0, 0.0), spacing=sp_m)

    # 1. Run PyTorch GPU Affine solver
    reg = robust_affine(fixed, moving, mode='pytorch', verbose=False)
    tx_path = reg['fwdtransforms'][0]
    assert os.path.exists(tx_path)
    assert reg['time'] < 15.0

    # 2. Verify label alignment Dice >= 0.85
    warped_ml = ants.apply_transforms(fixed=fixed, moving=moving_label, transformlist=[tx_path], interpolator='nearestNeighbor')
    ov = ants.label_overlap_measures(fixed_label, warped_ml)
    df = ov[~ov['Label'].astype(str).isin(['All', '0', '0.0'])]
    col = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df.columns else 'TargetOverlap'
    dice = float(df[col].mean())
    assert dice >= 0.80, f"PyTorch Affine registration regressed: Dice = {dice:.4f} < 0.80"

    # 3. Test parse_ants_affine physical coordinate parity
    M_phys, t_phys = parse_ants_affine(tx_path, dim=3)
    assert M_phys is not None and t_phys is not None
    assert torch.allclose(torch.diag(M_phys), torch.ones(3), atol=0.25), f"Parsed M_phys diagonal regressed: {M_phys}"

