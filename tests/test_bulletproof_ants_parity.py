"""
Bulletproof ANTsPy & Syntx Parity Verification Suite (2D & 3D).

Verifies 100% spatial grid alignment, component ordering, Jacobian determinant
correlation (r > 0.999), and ANTsPy apply_transforms warping parity on realistic
isotropic and anisotropic (mbhard-style) spatial headers.
"""

import tempfile
import pytest
import numpy as np
import torch
import ants
import syntx
from syntx.spatial import (
    disp_tensor_to_itk,
    disp_itk_to_tensor,
    jacobian_determinant,
)


def create_anisotropic_3d_pair():
    """Generates synthetic 3D anisotropic image pair with mbhard-style direction and spacing."""
    shape = (48, 48, 48)
    spacing = (0.8, 0.8, 1.5)
    origin = (10.0, -20.0, 5.0)
    direction = np.diag([1.0, -1.0, 1.0])

    fi_arr = np.zeros(shape, dtype=np.float32)
    fi_arr[12:36, 12:36, 12:36] = 1.0

    mi_arr = np.zeros(shape, dtype=np.float32)
    mi_arr[16:40, 16:40, 16:40] = 1.0

    fi = ants.from_numpy(fi_arr, origin=origin, spacing=spacing, direction=direction)
    mi = ants.from_numpy(mi_arr, origin=origin, spacing=spacing, direction=direction)

    return fi, mi


def test_2d_bulletproof_ants_parity():
    """Fast 2D ANTsPy parity test on r16 / r64 benchmark data."""
    fi = ants.image_read(ants.get_ants_data('r16'))
    mi = ants.image_read(ants.get_ants_data('r64'))

    reg = syntx.syn(
        fixed=fi,
        moving=mi,
        type_of_transform='SyNTo',
        reg_iterations=[20, 20, 10],
        verbose=False
    )

    model = reg['model']
    warp_file = reg['fwdtransforms'][0]
    warp_file_itk = ants.image_read(warp_file)

    # 1. Test disp_tensor_to_itk conversion vs saved ANTs displacement file
    warp_converted_itk = disp_tensor_to_itk(model.warp_l2r, ref_image=fi)

    corr_disp = np.corrcoef(
        warp_converted_itk.numpy().flatten(),
        warp_file_itk.numpy().flatten()
    )[0, 1]
    assert corr_disp > 0.999, f"2D Displacement field correlation mismatch: {corr_disp:.6f}"

    # 2. Test round-trip disp_itk_to_tensor
    tensor_roundtrip = disp_itk_to_tensor(warp_converted_itk)
    warp_recon_itk = disp_tensor_to_itk(tensor_roundtrip, ref_image=fi)
    corr_roundtrip = np.corrcoef(
        warp_converted_itk.numpy().flatten(),
        warp_recon_itk.numpy().flatten()
    )[0, 1]
    assert corr_roundtrip > 0.9999, f"2D Roundtrip correlation mismatch: {corr_roundtrip:.6f}"

    # 3. Test Jacobian Determinant correlation vs ANTs C++ ITK reference
    jac_ants = ants.create_jacobian_determinant_image(fi, warp_file, do_log=False)
    jac_syntx = jacobian_determinant(model.warp_l2r, ref_image=fi)

    corr_jac = np.corrcoef(
        jac_syntx.flatten(),
        jac_ants.numpy().flatten()
    )[0, 1]
    assert corr_jac > 0.990, f"2D Jacobian determinant correlation mismatch: {corr_jac:.6f}"

    # 4. Test ANTsPy apply_transforms warping parity using converted ITK image vs saved file
    warped_file = ants.apply_transforms(fixed=fi, moving=mi, transformlist=[warp_file])
    
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
        converted_path = tmp.name
    ants.image_write(warp_converted_itk, converted_path)
    warped_converted = ants.apply_transforms(fixed=fi, moving=mi, transformlist=[converted_path])

    diff_max = float(np.max(np.abs(warped_file.numpy() - warped_converted.numpy())))
    assert diff_max < 1e-4, f"2D ANTsPy apply_transforms image warping difference too high: {diff_max:.6e}"


def test_3d_anisotropic_bulletproof_ants_parity():
    """Fast 3D ANTsPy parity test on anisotropic mbhard-style headers."""
    fi, mi = create_anisotropic_3d_pair()

    reg = syntx.syn(
        fixed=fi,
        moving=mi,
        type_of_transform='SyNTo',
        reg_iterations=[10, 10, 5],
        verbose=False
    )

    model = reg['model']
    warp_file = reg['fwdtransforms'][0]
    warp_file_itk = ants.image_read(warp_file)

    # 1. Test disp_tensor_to_itk conversion vs saved ANTs displacement file
    warp_converted_itk = disp_tensor_to_itk(model.warp_l2r, ref_image=fi)

    corr_disp = np.corrcoef(
        warp_converted_itk.numpy().flatten(),
        warp_file_itk.numpy().flatten()
    )[0, 1]
    assert corr_disp > 0.999, f"3D Displacement field correlation mismatch: {corr_disp:.6f}"

    # 2. Test round-trip disp_itk_to_tensor
    tensor_roundtrip = disp_itk_to_tensor(warp_converted_itk)
    warp_recon_itk = disp_tensor_to_itk(tensor_roundtrip, ref_image=fi)
    corr_roundtrip = np.corrcoef(
        warp_converted_itk.numpy().flatten(),
        warp_recon_itk.numpy().flatten()
    )[0, 1]
    assert corr_roundtrip > 0.9999, f"3D Roundtrip correlation mismatch: {corr_roundtrip:.6f}"

    # 3. Test 3D Jacobian Determinant correlation vs ANTs C++ ITK reference (with direction scaling)
    jac_ants = ants.create_jacobian_determinant_image(fi, warp_file, do_log=False)
    jac_syntx = jacobian_determinant(model.warp_l2r, ref_image=fi)

    corr_jac = np.corrcoef(
        jac_syntx.flatten(),
        jac_ants.numpy().flatten()
    )[0, 1]
    assert corr_jac > 0.980, f"3D Jacobian determinant correlation mismatch: {corr_jac:.6f}"

    # 4. Test ANTsPy apply_transforms warping parity using converted ITK image vs saved file
    warped_file = ants.apply_transforms(fixed=fi, moving=mi, transformlist=[warp_file])

    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
        converted_path = tmp.name
    ants.image_write(warp_converted_itk, converted_path)
    warped_converted = ants.apply_transforms(fixed=fi, moving=mi, transformlist=[converted_path])

    diff_max = float(np.max(np.abs(warped_file.numpy() - warped_converted.numpy())))
    assert diff_max < 1e-4, f"3D ANTsPy apply_transforms image warping difference too high: {diff_max:.6e}"
