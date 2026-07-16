import os
import numpy as np
import pytest
import ants
import torch
import torch.nn.functional as F
import tempfile

from syntx.syn import registration
from syntx.transform import SyNToTransform

def test_ants_component_ordering_3d():
    # 1. Create a 3D image with a single bright spot in the center
    data = np.zeros((16, 24, 32), dtype=np.float32)
    data[8, 12, 16] = 100.0
    img = ants.from_numpy(data, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0))
    
    # 2. Create a displacement field with displacement only in numpy Component 0
    disp_x = np.zeros((16, 24, 32, 3), dtype=np.float32)
    disp_x[..., 0] = -5.0
    
    warp_img_x = ants.from_numpy(disp_x, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), has_components=True)
    
    f_x = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False).name
    try:
        ants.image_write(warp_img_x, f_x)
        warped_x = ants.apply_transforms(fixed=img, moving=img, transformlist=[f_x])
        warped_x_np = warped_x.numpy()
        peak_idx = np.unravel_index(np.argmax(warped_x_np), warped_x_np.shape)
        print(f"Original peak: (8, 12, 16). New peak with numpy Component 0: {peak_idx}")
    finally:
        if os.path.exists(f_x):
            os.remove(f_x)
            
    # 3. Create a displacement field with displacement only in numpy Component 2
    disp_z = np.zeros((16, 24, 32, 3), dtype=np.float32)
    disp_z[..., 2] = -5.0
    warp_img_z = ants.from_numpy(disp_z, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), has_components=True)
    
    f_z = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False).name
    try:
        ants.image_write(warp_img_z, f_z)
        warped_z = ants.apply_transforms(fixed=img, moving=img, transformlist=[f_z])
        warped_z_np = warped_z.numpy()
        peak_idx_z = np.unravel_index(np.argmax(warped_z_np), warped_z_np.shape)
        print(f"New peak with numpy Component 2: {peak_idx_z}")
    finally:
        if os.path.exists(f_z):
            os.remove(f_z)
            
    # Without manual component swap:
    # - numpy Component 0 maps to ITK component 2, which shifts along the Z/depth axis (axis 0 of numpy array)
    # - numpy Component 2 maps to ITK component 0, which shifts along the X/col axis (axis 2 of numpy array)
    assert peak_idx[0] == 13, f"Expected shift along Z (axis 0), got peak at {peak_idx}"
    assert peak_idx_z[2] == 21, f"Expected shift along X (axis 2), got peak at {peak_idx_z}"


def test_registration_versus_transform_export_3d(tmp_path):
    fixed_data = np.zeros((16, 24, 32), dtype=np.float32)
    moving_data = np.zeros((16, 24, 32), dtype=np.float32)
    zc, yc, xc = 8, 12, 16
    for z in range(16):
        for y in range(24):
            for x in range(32):
                dist_sq_fixed = (z - zc)**2 + (y - yc)**2 + (x - xc)**2
                if dist_sq_fixed <= 3**2:
                    fixed_data[z, y, x] = 100.0
                dist_sq_moving = (z - (zc + 1.5))**2 + (y - (yc + 1.0))**2 + (x - (xc - 1.0))**2
                if dist_sq_moving <= 5**2:
                    moving_data[z, y, x] = 100.0
    
    fixed = ants.from_numpy(fixed_data, spacing=(1.0, 1.5, 2.0), origin=(10, 20, 30))
    moving = ants.from_numpy(moving_data, spacing=(1.0, 1.5, 2.0), origin=(10, 20, 30))
    
    # Run SyNTo registration using PyTorch backend and capture results
    res = registration(
        fixed=fixed,
        moving=moving,
        type_of_transform='SyNTo',
        backend='pytorch',
        levels=[1],
        affine_iterations=[0],
        reg_iterations=[50],
        grad_step=1.5,
        flow_sigma=1.0
    )
    
    # Registration returns warpedmovout and list of fwdtransforms
    fwd_tx_list = res['fwdtransforms']
    fwd_warp_file = next((tx for tx in fwd_tx_list if tx.endswith('.nii.gz')), None)
    assert fwd_warp_file is not None
    
    # Load the warp file from registration
    warp_reg_img = ants.image_read(fwd_warp_file)
    warp_reg_np = warp_reg_img.numpy()
    
    # Manually recreate the same SyNToTransform.
    # To get the EXACT warp field that was used, we can reconstruct it from the registered file
    # by reversing the component swap.
    # syn.py output warp_reg_np has components in order [Z, Y, X].
    # So to get the unswapped warp_manual_np, we swap the components back:
    unswapped_warp_np = warp_reg_np[..., [2, 1, 0]]
    
    # Let's save this manually created unswapped warp to a file using ants
    warp_manual_img = ants.from_numpy(
        unswapped_warp_np,
        origin=fixed.origin,
        spacing=fixed.spacing,
        direction=fixed.direction,
        has_components=True
    )
    manual_warp_file = os.path.join(tmp_path, "manual_warp.nii.gz")
    ants.image_write(warp_manual_img, manual_warp_file)
    
    # Let's verify that warp_manual_img indeed corresponds to the unswapped field
    warp_manual_np = warp_manual_img.numpy()
    np.testing.assert_allclose(warp_reg_np[..., 0], warp_manual_np[..., 2], atol=1e-5)
    np.testing.assert_allclose(warp_reg_np[..., 1], warp_manual_np[..., 1], atol=1e-5)
    np.testing.assert_allclose(warp_reg_np[..., 2], warp_manual_np[..., 0], atol=1e-5)
    
    # Apply both warps using ANTs and compute similarity
    warped_reg_applied = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=fwd_tx_list)
    warped_manual_applied = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=[manual_warp_file] + fwd_tx_list[1:])
    
    mse_reg = np.mean((fixed.numpy() - warped_reg_applied.numpy())**2)
    mse_manual = np.mean((fixed.numpy() - warped_manual_applied.numpy())**2)
    mse_unregistered = np.mean((fixed.numpy() - moving.numpy())**2)
    
    print(f"MSE Unregistered: {mse_unregistered:.4f}")
    print(f"MSE with Registration Warp (swapped, syn.py): {mse_reg:.4f}")
    print(f"MSE with Manual Warp (unswapped, transform.py): {mse_manual:.4f}")
    
    # Assert that the swapped version is the one that successfully registers the images
    # (i.e. has much lower MSE).
    assert mse_reg < mse_unregistered, "Registration warp did not improve alignment!"
    assert mse_reg < mse_manual, "The unswapped warp performed better or identically to the swapped warp!"
