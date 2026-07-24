import pytest
import torch
import torch.nn.functional as F
import jax
import jax.numpy as jnp
import numpy as np
import ants

from syntx.syn import registration, grid_sample_bspline_torch, grid_sample_nd
from syntx.syn_jax import jax_grid_sample_bspline, jax_grid_sample

def test_bspline_grid_sample_parity_2d_and_3d():
    """Verify PyTorch and JAX 2D and 3D B-spline grid sample numerical parity."""
    np.random.seed(42)
    
    # 2D Test
    img_2d_np = np.random.randn(2, 3, 16, 16).astype(np.float32)
    grid_2d_np = (np.random.rand(2, 12, 12, 2) * 2.0 - 1.0).astype(np.float32)
    
    out_pt_2d = grid_sample_bspline_torch(torch.from_numpy(img_2d_np), torch.from_numpy(grid_2d_np)).numpy()
    out_jx_2d = np.array(jax_grid_sample_bspline(jnp.array(img_2d_np), jnp.array(grid_2d_np)))
    
    diff_2d = np.max(np.abs(out_pt_2d - out_jx_2d))
    assert diff_2d < 1e-4, f"2D B-spline parity failure: max diff = {diff_2d}"
    
    # 3D Test
    img_3d_np = np.random.randn(2, 2, 16, 16, 16).astype(np.float32)
    grid_3d_np = (np.random.rand(2, 10, 10, 10, 3) * 2.0 - 1.0).astype(np.float32)
    
    out_pt_3d = grid_sample_bspline_torch(torch.from_numpy(img_3d_np), torch.from_numpy(grid_3d_np)).numpy()
    out_jx_3d = np.array(jax_grid_sample_bspline(jnp.array(img_3d_np), jnp.array(grid_3d_np)))
    
    diff_3d = np.max(np.abs(out_pt_3d - out_jx_3d))
    assert diff_3d < 1e-4, f"3D B-spline parity failure: max diff = {diff_3d}"


def test_derivative_smoothing_bspline_vs_linear():
    """
    Measure and assert that B-Spline derivative interpolation significantly reduces 
    1st (||grad u||) and 2nd (||grad^2 u||) spatial derivative norms and fluctuations 
    compared to linear interpolation.
    """
    torch.manual_seed(42)
    img = torch.randn(1, 1, 32, 32, 32)
    
    # Grid with fractional coordinate offsets and smooth variation
    z_g = torch.linspace(-0.8, 0.8, 32)
    y_g = torch.linspace(-0.8, 0.8, 32)
    x_g = torch.linspace(-0.8, 0.8, 32)
    grid_mesh = torch.stack(torch.meshgrid(z_g, y_g, x_g, indexing='ij')[::-1], dim=-1).unsqueeze(0)
    grid_perturbed = grid_mesh + torch.randn(1, 32, 32, 32, 3) * 0.05
    
    linear_out = grid_sample_nd(img, grid_perturbed, mode='bilinear', interpolator='linear')
    bspline_out = grid_sample_nd(img, grid_perturbed, interpolator='bspline')
    
    def calc_derivatives(tensor):
        # 1st spatial derivatives
        gz = torch.gradient(tensor, dim=2)[0]
        gy = torch.gradient(tensor, dim=3)[0]
        gx = torch.gradient(tensor, dim=4)[0]
        grad_1st = torch.sqrt(gz**2 + gy**2 + gx**2 + 1e-10)
        
        # 2nd spatial derivatives
        gzz = torch.gradient(gz, dim=2)[0]
        gyy = torch.gradient(gy, dim=3)[0]
        gxx = torch.gradient(gx, dim=4)[0]
        grad_2nd = torch.sqrt(gzz**2 + gyy**2 + gxx**2 + 1e-10)
        
        return (
            grad_1st.mean().item(),
            torch.std(grad_1st).item(),
            grad_2nd.mean().item(),
            torch.std(grad_2nd).item()
        )
    
    lin_mean1, lin_std1, lin_mean2, lin_std2 = calc_derivatives(linear_out)
    bsp_mean1, bsp_std1, bsp_mean2, bsp_std2 = calc_derivatives(bspline_out)
    
    print(f"Linear:  1st mean={lin_mean1:.4f}, 1st std={lin_std1:.4f}, 2nd mean={lin_mean2:.4f}, 2nd std={lin_std2:.4f}")
    print(f"BSpline: 1st mean={bsp_mean1:.4f}, 1st std={bsp_std1:.4f}, 2nd mean={bsp_mean2:.4f}, 2nd std={bsp_std2:.4f}")
    
    # Assert B-spline significantly reduces derivative norms and fluctuations
    assert bsp_mean1 < lin_mean1, "B-spline 1st derivative norm should be smaller than linear"
    assert bsp_std1 < lin_std1, "B-spline 1st derivative std (fluctuation) should be smaller than linear"
    assert bsp_mean2 < lin_mean2, "B-spline 2nd derivative norm should be smaller than linear"
    assert bsp_std2 < lin_std2, "B-spline 2nd derivative std (fluctuation) should be smaller than linear"


def test_registration_bspline_pytorch_jax_dice_parity():
    """Verify PyTorch and JAX bspline registration Dice parity (<= 0.015 Dice gap)."""
    shape = (32, 32, 32)
    arr_f = np.zeros(shape, dtype=np.float32)
    arr_m = np.zeros(shape, dtype=np.float32)
    
    z, y, x = np.ogrid[:32, :32, :32]
    mask_f = (x - 16)**2 + (y - 16)**2 + (z - 16)**2 <= 9**2
    arr_f[mask_f] = 1.0
    
    mask_m = (x - 17)**2 + (y - 15)**2 + (z - 16)**2 <= 9**2
    arr_m[mask_m] = 1.0
    
    fixed = ants.from_numpy(arr_f)
    moving = ants.from_numpy(arr_m)
    
    res_pt = registration(
        fixed, moving, backend='pytorch', interpolator='bspline', 
        reg_iterations=[10, 5], affine_iterations=[10, 5], verbose=False
    )
    
    res_jx = registration(
        fixed, moving, backend='jax', interpolator='bspline', 
        reg_iterations=[10, 5], affine_iterations=[10, 5], verbose=False
    )
    
    fixed_mask = ants.from_numpy((arr_f > 0.5).astype(np.uint8))
    warped_pt_mask = ants.from_numpy((res_pt['warpedmovout'].numpy() > 0.5).astype(np.uint8))
    warped_jx_mask = ants.from_numpy((res_jx['warpedmovout'].numpy() > 0.5).astype(np.uint8))
    
    overlap_pt = ants.label_overlap_measures(fixed_mask, warped_pt_mask)
    overlap_jx = ants.label_overlap_measures(fixed_mask, warped_jx_mask)
    
    dice_pt = float(overlap_pt.loc[overlap_pt['Label'] == 1, 'MeanOverlap'].values[0])
    dice_jx = float(overlap_jx.loc[overlap_jx['Label'] == 1, 'MeanOverlap'].values[0])
    
    dice_gap = abs(dice_pt - dice_jx)
    print(f"PyTorch Dice: {dice_pt:.4f}, JAX Dice: {dice_jx:.4f}, Dice Gap: {dice_gap:.4f}")
    assert dice_gap <= 0.015, f"PyTorch and JAX Dice gap ({dice_gap:.4f}) exceeds threshold 0.015"


def test_registration_interpolator_parameter_propagation():
    """Verify interpolator='linear' vs interpolator='bspline' propagation in registration."""
    shape = (24, 24, 24)
    arr_f = np.random.randn(*shape).astype(np.float32)
    arr_m = np.random.randn(*shape).astype(np.float32)
    
    fixed = ants.from_numpy(arr_f)
    moving = ants.from_numpy(arr_m)
    
    # Test linear
    res_lin_pt = registration(fixed, moving, backend='pytorch', interpolator='linear', reg_iterations=[2, 2], affine_iterations=[2, 2])
    res_lin_jx = registration(fixed, moving, backend='jax', interpolator='linear', reg_iterations=[2, 2], affine_iterations=[2, 2])
    
    # Test bspline
    res_bsp_pt = registration(fixed, moving, backend='pytorch', interpolator='bspline', reg_iterations=[2, 2], affine_iterations=[2, 2])
    res_bsp_jx = registration(fixed, moving, backend='jax', interpolator='bspline', reg_iterations=[2, 2], affine_iterations=[2, 2])
    
    assert res_lin_pt['warpedmovout'].shape == fixed.shape
    assert res_lin_jx['warpedmovout'].shape == fixed.shape
    assert res_bsp_pt['warpedmovout'].shape == fixed.shape
    assert res_bsp_jx['warpedmovout'].shape == fixed.shape


def test_nearest_neighbor_label_preservation():
    """
    Verify that interpolator='nearestNeighbor' and interpolator='nearest' in both PyTorch 
    and JAX strictly preserve discrete integer label values without non-integer spatial blurring.
    """
    label_values = np.array([0.0, 1.0, 2.0, 5.0, 10.0], dtype=np.float32)
    shape = (1, 1, 16, 16, 16)
    
    np.random.seed(42)
    label_data = np.random.choice(label_values, size=shape).astype(np.float32)
    
    z_g = np.linspace(-0.85, 0.85, 16, dtype=np.float32)
    y_g = np.linspace(-0.85, 0.85, 16, dtype=np.float32)
    x_g = np.linspace(-0.85, 0.85, 16, dtype=np.float32)
    mesh = np.stack(np.meshgrid(z_g, y_g, x_g, indexing='ij')[::-1], axis=-1)[None, ...]
    grid_data = mesh + (np.random.rand(1, 16, 16, 16, 3).astype(np.float32) * 0.1 - 0.05)
    
    pt_label = torch.from_numpy(label_data)
    pt_grid = torch.from_numpy(grid_data)
    jx_label = jnp.array(label_data)
    jx_grid = jnp.array(grid_data)
    
    # 1. Linear interpolation produces non-integer values
    lin_pt = grid_sample_nd(pt_label, pt_grid, mode='bilinear', interpolator='linear').numpy()
    lin_non_int = np.any(np.mod(lin_pt, 1.0) != 0.0)
    assert lin_non_int, "Bilinear interpolation expected to produce non-integer blurred values"
    
    # 2. PyTorch nearest neighbor tests
    for nn_str in ('nearestNeighbor', 'nearest', 'nearest_neighbor', 'NearestNeighbor'):
        out_pt_interp = grid_sample_nd(pt_label, pt_grid, interpolator=nn_str).numpy()
        unique_pt_interp = np.unique(out_pt_interp)
        assert np.isin(unique_pt_interp, label_values).all(), f"PyTorch with interpolator='{nn_str}' produced non-label values: {unique_pt_interp}"
        assert np.all(np.mod(out_pt_interp, 1.0) == 0.0), f"PyTorch with interpolator='{nn_str}' produced non-integer values"

        out_pt_mode = grid_sample_nd(pt_label, pt_grid, mode=nn_str).numpy()
        unique_pt_mode = np.unique(out_pt_mode)
        assert np.isin(unique_pt_mode, label_values).all(), f"PyTorch with mode='{nn_str}' produced non-label values: {unique_pt_mode}"
        assert np.all(np.mod(out_pt_mode, 1.0) == 0.0), f"PyTorch with mode='{nn_str}' produced non-integer values"

    # 3. JAX nearest neighbor tests
    for nn_str in ('nearestNeighbor', 'nearest', 'nearest_neighbor', 'NearestNeighbor'):
        out_jx_interp = np.array(jax_grid_sample(jx_label, jx_grid, interpolator=nn_str))
        unique_jx_interp = np.unique(out_jx_interp)
        assert np.isin(unique_jx_interp, label_values).all(), f"JAX with interpolator='{nn_str}' produced non-label values: {unique_jx_interp}"
        assert np.all(np.mod(out_jx_interp, 1.0) == 0.0), f"JAX with interpolator='{nn_str}' produced non-integer values"

        out_jx_mode = np.array(jax_grid_sample(jx_label, jx_grid, mode=nn_str))
        unique_jx_mode = np.unique(out_jx_mode)
        assert np.isin(unique_jx_mode, label_values).all(), f"JAX with mode='{nn_str}' produced non-label values: {unique_jx_mode}"
        assert np.all(np.mod(out_jx_mode, 1.0) == 0.0), f"JAX with mode='{nn_str}' produced non-integer values"

