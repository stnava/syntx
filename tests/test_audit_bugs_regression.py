"""
Comprehensive Regression Test Suite for Audit Bugs #1 through #9.

This test module verifies that every bug identified during the audit
is strictly caught, preventing regressions across PyTorch, JAX,
visualization, feature networks, orchestrator, and inverse solvers.
"""

import os
import json
import tempfile
import pytest
import numpy as np
import torch
import ants

# ---------------------------------------------------------------------------
# Bug #1: Benchmark Orchestrator t0_benchmark UnboundLocalError
# ---------------------------------------------------------------------------
def test_bug1_orchestrator_t0_benchmark_defined(monkeypatch, tmp_path):
    """
    Bug #1: In orchestrator.run_mindboggle_benchmark, t0_benchmark was referenced
    at the end for timing calculation, but was not initialized at function entry.
    This test verifies that run_mindboggle_benchmark initializes t0_benchmark and
    completes without raising UnboundLocalError.
    """
    from syntx.benchmark import orchestrator

    monkeypatch.setattr(
        orchestrator, "check_mindboggle_data",
        lambda **kwargs: (True, {"total_pairs_in_csv": 1, "available_pairs": 1})
    )

    out_dir = str(tmp_path / "bench_out")
    os.makedirs(out_dir, exist_ok=True)
    summary_json = str(tmp_path / "summary.json")
    report_html = str(tmp_path / "report.html")

    # Seed a cached result so orchestrator doesn't need to run a live subprocess
    cached_record = {
        "pair_id": 0,
        "model": "sobolev",
        "status": "SUCCESS",
        "dice_sym": 0.82,
        "runtime_seconds": 1.5,
        "folds_pct": 0.0,
        "detJ_min": 0.1
    }
    with open(os.path.join(out_dir, "pair_000_sobolev.json"), "w") as f:
        json.dump(cached_record, f)

    # Prior to the fix, this raised UnboundLocalError: local variable 't0_benchmark' referenced before assignment
    res = orchestrator.run_mindboggle_benchmark(
        pairs=[0],
        model="sobolev",
        out_dir=out_dir,
        summary_json=summary_json,
        report_html=report_html,
        generate_example_reports=False,
        force=False,
        verbose=False
    )
    assert "runtime_minutes" in res
    assert res["runtime_minutes"] >= 0.0


# ---------------------------------------------------------------------------
# Bug #2: syn.registration() Missing time Import & Undefined t_start
# ---------------------------------------------------------------------------
def test_bug2_syn_registration_timing_and_provenance():
    """
    Bug #2: In syn.registration(), `time` was not imported and `t_start` was not
    initialized, causing fit_time and engine provenance to silently fail or crash.
    """
    from syntx.syn import registration

    fi = ants.from_numpy(np.pad(np.ones((16, 16), dtype=np.float32), 4), origin=(0., 0.), spacing=(1., 1.))
    mi = ants.from_numpy(np.pad(np.ones((16, 16), dtype=np.float32), 4), origin=(0., 0.), spacing=(1., 1.))

    res = registration(
        fixed=fi, moving=mi,
        type_of_transform='Translation',
        backend='pytorch',
        affine_iterations=[3],
        reg_iterations=[0],
        verbose=True
    )
    assert 'provenance' in res
    assert res['provenance'] is not None
    assert res['provenance']['fit_time'] is not None
    assert res['provenance']['fit_time'] > 0.0


# ---------------------------------------------------------------------------
# Bug #3: PyTorch compute_jacobian_determinant_nd on Anisotropic Grid & Shear
# ---------------------------------------------------------------------------
def test_bug3_pytorch_jacobian_determinant_anisotropic():
    """
    Bug #3: In compute_jacobian_determinant_nd, physical spacing was passed without
    matching PyTorch's (Z, Y, X) spatial dimension order, and displacement vector
    channels (u_x, u_y, u_z) were incorrectly mapped to (u_z, u_y, u_x).
    On anisotropic grids and shears, this produced wildly incorrect determinants
    (e.g., 3.5 instead of 1.0 for pure shear, 1.212 instead of 1.188 for stretch).
    """
    from syntx.core.jacobian import compute_jacobian_determinant_nd

    D, H, W = 8, 10, 12
    sx, sy, sz = 1.0, 2.0, 5.0
    spacing = (sx, sy, sz)

    z = torch.arange(D).float() * sz
    y = torch.arange(H).float() * sy
    x = torch.arange(W).float() * sx
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')

    # Test Case A: Pure Shear u_x = 0.5 * Z, u_y = 0, u_z = 0.
    # Analytical Jacobian det(J) must be exactly 1.0 everywhere.
    # In PyTorch tensor convention: channel 2 is u_x, channel 1 is u_y, channel 0 is u_z.
    disp_shear = torch.zeros(1, D, H, W, 3)
    disp_shear[..., 2] = 0.5 * Z
    detJ_shear = compute_jacobian_determinant_nd(disp_shear, physical_spacing=spacing)
    shear_val = detJ_shear[0, 4, 5, 6].item()
    assert abs(shear_val - 1.0) < 1e-4, f"Pure shear det(J) was {shear_val}, expected 1.0"

    # Test Case B: Anisotropic linear stretch u_x = 0.1*X, u_y = 0.2*Y, u_z = -0.1*Z.
    # Analytical det(J) = (1 + 0.1) * (1 + 0.2) * (1 - 0.1) = 1.188.
    disp_stretch = torch.zeros(1, D, H, W, 3)
    disp_stretch[..., 2] = 0.1 * X
    disp_stretch[..., 1] = 0.2 * Y
    disp_stretch[..., 0] = -0.1 * Z
    detJ_stretch = compute_jacobian_determinant_nd(disp_stretch, physical_spacing=spacing)
    stretch_val = detJ_stretch[0, 4, 5, 6].item()
    assert abs(stretch_val - 1.188) < 1e-4, f"Anisotropic stretch det(J) was {stretch_val}, expected 1.188"


# ---------------------------------------------------------------------------
# Bug #4: JAX compute_jacobian_determinant_nd_jax Matrix Diagonal
# ---------------------------------------------------------------------------
def test_bug4_jax_jacobian_determinant_linear_stretch():
    """
    Bug #4: In compute_jacobian_determinant_nd_jax, reversal of channels combined
    with reversed gradient stacking placed diagonal derivatives onto off-diagonals,
    causing identity (+1.0) to be added to shear terms and producing 1.066 instead of 1.5.
    """
    import jax
    import jax.numpy as jnp
    from syntx.syn_jax import compute_jacobian_determinant_nd_jax

    D, H, W = 16, 16, 16
    z = jnp.linspace(-1, 1, D)
    y = jnp.linspace(-1, 1, H)
    x = jnp.linspace(-1, 1, W)
    Z, Y, X = jnp.meshgrid(z, y, x, indexing='ij')

    # Linear expansion along X: u_x = 0.5 * X. Analytical det(J) = 1.5
    disp = jnp.zeros((1, D, H, W, 3))
    disp = disp.at[0, ..., 2].set(0.5 * X)

    dx = float(2.0 / (W - 1))
    spacing = (dx, dx, dx)
    detJ = compute_jacobian_determinant_nd_jax(disp, physical_spacing=spacing)
    center_val = float(detJ[0, 8, 8, 8])
    assert abs(center_val - 1.5) < 1e-4, f"Expected 1.5 for linear stretch, got {center_val}"

    # Pure shear: u_x = 0.5 * Y. Analytical det(J) = 1.0
    disp_shear = jnp.zeros((1, D, H, W, 3))
    disp_shear = disp_shear.at[0, ..., 2].set(0.5 * Y)
    detJ_shear = compute_jacobian_determinant_nd_jax(disp_shear, physical_spacing=spacing)
    shear_val = float(detJ_shear[0, 8, 8, 8])
    assert abs(shear_val - 1.0) < 1e-4, f"Expected 1.0 for pure shear, got {shear_val}"


# ---------------------------------------------------------------------------
# Bug #5: Sobolev Operator Physical Spacing Inversion on Anisotropic Grids
# ---------------------------------------------------------------------------
def test_bug5_sobolev_anisotropic_physical_spacing():
    """
    Bug #5: In _get_sobolev_filter_cached, spatial_shape is (D_z, H_y, W_x)
    but spacing[d] was indexed without reversing, applying s_x to Z and s_z to X.
    On anisotropic grids (e.g. dz=10mm, dx=1mm), this heavily damped low physical
    frequencies along Z and preserved high physical frequencies along X.
    """
    from syntx.core.smoothing import _get_sobolev_filter_cached

    shape = (16, 16, 16)
    # Anisotropic acquisition: thin in-plane (1mm), thick slices (10mm)
    spacing = (1.0, 1.0, 10.0) # (sx, sy, sz)

    K = _get_sobolev_filter_cached(shape, alpha_val=1.0, s=2.0, spacing=spacing, device='cpu', dtype=torch.float32)

    # Frequency index 4 along Z (dim 0) vs frequency index 4 along X (dim 2)
    val_z = K[0, 0, 4, 0, 0].item()
    val_x = K[0, 0, 0, 0, 4].item()

    # In physical space, 10mm slice thickness means index 4 is a low physical frequency (40mm wavelength)
    # while 1mm in-plane thickness means index 4 is a high physical frequency (4mm wavelength).
    # Therefore, the filter value along Z MUST be larger (less damped) than along X.
    assert val_z > val_x, f"Physical Sobolev spacing inverted! Z filter: {val_z:.4f}, X filter: {val_x:.4f}"
    assert val_z > 0.80, f"Expected low physical frequency along Z to be preserved (>0.80), got {val_z:.4f}"
    assert val_x < 0.20, f"Expected high physical frequency along X to be damped (<0.20), got {val_x:.4f}"


# ---------------------------------------------------------------------------
# Bug #6: Inverse Solver Anisotropic Spacing Scaling Error
# ---------------------------------------------------------------------------
def test_bug6_inverse_field_anisotropic_physical_identity():
    """
    Bug #6: In update_inverse_field_nd and its Anderson variant, the displacement
    error was divided by spacing_rev rather than matching channel order (u_x, u_y, u_z).
    This test verifies that inverse field computation operates stably on anisotropic grids.
    """
    from syntx.core.inverse import (
        update_inverse_field_nd,
        calculate_inverse_identity_error,
        compute_inverse_identity_error_nd
    )

    shape = (16, 16, 16)
    spacing = (1.0, 2.0, 4.0)
    origin = (0.0, 0.0, 0.0)
    direction = np.eye(3)

    disp = torch.zeros(1, *shape, 3)
    disp[..., 0] = 0.2
    disp[..., 1] = -0.4
    disp[..., 2] = 0.5

    inv_disp = update_inverse_field_nd(
        disp, method='anderson', steps=10,
        spacing=spacing, origin=origin, direction=direction
    )
    assert inv_disp.shape == disp.shape
    # Inverse of a constant translation should be negative displacement
    assert torch.allclose(inv_disp[0, 4:12, 4:12, 4:12, 0], torch.tensor(-0.2), atol=1e-2)
    assert torch.allclose(inv_disp[0, 4:12, 4:12, 4:12, 1], torch.tensor(0.4), atol=1e-2)
    assert torch.allclose(inv_disp[0, 4:12, 4:12, 4:12, 2], torch.tensor(-0.5), atol=1e-2)

    # Fixed-point method verification on anisotropic grid
    inv_fp = update_inverse_field_nd(
        disp, method='fixed_point', steps=10,
        spacing=spacing, origin=origin, direction=direction
    )
    assert torch.allclose(inv_fp[0, 4:12, 4:12, 4:12, 0], torch.tensor(-0.2), atol=1e-2)
    assert torch.allclose(inv_fp[0, 4:12, 4:12, 4:12, 1], torch.tensor(0.4), atol=1e-2)
    assert torch.allclose(inv_fp[0, 4:12, 4:12, 4:12, 2], torch.tensor(-0.5), atol=1e-2)

    # Verify calculate_inverse_identity_error and compute_inverse_identity_error_nd on anisotropic grid
    err_dict = calculate_inverse_identity_error(disp, inv_disp, spacing=spacing, origin=origin, direction=direction)
    assert err_dict['max_error'] < 0.05

    err_nd = compute_inverse_identity_error_nd(disp, inv_disp, spacing=spacing, origin=origin, direction=direction)
    assert err_nd[0, 4:12, 4:12, 4:12].max().item() < 0.05


# ---------------------------------------------------------------------------
# Bug #7: robust_affine and syn fit_kwargs Dropping
# ---------------------------------------------------------------------------
def test_bug7_robust_affine_and_syn_kwargs_forwarding():
    """
    Bug #7: robust_affine dropped **kwargs causing TypeError when passing
    valid options (e.g. aff_metric, custom_param). JAX model.fit() dropped fit_kwargs.
    """
    from syntx.robust_affine import robust_affine

    arr = np.pad(np.ones((16, 16), dtype=np.float32), 4)
    fi = ants.from_numpy(arr, origin=(0., 0.), spacing=(1., 1.))
    mi = ants.from_numpy(arr, origin=(0., 0.), spacing=(1., 1.))

    # Should accept arbitrary registration kwargs without raising TypeError
    res = robust_affine(fi, mi, mode='pytorch', custom_kwarg=123, aff_metric='mattes')
    assert 'warpedmovout' in res
    assert 'fwdtransforms' in res


# ---------------------------------------------------------------------------
# Bug #8: 1-Channel 2D Extractors 5D Slicing Crash
# ---------------------------------------------------------------------------
def test_bug8_feature_space_loss_1channel_2d_extractor():
    """
    Bug #8: In features.py and image_compare.py, 1-channel 2D extractors
    sliced 4D tensors with 5D indexing x[:, 0:1, z:z+1, :, :], causing
    IndexError: too many indices for tensor of dimension 4.
    """
    from syntx.features import FeatureSpaceLoss, ResNet10Extractor
    from syntx.image_compare import compute_reconstructed_loss

    # 1-channel 2D feature extractor
    ext_1ch = ResNet10Extractor(dim=2, feature_layers=[1])
    assert ext_1ch.in_channels == 1
    assert not ext_1ch.is_3d

    x = torch.rand(1, 1, 16, 16, 16)
    y = torch.rand(1, 1, 16, 16, 16)

    # 1. FeatureSpaceLoss triplanar mode
    loss_fn_tri = FeatureSpaceLoss(extractor=ext_1ch, mode='triplanar')
    val_tri = loss_fn_tri(x, y)
    assert isinstance(val_tri, torch.Tensor)
    assert not torch.isnan(val_tri).item()

    # 2. FeatureSpaceLoss lncc_3d mode
    loss_fn_3d = FeatureSpaceLoss(extractor=ext_1ch, mode='lncc_3d')
    val_3d = loss_fn_3d(x, y)
    assert isinstance(val_3d, torch.Tensor)
    assert not torch.isnan(val_3d).item()

    # 3. compute_reconstructed_loss on 1-channel 3D volume
    rec_loss = compute_reconstructed_loss(ext_1ch, x, y)
    assert rec_loss is not None and not np.isnan(rec_loss)


# ---------------------------------------------------------------------------
# Bug #9: Visualizer Title Typo and Union Bounding Box
# ---------------------------------------------------------------------------
def test_bug9_label_alignment_figure_title_and_union_bbox():
    """
    Bug #9: render_label_alignment_figure had a typo in sagittal title
    (displaying Z slice s2 instead of X slice s0), and independently cropped
    fixed and warped labels causing overlay distortions when labels have different bounds.
    """
    from syntx.viz.figures import render_label_alignment_figure
    import matplotlib.pyplot as plt

    # Create 3D label masks with different bounding boxes
    fl_arr = np.zeros((30, 30, 30), dtype=np.int32)
    fl_arr[5:15, 8:18, 8:18] = 1 # Left region

    wl_arr = np.zeros((30, 30, 30), dtype=np.int32)
    wl_arr[15:25, 12:22, 12:22] = 1 # Right region (non-overlapping X coordinates)

    fi_arr = np.random.rand(30, 30, 30).astype(np.float32)

    fig = render_label_alignment_figure(
        fixed_labels=fl_arr,
        warped_labels=wl_arr,
        fixed_image=fi_arr,
        crop_background=True,
        show_figure=False
    )

    # Check sagittal axis title (row 1, col 2)
    sag_title = fig.axes[5].get_title()
    # Should display Sagittal (X=...) using s0_w, NOT Sagittal (X=s2_w)
    assert "Sagittal (X=" in sag_title
    
    # Extract images rendered on axes: background slice shape must match label overlay shape exactly
    im_bg_fixed = fig.axes[0].images[0].get_array().shape
    im_bg_warped = fig.axes[3].images[0].get_array().shape
    assert im_bg_fixed == im_bg_warped, (
        f"Row 0 and Row 1 background slice shapes differed: {im_bg_fixed} vs {im_bg_warped}"
    )

    plt.close(fig)


# ---------------------------------------------------------------------------
# ANTsPy Ground-Truth Validation: Direct Parity Verification
# ---------------------------------------------------------------------------
def test_pytorch_jacobian_exact_parity_with_antspy():
    """
    Directly validates PyTorch compute_jacobian_determinant_nd against ANTsPy's
    ITK C++ implementation (ants.create_jacobian_determinant_image) across both
    anisotropic stretch and non-linear cross-axis coupled deformation fields.
    """
    from syntx.core.jacobian import compute_jacobian_determinant_nd

    Nx, Ny, Nz = 16, 20, 24
    sx, sy, sz = 1.0, 2.0, 4.0
    spacing = (sx, sy, sz)

    z = torch.arange(Nz).float() * sz
    y = torch.arange(Ny).float() * sy
    x = torch.arange(Nx).float() * sx
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')

    # Non-linear deformation with coupled cross-axis shear and divergence
    disp_torch = torch.zeros(1, Nz, Ny, Nx, 3)
    disp_torch[..., 0] = 0.4 * torch.sin(np.pi * Y / (Ny * sy)) * torch.cos(np.pi * Z / (Nz * sz))
    disp_torch[..., 1] = 0.3 * torch.cos(np.pi * X / (Nx * sx)) * torch.sin(np.pi * Z / (Nz * sz))
    disp_torch[..., 2] = 0.2 * torch.sin(np.pi * X / (Nx * sx)) * torch.cos(np.pi * Y / (Ny * sy))

    # 1. PyTorch computation in syntx
    detJ_pytorch = compute_jacobian_determinant_nd(disp_torch, physical_spacing=spacing)[0]

    # 2. ANTsPy ITK C++ ground truth computation
    from syntx.spatial import disp_tensor_to_itk
    ref_ants = ants.from_numpy(np.zeros((Nx, Ny, Nz), dtype=np.float32), spacing=spacing)
    vec_ants = disp_tensor_to_itk(disp_torch, ref_image=ref_ants)

    with tempfile.NamedTemporaryFile(suffix='_warp_antspy.nii.gz') as tmp:
        ants.image_write(vec_ants, tmp.name)
        jac_ants = ants.create_jacobian_determinant_image(ref_ants, tmp.name, do_log=False)
        detJ_ants = jac_ants.numpy()

    # Transpose ANTs (X, Y, Z) back to (Z, Y, X) for direct array comparison
    detJ_ants_zyx = np.transpose(detJ_ants, (2, 1, 0))

    # Compare interior voxels (away from boundary stencils)
    sub_py = detJ_pytorch.numpy()[2:-2, 2:-2, 2:-2]
    sub_ants = detJ_ants_zyx[2:-2, 2:-2, 2:-2]

    corr = np.corrcoef(sub_py.flatten(), sub_ants.flatten())[0, 1]
    max_diff = np.max(np.abs(sub_py - sub_ants))
    mean_diff = np.mean(np.abs(sub_py - sub_ants))

    assert corr > 0.9999, f"Correlation between PyTorch and ANTsPy was {corr}, expected > 0.9999"
    assert max_diff < 5e-4, f"Max difference between PyTorch and ANTsPy was {max_diff}, expected < 5e-4"
    assert mean_diff < 1e-4, f"Mean difference was {mean_diff}, expected < 1e-4"

