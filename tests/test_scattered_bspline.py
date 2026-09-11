"""Unit and integration tests for ANTsTorch B-spline scattered data facilities in syntx.scattered.

Tests cover:
- Differentiable multi-level B-spline projection in 2D and 3D
- Autograd gradient flow into both coordinates and feature channels
- BSplineScatteredProjector module caching and evaluation
- Closed-form C^2 continuous landmark warp fitting and coordinate warping accuracy
- B-spline fluid velocity regularization (BSplineSyN operator)
- End-to-end SyN registration with B-spline projection, regularization, and landmark warm-start
- Full backward compatibility with existing Gaussian projection and SyN workflows
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np

from syntx.scattered import (
    has_antstorch,
    project_scattered_to_grid,
    ScatteredProjector,
    BSplineScatteredProjector,
    fit_bspline_landmark_warp,
    apply_bspline_fluid_regularizer,
    bspline_syn_scattered,
    SyNScattered,
    ScatteredRegistrationConfig,
    warp_scattered_coordinates,
    pushforward_scattered_to_grid,
)


@pytest.mark.skipif(not has_antstorch(), reason="ANTsTorch is required for B-spline scattered tests")
def test_has_antstorch():
    assert has_antstorch() is True


@pytest.mark.skipif(not has_antstorch(), reason="ANTsTorch is required for B-spline scattered tests")
def test_project_scattered_bspline_2d():
    torch.manual_seed(42)
    N = 100
    pts = (torch.rand(N, 2) * 1.6) - 0.8  # in [-0.8, 0.8]
    vals = torch.sin(pts[:, 0:1] * 3.0) * torch.cos(pts[:, 1:2] * 3.0)

    # 1. Project single channel
    grid, density = project_scattered_to_grid(
        points=pts,
        values=vals,
        grid_shape=(32, 32),
        domain_bounds=(-1.0, 1.0),
        method='bspline',
        number_of_fitting_levels=3,
        mesh_size=2,
        return_density=True,
    )
    assert grid.shape == (1, 1, 32, 32)
    assert density.shape == (1, 1, 32, 32)
    assert torch.isfinite(grid).all()
    assert torch.isfinite(density).all()

    # 2. Project multi-channel
    vals_mc = torch.randn(N, 4)
    grid_mc = project_scattered_to_grid(
        points=pts,
        values=vals_mc,
        grid_shape=(32, 32),
        domain_bounds=(-1.0, 1.0),
        method='bspline',
        number_of_fitting_levels=3,
        mesh_size=2,
    )
    assert grid_mc.shape == (1, 4, 32, 32)
    assert torch.isfinite(grid_mc).all()


@pytest.mark.skipif(not has_antstorch(), reason="ANTsTorch is required for B-spline scattered tests")
def test_project_scattered_bspline_3d():
    torch.manual_seed(42)
    N = 150
    pts = (torch.rand(N, 3) * 1.6) - 0.8
    vals = torch.sum(pts**2, dim=-1, keepdim=True)

    grid = project_scattered_to_grid(
        points=pts,
        values=vals,
        grid_shape=(24, 24, 24),
        domain_bounds=(-1.0, 1.0),
        method='bspline',
        number_of_fitting_levels=2,
        mesh_size=1,
    )
    assert grid.shape == (1, 1, 24, 24, 24)
    assert torch.isfinite(grid).all()


@pytest.mark.skipif(not has_antstorch(), reason="ANTsTorch is required for B-spline scattered tests")
def test_project_scattered_bspline_autograd():
    torch.manual_seed(42)
    N = 40
    pts = (torch.rand(N, 2) * 0.8).detach().requires_grad_(True)
    vals = torch.randn(N, 2, requires_grad=True)

    grid = project_scattered_to_grid(
        points=pts,
        values=vals,
        grid_shape=(20, 20),
        domain_bounds=(0.0, 1.0),
        method='bspline',
        number_of_fitting_levels=2,
        mesh_size=1,
    )
    loss = (grid**2).sum()
    loss.backward()

    assert pts.grad is not None
    assert vals.grad is not None
    assert torch.isfinite(pts.grad).all()
    assert torch.isfinite(vals.grad).all()
    assert (pts.grad.abs() > 0).any()
    assert (vals.grad.abs() > 0).any()


@pytest.mark.skipif(not has_antstorch(), reason="ANTsTorch is required for B-spline scattered tests")
def test_bspline_scattered_projector_module():
    torch.manual_seed(42)
    projector = BSplineScatteredProjector(
        grid_shape=(28, 28),
        domain_bounds=(-1.0, 1.0),
        number_of_fitting_levels=2,
        mesh_size=1,
    )
    pts = (torch.rand(50, 2) * 1.6) - 0.8
    vals = torch.randn(50, 1)

    out = projector(pts, vals)
    assert out.shape == (1, 1, 28, 28)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not has_antstorch(), reason="ANTsTorch is required for B-spline scattered tests")
def test_fit_bspline_landmark_warp_2d():
    torch.manual_seed(42)
    domain_bounds = ((-1.0, -1.0), (1.0, 1.0))
    fixed_lm = torch.tensor([
        [-0.5, -0.5],
        [0.5, -0.5],
        [0.0, 0.5],
        [0.0, 0.0],
    ], dtype=torch.float32)

    # Apply small shifts
    shifts = torch.tensor([
        [0.08, -0.04],
        [-0.06, 0.05],
        [0.02, 0.07],
        [0.00, -0.03],
    ], dtype=torch.float32)
    moving_lm = fixed_lm + shifts

    warp = fit_bspline_landmark_warp(
        fixed_landmarks=fixed_lm,
        moving_landmarks=moving_lm,
        grid_shape=(32, 32),
        domain_bounds=domain_bounds,
        number_of_fitting_levels=4,
        mesh_size=2,
        enforce_stationary_boundary=False,
    )
    assert warp.shape == (1, 32, 32, 2)
    assert torch.isfinite(warp).all()

    # Verify warp maps fixed landmarks to moving landmarks
    warped_lm = warp_scattered_coordinates(
        coords=fixed_lm,
        displacement_field=warp,
        domain_bounds=domain_bounds,
        is_physical=True,
        vector_convention='xyz',
    )
    err = (warped_lm - moving_lm).abs().max().item()
    assert err < 1e-2, f"Landmark warp error too high: {err}"


@pytest.mark.skipif(not has_antstorch(), reason="ANTsTorch is required for B-spline scattered tests")
def test_fit_bspline_landmark_warp_3d():
    torch.manual_seed(42)
    domain_bounds = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    fixed_lm = torch.tensor([
        [-0.4, -0.4, 0.0],
        [0.4, 0.4, -0.3],
        [0.0, 0.2, 0.5],
    ], dtype=torch.float32)
    moving_lm = fixed_lm + torch.tensor([
        [0.05, -0.05, 0.02],
        [-0.04, 0.03, -0.02],
        [0.01, 0.02, 0.04],
    ], dtype=torch.float32)

    warp = fit_bspline_landmark_warp(
        fixed_landmarks=fixed_lm,
        moving_landmarks=moving_lm,
        grid_shape=(24, 24, 24),
        domain_bounds=domain_bounds,
        number_of_fitting_levels=3,
        mesh_size=1,
        enforce_stationary_boundary=False,
    )
    assert warp.shape == (1, 24, 24, 24, 3)
    assert torch.isfinite(warp).all()


@pytest.mark.skipif(not has_antstorch(), reason="ANTsTorch is required for B-spline scattered tests")
def test_apply_bspline_fluid_regularizer():
    torch.manual_seed(42)
    # Create a noisy velocity field
    v = torch.randn(1, 32, 32, 2)
    bounds = (torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]))

    v_smoothed = apply_bspline_fluid_regularizer(
        velocity_field=v,
        grid_shape=(32, 32),
        domain_bounds=bounds,
        mesh_size=2,
    )
    assert v_smoothed.shape == v.shape
    assert torch.isfinite(v_smoothed).all()

    # Smoothing should dramatically reduce total gradient energy / variance
    orig_var = torch.var(v)
    sm_var = torch.var(v_smoothed)
    assert sm_var < orig_var * 0.5


@pytest.mark.skipif(not has_antstorch(), reason="ANTsTorch is required for B-spline scattered tests")
def test_pushforward_scattered_bspline():
    torch.manual_seed(42)
    pts = (torch.rand(60, 2) * 1.6) - 0.8
    fts = torch.randn(60, 1)

    grid = pushforward_scattered_to_grid(
        coords=pts,
        features=fts,
        grid_shape=(32, 32),
        domain_bounds=(-1.0, 1.0),
        method='bspline',
        number_of_fitting_levels=2,
        mesh_size=1,
    )
    assert grid.shape == (1, 1, 32, 32)
    assert torch.isfinite(grid).all()


@pytest.mark.skipif(not has_antstorch(), reason="ANTsTorch is required for B-spline scattered tests")
def test_bspline_syn_scattered_registration():
    torch.manual_seed(42)
    # Generate synthetic source and target point clouds
    N = 80
    theta = torch.linspace(0, 2 * np.pi, N)[:-1]
    r_target = 0.5 + 0.1 * torch.sin(3 * theta)
    pts_f = torch.stack([r_target * torch.cos(theta), r_target * torch.sin(theta)], dim=-1)

    # Source has an additional expansion and rotation
    r_source = 0.55 + 0.1 * torch.sin(3 * theta + 0.3)
    pts_m = torch.stack([r_source * torch.cos(theta + 0.1), r_source * torch.sin(theta + 0.1)], dim=-1)

    # Run SyN with B-spline projection and B-spline regularizer
    res = bspline_syn_scattered(
        fixed_points=pts_f,
        moving_points=pts_m,
        grid_res=32,
        domain_bounds=(-1.0, 1.0),
        projection_method='bspline',
        regularizer='bspline',
        number_of_fitting_levels=2,
        mesh_size=1,
        iterations=15,
        fluid_sigma=1.5,
    )

    assert 'warp_l2r' in res
    assert 'warp_r2l' in res
    assert 'warped_moving_points' in res
    assert res['warp_l2r'].shape == (1, 32, 32, 2)
    assert res['warp_r2l'].shape == (1, 32, 32, 2)
    assert torch.isfinite(res['warp_l2r']).all()
    assert torch.isfinite(res['warp_r2l']).all()
    assert len(res['loss_history']) > 0


@pytest.mark.skipif(not has_antstorch(), reason="ANTsTorch is required for B-spline scattered tests")
def test_landmark_initialized_syn():
    torch.manual_seed(42)
    pts_f = torch.tensor([[-0.5, -0.5], [0.5, -0.5], [0.0, 0.5]], dtype=torch.float32)
    pts_m = torch.tensor([[-0.45, -0.48], [0.48, -0.52], [0.02, 0.54]], dtype=torch.float32)

    config = ScatteredRegistrationConfig(
        dim=2,
        grid_res=32,
        domain_bounds=(-1.0, 1.0),
        landmark_init=True,
        initial_landmarks=(pts_f, pts_m),
        iterations=5,
    )
    model = SyNScattered(config)
    model.fit(fixed_points=pts_f, moving_points=pts_m)

    assert torch.isfinite(model.warp_r2l).all()
    # Initialized warp should have non-zero magnitude due to landmark warm-start
    assert model.warp_r2l.abs().sum().item() > 0.0


def test_backward_compatibility_gaussian_mode():
    """Verify that existing default Gaussian projection works completely unchanged."""
    torch.manual_seed(42)
    pts = torch.tensor([[0.0, 0.0], [0.5, 0.5]], dtype=torch.float32)
    vals = torch.tensor([[1.0], [2.0]], dtype=torch.float32)

    grid = project_scattered_to_grid(
        points=pts,
        values=vals,
        grid_shape=32,
        domain_bounds=(-1.0, 1.0),
        sigma=0.05,
        method='gaussian',
    )
    assert grid.shape == (1, 1, 32, 32)
    assert torch.isfinite(grid).all()
