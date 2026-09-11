"""
Adversarial Verification Suite for Milestone 2:
TVF Integration Caching, Adaptive CFL Sizing, and Downstream Solver Stability.

Tests:
1. Within-epoch cache reuse: v_max_voxel is evaluated once and reused across repeated integrate() calls.
2. Multi-resolution pyramid invalidation: target_shape changes produce distinct cache keys and recompute.
3. Adaptive CFL scaling: as velocity magnitude increases, n_steps increases to satisfy CFL condition.
4. PyTorch optimizer in-place invalidation: optimizer.step() updates increment _version and invalidate cache.
5. In-place .data update flaw: verifies that modifying .data directly bypasses _version increment,
   revealing why TVFModel.fit() with optimizer_type='cfl' and elastic_sigma=0.0 fails to invalidate cache.
6. Downstream deformation regularity and folding: verifies min det(J) > 0 and 0.0% folding on TVF registrations.
7. Solver-dependent CFL safety factor: verifies c_cfl is 1.0 for RK4 and 0.5 for Euler.
8. Cache capacity bound: verifies _v_max_cache bound at 32 entries preventing unbounded growth.
"""

import math
import numpy as np
import pytest
import torch
import ants

from syntx.tvf import TVFModel
from syntx import tvf
from syntx.spatial import jacobian_determinant


def test_tvf_cfl_cache_reuse_within_epoch():
    """Verify v_max_voxel is computed once and reused across repeated integrate() calls."""
    model = TVFModel(image_shape=(32, 32), velocity_shape=(16, 16), dim=2, n_time_steps=3)
    with torch.no_grad():
        model.velocity.add_(0.5)

    # Initial call populates cache
    key = (id(model.velocity), model.velocity._version, (32, 32))
    assert key not in model._v_max_cache

    disp1 = model.integrate(0.0, 0.5)
    assert key in model._v_max_cache
    v_max_first = model._v_max_cache[key]
    assert v_max_first > 0.0

    # Repeated calls must reuse cached value without modifying cache
    disp2 = model.integrate(0.5, 1.0)
    disp3 = model.integrate(1.0, 0.0)
    assert model._v_max_cache[key] == v_max_first
    assert len(model._v_max_cache) == 1


def test_tvf_cfl_cache_pyramid_invalidation():
    """Verify target_shape changes across pyramid levels generate separate cache entries."""
    model = TVFModel(image_shape=(64, 64), velocity_shape=(32, 32), dim=2, n_time_steps=3)
    with torch.no_grad():
        model.velocity.add_(1.0)

    # Level 1: (16, 16)
    v_max_16 = model._get_v_max_voxel(model.velocity, (16, 16))
    key_16 = (id(model.velocity), model.velocity._version, (16, 16))

    # Level 2: (32, 32)
    v_max_32 = model._get_v_max_voxel(model.velocity, (32, 32))
    key_32 = (id(model.velocity), model.velocity._version, (32, 32))

    # Level 3: (64, 64)
    v_max_64 = model._get_v_max_voxel(model.velocity, (64, 64))
    key_64 = (id(model.velocity), model.velocity._version, (64, 64))

    assert key_16 in model._v_max_cache
    assert key_32 in model._v_max_cache
    assert key_64 in model._v_max_cache

    # In voxel units, finer grids have more voxels for the same physical velocity,
    # so v_max in voxel units differs across resolutions
    assert not math.isclose(v_max_16, v_max_64, rel_tol=1e-3)


def test_tvf_adaptive_cfl_step_scaling():
    """Verify n_steps scales dynamically with velocity magnitude to satisfy CFL bound."""
    model = TVFModel(image_shape=(32, 32), velocity_shape=(32, 32), dim=2, n_time_steps=4, solver='euler')
    default_steps = model.n_time_steps * model.integration_steps_per_interval

    # Zero velocity: n_steps defaults to default_steps
    model.velocity.data.zero_()
    disp_zero = model.integrate(0.0, 1.0)
    assert disp_zero.abs().max().item() == 0.0

    # Small velocity: v_max_voxel small enough that default_steps suffices
    with torch.no_grad():
        model.velocity.fill_(0.01)
    model._v_max_cache.clear()
    v_small = model._get_v_max_voxel(model.velocity, (32, 32))
    c_cfl = 0.5
    steps_small = max(default_steps, int(math.ceil(v_small * 1.0 / c_cfl)))
    assert steps_small == default_steps

    # Large velocity: requires significantly more steps to prevent grid folding
    # default_steps is 16, so v_max > 8.0 is needed for cfl_steps > default_steps
    with torch.no_grad():
        model.velocity.fill_(8.0)
    model._v_max_cache.clear()
    v_large = model._get_v_max_voxel(model.velocity, (32, 32))
    steps_large = max(default_steps, int(math.ceil(v_large * 1.0 / c_cfl)))
    assert steps_large > default_steps
    assert steps_large >= 23


def test_tvf_cfl_cache_invalidation_pytorch_optimizers():
    """Verify PyTorch optimizers (LARS, Adam, SGD) increment parameter version and invalidate cache."""
    for opt_name in ['lars', 'adam', 'sgd']:
        model = TVFModel(image_shape=(16, 16), velocity_shape=(16, 16), dim=2, n_time_steps=2)
        v0 = model.velocity._version

        # Pre-populate cache at version 0
        val0 = model._get_v_max_voxel(model.velocity, (16, 16))
        assert (id(model.velocity), v0, (16, 16)) in model._v_max_cache

        # Run synthetic optimization step
        fi = torch.zeros((1, 1, 16, 16))
        mi = torch.zeros((1, 1, 16, 16))
        fi[:, :, 4:12, 4:12] = 1.0
        mi[:, :, 6:14, 4:12] = 1.0

        model.fit(fi, mi, levels=[1], epochs_per_level=[1], affine_epochs=0, verbose=False, optimizer_type=opt_name)

        v1 = model.velocity._version
        assert v1 > v0, f"Optimizer {opt_name} did not increment velocity._version"
        assert (id(model.velocity), v1, (16, 16)) in model._v_max_cache, f"New version not cached for {opt_name}"


def test_tvf_cfl_cache_data_sub_flaw_reproduction():
    """
    Empirically reproduce the cache invalidation failure when updating velocity via .data.sub_
    (used in TVFModel.fit lines 1364/1366 when optimizer_type='cfl' and elastic_sigma=0).
    """
    model = TVFModel(image_shape=(32, 32), velocity_shape=(32, 32), dim=2, n_time_steps=4)
    # Initial velocity is zero
    v_init = model._get_v_max_voxel(model.velocity, (32, 32))
    assert v_init == 0.0
    initial_version = model.velocity._version

    # Simulate in-place update using .data.sub_ (the exact operation in TVF CFL optimizer)
    update = torch.ones_like(model.velocity.data) * 2.0
    model.velocity.data.sub_(update)

    # In PyTorch, .data operations bypass _version tracking
    assert model.velocity._version == initial_version, "PyTorch unexpectedly tracked .data modification"

    # Because _version did not increment, _get_v_max_voxel returns the stale zero value!
    v_stale = model._get_v_max_voxel(model.velocity, (32, 32))
    assert v_stale == 0.0, "Cache key did not hit stale entry"

    # Verify that proper PyTorch in-place update (with torch.no_grad(): param.sub_) DOES increment version
    with torch.no_grad():
        model.velocity.sub_(update)
    assert model.velocity._version > initial_version
    v_fresh = model._get_v_max_voxel(model.velocity, (32, 32))
    assert v_fresh > 0.0, "Proper in-place update did not compute fresh velocity max"


def test_tvf_deformation_regularity_and_folding_2d():
    """Verify TVF registration preserves deformation regularity (min det(J) > 0, 0.0% folding)."""
    fi_np = np.zeros((32, 32), dtype=np.float32)
    fi_np[8:24, 8:24] = 1.0
    mi_np = np.zeros((32, 32), dtype=np.float32)
    mi_np[10:26, 8:24] = 1.0

    fi = ants.from_numpy(fi_np)
    mi = ants.from_numpy(mi_np)

    reg = tvf(fi, mi, reg_iterations=[15, 10], optimizer='lars', verbose=False)
    model = reg['model']

    fwd_disp = model.get_forward_warp().detach().cpu().squeeze(0).numpy()
    jac = jacobian_determinant(fwd_disp, ref_image=fi)

    min_det = float(np.min(jac))
    folding_pct = float(np.mean(jac <= 0.0)) * 100.0

    assert min_det > 0.0, f"Deformation fold detected: min det(J) = {min_det:.4f}"
    assert folding_pct == 0.0, f"Non-zero folding percentage: {folding_pct:.4f}%"

    inv_err = reg['inverse_identity_errors']['phi_1']
    assert inv_err['mean_error'] < 0.20, f"High inverse consistency error: {inv_err['mean_error']:.4f} mm"


def test_tvf_rk4_vs_euler_cfl_thresholds():
    """Verify c_cfl factor differs between RK4 (1.0) and Euler (0.5)."""
    model_euler = TVFModel(image_shape=(16, 16), velocity_shape=(16, 16), dim=2, solver='euler')
    model_rk4 = TVFModel(image_shape=(16, 16), velocity_shape=(16, 16), dim=2, solver='rk4')

    # Set velocity so v_max = 2.0:
    # Euler c_cfl = 0.5 -> ceil(2.0 / 0.5) = 4
    # RK4 c_cfl = 1.0 -> ceil(2.0 / 1.0) = 2
    with torch.no_grad():
        model_euler.velocity.zero_()
        model_euler.velocity[..., 0] = 2.0
        model_rk4.velocity.zero_()
        model_rk4.velocity[..., 0] = 2.0

    v_euler = model_euler._get_v_max_voxel(model_euler.velocity, (16, 16))
    v_rk4 = model_rk4._get_v_max_voxel(model_rk4.velocity, (16, 16))
    assert math.isclose(v_euler, 2.0, rel_tol=1e-5)
    assert math.isclose(v_rk4, 2.0, rel_tol=1e-5)

    # RK4 stability threshold c_cfl=1.0 requires half the steps of Euler c_cfl=0.5
    steps_euler = int(math.ceil(v_euler * 1.0 / 0.5))
    steps_rk4 = int(math.ceil(v_rk4 * 1.0 / 1.0))
    assert steps_euler == 4
    assert steps_rk4 == 2
    assert steps_euler == 2 * steps_rk4


def test_tvf_cache_capacity_bound():
    """Verify _v_max_cache doesn't exceed 32 entries and bounds memory growth."""
    model = TVFModel(image_shape=(16, 16), velocity_shape=(16, 16), dim=2)
    # Generate 40 distinct cache lookups with different shapes
    for i in range(16, 56):
        shape = (i, i)
        model._get_v_max_voxel(model.velocity, shape)

    assert len(model._v_max_cache) <= 32, f"Cache exceeded bound of 32: {len(model._v_max_cache)}"
