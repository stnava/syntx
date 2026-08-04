"""
tests/test_tvf_bugs.py — Unit and Regression Tests for TVF Bug Fixes
====================================================================
"""

import pytest
import torch
import numpy as np
from syntx.tvf import TVFModel


def test_problem_1_temporal_gradient_weighting():
    """
    Test Problem 1: Ensures velocity parameter gradients across time keyframes t_k
    are temporally weighted rather than being identically assigned uniform values.
    """
    device = 'cpu'
    model = TVFModel(
        dim=2,
        image_shape=(32, 32),
        velocity_shape=(32, 32),
        n_time_steps=5,
        fluid_sigma=1.0,
        elastic_sigma=0.0
    ).to(device)

    fixed = torch.randn(1, 1, 32, 32, device=device)
    moving = torch.randn(1, 1, 32, 32, device=device)

    # Run one epoch in analytical gradient mode
    model.fit(
        fixed, moving,
        epochs_per_level=[1],
        levels=[1],
        use_analytical_gradients=True,
        verbose=False
    )

    grad = model.velocity.grad  # Shape: (5, 1, 32, 32, 2)
    assert grad is not None, "Velocity gradient should not be None after fit()"

    # Verify that gradients across different keyframes are NOT all identical
    g0 = grad[0].cpu().numpy()
    g2 = grad[2].cpu().numpy()  # Midpoint keyframe (t=0.5)

    diff = np.abs(g0 - g2).max()
    assert diff > 1e-6, f"Expected non-zero temporal variation in gradients across keyframes, got max diff={diff}"


def test_problem_2_antisymmetric_drift_projection():
    """
    Test Problem 2: Ensures project_antisymmetric enforces midpoint velocity zeroing (v(x, 0.5) = 0)
    for strict temporal anti-symmetry across PyTorch and JAX.
    """
    device = 'cpu'
    model = TVFModel(
        dim=2,
        image_shape=(32, 32),
        velocity_shape=(32, 32),
        n_time_steps=5, # odd number: midpoint at index 2 (t=0.5)
        fluid_sigma=1.0,
        elastic_sigma=0.0
    ).to(device)

    # Initialize non-zero velocity at midpoint keyframe
    with torch.no_grad():
        model.velocity.data[2] = torch.ones_like(model.velocity.data[2]) * 2.0

    model.project_antisymmetric()

    mid_vel_norm = torch.norm(model.velocity.data[2]).item()
    assert mid_vel_norm < 1e-5, f"Midpoint velocity should be zeroed out by project_antisymmetric for anti-symmetry, got norm={mid_vel_norm}"


def test_problem_3_velocity_cfl_clamping():
    """
    Test Problem 3: Ensures velocity vector magnitudes are clamped within CFL voxel limits.
    """
    device = 'cpu'
    model = TVFModel(
        dim=2,
        image_shape=(32, 32),
        velocity_shape=(32, 32),
        spacing=(1.0, 1.0),
        n_time_steps=3,
        fluid_sigma=1.0,
        elastic_sigma=0.0
    ).to(device)

    # Inject extreme unconstrained velocity values
    with torch.no_grad():
        model.velocity.data.fill_(100.0)

    fixed = torch.randn(1, 1, 32, 32, device=device)
    moving = torch.randn(1, 1, 32, 32, device=device)

    model.fit(
        fixed, moving,
        epochs_per_level=[1],
        levels=[1],
        cfl_momentum=0.9,
        cfl_max=0.4,
        verbose=False
    )


    max_vel = torch.norm(model.velocity.data, dim=-1).max().item()
    # Spacing is 1.0 mm, max allowed physical step per keyframe is 0.4 * spacing = 0.4 mm
    assert max_vel <= 0.5, f"Expected velocity magnitude to be clamped within CFL bound <= 0.5 mm, got max_vel={max_vel}"


