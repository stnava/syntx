import pytest
import torch
import numpy as np

from syntx.core.inverse import (
    update_inverse_field_nd,
    update_inverse_field_nd_anderson,
    update_inverse_field_nd_hybrid_lm,
    compute_inverse_identity_error_nd,
    calculate_inverse_identity_error,
    integrate_time_varying_velocity_field,
)


def test_inverse_identity_error_zero_displacement():
    disp = torch.zeros(1, 16, 16, 2)
    err = compute_inverse_identity_error_nd(disp, disp, is_displacement=True)
    assert torch.allclose(err, torch.zeros_like(err), atol=1e-6)

    err_dict = calculate_inverse_identity_error(disp, disp, spacing=[1.0, 1.0], origin=[0.0, 0.0], direction=np.eye(2))
    assert np.isclose(err_dict['max_error'], 0.0, atol=1e-5)
    assert np.isclose(err_dict['mean_error'], 0.0, atol=1e-5)


def test_update_inverse_field_small_displacement():
    # Small synthetic translation
    disp = torch.zeros(1, 16, 16, 2)
    disp[..., 0] = 0.05
    disp[..., 1] = -0.05

    inv_anderson = update_inverse_field_nd(disp, method='anderson', steps=15)
    assert inv_anderson.shape == disp.shape

    # Inverse should be approximately negative displacement
    assert torch.allclose(inv_anderson[0, 4:12, 4:12, 0], -disp[0, 4:12, 4:12, 0], atol=1e-2)
    assert torch.allclose(inv_anderson[0, 4:12, 4:12, 1], -disp[0, 4:12, 4:12, 1], atol=1e-2)


def test_integrate_time_varying_velocity_field_zero():
    v = [torch.zeros(1, 16, 16, 2) for _ in range(4)]
    phi = integrate_time_varying_velocity_field(v, dt=0.25, mode='forward')
    assert torch.allclose(phi, torch.zeros_like(v[0]), atol=1e-6)
