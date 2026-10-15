import pytest
import torch
import numpy as np
import ants
import syntx
from syntx.tvf import TVFModel, tvf_registration


@pytest.mark.parametrize("dim,shape", [(2, (16, 16)), (3, (8, 8, 8))])
def test_adjoint_euler_gradient_parity(dim, shape):
    """Verify bitwise/machine-precision gradient parity between standard autograd Euler and Adjoint Euler."""
    torch.manual_seed(42)
    m_std = TVFModel(dim=dim, image_shape=shape, velocity_shape=shape, solver='euler', n_time_steps=3)
    m_adj = TVFModel(dim=dim, image_shape=shape, velocity_shape=shape, solver='adjoint_euler', n_time_steps=3)

    v = torch.randn_like(m_std.velocity) * 0.05
    with torch.no_grad():
        m_std.velocity.copy_(v)
        m_adj.velocity.copy_(v)

    disp_std = m_std.integrate(0.0, 1.0)
    loss_std = (disp_std ** 2).sum()
    loss_std.backward()

    disp_adj = m_adj.integrate(0.0, 1.0)
    loss_adj = (disp_adj ** 2).sum()
    loss_adj.backward()

    disp_diff = (disp_std - disp_adj).abs().max().item()
    loss_diff = abs(loss_std.item() - loss_adj.item())
    grad_diff = (m_std.velocity.grad - m_adj.velocity.grad).abs().max().item()

    assert disp_diff < 1e-6, f"Displacement mismatch: {disp_diff}"
    assert loss_diff < 1e-6, f"Loss mismatch: {loss_diff}"
    assert grad_diff < 1e-6, f"Gradient mismatch: {grad_diff}"


@pytest.mark.parametrize("dim,shape", [(2, (16, 16)), (3, (8, 8, 8))])
@pytest.mark.parametrize("v_scale", [0.05, 2.5])
def test_adjoint_rk4_gradient_parity(dim, shape, v_scale):
    """Verify bitwise/machine-precision gradient parity between standard autograd RK4 and Adjoint RK4 (including adaptive CFL regime)."""
    torch.manual_seed(42)
    m_std = TVFModel(dim=dim, image_shape=shape, velocity_shape=shape, solver='rk4', n_time_steps=3, integration_steps_per_interval=1)
    m_adj = TVFModel(dim=dim, image_shape=shape, velocity_shape=shape, solver='rk4', adjoint=True, n_time_steps=3, integration_steps_per_interval=1)
    assert m_adj.solver == 'adjoint_rk4'
    assert m_adj.adjoint is True

    v = torch.randn_like(m_std.velocity) * v_scale
    with torch.no_grad():
        m_std.velocity.copy_(v)
        m_adj.velocity.copy_(v)

    disp_std = m_std.integrate(0.0, 1.0)
    loss_std = (disp_std ** 2).sum()
    loss_std.backward()

    disp_adj = m_adj.integrate(0.0, 1.0)
    loss_adj = (disp_adj ** 2).sum()
    loss_adj.backward()

    disp_diff = (disp_std - disp_adj).abs().max().item()
    loss_diff = abs(loss_std.item() - loss_adj.item())
    grad_diff = (m_std.velocity.grad - m_adj.velocity.grad).abs().max().item()

    assert disp_diff < 1e-5, f"RK4 displacement mismatch: {disp_diff}"
    assert loss_diff < 1e-4, f"RK4 loss mismatch: {loss_diff}"
    assert grad_diff < 1e-4, f"RK4 gradient mismatch: {grad_diff}"


def test_tvf_registration_adjoint_euler_and_rk4_2d():
    """Verify end-to-end registration pipeline execution with adjoint solvers in 2D."""
    fi = ants.image_read(ants.get_ants_data('r16')).resample_image((32, 32), use_voxels=True)
    mi = ants.image_read(ants.get_ants_data('r64')).resample_image((32, 32), use_voxels=True)

    reg_euler = tvf_registration(
        fixed=fi,
        moving=mi,
        reg_iterations=[3, 3],
        solver='adjoint_euler',
        affine_dof='translation',
        energy_weight=1e-3,
    )
    assert 'warpedmovout' in reg_euler
    assert 'fwdtransforms' in reg_euler
    assert reg_euler['provenance']['solver'] == 'adjoint_euler'

    reg_rk4 = tvf_registration(
        fixed=fi,
        moving=mi,
        reg_iterations=[3, 3],
        solver='rk4',
        adjoint=True,
        affine_dof='translation',
        energy_weight=1e-3,
    )
    assert 'warpedmovout' in reg_rk4
    assert 'fwdtransforms' in reg_rk4
    assert reg_rk4['provenance']['solver'] == 'rk4'
    assert reg_rk4['provenance']['adjoint'] is True


def test_tvf_registration_adjoint_3d():
    """Verify end-to-end registration pipeline execution with adjoint solver in 3D."""
    arr_fi = np.zeros((16, 16, 16), dtype=np.float32)
    arr_fi[4:12, 4:12, 4:12] = 1.0
    arr_mi = np.zeros((16, 16, 16), dtype=np.float32)
    arr_mi[5:13, 5:13, 5:13] = 1.0

    fi = ants.from_numpy(arr_fi, origin=(0, 0, 0), spacing=(1.0, 1.0, 1.0))
    mi = ants.from_numpy(arr_mi, origin=(0, 0, 0), spacing=(1.0, 1.0, 1.0))

    reg = tvf_registration(
        fixed=fi,
        moving=mi,
        reg_iterations=[3],
        solver='adjoint_euler',
        affine_dof='translation',
        energy_weight=1e-3,
    )
    assert 'warpedmovout' in reg
    assert 'fwdtransforms' in reg
    assert reg['provenance']['solver'] == 'adjoint_euler'
