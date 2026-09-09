import pytest
import torch
import numpy as np
import torch.nn.functional as F

from syntx.core.losses import (
    compute_image_distance_transform,
    compute_soft_distance_transform,
    distance_transform_loss,
)
from syntx.scattered.projection import compute_distance_transform_to_grid
from syntx.scattered.solver import syn_scattered, ScatteredRegistrationConfig
from syntx.syn import syn


def test_image_distance_transform_2d():
    """Verify exact Euclidean distance transform on 2D binary square."""
    img = torch.zeros(64, 64)
    img[24:40, 24:40] = 1.0  # 16x16 center square

    # Unsigned distance
    d_out = compute_image_distance_transform(img)
    assert d_out.shape == (64, 64)
    # Inside the square: distance is 0.0
    assert torch.all(d_out[25:39, 25:39] == 0.0)
    # Outside the square: distance increases with Euclidean distance
    assert d_out[20, 32].item() == pytest.approx(4.0, abs=0.1)
    assert d_out[0, 0].item() > 20.0

    # Signed distance
    d_signed = compute_image_distance_transform(img, signed=True)
    # Inside should be negative
    assert d_signed[32, 32].item() < -6.0
    # Outside should be positive
    assert d_signed[0, 0].item() > 20.0

    # Smooth potential
    pot = compute_image_distance_transform(img, tau=5.0)
    assert pot.min() >= 0.0 and pot.max() <= 1.0
    # On surface, potential is 1.0
    assert pot[32, 32].item() == pytest.approx(1.0, abs=1e-3)


def test_image_distance_transform_3d():
    """Verify exact Euclidean distance transform on 3D sphere mask."""
    res = 32
    coords = torch.linspace(-1.0, 1.0, res)
    z, y, x = torch.meshgrid(coords, coords, coords, indexing='ij')
    r = torch.sqrt(x**2 + y**2 + z**2)
    mask = (r < 0.50).float()

    d_out = compute_image_distance_transform(mask, tau=5.0)
    assert d_out.shape == (res, res, res)
    assert d_out.min() >= 0.0 and d_out.max() <= 1.0


def test_soft_distance_transform_autograd():
    """Verify that soft distance transform supports autograd backpropagation."""
    img = torch.zeros(1, 1, 32, 32, requires_grad=True)
    with torch.enable_grad():
        d = compute_soft_distance_transform(img, sigma=2.0)
        loss = d.sum()
        loss.backward()
    assert img.grad is not None
    assert torch.isfinite(img.grad).all()


def test_distance_transform_loss_modes():
    """Verify distance_transform_loss across modes and autograd gradients."""
    I = torch.zeros(1, 1, 32, 32)
    I[:, :, 10:22, 10:22] = 1.0
    J = torch.zeros(1, 1, 32, 32)
    J[:, :, 12:24, 12:24] = 1.0

    grid = torch.zeros(1, 32, 32, 2, requires_grad=True)
    phi = F.affine_grid(torch.eye(2, 3).unsqueeze(0), [1, 1, 32, 32], align_corners=True) + grid
    I_warped = F.grid_sample(I, phi, align_corners=True)

    for mode in ['potential_lncc', 'potential_mse', 'edt_mse', 'edt_l1']:
        loss = distance_transform_loss(I_warped, J, mode=mode, tau=5.0)
        assert torch.isfinite(loss), f"Loss is not finite for mode={mode}"
        loss.backward(retain_graph=True)
        assert grid.grad is not None and torch.isfinite(grid.grad).all()
        grid.grad.zero_()


def test_scattered_distance_transform():
    """Verify compute_distance_transform_to_grid on 3D point cloud."""
    points = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ])
    dt = compute_distance_transform_to_grid(
        points,
        grid_shape=(16, 16, 16),
        domain_bounds=(-1.0, 1.0),
    )
    assert dt.shape == (1, 1, 16, 16, 16)
    assert dt.min() >= 0.0

    # Potential mode
    pot = compute_distance_transform_to_grid(
        points,
        grid_shape=(16, 16, 16),
        domain_bounds=(-1.0, 1.0),
        potential_tau=0.20,
    )
    assert pot.min() >= 0.0 and pot.max() <= 1.0


def test_syn_with_dt_metric():
    """Verify SyN with similarity_metric='dt' on 2D images."""
    import ants
    import numpy as np

    arr_f = np.zeros((32, 32), dtype=np.float32)
    arr_f[10:22, 10:22] = 1.0
    arr_m = np.zeros((32, 32), dtype=np.float32)
    arr_m[12:24, 12:24] = 1.0

    img_f = ants.from_numpy(arr_f)
    img_m = ants.from_numpy(arr_m)

    res = syn(
        fixed=img_f,
        moving=img_m,
        similarity_metric='dt',
        reg_iterations=[10, 5],
        fast_smooth=True,
    )
    assert res['warpedmovout'] is not None
    assert isinstance(res['warpedmovout'], ants.ANTsImage)


def test_scattered_syn_with_dt_tau():
    """Verify syn_scattered with native distance_transform_tau on 3D sphere mesh."""
    theta = torch.linspace(0, 2 * np.pi, 8)
    phi = torch.linspace(0.2 * np.pi, 0.8 * np.pi, 8)
    tt, pp = torch.meshgrid(theta, phi, indexing='ij')
    vx = 0.5 * torch.sin(pp) * torch.cos(tt)
    vy = 0.5 * torch.sin(pp) * torch.sin(tt)
    vz = 0.5 * torch.cos(pp)
    V_fix = torch.stack([vx.flatten(), vy.flatten(), vz.flatten()], dim=-1)

    V_mov = V_fix.clone()
    V_mov[:, 0] += 0.04 * torch.sin(np.pi * V_fix[:, 1])

    cfg = ScatteredRegistrationConfig(
        dim=3,
        grid_res=32,
        window_size=5,
        epochs_per_level=[10, 10],
        levels=[2, 1],
        fluid_sigma=2.5,
        distance_transform_tau=0.10,
        regularizer='gaussian',
    )
    res = syn_scattered(
        fixed_points=V_fix,
        moving_points=V_mov,
        config=cfg,
    )
    err_init = torch.norm(V_mov - V_fix, dim=-1).mean().item()
    err_final = torch.norm(res.warped_moving_points - V_fix, dim=-1).mean().item()
    assert err_final < err_init, f"Registration failed to reduce error ({err_final:.4f} >= {err_init:.4f})"
    assert res.folding_percentage < 0.01


def test_image_distance_transform_physical_space_ants():
    """Verify compute_image_distance_transform respects physical spacing from ANTsImage."""
    import ants
    # 2D ANTsImage with anisotropic spacing (2.0 mm, 3.0 mm)
    arr = np.zeros((20, 20), dtype=np.float32)
    arr[10, 10] = 1.0
    img = ants.from_numpy(arr, spacing=(2.0, 3.0))

    dt = compute_image_distance_transform(img)
    # Step along X (axis 0, spacing 2.0 mm)
    assert np.isclose(dt[11, 10].item(), 2.0)
    # Step along Y (axis 1, spacing 3.0 mm)
    assert np.isclose(dt[10, 11].item(), 3.0)

    # 3D ANTsImage with anisotropic spacing (1.0 mm, 1.5 mm, 2.0 mm)
    arr_3d = np.zeros((15, 15, 15), dtype=np.float32)
    arr_3d[7, 7, 7] = 1.0
    img_3d = ants.from_numpy(arr_3d, spacing=(1.0, 1.5, 2.0))
    dt_3d = compute_image_distance_transform(img_3d)
    assert np.isclose(dt_3d[8, 7, 7].item(), 1.0)
    assert np.isclose(dt_3d[7, 8, 7].item(), 1.5)
    assert np.isclose(dt_3d[7, 7, 8].item(), 2.0)


def test_image_distance_transform_physical_space_tensor():
    """Verify compute_image_distance_transform with PyTorch tensor and sampling_spacing."""
    t = torch.zeros(20, 20)
    t[10, 10] = 1.0
    dt_pt = compute_image_distance_transform(t, sampling_spacing=(2.0, 3.0))
    # In PyTorch tensor, row index is Y (spacing 3.0), col index is X (spacing 2.0)
    assert np.isclose(dt_pt[11, 10].item(), 3.0)
    assert np.isclose(dt_pt[10, 11].item(), 2.0)

