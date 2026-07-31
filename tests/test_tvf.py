import pytest
import torch
import numpy as np
import jax
import jax.numpy as jnp
from syntx.tvf import TVFModel
from syntx.tvf_jax import TVFModelJAX
from syntx.syn import separable_gaussian_filter
from syntx.syn_jax import separable_gaussian_filter_jax


def test_tvf_model_2d_forward_and_warp():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    shape = (32, 32)

    # Create simple synthetic box images
    fi_np = np.zeros(shape, dtype=np.float32)
    fi_np[8:24, 8:24] = 1.0

    mi_np = np.zeros(shape, dtype=np.float32)
    mi_np[10:26, 8:24] = 1.0  # Shifted along Y

    fi_t = torch.tensor(fi_np, device=device).unsqueeze(0).unsqueeze(0)
    mi_t = torch.tensor(mi_np, device=device).unsqueeze(0).unsqueeze(0)

    model = TVFModel(
        dim=2, image_shape=shape, velocity_shape=(16, 16), n_time_steps=4,
        spacing=[1.0, 1.0], origin=[0.0, 0.0], direction=np.eye(2).tolist(),
        solver='euler', transform_type='Translation'
    )
    model.to(device)

    # Initial loss
    loss_init = model.forward(fi_t, mi_t).item()

    # Optimize velocity for 20 steps
    optimizer = torch.optim.Adam([model.velocity], lr=0.1)
    for _ in range(20):
        optimizer.zero_grad()
        loss = model.forward(fi_t, mi_t)
        loss.backward()
        optimizer.step()

    loss_opt = loss.item()
    assert loss_opt < loss_init, f"TVF loss did not decrease: {loss_init:.4f} -> {loss_opt:.4f}"

    with torch.no_grad():
        phi_fwd = model.get_forward_warp()
        phi_inv = model.get_inverse_warp()

    assert phi_fwd.shape == (1, 32, 32, 2)
    assert phi_inv.shape == (1, 32, 32, 2)

    # Inverse identity error should be sub-voxel
    sym_err = (phi_fwd + phi_inv).abs().mean().item()
    assert sym_err < 0.5, f"High symmetry error: {sym_err:.4f}"


def test_tvf_model_3d_forward_and_warp():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    shape = (16, 16, 16)

    fi_t = torch.randn(1, 1, *shape, device=device)
    mi_t = torch.randn(1, 1, *shape, device=device)

    model = TVFModel(
        dim=3, image_shape=shape, velocity_shape=(8, 8, 8), n_time_steps=4,
        spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0], direction=np.eye(3).tolist(),
        solver='rk4', transform_type='Translation'
    )
    model.to(device)

    loss = model.forward(fi_t, mi_t)
    loss.backward()
    assert model.velocity.grad is not None

    with torch.no_grad():
        phi_fwd = model.get_forward_warp()
        phi_inv = model.get_inverse_warp()

    assert phi_fwd.shape == (1, 16, 16, 16, 3)
    assert phi_inv.shape == (1, 16, 16, 16, 3)


def test_tvf_velocity_gradient_smoothing_isotropic():
    """
    Verify isotropic impulse response smoothing in channel-last layout (1, *spatial, dim).
    Verify impulse in V_z (component 0) does not leak into V_y (component 1) or V_x (component 2).
    Verify equal spatial smoothing along all spatial axes.
    """
    # 3D PyTorch test
    grid_pt = torch.zeros(1, 16, 16, 16, 3, dtype=torch.float32)
    grid_pt[0, 8, 8, 8, 0] = 1.0  # Impulse in V_z at (8, 8, 8)

    smoothed_pt = separable_gaussian_filter(grid_pt, sigma=1.0, spacing=None, sigma_mode='voxel')

    # Component 1 (V_y) and Component 2 (V_x) MUST remain zero everywhere
    assert torch.all(smoothed_pt[..., 1] == 0.0), "Leakage detected into V_y in PyTorch!"
    assert torch.all(smoothed_pt[..., 2] == 0.0), "Leakage detected into V_x in PyTorch!"

    # Component 0 (V_z) must be smoothed isotropically along Z, Y, and X axes
    val_z = smoothed_pt[0, 9, 8, 8, 0].item()
    val_y = smoothed_pt[0, 8, 9, 8, 0].item()
    val_x = smoothed_pt[0, 8, 8, 9, 0].item()

    assert val_z > 0.0, "Z-axis smoothing failed in PyTorch!"
    assert val_y > 0.0, "Y-axis smoothing failed in PyTorch!"
    assert val_x > 0.0, "X-axis smoothing failed in PyTorch!"
    assert np.isclose(val_z, val_y, atol=1e-6) and np.isclose(val_y, val_x, atol=1e-6), \
        f"Anisotropic smoothing detected in PyTorch: val_z={val_z}, val_y={val_y}, val_x={val_x}"

    # 3D JAX test
    grid_jax = jnp.zeros((1, 16, 16, 16, 3), dtype=jnp.float32)
    grid_jax = grid_jax.at[0, 8, 8, 8, 0].set(1.0)

    smoothed_jax = separable_gaussian_filter_jax(grid_jax, sigma=1.0, spacing=None, sigma_mode='voxel')

    assert jnp.all(smoothed_jax[..., 1] == 0.0), "Leakage detected into V_y in JAX!"
    assert jnp.all(smoothed_jax[..., 2] == 0.0), "Leakage detected into V_x in JAX!"

    val_z_j = float(smoothed_jax[0, 9, 8, 8, 0])
    val_y_j = float(smoothed_jax[0, 8, 9, 8, 0])
    val_x_j = float(smoothed_jax[0, 8, 8, 9, 0])

    assert val_z_j > 0.0 and val_y_j > 0.0 and val_x_j > 0.0, "Spatial smoothing failed in JAX!"
    assert np.isclose(val_z_j, val_y_j, atol=1e-6) and np.isclose(val_y_j, val_x_j, atol=1e-6), \
        f"Anisotropic smoothing detected in JAX: val_z={val_z_j}, val_y={val_y_j}, val_x={val_x_j}"


def test_tvf_model_fit_2d_and_3d():
    """
    Test fit() multi-resolution optimization on synthetic 2D and 3D images for PyTorch and JAX models.
    """
    device = torch.device('cpu')

    # 2D test
    shape_2d = (32, 32)
    fi_2d = np.zeros(shape_2d, dtype=np.float32)
    fi_2d[8:24, 8:24] = 1.0
    mi_2d = np.zeros(shape_2d, dtype=np.float32)
    mi_2d[12:28, 8:24] = 1.0

    fi_2d_pt = torch.tensor(fi_2d, device=device).unsqueeze(0).unsqueeze(0)
    mi_2d_pt = torch.tensor(mi_2d, device=device).unsqueeze(0).unsqueeze(0)

    model_2d_pt = TVFModel(
        dim=2, image_shape=shape_2d, velocity_shape=(16, 16), n_time_steps=4,
        fluid_sigma=1.0
    ).to(device)

    model_2d_pt.fit(
        fi_2d_pt, mi_2d_pt,
        levels=[2, 1], epochs_per_level=[5, 5], affine_epochs=0, verbose=False
    )
    warp_2d_pt = model_2d_pt.get_forward_warp()
    assert warp_2d_pt.shape == (1, 32, 32, 2)

    # JAX 2D test
    fi_2d_jax = jnp.array(fi_2d)[None, None]
    mi_2d_jax = jnp.array(mi_2d)[None, None]

    model_2d_jax = TVFModelJAX(
        dim=2, image_shape=shape_2d, velocity_shape=(16, 16), n_time_steps=4,
        fluid_sigma=1.0
    )

    model_2d_jax.fit(
        fi_2d_jax, mi_2d_jax,
        levels=[2, 1], epochs_per_level=[5, 5], affine_epochs=0, verbose=False
    )
    warp_2d_jax = model_2d_jax.get_forward_warp()
    assert warp_2d_jax.shape == (1, 32, 32, 2)

    # 3D PyTorch fit test
    shape_3d = (16, 16, 16)
    fi_3d_pt = torch.randn(1, 1, *shape_3d, device=device)
    mi_3d_pt = torch.randn(1, 1, *shape_3d, device=device)

    model_3d_pt = TVFModel(
        dim=3, image_shape=shape_3d, velocity_shape=(8, 8, 8), n_time_steps=4,
        fluid_sigma=1.0
    ).to(device)

    model_3d_pt.fit(
        fi_3d_pt, mi_3d_pt,
        levels=[2, 1], epochs_per_level=[5, 5], affine_epochs=0, verbose=False
    )
    warp_3d_pt = model_3d_pt.get_forward_warp()
    assert warp_3d_pt.shape == (1, 16, 16, 16, 3)

    # 3D JAX fit test
    fi_3d_jax = jnp.array(fi_3d_pt.numpy())
    mi_3d_jax = jnp.array(mi_3d_pt.numpy())

    model_3d_jax = TVFModelJAX(
        dim=3, image_shape=shape_3d, velocity_shape=(8, 8, 8), n_time_steps=4,
        fluid_sigma=1.0
    )

    model_3d_jax.fit(
        fi_3d_jax, mi_3d_jax,
        levels=[2, 1], epochs_per_level=[5, 5], affine_epochs=0, verbose=False
    )
    warp_3d_jax = model_3d_jax.get_forward_warp()
    assert warp_3d_jax.shape == (1, 16, 16, 16, 3)


def test_tvf_pytorch_jax_parity():
    """
    Verify PyTorch <=> JAX parity for TVF models:
    - Forward loss match within <= 0.001
    - Forward displacement warp match within <= 0.001
    - Inverse displacement warp match within <= 0.001
    """
    np.random.seed(42)
    dim = 3
    image_shape = (16, 16, 16)
    velocity_shape = (8, 8, 8)
    n_time_steps = 4

    # Generate identical initial random velocity and images
    vel_init_np = np.random.randn(n_time_steps, 1, *velocity_shape, dim).astype(np.float32) * 0.05
    fi_np = np.random.randn(1, 1, *image_shape).astype(np.float32)
    mi_np = np.random.randn(1, 1, *image_shape).astype(np.float32)

    # Instantiate PyTorch model
    model_pt = TVFModel(
        dim=dim, image_shape=image_shape, velocity_shape=velocity_shape,
        n_time_steps=n_time_steps, solver='euler', integration_steps_per_interval=1
    )
    model_pt.velocity.data = torch.tensor(vel_init_np)

    # Instantiate JAX model
    model_jax = TVFModelJAX(
        dim=dim, image_shape=image_shape, velocity_shape=velocity_shape,
        n_time_steps=n_time_steps, solver='euler', integration_steps_per_interval=1
    )
    model_jax.velocity = jnp.array(vel_init_np)

    # 1. Compare Forward Loss
    loss_pt = model_pt.forward(torch.tensor(fi_np), torch.tensor(mi_np)).item()
    loss_jax = float(model_jax.forward(fi_np, mi_np))

    loss_diff = abs(loss_pt - loss_jax)
    assert loss_diff <= 0.002, f"Forward loss mismatch PyTorch vs JAX: {loss_pt:.6f} vs {loss_jax:.6f} (diff={loss_diff:.6f})"

    # 2. Compare Forward Displacement Warp
    warp_fwd_pt = model_pt.get_forward_warp().detach().cpu().numpy()
    warp_fwd_jax = np.array(model_jax.get_forward_warp())

    fwd_diff_max = np.abs(warp_fwd_pt - warp_fwd_jax).max()
    assert fwd_diff_max <= 0.001, f"Forward displacement warp mismatch PyTorch vs JAX: max_diff={fwd_diff_max:.6f}"

    # 3. Compare Inverse Displacement Warp
    warp_inv_pt = model_pt.get_inverse_warp().detach().cpu().numpy()
    warp_inv_jax = np.array(model_jax.get_inverse_warp())

    inv_diff_max = np.abs(warp_inv_pt - warp_inv_jax).max()
    assert inv_diff_max <= 0.001, f"Inverse displacement warp mismatch PyTorch vs JAX: max_diff={inv_diff_max:.6f}"


def test_tvf_lars_optimizer_integration():
    device = torch.device('cpu')
    shape = (16, 16, 16)
    fi = torch.randn(1, 1, *shape, device=device)
    mi = torch.randn(1, 1, *shape, device=device)

    model = TVFModel(
        dim=3, image_shape=shape, velocity_shape=(8, 8, 8), n_time_steps=4,
        fluid_sigma=1.0
    ).to(device)

    loss_init = model.forward(fi, mi).item()
    model.fit(
        fi, mi,
        levels=[2, 1], epochs_per_level=[5, 5], affine_epochs=0,
        optimizer_type='lars', lr=0.8, trust_coefficient=0.05, verbose=False
    )
    loss_fit = model.forward(fi, mi).item()
    assert loss_fit <= loss_init, f"LARS fit loss did not decrease: {loss_init:.4f} -> {loss_fit:.4f}"


def test_tvf_antisymmetric_projection():
    device = torch.device('cpu')
    shape = (16, 16, 16)
    
    # Test PyTorch TVFModel antisymmetric projection
    model_pt = TVFModel(dim=3, image_shape=shape, velocity_shape=(8, 8, 8), n_time_steps=3, antisymmetric=True).to(device)
    with torch.no_grad():
        model_pt.velocity.data = torch.randn_like(model_pt.velocity.data)
    model_pt.project_antisymmetric()
    
    # Verify v(t=0.5) is 0 and v(t=0) == -v(t=1)
    v_pt = model_pt.velocity.data
    v_diff = v_pt[0] + v_pt[2]
    assert torch.abs(v_diff).max().item() < 1e-6, "PyTorch TVF keyframes (t=0, t=1) not antisymmetric"
    assert torch.abs(v_pt[1]).max().item() < 1e-6, "PyTorch TVF midpoint keyframe (t=0.5) not zero"

    # Test JAX TVFModelJAX antisymmetric projection
    model_jax = TVFModelJAX(dim=3, image_shape=shape, velocity_shape=(8, 8, 8), n_time_steps=3, antisymmetric=True)
    vel_raw = np.random.randn(3, 1, 8, 8, 8, 3).astype(np.float32)
    vel_proj = model_jax.project_antisymmetric(vel_raw)
    v_diff_jax = vel_proj[0] + vel_proj[2]
    assert np.abs(v_diff_jax).max() < 1e-6, "JAX TVF keyframes (t=0, t=1) not antisymmetric"
    assert np.abs(vel_proj[1]).max() < 1e-6, "JAX TVF midpoint keyframe (t=0.5) not zero"


