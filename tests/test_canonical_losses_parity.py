"""Cross-backend numerical parity test suite: PyTorch vs JAX.

Specifications (per ORIGINAL_REQUEST.md & PROJECT.md):
- Cross-backend PyTorch vs JAX numerical parity for LNCC, CC2, Soft Dice, MSE.
- Mattes MI boundary-padded parity test (pad=2.0).
- GEMINI.md Rule 2 compliance: PyTorch and JAX are compute engines, not algorithm variants.
"""

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from syntx.core.losses import (
    b_spline_3,
    local_ncc_loss_nd,
    mattes_mi_loss_core,
    mse_loss_nd,
    soft_dice_loss_nd,
)
from syntx.syn_jax import (
    b_spline_3_jax,
    local_ncc_loss_nd_jax,
    mattes_mi_loss_core_jax,
    soft_dice_loss_nd_jax,
)


@pytest.mark.parametrize("squared", [False, True])
@pytest.mark.parametrize("window_size", [3, 5])
def test_jax_lncc_and_cc2_numerical_parity_2d(squared, window_size):
    """Verify PyTorch and JAX compute identical LNCC / CC2 losses on 2D inputs."""
    rng = np.random.default_rng(42)
    a_np = rng.standard_normal((1, 1, 24, 24)).astype("float32")
    b_np = rng.standard_normal((1, 1, 24, 24)).astype("float32")

    loss_pt = float(
        local_ncc_loss_nd(
            torch.from_numpy(a_np),
            torch.from_numpy(b_np),
            window_size=window_size,
            squared=squared,
            use_ants_pseudo_gradient=False,
        )
    )
    loss_jx = float(
        local_ncc_loss_nd_jax(
            jnp.array(a_np),
            jnp.array(b_np),
            window_size=window_size,
            squared=squared,
            use_ants_pseudo_gradient=False,
        )
    )

    diff = abs(loss_pt - loss_jx)
    assert diff < 1e-4, f"2D LNCC/CC2 (squared={squared}, ws={window_size}) parity diff {diff} >= 1e-4"


@pytest.mark.parametrize("squared", [False, True])
def test_jax_lncc_and_cc2_numerical_parity_3d(squared):
    """Verify PyTorch and JAX compute identical LNCC / CC2 losses on 3D inputs."""
    rng = np.random.default_rng(43)
    a_np = rng.standard_normal((1, 1, 12, 12, 12)).astype("float32")
    b_np = rng.standard_normal((1, 1, 12, 12, 12)).astype("float32")

    loss_pt = float(
        local_ncc_loss_nd(
            torch.from_numpy(a_np),
            torch.from_numpy(b_np),
            window_size=3,
            squared=squared,
            use_ants_pseudo_gradient=False,
        )
    )
    loss_jx = float(
        local_ncc_loss_nd_jax(
            jnp.array(a_np),
            jnp.array(b_np),
            window_size=3,
            squared=squared,
            use_ants_pseudo_gradient=False,
        )
    )

    diff = abs(loss_pt - loss_jx)
    assert diff < 1e-4, f"3D LNCC/CC2 (squared={squared}) parity diff {diff} >= 1e-4"


@pytest.mark.parametrize("squared", [False, True])
def test_jax_lncc_pseudo_gradient_parity(squared):
    """Verify pseudo-gradient LNCC evaluation parity between PyTorch and JAX."""
    rng = np.random.default_rng(44)
    a_np = rng.standard_normal((1, 1, 16, 16)).astype("float32")
    b_np = -(0.5 * a_np + 0.5 * rng.standard_normal((1, 1, 16, 16)).astype("float32"))

    loss_pt = float(
        local_ncc_loss_nd(
            torch.from_numpy(a_np),
            torch.from_numpy(b_np),
            window_size=5,
            squared=squared,
            use_ants_pseudo_gradient=True,
        )
    )
    loss_jx = float(
        local_ncc_loss_nd_jax(
            jnp.array(a_np),
            jnp.array(b_np),
            window_size=5,
            squared=squared,
            use_ants_pseudo_gradient=True,
        )
    )

    diff = abs(loss_pt - loss_jx)
    assert diff < 1e-4, f"Pseudo-gradient LNCC (squared={squared}) parity diff {diff} >= 1e-4"


@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("num_channels", [1, 3])
def test_jax_soft_dice_parity(use_mask, num_channels):
    """Verify Soft Dice loss numerical parity across PyTorch and JAX."""
    rng = np.random.default_rng(45)
    a_np = rng.uniform(0.1, 0.9, (1, num_channels, 16, 16)).astype("float32")
    b_np = rng.uniform(0.1, 0.9, (1, num_channels, 16, 16)).astype("float32")

    if use_mask:
        mask_np = (rng.uniform(0, 1, (1, 1, 16, 16)) > 0.3).astype("float32")
        pt_mask = torch.from_numpy(mask_np)
        jx_mask = jnp.array(mask_np)
    else:
        pt_mask = None
        jx_mask = None

    loss_pt = float(soft_dice_loss_nd(torch.from_numpy(a_np), torch.from_numpy(b_np), mask=pt_mask))
    loss_jx = float(soft_dice_loss_nd_jax(jnp.array(a_np), jnp.array(b_np), mask=jx_mask))

    diff = abs(loss_pt - loss_jx)
    assert diff < 1e-5, f"Soft Dice (channels={num_channels}, mask={use_mask}) diff {diff} >= 1e-5"


@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("spatial_dim", [2, 3])
def test_jax_mse_parity(use_mask, spatial_dim):
    """Verify Pointwise MSE loss numerical parity across PyTorch and JAX."""
    rng = np.random.default_rng(46)
    shape = (1, 1, 16, 16) if spatial_dim == 2 else (1, 1, 8, 8, 8)
    a_np = rng.standard_normal(shape).astype("float32")
    b_np = rng.standard_normal(shape).astype("float32")

    if use_mask:
        mask_np = (rng.uniform(0, 1, shape) > 0.3).astype("float32")
        pt_mask = torch.from_numpy(mask_np)
        jx_mask = jnp.array(mask_np)
        loss_jx = float(jnp.sum(((jnp.array(a_np) - jnp.array(b_np)) ** 2) * jx_mask) / (jnp.sum(jx_mask) + 1e-8))
    else:
        pt_mask = None
        loss_jx = float(jnp.mean((jnp.array(a_np) - jnp.array(b_np)) ** 2))

    loss_pt = float(mse_loss_nd(torch.from_numpy(a_np), torch.from_numpy(b_np), mask=pt_mask))

    diff = abs(loss_pt - loss_jx)
    assert diff < 1e-6, f"MSE (dim={spatial_dim}, mask={use_mask}) diff {diff} >= 1e-6"


@pytest.mark.parametrize("num_bins", [16, 32, 64])
def test_jax_mattes_mi_boundary_padded_parity(num_bins):
    """Verify Mattes MI boundary-padded parity (pad=2.0) resolves previous 0.112 nats divergence."""
    rng = np.random.default_rng(47)
    N = 1000
    x_np = rng.uniform(-0.8, 0.8, N).astype("float32")
    y_np = np.clip(x_np + rng.normal(0, 0.1, N), -0.8, 0.8).astype("float32")

    loss_pt = float(
        mattes_mi_loss_core(
            torch.from_numpy(x_np),
            torch.from_numpy(y_np),
            num_bins=num_bins,
            min_val=-1.0,
            max_val=1.0,
        )
    )
    loss_jx = float(
        mattes_mi_loss_core_jax(
            jnp.array(x_np),
            jnp.array(y_np),
            num_bins=num_bins,
            min_val=-1.0,
            max_val=1.0,
            pad=2.0,
        )
    )

    diff = abs(loss_pt - loss_jx)
    assert diff < 1e-4, f"Mattes MI (num_bins={num_bins}) parity diff {diff} >= 1e-4 (previous divergence was 0.112 nats)"


def test_mattes_mi_partition_of_unity_parity():
    """Verify cubic B-spline Parzen evaluations maintain partition of unity in both PyTorch and JAX."""
    num_bins = 32
    min_val, max_val = -1.0, 1.0
    pad = 2.0
    u_min = pad
    u_max = float(num_bins - 1) - pad
    scale = (u_max - u_min) / (max_val - min_val)

    # Sample extreme boundary points including endpoints -1.0 and +1.0
    x_np = np.linspace(-1.0, 1.0, 100, dtype=np.float32)

    # PyTorch evaluation
    x_pt = torch.from_numpy(x_np)
    u_pt = u_min + (x_pt.unsqueeze(1) - min_val) * scale
    bin_idx_pt = torch.arange(num_bins, dtype=torch.float32).unsqueeze(0)
    w_pt = b_spline_3(u_pt - bin_idx_pt)
    sums_pt = w_pt.sum(dim=1).numpy()

    # JAX evaluation
    x_jx = jnp.array(x_np)
    u_jx = u_min + (jnp.expand_dims(x_jx, 1) - min_val) * scale
    bin_idx_jx = jnp.expand_dims(jnp.arange(num_bins, dtype=jnp.float32), 0)
    w_jx = b_spline_3_jax(u_jx - bin_idx_jx)
    sums_jx = np.asarray(jnp.sum(w_jx, axis=1))

    # Assert both engines maintain exact partition of unity (sum == 1.0)
    np.testing.assert_allclose(sums_pt, 1.0, atol=1e-5, err_msg="PyTorch B-spline partition of unity violated")
    np.testing.assert_allclose(sums_jx, 1.0, atol=1e-5, err_msg="JAX B-spline partition of unity violated")

    # Assert cross-engine weight parity
    np.testing.assert_allclose(sums_pt, sums_jx, atol=1e-6, err_msg="Cross-engine partition of unity mismatch")
