"""
Unit tests targeting high code coverage for syntx.syn, syntx.syn_jax, syntx.tvf, and syntx.tvf_jax.
"""

import os
import numpy as np
import torch
import pytest
import ants

from syntx.syn import (
    HierarchicalAffine,
    SyNTo,
    local_ncc_loss_nd,
    mattes_mi_loss_nd,
    compute_physical_jacobian_determinant,
    calculate_inverse_identity_error,
    separable_gaussian_filter,
    compose_grids,
    get_physical_grid_torch,
    physical_to_normalized_torch_cached,
    registration as syn_registration
)

from syntx.tvf import (
    TVFModel,
    tvf_registration
)

try:
    from syntx.syn_jax import SyNTo as SyNToJAX
    from syntx.tvf_jax import TVFModelJAX
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def test_hierarchical_affine_all_types():
    for ttype in ['Translation', 'Rigid', 'Similarity', 'Affine']:
        aff = HierarchicalAffine(dim=2, transform_type=ttype)
        aff.translation.data.fill_(1.0)
        aff.clamp_parameters()
        mat = aff.get_matrix()
        assert mat.shape == (3, 3)

        aff3d = HierarchicalAffine(dim=3, transform_type=ttype)
        aff3d.clamp_parameters()
        mat3d = aff3d.get_matrix()
        assert mat3d.shape == (4, 4)


def test_local_ncc_and_ants_pseudo_gradient():
    I = torch.randn(1, 1, 16, 16, requires_grad=True)
    J = torch.randn(1, 1, 16, 16, requires_grad=True)
    mask = torch.ones(1, 1, 16, 16)

    # Standard LNCC with mask
    loss_mask = local_ncc_loss_nd(I, J, mask=mask, window_size=5)
    assert isinstance(loss_mask, torch.Tensor)

    # ANTs Pseudo Gradient LNCC
    loss_ants = local_ncc_loss_nd(I, J, mask=mask, window_size=5, use_ants_pseudo_gradient=True)
    assert isinstance(loss_ants, torch.Tensor)
    loss_ants.backward()


def test_synto_model_2d_advanced_options():
    device = 'cpu'
    model = SyNTo(
        dim=2,
        grid_shape=(16, 16),
        spacing=[1.0, 1.0],
        origin=[0.0, 0.0],
        fluid_sigma=2.0,
        elastic_sigma=0.5,
        boundary_suppression_thresh=0.05,
        image_grad_clip=5.0,
        antisymmetric=True
    ).to(device)

    fixed = torch.randn(1, 1, 16, 16, device=device)
    moving = torch.randn(1, 1, 16, 16, device=device)

    # Fit with Sobolev regularizer and composite similarity metric
    model.fit(
        fixed, moving,
        levels=[1],
        epochs_per_level=[2],
        affine_epochs=1,
        similarity_metric='lncc',
        regularizer='sobolev',
        sobolev_alpha=0.1,
        verbose=False
    )

    fwd_warp = model.warp_l2r
    inv_warp = model.warp_l2r_inv
    assert fwd_warp.shape == (1, 16, 16, 2)
    assert inv_warp.shape == (1, 16, 16, 2)

    # Test Jacobian determinant and inverse identity error computation
    jac_det = compute_physical_jacobian_determinant(fwd_warp, direction=model.direction, spacing=model.spacing)
    assert jac_det.shape == (1, 16, 16)

    inv_err = calculate_inverse_identity_error(
        fwd_warp, inv_warp,
        spacing=model.spacing, origin=model.origin, direction=model.direction
    )
    assert isinstance(inv_err, dict)
    assert inv_err['error_map'].shape == (16, 16)


def test_tvf_model_advanced_options():
    device = 'cpu'
    model = TVFModel(
        dim=2,
        image_shape=(16, 16),
        velocity_shape=(16, 16),
        spacing=[1.0, 1.0],
        origin=[0.0, 0.0],
        fluid_sigma=1.0,
        elastic_sigma=0.05,
        n_time_steps=3
    ).to(device)

    fixed = torch.randn(1, 1, 16, 16, device=device)
    moving = torch.randn(1, 1, 16, 16, device=device)

    # Fit with physical sigma mode, multipoint loss, and Sobolev regularizer
    model.fit(
        fixed, moving,
        levels=[1],
        epochs_per_level=[2],
        affine_epochs=1,
        optimizer_type='lars',
        sigma_mode='physical',
        regularizer='sobolev',
        sobolev_alpha=0.1,
        multipoint_loss=[0.0, 0.5, 1.0],
        fast_smooth=True,
        verbose=False
    )

    fwd_warp = model.get_forward_warp()
    inv_warp = model.get_inverse_warp()
    assert fwd_warp.shape == (1, 16, 16, 2)
    assert inv_warp.shape == (1, 16, 16, 2)


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_syn_jax_and_tvf_jax_advanced():
    import jax.numpy as jnp

    model_syn_jax = SyNToJAX(
        dim=2,
        grid_shape=(16, 16),
        spacing=[1.0, 1.0],
        origin=[0.0, 0.0],
        fluid_sigma=1.0,
        elastic_sigma=0.0
    )

    I_jax = jnp.ones((1, 1, 16, 16), dtype=jnp.float32)
    J_jax = jnp.ones((1, 1, 16, 16), dtype=jnp.float32)

    model_syn_jax.fit(
        I_jax, J_jax,
        levels=[1],
        epochs_per_level=[2],
        affine_epochs=1,
        verbose=False
    )
    assert model_syn_jax.warp_l2r is not None

    model_tvf_jax = TVFModelJAX(
        dim=2,
        image_shape=(16, 16),
        velocity_shape=(16, 16),
        spacing=[1.0, 1.0],
        origin=[0.0, 0.0],
        n_time_steps=3
    )

    model_tvf_jax.fit(
        I_jax, J_jax,
        levels=[1],
        epochs_per_level=[2],
        affine_epochs=1,
        verbose=False
    )
    assert model_tvf_jax.get_forward_warp() is not None
