import pytest
import math
import numpy as np
import torch
import torch.nn.functional as F
import jax
import jax.numpy as jnp
import ants

from syntx.syn import (
    get_rotation_matrix,
    TriPlanarVGG3DLoss,
    HierarchicalAffine,
    separable_gaussian_filter,
    local_ncc_loss_nd,
    mattes_mi_loss_core,
    compute_physical_jacobian_determinant,
    compute_jacobian_determinant_nd,
    registration,
    check_convergence
)

from syntx.syn_jax import (
    get_rotation_matrix_jax,
    get_affine_matrix_jax,
    separable_gaussian_filter_jax,
    local_ncc_loss_nd_jax,
    mattes_mi_loss_core_jax,
    mattes_mi_loss_nd_jax,
    compute_physical_jacobian_determinant_jax,
    compute_jacobian_determinant_nd_jax,
    check_convergence as check_convergence_jax,
    to_jax_array,
    upscale_field_jax
)


def get_test_image_2d():
    vol = np.zeros((32, 32), dtype=float)
    y, x = np.ogrid[:32, :32]
    mask1 = (x - 16)**2 + (y - 16)**2 < 8**2
    vol[mask1] = 10
    img = ants.from_numpy(vol, spacing=(1, 1), origin=(0, 0))
    return ants.smooth_image(img, 1.0)


def test_get_rotation_matrix():
    # 2D
    r2 = get_rotation_matrix(torch.tensor([0.1]), 2)
    assert r2.shape == (2, 2)

    # 3D zero
    r3_zero = get_rotation_matrix(torch.zeros(3), 3)
    assert torch.allclose(r3_zero, torch.eye(3))

    # 3D non-zero
    r3_nonzero = get_rotation_matrix(torch.tensor([0.1, 0.2, 0.3]), 3)
    assert r3_nonzero.shape == (3, 3)

    # Invalid dimension
    with pytest.raises(ValueError, match="Only 2D and 3D are supported."):
        get_rotation_matrix(torch.zeros(4), 4)


def test_get_rotation_matrix_jax():
    # 2D
    r2 = get_rotation_matrix_jax(jnp.array([0.1]), 2)
    assert r2.shape == (2, 2)

    # 3D zero
    r3_zero = get_rotation_matrix_jax(jnp.zeros(3), 3)
    assert jnp.allclose(r3_zero, jnp.eye(3))

    # 3D non-zero
    r3_nonzero = get_rotation_matrix_jax(jnp.array([0.1, 0.2, 0.3]), 3)
    assert r3_nonzero.shape == (3, 3)

    # Invalid dimension
    with pytest.raises(ValueError, match="Only 2D and 3D are supported."):
        get_rotation_matrix_jax(jnp.zeros(4), 4)


def test_triplanar_vgg3d_loss():
    img_in = torch.randn(1, 1, 16, 16, 16)
    img_tg = torch.randn(1, 1, 16, 16, 16)

    # mode='lncc_3d'
    loss_fn = TriPlanarVGG3DLoss(dim=3, feature_layers=[2], mode='lncc_3d')
    loss = loss_fn(img_in, img_tg)
    assert loss.ndim == 0

    # mode='lncc'
    loss_fn = TriPlanarVGG3DLoss(dim=3, feature_layers=[2], mode='lncc', num_slices=2)
    loss = loss_fn(img_in, img_tg)
    assert loss.ndim == 0

    # mode='patch_grid'
    loss_fn = TriPlanarVGG3DLoss(dim=3, feature_layers=[2], mode='patch_grid', patch_size=8, num_patches=2, num_slices=2)
    loss = loss_fn(img_in, img_tg)
    assert loss.ndim == 0

    # mode='patch_walk'
    loss_fn = TriPlanarVGG3DLoss(dim=3, feature_layers=[2], mode='patch_walk', patch_size=8, num_patches=2, num_slices=2)
    loss = loss_fn(img_in, img_tg)
    assert loss.ndim == 0


def test_hierarchical_affine():
    # Translation
    ha_trans = HierarchicalAffine(dim=3, transform_type='Translation')
    t_matrix = ha_trans.get_matrix()
    assert t_matrix.shape == (4, 4)

    # Rigid
    ha_rigid = HierarchicalAffine(dim=3, transform_type='Rigid')
    r_matrix = ha_rigid.get_matrix()
    assert r_matrix.shape == (4, 4)

    # Similarity
    ha_sim = HierarchicalAffine(dim=3, transform_type='Similarity')
    s_matrix = ha_sim.get_matrix()
    assert s_matrix.shape == (4, 4)


def test_get_affine_matrix_jax():
    params = {
        'translation': jnp.zeros(3),
        'omega': jnp.zeros(3),
        'scale': jnp.ones(1),
        'anisotropic_scale': jnp.ones(3),
        'shear': jnp.zeros(3)
    }
    T_affine = get_affine_matrix_jax(params, 3, 'Affine')
    T_other = get_affine_matrix_jax(params, 3, 'Rigid')
    assert T_affine.shape == (4, 4)
    assert T_other.shape == (4, 4)

    params_2d = {
        'translation': jnp.zeros(2),
        'omega': jnp.zeros(1),
        'scale': jnp.ones(1),
        'anisotropic_scale': jnp.ones(2),
        'shear': jnp.zeros(1)
    }
    T_affine_2d = get_affine_matrix_jax(params_2d, 2, 'Affine')
    assert T_affine_2d.shape == (3, 3)

    params_invalid = {
        'translation': jnp.zeros(4),
        'omega': jnp.zeros(6),
        'scale': jnp.ones(1),
        'anisotropic_scale': jnp.ones(4),
        'shear': jnp.zeros(6)
    }
    with pytest.raises(ValueError, match="Only 2D and 3D are supported."):
        get_affine_matrix_jax(params_invalid, 4, 'Affine')


def test_separable_gaussian_filter():
    grid = torch.randn(1, 8, 8, 2)
    # sigma <= 0.0
    res_zero = separable_gaussian_filter(grid, 0.0)
    assert torch.allclose(res_zero, grid)

    # spacing causing sig <= 0.0
    res_spacing = separable_gaussian_filter(grid, 1.0, spacing=[1.0, -1.0])
    assert res_spacing.shape == grid.shape


def test_separable_gaussian_filter_jax():
    grid = jnp.zeros((1, 8, 8, 2))
    # sigma <= 0.0
    res_zero = separable_gaussian_filter_jax(grid, 0.0)
    assert jnp.allclose(res_zero, grid)

    # spacing causing sig <= 0.0
    res_spacing = separable_gaussian_filter_jax(grid, 1.0, spacing=[1.0, -1.0])
    assert res_spacing.shape == grid.shape


def test_local_ncc_loss():
    # 3D pool
    i3 = torch.randn(1, 1, 8, 8, 8)
    j3 = torch.randn(1, 1, 8, 8, 8)
    l3 = local_ncc_loss_nd(i3, j3, window_size=5)
    assert l3.ndim == 0

    # Mask
    mask = torch.ones(1, 1, 8, 8, 8)
    l_mask = local_ncc_loss_nd(i3, j3, mask=mask, window_size=5)
    assert l_mask.ndim == 0

    # Invalid dim ValueError
    i1 = torch.randn(1, 1, 8)
    j1 = torch.randn(1, 1, 8)
    with pytest.raises(ValueError, match="Only 2D and 3D images are supported"):
        local_ncc_loss_nd(i1, j1)


def test_local_ncc_loss_jax():
    # 3D pool
    i3 = jnp.zeros((1, 1, 8, 8, 8))
    j3 = jnp.zeros((1, 1, 8, 8, 8))
    l3 = local_ncc_loss_nd_jax(i3, j3)
    assert l3.ndim == 0

    # Mask
    mask = jnp.ones((1, 1, 8, 8, 8))
    l_mask = local_ncc_loss_nd_jax(i3, j3, mask=mask)
    assert l_mask.ndim == 0


def test_mattes_mi_loss_core():
    i2 = torch.randn(1, 1, 8, 8)
    j2 = torch.randn(1, 1, 8, 8)

    # mask
    mask = torch.ones(1, 1, 8, 8)
    l_mask = mattes_mi_loss_core(i2, j2, mask=mask)
    assert l_mask.ndim == 0

    # sampling_percentage
    l_sample = mattes_mi_loss_core(i2, j2, sampling_percentage=0.5)
    assert l_sample.ndim == 0

    # empty (using mask of zeros)
    mask_zero = torch.zeros(1, 1, 8, 8)
    l_empty = mattes_mi_loss_core(i2, j2, mask=mask_zero)
    assert l_empty.ndim == 0
    assert l_empty.item() == 0.0


def test_mattes_mi_loss_core_jax():
    i2 = jnp.zeros((1, 1, 8, 8))
    j2 = jnp.zeros((1, 1, 8, 8))

    # mask
    mask = jnp.ones((1, 1, 8, 8))
    l_mask = mattes_mi_loss_core_jax(i2, j2, mask=mask)
    assert l_mask.ndim == 0

    # sampling_percentage
    l_sample = mattes_mi_loss_core_jax(i2, j2, sampling_percentage=0.5)
    assert l_sample.ndim == 0

    # empty (using mask of zeros)
    mask_zero = jnp.zeros((1, 1, 8, 8))
    l_empty = mattes_mi_loss_core_jax(i2, j2, mask=mask_zero)
    assert l_empty.ndim == 0
    assert l_empty == 0.0


def test_compute_physical_jacobian_determinant():
    # 2D (covers lines 730-734)
    warp_2d = torch.zeros(1, 8, 8, 2)
    dir_2d = torch.eye(2)
    spacing_2d = torch.ones(2)
    jac_2d = compute_physical_jacobian_determinant(warp_2d, dir_2d, spacing_2d)
    assert jac_2d.shape == (1, 8, 8)

    # 4D (covers line 747)
    warp_4d = torch.zeros(1, 8, 8, 8, 8, 4)
    dir_4d = torch.eye(4)
    spacing_4d = torch.ones(4)
    jac_4d = compute_physical_jacobian_determinant(warp_4d, dir_4d, spacing_4d)
    assert jac_4d.shape == (1, 8, 8, 8, 8)


def test_compute_physical_jacobian_determinant_jax():
    # 2D
    warp_2d = jnp.zeros((1, 8, 8, 2))
    dir_2d = jnp.eye(2)
    spacing_2d = jnp.ones(2)
    jac_2d = compute_physical_jacobian_determinant_jax(warp_2d, dir_2d, spacing_2d)
    assert jac_2d.shape == (1, 8, 8)

    # 4D
    warp_4d = jnp.zeros((1, 8, 8, 8, 8, 4))
    dir_4d = jnp.eye(4)
    spacing_4d = jnp.ones(4)
    jac_4d = compute_physical_jacobian_determinant_jax(warp_4d, dir_4d, spacing_4d)
    assert jac_4d.shape == (1, 8, 8, 8, 8)


def test_to_jax_array():
    # PyTorch
    x_pt = torch.tensor([1.0, 2.0])
    res_pt = to_jax_array(x_pt)
    assert isinstance(res_pt, jax.Array)

    # Numpy
    x_np = np.array([1.0, 2.0])
    res_np = to_jax_array(x_np)
    assert isinstance(res_np, jax.Array)

    # List
    x_list = [1.0, 2.0]
    res_list = to_jax_array(x_list)
    assert isinstance(res_list, jax.Array)


def test_upscale_field_jax():
    field = jnp.zeros((1, 8, 8, 2))
    res = upscale_field_jax(field, (16, 16))
    assert res.shape == (1, 16, 16, 2)


def test_check_convergence():
    assert not check_convergence([1.0, 0.9])
    assert check_convergence([1.0] * 20, window_size=5, slope_threshold=1e-3)


def test_check_convergence_jax():
    assert not check_convergence_jax([1.0, 0.9])
    assert check_convergence_jax([1.0] * 20, window_size=5, slope_threshold=1e-3)


def test_registration_options(tmp_path):
    import os
    fixed = get_test_image_2d()
    moving = get_test_image_2d()

    # Backend unknown
    with pytest.raises(ValueError, match="Unknown backend"):
        registration(fixed, moving, backend='unknown')

    # Rigid transform
    res_rigid = registration(
        fixed, moving,
        type_of_transform='Rigid',
        backend='pytorch',
        levels=[2, 1],
        affine_iterations=[2, 1],
        reg_iterations=[0, 0],
        inverse_method='fixed_point'
    )
    assert 'fwdtransforms' in res_rigid

    # Translation transform
    res_trans = registration(
        fixed, moving,
        type_of_transform='Translation',
        backend='pytorch',
        levels=[2, 1],
        affine_iterations=[2, 1],
        reg_iterations=[0, 0]
    )
    assert 'fwdtransforms' in res_trans

    # Affine transform (linear only)
    res_affine = registration(
        fixed, moving,
        type_of_transform='Affine',
        backend='pytorch',
        levels=[2, 1],
        affine_iterations=[2, 1],
        reg_iterations=[0, 0]
    )
    assert 'fwdtransforms' in res_affine

    # Initial transform (list)
    tx = ants.create_ants_transform(transform_type='Euler2DTransform', dimension=2, translation=(0.5, 0.5))
    tx_path = os.path.join(tmp_path, "init_tx.mat")
    ants.write_transform(tx, tx_path)

    res_init = registration(
        fixed, moving,
        type_of_transform='SyNTo',
        backend='pytorch',
        levels=[2, 1],
        affine_iterations=[2, 1],
        reg_iterations=[2, 1],
        initial_transform=[tx_path]
    )
    assert 'fwdtransforms' in res_init

    # Initial transform (single object)
    res_init_single = registration(
        fixed, moving,
        type_of_transform='SyNTo',
        backend='pytorch',
        levels=[2, 1],
        affine_iterations=[2, 1],
        reg_iterations=[2, 1],
        initial_transform=tx_path
    )
    assert 'fwdtransforms' in res_init_single


def test_registration_options_jax(tmp_path):
    import os
    fixed = get_test_image_2d()
    moving = get_test_image_2d()

    # Rigid JAX
    res_rigid = registration(
        fixed, moving,
        type_of_transform='Rigid',
        backend='jax',
        levels=[2, 1],
        affine_iterations=[2, 1],
        reg_iterations=[0, 0],
        inverse_method='fixed_point'
    )
    assert 'fwdtransforms' in res_rigid

    # Translation JAX
    res_trans = registration(
        fixed, moving,
        type_of_transform='Translation',
        backend='jax',
        levels=[2, 1],
        affine_iterations=[2, 1],
        reg_iterations=[0, 0]
    )
    assert 'fwdtransforms' in res_trans

    # Affine JAX (linear only)
    res_affine = registration(
        fixed, moving,
        type_of_transform='Affine',
        backend='jax',
        levels=[2, 1],
        affine_iterations=[2, 1],
        reg_iterations=[0, 0]
    )
    assert 'fwdtransforms' in res_affine

    # Initial transform JAX
    tx = ants.create_ants_transform(transform_type='Euler2DTransform', dimension=2, translation=(0.5, 0.5))
    tx_path = os.path.join(tmp_path, "init_tx_jax.mat")
    ants.write_transform(tx, tx_path)

    res_init = registration(
        fixed, moving,
        type_of_transform='SyNTo',
        backend='jax',
        levels=[2, 1],
        affine_iterations=[2, 1],
        reg_iterations=[2, 1],
        initial_transform=tx_path
    )
    assert 'fwdtransforms' in res_init


def test_check_convergence_denom_zero():
    from syntx.syn_jax import check_convergence as check_convergence_jax
    assert not check_convergence_jax([1.0, 1.0], window_size=1)


def test_inverse_field_smoothing():
    from syntx.syn import update_inverse_field_nd
    from syntx.syn_jax import update_inverse_field_nd_jax
    
    # PyTorch
    field = torch.zeros(1, 8, 8, 2)
    W_inv = torch.zeros(1, 8, 8, 2)
    res = update_inverse_field_nd(field, W_inv, steps=2, smoothing_sigma=1.0)
    assert res.shape == (1, 8, 8, 2)

    # JAX
    field_jax = jnp.zeros((1, 8, 8, 2))
    W_inv_jax = jnp.zeros((1, 8, 8, 2))
    res_jax = update_inverse_field_nd_jax(field_jax, W_inv_jax, steps=2, smoothing_sigma=1.0)
    assert res_jax.shape == (1, 8, 8, 2)


def test_spatial_jacobian_1d():
    from syntx.syn_jax import _spatial_jacobian_nd_jax
    field = jnp.zeros((1, 8, 1))
    res = _spatial_jacobian_nd_jax(field)
    assert res.shape == (1, 8, 1, 1)


def test_mattes_mi_loss_nd_jax():
    i2 = jnp.zeros((1, 1, 8, 8))
    j2 = jnp.zeros((1, 1, 8, 8))
    l = mattes_mi_loss_nd_jax(i2, j2)
    assert l.ndim == 0


def test_compute_jacobian_determinant_nd_extra():
    # PyTorch 3D with physical_spacing
    warp_3d = torch.zeros(1, 8, 8, 8, 3)
    res_3d = compute_jacobian_determinant_nd(warp_3d, physical_spacing=[1.0, 1.0, 1.0])
    assert res_3d.shape == (1, 8, 8, 8)

    # PyTorch invalid dim
    warp_4d = torch.zeros(1, 8, 8, 8, 8, 4)
    with pytest.raises(ValueError, match="Only 2D and 3D are supported"):
        compute_jacobian_determinant_nd(warp_4d)

    # JAX 3D with physical_spacing
    warp_3d_jax = jnp.zeros((1, 8, 8, 8, 3))
    res_3d_jax = compute_jacobian_determinant_nd_jax(warp_3d_jax, physical_spacing=[1.0, 1.0, 1.0])
    assert res_3d_jax.shape == (1, 8, 8, 8)

    # JAX invalid dim
    warp_4d_jax = jnp.zeros((1, 8, 8, 8, 8, 4))
    with pytest.raises(ValueError, match="Only 2D and 3D are supported"):
        compute_jacobian_determinant_nd_jax(warp_4d_jax)

    # JAX 1D (covers line 514)
    warp_1d_jax = jnp.zeros((1, 8, 1))
    with pytest.raises(ValueError, match="Only 2D and 3D are supported"):
        compute_jacobian_determinant_nd_jax(warp_1d_jax)
