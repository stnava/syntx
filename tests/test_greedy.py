"""
Unit tests for syntx.greedy (unidirectional Eulerian compositive registration).
"""

import os
import pytest
import numpy as np
import torch
import ants
import syntx


@pytest.fixture
def sample_2d_images():
    """Load standard 2D test pair r16 / r64."""
    fi = ants.image_read(ants.get_ants_data('r16'))
    mi = ants.image_read(ants.get_ants_data('r64'))
    return fi, mi


@pytest.fixture
def sample_3d_images():
    """Create small synthetic 3D sphere images for fast unit testing."""
    shape = (32, 32, 32)
    fi_np = np.zeros(shape, dtype=np.float32)
    mi_np = np.zeros(shape, dtype=np.float32)

    # Sphere at center for fixed
    z, y, x = np.ogrid[:32, :32, :32]
    mask_f = ((x - 16)**2 + (y - 16)**2 + (z - 16)**2) <= 8**2
    fi_np[mask_f] = 1.0

    # Shifted sphere for moving
    mask_m = ((x - 14)**2 + (y - 18)**2 + (z - 15)**2) <= 8**2
    mi_np[mask_m] = 1.0

    fi = ants.from_numpy(fi_np, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    mi = ants.from_numpy(mi_np, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    return fi, mi


def test_greedy_2d_basic(sample_2d_images):
    """Test 2D greedy registration on r16 / r64."""
    fi, mi = sample_2d_images

    initial_mse = float(np.mean((fi.numpy() - mi.numpy()) ** 2))

    res = syntx.greedy(
        fixed=fi,
        moving=mi,
        reg_iterations=[25, 15],
        scales=[2, 1],
        learning_rate=0.4,
        verbose=False,
    )

    assert 'warpedmovout' in res
    assert 'fwdtransforms' in res
    assert 'model' in res
    assert 'provenance' in res

    warped = res['warpedmovout']
    assert isinstance(warped, ants.ANTsImage)
    assert warped.shape == fi.shape

    # Check forward transform exists on disk
    fwd_tx = res['fwdtransforms'][0]
    assert os.path.exists(fwd_tx)

    # Alignment should improve MSE
    final_mse = float(np.mean((fi.numpy() - warped.numpy()) ** 2))
    assert final_mse < initial_mse, f"Expected MSE improvement, got initial={initial_mse:.4f}, final={final_mse:.4f}"

    # Verify provenance dictionary
    prov = res['provenance']
    assert prov['algorithm'] == 'syntx.greedy'
    assert prov['formulation'] == 'eulerian_compositive'
    assert 'runtime_total_sec' in prov


def test_greedy_3d_synthetic(sample_3d_images):
    """Test 3D greedy registration on small synthetic volumes."""
    fi, mi = sample_3d_images

    initial_corr = float(np.corrcoef(fi.numpy().flatten(), mi.numpy().flatten())[0, 1])

    res = syntx.greedy(
        fixed=fi,
        moving=mi,
        reg_iterations=[15, 10],
        scales=[2, 1],
        learning_rate=0.4,
        initial_transform=False,  # Test identity initial transform
        verbose=False,
    )

    warped = res['warpedmovout']
    assert warped.shape == fi.shape

    final_corr = float(np.corrcoef(fi.numpy().flatten(), warped.numpy().flatten())[0, 1])
    assert final_corr > initial_corr, f"Expected correlation increase, got initial={initial_corr:.4f}, final={final_corr:.4f}"


def test_greedy_delegation_via_syn(sample_2d_images):
    """Test calling syntx.syn with type_of_transform='greedy'."""
    fi, mi = sample_2d_images

    res = syntx.syn(
        fixed=fi,
        moving=mi,
        type_of_transform='greedy',
        reg_iterations=[10, 5],
        scales=[2, 1],
        verbose=False,
    )

    assert 'warpedmovout' in res
    assert res['provenance']['algorithm'] == 'syntx.greedy'


def test_greedy_user_initial_transform(sample_2d_images):
    """Test passing an explicit initial transform to syntx.greedy."""
    fi, mi = sample_2d_images

    # Pre-compute affine transform
    aff = syntx.robust_affine(fi, mi, mode='auto', verbose=False)
    aff_tx = aff['fwdtransforms'][0]

    res = syntx.greedy(
        fixed=fi,
        moving=mi,
        initial_transform=aff_tx,
        reg_iterations=[10, 5],
        scales=[2, 1],
        verbose=False,
    )

    assert 'warpedmovout' in res
    assert res['provenance']['runtime_affine_sec'] >= 0.0
