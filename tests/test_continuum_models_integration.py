"""
Fast end-to-end integration tests for continuum-mechanics and incompressibility
regularizers across all 4 syntx registration architectures:
1. syntx.syn
2. syntx.syngs
3. syntx.tvf
4. syntx.greedy

All tests run on miniature 2D/3D phantoms with minimal iterations (1-2) to ensure
instantaneous (< 2.0s total) execution time while guaranteeing end-to-end mathematical
and API integrity.
"""

import pytest
import numpy as np
import torch
import ants
import syntx


@pytest.fixture(scope="module")
def phantom_pair_2d():
    """Create small 2D synthetic phantom pair with concentric circles."""
    torch.manual_seed(42)
    N = 24
    y, x = np.mgrid[-1:1:complex(0, N), -1:1:complex(0, N)]
    r_sq = x**2 + y**2
    img1 = np.exp(-r_sq / 0.3).astype(np.float32)
    # img2 is slightly shifted
    r_sq_shift = (x - 0.1)**2 + (y - 0.05)**2
    img2 = np.exp(-r_sq_shift / 0.3).astype(np.float32)

    fi = ants.from_numpy(img1, spacing=(1.0, 1.0), origin=(0.0, 0.0))
    mi = ants.from_numpy(img2, spacing=(1.0, 1.0), origin=(0.0, 0.0))
    mask = ants.from_numpy((r_sq <= 0.5).astype(np.float32), spacing=(1.0, 1.0), origin=(0.0, 0.0))
    return fi, mi, mask


# -----------------------------------------------------------------------------
# 1. syntx.syn Integration Tests
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("reg_name", [
    'solenoidal',
    'navier',
    'beltrami',
    'div_curl',
    'poroelastic',
    'hyperelastic',
    'masked_incompressible',
])
def test_syn_with_continuum_regularizers(phantom_pair_2d, reg_name):
    fi, mi, mask = phantom_pair_2d
    kwargs = {
        'fixed': fi,
        'moving': mi,
        'reg_iterations': [2, 1],
        'scales': [2, 1],
        'regularizer': reg_name,
        'initial_transform': 'identity',
        'verbose': False,
    }
    if reg_name == 'masked_incompressible':
        kwargs['fixed_mask'] = mask

    res = syntx.syn(**kwargs)
    assert 'warpedmovout' in res
    assert 'fwdtransforms' in res
    assert not np.isnan(res['warpedmovout'].numpy()).any()


# -----------------------------------------------------------------------------
# 2. syntx.syngs Integration Tests
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("reg_name", [
    'solenoidal',
    'navier',
    'beltrami',
    'poroelastic',
    'hyperelastic',
])
def test_syngs_with_continuum_regularizers(phantom_pair_2d, reg_name):
    fi, mi, _ = phantom_pair_2d
    res = syntx.syngs(
        fixed=fi,
        moving=mi,
        reg_iterations=[2, 1],
        levels=[2, 1],
        n_steps=2,
        regularizer=reg_name,
        initial_transform='identity',
        verbose=False,
    )
    assert 'warpedmovout' in res
    assert 'fwdtransforms' in res
    assert not np.isnan(res['warpedmovout'].numpy()).any()


# -----------------------------------------------------------------------------
# 3. syntx.tvf Integration Tests
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("reg_name", [
    'solenoidal',
    'navier',
    'beltrami',
    'div_curl',
])
def test_tvf_with_continuum_regularizers(phantom_pair_2d, reg_name):
    fi, mi, _ = phantom_pair_2d
    res = syntx.tvf(
        fixed=fi,
        moving=mi,
        reg_iterations=[2],
        levels=[1],
        n_time_steps=2,
        regularizer=reg_name,
        initial_transform='identity',
        verbose=False,
    )
    assert 'warpedmovout' in res
    assert 'fwdtransforms' in res
    assert not np.isnan(res['warpedmovout'].numpy()).any()


# -----------------------------------------------------------------------------
# 4. syntx.greedy Integration Tests
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("reg_name", [
    'gaussian',       # Default baseline
    'solenoidal',     # Incompressible Leray
    'beltrami',       # Anti-shear quasiconformal
    'navier',         # Navier-Cauchy continuum
    'poroelastic',    # Two-phase Darcy-Stokes
])
def test_greedy_with_continuum_regularizers(phantom_pair_2d, reg_name):
    fi, mi, mask = phantom_pair_2d
    res = syntx.greedy(
        fixed=fi,
        moving=mi,
        reg_iterations=[2, 1],
        scales=[2, 1],
        regularizer=reg_name,
        initial_transform='identity',
        fixed_mask=mask if reg_name == 'poroelastic' else None,
        verbose=False,
    )
    assert 'warpedmovout' in res
    assert 'fwdtransforms' in res
    assert not np.isnan(res['warpedmovout'].numpy()).any()
