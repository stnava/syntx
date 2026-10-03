"""
Tests for package-wide interface consistency:
- `levels` argument across all registration models (robust_affine, syn, tvf, syngs, greedy, auto_reg)
- `iterations` and `reg_iterations` interoperability across all registration models
- Affine-only delegation through auto_reg and syn with custom levels and iterations
"""

import pytest
import numpy as np
import torch
import ants
import syntx


@pytest.fixture(scope="module")
def small_pair_2d():
    """Lightweight 2D pair for rapid interface testing."""
    np.random.seed(42)
    # 64x64 phantom with a centered circle
    shape = (64, 64)
    y, x = np.ogrid[:shape[0], :shape[1]]
    center = (32, 32)
    mask1 = ((x - center[1]) ** 2 + (y - center[0]) ** 2) < 16 ** 2
    img1 = mask1.astype(np.float32)

    # Shifted and slightly scaled circle
    center2 = (35, 30)
    mask2 = ((x - center2[1]) ** 2 / 1.1 + (y - center2[0]) ** 2 * 1.1) < 16 ** 2
    img2 = mask2.astype(np.float32)

    fi = ants.from_numpy(img1, spacing=(1.0, 1.0), origin=(0.0, 0.0))
    mi = ants.from_numpy(img2, spacing=(1.0, 1.0), origin=(0.0, 0.0))
    return fi, mi


def test_robust_affine_levels_and_iterations(small_pair_2d):
    fi, mi = small_pair_2d

    # 1. Custom levels and iterations
    aff = syntx.robust_affine(fi, mi, levels=[4, 2], iterations=[15, 10], verbose=False)
    assert 'fwdtransforms' in aff
    assert 'warpedmovout' in aff
    assert len(aff['fwdtransforms']) == 1

    # 2. reg_iterations alias in robust_affine
    aff_alias = syntx.robust_affine(fi, mi, levels=[4, 2], reg_iterations=[10, 5], verbose=False)
    assert 'fwdtransforms' in aff_alias

    # 3. Rigid dof with custom levels
    aff_rigid = syntx.robust_affine(fi, mi, levels=[4, 2], iterations=[10, 5], dof='rigid', verbose=False)
    assert 'fwdtransforms' in aff_rigid


def test_syn_iterations_alias(small_pair_2d):
    fi, mi = small_pair_2d
    # 1. Test syn with iterations alias (instead of reg_iterations)
    res = syntx.syn(fi, mi, levels=[4, 2], iterations=[5, 2], verbose=False)
    assert 'warpedmovout' in res
    assert 'fwdtransforms' in res

    # 2. Test syn with reg_iterations
    res2 = syntx.syn(fi, mi, levels=[4, 2], reg_iterations=[5, 2], verbose=False)
    assert 'warpedmovout' in res2
    assert 'fwdtransforms' in res2


def test_tvf_iterations_alias(small_pair_2d):
    fi, mi = small_pair_2d
    # 1. Test tvf with iterations alias and custom levels
    res = syntx.tvf(fi, mi, levels=[4, 2], iterations=[5, 2], n_time_steps=1, verbose=False)
    assert 'warpedmovout' in res
    assert 'fwdtransforms' in res

    # 2. Test tvf with reg_iterations and custom levels
    res2 = syntx.tvf(fi, mi, levels=[4, 2], reg_iterations=[5, 2], n_time_steps=1, verbose=False)
    assert 'warpedmovout' in res2
    assert 'fwdtransforms' in res2


def test_syngs_iterations_alias(small_pair_2d):
    fi, mi = small_pair_2d
    # 1. Test syngs with iterations alias and custom levels
    res = syntx.syngs(fi, mi, levels=[4, 2], iterations=[5, 2], verbose=False)
    assert 'warpedmovout' in res
    assert 'fwdtransforms' in res

    # 2. Test syngs with reg_iterations and custom levels
    res2 = syntx.syngs(fi, mi, levels=[4, 2], reg_iterations=[5, 2], verbose=False)
    assert 'warpedmovout' in res2
    assert 'fwdtransforms' in res2


def test_greedy_iterations_alias(small_pair_2d):
    fi, mi = small_pair_2d
    # 1. Test greedy with levels and iterations alias
    res = syntx.greedy(fi, mi, levels=[4, 2], iterations=[5, 2], verbose=False)
    assert 'warpedmovout' in res
    assert 'fwdtransforms' in res

    # 2. Test greedy with levels and reg_iterations
    res2 = syntx.greedy(fi, mi, levels=[4, 2], reg_iterations=[5, 2], verbose=False)
    assert 'warpedmovout' in res2
    assert 'fwdtransforms' in res2


def test_auto_reg_consistency(small_pair_2d):
    fi, mi = small_pair_2d

    # 1. auto_reg with type_of_transform='Affine' and custom levels + iterations
    res_aff = syntx.auto_reg(fi, mi, type_of_transform='Affine', levels=[4, 2], iterations=[10, 5], diagnose=False)
    assert 'warpedmovout' in res_aff
    assert 'fwdtransforms' in res_aff

    # 2. auto_reg with type_of_transform='SyN' and custom levels + reg_iterations
    res_syn = syntx.auto_reg(fi, mi, type_of_transform='SyN', levels=[4, 2], reg_iterations=[5, 2], diagnose=False)
    assert 'warpedmovout' in res_syn
    assert 'fwdtransforms' in res_syn

    # 3. auto_reg with type_of_transform='TVF' and custom levels + iterations
    res_tvf = syntx.auto_reg(fi, mi, type_of_transform='TVF', levels=[4, 2], iterations=[5, 2], n_time_steps=1, diagnose=False)
    assert 'warpedmovout' in res_tvf
    assert 'fwdtransforms' in res_tvf


def test_syn_scattered_interface():
    pts_fix = torch.tensor([[0.0, 0.0], [0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32)
    pts_mov = torch.tensor([[0.1, 0.1], [0.6, 0.4], [-0.4, -0.6]], dtype=torch.float32)

    # 1. Test syn_scattered with reg_iterations alias and levels
    res = syntx.syn_scattered(
        fixed_points=pts_fix,
        moving_points=pts_mov,
        levels=[4, 2],
        reg_iterations=[5, 5],
        initial_transform='identity',
    )
    assert res.warp_fwd is not None

    # 2. Test syn_scattered with iterations and levels
    res2 = syntx.syn_scattered(
        fixed_points=pts_fix,
        moving_points=pts_mov,
        levels=[4, 2],
        iterations=[5, 5],
        initial_transform='identity',
    )
    assert res2.warp_fwd is not None

