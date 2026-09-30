"""
tests/test_robust_cross_modal_rigid.py
=======================================

Unit tests for syntx.robust_affine.robust_cross_modal_rigid -- the shared two-stage
(coarse robust_affine rigid + fine multi-resolution Mattes MI ants.registration)
cross-modal rigid registration recipe, migrated here from
antsxfunctional.pet.registration.register_t1_to_pet so every cross-modal rigid
registration in the ecosystem (T1-to-PET, and potentially others) shares one
implementation.
"""

import numpy as np
import ants

from syntx.robust_affine import robust_cross_modal_rigid


def _synthetic_sphere_pair(dim=24, offset=(1.5, -1.0, 0.5)):
    grid = np.linspace(-1, 1, dim)
    zz, yy, xx = np.meshgrid(grid, grid, grid, indexing="ij")
    r = np.sqrt(xx**2 + yy**2 + zz**2)
    arr = np.clip(1.0 - r, 0, 1.0).astype(np.float32)
    fixed = ants.from_numpy(arr, origin=(0, 0, 0), spacing=(1.0, 1.0, 1.0))
    moving = ants.from_numpy(arr, origin=offset, spacing=(1.0, 1.0, 1.0))
    return fixed, moving


def test_robust_cross_modal_rigid_recovers_known_offset():
    """A known rigid (translation-only) offset between two identically-shaped synthetic
    volumes should be recovered with a near-identity rotation (determinant ~1.0) and the
    warped moving image should land back on the fixed image's grid/content."""
    fixed, moving = _synthetic_sphere_pair()
    mask = ants.threshold_image(moving, 0.1, 2.0)

    res = robust_cross_modal_rigid(
        fixed=fixed, moving=moving, moving_mask=mask, seed=42,
        aff_sampling=32, aff_random_sampling_rate=0.2, verbose=False,
    )

    assert "fwdtransforms" in res and len(res["fwdtransforms"]) > 0
    assert "invtransforms" in res
    assert abs(res["determinant"] - 1.0) < 0.01
    assert res["time"] > 0
    assert res["warpedmovout"] is not None

    warped_arr = res["warpedmovout"].numpy()
    fixed_arr = fixed.numpy()
    # After successful registration, the warped moving sphere should correlate strongly
    # with the fixed sphere (loose threshold -- this is a coarse synthetic sanity check,
    # not a precision benchmark).
    corr = np.corrcoef(warped_arr.ravel(), fixed_arr.ravel())[0, 1]
    assert corr > 0.8, f"expected strong correlation after registration, got {corr:.3f}"


def test_robust_cross_modal_rigid_without_mask_falls_back_to_otsu():
    """moving_mask=None should not raise -- falls back to an Otsu foreground mask."""
    fixed, moving = _synthetic_sphere_pair()
    res = robust_cross_modal_rigid(
        fixed=fixed, moving=moving, moving_mask=None, seed=7,
        aff_sampling=32, aff_random_sampling_rate=0.2, verbose=False,
    )
    assert abs(res["determinant"] - 1.0) < 0.01
