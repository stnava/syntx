"""Regression tests for syntx.syn midpoint_only/export_midpoint/return_warped fastpaths,
displacement-field write-through caching, and resample_series precomputed_field_jacobian."""

from __future__ import annotations

import os
from unittest.mock import patch

import ants
import numpy as np
import pytest

import syntx
from syntx.transforms import (
    TransformChain,
    compose_displacement,
    jacobian_modulation,
    resample_series,
)


def _pair_3d(size: int = 16):
    base = np.zeros((size, size, size), dtype=np.float32)
    base[4:12, 4:12, 4:12] = 500.0
    base[6:10, 6:10, 6:10] = 1000.0
    fi = ants.from_numpy(np.roll(base, -1, axis=1), spacing=(2.0, 2.0, 2.0))
    mi = ants.from_numpy(np.roll(base, 1, axis=1), spacing=(2.0, 2.0, 2.0))
    return fi, mi


def test_syn_midpoint_only_skips_full_warps_and_matches_standard_midpoint():
    fi, mi = _pair_3d()
    common_kw = dict(
        dof="rigid",
        regularizer="gaussian",
        antisymmetric=True,
        restrict_transformation=(0.0, 1.0, 0.0),
        levels=[2, 1],
        reg_iterations=[6, 4],
        device="cpu",
        seed=42,
    )
    reg_full = syntx.syn(fi, mi, **common_kw)
    reg_mid = syntx.syn(fi, mi, midpoint_only=True, **common_kw)

    # Full end-to-end warps and inverse error check are skipped under midpoint_only=True
    assert reg_mid["warpedmovout"] is None
    assert reg_mid["warpedfixout"] is None
    assert reg_mid["fwd_warp_image"] is None
    assert reg_mid["inv_warp_image"] is None
    assert reg_mid["inverse_identity_errors"] == {}
    assert all(p.endswith(".mat") for p in reg_mid["fwdtransforms"])
    assert len(reg_mid["fwdtransforms"]) == 1

    # Midpoint outputs and in-memory midpoint images match full run identically
    assert reg_mid["fwd_midpoint_warp"] is not None and os.path.exists(reg_mid["fwd_midpoint_warp"])
    assert reg_mid["inv_midpoint_warp"] is not None and os.path.exists(reg_mid["inv_midpoint_warp"])
    assert reg_mid["fwd_midpoint_image"] is not None
    assert reg_mid["inv_midpoint_image"] is not None
    np.testing.assert_allclose(
        reg_mid["fwd_midpoint_image"].numpy(),
        ants.image_read(reg_full["fwd_midpoint_warp"]).numpy(),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        reg_mid["midpoint_fixed"].numpy(),
        reg_full["midpoint_fixed"].numpy(),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        reg_mid["midpoint_moving"].numpy(),
        reg_full["midpoint_moving"].numpy(),
        atol=1e-6,
    )


def test_syn_export_midpoint_false_and_return_warped_false():
    fi, mi = _pair_3d()
    reg = syntx.syn(
        fi,
        mi,
        type_of_transform="SyNOnly",
        initial_transform="Identity",
        restrict_transformation=(0.0, 1.0, 0.0),
        levels=[2, 1],
        reg_iterations=[4, 2],
        export_midpoint=False,
        return_warped=False,
        device="cpu",
        seed=42,
    )
    assert reg["warpedmovout"] is None
    assert reg["warpedfixout"] is None
    assert reg["midpoint_fixed"] is None
    assert reg["midpoint_moving"] is None
    assert reg["fwd_midpoint_warp"] is None
    assert reg["inv_midpoint_warp"] is None
    assert reg["fwd_warp_image"] is not None
    assert any(p.endswith(".nii.gz") for p in reg["fwdtransforms"])
    assert "phi_1" in reg["inverse_identity_errors"]


def test_compose_displacement_write_through_cache_and_precomputed_jacobian(tmp_path):
    fi, mi = _pair_3d()
    disp_arr = np.zeros((16, 16, 16, 3), dtype=np.float32)
    disp_arr[..., 1] = np.linspace(-1.5, 1.5, 16, dtype=np.float32)[None, :, None]
    raw_warp_path = str(tmp_path / "raw_warp.nii.gz")
    ants.image_write(
        ants.from_numpy(disp_arr, spacing=(2.0, 2.0, 2.0), has_components=True),
        raw_warp_path,
    )

    composed_path = str(tmp_path / "composed.nii.gz")
    compose_displacement([raw_warp_path], [False], fi, composed_path, device="cpu")

    vols = [fi, mi, fi * 0.5]
    img4d = ants.list_to_ndimage(
        ants.from_numpy(np.zeros((16, 16, 16, 3), dtype=np.float32), spacing=(2.0, 2.0, 2.0, 1.0)),
        vols,
    )
    chains: list[TransformChain | None] = [None, None, None]

    # Both jacobian_modulation(composed_path, fi) and resample_series hit write-through cache without image_read
    with patch("ants.image_read", wraps=ants.image_read) as spy_read:
        jac = jacobian_modulation(composed_path, fi).numpy()
        out_fast = resample_series(
            img4d,
            chains,
            composed_path,
            fi,
            jacobian=True,
            device="cpu",
            precomputed_field_jacobian=jac,
        )
        assert spy_read.call_count == 0

    out_ref = resample_series(img4d, chains, composed_path, fi, jacobian=True, device="cpu")
    np.testing.assert_allclose(out_fast.numpy(), out_ref.numpy(), atol=1e-6)

    # Shape validation on precomputed_field_jacobian
    with pytest.raises(ValueError, match="precomputed_field_jacobian shape"):
        resample_series(
            img4d,
            chains,
            composed_path,
            fi,
            jacobian=True,
            device="cpu",
            precomputed_field_jacobian=np.ones((8, 8, 8), dtype=np.float32),
        )

    # Overwriting composed_path invalidates cache cleanly
    disp_arr2 = disp_arr * 2.0
    ants.image_write(
        ants.from_numpy(disp_arr2, spacing=(2.0, 2.0, 2.0), has_components=True),
        composed_path,
    )
    jac2 = jacobian_modulation(composed_path, fi).numpy()
    assert not np.allclose(jac, jac2)
