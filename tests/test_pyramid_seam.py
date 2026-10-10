"""The private ``_level_builder`` seam of robust_affine and motion_batched (used by benchmarks/pyramid_compare.py) must leave the
default behaviour untouched: a builder that reproduces the default pyramid gives the identical transform, and a builder that
violates the contract raises instead of being ignored."""
import os
import sys

import ants
import numpy as np
import pytest

import syntx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks"))
import pyramid_compare as pc  # noqa: E402

pytest.importorskip("siq")


def _params(reg):
    return np.asarray(ants.read_transform(reg["fwdtransforms"][0]).parameters)


@pytest.mark.parametrize("r, label", [(1, "P1"), (4, "DEFAULT_EQUIV")])
def test_default_equals_equivalent_builder_cpu(r, label):
    """isotropic default = voxel pooling (P1); anisotropic default = the millimetre pyramid for the fixed image, pooling for an isotropic moving image: bit-identical transforms."""
    fixed, mov, _ = pc.slab_pair(r)
    base = syntx.robust_affine(fixed, mov, mode="pytorch", dof="rigid", device="cpu", seed=42)
    alt = syntx.robust_affine(fixed, mov, mode="pytorch", dof="rigid", device="cpu", seed=42, _level_builder=pc.get_policy(label))
    assert np.array_equal(_params(base), _params(alt))


def test_bad_clamp_rule_raises():
    fixed, mov, _ = pc.slab_pair(1)

    def bad(role, arr, sp, ctx):
        raise AssertionError("not reached")
    bad.min_shape_rule = "bogus"
    with pytest.raises(syntx.RobustAffineSolverError):
        syntx.robust_affine(fixed, mov, mode="pytorch", dof="rigid", device="cpu", _level_builder=bad)


def test_valid_margin_masks_edge_slices_only():
    import torch
    v = pc.valid_margin((11, 64, 64), (0.43, 0.43, 3.5), [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], (11, 64, 64), 1, "cpu")
    assert v.shape == (1, 1, 11, 64, 64)
    per_slice = v.reshape(11, -1).all(dim=1)
    assert per_slice.tolist() == [False] + [True] * 9 + [False]
    assert pc.valid_margin((32, 32, 32), (1.0, 1.0, 1.0), [1.0] * 3, [0.0] * 3, (32, 32, 32), 1, "cpu") is None


def test_motion_default_equals_p1_builder_cpu():
    from syntx.motion_batched import _batched_rigid_register_pass_core as core
    ref, frames, _ = pc._motion_series((32, 36, 32), (2.0, 2.0, 2.0), [(1.0, -1.0, (1.5, -1.0, 1.0))])
    a = core(ref, frames, device="cpu", seed_mode="narrow", schedule_mode="fast")[0]
    b = core(ref, frames, device="cpu", seed_mode="narrow", schedule_mode="fast", _level_builder=pc.get_policy("P1"))[0]
    # the builder path normalises coordinates through (f, off) instead of sp*level: equal up to float rounding
    np.testing.assert_allclose(ants.read_transform(a[0][0]).parameters, ants.read_transform(b[0][0]).parameters, atol=2e-3)
