"""Regression tests for robust_affine audit findings (docs/DOCSTRING_AUDIT.md). Instant: every
check fails before any computation."""
import ants
import numpy as np
import pytest

from syntx.robust_affine import robust_affine, ROBUST_AFFINE_MODES


def _img():
    return ants.from_numpy(np.zeros((8, 8), np.float32))


def test_unknown_mode_raises_instead_of_running_ants():
    with pytest.raises(ValueError, match="unknown mode 'bogus'"):
        robust_affine(_img(), _img(), mode="bogus")


def test_mode_fast_is_not_an_alias_any_more():
    with pytest.raises(ValueError, match="preset='fast'"):
        robust_affine(_img(), _img(), mode="fast")


def test_backend_parameter_removed():
    with pytest.raises(TypeError, match="no 'backend' parameter"):
        robust_affine(_img(), _img(), backend="pytorch")


def test_documented_modes_are_accepted_names():
    assert {"auto", "pytorch", "ants_fast", "ants", "com_only", "translation_only",
            "tournament"} <= ROBUST_AFFINE_MODES


def _pair_with(bad_value=None, which=None):
    rng = np.random.default_rng(0)
    imgs = {k: ants.from_numpy(rng.random((16, 16, 16)).astype('float32'))
            for k in ('fixed', 'moving')}
    if which is not None:
        arr = imgs[which].numpy()
        arr[3, 4, 5] = bad_value
        imgs[which] = ants.from_numpy(arr)
    return imgs['fixed'], imgs['moving']


@pytest.mark.parametrize("which,bad", [("fixed", np.nan), ("moving", np.nan),
                                       ("moving", np.inf), ("fixed", -np.inf)])
def test_nonfinite_input_raises_named_error(which, bad):
    fixed, moving = _pair_with(bad, which)
    with pytest.raises(ValueError, match=rf"{which} contains 1 non-finite voxel.*mask"):
        robust_affine(fixed, moving, dof="rigid")


def test_nonfinite_input_raises_in_registration_multichannel():
    import syntx
    fixed, moving = _pair_with(np.nan, "moving")
    with pytest.raises(ValueError, match=r"moving\[1\] contains 1 non-finite"):
        syntx.registration(fixed, [fixed, moving])


def test_finite_control_passes_validation():
    fixed, moving = _pair_with()
    out = robust_affine(fixed, moving, dof="rigid", mode="com_only")
    assert np.all(np.isfinite(ants.read_transform(out['fwdtransforms'][0]).parameters))
