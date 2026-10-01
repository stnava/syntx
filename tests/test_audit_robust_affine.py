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
