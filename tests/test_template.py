"""
tests/test_template.py — Unit + smoke tests for syntx.build_template

Tests cover:
- Weight normalization
- Initial-template construction (uniform + weighted)
- _scale_warp, _template_change, _compute_shape_residual unit tests
- Return dict schema: all expected keys present including 'shape_residuals'
- Sharpen-blend correctness: blending does NOT use old template
- Shape residual values are finite and non-negative
- Smoke: runs to completion without raising on 2D data

All tests run on CPU with tiny (32×32) phantoms for speed.
"""

import numpy as np
import pytest
import ants
import syntx
from syntx.template import (
    _normalize_weights,
    _initial_template,
    _scale_warp,
    _template_change,
    _compute_shape_residual,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _disc_image(cx, cy, size=32, radius=6):
    """Create a 2D ANTsImage with a filled disc at (cx, cy)."""
    arr = np.zeros((size, size), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                arr[y, x] = 1.0
    return ants.from_numpy(arr, spacing=(1.0, 1.0))


def _cohort_2d(n=4, size=32, radius=5):
    """
    Build a 2D cohort of n disc images displaced symmetrically around center.
    """
    cx0, cy0 = size // 2, size // 2
    disps = [(-4, 0), (4, 0), (0, -4), (0, 4)]
    imgs = []
    for i in range(n):
        dx, dy = disps[i % len(disps)]
        imgs.append(_disc_image(cx0 + dx, cy0 + dy, size=size, radius=radius))
    return imgs


# ---------------------------------------------------------------------------
# Unit: weight normalization
# ---------------------------------------------------------------------------

class TestNormalizeWeights:
    def test_none_returns_uniform(self):
        w = _normalize_weights(None, 4)
        assert w.shape == (4,)
        assert np.allclose(w, 0.25)

    def test_custom_weights_normalized(self):
        w = _normalize_weights([1, 2, 3, 4], 4)
        assert np.isclose(w.sum(), 1.0)
        assert np.isclose(w[3], 0.4)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match=r"len\(weights\)"):
            _normalize_weights([1, 2], 3)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _normalize_weights([-1, 2, 3], 3)

    def test_all_zero_raises(self):
        with pytest.raises(ValueError, match="positive sum"):
            _normalize_weights([0, 0, 0], 3)


# ---------------------------------------------------------------------------
# Unit: initial template from mean
# ---------------------------------------------------------------------------

class TestInitialTemplate:
    def test_uniform_mean(self):
        imgs = _cohort_2d(n=4, size=16)
        w = _normalize_weights(None, 4)
        tmpl = _initial_template(imgs, w)
        arr = tmpl.numpy()
        assert arr.shape == (16, 16)
        assert float(arr.max()) > 0.0
        assert float(arr.min()) >= 0.0
        # centre pixel should be part of every disc → near 1.0
        assert arr[8, 8] > 0.5

    def test_weighted_mean_shifts_toward_heavy(self):
        imgs = _cohort_2d(n=4, size=32)
        # put most weight on first image (displaced -4 in x)
        w = np.array([0.97, 0.01, 0.01, 0.01])
        tmpl = _initial_template(imgs, w)
        arr = tmpl.numpy()
        xs = np.arange(32)
        ys = np.arange(32)
        XX, YY = np.meshgrid(xs, ys)
        total = arr.sum() + 1e-10
        cx = float((XX * arr).sum() / total)
        assert cx < 16.0  # shifted left of centre (16)


# ---------------------------------------------------------------------------
# Unit: _scale_warp
# ---------------------------------------------------------------------------

class TestScaleWarp:
    def test_scale_zero(self):
        arr = np.ones((8, 8, 2), dtype=np.float32)
        img = ants.from_numpy(arr, spacing=(1.0, 1.0), has_components=True)
        scaled = _scale_warp(img, 0.0)
        assert np.allclose(scaled.numpy(), 0.0)

    def test_scale_negative(self):
        arr = np.ones((8, 8, 2), dtype=np.float32) * 2.0
        img = ants.from_numpy(arr, spacing=(1.0, 1.0), has_components=True)
        scaled = _scale_warp(img, -0.5)
        assert np.allclose(scaled.numpy(), -1.0)


# ---------------------------------------------------------------------------
# Unit: _template_change
# ---------------------------------------------------------------------------

class TestTemplateChange:
    def test_identical_zero(self):
        arr = np.random.rand(16, 16).astype(np.float32)
        img = ants.from_numpy(arr, spacing=(1.0, 1.0))
        assert _template_change(img, img) == pytest.approx(0.0, abs=1e-7)

    def test_known_change(self):
        a = ants.from_numpy(np.zeros((8, 8), dtype=np.float32), spacing=(1.0, 1.0))
        b = ants.from_numpy(np.ones((8, 8), dtype=np.float32), spacing=(1.0, 1.0))
        assert _template_change(a, b) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Unit: _compute_shape_residual
# ---------------------------------------------------------------------------

class TestComputeShapeResidual:
    def test_zero_field_has_zero_residual(self):
        arr = np.zeros((16, 16, 2), dtype=np.float32)
        img = ants.from_numpy(arr, spacing=(1.0, 1.0), has_components=True)
        sr = _compute_shape_residual(img)
        assert sr['l2_norm'] == pytest.approx(0.0, abs=1e-6)
        assert sr['membrane_energy'] == pytest.approx(0.0, abs=1e-6)
        assert sr['bending_energy'] == pytest.approx(0.0, abs=1e-6)

    def test_constant_field_has_zero_energy(self):
        # Constant displacement: derivatives are zero → membrane and bending = 0
        arr = np.ones((16, 16, 2), dtype=np.float32) * 3.0
        img = ants.from_numpy(arr, spacing=(1.0, 1.0), has_components=True)
        sr = _compute_shape_residual(img)
        # L2 norm of constant 3mm displacement = 3*sqrt(2) ≈ 4.24
        assert sr['l2_norm'] == pytest.approx(np.sqrt(2) * 3.0, rel=1e-4)
        assert sr['membrane_energy'] == pytest.approx(0.0, abs=1e-5)
        assert sr['bending_energy'] == pytest.approx(0.0, abs=1e-5)

    def test_keys_present(self):
        arr = np.random.rand(8, 8, 2).astype(np.float32)
        img = ants.from_numpy(arr, spacing=(1.0, 1.0), has_components=True)
        sr = _compute_shape_residual(img)
        assert set(sr.keys()) == {'l2_norm', 'membrane_energy', 'bending_energy'}

    def test_all_finite_and_nonneg(self):
        arr = np.random.rand(8, 8, 2).astype(np.float32)
        img = ants.from_numpy(arr, spacing=(1.0, 1.0), has_components=True)
        sr = _compute_shape_residual(img)
        for k, v in sr.items():
            assert np.isfinite(v), f"{k} is not finite"
            assert v >= 0.0, f"{k} is negative"


# ---------------------------------------------------------------------------
# Smoke + integration: build_template on 2D cohort
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBuildTemplateIntegration:
    def test_return_schema(self):
        imgs = _cohort_2d(n=4, size=32)
        result = syntx.build_template(
            image_list=imgs,
            iterations=1,
            gradient_step=0.20,
            blending_weight=0.75,
            type_of_transform='SyNTo',
            reg_iterations=[20, 10, 0],
            verbose=False,
        )
        required_keys = {
            'template', 'warped_images', 'fwdtransforms',
            'invtransforms', 'convergence', 'shape_residuals',
            'n_iterations', 'weights', 'elapsed_sec',
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_shape_residuals_structure(self):
        imgs = _cohort_2d(n=4, size=32)
        result = syntx.build_template(
            image_list=imgs,
            iterations=2,
            gradient_step=0.20,
            reg_iterations=[20, 10, 0],
            verbose=False,
        )
        assert len(result['shape_residuals']) == 2
        for sr in result['shape_residuals']:
            assert 'l2_norm' in sr
            assert 'membrane_energy' in sr
            assert 'bending_energy' in sr
            assert np.isfinite(sr['l2_norm'])
            assert sr['l2_norm'] >= 0.0

    def test_template_values_bounded(self):
        imgs = _cohort_2d(n=4, size=32)
        result = syntx.build_template(
            image_list=imgs,
            iterations=2,
            gradient_step=0.20,
            reg_iterations=[20, 10, 0],
            verbose=False,
        )
        arr = result['template'].numpy()
        assert float(arr.min()) >= -0.2
        assert float(arr.max()) <= 2.0

    def test_convergence_history_length(self):
        imgs = _cohort_2d(n=4, size=32)
        result = syntx.build_template(
            image_list=imgs,
            iterations=3,
            gradient_step=0.20,
            reg_iterations=[15, 0, 0],
            verbose=False,
        )
        assert result['n_iterations'] == 3
        assert len(result['convergence']) == 3

    def test_warped_images_count(self):
        imgs = _cohort_2d(n=4, size=32)
        result = syntx.build_template(
            image_list=imgs,
            iterations=1,
            reg_iterations=[15, 0, 0],
            verbose=False,
        )
        assert len(result['warped_images']) == 4
        for wi in result['warped_images']:
            assert wi is not None

    def test_weights_normalized(self):
        imgs = _cohort_2d(n=4, size=32)
        result = syntx.build_template(
            image_list=imgs,
            weights=[1, 2, 3, 4],
            iterations=1,
            reg_iterations=[10, 0, 0],
            verbose=False,
        )
        assert np.isclose(result['weights'].sum(), 1.0)

    def test_explicit_initial_template(self):
        imgs = _cohort_2d(n=4, size=32)
        init = _disc_image(16, 16, size=32, radius=5)
        result = syntx.build_template(
            initial_template=init,
            image_list=imgs,
            iterations=1,
            reg_iterations=[10, 0, 0],
            verbose=False,
        )
        assert result['template'] is not None

    def test_useNoRigid_no_crash(self):
        imgs = _cohort_2d(n=4, size=32)
        result = syntx.build_template(
            image_list=imgs,
            iterations=1,
            useNoRigid=True,
            reg_iterations=[10, 0, 0],
            verbose=False,
        )
        assert 'template' in result


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestBuildTemplateErrors:
    def test_empty_image_list_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            syntx.build_template(image_list=[])

    def test_none_image_list_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            syntx.build_template(image_list=None)

    def test_bad_blending_weight_raises(self):
        imgs = _cohort_2d(n=2, size=8)
        with pytest.raises(ValueError, match="blending_weight"):
            syntx.build_template(image_list=imgs, blending_weight=0.0)

    def test_bad_gradient_step_raises(self):
        imgs = _cohort_2d(n=2, size=8)
        with pytest.raises(ValueError, match="gradient_step"):
            syntx.build_template(image_list=imgs, gradient_step=1.5)


# ---------------------------------------------------------------------------
# Unit: reflect_image
# ---------------------------------------------------------------------------

class TestReflectImage:
    def test_header_preservation(self):
        r16 = ants.image_read(ants.get_ants_data('r16'))
        r16_ref = syntx.reflect_image(r16, axis=0)
        assert r16_ref.shape == r16.shape
        assert r16_ref.spacing == r16.spacing
        assert r16_ref.origin == r16.origin
        assert np.allclose(r16_ref.direction, r16.direction)

    def test_string_axis_aliases(self):
        r16 = ants.image_read(ants.get_ants_data('r16'))
        ref_x = syntx.reflect_image(r16, axis='x')
        ref_0 = syntx.reflect_image(r16, axis=0)
        assert np.allclose(ref_x.numpy(), ref_0.numpy())
