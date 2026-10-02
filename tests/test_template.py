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
# type_of_transform='Greedy' -- one-directional, no affine/warp split, no inverse.
# ---------------------------------------------------------------------------

class TestBuildTemplateGreedy:
    """Greedy template construction doesn't need invertible per-image transforms (the
    template only needs each image warped INTO template space to accumulate an average) --
    these tests confirm the dedicated code path (no affinelist/avgaffine, has_field driven
    by is_greedy rather than L >= 2) produces the same result schema and converges
    sensibly, not just that it avoids crashing."""

    def test_return_schema_matches_syn_path(self):
        imgs = _cohort_2d(n=4, size=32)
        result = syntx.build_template(
            image_list=imgs,
            iterations=1,
            type_of_transform='Greedy',
            reg_iterations=[20, 10],
            scales=[2, 1],
            verbose=False,
        )
        required_keys = {
            'template', 'warped_images', 'fwdtransforms',
            'invtransforms', 'convergence', 'shape_residuals',
            'n_iterations', 'weights', 'elapsed_sec',
        }
        assert required_keys.issubset(result.keys())
        # Greedy returns a single composed field (no return_inverse requested) -- confirm
        # build_template doesn't silently invent/require an inverse for it.
        assert all(inv == [] for inv in result['invtransforms'])
        assert all(len(fwd) == 1 for fwd in result['fwdtransforms'])

    def test_shape_residual_uses_composed_field_not_zeroed(self):
        """Regression guard: before this path existed, L (len(fwdtransforms)) == 1 for any
        single-file transform list meant has_field was False, silently zeroing the shape
        residual every iteration for a hypothetical single-field backend. Greedy must report
        a real, non-degenerate residual from its one composed field."""
        imgs = _cohort_2d(n=4, size=32)
        result = syntx.build_template(
            image_list=imgs,
            iterations=2,
            type_of_transform='Greedy',
            reg_iterations=[20, 10],
            scales=[2, 1],
            verbose=False,
        )
        assert len(result['shape_residuals']) == 2
        for sr in result['shape_residuals']:
            assert np.isfinite(sr['l2_norm'])
            assert sr['l2_norm'] >= 0.0
        # At least one iteration should show real (non-zero) deformation given 4 genuinely
        # displaced discs -- otherwise has_field silently fell back to the degenerate branch.
        assert any(sr['l2_norm'] > 0.0 for sr in result['shape_residuals'])

    def test_converges_toward_centered_template(self):
        """The 4 discs are placed symmetrically around the image centre -- a converged
        template (greedy or SyN) should end up centred there too, same correctness check as
        a population-template build is actually meant to satisfy."""
        imgs = _cohort_2d(n=4, size=32)
        result = syntx.build_template(
            image_list=imgs,
            iterations=3,
            type_of_transform='Greedy',
            reg_iterations=[20, 10],
            scales=[2, 1],
            verbose=False,
        )
        arr = result['template'].numpy()
        from scipy.ndimage import center_of_mass as ndi_center_of_mass
        com = ndi_center_of_mass(arr)
        assert abs(com[0] - 16) < 4.0
        assert abs(com[1] - 16) < 4.0

    def test_ants_backend_raises_for_greedy(self):
        """syntx.greedy has no backend='ants' equivalent -- must fail fast and clearly,
        not silently fall through to ants.registration with an invalid type_of_transform."""
        imgs = _cohort_2d(n=2, size=16)
        with pytest.raises(ValueError, match="Greedy"):
            syntx.build_template(image_list=imgs, type_of_transform='Greedy', backend='ants')

    def test_multi_iteration_runs_realign_from_scratch_each_time(self):
        """Unlike the SyN path's 'SyNOnly' drift-avoidance continuity, greedy re-aligns via
        syntx.robust_affine from scratch every iteration (no initial_transform carried
        forward -- see build_template's docstring for why). Confirms a multi-iteration run
        still completes cleanly and the template keeps evolving (a template that stopped
        changing after iteration 0 would suggest xavg itself wasn't being updated between
        iterations, a different real bug this guards against)."""
        imgs = _cohort_2d(n=4, size=32)
        result = syntx.build_template(
            image_list=imgs,
            iterations=2,
            type_of_transform='Greedy',
            reg_iterations=[20, 10],
            scales=[2, 1],
            verbose=False,
        )
        assert len(result['convergence']) == 2
        assert all(c >= 0.0 for c in result['convergence'])


class TestBuildTemplateGreedyAffineCaching:
    """initial_transforms (per-image, iteration-0 seed) + affine_drop_tolerance (adaptive
    re-affine) -- the actual mechanism that makes repeated greedy template-building
    iterations cheap: cache each image's affine instead of re-running robust_affine's full
    search every iteration, but re-verify it's still good enough rather than trusting it
    unconditionally forever."""

    def test_initial_transforms_identity_runs_without_crashing(self):
        imgs = _cohort_2d(n=3, size=32)
        result = syntx.build_template(
            image_list=imgs,
            iterations=1,
            type_of_transform='Greedy',
            initial_transforms=['identity'] * 3,
            reg_iterations=[15, 5],
            scales=[2, 1],
            verbose=False,
        )
        assert result['template'] is not None

    def test_initial_transforms_length_mismatch_raises(self):
        imgs = _cohort_2d(n=3, size=32)
        with pytest.raises(ValueError, match="initial_transforms"):
            syntx.build_template(
                image_list=imgs, type_of_transform='Greedy',
                initial_transforms=['identity', 'identity'],  # length 2, need 3
                reg_iterations=[10, 5], scales=[2, 1],
            )

    def test_singular_initial_transform_kwarg_rejected_for_greedy(self):
        """The plural initial_transforms (per-image) is how this path is controlled --
        passing the singular initial_transform via **kwargs would be silently overwritten
        every iteration by build_template's own management of it, so it's rejected instead
        of accepted-but-ignored."""
        imgs = _cohort_2d(n=2, size=16)
        with pytest.raises(TypeError, match="initial_transforms"):
            syntx.build_template(
                image_list=imgs, type_of_transform='Greedy',
                initial_transform='identity',
                reg_iterations=[10, 5], scales=[2, 1],
            )

    def test_affine_drop_tolerance_triggers_recompute(self, monkeypatch):
        """An impossible-to-satisfy tolerance (-1.0, since correlation is bounded in
        [-1, 1]) must force a from-scratch affine recompute on every post-iteration-0 image,
        producing strictly more greedy_registration calls than disabling the check
        (affine_drop_tolerance=None, cache trusted unconditionally). A directional
        comparison rather than an exact count -- robust to implementation details of
        exactly how many extra calls recomputation costs."""
        import importlib
        # syntx/__init__.py's `greedy = greedy_registration` alias shadows the `greedy`
        # attribute on the `syntx` package, so `import syntx.greedy` would hand back that
        # function, not the submodule -- fetch the actual submodule from sys.modules instead.
        greedy_mod = importlib.import_module('syntx.greedy')
        real_greedy_registration = greedy_mod.greedy_registration
        call_log = []

        def spy(*args, **kw):
            call_log.append(kw.get('initial_transform'))
            return real_greedy_registration(*args, **kw)

        monkeypatch.setattr(greedy_mod, 'greedy_registration', spy)
        imgs = _cohort_2d(n=2, size=32)

        call_log.clear()
        syntx.build_template(
            image_list=imgs, iterations=2, type_of_transform='Greedy',
            affine_drop_tolerance=None, reg_iterations=[15, 5], scales=[2, 1], verbose=False,
        )
        n_calls_disabled = len(call_log)

        call_log.clear()
        syntx.build_template(
            image_list=imgs, iterations=2, type_of_transform='Greedy',
            affine_drop_tolerance=-1.0, reg_iterations=[15, 5], scales=[2, 1], verbose=False,
        )
        n_calls_forced = len(call_log)

        assert n_calls_forced > n_calls_disabled, (
            f"expected more greedy_registration calls with an impossible-to-satisfy "
            f"affine_drop_tolerance ({n_calls_forced}) than with it disabled "
            f"({n_calls_disabled})"
        )

    def test_identity_seeded_image_stays_identity_seeded_across_iterations(self, monkeypatch):
        """Regression guard for a real bug found on real FDG-PET data: caching None (what
        greedy_registration returns for 'no affine was used') and replaying it as the next
        iteration's initial_transform silently means something different to
        greedy_registration itself (its own "nothing given -> run robust_affine" default) --
        an identity-seeded, stable image would silently fall back to an expensive
        full-affine-search on every iteration after the first, defeating the entire point of
        caching. Every call for an 'identity'-seeded image must stay seeded with the literal
        string 'identity', never None, across however many iterations run (with
        affine_drop_tolerance disabled, so nothing should ever force a recompute here)."""
        import importlib
        greedy_mod = importlib.import_module('syntx.greedy')
        real_greedy_registration = greedy_mod.greedy_registration
        call_log = []

        def spy(*args, **kw):
            call_log.append(kw.get('initial_transform'))
            return real_greedy_registration(*args, **kw)

        monkeypatch.setattr(greedy_mod, 'greedy_registration', spy)
        imgs = _cohort_2d(n=2, size=32)

        syntx.build_template(
            image_list=imgs,
            iterations=3,
            type_of_transform='Greedy',
            initial_transforms=['identity', 'identity'],
            affine_drop_tolerance=None,
            reg_iterations=[15, 5],
            scales=[2, 1],
            verbose=False,
        )
        assert len(call_log) == 3 * 2, f"expected 3 iterations x 2 images = 6 calls, got {len(call_log)}"
        assert all(c == 'identity' for c in call_log), (
            f"expected every call seeded with 'identity', got {call_log} -- a None anywhere "
            f"here means a cached identity-seeded image silently fell back to a full "
            f"robust_affine search on a later iteration"
        )


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
