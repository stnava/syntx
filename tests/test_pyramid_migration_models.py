"""Pass 3b: the deformable models and their one-off resamples use the central pyramid (``syntx.pyramid``).

Golden tests: the migrated call sites reproduce the incumbent pyramid code (``tests/legacy_pyramid_reference.py``, the code that
used to live in ``syntx/pyramid.py`` and in ``greedy`` / ``tvf_adj`` / ``scattered``) bit for bit. Regression tests: the latent
crash / NaN / wrong-axis bugs that were fixed on the way.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F


def _smooth(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.interpolate(torch.rand(1, 1, 5, 5, 5, generator=g), size=shape, mode='trilinear', align_corners=True)


# ---------------------------------------------------------------------------------------------------------- greedy
def test_greedy_moving_image_gets_its_own_antialias_sigma(monkeypatch):
    """greedy applied the FIXED image's anti-alias sigma to the moving image when the sizes differ (identical for equal shapes)."""
    import importlib
    from syntx.pyramid import build_pyramid as real_build, plan_levels
    G = importlib.import_module("syntx.greedy")

    fixed, moving = _smooth((40, 48, 36)), _smooth((30, 44, 52), seed=1)
    seen = {}

    def spy(image, *a, **k):
        out = real_build(image, *a, **k)
        seen[tuple(image.shape[2:])] = (out[0].plan.sigma_vox_zyx, tuple(out[0].tensor.shape[2:]))
        return out

    monkeypatch.setattr(G, 'build_pyramid', spy)
    model = G.GreedyRegistrationModel(dim=3, similarity_metric='cc2', lncc_radius=2)
    model.fit(fixed, moving, torch.eye(3, 4)[None], scales=[4], iterations=[1])
    for img in (fixed, moving):
        want = plan_levels(img.shape[2:], (1, 1, 1), [4], 'legacy_floor16')[0]
        assert seen[tuple(img.shape[2:])] == (want.sigma_vox_zyx, want.shape_zyx)
    assert seen[(40, 48, 36)][0] != seen[(30, 44, 52)][0]


def test_greedy_full_resolution_level_on_a_thin_axis_does_not_crash():
    """An axis shorter than 16 voxels at scale 1: the warp grid was floored to 16 while the images stayed native -> RuntimeError."""
    import syntx
    fixed, moving = _smooth((10, 40, 40)), torch.roll(_smooth((10, 40, 40)), (1, 2, -1), (2, 3, 4))
    from syntx.greedy import GreedyRegistrationModel
    model = GreedyRegistrationModel(dim=3, similarity_metric='cc2', lncc_radius=2)
    warp = model.fit(fixed, moving, torch.eye(3, 4)[None], scales=[2, 1], iterations=[2, 2])
    assert tuple(warp.shape) == (1, 10, 40, 40, 3) and torch.isfinite(warp).all()


# ---------------------------------------------------------------------------------------------------------- build_image_pyramid
from . import legacy_pyramid_reference as legacy  # noqa: E402


@pytest.mark.parametrize("shape,levels,sp", [((33, 40, 47), [4, 2, 1], (1.0, 1.0, 1.0)), ((24, 48, 40), [3, 2, 1], (2.0, 1.0, 0.5)),
                                              ((41, 29), [2, 1], (1.0, 1.0))])
def test_build_image_pyramid_delegate_equals_the_incumbent(shape, levels, sp):
    from syntx.pyramid import build_image_pyramid
    x = _smooth(shape) if len(shape) == 3 else torch.rand(1, 1, *shape)
    for kw in (dict(), dict(smoothing_sigmas=[1.0, 0.5, 0.0][:len(levels)]), dict(smoothing_sigmas=0.7)):
        a, b = legacy.build_image_pyramid(x, sp, levels, **kw), build_image_pyramid(x, sp, levels, **kw)
        assert len(a) == len(b) and all(torch.equal(u, v) for u, v in zip(a, b))


@pytest.mark.parametrize("shape,level,want", [((3, 64, 64), 4, (2, 16, 16)), ((3, 64, 64), 8, (2, 8, 8)), ((11, 128, 128), 8, (2, 16, 16))])
def test_build_image_pyramid_thin_axis_no_crash_no_collapse(shape, level, want):
    """The incumbent raised ('sizes should be greater than 0') on (3, 64, 64) at level 4 and returned a 1-voxel axis for 11 slices at level 8."""
    from syntx.pyramid import build_image_pyramid
    if shape == (3, 64, 64):
        with pytest.raises(RuntimeError):
            legacy.build_image_pyramid(torch.rand(1, 1, *shape), (1.0,) * 3, [level])
    out = build_image_pyramid(torch.rand(1, 1, *shape), (1.0,) * 3, [level])[0]
    assert tuple(out.shape[2:]) == want and torch.isfinite(out).all()


def test_build_image_pyramid_argument_errors():
    from syntx.pyramid import build_image_pyramid, PyramidLevelError
    x = torch.rand(1, 1, 16, 16, 16)
    with pytest.raises(ValueError):
        build_image_pyramid(x, (1.0,) * 3, [2, 1], smoothing_sigmas=[1.0])
    with pytest.raises(ValueError):
        build_image_pyramid(x, None, [2, 1], sigma_mode='physical')
    with pytest.raises(PyramidLevelError):
        build_image_pyramid(x, (1.0,) * 3, [0.5])
    assert len(build_image_pyramid(x, None, [2, 1])) == 2        # voxel mode never needs the spacing


# ---------------------------------------------------------------------------------------------------------- tvf_adj
@pytest.mark.parametrize("shape", [(31, 45, 50), (6, 40, 40), (12, 33, 21)])
@pytest.mark.parametrize("level", [4, 2, 1])
def test_tvf_adj_level_image_equals_the_incumbent_interpolate(shape, level):
    """tvf_adj: F.interpolate(align_corners) to max(8, s // level), no smoothing (floor 8, sigma 0 of legacy_floor16)."""
    from types import SimpleNamespace
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from syntx.tvf_adj import TVFRegistrationAdjoint
    x = _smooth(shape)
    curr = [max(8, s // level) for s in shape]
    ref = F.interpolate(x, size=curr, mode='trilinear', align_corners=True)
    got = TVFRegistrationAdjoint._level_image(SimpleNamespace(dim=3), x, level, curr)
    assert got.shape == ref.shape and torch.equal(got, ref)


# ---------------------------------------------------------------------------------------------------------- JAX trio
def _jax_pair(shape=(16, 16, 16)):
    return (_smooth(shape).numpy().astype(np.float32), torch.roll(_smooth(shape), (1, 2, -1), (2, 3, 4)).numpy().astype(np.float32))


@pytest.mark.parametrize("shape,level", [((32, 32, 32), 2), ((32, 40, 36), 4), ((12, 40, 40), 2), ((6, 33, 21), 2)])
@pytest.mark.parametrize("sigma", [0.0, 1.0])
def test_pyramid_level_jax_matches_the_incumbent_jax_resampling(shape, level, sigma):
    """The JAX models' level images (central ``build_pyramid_jax``) equal the incumbent: ITK Gaussian on every axis (channels last),
    then ``interpolate_jax`` (align_corners) to ``max(8, s // level)`` -- to float32 rounding, shapes exactly."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from syntx.syn_jax import pyramid_level_jax, interpolate_jax, separable_gaussian_filter_jax
    x = jnp.asarray(_smooth(shape).numpy())
    got = pyramid_level_jax(x, level, sigma)
    xs = jnp.moveaxis(separable_gaussian_filter_jax(jnp.moveaxis(x, 1, -1), sigma=sigma, spacing=None, sigma_mode='voxel'), -1, 1) if sigma > 0 else x
    want = interpolate_jax(xs, tuple(max(8, s // level) for s in shape), 3)
    assert got.shape == want.shape
    assert float(jnp.abs(got - want).max()) < 2e-5


def test_pyramid_level_jax_smooths_every_axis():
    """syngs_jax / tvf_jax passed a (D, H, W) array to the channels-last filter ``(B, *spatial, C)``: only the y axis was smoothed."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from syntx.syn_jax import pyramid_level_jax
    x = np.zeros((1, 1, 33, 33, 33), np.float32); x[0, 0, 16, 16, 16] = 1.0
    out = np.asarray(pyramid_level_jax(jnp.asarray(x), 2, 1.5))[0, 0]
    c = np.array(out.shape) // 2
    prof = [out[tuple(slice(None) if i == a else c[i] for i in range(3))] for a in range(3)]
    for a in (1, 2):
        assert np.allclose(prof[0], prof[a], atol=1e-6), f"axis {a} is smoothed differently from axis 0"
    assert prof[0][c[0]] < 1.0 and prof[0][c[0] + 1] > 0                      # the impulse is spread along z, not only y


@pytest.mark.parametrize("which", ["syngs", "tvf"])
def test_jax_models_build_level_images_through_the_central_pyramid(which, monkeypatch):
    pytest.importorskip("jax")
    import importlib
    if which == "syngs":
        mod = importlib.import_module("syntx.syngs_jax")
        model = mod.GeodesicShootingModelJAX(dim=3, image_shape=(16,) * 3, velocity_shape=(16,) * 3)
    else:
        mod = importlib.import_module("syntx.tvf_jax")
        model = mod.TVFModelJAX(dim=3, image_shape=(16,) * 3, velocity_shape=(16,) * 3)
    f, m = _jax_pair()
    seen = []
    real = mod.pyramid_level_jax
    monkeypatch.setattr(mod, 'pyramid_level_jax', lambda image, level, sigma=0.0, **k: (seen.append((level, sigma)), real(image, level, sigma, **k))[1])
    model.fit(f, m, levels=[2, 1], epochs_per_level=[1, 1], affine_epochs=1, lncc_radius=2, smooth_pyramid=True)
    assert seen and all(level == 2 and abs(sigma - 1.0) < 1e-12 for level, sigma in seen)       # level 1 is the native image


def test_pyramid_level_jax_moving_on_another_grid_goes_to_the_fixed_level_grid():
    """Incumbent: a moving image of another shape is smoothed and resampled to the FIXED level shape (kept by the migration)."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from syntx.syn_jax import pyramid_level_jax, interpolate_jax, separable_gaussian_filter_jax
    x = jnp.asarray(_smooth((20, 28, 32)).numpy())
    got = pyramid_level_jax(x, 2, 1.0, target_shape=(12, 16, 14))
    sm = jnp.moveaxis(separable_gaussian_filter_jax(jnp.moveaxis(x, 1, -1), sigma=1.0, spacing=None, sigma_mode='voxel'), -1, 1)
    want = interpolate_jax(sm, (12, 16, 14), 3)
    assert got.shape == want.shape and float(jnp.abs(got - want).max()) < 2e-5
