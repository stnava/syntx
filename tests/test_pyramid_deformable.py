"""Pass 3d: the deformable models share ONE pyramid convention (``deformable`` preset of the central planner).

Correctness tests, not golden equality with the removed legacy presets: corner-aligned grid whose spacing is exactly
``s (N - 1) / (n - 1)``, native grid at level 1, anti-aliasing (sigma = 0.5 x the axis's coarsening factor), thin axes, each image on
its own grid, one CFL shrink ratio from the actual per-axis spacings, and per-model regression tests for the inconsistencies of
docs/PYRAMID.md Section 10 (each asserts the corrected behaviour and, where cheap, that the old formula gives a different number).
"""
import importlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from syntx.pyramid import (MIN_VOXELS_PER_AXIS, build_pyramid, cfl_shrink_ratio, level_spacing, plan_levels)

CASES = [((64, 64, 64), (1.0, 1.0, 1.0)), ((32, 64, 64), (1.0, 1.0, 2.0)), ((14, 64, 64), (1.0, 1.0, 3.0)),
         ((40, 60, 48), (1.5, 1.0, 1.0)), ((3, 64, 64), (1.0, 1.0, 1.0))]


def _smooth(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.interpolate(torch.rand(1, 1, 5, 5, 5, generator=g), size=shape, mode='trilinear', align_corners=True)


def _ramp(shape, ax):
    idx = torch.arange(shape[ax], dtype=torch.float32)
    sh = [1, 1, 1, 1, 1]; sh[2 + ax] = -1
    return (idx.reshape(sh) * torch.ones(1, 1, *shape)).contiguous()


# ----------------------------------------------------------------------------------------------- the grid convention
@pytest.mark.parametrize("shape, sp", CASES)
def test_deformable_grid_is_corner_aligned_with_exact_spacing_and_native_level_one(shape, sp):
    for lv in plan_levels(shape, sp, [8, 4, 2, 1], 'deformable'):
        for n, m, f, off, s, ws in zip(shape[::-1], lv.shape_zyx[::-1], lv.f_xyz, lv.off_xyz, sp, lv.spacing_xyz):
            assert off == 0.0
            assert m == n or abs(f - (n - 1.0) / (m - 1.0)) < 1e-12              # first and last samples on the first and last native samples
            assert abs(ws - s * (n - 1.0) / (m - 1.0)) < 1e-9 if m > 1 else True  # the spacing that CFL / regularisation / Jacobian use
            assert m >= min(MIN_VOXELS_PER_AXIS, n)                                # the ONE floor; a thin axis keeps its native samples
        if lv.level == 1:
            assert lv.is_identity and lv.shape_zyx == tuple(shape)                 # a warp field lives on the native grid at level 1
        assert list(lv.spacing_xyz) == pytest.approx(level_spacing(sp, shape, lv.shape_zyx))      # bookkeeping == actual grid


@pytest.mark.parametrize("shape, sp", CASES[:4])
def test_deformable_corner_samples_and_ramp_coordinates(shape, sp):
    for L in (2, 4):
        for ax in range(3):
            lv = build_pyramid(_ramp(shape, ax), sp, [L], 'deformable', sigma=0.0)[0]
            a = 2 - ax
            got = lv.tensor[0, 0].movedim(ax, -1).reshape(-1, lv.shape[ax])[0].double()
            expect = torch.arange(lv.shape[ax], dtype=torch.float64) * lv.f[a]
            assert abs(float(got[0])) < 1e-5 and abs(float(got[-1]) - (shape[ax] - 1)) < 1e-4     # corners map to corners
            assert float((got - expect).abs().max()) < 1e-4


def test_deformable_never_upsamples_a_thick_axis_and_levels_are_millimetre_isotropic():
    p = plan_levels((14, 64, 64), (1.0, 1.0, 3.0), [4, 2, 1], 'deformable')
    assert p[0].shape_zyx == (14, 17, 17) and p[0].spacing_xyz[2] == 3.0             # 3 mm slab axis (3 < 4 mm target but 14 voxels < floor) native
    assert p[2].is_identity
    q = plan_levels((32, 64, 64), (1.0, 1.0, 2.0), [4], 'deformable')[0]
    assert q.shape_zyx == (17, 17, 17)                                              # 2 mm slices and 1 mm pixels reach one ~4 mm isotropic grid
    assert max(q.spacing_xyz) / min(q.spacing_xyz) < 1.03


def test_one_floor_for_every_model():
    from syntx import pyramid
    assert pyramid.PRESETS['deformable'].floor == pyramid.MIN_VOXELS_PER_AXIS == 16
    for shape in [(64, 64, 64), (40, 33, 70)]:
        for lv in plan_levels(shape, (1.0,) * 3, [16, 8, 4], 'deformable'):
            assert min(lv.shape_zyx) >= min(16, min(shape))


# ----------------------------------------------------------------------------------------------- the smoother
def test_default_sigma_is_half_the_axis_coarsening_factor_and_never_on_native_axes():
    for lv in plan_levels((14, 64, 64), (1.0, 1.0, 3.0), [4, 2], 'deformable'):
        for sg, f in zip(lv.sigma_vox_zyx[::-1], lv.f_xyz):
            assert sg == (pytest.approx(0.5 * f) if f > 1 + 1e-9 else 0.0)
        assert lv.sigma_vox_zyx[0] == 0.0                                           # the native thin axis is not smoothed
    # an explicit sigma never touches an axis that is not coarsened either
    assert plan_levels((14, 64, 64), (1.0, 1.0, 3.0), [4], 'deformable', sigma=2.0)[0].sigma_vox_zyx[0] == 0.0


def test_alias_suppression_nyquist_pattern_and_constant_preserved():
    x = torch.zeros(1, 1, 64, 64, 64)
    x += ((torch.arange(64) % 2).float() * 2 - 1).reshape(1, 1, 1, 1, 64)           # period-2 pattern: pure aliasing hazard at level 4
    aa = build_pyramid(x, (1.0,) * 3, [4], 'deformable')[0].tensor
    naive = build_pyramid(x, (1.0,) * 3, [4], 'deformable', sigma=0.0)[0].tensor
    assert float(naive.abs().max()) > 0.5                                           # point sampling aliases the pattern to a large false signal
    assert float(aa[..., 2:-2].abs().max()) < 0.05                                  # the anti-aliased level does not
    c = torch.full((1, 1, 40, 40, 40), 0.3)
    assert float((build_pyramid(c, (1.0,) * 3, [4], 'deformable')[0].tensor - 0.3).abs().max()) < 1e-6   # no zero-padding darkening


def test_default_smoothing_keeps_a_ramp_in_the_interior():
    lv = build_pyramid(_ramp((64, 64, 64), 2), (1.0,) * 3, [4], 'deformable')[0]
    got = lv.tensor[0, 0, 8, 8].double(); expect = torch.arange(lv.shape[2], dtype=torch.float64) * lv.f[0]
    assert float((got - expect)[3:-3].abs().max()) < 1e-3


# ----------------------------------------------------------------------------------------------- thin axes
@pytest.mark.parametrize("shape,level,want", [((3, 64, 64), 4, (3, 17, 17)), ((3, 64, 64), 8, (3, 16, 16)), ((11, 128, 128), 8, (11, 17, 17)),
                                              ((2, 40, 40), 4, (2, 16, 16))])
def test_thin_axis_no_crash_no_collapse(shape, level, want):
    out = build_pyramid(torch.rand(1, 1, *shape), (1.0,) * 3, [level], 'deformable')[0].tensor
    assert tuple(out.shape[2:]) == want and torch.isfinite(out).all()


# ----------------------------------------------------------------------------------------------- each image on its own grid
def _blob(shape, sp, centre_mm, sigma_mm=6.0):
    ax = [torch.arange(n, dtype=torch.float64) * s for n, s in zip(shape[::-1], sp)]       # x, y, z
    X, Y, Z = torch.meshgrid(*ax, indexing='ij')                                             # (x, y, z)
    g = torch.exp(-((X - centre_mm[0]) ** 2 + (Y - centre_mm[1]) ** 2 + (Z - centre_mm[2]) ** 2) / (2 * sigma_mm ** 2))
    return g.permute(2, 1, 0)[None, None].float().contiguous()                              # (z, y, x)


@pytest.mark.parametrize("level", [2, 4])
def test_moving_image_on_its_own_grid_keeps_physical_coordinates(level):
    """A blob at one physical position, sampled on a fixed grid (1 mm iso) and on a moving grid of another shape and spacing: at every
    level the intensity centroid in physical coordinates (from each Level's own f, off) is the same point."""
    c = (30.0, 35.0, 28.0)
    fixed = _blob((60, 64, 64), (1.0, 1.0, 1.0), c)
    moving = _blob((36, 50, 70), (0.9, 1.3, 1.7), c)
    cents = []
    for img, sp in ((fixed, (1.0, 1.0, 1.0)), (moving, (0.9, 1.3, 1.7))):
        lv = build_pyramid(img, sp, [level], 'deformable', reference_spacing=(1.0, 1.0, 1.0))[0]
        pc = lv.physical_coordinates(); w = lv.tensor[0, 0].numpy().astype(np.float64)
        cents.append((pc * w[..., None]).reshape(-1, 3).sum(0) / w.sum())
    assert np.abs(cents[0] - np.array(c)).max() < 0.05 and np.abs(cents[1] - np.array(c)).max() < 0.05
    if level == 2:                                                                          # each image keeps its own level grid (not the fixed one)
        assert build_pyramid(moving, (0.9, 1.3, 1.7), [2], 'deformable', reference_spacing=(1.0,) * 3)[0].shape != build_pyramid(fixed, (1.0,) * 3, [2], 'deformable')[0].shape


# ----------------------------------------------------------------------------------------------- CFL shrink ratio
def test_cfl_shrink_ratio_uses_every_axis_not_z_only():
    sp = (1.0, 1.0, 3.0); shape = (11, 64, 64)
    fine = level_spacing(sp, shape, shape)
    for L in (2, 4):
        lv = plan_levels(shape, sp, [L], 'deformable')[0]
        r = cfl_shrink_ratio(lv.spacing_xyz, fine)
        z_only = lv.shape_zyx[0] / shape[0]
        assert r == pytest.approx(min(f / c for f, c in zip(fine, lv.spacing_xyz)))
        assert r < 1.0
        if L == 4:
            assert z_only == 1.0 and abs(r - 1.0) > 0.2                             # the old z-only rule gave 1.0 (no shrink) for this slab
    assert cfl_shrink_ratio(fine, fine) == 1.0


def _syn_pair(shape, seed=0):
    return _smooth(shape, seed), torch.roll(_smooth(shape, seed), (1, 2, -1), (2, 3, 4))


def test_syn_cfl_shrink_ratio_follows_the_actual_spacings_on_a_thin_slab():
    from syntx.syn import SyNTo
    shape = (11, 48, 48); sp = [1.0, 1.0, 3.0]
    f, m = _syn_pair(shape)
    model = SyNTo(dim=3, grid_shape=shape, spacing=sp)
    model.fit(f, m, fixed_spacing=sp, moving_spacing=sp, levels=[4, 2, 1], epochs_per_level=[1, 1, 1], verbose=False)
    info = model.pyramid_info
    assert [i['level'] for i in info] == [4, 2, 1]
    fine = level_spacing(sp, shape, shape)
    for i in info:
        assert tuple(i['fixed_spacing']) == pytest.approx(tuple(level_spacing(sp, shape, i['fixed_shape'])))
        assert i['shrink_ratio'] == pytest.approx(cfl_shrink_ratio(i['fixed_spacing'], fine))
    assert info[-1]['shrink_ratio'] == 1.0
    z_only = info[0]['fixed_shape'][0] / shape[0]
    assert abs(info[0]['shrink_ratio'] - z_only) > 0.1                              # the z-only rule (11 slices stay 11: 1.0) was wrong here


def test_syn_moving_image_has_its_own_level_grid_and_spacing():
    from syntx.syn import SyNTo
    f = _smooth((32, 40, 36)); m = _smooth((28, 36, 40), seed=1)
    sp, msp = [1.0, 1.0, 1.0], [1.2, 1.0, 0.8]
    model = SyNTo(dim=3, grid_shape=(32, 40, 36), spacing=sp)
    model.fit(f, m, fixed_spacing=sp, moving_spacing=msp, levels=[2, 1], epochs_per_level=[1, 1], verbose=False)
    for i in model.pyramid_info:
        assert tuple(i['moving_spacing']) == pytest.approx(tuple(level_spacing(msp, (28, 36, 40), i['moving_shape'])))
    assert model.pyramid_info[0]['moving_shape'] != model.pyramid_info[0]['fixed_shape']


# ----------------------------------------------------------------------------------------------- tvf
def test_tvf_level_spacing_is_the_actual_velocity_grid_spacing():
    """tvf normalised the CFL step with ``sp * level`` while the velocity grid has spacing ``sp (N - 1) / (n - 1)``."""
    from syntx.tvf import TVFModel
    shape = (48, 48, 48); sp = [1.0, 1.0, 1.0]
    f, m = _syn_pair(shape)
    model = TVFModel(dim=3, image_shape=shape, velocity_shape=shape, spacing=sp, moving_shape=shape, moving_spacing=sp)
    model.fit(f, m, levels=[4, 2, 1], epochs_per_level=[1, 1, 1])
    info = model.pyramid_info
    assert [i['level'] for i in info] == [4, 2, 1]
    for i in info:
        want = level_spacing(sp, shape, i['velocity_shape'])
        assert tuple(i['velocity_spacing']) == pytest.approx(tuple(want))
        assert tuple(i['image_shape']) == tuple(i['velocity_shape'])               # one grid for the image level and the velocity
    assert info[0]['velocity_spacing'][0] != pytest.approx(4.0)                    # the old sp * level (4.0) differs from (N-1)/(n-1) = 3.9...
    assert info[-1]['velocity_spacing'] == pytest.approx((1.0, 1.0, 1.0)) and info[-1]['shrink_ratio'] == 1.0


# ----------------------------------------------------------------------------------------------- greedy
def test_greedy_levels_follow_the_spacing_and_moving_gets_its_own_sigma(monkeypatch):
    G = importlib.import_module("syntx.greedy")
    fixed, moving = _smooth((40, 48, 36)), _smooth((30, 44, 52), seed=1)
    seen = {}
    real = G.build_pyramid

    def spy(image, spacing, *a, **k):
        out = real(image, spacing, *a, **k)
        seen[tuple(image.shape[2:])] = (out[0].plan.sigma_vox_zyx, tuple(out[0].tensor.shape[2:]), tuple(spacing))
        return out
    monkeypatch.setattr(G, 'build_pyramid', spy)
    model = G.GreedyRegistrationModel(dim=3, similarity_metric='cc2', lncc_radius=2)
    model.fit(fixed, moving, torch.eye(3, 4)[None], scales=[4], iterations=[1], spacing=(1.0, 1.0, 1.0), moving_spacing=(1.0, 1.0, 1.0))
    for img in (fixed, moving):
        want = plan_levels(img.shape[2:], (1, 1, 1), [4], 'deformable')[0]
        assert seen[tuple(img.shape[2:])][:2] == (want.sigma_vox_zyx, want.shape_zyx)
    assert seen[(40, 48, 36)][0] != seen[(30, 44, 52)][0]


def test_greedy_thick_axis_is_not_shrunk_with_the_thin_ones_and_thin_axis_does_not_crash():
    from syntx.greedy import GreedyRegistrationModel
    fixed, moving = _smooth((12, 48, 48)), torch.roll(_smooth((12, 48, 48)), (1, 2, -1), (2, 3, 4))
    model = GreedyRegistrationModel(dim=3, similarity_metric='cc2', lncc_radius=2)
    warp = model.fit(fixed, moving, torch.eye(3, 4)[None], scales=[2, 1], iterations=[2, 2], spacing=(1.0, 1.0, 3.0))
    assert tuple(warp.shape) == (1, 12, 48, 48, 3) and torch.isfinite(warp).all()


# ----------------------------------------------------------------------------------------------- tvf_adj
def test_tvf_adj_spacing_is_corner_aligned():
    """tvf_adj used ``sp * (N / n)`` against the ``(N - 1) / (n - 1)`` grid of its images and velocity."""
    ants = pytest.importorskip("ants")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from syntx.tvf_adj import TVFRegistrationAdjoint
    a = ants.from_numpy(_smooth((40, 40, 40))[0, 0].numpy().T.copy(), spacing=(1.0, 1.0, 1.0))
    b = ants.from_numpy(torch.roll(_smooth((40, 40, 40)), (1, 1, 1), (2, 3, 4))[0, 0].numpy().T.copy(), spacing=(1.0, 1.0, 1.0))
    model = TVFRegistrationAdjoint(a, b, levels=[2, 1], reg_iterations=[1, 1])
    model.fit()
    for i in model.pyramid_info:
        want = level_spacing(model.spacing[::-1], model.shape, i['shape'])[::-1]
        assert tuple(i['spacing_zyx']) == pytest.approx(tuple(want))
    assert model.pyramid_info[0]['spacing_zyx'][0] == pytest.approx(1.0 * 39 / 20)  # 21 samples over 39 mm; not N / n = 40 / 21 = 1.905


# ----------------------------------------------------------------------------------------------- JAX
def test_pyramid_level_jax_smooths_every_axis_and_matches_torch():
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from syntx.syn_jax import pyramid_level_jax
    x = np.zeros((1, 1, 33, 33, 33), np.float32); x[0, 0, 16, 16, 16] = 1.0
    out = np.asarray(pyramid_level_jax(jnp.asarray(x), 2, (1.0,) * 3, sigma=1.5))[0, 0]
    c = np.array(out.shape) // 2
    prof = [out[tuple(slice(None) if i == a else c[i] for i in range(3))] for a in range(3)]
    for a in (1, 2):
        assert np.allclose(prof[0], prof[a], atol=1e-6), f"axis {a} is smoothed differently from axis 0"
    ref = build_pyramid(torch.from_numpy(x), (1.0,) * 3, [2], 'deformable', sigma=1.5)[0].tensor.numpy()
    assert out.shape == ref[0, 0].shape and np.abs(out - ref[0, 0]).max() < 1e-5


def test_resize_velocity_jax_is_align_corners_not_half_pixel():
    """tvf_jax resized the velocity with ``jax.image.resize`` (half-pixel): a linear ramp comes back shifted by a fraction of a voxel
    and its end values are not the end values; the level spacing assumes align-corners."""
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from syntx.tvf_jax import resize_velocity_jax
    n, m = 33, 17
    ramp = jnp.broadcast_to(jnp.arange(n, dtype=jnp.float32).reshape(1, 1, 1, n, 1), (1, n, n, n, 3))
    got = resize_velocity_jax(ramp, (m, m, m), 3)
    expect = jnp.arange(m, dtype=jnp.float32) * (n - 1) / (m - 1)
    assert float(jnp.abs(got[0, 3, 3, :, 0] - expect).max()) < 1e-4 and got.shape == (1, m, m, m, 3)
    half = jax.image.resize(ramp[0, :, :, :, 0], (m, m, m), method='linear')
    assert float(jnp.abs(half[3, 3] - expect).max()) > 0.4                           # the old convention is off by ~0.5 voxel at the ends


def test_syn_jax_moving_image_has_its_own_level_grid():
    pytest.importorskip("jax")
    from syntx.syn_jax import SyNTo as SyNToJax
    f = _smooth((32, 40, 36)).numpy().astype(np.float32); m = _smooth((28, 36, 40), seed=1).numpy().astype(np.float32)
    sp, msp = [1.0, 1.0, 1.0], [1.2, 1.0, 0.8]
    model = SyNToJax(dim=3, grid_shape=(32, 40, 36), spacing=sp)
    model.fit(f, m, fixed_spacing=sp, moving_spacing=msp, levels=[2, 1], epochs_per_level=[1, 1], verbose=False)
    for i in model.pyramid_info:
        assert tuple(i['moving_spacing']) == pytest.approx(tuple(level_spacing(msp, (28, 36, 40), i['moving_shape'])))
    assert model.pyramid_info[0]['moving_shape'] != model.pyramid_info[0]['fixed_shape']       # was resampled onto the FIXED level grid


@pytest.mark.parametrize("which", ["syngs", "tvf"])
def test_jax_models_build_level_images_through_the_central_pyramid(which, monkeypatch):
    pytest.importorskip("jax")
    if which == "syngs":
        mod = importlib.import_module("syntx.syngs_jax")
        model = mod.GeodesicShootingModelJAX(dim=3, image_shape=(24,) * 3, velocity_shape=(24,) * 3)
    else:
        mod = importlib.import_module("syntx.tvf_jax")
        model = mod.TVFModelJAX(dim=3, image_shape=(24,) * 3, velocity_shape=(24,) * 3)
    f, m = (a.numpy().astype(np.float32) for a in _syn_pair((24,) * 3))
    seen = []
    real = mod.pyramid_level_jax
    monkeypatch.setattr(mod, 'pyramid_level_jax', lambda image, level, *a, **k: (seen.append((level, k.get('sigma'))), real(image, level, *a, **k))[1])
    model.fit(f, m, levels=[2, 1], epochs_per_level=[1, 1], affine_epochs=1, lncc_radius=2, smooth_pyramid=True)
    assert [lv for lv, _ in seen] == [2, 2, 1, 1] and all(sg is None for _, sg in seen)      # fixed + moving at every level, planner's sigma



# ----------------------------------------------------------------------------------------------- tvf CFL scale (found by the validation)
def test_tvf_cfl_step_is_scaled_up_to_the_step_length():
    """The CFL scale was clamped at 1, so a small gradient was never scaled to the step length and TVF stayed at the identity."""
    from syntx.tvf import TVFModel
    shape = (24, 24, 24)
    f, m = _syn_pair(shape)
    model = TVFModel(dim=3, image_shape=shape, velocity_shape=shape, spacing=[1.0] * 3, moving_shape=shape, moving_spacing=[1.0] * 3)
    model.fit(f, m, levels=[1], epochs_per_level=[3], optimizer_type='cfl', cfl_step=0.5, cfl_momentum=0.0)
    assert float(model.velocity.abs().max()) > 0.1        # with the clamp: ~1e-3
