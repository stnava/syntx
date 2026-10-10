"""Golden and property tests of the CENTRAL pyramid (``syntx.pyramid.plan_levels`` / ``build_pyramid``).

Golden: each legacy preset reproduces the incumbent implementation that is still live in the package
(``build_image_pyramid``, ``build_anti_aliased_pyramid``, ``robust_affine._mm_pyramid_level``), the robust_affine /
motion_batched / greedy / tvf_adj inline operators (re-stated here from their source lines), bit-identically where the operator is
the same torch call, otherwise to the stated tolerance. ``mm_iso`` differs from ``_mm_pyramid_level`` only by the documented
coordinate-bias fix (centroid-exact area average) and by the smoothing of the central module (replicate padding, ITK Bessel
kernel instead of zero padding and a truncated Gaussian). Properties hold for every preset.
"""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from syntx import pyramid as pm
from . import legacy_pyramid_reference as legacy
from syntx.pyramid import (HARD_MIN_VOXELS, MIN_VOXELS_PER_AXIS, PRESETS, PyramidLevelError, build_pyramid, plan_levels,
                           warp_to_moving_norm, level_spacing, cap_ants_image)

ALL = sorted(PRESETS)
SHAPES_3D = [(32, 48, 40), (31, 45, 50), (40, 40, 40), (16, 64, 64)]          # isotropic-ish, odd sizes
LEVELS = [8, 4, 3, 2, 1]


def _rand(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 1, *shape, generator=g)


def _smooth(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    small = torch.rand(1, 1, 5, 5, 5, generator=g)
    return F.interpolate(small, size=shape, mode='trilinear', align_corners=True)


# ------------------------------------------------------------------------------------------------ golden: legacy_interp
@pytest.mark.parametrize("shape", SHAPES_3D + [(33, 47)])
def test_legacy_interp_is_build_image_pyramid_bit_identical(shape):
    x = _rand(shape)
    sp = (1.0,) * len(shape)
    levels = [3, 2, 1] if len(shape) == 2 else LEVELS
    inc = legacy.build_image_pyramid(x, sp, levels)
    new = build_pyramid(x, sp, levels, 'legacy_interp')
    for a, lv in zip(inc, new):
        assert a.shape == lv.tensor.shape
        assert torch.equal(a, lv.tensor)                                       # same torch ops: bit-identical
    for lv in new:                                                              # corner-aligned bookkeeping
        for n, m, f, off in zip(lv.native_shape[::-1], lv.shape[::-1], lv.f, lv.off):
            assert off == 0.0 and (m == n or abs(f - (n - 1.0) / (m - 1.0)) < 1e-12)


@pytest.mark.parametrize("kwargs", [dict(smoothing_sigmas=[1.0, 0.0, 0.5], sigma_mode='voxel'),
                                    dict(smoothing_sigmas=1.5, sigma_mode='voxel'),
                                    dict(smoothing_sigmas=[2.0, 1.0, 0.5], sigma_mode='physical')])
def test_legacy_interp_explicit_sigma_modes_bit_identical(kwargs):
    x = _rand((40, 44, 48)); sp = (0.8, 1.0, 2.0)
    levels = [4, 2, 1]
    inc = legacy.build_image_pyramid(x, sp, levels, **kwargs)
    new = build_pyramid(x, sp, levels, 'legacy_interp', sigma=kwargs['smoothing_sigmas'], sigma_mode=kwargs['sigma_mode'])
    for a, lv in zip(inc, new):
        assert torch.equal(a, lv.tensor)


# ------------------------------------------------------------------------------------------------ golden: legacy_floor16
@pytest.mark.parametrize("shape", SHAPES_3D + [(6, 40, 40), (10, 70, 33)])     # thin axes: incumbent up-samples them to 16
def test_legacy_floor16_is_build_anti_aliased_pyramid_bit_identical(shape):
    x = _rand(shape)
    inc = legacy.build_anti_aliased_pyramid(x, LEVELS)
    new = build_pyramid(x, (1.0,) * 3, LEVELS, 'legacy_floor16')
    for a, lv in zip(inc, new):
        assert a.shape == lv.tensor.shape and torch.equal(a, lv.tensor)
    for lv in new:
        assert all(m >= min(MIN_VOXELS_PER_AXIS, 10**9) or lv.level <= 1 for m in lv.shape)


@pytest.mark.parametrize("min_size", [8, 4])
def test_legacy_floor16_other_floors_and_tvf_adj_no_smoothing(min_size):
    x = _rand((31, 45, 50))
    inc = legacy.build_anti_aliased_pyramid(x, [4, 2], min_size=min_size)
    new = build_pyramid(x, (1.0,) * 3, [4, 2], 'legacy_floor16', floor=min_size)
    assert all(torch.equal(a, lv.tensor) for a, lv in zip(inc, new))
    # tvf_adj.py:298 (floor 8, no anti-alias) and scattered/solver.py (floor 4): F.interpolate(align_corners) to max(s // level, floor)
    for L in (4, 2):
        size = [max(s // L, min_size) for s in x.shape[2:]]
        ref = F.interpolate(x, size=size, mode='trilinear', align_corners=True)
        got = build_pyramid(x, (1.0,) * 3, [L], 'legacy_floor16', floor=min_size, sigma=0.0)[0].tensor
        assert torch.equal(ref, got)


def test_floor16_moving_image_gets_its_own_sigma_and_size():
    """greedy used the FIXED image's sigma for the moving image; the central module computes both from each image's own shape."""
    fixed, moving = _rand((60, 60, 60)), _rand((30, 80, 50), seed=1)
    pf = plan_levels(fixed.shape[2:], (1, 1, 1), [4], 'legacy_floor16')[0]
    pmv = plan_levels(moving.shape[2:], (1, 1, 1), [4], 'legacy_floor16')[0]
    assert pf.shape_zyx == (16, 16, 16) and pmv.shape_zyx == (16, 20, 16)
    assert pf.sigma_vox_zyx != pmv.sigma_vox_zyx
    assert torch.equal(build_pyramid(moving, (1, 1, 1), [4], 'legacy_floor16')[0].tensor, legacy.build_anti_aliased_pyramid(moving, [4])[0])


# ------------------------------------------------------------------------------------------------ golden: legacy_avgpool
@pytest.mark.parametrize("shape", SHAPES_3D)
def test_legacy_avgpool_is_avg_pool3d_bit_identical(shape):
    x = _rand(shape)
    for L in (1, 2, 3, 4):
        ref = F.avg_pool3d(x, kernel_size=L, stride=L) if L > 1 else x
        lv = build_pyramid(x, (1.0,) * 3, [L], 'legacy_avgpool')[0]
        assert torch.equal(ref, lv.tensor)
        assert list(lv.f) == [float(L)] * 3 and list(lv.off) == [(L - 1) / 2.0] * 3        # _pool_level's bookkeeping


def test_legacy_avgpool_2d_and_motion_stack():
    x = _rand((33, 47))[:, :, :, :]
    assert torch.equal(build_pyramid(x, (1.0, 1.0), [2], 'legacy_avgpool')[0].tensor, F.avg_pool2d(x, 2, 2))
    stack = torch.rand(5, 1, 20, 24, 28)                                         # motion_batched: a frame stack, same operator
    assert torch.equal(build_pyramid(stack, (1.0,) * 3, [4], 'legacy_avgpool')[0].tensor, F.avg_pool3d(stack, 4, 4))


def test_legacy_avgpool_smoothing_difference_is_only_kernel_and_border():
    """_pool_level smooths with a zero-padded truncated Gaussian (darkens the border); the central module uses replicate padding and
    the ITK discrete Gaussian: interior agrees to 5e-3 on a smooth image, and a constant image stays constant (the old one darkens)."""
    from syntx.landmarks.blob import _separable_gaussian3d
    x = _smooth((32, 48, 40))
    old = F.avg_pool3d(_separable_gaussian3d(x, 1.0), 2, 2)
    new = build_pyramid(x, (1.0,) * 3, [2], 'legacy_avgpool', sigma=1.0)[0].tensor
    assert float((old - new)[..., 3:-3, 3:-3, 3:-3].abs().max()) < 5e-3
    const = torch.full((1, 1, 32, 32, 32), 0.7)
    assert float(F.avg_pool3d(_separable_gaussian3d(const, 1.0), 2, 2).min()) < 0.69                # the old border darkening
    assert float((build_pyramid(const, (1.0,) * 3, [2], 'legacy_avgpool', sigma=1.0)[0].tensor - 0.7).abs().max()) < 1e-6


# ------------------------------------------------------------------------------------------------ golden: mm_iso
def _old_mm(x, sp, L, sigma_mm=0.0):
    from syntx.robust_affine import _mm_pyramid_level
    return _mm_pyramid_level(x, sp, L * min(sp), sigma_mm, 2 ** 24)


@pytest.mark.parametrize("shape, sp", [((16, 40, 32), (1.0, 1.0, 1.0)), ((16, 40, 32), (0.75, 0.75, 3.5)), ((32, 64, 48), (1.0, 1.0, 2.0)),
                                       ((8, 40, 40), (0.75, 0.75, 6.0))])
@pytest.mark.parametrize("L", [1, 2, 4])
def test_mm_iso_equals_mm_pyramid_level_bit_identical_when_f_is_integral(shape, sp, L):
    """Integer area factors have no fractional end weights, so the corrected matrix IS the old one: same bytes."""
    x = _rand(shape)
    old, f, off = _old_mm(x, sp, L)
    lv = build_pyramid(x, sp, [L], 'mm_iso', floor=HARD_MIN_VOXELS)[0]
    assert tuple(old.shape[2:]) == lv.shape and np.allclose(f, lv.f, atol=1e-12) and np.allclose(off, lv.off, atol=1e-12)
    assert torch.equal(old, lv.tensor)


@pytest.mark.parametrize("shape, sp", [((17, 41, 33), (1.0, 1.0, 1.0)), ((11, 90, 77), (0.75, 0.75, 3.5)), ((19, 50, 61), (1.0, 1.3, 2.0))])
@pytest.mark.parametrize("L", [2, 3, 5])
def test_mm_iso_fractional_f_same_grid_and_only_the_bias_fix_differs(shape, sp, L):
    x = _smooth(shape)
    old, f, off = _old_mm(x, sp, L)
    lv = build_pyramid(x, sp, [L], 'mm_iso', floor=HARD_MIN_VOXELS)[0]
    assert tuple(old.shape[2:]) == lv.shape and np.allclose(f, lv.f, atol=1e-12) and np.allclose(off, lv.off, atol=1e-12)
    # on a smooth image the bias fix moves values by (bias x gradient) only: bounded well under the image range
    assert float((old - lv.tensor).abs().max()) < 2e-2


def _ramp(shape, ax_t):
    idx = torch.arange(shape[ax_t], dtype=torch.float32)
    sh = [1, 1, 1, 1, 1]; sh[2 + ax_t] = -1
    return (idx.reshape(sh) * torch.ones(1, 1, *shape)).contiguous()


def _ramp_error_vox(builder, shape, sp, L):
    """max over axes of |ramp output - (j f + off)| / f, working voxels, whole axis including the edges."""
    worst = 0.0
    for ax_t in range(3):
        out, f, off = builder(_ramp(shape, ax_t), sp, L)
        a = 2 - ax_t
        m = out.shape[2 + ax_t]
        sel = [0, 0, 0, 0, 0]
        for o in range(3):
            sel[2 + o] = slice(None) if o == ax_t else out.shape[2 + o] // 2
        got = out[tuple(sel)].double()
        expect = torch.arange(m, dtype=torch.float64) * f[a] + off[a]
        worst = max(worst, float((got - expect).abs().max()) / f[a])
    return worst


def _new_builder(x, sp, L):
    lv = build_pyramid(x, sp, [L], 'mm_iso', floor=HARD_MIN_VOXELS)[0]
    return lv.tensor, lv.f, lv.off


def _old_builder(x, sp, L):
    return _old_mm(x, sp, L)


RAMP_CONFIGS = [((33, 61, 61), (1.0, 1.0, 1.0), 1.5), ((40, 40, 40), (1.0, 1.0, 1.0), 2.0), ((17, 100, 100), (0.75, 0.75, 3.5), 2.0),
                ((50, 61, 40), (1.0, 1.0, 1.0), 3.0)]


def test_coordinate_bias_old_vs_new_ramp():
    """Pass 1 criterion C6 (ramp through (f, off) within 0.05 working voxel): the old area average is off by ~0.06 working voxel at
    f ~ 1.5 (fractional end weights); the centroid-exact matrix is below 1e-4 everywhere (the requirement was < 0.01)."""
    olds, news = [], []
    for shape, sp, _ in RAMP_CONFIGS:
        for L in (2, 3, 4, 5):
            olds.append(_ramp_error_vox(_old_builder, shape, sp, L))
            news.append(_ramp_error_vox(_new_builder, shape, sp, L))
    for L in (4.0 / 3.0,):                                   # f = 4/3: the worst case of the old matrix (0.0625 working voxel)
        for shape, sp in [((48, 60, 60), (0.75, 0.75, 0.75)), ((96, 60, 60), (0.75, 0.75, 0.75))]:
            olds.append(_ramp_error_vox(_old_builder, shape, sp, L))
            news.append(_ramp_error_vox(_new_builder, shape, sp, L))
    print(f"ramp bias, working voxels: old max {max(olds):.4f}, new max {max(news):.2e}")
    assert max(olds) > 0.05                       # the test can see the old defect
    assert max(news) < 1e-4 and max(news) < 0.01


@pytest.mark.parametrize("n, m", [(100, 67), (61, 40), (64, 32), (50, 20), (33, 22)])
def test_coordinate_bias_old_vs_new_gaussian_bump_centroid(n, m):
    """Delta test with a smoothed delta (a Gaussian bump of sigma 2 f; a literal one-sample delta only measures the sampling phase,
    0.25 working voxel even for the exact integer f = 2): the centroid of the output in full-res coordinates matches the input's."""
    from syntx.robust_affine import _axis_resample_matrix
    f = n / m; off = (f - 1) / 2; e = np.arange(m) * f + off

    def cen(W, p, sg):
        y = W @ np.exp(-0.5 * ((np.arange(n) - p) / sg) ** 2)
        return (y * e).sum() / y.sum()
    Wn, Wo = pm.area_matrix(n, m, f), _axis_resample_matrix(n, m, f, off)
    ps = np.linspace(n * 0.4, n * 0.6, 17)
    err_new = max(abs(cen(Wn, p, 2 * f) - p) for p in ps) / f
    err_old = max(abs(cen(Wo, p, 2 * f) - p) for p in ps) / f
    assert err_new < 0.01 and err_old < 0.01 + 0.01                              # the bump is symmetric: both are small, the ramp separates them


def test_area_matrix_rows_normalised_nonnegative_and_exact_for_integer_f():
    from syntx.robust_affine import _axis_resample_matrix
    for n, m in [(64, 32), (60, 15), (100, 67), (17, 8), (301, 7)]:
        f = n / m
        W = pm.area_matrix(n, m, f)
        assert W.min() >= 0 and np.allclose(W.sum(1), 1.0)
        assert np.allclose(W @ np.arange(n), np.arange(m) * f + (f - 1) / 2, atol=1e-9)
        if f == int(f):
            assert np.array_equal(W, _axis_resample_matrix(n, m, f, (f - 1) / 2))


def test_mm_iso_smoothing_matches_old_interior_on_smooth_image():
    x = _smooth((32, 48, 40))
    old, f, off = _old_mm(x, (1.0,) * 3, 4, sigma_mm=1.0)
    new = build_pyramid(x, (1.0,) * 3, [4], 'mm_iso', sigma=1.0, sigma_mode='physical', floor=HARD_MIN_VOXELS)[0].tensor
    assert old.shape == new.shape
    assert float((old - new)[..., 1:-1, 1:-1, 1:-1].abs().max()) < 5e-3


def test_mm_iso_never_smooths_an_axis_that_is_not_coarsened():
    p = plan_levels((11, 128, 128), (0.75, 0.75, 3.5), [2], 'mm_iso', sigma=1.0, sigma_mode='voxel')[0]
    # level 2 = 1.5 mm: in-plane coarsened, the 3.5 mm slice axis is finer than the target (linear): not smoothed
    assert p.sigma_vox_zyx == (0.0, 1.0, 1.0)


# ------------------------------------------------------------------------------------------------ properties, every preset
THIN = [(2, 40, 40), (3, 64, 64), (11, 128, 128), (5, 33, 70), (64, 64, 64), (7, 33, 65)]


@pytest.mark.parametrize("policy", ALL)
@pytest.mark.parametrize("shape", THIN)
def test_thin_axis_never_zero_or_one_voxel_never_raises(policy, shape):
    x = _rand(shape)
    for sp in [(1.0, 1.0, 1.0), (0.43, 0.43, 3.5)]:
        for lv in build_pyramid(x, sp, [1, 2, 4, 8, 16], policy):
            assert min(lv.shape) >= HARD_MIN_VOXELS, (policy, shape, lv.level, lv.shape)
            assert tuple(lv.tensor.shape[2:]) == lv.shape
            assert torch.isfinite(lv.tensor).all()


@pytest.mark.parametrize("policy", ALL)
def test_single_sample_axis_passes_through(policy):
    x = _rand((1, 40, 40))
    for lv in build_pyramid(x, (1.0, 1.0, 1.0), [1, 4], policy):
        assert lv.shape[0] == 1 and torch.isfinite(lv.tensor).all()


@pytest.mark.parametrize("policy", ALL)
def test_floor_keeps_a_thin_axis_native_except_greedys_upsampling(policy):
    pol = PRESETS[policy]
    for lv in build_pyramid(_rand((9, 64, 64)), (1.0, 1.0, 1.0), [8], policy, floor=12):
        if pol.floor_upsamples:
            assert lv.shape[0] == 12                                              # greedy semantics: up-sampled to the floor
        else:
            assert lv.shape[0] == 9                                               # native sampling kept


@pytest.mark.parametrize("policy", ALL)
def test_no_nan_at_sigma_zero_and_constant_preserved(policy):
    c = torch.full((1, 1, 24, 30, 36), 0.5)
    for lv in build_pyramid(c, (1.0, 1.0, 2.0), [1, 2, 4], policy, sigma=0.0):
        assert torch.isfinite(lv.tensor).all() and float((lv.tensor - 0.5).abs().max()) < 1e-6
    for lv in build_pyramid(c, (1.0, 1.0, 2.0), [2, 4], policy, sigma=[0.0, 0.0]):
        assert torch.isfinite(lv.tensor).all()
    if PRESETS[policy].kernel == 'bessel':                                         # replicate padding: no border darkening
        for lv in build_pyramid(c, (1.0, 1.0, 1.0), [2], policy, sigma=1.5):
            assert float((lv.tensor - 0.5).abs().max()) < 1e-5


@pytest.mark.parametrize("policy", ALL)
def test_coordinates_ramp_through_f_off(policy):
    """Whole-axis check at sigma 0: a ramp of native indices comes back as j f + off (working-voxel error < 1e-4)."""
    for shape, sp in [((33, 61, 61), (1.0, 1.0, 1.0)), ((17, 100, 90), (0.75, 0.75, 3.5)), ((25, 40, 31), (1.0, 1.0, 1.0))]:
        for L in (1, 2, 3, 4):
            for ax_t in range(3):
                lv = build_pyramid(_ramp(shape, ax_t), sp, [L], policy, sigma=0.0, floor=None)[0]
                a = 2 - ax_t
                m = lv.shape[ax_t]
                sel = [0, 0] + [slice(None) if o == ax_t else lv.shape[o] // 2 for o in range(3)]
                got = lv.tensor[tuple(sel)].double()
                expect = torch.arange(m, dtype=torch.float64) * lv.f[a] + lv.off[a]
                assert float((got - expect).abs().max()) / lv.f[a] < 1e-4, (policy, shape, L, ax_t)


@pytest.mark.parametrize("policy", ALL)
def test_deterministic(policy):
    x = _rand((20, 33, 41))
    a = build_pyramid(x, (1.0, 1.0, 2.0), [4, 2, 1], policy, sigma=0.7)
    b = build_pyramid(x, (1.0, 1.0, 2.0), [4, 2, 1], policy, sigma=0.7)
    assert all(torch.equal(u.tensor, v.tensor) for u, v in zip(a, b))
    assert plan_levels(x.shape[2:], (1, 1, 2), [4, 2, 1], policy) == plan_levels(x.shape[2:], (1, 1, 2), [4, 2, 1], policy)


@pytest.mark.parametrize("policy", ALL)
def test_cpu_mps_parity(policy):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    x = _smooth((20, 33, 41))
    cpu = build_pyramid(x, (1.0, 1.0, 2.0), [4, 2], policy, sigma=0.8)
    mps = build_pyramid(x.to('mps'), (1.0, 1.0, 2.0), [4, 2], policy, sigma=0.8)
    for a, b in zip(cpu, mps):
        assert float((a.tensor - b.tensor.cpu()).abs().max()) < 1e-4


@pytest.mark.parametrize("policy", ALL)
def test_mask_resampled_consistently_with_the_image(policy):
    x = _rand((24, 40, 36)); mk = (x > 0.5).float()
    for lv, ref in zip(build_pyramid(x, (1.0, 1.0, 2.0), [4, 2, 1], policy, mask=mk, sigma=0.5),
                       build_pyramid(mk, (1.0, 1.0, 2.0), [4, 2, 1], policy, sigma=0.5)):
        assert torch.equal(lv.mask, ref.tensor)
        assert float(lv.mask.min()) >= -1e-6 and float(lv.mask.max()) <= 1 + 1e-6          # linear, never nearest or extrapolating
    with pytest.raises(PyramidLevelError):
        build_pyramid(x, (1.0, 1.0, 2.0), [2], policy, mask=mk[:, :, :-1])


def test_moving_image_on_its_own_grid_shares_the_millimetre_target():
    fixed = _rand((11, 96, 96)); moving = _rand((40, 40, 40), seed=3)
    fsp, msp = (0.75, 0.75, 3.5), (1.0, 1.0, 1.0)
    for L in (2, 4, 8):
        lf = build_pyramid(fixed, fsp, [L], 'mm_iso')[0]
        lm = build_pyramid(moving, msp, [L], 'mm_iso', reference_spacing=fsp, valid_margin=0)[0]
        assert lm.valid is None
        assert abs(lm.spacing[0] - lf.spacing[0]) < 0.1 * lf.spacing[0]               # the same working spacing in mm (rounded to a whole voxel count)
        assert lm.native_shape == (40, 40, 40) and lf.native_shape == (11, 96, 96)    # each on its own grid
    # a moving image finer than the target is never resampled below its own spacing
    lm = build_pyramid(moving, msp, [1], 'mm_iso', reference_spacing=(0.3, 0.3, 0.3))[0]
    assert lm.shape == (40, 40, 40)


def test_valid_margin_only_for_an_anisotropic_image_and_only_when_asked():
    fixed = _rand((11, 64, 64)); sp = (0.75, 0.75, 3.5)
    assert build_pyramid(fixed, sp, [2], 'mm_iso')[0].valid is None
    lv = build_pyramid(fixed, sp, [1, 2, 4], 'mm_iso_margin')
    for l in lv:
        v = l.valid
        c = np.arange(l.shape[0]) * l.f[2] + l.off[2]
        col = v[0, 0, :, 5, 5].numpy()
        assert (col == ((c >= 1 - 1e-6) & (c <= 11 - 2 + 1e-6))).all() and not col[0] and not col[-1] and col[1:-1].any()
    assert build_pyramid(_rand((32, 32, 32)), (1.0,) * 3, [2], 'mm_iso_margin')[0].valid is None


# ------------------------------------------------------------------------------------------------ geometry helpers
def _rot_dir():
    a = 0.3
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


@pytest.mark.parametrize("policy, L", [('legacy_avgpool', 2), ('legacy_avgpool', 4), ('mm_iso', 3), ('legacy_interp', 4)])
def test_warp_to_moving_norm_matches_the_solvers_formulas(policy, L):
    sp = (0.9, 1.1, 2.0); org = (10.0, -5.0, 3.0); dr = _rot_dir()
    x = _rand((24, 40, 36))
    lv = build_pyramid(x, sp, [L], policy, origin=org, direction=dr, sigma=0.0)[0]
    y = torch.rand(50, 3) * 30
    got = warp_to_moving_norm(y, lv)
    o, s_, d = (torch.tensor(v, dtype=torch.float32) for v in (org, sp, dr))
    y_vox = (y - o) @ torch.inverse(d).t() / s_                                    # robust_affine.warp_to_moving_norm (e['o_mov'], f_mov)
    shape_xyz = torch.tensor(lv.shape[::-1], dtype=torch.float32)
    ref = 2.0 * (((y_vox - torch.tensor(lv.off, dtype=torch.float32)) / torch.tensor(lv.f, dtype=torch.float32)) / (shape_xyz - 1.0)) - 1.0
    assert torch.allclose(got, ref, atol=1e-5)
    if policy == 'legacy_avgpool':                                                  # motion_batched's non-seam path (orig_lvl, sp * level)
        orig_lvl = o + d @ (((L - 1) / 2.0) * s_)
        y_lvl = (y - orig_lvl) @ torch.inverse(d).t() / (s_ * L)
        assert torch.allclose(got, 2.0 * (y_lvl / (shape_xyz - 1.0)) - 1.0, atol=1e-5)
    # sampling the level tensor at the mapped coordinates of the level's own voxels recovers it (the helper is consistent with grid_sample)
    pts = torch.as_tensor(lv.physical_coordinates(), dtype=torch.float32)
    g = warp_to_moving_norm(pts, lv).reshape(1, *lv.shape, 3)[:, :, :, :, :]
    back = F.grid_sample(lv.tensor, g.reshape(1, *lv.shape, 3), mode='bilinear', align_corners=True)
    assert float((back - lv.tensor).abs().max()) < 1e-4


def test_physical_coordinates_helper():
    sp = (0.9, 1.1, 2.0); org = (10.0, -5.0, 3.0); dr = _rot_dir()
    lv = build_pyramid(_rand((24, 40, 36)), sp, [4], 'legacy_avgpool', origin=org, direction=dr)[0]
    pc = lv.physical_coordinates()
    assert pc.shape == (*lv.shape, 3)
    j = np.array([2, 3, 1])                                                         # x, y, z working index
    full = j * 4.0 + 1.5
    expect = np.array(org) + dr @ (full * np.array(sp))
    assert np.allclose(pc[1, 3, 2], expect)                                          # tensor index (z, y, x) = (1, 3, 2)


def test_level_spacing_is_the_former_duplicates():
    """``syngs._level_spacing`` and ``syn_jax.level_spacing_itk`` were deleted (Pass 3b); the shared function is the old formula."""
    def old(spacing_xyz, orig_shape_zyx, curr_shape_zyx):
        return [sp * (float(o - 1) / float(c - 1)) if c > 1 else sp
                for sp, o, c in zip(spacing_xyz, reversed(tuple(orig_shape_zyx)), reversed(tuple(curr_shape_zyx)))]
    for sp, o, c in [((1.0, 2.0, 3.0), (30, 40, 50), (15, 20, 25)), ((0.5, 0.5, 0.5), (11, 64, 64), (11, 16, 16)), ((1.0, 1.0, 1.0), (1, 8, 8), (1, 4, 4))]:
        assert level_spacing(sp, o, c) == old(sp, o, c)
    import importlib
    assert not hasattr(importlib.import_module('syntx.syngs'), '_level_spacing')


def test_planner_is_pure_python_and_named_errors():
    plans = plan_levels((64, 64, 64), (1.0, 1.0, 1.0), [4, 1], 'mm_iso')
    assert all(isinstance(v, (int, float)) for p in plans for v in p.f_xyz + p.off_xyz + p.spacing_xyz)
    with pytest.raises(PyramidLevelError):
        plan_levels((64, 64, 64), (1.0,) * 3, [0.5])
    with pytest.raises(PyramidLevelError):
        plan_levels((64, 64, 64), (1.0,) * 3, [2], 'nope')
    with pytest.raises(PyramidLevelError):
        plan_levels((64, 64, 64), (1.0,) * 3, [2], 'mm_iso', align_corners=True)
    with pytest.raises(PyramidLevelError):
        plan_levels((64, 64, 64), (1.0,) * 3, [2, 1], 'mm_iso', sigma=[1.0])
    with pytest.raises(PyramidLevelError):
        plan_levels((64, 64, 64), (1.0,) * 3, [2], 'mm_iso', floor=1)
    with pytest.raises(PyramidLevelError):
        build_pyramid(torch.zeros(1, 1, 8, 8, 8, dtype=torch.int64), (1.0,) * 3, [2])


def test_max_voxels_budget_raises_target_like_the_old_budget():
    x = _rand((40, 60, 60))
    lv = build_pyramid(x, (1.0,) * 3, [1], 'mm_iso', max_voxels=20000)[0]
    assert float(np.prod(lv.shape)) <= 20000 * 1.2
    old, f, off = _old_mm(x, (1.0,) * 3, 1) if False else __import__('syntx.robust_affine', fromlist=['_mm_pyramid_level'])._mm_pyramid_level(x, (1.0,) * 3, 1.0, 0.0, 20000)
    assert tuple(old.shape[2:]) == lv.shape


def test_cell_aligned_option_matches_old_centre_to_centre_false():
    from syntx.robust_affine import _mm_pyramid_level
    x = _rand((11, 40, 40)); sp = (0.75, 0.75, 3.5)
    old, f, off = _mm_pyramid_level(x, sp, 1.5, 0.0, 2 ** 24, centre_to_centre=False)
    lv = build_pyramid(x, sp, [2], 'mm_iso', centre_to_centre=False, floor=HARD_MIN_VOXELS)[0]
    assert tuple(old.shape[2:]) == lv.shape and np.allclose(f, lv.f) and np.allclose(off, lv.off)


# ------------------------------------------------------------------------------------------------ JAX
@pytest.mark.parametrize("policy", ALL)
def test_jax_matches_torch_for_every_preset(policy):
    pytest.importorskip("jax")
    x = _smooth((20, 33, 41))
    for sp in [(1.0, 1.0, 1.0), (0.75, 0.75, 3.0)]:
        for sig in (None, 0.0, 0.9):
            t = build_pyramid(x, sp, [4, 2, 1], policy, sigma=sig)
            j = pm.build_pyramid_jax(x.numpy(), sp, [4, 2, 1], policy, sigma=sig)
            for a, b in zip(t, j):
                assert a.shape == b.shape and a.f == b.f and a.off == b.off
                assert float(np.abs(a.tensor.numpy() - np.asarray(b.tensor)).max()) < 1e-5


def test_jax_absent_raises_named_import_error(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == 'jax' or name.startswith('jax.'):
            raise ImportError("no jax")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', fake)
    with pytest.raises(ImportError, match="jax"):
        pm.build_pyramid_jax(np.zeros((1, 1, 8, 8, 8), np.float32), (1.0,) * 3, [2])


# ------------------------------------------------------------------------------------------------ ANTs wrapper
def _ants_img(shape, spacing, seed=0):
    import ants
    rng = np.random.default_rng(seed)
    a = rng.random(shape).astype('float32')
    return ants.from_numpy(a, spacing=spacing, origin=(3.0, -2.0, 5.0))


@pytest.mark.parametrize("shape, sp, cap", [((50, 60, 70), (1.0, 1.0, 1.0), 2.0), ((50, 60, 70), (0.5, 0.7, 1.1), 2.0),
                                           ((33, 33, 33), (1.0, 1.0, 1.0), 2.5), ((40, 40, 12), (1.0, 1.0, 3.0), 2.0)])
def test_cap_wrapper_equals_ants_linear_resample_when_the_cap_applied(shape, sp, cap):
    import ants
    from syntx.imaging_utils import DEFAULT_MIN_VOXELS_PER_AXIS
    assert DEFAULT_MIN_VOXELS_PER_AXIS == MIN_VOXELS_PER_AXIS
    img = _ants_img(shape, sp)
    spc = sp
    safe = min(cap, min(n * s for n, s in zip(shape, spc)) / MIN_VOXELS_PER_AXIS)
    assert safe > min(spc)                                                          # the path the old code handled
    ref = ants.resample_image(img, [safe] * 3, interp_type=0)
    got = cap_ants_image(img, cap)
    assert got.shape == ref.shape and np.allclose(got.spacing, ref.spacing) and np.allclose(got.origin, ref.origin)
    assert np.allclose(got.direction, ref.direction)
    a, b = got.numpy(), ref.numpy()
    inner = tuple(slice(0, s - 1) for s in a.shape)                                  # last sample may lie past n-1 (ANTs: 0, here: clamped)
    assert float(np.abs(a[inner] - b[inner]).max()) < 1e-4


def test_cap_wrapper_thin_slab_is_no_longer_defeated():
    img = _ants_img((60, 60, 4), (1.0, 1.0, 3.0))                                    # shortest extent 12 mm <= 16 finest voxels: old code returned img
    got = cap_ants_image(img, 2.0)
    assert got.shape[0] == 30 and got.shape[1] == 30 and got.shape[2] >= 4           # in-plane capped; the thin axis never collapses
    assert np.allclose(got.spacing[:2], 2.0)
    from syntx.imaging_utils import cap_resolution_for_registration
    via_public = cap_resolution_for_registration(img, 2.0)
    assert via_public.shape == got.shape


def test_cap_wrapper_already_coarse_returns_the_image():
    img = _ants_img((20, 20, 20), (2.0, 2.0, 2.0))
    assert cap_ants_image(img, 2.0) is img
