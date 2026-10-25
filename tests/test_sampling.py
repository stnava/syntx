"""syntx.sampling: the metric sample-point menu (registry, normalisation, region, weighted draw, physical space)."""

from __future__ import annotations

import ants
import numpy as np
import pytest
import torch

from syntx.sampling import (
    GradientSampling,
    PointSampler,
    SAMPLING_STRATEGIES,
    draw_indices,
    get_sampling_strategy,
    normalize_robust,
    sample_budget,
)


def _oblique_phantom(shape=(30, 34, 22), spacing=(1.2, 1.0, 2.5)):
    """Ellipsoidal head (smooth edge) with an inner block, on an oblique anisotropic grid."""
    g = np.indices(shape).astype(np.float64)
    c = (np.array(shape) - 1) / 2.0
    r = np.sqrt(sum(((g[i] - c[i]) / (0.38 * shape[i])) ** 2 for i in range(3)))
    arr = 100.0 / (1.0 + np.exp((r - 1.0) / 0.05))
    arr[12:18, 14:20, 8:13] += 60.0
    th = np.radians(20.0)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    return ants.from_numpy(arr.astype(np.float32), origin=(10.0, -5.0, 3.0), spacing=spacing, direction=R)


def test_menu_registry():
    assert {"gradient", "uniform", "gradient_floor"} <= set(SAMPLING_STRATEGIES)
    assert get_sampling_strategy("gradient_floor", floor=0.2).params()["floor"] == 0.2
    s = GradientSampling(z=3.0)
    assert get_sampling_strategy(s) is s
    assert get_sampling_strategy("Gradient", upper=0.9).params() == {"z": 4.0, "upper": 0.9}
    with pytest.raises(ValueError, match="unknown sampling strategy"):
        get_sampling_strategy("stratified")
    with pytest.raises(ValueError):
        get_sampling_strategy(s, z=5.0)


def test_normalize_robust_percentiles_and_header():
    img = _oblique_phantom()
    n = normalize_robust(img)
    a = n.numpy()
    assert a.min() >= 0.0 and a.max() <= 1.0
    assert n.origin == img.origin and n.spacing == img.spacing and np.allclose(n.direction, img.direction)
    pos = img.numpy()[img.numpy() > 0]
    assert abs(float(np.mean(a[img.numpy() > 0] >= 1.0)) - float(np.mean(pos >= np.percentile(pos, 98)))) < 0.01


def test_sample_budget():
    assert sample_budget(0.1, 0.2) == pytest.approx(0.5)
    assert sample_budget(0.3, 0.2) == 1.0
    assert sample_budget(1.0, 0.2) == 1.0
    with pytest.raises(ValueError):
        sample_budget(0.0, 0.2)


def test_draw_is_weighted_without_replacement_and_seeded():
    w = torch.tensor([0.0, 1.0, 1.0, 8.0, 8.0, 0.0] * 500, dtype=torch.float64)
    counts = np.zeros(6)
    for seed in range(40):
        idx = draw_indices(w, w.numel(), 300, torch.Generator().manual_seed(seed))
        assert idx.unique().numel() == idx.numel()
        np.add.at(counts, (idx % 6).numpy(), 1)
    assert counts[0] == 0 and counts[5] == 0                     # zero weight never drawn
    assert counts[3] / counts[1] > 5.0                           # 8:1 weights (without replacement flattens a little)
    a = draw_indices(w, w.numel(), 100, torch.Generator().manual_seed(3))
    b = draw_indices(w, w.numel(), 100, torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_gradient_sampler_concentrates_on_edges_in_physical_space():
    img = _oblique_phantom()
    ps = PointSampler(img, strategy="gradient", sample_fraction=0.05, stage_sampling=0.2)
    pool = ps.pool()
    assert ps.n_pool == pool["phys"].shape[0] >= 2000
    # the pool is physical: its points map back into the grid through the image's own header
    idx = [ants.transform_physical_point_to_index(img, p.tolist()) for p in pool["phys"][:50].cpu().numpy()]
    assert np.allclose(np.array(idx), pool["vox_xyz"][:50].cpu().numpy(), atol=1e-3)
    # the pool is uniform over the candidate region; a draw is weighted by |grad n| at its scale
    g = ants.iMath(ps.normalized, "Grad", ps.sigma_mm).numpy()
    v = np.round(pool["vox_xyz"].cpu().numpy()).astype(int)
    g_pool = g[v[:, 0], v[:, 1], v[:, 2]]
    k = ps.draw(200)
    assert k.numel() == 200 and k.unique().numel() == 200
    # a draw proportional to the weights has E[g] = sum(g^2) / sum(g) over the pool (a 10 % draw without replacement is
    # slightly below it), above the uniform mean
    w = ps.weights().cpu().numpy()
    expected = float((w * w).sum() / w.sum())
    got = float(w[k.cpu().numpy()].mean())
    assert got > float(w.mean()) and abs(got - expected) < 0.15 * expected
    prov = ps.provenance()
    assert prov["strategy"] == "gradient" and prov["pool_is_physical"] and prov["region_voxels"] == int(ps.region.sum())


def test_one_physical_pool_serves_every_scale_and_mask_is_opt_in():
    img = _oblique_phantom()
    m = np.zeros(img.shape, dtype=bool)
    m[: img.shape[0] // 2] = True
    ps = PointSampler(img, sample_mask=m, sample_fraction=0.1)
    pool = ps.pool()
    w_fine, w_coarse = ps.weights(), ps.weights(4.0 * ps.sigma_mm, use_sample_mask=False)
    assert w_fine.shape == w_coarse.shape == (ps.n_pool,)                 # same points at both scales
    assert not torch.allclose(w_fine, w_coarse)                            # the strategy acts at each scale
    v = np.round(pool["vox_xyz"].cpu().numpy()).astype(int)
    inside = m[v[:, 0], v[:, 1], v[:, 2]]
    assert float(w_fine[torch.from_numpy(~inside)].abs().sum()) == 0.0     # mask applied at the fine scale
    assert float(w_coarse[torch.from_numpy(~inside)].sum()) > 0.0           # not at the coarse scale
    assert ps.weights() is w_fine                                          # cached per (scale, mask)


def test_gradient_sigma_is_in_mm():
    """syntx.sampling.gradient_magnitude_mm: the same physical edge on two grids gives the same gradient per mm."""
    from syntx.sampling import gradient_magnitude_mm

    x = np.linspace(-20, 20, 81)
    prof = 1.0 / (1.0 + np.exp(-x / 2.0))
    a = np.tile(prof[:, None, None], (1, 8, 8)).astype(np.float32)
    fine = ants.from_numpy(a, spacing=(0.5, 1.0, 1.0))
    coarse = ants.from_numpy(a[::2].copy(), spacing=(1.0, 1.0, 1.0))
    gf = gradient_magnitude_mm(fine, 1.0)[40, 4, 4]
    gc = gradient_magnitude_mm(coarse, 1.0)[20, 4, 4]
    assert abs(gf - gc) / gc < 0.1
    assert abs(gf - 0.125) < 0.02          # logistic slope at the centre 1 / (4 * 2 mm), slightly smoothed


def test_empty_image_raises():
    img = ants.from_numpy(np.zeros((20, 20, 20), dtype=np.float32))
    with pytest.raises(ValueError, match="positive voxels"):
        PointSampler(img)


def _head_in_rayleigh_noise(sigma=2.0, zero_fraction=0.0, seed=0):
    """Head (ellipsoid, tissue intensity varying smoothly 100-160) in Rayleigh background noise (scale sigma); optional
    exact zeros in the background. Tissue is continuous, as in real images (a constant plateau would be saturated)."""
    rng = np.random.default_rng(seed)
    shape = (48, 52, 40)
    g = np.indices(shape).astype(np.float64)
    c = (np.array(shape) - 1) / 2.0
    r = np.sqrt(sum(((g[i] - c[i]) / (0.3 * shape[i])) ** 2 for i in range(3)))
    head = r < 1.0
    arr = rng.rayleigh(sigma, size=shape)
    arr[head] = 100.0 + 60.0 * (1.0 - r[head])
    if zero_fraction > 0:
        bgi = np.flatnonzero(~head.ravel())
        arr.ravel()[rng.choice(bgi, int(zero_fraction * arr.size), replace=False)] = 0.0
    return ants.from_numpy(arr.astype(np.float32), spacing=(1.5, 1.5, 2.5)), head


def test_noise_floor_region_is_the_head():
    img, head = _head_in_rayleigh_noise()
    ps = PointSampler(img)
    b = ps.provenance()["background"]
    assert b["background"] == "noise"
    assert abs(b["mu"] - 2.0 * np.sqrt(np.log(4.0))) < 0.3          # Rayleigh median = sigma sqrt(2 ln 2)
    # z = 4 puts the floor near 3.8 sigma for Rayleigh noise: a tail of ~7.6e-4 of the background (independent audit)
    assert (ps.region & ~head).sum() < 0.002 * (~head).sum()
    assert (head & ~ps.region).sum() < 0.03 * head.sum()             # head kept (top 2 % saturated core may drop)


def test_exact_zeros_are_excluded_from_the_noise_estimate():
    img, head = _head_in_rayleigh_noise(zero_fraction=0.05)
    b = PointSampler(img).provenance()["background"]
    assert b["background"] == "noise" and b["zero_fraction"] > 0.04
    assert abs(b["mu"] - 2.0 * np.sqrt(np.log(4.0))) < 0.3          # zeros did not pull the floor down


def test_zero_filled_background_falls_back_to_positive_voxels_with_warning():
    img, head = _head_in_rayleigh_noise()
    a = img.numpy()
    a[~head] = 0.0                    # background exactly 0 (no noise to measure)
    img = img.new_image_like(a)
    with pytest.warns(UserWarning, match="zero-filled"):
        ps = PointSampler(img)
    assert ps.provenance()["background"]["background"] == "zero-filled"
    assert np.array_equal(ps.region, (img.numpy() > 0) & (ps.normalized.numpy() < 0.98))


def test_uniform_and_floor_strategies_share_the_region():
    img = _oblique_phantom()
    pools = {}
    for name in ("gradient", "uniform", "gradient_floor"):
        ps = PointSampler(img, strategy=name, sample_fraction=0.1)
        ps.pool()
        w = ps.weights().cpu().numpy()
        pools[name] = (ps, w)
    assert all(np.array_equal(pools["gradient"][0].region, ps.region) for ps, _ in pools.values())
    w_u, w_f = pools["uniform"][1], pools["gradient_floor"][1]
    assert np.allclose(w_u[w_u > 0], 1.0)                       # region indicator
    assert w_f.min() >= 0.1 - 1e-6 and w_f.max() <= 1.1 + 1e-6  # floor + relative gradient


def test_clipped_gaussian_background_is_noise_not_zero_filled():
    """Zero-mean noise clipped at 0 (about half the background exactly 0): the positive half is half-normal, the floor is
    z sigma, and the noise is excluded (no fallback to every positive voxel)."""
    rng = np.random.default_rng(0)
    img, head = _head_in_rayleigh_noise()
    a = img.numpy()
    a[~head] = np.clip(rng.normal(0.0, 2.0, size=int((~head).sum())), 0.0, None)
    img = img.new_image_like(a.astype(np.float32))
    ps = PointSampler(img)
    b = ps.provenance()["background"]
    assert b["background"] == "clipped_noise"
    assert abs(b["sd"] - 2.0) < 0.2
    assert (ps.region & ~head).sum() < 0.001 * (~head).sum()


def test_dense_grid_entries():
    from syntx.sampling import regular_stride, uniform_cube_points

    assert regular_stride(None) == 1 and regular_stride(1.0) == 1 and regular_stride(0.2) == 5 and regular_stride(0.3) == 3
    with pytest.raises(ValueError):
        regular_stride(0.0)
    a = uniform_cube_points(1000, 3, np.random.default_rng(7))
    b = uniform_cube_points(1000, 3, np.random.default_rng(7))
    assert a.dtype == np.float32 and a.shape == (1000, 3) and np.array_equal(a, b)
    assert a.min() >= -1.0 and a.max() <= 1.0



def test_eval_cap_holds_for_every_sample_fraction():
    img, _ = _head_in_rayleigh_noise()
    big = ants.resample_image(img, (0.6, 0.6, 1.0), use_voxels=False, interp_type=0)
    from syntx import sampling as S

    for sf in (0.15, 0.2, 0.5):
        ps = PointSampler(big, sample_fraction=sf, stage_sampling=0.2, max_eval_points=5000)
        ps.pool()
        assert ps.n_pool <= 5000 / 0.2
        assert ps.eval_points(0.2, ps.n_pool) <= 5000 and ps.eval_points(0.3, ps.n_pool) <= 5000
    dense = PointSampler(big, sample_fraction=1.0, max_eval_points=5000)
    dense.pool()
    assert dense.eval_points(1.0, dense.n_pool) == dense.n_pool       # dense sampling is never capped
    assert S.MAX_EVAL_POINTS > 0


@pytest.mark.parametrize("nz", [1, 3])
def test_thin_slabs_do_not_crash(nz):
    """ITK's recursive Gaussian aborts the process below 4 voxels on an axis; the shared gradient handles any grid."""
    rng = np.random.default_rng(0)
    a = rng.rayleigh(2.0, size=(40, 40, nz)).astype(np.float32)
    a[10:30, 10:30, :] = 100.0 + rng.normal(0, 3, size=(20, 20, nz))
    ps = PointSampler(ants.from_numpy(a, spacing=(1.0, 1.0, 3.0)), strategy="gradient_floor")
    ps.pool()
    w = ps.weights()
    assert torch.isfinite(w).all() and ps.n_pool > 0


def test_integer_low_noise_magnitude():
    img, head = _head_in_rayleigh_noise(sigma=0.8)
    a = np.round(img.numpy()).astype(np.float32)
    ps = PointSampler(img.new_image_like(a))
    b = ps.provenance()["background"]
    assert b["background"] == "noise" and b["sd"] > 0
    assert (head & (a <= b["threshold"])).sum() == 0                  # the noise floor keeps every tissue voxel


def test_zero_padding_around_multicoil_noise_is_not_zero_filled():
    """Zero padding (contiguous blocks) is set aside; the noncentral-chi noise inside the field of view sets the floor."""
    rng = np.random.default_rng(1)
    img, head = _head_in_rayleigh_noise()
    a = img.numpy()
    n_coil = 8
    noise = np.sqrt(sum(rng.normal(0, 2.0, size=a.shape) ** 2 for _ in range(2 * n_coil)))
    a[~head] = noise[~head]
    a[:6], a[-6:] = 0.0, 0.0
    ps = PointSampler(img.new_image_like(a.astype(np.float32)))
    b = ps.provenance()["background"]
    assert b["background"] == "noise" and b["padding_zero_fraction"] > 0.2
    inside = np.zeros_like(head)
    inside[6:-6] = True
    assert (ps.region & ~head & inside).sum() < 0.002 * (~head & inside).sum()
    assert (head & ~ps.region).sum() < 0.03 * head.sum()


def test_object_filling_the_field_of_view_has_no_background():
    rng = np.random.default_rng(2)
    from scipy.ndimage import gaussian_filter

    tex = gaussian_filter(rng.normal(size=(40, 40, 40)), 2.0)
    a = (100.0 + 400.0 * (tex - tex.min()) / (tex.max() - tex.min()) + rng.normal(0, 1.0, tex.shape)).astype(np.float32)
    ps = PointSampler(ants.from_numpy(a))
    assert ps.provenance()["background"]["background"] == "no_background"
    assert ps.region.mean() > 0.9


def test_shared_strategy_object_keeps_each_samplers_provenance():
    st = get_sampling_strategy("uniform")
    img, head = _head_in_rayleigh_noise()
    p1 = PointSampler(img, strategy=st)
    a = img.numpy()
    a[~head] = 0.0
    with pytest.warns(UserWarning):
        PointSampler(img.new_image_like(a), strategy=st)
    assert p1.provenance()["background"]["background"] == "noise"


def test_draw_returns_at_most_the_positive_weights():
    w = torch.tensor([0.0, 1.0, 2.0, 3.0])
    idx = draw_indices(w, 4, 10, torch.Generator().manual_seed(0))
    assert idx.tolist() == [1, 2, 3] and idx.dtype == torch.int64
    with pytest.raises(ValueError, match="upload the weights once"):
        draw_indices(w, 4, 2, torch.Generator(device="mps").manual_seed(0) if torch.backends.mps.is_available() else _FakeGen())


class _FakeGen:
    device = torch.device("meta")
