"""Unit tests for syntx.joint_label_fusion."""

import ants
import numpy as np
import pytest
import syntx


def test_joint_label_fusion_basic():
    shape = (32, 32, 32)
    target_np = np.zeros(shape, dtype=np.float32)
    target_np[10:22, 10:22, 10:22] = 80.0
    target = ants.from_numpy(target_np)

    atlases = []
    labels = []
    for i in range(4):
        # Slightly noisy atlas intensity
        a_np = target_np + np.random.normal(0, 4, shape).astype(np.float32)
        atlases.append(ants.from_numpy(a_np))

        l_np = np.zeros(shape, dtype=np.uint32)
        l_np[10:22, 10:22, 10:22] = 1 # Ground truth cube: 12x12x12 = 1728 voxels
        # Atlas 0 has a spurious noisy blob outside
        if i == 0:
            l_np[0:4, 0:4, 0:4] = 1
        labels.append(ants.from_numpy(l_np))

    # 1. Test mode="patch"
    res_patch = syntx.joint_label_fusion(target, atlases, labels, mode="patch", rad=2)
    assert isinstance(res_patch["segmentation"], ants.ANTsImage)
    assert 1 in res_patch["probability_images"]
    seg_patch = res_patch["segmentation"].numpy()
    assert seg_patch[0:4, 0:4, 0:4].sum() == 0, "Spurious noise was not suppressed"
    assert np.isclose(seg_patch[10:22, 10:22, 10:22].sum(), 1728, atol=10)

    # 2. Test mode="joint"
    res_joint = syntx.joint_label_fusion(target, atlases, labels, mode="joint", rad=2)
    assert isinstance(res_joint["segmentation"], ants.ANTsImage)
    seg_joint = res_joint["segmentation"].numpy()
    assert seg_joint[0:4, 0:4, 0:4].sum() == 0, "Spurious noise was not suppressed in joint mode"
    assert np.isclose(seg_joint[10:22, 10:22, 10:22].sum(), 1728, atol=10)
    assert res_joint["timing_s"] > 0


def test_joint_label_fusion_multilabel():
    shape = (30, 30, 30)
    target = ants.from_numpy(np.random.normal(50, 10, shape).astype(np.float32))

    atlases = [target for _ in range(3)]
    labels = []
    for _ in range(3):
        l_np = np.zeros(shape, dtype=np.uint32)
        l_np[5:15, 5:15, 5:15] = 1
        l_np[15:25, 15:25, 15:25] = 2
        labels.append(ants.from_numpy(l_np))

    res = syntx.joint_label_fusion(target, atlases, labels, mode="joint", rad=1)
    seg_np = res["segmentation"].numpy()
    assert set(np.unique(seg_np)) == {0, 1, 2}
    assert 1 in res["probability_images"]
    assert 2 in res["probability_images"]


def test_joint_label_fusion_empty_raises():
    target = ants.from_numpy(np.zeros((10, 10, 10), dtype=np.float32))
    with pytest.raises(ValueError, match="equal non-zero length"):
        syntx.joint_label_fusion(target, [], [])


def test_joint_label_fusion_2d_beats_majority_vote():
    """Fast, targeted 2D test on 24x24 image proving JLF outperforms majority voting.

    Scenario:
    - Target image has a true 8x8 foreground square [8:16, 8:16] with bright intensity (~90)
      surrounded by dark background (~20).
    - Atlas 1 is accurate (correct segmentation and matching intensities).
    - Atlas 2 and Atlas 3 are misregistered: both carry a correlated false protrusion
      into the background [8:16, 16:20] with bright intensity (~90).
    - In the false protrusion, 2 out of 3 atlases vote foreground (1).
    - Majority voting accepts the false protrusion (Dice = 0.8000, 32 false positive voxels).
    - JLF (patch and joint) evaluates local patch intensity similarity, detecting that the
      target is background (~20) where Atlas 2/3 claim foreground, thereby downweighting
      the misregistered atlases and achieving Dice >= 0.99 (Dice = 1.0000 in joint mode).
    - Runs in < 15 ms.
    """
    np.random.seed(42)
    shape = (24, 24)
    gt_seg = np.zeros(shape, dtype=np.uint32)
    gt_seg[8:16, 8:16] = 1

    bg = np.random.normal(20, 3, shape).astype(np.float32)
    target_np = bg.copy()
    target_np[8:16, 8:16] = np.random.normal(90, 5, (8, 8)).astype(np.float32)
    target = ants.from_numpy(target_np)

    # Atlas 1: accurate
    a1_np = target_np + np.random.normal(0, 1, shape).astype(np.float32)
    l1_np = gt_seg.copy()

    # Atlas 2 & 3: correlated false protrusion at [8:16, 16:20]
    a2_np = bg + np.random.normal(0, 1, shape).astype(np.float32)
    a2_np[8:16, 8:16] = np.random.normal(90, 5, (8, 8)).astype(np.float32)
    a2_np[8:16, 16:20] = np.random.normal(90, 5, (8, 4)).astype(np.float32)
    l2_np = gt_seg.copy()
    l2_np[8:16, 16:20] = 1

    a3_np = bg + np.random.normal(0, 1, shape).astype(np.float32)
    a3_np[8:16, 8:16] = np.random.normal(90, 5, (8, 8)).astype(np.float32)
    a3_np[8:16, 16:20] = np.random.normal(90, 5, (8, 4)).astype(np.float32)
    l3_np = gt_seg.copy()
    l3_np[8:16, 16:20] = 1

    atlases = [ants.from_numpy(a1_np), ants.from_numpy(a2_np), ants.from_numpy(a3_np)]
    labels = [ants.from_numpy(l1_np), ants.from_numpy(l2_np), ants.from_numpy(l3_np)]

    def dice(pred, gt):
        inter = np.sum((pred == 1) & (gt == 1))
        return float(2.0 * inter / (np.sum(pred == 1) + np.sum(gt == 1)))

    # 1. Majority voting
    maj_vote = (np.mean([l1_np, l2_np, l3_np], axis=0) >= 0.5).astype(np.uint32)
    dice_maj = dice(maj_vote, gt_seg)
    fp_maj = int(np.sum(maj_vote[8:16, 16:20]))
    assert dice_maj == pytest.approx(0.8000, abs=0.01)
    assert fp_maj == 32, "Majority vote must include all 32 false positive voxels"

    # 2. JLF Fast Patch Mode
    res_patch = syntx.joint_label_fusion(target, atlases, labels, mode="patch", rad=1, beta=2.0)
    seg_patch = res_patch["segmentation"].numpy()
    dice_patch = dice(seg_patch, gt_seg)
    fp_patch = int(np.sum(seg_patch[8:16, 16:20]))
    assert dice_patch > dice_maj, f"JLF patch Dice ({dice_patch:.4f}) should exceed majority vote ({dice_maj:.4f})"
    assert fp_patch < fp_maj

    # 3. JLF Wang Joint Mode
    res_joint = syntx.joint_label_fusion(target, atlases, labels, mode="joint", rad=1, beta=4.0, rho=0.01)
    seg_joint = res_joint["segmentation"].numpy()
    dice_joint = dice(seg_joint, gt_seg)
    fp_joint = int(np.sum(seg_joint[8:16, 16:20]))
    assert dice_joint > dice_maj, f"JLF joint Dice ({dice_joint:.4f}) should exceed majority vote ({dice_maj:.4f})"
    assert dice_joint == pytest.approx(1.0, abs=0.01), "JLF joint mode should achieve near-perfect Dice"
    assert fp_joint == 0, "JLF joint mode should completely eliminate correlated false positive protrusion"
    # the first call on an accelerator pays one-off kernel setup (110-150 ms vs 3-6 ms afterwards)
    res_joint = syntx.joint_label_fusion(target, atlases, labels, mode="joint", rad=1, beta=4.0, rho=0.01)
    assert res_joint["timing_s"] < 0.1, "24x24 test must execute in under 100ms"


def _shifted_phantom(shape, shift, dim):
    """Textured blob target plus atlases that are the target rolled by +-shift along each axis."""
    rng = np.random.default_rng(0)
    tex = ants.smooth_image(ants.from_numpy(rng.normal(0, 1, shape).astype(np.float32)), 1.5).numpy()
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    grids = np.mgrid[tuple(slice(0, n) for n in shape)]
    blob = sum((g - n // 2) ** 2 for g, n in zip(grids, shape)) < (min(shape) // 4) ** 2
    t_np = (0.5 * tex + 0.5 * blob).astype(np.float32)
    kw = dict(spacing=(1.0, 1.5, 2.0)[:dim])
    target = ants.from_numpy(t_np, **kw)
    atlases, labels = [], []
    for ax in range(dim):
        for sgn in (1, -1):
            atlases.append(ants.from_numpy(np.roll(t_np, sgn * shift, axis=ax), **kw))
            labels.append(ants.from_numpy(np.roll(blob, sgn * shift, axis=ax).astype(np.uint32), **kw))
    return target, atlases, labels, blob.astype(np.uint32)


def _dice1(pred, gt):
    return 2.0 * np.sum((pred == 1) & (gt == 1)) / (np.sum(pred == 1) + np.sum(gt == 1))


@pytest.mark.parametrize("mode", ["patch", "joint"])
def test_r_search_zero_is_default_path(mode):
    target, atlases, labels, _ = _shifted_phantom((24, 24, 24), 2, 3)
    a = syntx.joint_label_fusion(target, atlases, labels, mode=mode, device="cpu")
    b = syntx.joint_label_fusion(target, atlases, labels, mode=mode, r_search=0, device="cpu")
    assert np.array_equal(a["segmentation"].numpy(), b["segmentation"].numpy())
    for c in a["probability_images"]:
        assert np.array_equal(a["probability_images"][c].numpy(), b["probability_images"][c].numpy())


@pytest.mark.parametrize("dim,shape", [(2, (40, 40)), (3, (28, 28, 28))])
@pytest.mark.parametrize("mode", ["patch", "joint"])
def test_r_search_recovers_shifted_atlases(dim, shape, mode):
    target, atlases, labels, gt = _shifted_phantom(shape, 2, dim)
    d0 = _dice1(syntx.joint_label_fusion(target, atlases, labels, mode=mode, r_search=0,
                                         device="cpu")["segmentation"].numpy(), gt)
    res = syntx.joint_label_fusion(target, atlases, labels, mode=mode, r_search=2, device="cpu")
    d2 = _dice1(res["segmentation"].numpy(), gt)
    assert d2 >= 0.99 and d2 > d0
    assert res["r_search"] == 2
    # geometry is the target's, anisotropic spacing included
    assert ants.image_physical_space_consistency(res["segmentation"], target)


def test_r_search_deterministic_and_validated():
    target, atlases, labels, _ = _shifted_phantom((24, 24, 24), 1, 3)
    r1 = syntx.joint_label_fusion(target, atlases, labels, r_search=1, device="cpu")
    r2 = syntx.joint_label_fusion(target, atlases, labels, r_search=1, device="cpu")
    assert np.array_equal(r1["segmentation"].numpy(), r2["segmentation"].numpy())
    with pytest.raises(ValueError, match="r_search"):
        syntx.joint_label_fusion(target, atlases, labels, r_search=-1)


def _noisy_multilabel(shape=(48, 48), k=5, seed=3):
    """Target with 3 classes and k atlases whose labels carry independent boundary errors."""
    rng = np.random.default_rng(seed)
    base = np.zeros(shape, dtype=np.uint32)
    base[8:24, 8:40] = 1
    base[28:42, 10:38] = 2
    t_np = (20 + 40 * (base == 1) + 80 * (base == 2) + rng.normal(0, 3, shape)).astype(np.float32)
    target = ants.from_numpy(t_np)
    atlases, labels = [], []
    for _ in range(k):
        dy, dx = rng.integers(-2, 3, 2)
        lab = np.roll(base, (int(dy), int(dx)), axis=(0, 1))
        img = (20 + 40 * (lab == 1) + 80 * (lab == 2) + rng.normal(0, 3, shape)).astype(np.float32)
        atlases.append(ants.from_numpy(img))
        labels.append(ants.from_numpy(lab))
    return target, atlases, labels, base


@pytest.mark.parametrize("mode", ["patch", "joint"])
def test_consensus_skip_equals_dense(mode):
    target, atlases, labels, _ = _noisy_multilabel()
    kw = dict(mode=mode, rad=2, beta=2.0, device="cpu", solver="lu")
    dense = syntx.joint_label_fusion(target, atlases, labels, skip_consensus=False, **kw)
    fast = syntx.joint_label_fusion(target, atlases, labels, skip_consensus=True, **kw)
    assert fast["disagreement_fraction"] < dense["disagreement_fraction"] == 1.0
    assert np.array_equal(dense["segmentation"].numpy(), fast["segmentation"].numpy())
    for c in dense["probability_images"]:
        assert np.allclose(dense["probability_images"][c].numpy(), fast["probability_images"][c].numpy(), atol=1e-5)


@pytest.mark.parametrize("mode", ["joint"])
def test_cholesky_matches_lu(mode):
    target, atlases, labels, _ = _noisy_multilabel()
    a = syntx.joint_label_fusion(target, atlases, labels, mode=mode, device="cpu", solver="lu")
    b = syntx.joint_label_fusion(target, atlases, labels, mode=mode, device="cpu", solver="cholesky")
    assert b["cholesky_failed"] == 0
    assert (a["segmentation"].numpy() != b["segmentation"].numpy()).mean() < 1e-3
    for c in a["probability_images"]:
        assert np.allclose(a["probability_images"][c].numpy(), b["probability_images"][c].numpy(), atol=1e-3)


def test_probabilities_sum_to_one_and_include_background():
    target, atlases, labels, _ = _noisy_multilabel()
    res = syntx.joint_label_fusion(target, atlases, labels, mode="joint", device="cpu")
    total = sum(p.numpy() for p in res["probability_images"].values())
    assert 0 in res["probability_images"]
    assert np.allclose(total[12:36, 12:36], 1.0, atol=1e-4)


def test_wang_prefers_the_accurate_atlas():
    shape = (40, 40)
    rng = np.random.default_rng(1)
    truth = np.zeros(shape, dtype=np.uint32)
    truth[10:30, 10:30] = 1
    t_np = (30 + 50 * truth + rng.normal(0, 2, shape)).astype(np.float32)
    good_lab = truth.copy()
    bad_lab = np.roll(truth, 3, axis=1)
    bad_img = (30 + 50 * bad_lab + rng.normal(0, 2, shape)).astype(np.float32)
    bad2_img = (30 + 50 * bad_lab + rng.normal(0, 2, shape)).astype(np.float32)
    target = ants.from_numpy(t_np)
    atlases = [ants.from_numpy(t_np + rng.normal(0, 1, shape).astype(np.float32)),
               ants.from_numpy(bad_img), ants.from_numpy(bad2_img)]
    labels = [ants.from_numpy(good_lab), ants.from_numpy(bad_lab), ants.from_numpy(bad_lab)]
    seg = syntx.joint_label_fusion(target, atlases, labels, mode="joint", beta=2.0, rho=0.01,
                                   device="cpu")["segmentation"].numpy()
    assert _dice1(seg, truth) > 0.97  # majority vote would adopt the shifted label (2 of 3)
    assert _dice1(np.roll(truth, 3, axis=1), truth) < 0.9


def test_inputs_not_modified_and_outputs_masked_geometry():
    target, atlases, labels, _ = _noisy_multilabel()
    before = [l.numpy().copy() for l in labels] + [a.numpy().copy() for a in atlases] + [target.numpy().copy()]
    res = syntx.joint_label_fusion(target, atlases, labels, device="cpu")
    after = [l.numpy() for l in labels] + [a.numpy() for a in atlases] + [target.numpy()]
    assert all(np.array_equal(b, a) for b, a in zip(before, after))
    assert ants.image_physical_space_consistency(res["segmentation"], target)


def test_device_is_honoured_and_timings_reported():
    target, atlases, labels, _ = _noisy_multilabel()
    res = syntx.joint_label_fusion(target, atlases, labels, device="cpu", r_search=1)
    assert res["device"] == "cpu"
    st = res["stage_timings_s"]
    assert set(st) == {"search", "weights", "vote"} and st["search"] > 0 and st["weights"] > 0
    assert 0.0 <= res["disagreement_fraction"] <= 1.0


def test_return_probabilities_false_matches_segmentation():
    target, atlases, labels, _ = _noisy_multilabel()
    a = syntx.joint_label_fusion(target, atlases, labels, device="cpu")
    b = syntx.joint_label_fusion(target, atlases, labels, device="cpu", return_probabilities=False)
    assert b["probability_images"] == {}
    assert np.array_equal(a["segmentation"].numpy(), b["segmentation"].numpy())
    assert np.array_equal(a["consensus_envelope"].numpy(), b["consensus_envelope"].numpy())


def test_invalid_mode_and_solver_raise():
    target, atlases, labels, _ = _noisy_multilabel(k=2)
    with pytest.raises(ValueError, match="mode"):
        syntx.joint_label_fusion(target, atlases, labels, mode="nope")
    with pytest.raises(ValueError, match="solver"):
        syntx.joint_label_fusion(target, atlases, labels, solver="qr")
    with pytest.raises(ValueError, match="mode"):
        syntx.joint_label_fusion(target, atlases, labels, mode="wang")  # removed: 'joint' is Wang's model


def test_gather_weights_equal_box_weights_at_zero_shift():
    """With the zero offset only, whole-patch gather must reproduce the box-sum weights (interior)."""
    import torch
    import importlib; mod = importlib.import_module("syntx.joint_label_fusion")
    rng = np.random.default_rng(5)
    sub, rad, K = (26, 26), 2, 4
    t = torch.from_numpy(rng.random(sub).astype(np.float32))[None, None]
    atl = [torch.from_numpy((rng.random(sub) * 0.3 + t[0, 0].numpy()).astype(np.float32))[None, None] for _ in range(K)]
    pad = lambda x: torch.nn.functional.pad(x, (rad,) * 4, mode="replicate")  # noqa: E731
    didx = torch.nonzero(torch.from_numpy(np.pad(np.ones((26 - 2 * rad - 2,) * 2, bool), rad + 1)).reshape(-1)).squeeze(1)
    offs = [(0, 0)]
    off_ids = [torch.zeros((1, 1) + sub, dtype=torch.long) for _ in range(K)]
    stats = {"cholesky_failed": 0}
    wg = mod._gather_weights(pad(t), [pad(a) for a in atl], off_ids, offs, didx, sub, rad, rad, "joint",
                             4.0, 0.01, "lu", True, stats, torch.device("cpu"), patch_norm=False)
    P = float((2 * rad + 1) ** 2)
    absd = [(a - t).abs() for a in atl]
    G = torch.empty((didx.numel(), K, K))
    for i in range(K):
        for j in range(K):
            G[:, i, j] = (mod._box_sum(absd[i] * absd[j], rad) / P)[0, 0].reshape(-1)[didx]
    wb = mod._wang_weights(G, 4.0, 0.01, "lu", True, stats, p_eff=P)
    assert torch.allclose(wg, wb, atol=1e-4)

    # standardised patches: G_ij = r_ij - r_it - r_jt + r_tt from box-sum moments (independent code path)
    wn = mod._gather_weights(pad(t), [pad(a) for a in atl], off_ids, offs, didx, sub, rad, rad, "joint",
                             4.0, 0.01, "lu", True, stats, torch.device("cpu"), patch_norm=True)
    bs = lambda x: mod._box_sum(x, rad)  # noqa: E731
    C = lambda x, y: bs(x * y) - bs(x) * bs(y) / P  # noqa: E731
    s = lambda x: torch.sqrt(torch.clamp_min(C(x, x), mod._VAR_FLOOR))  # noqa: E731
    r = lambda x, y: C(x, y) / (s(x) * s(y))  # noqa: E731
    Gn = torch.empty((didx.numel(), K, K))
    for i in range(K):
        for j in range(K):
            g = r(atl[i], atl[j]) - r(atl[i], t) - r(atl[j], t) + C(t, t) / s(t) ** 2
            Gn[:, i, j] = g[0, 0].reshape(-1)[didx]
    assert torch.allclose(wn, mod._wang_weights(Gn, 4.0, 0.01, "lu", True, stats), atol=1e-3)


def test_joint_is_not_fooled_by_noise_only_in_one_atlas():
    """A blob carried by one atlas in a featureless noisy region must lose the 1-vs-3 vote.

    Unshrunk raw residuals made the weights arbitrary in featureless regions and ~11% of such
    voxels flipped (143/1280, ANTs 0/1280); the empirical-Bayes shrinkage fixes it at the source.
    """
    shape = (32, 32, 32)
    t = np.zeros(shape, np.float32); t[10:22, 10:22, 10:22] = 80.0
    target = ants.from_numpy(t)
    flipped = 0
    for seed in range(10):
        rng = np.random.default_rng(seed)
        atl, lab = [], []
        for i in range(4):
            atl.append(ants.from_numpy(t + rng.normal(0, 4, shape).astype(np.float32)))
            l = np.zeros(shape, np.uint32); l[10:22, 10:22, 10:22] = 1
            if i == 0:
                l[0:4, 0:4, 0:4] = 1
            lab.append(ants.from_numpy(l))
        seg = syntx.joint_label_fusion(target, atl, lab, mode="joint", rad=2, device="cpu")["segmentation"].numpy()
        flipped += int(seg[0:4, 0:4, 0:4].sum())
    assert flipped == 0


@pytest.mark.parametrize("mode", ["patch", "joint"])
def test_search_skip_consensus_matches_dense_reference_on_shifted_atlases(mode):
    target, atlases, labels, gt = _shifted_phantom((28, 28, 28), 2, 3)
    s = syntx.joint_label_fusion(target, atlases, labels, mode=mode, r_search=2, device="cpu")
    d = syntx.joint_label_fusion(target, atlases, labels, mode=mode, r_search=2, device="cpu",
                                 skip_consensus=False, solver="lu")
    a, b = s["segmentation"].numpy(), d["segmentation"].numpy()
    assert _dice1(a, gt) >= 0.99 and _dice1(b, gt) >= 0.99
    assert np.mean(a != b) < 5e-3  # documented approximation under a search (see Notes)


@pytest.mark.parametrize("mode", ["patch", "joint"])
def test_tied_votes_are_deterministic_and_go_to_the_smaller_label(mode):
    """Two equally good atlases that disagree tie exactly; round-off (solver, path) must not decide."""
    rng = np.random.default_rng(3)
    shape = (16, 16)
    t = rng.random(shape).astype(np.float32)
    target = ants.from_numpy(t)
    atlases = [ants.from_numpy(t.copy()), ants.from_numpy(t.copy())]
    l0 = np.zeros(shape, np.uint32)
    l1 = np.zeros(shape, np.uint32); l1[4:12, 4:12] = 2
    labels = [ants.from_numpy(l0), ants.from_numpy(l1)]
    kw = dict(mode=mode, rad=1, device="cpu")
    a = syntx.joint_label_fusion(target, atlases, labels, **kw)["segmentation"].numpy()
    b = syntx.joint_label_fusion(target, atlases, labels, solver="lu", skip_consensus=False, **kw)["segmentation"].numpy()
    assert np.array_equal(a, b)
    assert a[6:10, 6:10].max() == 0  # tie -> smaller label (background)
