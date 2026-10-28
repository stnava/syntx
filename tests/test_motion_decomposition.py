"""syntx.motion_decomposition: planted term variances are recovered; the pose-domain shares reproduce planted components."""
from __future__ import annotations

import numpy as np

from syntx.motion_decomposition import (design_blocks, image_variance_decomposition, pose_step_decomposition, real_even_sh,
                                        sequential_sums_of_squares)


def test_real_even_sh_shapes_and_b0_rows():
    rng = np.random.default_rng(0)
    g = rng.standard_normal((40, 3)); g[:3] = 0
    assert real_even_sh(g, 2).shape == (40, 5) and real_even_sh(g, 4).shape == (40, 14)
    assert np.all(real_even_sh(g, 4)[:3] == 0)
    Y = real_even_sh(rng.standard_normal((4000, 3)), 4)
    C = Y.T @ Y / 4000
    assert np.allclose(np.diag(C) / np.diag(C).mean(), 1.0, atol=0.15)        # near-orthonormal on a uniform sphere


def test_image_variance_decomposition_recovers_planted_fractions():
    rng = np.random.default_rng(1)
    T, V = 120, 800
    bv = np.r_[[0] * 12, [1000] * 54, [2000] * 54]
    g = rng.standard_normal((T, 3)); g /= np.linalg.norm(g, axis=1, keepdims=True); g[bv == 0] = 0
    levels = (np.round(bv, -2) // 1000).astype(int)
    blocks = design_blocks(levels, "shells", bvecs=g, bvals=bv, lmax=4)
    names = [b[0] for b in blocks]
    assert names == ["level intercepts", "l=2 direction terms per shell", "l=4 direction terms per shell"]
    inter, l2, l4 = (b[1] for b in blocks)
    fd = np.abs(rng.standard_normal(T)); fd[0] = 0
    motion = np.c_[rng.standard_normal((T, 6)), fd]
    # planted: shell contrast (big), l2 (medium), l4 (small), motion-locked (small), noise
    sig = (inter @ (rng.standard_normal((3, V)) * 10.0) + l2 @ (rng.standard_normal((l2.shape[1], V)) * 1.0)
           + l4 @ (rng.standard_normal((l4.shape[1], V)) * 0.4) + fd[:, None] * (rng.standard_normal((1, V)) * 2.0)
           + rng.standard_normal((T, V)) * 1.0)
    series = sig.T.reshape(V, 1, 1, T)
    out = image_variance_decomposition(series, blocks, motion, np.ones((V, 1, 1), bool), fd=fd, n_voxels=V)
    fr = {t["term"]: t["fraction"] for t in out["terms"]}
    assert abs(sum(fr.values()) - 1.0) < 1e-9
    assert fr["level intercepts"] > 0.8 and fr["l=2 direction terms per shell"] > fr["l=4 direction terms per shell"] > 0
    assert 0.0 < fr["motion regressors after the design"] < fr["level intercepts"]
    assert out["residual_rms_fd_corr"] > 0.25                                  # the planted motion-locked term follows FD (noise 1.0 dominates the per-frame RMS)
    tot, ss = sequential_sums_of_squares(np.zeros((T, 5)), blocks[:1])
    assert tot == 0.0 and ss[-1][0] == "residual"


def test_pose_step_decomposition_shares():
    rng = np.random.default_rng(2)
    T = 60
    D = np.eye(3)
    head = np.cumsum(rng.standard_normal((T, 3)) * 0.05, axis=0)
    eddy = rng.standard_normal(T) * 0.5                                        # along phase encode (axis 1)
    bias = np.zeros((T, 6)); bias[:, 2] = np.where(np.arange(T) % 2 == 0, 0.1, -0.1)   # level-locked slice offset
    out = pose_step_decomposition(head, np.zeros((T, 3)), D, applied_pe_shift_mm=eddy, pe_axis=np.array([0, 1.0, 0]), removed_bias_mm_rad=bias)
    ax = out["axes"]
    assert ax["readout"]["applied_acquisition_term"]["step_sd_mm"] == 0.0 and ax["readout"]["removed_design_locked_term"]["step_sd_mm"] == 0.0
    assert ax["phase_encode"]["applied_acquisition_term"]["share"] > 0.8
    assert ax["slice"]["removed_design_locked_term"]["share"] > 0.8
    assert abs(ax["slice"]["removed_design_locked_term"]["step_sd_mm"] - 0.2) < 1e-3
