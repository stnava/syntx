"""Step 2d: the odd/even slice intensity term. On a corrected DWI series: residual after the shell model (intercepts + l=2 +
l=4), the per-volume odd-minus-even slice contrast c(t) of that residual, the variance explained when c(t) enters as a
temporal regressor with a free spatial pattern per voxel, and c(t)'s correlation with FD and the poses.
Usage: python benchmarks/slice_parity_term.py <dwi.nii.gz> <out dir>  (runs the joint engine; ~2 min)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np


def main(path, out):
    import ants
    import syntx
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from syntx.motion_decomposition import design_blocks, sequential_sums_of_squares

    os.makedirs(out, exist_ok=True)
    img = ants.image_read(path)
    bv = np.loadtxt(path.replace(".nii.gz", ".bval")); g = np.loadtxt(path.replace(".nii.gz", ".bvec")).T
    sc = json.load(open(path.replace(".nii.gz", ".json"))); pe = sc.get("PhaseEncodingDirection", "j"); ax = {"i": 0, "j": 1, "k": 2}[pe[0]]
    v = np.asarray(img.direction)[:3, ax] * (-1 if pe.endswith("-") else 1); v /= np.linalg.norm(v)
    design = syntx.make_design("shells", bvals=bv, bvecs=g, pe_axis=v); levels = np.asarray(design.levels)
    r = syntx.motion_correct_design(img, design, engine="joint", verbose=False)
    corr = r["motion_corrected"].numpy(); T = corr.shape[-1]
    fd = np.asarray(r["fd"], float); t = np.asarray(r["translations_mm"], float); rot = np.asarray(r["rotations_deg"], float)
    m = corr.mean(-1); mask = m > 0.3 * np.percentile(m, 99)
    blocks = design_blocks(levels, "shells", bvecs=g, bvals=bv, lmax=4, min_frames_per_column=3)
    X = np.hstack([B for _, B in blocks]); Q, _ = np.linalg.qr(X)
    Y = corr.reshape(-1, T)[mask.reshape(-1)].T.astype(np.float64); Y -= Y.mean(0, keepdims=True)
    R = Y - Q @ (Q.T @ Y)                                                  # (T, V) residual after the shell model
    zidx = np.nonzero(mask)[2]; parity = np.where(zidx % 2 == 0, 1.0, -1.0)
    c = (R * parity[None, :]).mean(axis=1)                                 # per volume: odd-minus-even slice contrast of the residual
    c_rel = c / (np.abs(R).mean() + 1e-9)
    # share of variance: the parity contrast as a temporal regressor with a free spatial pattern, entered after the design
    rng = np.random.default_rng(0); idx = rng.choice(Y.shape[1], size=min(60000, Y.shape[1]), replace=False)
    Ys = Y[:, idx]
    motion = np.c_[t, rot, fd]; motion -= motion.mean(0)
    tot, ss_a = sequential_sums_of_squares(Ys, blocks + [("motion", motion)])
    tot, ss_b = sequential_sums_of_squares(Ys, blocks + [("parity contrast c(t)", (c - c.mean())[:, None]), ("motion", motion)])
    share = {n: v_ / tot for n, v_, k in ss_b}
    corr_fd = float(np.corrcoef(c[1:], fd[1:])[0, 1])
    corr_pose = {lab: float(np.corrcoef(c, col)[0, 1]) for lab, col in zip(("tx", "ty", "tz", "rx", "ry", "rz"), np.c_[t, rot].T)}
    # spatial pattern of the parity term: its per-voxel coefficient, how slice-structured is it (fraction of its energy in the parity sign)
    coef = np.linalg.lstsq(np.c_[X, c[:, None]], Y, rcond=None)[0][-1]
    slice_struct = float(np.abs((coef * parity).mean()) / (np.abs(coef).mean() + 1e-9))
    res = {"parity_contrast_abs_mean_rel": float(np.abs(c_rel).mean()), "share_parity_pct": 100 * share["parity contrast c(t)"],
           "share_motion_after_pct": 100 * share["motion"], "share_residual_pct": 100 * share["residual"],
           "share_motion_without_parity_pct": 100 * dict((n, v_ / tot) for n, v_, k in ss_a)["motion"],
           "c_vs_fd_r": corr_fd, "c_vs_pose_r": corr_pose, "coef_parity_structure": slice_struct,
           "c_lag1_autocorr": float(np.corrcoef(c[:-1], c[1:])[0, 1])}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(c_rel, lw=0.8); axes[0].set_title("odd-minus-even slice residual contrast per volume (rel.)", fontsize=9); axes[0].set_xlabel("volume")
    for i in np.flatnonzero(bv < 50): axes[0].axvline(i, color="k", alpha=0.1)
    axes[1].scatter(fd[1:], c_rel[1:], s=8, c=levels[1:], cmap="viridis"); axes[1].set_xlabel("FD (mm)"); axes[1].set_ylabel("parity contrast"); axes[1].set_title(f"vs FD: r = {corr_fd:.2f}", fontsize=9)
    z = corr.shape[2] // 2; pat = np.zeros(mask.shape); pat[mask] = coef
    axes[2].imshow(np.rot90(pat[:, mask.shape[1] // 2, :]), cmap="coolwarm", vmin=-np.percentile(np.abs(coef), 95), vmax=np.percentile(np.abs(coef), 95), aspect="auto"); axes[2].set_title("spatial pattern of the parity term (coronal)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, "slice_parity_term.png"), dpi=110)
    json.dump(res, open(os.path.join(out, "slice_parity_term.json"), "w"), indent=1)
    print(json.dumps({k: (round(v_, 3) if isinstance(v_, float) else v_) for k, v_ in res.items()}))
    print("DONE", os.path.join(out, "slice_parity_term.png"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
