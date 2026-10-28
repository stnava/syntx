"""Why does the readout-axis jitter differ between engines on BLAST031? Per-volume, per-slice residual of the corrected series
against its shell-model prediction (intercept + l=2 + l=4 per shell), for unified and projected. Intra-volume motion or signal
dropout gives slice-wise stripes concentrated in the volumes with large readout steps; pulsation gives residual at the same
places in every volume. Writes a figure and a JSON with the correlation between a volume's slice-stripe index and its readout step.
Usage: python benchmarks/slice_residual_diagnostic.py <dwi.nii.gz> <out dir>
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
    from syntx.motion_decomposition import design_blocks

    os.makedirs(out, exist_ok=True)
    img = ants.image_read(path)
    bv = np.loadtxt(path.replace(".nii.gz", ".bval")); g = np.loadtxt(path.replace(".nii.gz", ".bvec")).T
    sc = json.load(open(path.replace(".nii.gz", ".json"))); pe = sc.get("PhaseEncodingDirection", "j"); ax = {"i": 0, "j": 1, "k": 2}[pe[0]]
    v = np.asarray(img.direction)[:3, ax] * (-1 if pe.endswith("-") else 1); v /= np.linalg.norm(v)
    design = syntx.make_design("shells", bvals=bv, bvecs=g, pe_axis=v)
    levels = np.asarray(design.levels); D = np.asarray(img.direction)[:3, :3]
    blocks = design_blocks(levels, "shells", bvecs=g, bvals=bv, lmax=4, min_frames_per_column=3)
    X = np.hstack([B for _, B in blocks]); T = img.shape[-1]
    res = {}
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    for i, eng in enumerate(("unified", "projected")):
        r = syntx.motion_correct_design(img, design, engine=eng, verbose=False)
        corr = r["motion_corrected"].numpy(); m = corr.mean(-1); mask = m > 0.3 * np.percentile(m, 99)
        Y = corr.reshape(-1, T)[mask.reshape(-1)].T.astype(np.float64)
        Q, _ = np.linalg.qr(X); R = (Y - Q @ (Q.T @ Y)).T                       # (V, T) residual against the shell model
        zidx = np.nonzero(mask)[2]
        nz = corr.shape[2]
        per_slice = np.zeros((nz, T)); cnt = np.zeros(nz)
        for z in range(nz):
            sel = zidx == z
            if sel.sum() > 20:
                per_slice[z] = np.sqrt((R[sel] ** 2).mean(0)); cnt[z] = sel.sum()
        ok = cnt > 20
        ps = per_slice[ok]
        ps_norm = ps / (np.median(ps, axis=1, keepdims=True) + 1e-9)             # per slice, relative to its own typical level
        stripe = np.std(ps_norm, axis=0)                                           # per volume: how uneven the residual is across slices
        t = np.asarray(r["translations_mm"], float); ro = t @ D[:, 0]
        step = np.r_[0, np.abs(np.diff(ro))]
        rr = float(np.corrcoef(stripe[1:], step[1:])[0, 1])
        frame_rms = np.sqrt((R ** 2).mean(0)); rf = float(np.corrcoef(frame_rms[1:], step[1:])[0, 1])
        res[eng] = {"readout_step_mad": float(1.4826 * np.median(np.abs(np.diff(ro) - np.median(np.diff(ro))))), "stripe_vs_readout_step_r": rr,
                    "frame_rms_vs_readout_step_r": rf, "stripe_mean": float(stripe.mean()), "n_slices": int(ok.sum())}
        im = axes[i, 0].imshow(ps_norm, aspect="auto", cmap="magma", vmin=0.6, vmax=1.6); axes[i, 0].set_title(f"{eng}: per-slice residual RMS (rel. to slice median) vs volume", fontsize=9)
        axes[i, 0].set_xlabel("volume"); axes[i, 0].set_ylabel("slice"); plt.colorbar(im, ax=axes[i, 0])
        axes[i, 1].plot(stripe, lw=0.8, label="slice-stripe index"); ax2 = axes[i, 1].twinx(); ax2.plot(step, lw=0.8, color="C1", label="|readout step|")
        axes[i, 1].set_title(f"stripe index vs |readout step|: r = {rr:.2f}", fontsize=9); axes[i, 1].set_xlabel("volume")
        axes[i, 2].scatter(step[1:], stripe[1:], s=8, c=levels[1:], cmap="viridis"); axes[i, 2].set_xlabel("|readout step| (mm)"); axes[i, 2].set_ylabel("stripe index"); axes[i, 2].set_title("per volume, colour = shell", fontsize=9)
        print(eng, json.dumps(res[eng]), flush=True)
    fig.tight_layout(); fig.savefig(os.path.join(out, "slice_residual_diagnostic.png"), dpi=110)
    json.dump(res, open(os.path.join(out, "slice_residual_diagnostic.json"), "w"), indent=1)
    print("DONE", os.path.join(out, "slice_residual_diagnostic.png"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
