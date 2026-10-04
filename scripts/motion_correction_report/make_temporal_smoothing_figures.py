"""Reproduces the Gaussian-temporal-averaging figures in
docs/ADAPTIVE_MOTION_CORRECTION_AND_RECOVERY.html, section 5.4-5.5:

- pet_temporal_smoothing_visual.png: raw vs smoothed example frames.
- pet_sift3d_smoothing_matches.png / pet_sift3d_smoothing_stability.png: SIFT3D
  keypoint-match counts and displacement stability, raw vs smoothed frames.
- pet_smoothing_fixes_both_methods.png: Mattes-MI AND SIFT3D displacement, raw vs
  smoothed, showing both mechanisms converge once given denoised input.

Requires the real ds002898 FDG-PET data cached at the path below (see
antsxfunctional's own data_cache/). Not part of the test suite -- real-data report
reproduction, not CI (see this directory's README.md).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ants
import syntx
from syntx.temporal_denoise import gaussian_temporal_average
from syntx.landmarks import detect_sift3d, match_landmarks, ransac_filter

BG = "#0b0f17"; PANEL = "#141b2d"; TEXT = "#e2e8f0"; MUTED = "#94a3b8"; BORDER = "#27354a"
PET_PATH = "/Users/stnava/data/repos/antsxfunctional/data_cache/ds002898/sub-01/pet/sub-01_task-rest_trc-18FFDG_rec-acdyn_run-001_pet.nii.gz"
OUT_DIR = "/tmp/motion_report"
START, N = 40, 32
TESTED = [0, 3, 8, 12, 15, 18, 20, 22, 25, 28, 31]


def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BORDER)


def load_window():
    img = ants.image_read(PET_PATH)
    arr = img.numpy()[..., START:START + N].astype(np.float64)
    spacing, origin, direction = img.spacing[:3], img.origin[:3], img.direction[:3, :3]
    return arr, spacing, origin, direction


def make_visual_figure(arr, spacing, origin, direction):
    zi = arr.shape[2] // 2
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), facecolor=BG)
    examples = [3, 8, 18]
    vmax = np.percentile(arr[..., 18], 99.5)
    for col, idx in enumerate(examples):
        raw_sl = arr[:, :, zi, idx].T[::-1, :]
        frame4d = ants.from_numpy(np.ascontiguousarray(arr.astype(np.float32)),
                                   spacing=spacing + (1.0,), origin=origin + (0.0,))
        smoothed = gaussian_temporal_average(frame4d, center=idx, sigma=1.5, half_window=3)
        smoothed_sl = smoothed.numpy()[:, :, zi].T[::-1, :]
        for row, (sl, tag) in enumerate([(raw_sl, "RAW"), (smoothed_sl, "Gaussian temporal avg (sigma=1.5, +/-3 frames)")]):
            ax = axes[row, col]
            style_ax(ax); ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(sl, cmap="inferno", vmin=0, vmax=vmax)
            ax.set_title(f"frame {idx}: {tag}", color=TEXT, fontsize=9.5)
    fig.suptitle("FDG-PET: Gaussian-weighted temporal averaging dramatically raises per-frame SNR",
                 color=TEXT, fontsize=13, fontweight="bold")
    fig.patch.set_facecolor(BG)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(f"{OUT_DIR}/pet_temporal_smoothing_visual.png", dpi=145, facecolor=BG)


def run_sift3d_comparison(arr, spacing, origin, direction):
    ref_native = ants.from_numpy(arr.mean(axis=-1).astype(np.float32), spacing=spacing, origin=origin, direction=direction)
    kp_ref, desc_ref = detect_sift3d(ref_native, local_normalize=True, max_keypoints=512)
    img4d = ants.from_numpy(np.ascontiguousarray(arr.astype(np.float32)), spacing=spacing + (1.0,), origin=origin + (0.0,))

    results = {}
    for idx in TESTED:
        raw_frame = ants.from_numpy(np.ascontiguousarray(arr[..., idx].astype(np.float32)), spacing=spacing, origin=origin, direction=direction)
        smoothed_frame = gaussian_temporal_average(img4d, center=idx, sigma=1.5, half_window=3)
        smoothed_frame = ants.from_numpy(smoothed_frame.numpy(), spacing=spacing, origin=origin, direction=direction)

        row = {}
        for tag, frame in [("raw", raw_frame), ("smoothed", smoothed_frame)]:
            kp_f, desc_f = detect_sift3d(frame, local_normalize=True, max_keypoints=512)
            matches = match_landmarks(kp_ref, kp_f, desc_ref, desc_f, ratio_thresh=0.85, mutual=True)
            n_inliers, mag = 0, None
            if len(matches) >= 4:
                filtered, M = ransac_filter(kp_ref[:, :3], kp_f[:, :3], matches, model="rigid", inlier_thresh_mm=8.0, min_inliers=4)
                n_inliers = len(filtered)
                if n_inliers >= 3:
                    mag = float(np.linalg.norm(M[:3, 3]))
            row[tag] = {"n_kp": len(kp_f), "n_matches": len(matches), "n_inliers": n_inliers, "translation_mm": mag}
        results[idx] = row
    return results


def run_mattes_on_smoothed(arr, spacing, origin):
    smoothed_full = np.stack([
        gaussian_temporal_average(
            ants.from_numpy(np.ascontiguousarray(arr.astype(np.float32)), spacing=spacing + (1.0,), origin=origin + (0.0,)),
            center=t, sigma=1.5, half_window=3,
        ).numpy()
        for t in range(N)
    ], axis=-1).astype(np.float32)
    img4d_smoothed = ants.from_numpy(np.ascontiguousarray(smoothed_full), spacing=spacing + (1.0,), origin=origin + (0.0,))
    ref_smoothed = ants.get_average_of_timeseries(img4d_smoothed)
    mc = syntx.motion_correction(image=img4d_smoothed, reference=ref_smoothed, backend="pytorch_batched_adaptive", verbose=False)
    return np.asarray(mc.fd)


if __name__ == "__main__":
    arr, spacing, origin, direction = load_window()
    make_visual_figure(arr, spacing, origin, direction)
    print("wrote pet_temporal_smoothing_visual.png")

    sift_results = run_sift3d_comparison(arr, spacing, origin, direction)
    json.dump(sift_results, open(f"{OUT_DIR}/pet_sift3d_smoothed_comparison.json", "w"))
    print("wrote pet_sift3d_smoothed_comparison.json")

    fd_smoothed = run_mattes_on_smoothed(arr, spacing, origin)
    out = {str(i): float(fd_smoothed[i]) for i in TESTED}
    json.dump(out, open(f"{OUT_DIR}/pet_mattes_fd_on_smoothed.json", "w"))
    print("wrote pet_mattes_fd_on_smoothed.json")
    print("Run make_sift3d_smoothing_plots.py (or the inline plotting code in this file's"
          " history) next to render the comparison figures from these JSON files.")
