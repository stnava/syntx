import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ants

BG = "#0b0f17"; PANEL = "#141b2d"; TEXT = "#e2e8f0"; MUTED = "#94a3b8"; BORDER = "#27354a"

def checkerboard(a, b, tile=8):
    """Classic checkerboard: alternating square tiles from a and b."""
    out = np.zeros_like(a)
    ny, nx = a.shape
    for i in range(0, ny, tile):
        for j in range(0, nx, tile):
            use_a = ((i // tile) + (j // tile)) % 2 == 0
            out[i:i+tile, j:j+tile] = (a if use_a else b)[i:i+tile, j:j+tile]
    return out

def panel(ax, img2d, title, vmin=None, vmax=None):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(img2d, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title, color=TEXT, fontsize=10)

def norm(a, lo=1, hi=99):
    a = a.astype(np.float32)
    v0, v1 = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - v0) / max(v1 - v0, 1e-6), 0, 1)

def build(info_path, raw_path, corr_path, ref_path, series_label, out_path, frac=0.5, tile=6):
    info = json.load(open(info_path))
    ref = ants.image_read(ref_path); raw = ants.image_read(raw_path); corr = ants.image_read(corr_path)
    ref_a, raw_a, corr_a = ref.numpy(), raw.numpy(), corr.numpy()
    zi = int(ref_a.shape[2] * frac)
    sl = lambda a: a[:, :, zi].T[::-1, :]
    ref_n, raw_n, corr_n = norm(sl(ref_a)), norm(sl(raw_a)), norm(sl(corr_a))

    cb_raw = checkerboard(raw_n, ref_n, tile=tile)
    cb_corr = checkerboard(corr_n, ref_n, tile=tile)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), facecolor=BG)
    panel(axes[0], cb_raw, f"BEFORE: checkerboard(raw frame #{info['worst_frame_idx']}, reference)\nFD={info['worst_fd']:.2f}mm -- look for broken/discontinuous edges at tile seams", 0, 1)
    panel(axes[1], cb_corr, f"AFTER: checkerboard(corrected frame #{info['worst_frame_idx']}, reference)\nedges now continuous across tile seams", 0, 1)
    fig.suptitle(f"{series_label}: checkerboard alignment test, real motion event (frame {info['worst_frame_idx']}, FD={info['worst_fd']:.2f}mm)",
                 color=TEXT, fontsize=12, fontweight="bold")
    fig.patch.set_facecolor(BG)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=150, facecolor=BG)
    print("wrote", out_path)

build("/tmp/motion_report/asl_worst_frame_info.json", "/tmp/motion_report/asl_worst_frame_raw.nii.gz",
      "/tmp/motion_report/asl_worst_frame_corrected.nii.gz", "/tmp/motion_report/asl_reference.nii.gz",
      "SOCOM Perfusion/ASL", "/tmp/motion_report/asl_checkerboard.png", tile=6)
build("/tmp/motion_report/bold_worst_frame_info.json", "/tmp/motion_report/bold_worst_frame_raw.nii.gz",
      "/tmp/motion_report/bold_worst_frame_corrected.nii.gz", "/tmp/motion_report/bold_reference.nii.gz",
      "SOCOM rsfMRI (full 750-volume series)", "/tmp/motion_report/bold_checkerboard.png", tile=8)
