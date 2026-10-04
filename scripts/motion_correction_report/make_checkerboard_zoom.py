import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ants

BG = "#0b0f17"; PANEL = "#141b2d"; TEXT = "#e2e8f0"; MUTED = "#94a3b8"; BORDER = "#27354a"

def checkerboard(a, b, tile=6):
    out = np.zeros_like(a)
    ny, nx = a.shape
    for i in range(0, ny, tile):
        for j in range(0, nx, tile):
            use_a = ((i // tile) + (j // tile)) % 2 == 0
            out[i:i+tile, j:j+tile] = (a if use_a else b)[i:i+tile, j:j+tile]
    return out

def norm(a, lo=1, hi=99):
    a = a.astype(np.float32)
    v0, v1 = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - v0) / max(v1 - v0, 1e-6), 0, 1)

def panel(ax, img2d, title):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(img2d, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, color=TEXT, fontsize=10)

info = json.load(open("/tmp/motion_report/asl_worst_frame_info.json"))
ref = ants.image_read("/tmp/motion_report/asl_reference.nii.gz")
raw = ants.image_read("/tmp/motion_report/asl_worst_frame_raw.nii.gz")
corr = ants.image_read("/tmp/motion_report/asl_worst_frame_corrected.nii.gz")
ref_a, raw_a, corr_a = ref.numpy(), raw.numpy(), corr.numpy()
zi = ref_a.shape[2] // 2
sl = lambda a: a[:, :, zi].T[::-1, :]
ref_n, raw_n, corr_n = norm(sl(ref_a)), norm(sl(raw_a)), norm(sl(corr_a))

cb_raw = checkerboard(raw_n, ref_n, tile=4)
cb_corr = checkerboard(corr_n, ref_n, tile=4)

# Crop to the anterior/frontal region (top of the image in our flip convention) where
# the brightest, most structurally distinctive tissue sits -- easiest region to judge
# edge continuity by eye.
h, w = cb_raw.shape
crop = cb_raw[int(h*0.08):int(h*0.45), int(w*0.15):int(w*0.85)], cb_corr[int(h*0.08):int(h*0.45), int(w*0.15):int(w*0.85)]

fig, axes = plt.subplots(1, 2, figsize=(11, 6.5), facecolor=BG)
panel(axes[0], crop[0], f"BEFORE correction (zoomed)\nraw frame #{info['worst_frame_idx']}, FD={info['worst_fd']:.2f}mm, checkerboarded vs reference")
panel(axes[1], crop[1], "AFTER correction (zoomed)\nsame tiling, corrected frame vs reference")
fig.suptitle("SOCOM ASL: zoomed checkerboard -- a real 6.1mm motion event, before vs after",
             color=TEXT, fontsize=13, fontweight="bold")
fig.patch.set_facecolor(BG)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig("/tmp/motion_report/asl_checkerboard_zoom.png", dpi=160, facecolor=BG)
print("wrote asl_checkerboard_zoom.png")
