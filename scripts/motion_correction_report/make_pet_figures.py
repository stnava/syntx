import json, base64, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ants

BG = "#0b0f17"; PANEL = "#141b2d"; TEXT = "#e2e8f0"; MUTED = "#94a3b8"; BORDER = "#27354a"
C_SLOW = "#64748b"; C_OLD = "#ef4444"; C_NEW = "#22c55e"

def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.title.set_color(TEXT)

def fig_to_b64(fig, dpi=140):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

pet = json.load(open("/tmp/motion_report/pet_results.json"))
figs = {}

fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=BG)
style_ax(ax)
ax.plot(pet["slow_fd"], color=C_SLOW, lw=1.8, marker="o", label=f"trusted slow ('pytorch', n={pet['n_slow']}, {pet['slow_time']:.0f}s)")
ax.plot(pet["old_fd"][:pet['n_slow']], color=C_OLD, lw=1.5, marker="s", label=f"OLD naive-mean + adaptive ({pet['old_time']:.1f}s, full n=32)")
ax.plot(pet["new_fd"][:pet['n_slow']], color=C_NEW, lw=1.5, marker="^", label=f"NEW low-motion-subset + adaptive ({pet['new_time']:.1f}s, full n=32)")
ax.set_xlabel("frame (post-uptake window, start=frame 40)", color=MUTED)
ax.set_ylabel("FD (mm)", color=MUTED)
ax.set_title("FDG-PET: neither reference strategy clearly improves on the other -- a genuinely harder problem than DWI/BOLD", fontsize=10, fontweight="bold")
ax.legend(facecolor=BG, labelcolor=TEXT, frameon=False, fontsize=8)
fig.patch.set_facecolor(BG)
fig.tight_layout()
figs["pet_fd_comparison"] = fig_to_b64(fig)

ref_naive = ants.image_read("/tmp/motion_report/pet_ref_naive.nii.gz")
ref_subset = ants.image_read("/tmp/motion_report/pet_ref_subset.nii.gz")
fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), facecolor=BG)
for ax, (im, title) in zip(axes, [(ref_naive, "Naive whole-series mean\n(32 post-uptake frames, still uptake-drift-blurred)"),
                                    (ref_subset, "Low-motion-subset reference\n(10 lowest-coarse-FD frames)")]):
    arr = im.numpy()
    z = arr.shape[2] // 2
    sl = arr[:, :, z].T[::-1, :]
    ax.imshow(sl, cmap="inferno")
    ax.set_title(title, color=TEXT, fontsize=9)
    ax.axis("off")
fig.patch.set_facecolor(BG)
fig.tight_layout()
figs["pet_reference_comparison"] = fig_to_b64(fig)

json.dump(figs, open("/tmp/motion_report/pet_figs.json", "w"))
print("wrote", len(figs), "PET figures")
