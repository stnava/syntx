import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG = "#0b0f17"; PANEL = "#141b2d"; TEXT = "#e2e8f0"; MUTED = "#94a3b8"; BORDER = "#27354a"

def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BORDER)

sift = json.load(open("/tmp/motion_report/pet_sift3d_results.json"))
mattes_fd_full = json.load(open("/tmp/motion_report/pet_mattes_fd_for_sift_comparison.json"))

idxs = sorted(sift.keys(), key=int)
sift_vals = [sift[i]["translation_mm"] for i in idxs]
mattes_vals = [mattes_fd_full.get(i) for i in idxs]
n_matches = [sift[i]["n_matches"] for i in idxs]

fig, ax1 = plt.subplots(figsize=(11, 4.6), facecolor=BG)
style_ax(ax1)
x = np.arange(len(idxs))
w = 0.35
sift_plot = [v if v is not None else 0 for v in sift_vals]
mattes_plot = [v if v is not None else 0 for v in mattes_vals]
bars1 = ax1.bar(x - w/2, mattes_plot, w, color="#ef4444", label="Mattes-MI (intensity-based) FD")
bars2 = ax1.bar(x + w/2, sift_plot, w, color="#22c55e", label="SIFT3D (feature-based) |t|")
for i, v in enumerate(sift_vals):
    if v is None:
        ax1.text(x[i] + w/2, 0.3, "no fit\n(too few\nmatches)", ha="center", color="#94a3b8", fontsize=7)
ax1.set_xticks(x); ax1.set_xticklabels([f"frame\n{i}\n(n={n})" for i, n in zip(idxs, n_matches)], color=TEXT, fontsize=8)
ax1.set_ylabel("displacement (mm)", color=MUTED)
ax1.set_title("FDG-PET: SIFT3D feature-based rigid estimate vs Mattes-MI intensity-based FD, same frames\n"
              "SIFT3D is stable (0.6-2.1mm) wherever it can fit at all; Mattes-MI swings erratically (0-9.4mm)",
              color=TEXT, fontsize=11, fontweight="bold")
ax1.legend(facecolor=BG, labelcolor=TEXT, frameon=False, fontsize=9)
fig.patch.set_facecolor(BG)
fig.tight_layout()
fig.savefig("/tmp/motion_report/pet_sift3d_comparison.png", dpi=145, facecolor=BG)
print("wrote pet_sift3d_comparison.png")
