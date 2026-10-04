import json, base64, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ants

BG = "#0b0f17"
PANEL = "#141b2d"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
BORDER = "#27354a"
C_SLOW = "#64748b"
C_OLD = "#ef4444"
C_NEW = "#22c55e"
C_B0 = "#38bdf8"
C_DWI = "#f59e0b"

def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)

def fig_to_b64(fig, dpi=140):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

figs = {}

# ============================= BOLD =============================
bold = json.load(open("/tmp/motion_report/bold_results.json"))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), facecolor=BG)
for ax, name in zip(axes, ["SOCOM", "PPMI"]):
    r = bold[name]
    style_ax(ax)
    ax.plot(r["slow_fd"], color=C_SLOW, lw=1.8, label=f"trusted slow ('pytorch', {r['slow_time']:.0f}s)")
    ax.plot(r["old_gate_fd"], color=C_OLD, lw=1.3, alpha=0.9, label=f"old correlation-gate ({r['old_gate_time']:.1f}s)")
    ax.plot(r["adaptive_fd"], color=C_NEW, lw=1.3, alpha=0.9, label=f"new adaptive gate ({r['adaptive_time']:.1f}s)")
    ax.set_title(f"BOLD -- {name} ({r['shape'][-1]} frames, {r['spacing'][0]}mm iso)", fontsize=11, fontweight="bold")
    ax.set_xlabel("frame"); ax.set_ylabel("FD (mm)")
    ax.legend(facecolor=BG, labelcolor=TEXT, frameon=False, fontsize=8, loc="upper right")
fig.patch.set_facecolor(BG)
fig.tight_layout()
figs["bold_fd_traces"] = fig_to_b64(fig)

fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
style_ax(ax)
labels = ["SOCOM\n(750→64 fr, 2mm)", "PPMI\n(240→64 fr, 3.5mm)"]
slow_t = [bold["SOCOM"]["slow_time"], bold["PPMI"]["slow_time"]]
old_t = [bold["SOCOM"]["old_gate_time"], bold["PPMI"]["old_gate_time"]]
new_t = [bold["SOCOM"]["adaptive_time"], bold["PPMI"]["adaptive_time"]]
x = np.arange(2); w = 0.25
ax.bar(x - w, slow_t, w, color=C_SLOW, label="trusted slow ('pytorch')")
ax.bar(x, old_t, w, color=C_OLD, label="old correlation-gate")
ax.bar(x + w, new_t, w, color=C_NEW, label="new adaptive gate")
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(labels, color=TEXT, fontsize=9)
ax.set_ylabel("wall-clock time (s, log scale)")
ax.set_title("BOLD motion correction: timing (64-frame truncation)", fontsize=11, fontweight="bold")
ax.legend(facecolor=BG, labelcolor=TEXT, frameon=False, fontsize=8)
fig.patch.set_facecolor(BG)
fig.tight_layout()
figs["bold_timing"] = fig_to_b64(fig)

# ============================= DWI =============================
dwi = json.load(open("/tmp/motion_report/dwi_results.json"))
bvals = np.array(dwi["bvals"])
b0_mask = bvals <= 50

fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=BG)
style_ax(ax)
x = np.arange(len(bvals))
ax.scatter(x[b0_mask], np.array(dwi["old_fd"])[b0_mask], color=C_B0, marker="o", s=50, label="b0 (old: blended mean ref)", zorder=3)
ax.scatter(x[~b0_mask], np.array(dwi["old_fd"])[~b0_mask], color=C_DWI, marker="o", s=20, alpha=0.6, label="DWI (old: blended mean ref)")
ax.scatter(x[b0_mask], np.array(dwi["b0_only_fd"]), color=C_NEW, marker="^", s=70, label="b0 (trusted: b0-only internal baseline)", zorder=4)
ax.axhline(np.mean(dwi["b0_only_fd"]), color=C_NEW, ls="--", lw=1, alpha=0.6)
ax.set_xlabel("volume index"); ax.set_ylabel("FD (mm)")
ax.set_title(f"DWI (SOCOM, {dwi['n_total']} volumes, {dwi['n_b0']} b0 + {dwi['n_total']-dwi['n_b0']} DWI): "
             f"the blended-mean reference inflates FD, worst for the minority b0 class", fontsize=10, fontweight="bold")
ax.legend(facecolor=BG, labelcolor=TEXT, frameon=False, fontsize=8)
fig.patch.set_facecolor(BG)
fig.tight_layout()
figs["dwi_fd_old_vs_trusted"] = fig_to_b64(fig)

fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=BG)
style_ax(ax)
groups = ["b0\n(blended mean,\nnarrow/adaptive)", "b0\n(grouped b0-ref,\nwide/full)", "b0\n(b0-only internal,\ntrusted)"]
vals = [np.mean(np.array(dwi["old_fd"])[b0_mask]), 3.490, np.mean(dwi["b0_only_fd"])]
colors = [C_OLD, "#a78bfa", C_NEW]
ax.bar(groups, vals, color=colors)
ax.set_ylabel("mean FD (mm)")
ax.set_title("b0 apparent motion by reference/search strategy", fontsize=10, fontweight="bold")
for i, v in enumerate(vals):
    ax.text(i, v + 0.15, f"{v:.2f}", ha="center", color=TEXT, fontsize=9)
fig.patch.set_facecolor(BG)
fig.tight_layout()
figs["dwi_b0_fd_comparison"] = fig_to_b64(fig)

# Reference image comparison -- mid-axial slice
ref_blend = ants.image_read("/tmp/motion_report/dwi_ref_blended.nii.gz")
ref_grouped = ants.image_read("/tmp/motion_report/dwi_ref_grouped.nii.gz")
fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), facecolor=BG)
for ax, (im, title) in zip(axes, [(ref_blend, "OLD: blended whole-series mean\n(92 DWI + 7 b0, dominated by DWI contrast)"),
                                    (ref_grouped, "NEW: b0-anchored low-motion-subset reference\n(built from 4 b0 frames only)")]):
    arr = im.numpy()
    z = arr.shape[2] // 2
    sl = arr[:, :, z].T[::-1, :]
    ax.imshow(sl, cmap="gray")
    ax.set_title(title, color=TEXT, fontsize=9)
    ax.axis("off")
fig.patch.set_facecolor(BG)
fig.tight_layout()
figs["dwi_reference_comparison"] = fig_to_b64(fig)

json.dump(figs, open("/tmp/motion_report/figs_partial.json", "w"))
print("wrote", len(figs), "figures (BOLD+DWI); PET pending")
