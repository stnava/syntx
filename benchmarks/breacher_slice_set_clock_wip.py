"""WIP: apply the slice-set clock (Stage 1) to breacher runs and plot FD before/after. 2026-10-31.

Usage: python benchmarks/breacher_slice_set_clock_wip.py OUT_PREFIX run1.nii.gz [run2 ...]
Before = Power FD of the raw full-volume poses in the acquisition frame; after = the same with the volume-locked
share removed (``fd_head_vl``). The band-limited FD of the production result is drawn for reference.
"""
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ants
import syntx
from syntx.contract import ComputeContext

prefix, paths = sys.argv[1], sys.argv[2:]
dev = ComputeContext().reg_device
out = {}
fig, axes = plt.subplots(len(paths), 1, figsize=(14, 3.2 * len(paths)), dpi=130, sharex=False)
axes = np.atleast_1d(axes)
for ax, path in zip(axes, paths):
    tag = path.split("/")[-1].split("_task")[0]
    img = ants.image_read(path)
    side = path.replace(".nii.gz", ".json")
    pe = np.asarray(syntx.resolve_phase_encode(img, sidecar=side, on_oblique="warn").physical_vector, float)
    t0 = time.time()
    res = syntx.motion_correct_design(img, syntx.make_design("none", n_frames=int(img.shape[-1]), pe_axis=pe), engine="joint",
                                      device=dev, sidecar=side, pe_axis=pe, debias_pe_pseudomotion=True)
    t1 = time.time()
    core = syntx.slice_set_clock(img, res, sidecar=side, device=dev)
    t2 = time.time()
    fd, fdh = core["fd"], core["fd_head_vl"]
    rec = {"seconds_full": round(t1 - t0, 1), "seconds_clock": round(t2 - t1, 1), "early_set_source": core["early_set_source"],
           "readings": core["readings"], "per_dof": core["per_dof"],
           "fd_mean_before": round(float(fd[1:].mean()), 4), "fd_mean_after": round(float(fdh[1:].mean()), 4),
           "pct_gt_0p5_before": round(float((fd[1:] > 0.5).mean() * 100), 1), "pct_gt_0p5_after": round(float((fdh[1:] > 0.5).mean() * 100), 1),
           "pct_gt_0p3_before": round(float((fd[1:] > 0.3).mean() * 100), 1), "pct_gt_0p3_after": round(float((fdh[1:] > 0.3).mean() * 100), 1),
           "fd_bandlimited_mean_production": res.get("summary", {}).get("fd_bandlimited_mean"),
           "fd": fd.tolist(), "fd_head_vl": fdh.tolist(), "tr": core["tr_s"], "band_hz": list(core["band_hz"]),
           "dof": core["dof"], "poses_full": np.round(core["poses_full"], 4).tolist(),
           "poses_early": np.round(core["poses_early"], 4).tolist(), "poses_late": np.round(core["poses_late"], 4).tolist(),
           "x_vl": np.round(core["x_vl"], 4).tolist(),
           "fd_bandlimited_production": (np.asarray(res["fd_bandlimited"], float).tolist() if res.get("fd_bandlimited") is not None else None)}
    out[tag] = rec
    print(tag, json.dumps({k: v for k, v in rec.items() if k not in ("fd", "fd_head_vl", "per_dof")}), flush=True)
    x = np.arange(len(fd))
    ax.plot(x, fd, color="#b0413e", lw=1.0, label=f"before: Power FD of raw poses (mean {rec['fd_mean_before']:.3f} mm, >0.5: {rec['pct_gt_0p5_before']:.1f}%)")
    ax.plot(x, fdh, color="#1f5f8b", lw=1.0, label=f"after: fd_head_vl (mean {rec['fd_mean_after']:.3f} mm, >0.5: {rec['pct_gt_0p5_after']:.1f}%)")
    fb = res.get("fd_bandlimited")
    if fb is not None:
        ax.plot(x, np.asarray(fb, float), color="#7a7a7a", lw=0.8, ls=":", label="production fd_bandlimited (Jenkinson) for reference")
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    r = core["readings"]
    ax.set_title(f"{tag}  clock reading t_sl={r['t_sl']}, t_pe={r['t_pe']}, pitch={r['pitch']}  "
                 f"(phase t_sl {core['per_dof']['t_sl']['phase_deg']:+.0f}°, coh {core['per_dof']['t_sl']['coherence']:.2f}, removed share {core['per_dof']['t_sl']['removed_share']:.2f})", fontsize=9)
    ax.set_ylabel("FD (mm)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("volume (TR 2.0 s)")
fig.tight_layout()
fig.savefig(prefix + ".png")
with open(prefix + ".json", "w") as fh:
    json.dump(out, fh, indent=1)
print("saved", prefix + ".png")
