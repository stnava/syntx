"""One-case study of the head prior in likelihood units (surround_scale='likelihood', weight in nats per mm) on a real
multi-shell DWI series: unified, joint and projected at several prior weights. Writes JSON, an HTML table and a figure of
the per-frame pose along the acquisition axes. Usage: python benchmarks/prior_units_case_study.py <dwi.nii.gz> <out dir>
[--weights 0,200,2000]"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from motion_cohort_gate import _pe_axis, _signatures  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dwi")
    ap.add_argument("out")
    ap.add_argument("--weights", default="0,200,2000")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    import ants
    import syntx

    os.makedirs(args.out, exist_ok=True)
    img = ants.image_read(args.dwi)
    bv = np.loadtxt(args.dwi.replace(".nii.gz", ".bval"))
    g = np.loadtxt(args.dwi.replace(".nii.gz", ".bvec")).T
    pe = _pe_axis(args.dwi, img)
    design = syntx.make_design("shells", bvals=bv, bvecs=g, pe_axis=pe)
    levels = np.asarray(design.levels)
    D = np.asarray(img.direction)[:3, :3]
    variants = [("unified", {}), ("joint", {})] + [(f"projected, prior {w:g} nats/mm", {"engine": "projected", "surround_weight": float(w)})
                                                   for w in (float(x) for x in args.weights.split(","))]
    results, traces = {}, {}
    for label, kw in variants:
        eng = kw.pop("engine", label)
        t0 = time.time()
        r = syntx.motion_correct_design(img, design, engine=eng, device=args.device, verbose=False, **kw)
        sig = _signatures(r, img, levels)
        sig["seconds"] = time.time() - t0
        pr = r.get("prior") or {}
        sig["prior_scale"] = (pr.get("scale_log") or [{}])[-1]
        results[label] = sig
        t = np.asarray(r["translations_mm"], float)
        traces[label] = {"t_ro": (t @ D[:, 0]).tolist(), "t_pe": (t @ D[:, 1]).tolist(), "t_sl": (t @ D[:, 2]).tolist()}
        print(label, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in sig.items() if k in ("fd_mean", "fd_max", "tsd_rel", "seconds")}), sig["step_mad_mm"], flush=True)
        json.dump({"results": results, "traces": traces, "bvals": bv.tolist(), "levels": levels.tolist()}, open(os.path.join(args.out, "case.json"), "w"), indent=1)
    _report(results, traces, bv, args.out)
    print("DONE", os.path.join(args.out, "prior_units_case_study.html"))


def _report(results, traces, bv, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for label, tr in traces.items():
        for ax, key in zip(axes, ("t_ro", "t_pe", "t_sl")):
            ax.plot(tr[key], lw=0.8, label=label)
    for ax, key in zip(axes, ("readout translation (B0-immune)", "phase-encode translation", "slice translation")):
        ax.set_ylabel("mm"); ax.set_title(key, fontsize=9)
        for i in np.flatnonzero(np.asarray(bv) < 50):
            ax.axvline(i, color="k", alpha=0.08)
    axes[0].legend(fontsize=7); axes[-1].set_xlabel("volume (vertical lines: b0)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "traces.png"), dpi=110); plt.close(fig)
    rows = []
    for label, s in results.items():
        st, lk = s["step_mad_mm"], s["design_locked_mm"]
        rows.append(f"<tr><td>{label}</td><td>{s['fd_mean']:.3f}</td><td>{s['fd_max']:.2f}</td><td>{st['t_ro']:.3f}</td><td>{st['t_pe']:.3f}</td>"
                    f"<td>{st['t_sl']:.3f}</td><td>{st['pitch']:.3f}</td><td>{lk['t_pe']:.3f}</td><td>{s['tsd_rel']:.4f}</td>"
                    f"<td>{s['frame_converged_frac']:.2f}</td><td>{s['seconds']:.0f}</td></tr>")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Prior in likelihood units: one DWI case</title>
<style>body{{font-family:sans-serif;margin:2em;max-width:1100px}} table{{border-collapse:collapse}} td,th{{border:1px solid #999;padding:3px 6px;font-size:13px}} th{{background:#eee}}</style></head>
<body><h1>Head prior in likelihood units: one real multi-shell DWI series</h1>
<p>The prior weight is in nats per mm of second difference beyond the 0.05 mm knee, entering the loss multiplied by the measured
sigma^2 / S of the stage's points (<code>surround_scale='likelihood'</code>). Step MAD: robust frame-to-frame step per axis
(mm; pitch at 50 mm). The readout step is the B0-immune floor: real jitter the prior must not shrink below what unified and joint
report; the phase-encode step carries eddy and contrast effects. design_locked: period-2 pose amplitude along PE relative to the
anchor level. tsd_rel: alignment (lower is better).</p>
<table><tr><th>engine</th><th>FD mean</th><th>FD max</th><th>step readout</th><th>step PE</th><th>step slice</th><th>step pitch</th>
<th>locked PE</th><th>tsd_rel</th><th>converged</th><th>s</th></tr>{''.join(rows)}</table>
<h3>Pose along the acquisition axes</h3><img src="traces.png" style="max-width:100%"></body></html>"""
    open(os.path.join(out, "prior_units_case_study.html"), "w").write(html)


if __name__ == "__main__":
    main()
