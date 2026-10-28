"""Plan step 2: the DWI design decision. On the two DWI cases, four engines: unified, joint, projected with l = 2 and
projected with l = 4 on the b >= 2500 shell. Per variant: readout / phase-encode / slice step MAD, FD, alignment,
convergence, the image-domain decomposition (l = 2 and l = 4 blocks, min 3 frames per column), the misattribution angle
between the rigid derivatives and the design-locked patterns, and time. The decision rule is evaluated and printed.

Decision rule (fixed before the run): a richer design wins only if its readout step is within 0.02 mm of unified's on
both cases, its residual share is not higher than joint's, and the l = 4 share after the engine falls below the motion
share. Otherwise joint stays the DWI default.

Usage: python benchmarks/dwi_design_decision.py <motion_cohort dir> <out dir> [--device auto]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from motion_cohort_gate import _pe_axis, _signatures  # noqa: E402

CASES = {
    "dwi031": ("FPA/sub-BLAST031/ses-01/dwi/sub-BLAST031_ses-01_run-001_dir-AP_dwi.nii.gz", "FPA BLAST031 dir-AP, high motion"),
    "dwi11": ("/Users/stnava/data/blast_cohorts/BIDS/SOCOM/sub-Blast-11/ses-01/dwi/sub-Blast-11_ses-01_run-001_dir-PA_dwi.nii.gz", "SOCOM Blast-11 dir-PA, low SNR"),
}
VARIANTS = [("unified", {}), ("joint", {}), ("projected l2", {"engine": "projected", "lmax_high_b": 2, "full_gauge": False}),
            ("projected l4", {"engine": "projected", "lmax_high_b": 4, "full_gauge": False}),
            ("projected l2 gauged", {"engine": "projected", "lmax_high_b": 2, "full_gauge": True})]
READOUT_TOL_MM = 0.02


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    import ants
    import syntx
    from syntx.motion_decomposition import design_blocks, image_variance_decomposition, misattribution_angles

    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, "dwi_design_decision.json")
    results = json.load(open(out_json)) if os.path.exists(out_json) else {}
    for key, (rel, label) in CASES.items():
        path = rel if rel.startswith("/") else os.path.join(args.root, rel)
        img = ants.image_read(path)
        bv = np.loadtxt(path.replace(".nii.gz", ".bval")); g = np.loadtxt(path.replace(".nii.gz", ".bvec")).T
        pe = _pe_axis(path, img)
        design = syntx.make_design("shells", bvals=bv, bvecs=g, pe_axis=pe)
        levels = np.asarray(design.levels)
        blocks = design_blocks(levels, "shells", bvecs=g, bvals=bv, lmax=4, min_frames_per_column=3)
        D = np.asarray(img.direction)[:3, :3]
        results.setdefault(key, {"label": label, "variants": {}})
        for vlabel, kw in VARIANTS:
            if vlabel in results[key]["variants"]:
                continue
            kw = dict(kw); eng = kw.pop("engine", vlabel)
            t0 = time.time()
            r = syntx.motion_correct_design(img, design, engine=eng, device=args.device, verbose=False, **kw)
            secs = time.time() - t0
            sig = _signatures(r, img, levels)
            corr = r["motion_corrected"].numpy(); m = corr.mean(-1); mask = m > 0.3 * np.percentile(m, 99)
            t = np.asarray(r["translations_mm"], float); rot = np.asarray(r["rotations_deg"], float); fd = np.asarray(r["fd"], float)
            dec = image_variance_decomposition(corr, blocks, np.c_[t, rot, fd], mask, fd=fd)
            com = np.asarray(r["group_means"][0].get_center_of_mass(), float) if r.get("group_means") else np.asarray(ants.get_center_of_mass(ants.slice_image(img, 3, 0)), float)
            ang = misattribution_angles(corr, blocks, img.spacing[:3], D, com, mask)
            rec = {"seconds": secs, "fd_mean": sig["fd_mean"], "fd_max": sig["fd_max"], "step_mad_mm": sig["step_mad_mm"], "tsd_rel": sig["tsd_rel"],
                   "frame_converged_frac": sig["frame_converged_frac"],
                   "terms": {x["term"]: x["fraction"] for x in dec["terms"]}, "resid_fd_corr": dec["residual_rms_fd_corr"],
                   "angles": ang["sets"], "design_columns": r["design"].get("columns"),
                   "trace_ro": (t @ D[:, 0]).tolist(), "trace_pe": (t @ D[:, 1]).tolist()}
            results[key]["variants"][vlabel] = rec
            json.dump(results, open(out_json, "w"), indent=1, default=float)
            print(key, vlabel, json.dumps({"fd": round(sig["fd_mean"], 3), "ro": round(sig["step_mad_mm"]["t_ro"], 3), "pe": round(sig["step_mad_mm"]["t_pe"], 3),
                                           "tsd": round(sig["tsd_rel"], 4), "l4": round(100 * rec["terms"].get("l=4 direction terms per shell", 0), 2),
                                           "motion": round(100 * rec["terms"]["motion regressors after the design"], 2), "resid": round(100 * rec["terms"]["residual"], 2),
                                           "min_angle": round(min(s_["min_angle_deg"] for s_ in ang["sets"]), 1) if ang["sets"] else None, "s": round(secs)}), flush=True)
    _report(results, args.out)
    print("DONE", os.path.join(args.out, "dwi_design_decision.html"))


def _report(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(results), 2, figsize=(15, 3.4 * len(results)))
    axes = np.atleast_2d(axes)
    rows, verdicts = [], []
    for i, (key, rec) in enumerate(results.items()):
        for vlabel, v in rec["variants"].items():
            axes[i, 0].plot(v["trace_ro"], lw=0.8, label=vlabel); axes[i, 1].plot(v["trace_pe"], lw=0.8, label=vlabel)
        axes[i, 0].set_title(f"{rec['label']}: readout translation (head)", fontsize=9); axes[i, 1].set_title("phase-encode translation (head)", fontsize=9)
        axes[i, 0].legend(fontsize=7); axes[i, 0].set_ylabel("mm")
        u = rec["variants"].get("unified"); j = rec["variants"].get("joint")
        for vlabel, v in rec["variants"].items():
            st = v["step_mad_mm"]; tm = v["terms"]
            l4 = tm.get("l=4 direction terms per shell", 0.0); mo = tm["motion regressors after the design"]; res = tm["residual"]
            ok_ro = u is not None and abs(st["t_ro"] - u["step_mad_mm"]["t_ro"]) <= READOUT_TOL_MM
            ok_res = j is not None and res <= j["terms"]["residual"] + 1e-9
            ok_l4 = l4 < mo
            ang = min(s_["min_angle_deg"] for s_ in v["angles"]) if v["angles"] else float("nan")
            verdicts.append((key, vlabel, ok_ro and ok_res and ok_l4))
            rows.append(f"<tr><td>{rec['label']}</td><td>{vlabel}</td><td>{v['fd_mean']:.3f}</td><td>{st['t_ro']:.3f}</td><td>{st['t_pe']:.3f}</td><td>{st['t_sl']:.3f}</td>"
                        f"<td>{v['tsd_rel']:.4f}</td><td>{100 * tm.get('l=2 direction terms per shell', 0):.2f}</td><td>{100 * l4:.2f}</td><td>{100 * mo:.2f}</td><td>{100 * res:.2f}</td>"
                        f"<td>{v['resid_fd_corr']:.2f}</td><td>{ang:.1f}</td><td>{v['frame_converged_frac']:.2f}</td><td>{v['seconds']:.0f}</td>"
                        f"<td>{'readout ok' if ok_ro else 'readout off'}, {'residual ok' if ok_res else 'residual worse'}, {'l4 < motion' if ok_l4 else 'l4 >= motion'}</td></tr>")
    fig.tight_layout(); figp = os.path.join(out, "dwi_design_decision.png"); fig.savefig(figp, dpi=110); plt.close(fig)
    winners = {v for k, v, ok in verdicts if ok and v != "unified"}
    both = [v for v in winners if all(ok for k, vv, ok in verdicts if vv == v)]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>DWI design decision</title>
<style>body{{font-family:sans-serif;margin:2em;max-width:1300px}} table{{border-collapse:collapse}} td,th{{border:1px solid #999;padding:3px 6px;font-size:12px}} th{{background:#eee}}</style></head>
<body><h1>DWI design decision (plan step 2)</h1>
<p>Four engines on two series. Step MAD: robust frame-to-frame step per axis (mm); the readout axis is the B0-immune floor. tsd_rel:
alignment (lower is better). l=2 / l=4 / motion / residual: percent of per-voxel temporal variance of the corrected series explained by
the direction terms, by the motion regressors after the design, and left as residual. r: per-frame residual RMS vs FD. min angle:
smallest principal angle (degrees) between the six rigid-derivative images and the design-locked spatial patterns of the corrected series
(small = a design pattern the pose can absorb). Rule: readout step within {READOUT_TOL_MM} mm of unified on both series, residual not above
joint's, l = 4 share below the motion share.</p>
<p><b>Verdict:</b> {("engine(s) passing the rule on both series: " + ", ".join(sorted(both))) if both else "no richer engine passes the rule on both series; joint stays the DWI default"}.</p>
<table><tr><th>series</th><th>engine</th><th>FD</th><th>step readout</th><th>step PE</th><th>step slice</th><th>tsd_rel</th><th>l=2 %</th><th>l=4 %</th><th>motion %</th><th>residual %</th><th>r</th><th>min angle</th><th>conv</th><th>s</th><th>rule</th></tr>{''.join(rows)}</table>
<img src="data:image/png;base64,{base64.b64encode(open(figp, 'rb').read()).decode()}" style="max-width:100%"></body></html>"""
    open(os.path.join(out, "dwi_design_decision.html"), "w").write(html)


if __name__ == "__main__":
    main()
