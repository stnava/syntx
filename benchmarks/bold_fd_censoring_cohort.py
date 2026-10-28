"""Cohort gate C5 (first half): raw vs band-limited FD censoring on every local resting BOLD series.

For each ``*_bold.nii.gz`` under the BIDS root: ``syntx.motion_correction(preset="bold")`` (the released engine), then the
fraction of frames above the scrubbing threshold on the raw FD and on ``fd_bandlimited`` (``syntx.band_limited_fd``), the band
found, the alternation amplitude and the DVARS to FD coupling. Writes ``fd_censoring.json`` incrementally and an HTML table.

Usage: python benchmarks/bold_fd_censoring_cohort.py <bids root> <out dir> [--cohorts breacher,SOCOM,ExpArt] [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

THRESHOLDS = (0.3, 0.5)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--cohorts", default="breacher,SOCOM,ExpArt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    import ants
    import syntx
    from antsxqc.metrics import fd_dvars_coupling

    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, "fd_censoring.json")
    results = json.load(open(out_json)) if os.path.exists(out_json) else {}
    files = []
    for c in args.cohorts.split(","):
        files += sorted(glob.glob(os.path.join(args.root, c, "sub-*", "ses-*", "func", "*_bold.nii.gz")))
    if args.limit:
        files = files[: args.limit]
    for f in files:
        name = "/".join(f.split(os.sep)[-5:-2])
        if name in results:
            continue
        img = ants.image_read(f)
        sidecar = f.replace(".nii.gz", ".json")
        tr = float(json.load(open(sidecar)).get("RepetitionTime", img.spacing[3])) if os.path.exists(sidecar) else float(img.spacing[3])
        t0 = time.time()
        r = syntx.motion_correction(img, reference="mean", type_of_transform="Rigid", preset="bold", device=args.device, verbose=False, tr=tr)
        fd = np.asarray(r["fd"], float)
        bl = np.asarray(r["fd_bandlimited"], float)
        rec = {"T": int(len(fd)), "tr": tr, "fd_mean": float(fd[1:].mean()), "fd_bandlimited_mean": float(bl[1:].mean()),
               "band": r["fd_band"], "dvars_fd_corr": float(fd_dvars_coupling(fd, np.asarray(r["dvars"], float))),
               "seconds": time.time() - t0}
        for th in THRESHOLDS:
            rec[f"frac_gt_{th}_raw"] = float(np.mean(fd > th))
            rec[f"frac_gt_{th}_bandlimited"] = float(np.mean(bl > th))
        results[name] = rec
        json.dump(results, open(out_json, "w"), indent=1)
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items() if k != "band"}), flush=True)
    _report(results, os.path.join(args.out, "bold_fd_censoring.html"))
    print("DONE", os.path.join(args.out, "bold_fd_censoring.html"))


def _report(results, path):
    rows = []
    for name, r in sorted(results.items()):
        b = r["band"] or {}
        band = "none" if not b.get("band_hz") else "%.2f-%.2f" % tuple(b["band_hz"])
        rows.append(f"<tr><td>{name}</td><td>{r['T']}</td><td>{r['tr']:.2f}</td><td>{r['fd_mean']:.3f}</td><td>{r['fd_bandlimited_mean']:.3f}</td>"
                    f"<td>{band}</td><td>{b.get('band_source', '-')}</td><td>{float(b.get('alternation_amplitude_mm') or 0):.3f}</td>"
                    f"<td>{100 * r['frac_gt_0.5_raw']:.0f}</td><td>{100 * r['frac_gt_0.5_bandlimited']:.0f}</td>"
                    f"<td>{100 * r['frac_gt_0.3_raw']:.0f}</td><td>{100 * r['frac_gt_0.3_bandlimited']:.0f}</td><td>{r['dvars_fd_corr']:.2f}</td></tr>")
    vals = np.array([[r["frac_gt_0.5_raw"], r["frac_gt_0.5_bandlimited"], r["frac_gt_0.3_raw"], r["frac_gt_0.3_bandlimited"]] for r in results.values()]) if results else np.zeros((0, 4))
    med = 100 * np.median(vals, axis=0) if len(vals) else [float("nan")] * 4
    n_band = sum(1 for r in results.values() if (r["band"] or {}).get("band_source") == "auto")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>BOLD FD censoring cohort</title>
<style>body{{font-family:sans-serif;margin:2em}} table{{border-collapse:collapse}} td,th{{border:1px solid #999;padding:3px 6px;font-size:12px}} th{{background:#eee}}</style></head>
<body><h1>Raw vs band-limited FD censoring on local resting BOLD ({len(results)} series)</h1>
<p>Released engine (<code>syntx.motion_correction(preset="bold")</code>). Band-limited FD removes the near-Nyquist physiological
alternation from the pose series (<code>syntx.band_limited_fd</code>); it was applied on {n_band} of {len(results)} series (the others
had no prominent band). Percent of frames above the threshold, raw vs band-limited; medians over series:
0.5 mm {med[0]:.0f} % vs {med[1]:.0f} %, 0.3 mm {med[2]:.0f} % vs {med[3]:.0f} %.</p>
<table><tr><th>series</th><th>T</th><th>TR</th><th>FD mean</th><th>FD band-limited</th><th>band Hz</th><th>source</th><th>alt. amp mm</th>
<th>&gt;0.5 raw %</th><th>&gt;0.5 bl %</th><th>&gt;0.3 raw %</th><th>&gt;0.3 bl %</th><th>DVARS~FD</th></tr>{''.join(rows)}</table></body></html>"""
    open(path, "w").write(html)


if __name__ == "__main__":
    main()
