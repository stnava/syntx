"""Real-data gate of the ``joint`` motion engine on the curated ``motion_cohort`` BIDS archive.

For every series: ``unified`` (the previous default) against ``joint`` (one optimisation over the whole timeline,
docs/MOTION_JOINT_OPTIMIZATION_PLAN.md). No ground truth exists on real data, so the gate reads physical signatures:

- ``fd_mean``: the reported head motion. Lower is not better by itself.
- step MAD per degree of freedom in the acquisition frame (phase encode, slice, readout translation; pitch): acquisition
  effects live in PE / slice, real motion also in readout. The readout step is the B0-immune floor and MUST NOT change
  by more than ``READOUT_TOL`` between engines (the prior must not shrink real jitter).
- ``design_locked``: the robust period-2 amplitude of the pose along the slice and PE axes (ASL tag/control, DWI shell
  alternation): what the design should absorb; lower is better.
- ``tsd``: median temporal SD of the corrected series inside the foreground (image alignment; lower is better).
- time.

Usage: ``python benchmarks/motion_cohort_gate.py <motion_cohort dir> <out dir> [--lams 0.02,0.05] [--only asl,dwi,bold]``
Writes ``gate.json`` and ``motion_cohort_gate.html``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

READOUT_TOL = 0.15      # relative change of the readout translation step allowed between engines ...
READOUT_TOL_MM = 0.02   # ... or this absolute change (the step is a floor quantity, 0.02-0.1 mm, a MAD over 16-100 steps)


def _series(root: str, only: set):
    out = []
    if "asl" in only:
        for f in sorted(glob.glob(root + "/*/sub-*/ses-*/asl/*_asl.nii.gz")):
            vt = [l.strip() for l in open(f.replace("_asl.nii.gz", "_aslcontext.tsv")).read().splitlines()[1:]]
            out.append(("asl", f, {"volume_types": vt}))
    if "dwi" in only:
        for f in sorted(glob.glob(root + "/*/sub-*/ses-*/dwi/*_dwi.nii.gz")):
            if "dir-PA" in f:
                continue
            bv = np.loadtxt(f.replace(".nii.gz", ".bval"))
            g = np.loadtxt(f.replace(".nii.gz", ".bvec")).T
            out.append(("dwi", f, {"bvals": bv, "bvecs": g}))
    if "bold" in only:
        for f in sorted(glob.glob(root + "/*/sub-*/ses-*/func/*_bold.nii.gz")):
            out.append(("bold", f, {}))
    return out


def _pe_axis(path: str, img) -> np.ndarray:
    sidecar = path.replace(".nii.gz", ".json")
    pe = "j"
    if os.path.exists(sidecar):
        pe = json.load(open(sidecar)).get("PhaseEncodingDirection", "j")
    ax = {"i": 0, "j": 1, "k": 2}[pe[0]]
    v = np.asarray(img.direction)[:3, ax]
    return v / np.linalg.norm(v)


def _signatures(r, img, levels):
    from scipy.spatial.transform import Rotation as Rot

    t = np.asarray(r["translations_mm"], float)
    rot = np.radians(np.asarray(r["rotations_deg"], float))
    rot_mm = 50.0 * Rot.from_euler("xyz", np.degrees(rot), degrees=True).as_rotvec()
    D = np.asarray(img.direction)[:3, :3]
    proj = {"t_pe": t @ D[:, 1], "t_sl": t @ D[:, 2], "t_ro": t @ D[:, 0], "pitch": rot_mm @ D[:, 0]}
    step = {k: float(1.4826 * np.median(np.abs(np.diff(v) - np.median(np.diff(v))))) for k, v in proj.items()}
    alt = np.where(np.arange(len(t)) % 2 == 0, 1.0, -1.0)
    locked = {}
    for k in ("t_pe", "t_sl"):
        v = proj[k]
        u = v[1:-1] - 0.5 * (v[:-2] + v[2:])
        if levels is not None and len(np.unique(levels)) > 1:
            s = np.where(levels[1:-1] == levels[0], 1.0, -1.0)   # relative to the anchor level
        else:
            s = alt[1:-1]
        locked[k] = float(abs(np.median(u * s)))
    arr = r["motion_corrected"].numpy()
    fg = np.mean(arr, axis=-1) > 0.1 * np.percentile(np.mean(arr, axis=-1), 99)
    tsd = float(np.median(np.std(arr[fg], axis=-1) / (np.mean(arr[fg], axis=-1) + 1e-6)))
    return {"fd_mean": float(np.mean(r["fd"])), "fd_max": float(np.max(r["fd"])), "step_mad_mm": step,
            "design_locked_mm": locked, "tsd_rel": tsd,
            "frame_converged_frac": float(np.mean(np.asarray(r.get("frame_converged", [True]), bool)))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--lams", default="0.02")
    ap.add_argument("--only", default="asl,dwi,bold")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--report-only", action="store_true", help="rebuild the HTML from <out>/gate.json")
    args = ap.parse_args(argv)
    if args.report_only:
        results = json.load(open(os.path.join(args.out, "gate.json")))
        _report(results, os.path.join(args.out, "motion_cohort_gate.html"))
        print("DONE", os.path.join(args.out, "motion_cohort_gate.html"))
        return
    import ants
    import syntx

    os.makedirs(args.out, exist_ok=True)
    lams = [float(x) for x in args.lams.split(",") if x]
    results = {}
    for kind, path, meta in _series(args.root, set(args.only.split(","))):
        name = kind + ":" + "/".join(path.split("/")[-5:-2]) + ("_AP" if "dir-AP" in path else "")
        img = ants.image_read(path)
        pe = _pe_axis(path, img)
        if kind == "asl":
            design = syntx.make_design("tag_control", volume_types=meta["volume_types"])
        elif kind == "dwi":
            design = syntx.make_design("shells", bvals=meta["bvals"], bvecs=meta["bvecs"], pe_axis=pe)
        else:
            design = syntx.make_design("none", n_frames=img.shape[-1])
        levels = np.asarray(design.levels)
        variants = [("unified", {})] + [("joint", {"surround_weight": lam}) for lam in lams]
        if kind == "dwi":
            variants.append(("joint", {"surround_weight": lams[0], "level_weights": True}))
        rec = {}
        for eng, extra in variants:
            label = eng if not extra else eng + ":" + ",".join(f"{k}={v}" for k, v in extra.items())
            t0 = time.time()
            r = syntx.motion_correct_design(img, design, engine=eng, device=args.device, verbose=False, **extra)
            sig = _signatures(r, img, levels)
            sig["seconds"] = time.time() - t0
            rec[label] = sig
            print(name, label, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in sig.items() if k in ("fd_mean", "seconds", "tsd_rel")}), flush=True)
        results[name] = rec
        json.dump(results, open(os.path.join(args.out, "gate.json"), "w"), indent=1)
    _report(results, os.path.join(args.out, "motion_cohort_gate.html"))
    print("DONE", os.path.join(args.out, "motion_cohort_gate.html"))


def _report(results: dict, path: str):
    rows = []
    verdicts = []
    for name, rec in results.items():
        base = rec.get("unified")
        for label, sig in rec.items():
            st = sig["step_mad_mm"]; lk = sig["design_locked_mm"]
            ro_change = None if base is None or label == "unified" else (st["t_ro"] - base["step_mad_mm"]["t_ro"]) / max(base["step_mad_mm"]["t_ro"], 1e-6)
            ro_abs = None if ro_change is None else abs(st["t_ro"] - base["step_mad_mm"]["t_ro"])
            ok = None if ro_change is None else (abs(ro_change) <= READOUT_TOL or ro_abs <= READOUT_TOL_MM)
            flag = "" if ro_change is None else ("ok" if ok else "readout step changed %+.0f%% (%.3f mm)" % (100 * ro_change, ro_abs))
            if ro_change is not None:
                verdicts.append((name, label, ok))
            rows.append(f"<tr><td>{name}</td><td>{label}</td><td>{sig['fd_mean']:.3f}</td><td>{sig['fd_max']:.2f}</td>"
                        f"<td>{st['t_pe']:.3f}</td><td>{st['t_sl']:.3f}</td><td>{st['t_ro']:.3f}</td><td>{st['pitch']:.3f}</td>"
                        f"<td>{lk['t_pe']:.3f}</td><td>{lk['t_sl']:.3f}</td><td>{sig['tsd_rel']:.4f}</td>"
                        f"<td>{sig['frame_converged_frac']:.2f}</td><td>{sig['seconds']:.0f}</td><td>{flag}</td></tr>")
    n_ok = sum(1 for _, _, ok in verdicts if ok)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>motion_cohort gate</title>
<style>body{{font-family:sans-serif;margin:2em}} table{{border-collapse:collapse}} td,th{{border:1px solid #999;padding:3px 6px;font-size:13px}} th{{background:#eee}}</style></head>
<body><h1>Joint motion engine: real-data gate on motion_cohort</h1>
<p>Each series registered by the previous default (<code>unified</code>) and by <code>joint</code> (one optimisation over the
whole acquisition timeline; head prior with a 0.05 mm Charbonnier knee at the given weight). Step MAD: robust frame-to-frame
step per degree of freedom in the acquisition frame (mm; pitch at 50 mm). Readout translation is the B0-immune floor: the
gate requires it to change by at most {int(READOUT_TOL * 100)} % or {READOUT_TOL_MM} mm (real jitter is not shrunk). On DWI the
readout step of <code>unified</code> also carries the cross-contrast link bias at every b0 switch (all DOFs), so a lower
readout step there is not shrinkage by itself; the ASL rows are the clean test. design_locked: robust amplitude
of the pose alternation locked to the design (what the acquisition terms should absorb). tsd_rel: median relative temporal
SD of the corrected series in the foreground (alignment).</p>
<p>Readout-floor gate: {n_ok} of {len(verdicts)} engine variants within tolerance.</p>
<table><tr><th>series</th><th>engine</th><th>FD mean</th><th>FD max</th><th>step t_pe</th><th>step t_sl</th><th>step t_ro</th>
<th>step pitch</th><th>locked t_pe</th><th>locked t_sl</th><th>tsd_rel</th><th>converged</th><th>s</th><th>gate</th></tr>
{''.join(rows)}</table></body></html>"""
    open(path, "w").write(html)


if __name__ == "__main__":
    main()
