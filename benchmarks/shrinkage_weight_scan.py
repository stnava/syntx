"""Decision 2026-10-28: shrinkage toward identity on at a small weight, set on the cohort by the readout-floor rule.
Joint engine with the per-frame gain, pose_shrinkage in likelihood units scanned on the three series; per weight: readout /
PE / slice step MAD, FD, alignment, convergence. Rule: the largest weight whose readout step is within 0.02 mm of the
weight-0 run on every series (shrinkage must not remove real jitter) and whose alignment is not worse.
Usage: python benchmarks/shrinkage_weight_scan.py <motion_cohort dir> <out dir> [--weights 0,100,1000,10000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from motion_cohort_gate import _pe_axis, _signatures  # noqa: E402

CASES = {
    "bold": ("breacher/sub-3315/ses-initial-day2/func/sub-3315_ses-initial-day2_task-rest_run-01_bold.nii.gz", "none"),
    "dwi031": ("FPA/sub-BLAST031/ses-01/dwi/sub-BLAST031_ses-01_run-001_dir-AP_dwi.nii.gz", "shells"),
    "dwi11": ("/Users/stnava/data/blast_cohorts/BIDS/SOCOM/sub-Blast-11/ses-01/dwi/sub-Blast-11_ses-01_run-001_dir-PA_dwi.nii.gz", "shells"),
}
READOUT_TOL_MM = 0.02


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--weights", default="0,100,1000,10000"); ap.add_argument("--device", default="auto")
    ap.add_argument("--only", default=",".join(CASES))
    args = ap.parse_args(argv)
    import ants
    import syntx

    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, "shrinkage_scan.json")
    results = json.load(open(out_json)) if os.path.exists(out_json) else {}
    weights = [float(w) for w in args.weights.split(",")]
    for key, (rel, kind) in CASES.items():
        if key not in args.only.split(","):
            continue
        path = rel if rel.startswith("/") else os.path.join(args.root, rel)
        img = ants.image_read(path); pe = _pe_axis(path, img)
        sc = json.load(open(path.replace(".nii.gz", ".json"))) if os.path.exists(path.replace(".nii.gz", ".json")) else {}
        tr = float(sc.get("RepetitionTime", img.spacing[3]))
        if kind == "shells":
            bv = np.loadtxt(path.replace(".nii.gz", ".bval")); g = np.loadtxt(path.replace(".nii.gz", ".bvec")).T
            design = syntx.make_design("shells", bvals=bv, bvecs=g, pe_axis=pe)
        else:
            design = syntx.make_design("none", n_frames=img.shape[-1])
        levels = np.asarray(design.levels)
        results.setdefault(key, {})
        for w in weights:
            wl = f"{w:g}"
            if wl in results[key]:
                continue
            t0 = time.time()
            r = syntx.motion_correct_design(img, design, engine="joint", device=args.device, verbose=False, tr=tr, pose_shrinkage=w,
                                            **({"surround_scale": "likelihood"} if w > 0 else {}))
            sig = _signatures(r, img, levels); sig["seconds"] = time.time() - t0
            sig["gain_range"] = None if r.get("frame_gain") is None else [float(min(r["frame_gain"])), float(max(r["frame_gain"]))]
            results[key][wl] = sig
            json.dump(results, open(out_json, "w"), indent=1, default=float)
            print(key, "mu", wl, json.dumps({"fd": round(sig["fd_mean"], 3), "ro": round(sig["step_mad_mm"]["t_ro"], 3), "pe": round(sig["step_mad_mm"]["t_pe"], 3),
                                             "sl": round(sig["step_mad_mm"]["t_sl"], 3), "tsd": round(sig["tsd_rel"], 4), "conv": sig["frame_converged_frac"], "gain": sig["gain_range"], "s": round(sig["seconds"])}), flush=True)
    # the rule
    ok_by_w = {}
    for w in weights:
        wl = f"{w:g}"; ok = True
        for key in [k for k in CASES if k in args.only.split(",")]:
            base = results[key]["0"]; cur = results[key].get(wl)
            if cur is None:
                ok = False; break
            if abs(cur["step_mad_mm"]["t_ro"] - base["step_mad_mm"]["t_ro"]) > READOUT_TOL_MM or cur["tsd_rel"] > base["tsd_rel"] + 1e-4:
                ok = False
        ok_by_w[wl] = ok
    chosen = max([w for w in weights if ok_by_w[f"{w:g}"]], default=0.0)
    results["_rule"] = {"readout_tol_mm": READOUT_TOL_MM, "ok_by_weight": ok_by_w, "chosen_weight": chosen}
    json.dump(results, open(out_json, "w"), indent=1, default=float)
    print("RULE", json.dumps(results["_rule"]))
    print("DONE", out_json)


if __name__ == "__main__":
    main()
