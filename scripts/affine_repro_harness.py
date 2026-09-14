#!/usr/bin/env python
"""
Affine reproducibility + speed harness (mbhard by default).

For each requested solver configuration the harness runs the affine N times, records the
transform parameters, symmetric DKT31 Dice, image similarity (fixed T1 vs affinely resampled
moving T1: lncc, ncc, mattes_mi, foreground NCC) and wall time, and reports

  * same-device reproducibility: max |Δparams| across repeats (0.0 = bitwise),
  * cross-device agreement (cpu vs mps/cuda) in parameters and Dice,
  * performance against the best ANTs C++ affine baseline stored in
    results/affine_baseline/mbhard_ants_affine_baseline.json (computed here if absent).

Acceptance rule used by tests/test_affine_reproducibility.py: same-device Δparams == 0,
Dice >= best ANTs affine Dice - 0.005, runtime < best ANTs affine runtime.

Usage:
  python scripts/affine_repro_harness.py --modes pytorch --devices mps cpu --repeats 2
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import ants
import syntx
from syntx.landmarks import preprocess_for_landmarks
from syntx.deformation_metrics import compute_bidirectional_dice
from syntx.image_compare import image_compare

BASE_DIR = "results/affine_baseline"
ANTS_JSON = os.path.join(BASE_DIR, "mbhard_ants_affine_baseline.json")


def load_pair(cache_dir=None):
    ds = syntx.benchmark_data("mbhard")
    fi, mi, fl, ml = ds["fixed"], ds["moving"], ds["fixed_label"], ds["moving_label"]
    assert fi.shape[0] > 100, "real Mindboggle data required"
    if cache_dir and os.path.exists(os.path.join(cache_dir, "fi_p.nii.gz")):
        fi_p, mi_p = ants.image_read(os.path.join(cache_dir, "fi_p.nii.gz")), ants.image_read(os.path.join(cache_dir, "mi_p.nii.gz"))
    else:
        fi_p, mi_p = preprocess_for_landmarks(fi), preprocess_for_landmarks(mi)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            ants.image_write(fi_p, os.path.join(cache_dir, "fi_p.nii.gz")); ants.image_write(mi_p, os.path.join(cache_dir, "mi_p.nii.gz"))
    return fi, mi, fl, ml, fi_p, mi_p


def score(fwd, fi, mi, fl, ml, fi_p, mi_p, mask):
    d = compute_bidirectional_dice(fl, ml, fi, mi, fwd, fwd, [True])
    w = ants.apply_transforms(fixed=fi_p, moving=mi_p, transformlist=fwd)
    sim = {m: float(image_compare(fi_p, w, m)) for m in ("lncc", "ncc", "mattes_mi")}
    sim["ncc_fg"] = float(np.corrcoef(fi_p.numpy()[mask], w.numpy()[mask])[0, 1])
    p = np.array(ants.read_transform(fwd[0]).parameters, float)
    return dict(dice_sym=float(d[2]), dice_fixed=float(d[0]), dice_moving=float(d[1]), params=p.tolist(), **sim)


def ants_baseline(fi, mi, fl, ml, fi_p, mi_p, mask, repeats=2, force=False):
    if os.path.exists(ANTS_JSON) and not force:
        return json.load(open(ANTS_JSON))
    os.makedirs(BASE_DIR, exist_ok=True)
    res = {}
    for name, kw in (("ants_Affine_default", dict(type_of_transform="Affine")),
                     ("ants_Affine_regular025", dict(type_of_transform="Affine", aff_random_sampling_rate=0.25, aff_sampling=32)),
                     ("ants_SyNQuick_a", dict(type_of_transform="antsRegistrationSyNQuick[a]")),
                     ("ants_SyN_a", dict(type_of_transform="antsRegistrationSyN[a]"))):
        for rep in range(repeats):
            t = time.time(); r = ants.registration(fixed=fi_p, moving=mi_p, random_seed=42, **kw); dt = time.time() - t
            res.setdefault(name, []).append(dict(rep=rep, time=dt, **score(r["fwdtransforms"], fi, mi, fl, ml, fi_p, mi_p, mask)))
            print(f"  {name} rep{rep}: {dt:.1f}s dice {res[name][-1]['dice_sym']:.4f}", flush=True)
    json.dump(res, open(ANTS_JSON, "w"), indent=1)
    return res


def best_ants(res):
    rows = [(k, np.mean([r["dice_sym"] for r in v]), np.mean([r["time"] for r in v])) for k, v in res.items()]
    return max(rows, key=lambda x: x[1])


def run_syntx(mode, device, repeats, fi, mi, fl, ml, fi_p, mi_p, mask, fresh_process=False, cache_dir=None):
    from syntx.robust_affine import robust_affine
    out = []
    for rep in range(repeats):
        if fresh_process:
            cmd = [sys.executable, "-c", (
                "import warnings,json,sys,ants,numpy as np;warnings.filterwarnings('ignore');"
                "from syntx.robust_affine import robust_affine;"
                f"fi=ants.image_read('{cache_dir}/fi_p.nii.gz');mi=ants.image_read('{cache_dir}/mi_p.nii.gz');"
                f"import time;t=time.time();r=robust_affine(fi,mi,mode='{mode}',device='{device}',seed=42);dt=time.time()-t;"
                "print('RESULT '+json.dumps(dict(fwd=r['fwdtransforms'],time=dt)))")]
            o = subprocess.run(cmd, capture_output=True, text=True)
            line = [l for l in o.stdout.splitlines() if l.startswith("RESULT ")][-1]
            rr = json.loads(line[7:]); fwd, dt = rr["fwd"], rr["time"]
        else:
            t = time.time(); r = robust_affine(fi_p, mi_p, mode=mode, device=device, seed=42); dt = time.time() - t
            fwd = r["fwdtransforms"]
        out.append(dict(rep=rep, time=dt, fresh_process=fresh_process, **score(fwd, fi, mi, fl, ml, fi_p, mi_p, mask)))
        print(f"  {mode}/{device}{'/fresh' if fresh_process else ''} rep{rep}: {dt:.1f}s dice {out[-1]['dice_sym']:.4f} p[:3] {np.round(out[-1]['params'][:3], 6)}", flush=True)
    return out


def summarize(runs):
    P = np.array([r["params"] for r in runs]); D = np.array([r["dice_sym"] for r in runs]); T = np.array([r["time"] for r in runs])
    return dict(n=len(runs), max_abs_param_spread=float(np.abs(P - P[0]).max()), dice_mean=float(D.mean()), dice_spread=float(D.max() - D.min()),
                time_mean=float(T.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["pytorch"])
    ap.add_argument("--devices", nargs="+", default=["mps", "cpu"])
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--fresh", action="store_true", help="also run each config once in a fresh interpreter")
    ap.add_argument("--cache-dir", default="results/affine_baseline/cache")
    ap.add_argument("--force-ants", action="store_true")
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "harness_latest.json"))
    args = ap.parse_args()
    fi, mi, fl, ml, fi_p, mi_p = load_pair(args.cache_dir)
    mask = fi_p.numpy() > 0.01
    print("ANTs C++ baseline", flush=True)
    ab = ants_baseline(fi, mi, fl, ml, fi_p, mi_p, mask, force=args.force_ants)
    bname, bdice, btime = best_ants(ab)
    print(f"best ANTs affine: {bname} dice {bdice:.4f} time {btime:.1f}s", flush=True)
    report = dict(ants_best=dict(name=bname, dice_sym=bdice, time=btime), ants=ab, syntx={})
    for mode in args.modes:
        for dev in args.devices:
            key = f"{mode}/{dev}"
            print(key, flush=True)
            runs = run_syntx(mode, dev, args.repeats, fi, mi, fl, ml, fi_p, mi_p, mask)
            if args.fresh:
                runs += run_syntx(mode, dev, 1, fi, mi, fl, ml, fi_p, mi_p, mask, fresh_process=True, cache_dir=args.cache_dir)
            s = summarize(runs)
            s["passes"] = dict(reproducible=bool(s["max_abs_param_spread"] == 0.0), dice_ok=bool(s["dice_mean"] >= bdice - 0.005), faster_than_ants=bool(s["time_mean"] < btime))
            report["syntx"][key] = dict(runs=runs, summary=s)
            print(f"  summary {key}: {s}", flush=True)
    devs = [k for k in report["syntx"] if k.endswith("/cpu")]
    for kc in devs:
        kg = kc.replace("/cpu", "/mps")
        if kg in report["syntx"]:
            pc = np.array(report["syntx"][kc]["runs"][0]["params"]); pg = np.array(report["syntx"][kg]["runs"][0]["params"])
            report["syntx"][kg]["summary"]["cpu_vs_gpu_max_abs_param_diff"] = float(np.abs(pc - pg).max())
            report["syntx"][kg]["summary"]["cpu_vs_gpu_dice_diff"] = float(report["syntx"][kg]["runs"][0]["dice_sym"] - report["syntx"][kc]["runs"][0]["dice_sym"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
