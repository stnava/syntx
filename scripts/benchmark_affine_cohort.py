#!/usr/bin/env python
"""
Affine cohort benchmark: syntx PyTorch affine vs ANTs C++ affine on N random Mindboggle pairs.

Per pair (seeded random draw from examples/pairs.csv):
  preprocessing  : NLM denoise (antstorch) + foreground 2-98 % normalisation (no N4), cached
  ANTs baseline  : ants.registration(type_of_transform='Affine', random_seed=42)
  syntx arms     : pt_default      robust_affine(mode='pytorch')  (run twice -> reproducibility)
                   landmark_affine least-squares affine on SIFT3D RANSAC inliers (match_sift3d_with_rotation_search)
                   pt_landmark     robust_affine(mode='pytorch', initial_transform=landmark_affine)
Metrics: symmetric DKT31 Dice (compute_bidirectional_dice), foreground NCC, whole-domain Mattes MI
of the affinely resampled moving T1 vs fixed T1, wall time, per-device parameter spread.

Success criterion ("equal or beat ANTs"): for the chosen syntx arm, mean Dice >= ANTs mean,
win-or-tie (Dice >= ANTs - 0.002) on >= 50 % of pairs, no pair worse than ANTs by > 0.01,
mean time < ANTs mean time, and bitwise reproducibility on every pair.

Usage: python scripts/benchmark_affine_cohort.py --n 10 --seed 7 [--device mps] [--no-landmarks]
"""
from __future__ import annotations

import argparse, json, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import ants, torch
import syntx
from syntx.benchmark.data import load_mindboggle_pair
from syntx.landmarks import preprocess_for_landmarks, match_sift3d_with_rotation_search
from syntx.robust_affine import robust_affine
from syntx.deformation_metrics import compute_bidirectional_dice
from syntx.core.losses import mattes_mi_loss_nd

OUT_DIR = "results/affine_baseline"
CACHE = os.path.join(OUT_DIR, "cache")


def preprocessed(pair_idx, p):
    os.makedirs(CACHE, exist_ok=True)
    fp, mp = os.path.join(CACHE, f"pair_{pair_idx:03d}_fi.nii.gz"), os.path.join(CACHE, f"pair_{pair_idx:03d}_mi.nii.gz")
    if os.path.exists(fp) and os.path.exists(mp):
        return ants.image_read(fp), ants.image_read(mp)
    fi_p, mi_p = preprocess_for_landmarks(p["fixed"]), preprocess_for_landmarks(p["moving"])
    ants.image_write(fi_p, fp); ants.image_write(mi_p, mp)
    return fi_p, mi_p


def score(fwd, p, fi_p, mi_p):
    d = compute_bidirectional_dice(p["fixed_label"], p["moving_label"], p["fixed"], p["moving"], fwd, fwd, [True])
    w = ants.apply_transforms(fixed=fi_p, moving=mi_p, transformlist=fwd)
    f, wv = fi_p.numpy(), w.numpy(); m = f > 0.01
    ncc = float(np.corrcoef(f[m], wv[m])[0, 1])
    ft = torch.from_numpy(f.transpose(2, 1, 0).copy())[None, None]; wt = torch.from_numpy(wv.transpose(2, 1, 0).copy())[None, None]
    mi = float(mattes_mi_loss_nd(wt, ft, mask=None, auto_mask=False, num_bins=64, fixed_range=(0.0, 1.0)).item())
    params = np.array(ants.read_transform(fwd[0]).parameters, float).tolist()
    return dict(dice_sym=float(d[2]), dice_fixed=float(d[0]), dice_moving=float(d[1]), ncc_fg=ncc, mattes_mi=mi, params=params)


def write_affine(M4, path):
    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3, matrix=np.asarray(M4[:3, :3], float),
                                    offset=tuple(float(x) for x in M4[:3, 3]))
    ants.write_transform(tx, path); return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="mps"); ap.add_argument("--no-landmarks", action="store_true")
    ap.add_argument("--pairs", type=int, nargs="*", default=None, help="explicit pair indices (overrides random draw)")
    ap.add_argument("--tag", default="cohort10")
    ap.add_argument("--pt-kwargs", default="{}", help="JSON of extra kwargs for robust_affine(mode='pytorch')")
    args = ap.parse_args()
    pt_kw = json.loads(args.pt_kwargs)
    rng = np.random.default_rng(args.seed)
    pairs = args.pairs if args.pairs else sorted(rng.choice(90, args.n, replace=False).tolist())
    print("pairs:", pairs, flush=True)
    rows = []
    for idx in pairs:
        p = load_mindboggle_pair(idx)
        ptype = p.get("type", "inter" if idx >= 42 else "intra")
        t = time.time(); fi_p, mi_p = preprocessed(idx, p); t_pre = time.time() - t
        row = dict(pair=idx, type=ptype, fixed=p.get("fixed_id", ""), moving=p.get("moving_id", ""), t_pre=t_pre, arms={})
        # ANTs baseline
        t = time.time(); ra = ants.registration(fixed=fi_p, moving=mi_p, type_of_transform="Affine", random_seed=42); dt = time.time() - t
        row["arms"]["ants"] = dict(time=dt, **score(ra["fwdtransforms"], p, fi_p, mi_p))
        # syntx default, twice
        runs = []
        for rep in range(2):
            t = time.time(); r = robust_affine(fi_p, mi_p, mode="pytorch", device=args.device, seed=42, **pt_kw); dt = time.time() - t
            runs.append(dict(time=dt, winner=r.get("init_candidate"), **score(r["fwdtransforms"], p, fi_p, mi_p)))
        spread = float(np.abs(np.array(runs[0]["params"]) - np.array(runs[1]["params"])).max())
        row["arms"]["pt_default"] = dict(runs[0], time=float(np.mean([x["time"] for x in runs])), param_spread=spread)
        if not args.no_landmarks:
            t = time.time()
            lm = match_sift3d_with_rotation_search(fi_p, mi_p, max_keypoints=2000, threshold=0.002, n_scales=10,
                                                   sigma_min=1.0, sigma_max=8.0, min_distance_mm=3.0, device=args.device)
            t_lm = time.time() - t
            inl = lm["inliers"]
            matB = write_affine(lm["affine"].astype(np.float64), os.path.join(CACHE, f"pair_{idx:03d}_landmark_affine.mat"))
            row["arms"]["landmark_affine"] = dict(time=t_lm, n_inliers=int(len(inl)), rotation_deg=float(lm["rotation_deg"]), **score([matB], p, fi_p, mi_p))
            t = time.time(); r = robust_affine(fi_p, mi_p, mode="pytorch", device=args.device, seed=42, initial_transform=matB, **pt_kw); dt = time.time() - t
            row["arms"]["pt_landmark"] = dict(time=dt + t_lm, winner=r.get("init_candidate"), **score(r["fwdtransforms"], p, fi_p, mi_p))
        rows.append(row)
        a = row["arms"]
        print(f"pair {idx:02d} {ptype:5s} | ANTs {a['ants']['dice_sym']:.4f} ({a['ants']['time']:.0f}s) | pt {a['pt_default']['dice_sym']:.4f} ({a['pt_default']['time']:.1f}s, spread {spread:.1e})"
              + (f" | lm-affine {a['landmark_affine']['dice_sym']:.4f} ({a['landmark_affine']['n_inliers']} inl) | pt+lm {a['pt_landmark']['dice_sym']:.4f}" if "pt_landmark" in a else ""), flush=True)
        os.makedirs(OUT_DIR, exist_ok=True)
        json.dump(dict(pairs=pairs, seed=args.seed, pt_kwargs=pt_kw, rows=rows), open(os.path.join(OUT_DIR, f"{args.tag}_affine.json"), "w"), indent=1)

    # summary --------------------------------------------------------------------
    arms = [k for k in rows[0]["arms"]]
    summary = {}
    ants_d = np.array([r["arms"]["ants"]["dice_sym"] for r in rows])
    for arm in arms:
        d = np.array([r["arms"][arm]["dice_sym"] for r in rows]); tt = np.array([r["arms"][arm]["time"] for r in rows])
        delta = d - ants_d
        summary[arm] = dict(dice_mean=float(d.mean()), dice_std=float(d.std()), time_mean=float(tt.mean()),
                            ncc_fg_mean=float(np.mean([r["arms"][arm]["ncc_fg"] for r in rows])),
                            mi_mean=float(np.mean([r["arms"][arm]["mattes_mi"] for r in rows])),
                            delta_vs_ants_mean=float(delta.mean()), wins_or_ties=int((delta >= -0.002).sum()), worst_delta=float(delta.min()),
                            reproducible=bool(all(r["arms"][arm].get("param_spread", 0.0) == 0.0 for r in rows)))
        s = summary[arm]
        s["beats_ants"] = bool(arm != "ants" and s["dice_mean"] >= summary["ants"]["dice_mean"] and s["wins_or_ties"] >= len(rows) / 2
                               and s["worst_delta"] >= -0.01 and s["time_mean"] < summary["ants"]["time_mean"] and s["reproducible"])
        print(f"{arm:16s} dice {s['dice_mean']:.4f}±{s['dice_std']:.4f}  Δ {s['delta_vs_ants_mean']:+.4f}  wins/ties {s['wins_or_ties']}/{len(rows)}  worst {s['worst_delta']:+.4f}  time {s['time_mean']:.1f}s  ncc {s['ncc_fg_mean']:.4f}  repro {s['reproducible']}  beats_ants {s['beats_ants']}", flush=True)
    json.dump(dict(pairs=pairs, seed=args.seed, pt_kwargs=pt_kw, rows=rows, summary=summary), open(os.path.join(OUT_DIR, f"{args.tag}_affine.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT_DIR, f"{args.tag}_affine.json"))


if __name__ == "__main__":
    main()
