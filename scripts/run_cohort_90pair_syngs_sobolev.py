#!/usr/bin/env python3
"""
Full 90-Pair Mindboggle Cohort Benchmark:
SyNGS (Geodesic Shooting with Balanced Sobolev α=0.35, CFL=0.25, RegAdam) vs. ANTs C++ SyN Baseline
===================================================================================================
Evaluates the complete 90-pair Mindboggle cohort (40 intra-study, 50 inter-study) on full volumes.
"""

import time
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import ants
import syntx
from syntx.benchmark.data import load_mindboggle_pair
from syntx.deformation_metrics import (
    compute_bidirectional_dice,
    compute_harmonic_energy,
    compute_bending_energy
)

def normalize_intensity(img: ants.ANTsImage) -> ants.ANTsImage:
    arr = img.numpy()
    pos = arr[arr > 0]
    if len(pos) > 0:
        p02 = float(np.percentile(pos, 2.0))
        p98 = float(np.percentile(pos, 98.0))
        if p98 <= p02 + 1e-4:
            p02 = 0.0
            p98 = float(pos.max())
    else:
        p02 = float(arr.min())
        p98 = float(arr.max())
    norm_arr = np.clip((arr - p02) / (p98 - p02 + 1e-6), 0.0, 1.0).astype(np.float32)
    return img.new_image_like(norm_arr)

def compute_inverse_identity_error(fi: ants.ANTsImage, fwd_warp_path: str, inv_warp_path: str) -> dict:
    """Computes real physical inverse identity composition error in mm."""
    try:
        fwd_img = ants.image_read(fwd_warp_path)
        inv_img = ants.image_read(inv_warp_path)
        fwd_channels = ants.split_channels(fwd_img)
        inv_channels = ants.split_channels(inv_img)
        composed_channels = [
            ants.apply_transforms(fixed=fi, moving=ch, transformlist=[inv_warp_path])
            for ch in fwd_channels
        ]
        err_sq = sum((c.numpy() + i.numpy()) ** 2 for c, i in zip(composed_channels, inv_channels))
        err_mag = np.sqrt(err_sq)
        return {
            "mean": float(np.mean(err_mag)),
            "p95": float(np.percentile(err_mag, 95.0)),
            "max": float(np.max(err_mag))
        }
    except Exception:
        return {"mean": float("nan"), "p95": float("nan"), "max": float("nan")}

def run_cohort_benchmark(
    alpha=0.35,
    cfl=0.25,
    flow_sigma=3.0,
    total_sigma=0.0,
    reg_iterations=(100, 100, 20),
    bootstrap_mode='antithetic',
    bootstrap_orig_weight=0.50,
    bootstrap_jitter_scale=0.25,
    output_csv="results/cohort_90pair_syngs_sobolev_summary.csv",
    output_json="results/cohort_90pair_syngs_sobolev_summary.json",
    force=False
):
    os.makedirs("results", exist_ok=True)
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

    df_pairs = pd.read_csv("examples/pairs.csv")
    total_pairs = len(df_pairs)
    print(f"====================================================================================================")
    print(f" STARTING COMPLETE 90-PAIR MINDBOGGLE BENCHMARK: SyNGS BALANCED SOBOLEV (α={alpha}, CFL={cfl}, Device: {device})")
    print(f" Total Cohort Size: {total_pairs} pairs (40 intra-study, 50 inter-study)")
    print(f"====================================================================================================\n", flush=True)

    results = []
    completed_pairs = set()

    if not force and os.path.exists(output_csv):
        try:
            df_existing = pd.read_csv(output_csv)
            completed_pairs = set(df_existing['pair'].tolist())
            results = df_existing.to_dict('records')
            print(f"Resuming benchmark: Found {len(completed_pairs)} already completed pairs in {output_csv}\n", flush=True)
        except Exception:
            completed_pairs = set()

    t0_all = time.time()

    for idx in range(total_pairs):
        if idx in completed_pairs and not force:
            continue

        row = df_pairs.iloc[idx]
        ptype = row['type']
        sub1, sub2 = row['subject1'], row['subject2']

        print(f"[{idx+1:02d}/{total_pairs}] Processing Pair {idx:02d} ({ptype.upper()}: {sub1} -> {sub2})...", flush=True)
        t_pair_start = time.time()

        p = load_mindboggle_pair(idx, "examples/pairs.csv")
        fi = normalize_intensity(p['fixed'])
        mi = normalize_intensity(p['moving'])
        fl, ml = p['fixed_label'], p['moving_label']
        fl_arr = fl.numpy()
        brain_mask = fl_arr > 0

        # Deterministic Robust Affine
        t_aff_0 = time.time()
        reg_aff = syntx.robust_affine(fi, mi, mode="auto", verbose=False)
        aff_0 = reg_aff["fwdtransforms"][0]
        aff_inv = reg_aff["invtransforms"][0]
        t_aff = time.time() - t_aff_0

        # Initial Affine DICE
        _, _, dice_aff = compute_bidirectional_dice(fl, ml, fi, mi, [aff_0], [aff_inv])

        # 1. ANTs C++ SyN Baseline (from cache or compute)
        ants_baseline_file = os.path.join("results", f"pair_{idx:03d}_ants_syn.json")
        if os.path.exists(ants_baseline_file):
            try:
                with open(ants_baseline_file, "r") as f:
                    ants_rec = json.load(f)
                dice_ants = float(ants_rec.get("dice_sym", float("nan")))
                dfix_ants = float(ants_rec.get("dice_fixed", float("nan")))
                dmov_ants = float(ants_rec.get("dice_moving", float("nan")))
                fold_whole_ants = float(ants_rec.get("folding_pct", 0.0))
                fold_brain_ants = float(ants_rec.get("folding_pct", 0.0))
                min_jac_whole_ants = float(ants_rec.get("min_jacobian", 0.0))
                min_jac_brain_ants = float(ants_rec.get("min_jacobian", 0.0))
                harm_ants = float(ants_rec.get("harmonic_energy", float("nan")))
                bend_ants = float(ants_rec.get("bending_energy", float("nan")))
                t_ants = float(ants_rec.get("runtime_seconds", 120.0))
            except Exception:
                ants_rec = None
        else:
            ants_rec = None

        if ants_rec is None or not np.isfinite(dice_ants):
            t0 = time.time()
            res_ants = ants.registration(
                fixed=fi, moving=mi, type_of_transform="SyN",
                initial_transform=aff_0,
                syn_metric="CC", syn_sampling=2,
                reg_iterations=(100, 100, 20),
                flow_sigma=3.0, total_sigma=0.0, grad_step=0.25, verbose=False
            )
            t_ants = time.time() - t0

            dfix_ants, dmov_ants, dice_ants = compute_bidirectional_dice(
                fl, ml, fi, mi, res_ants["fwdtransforms"], res_ants["invtransforms"]
            )
            warp_ants = res_ants["fwdtransforms"][0]
            jac_ants = ants.create_jacobian_determinant_image(fi, warp_ants, do_log=False).numpy()
            fold_whole_ants = float(np.mean(jac_ants <= 0.0) * 100.0)
            fold_brain_ants = float(np.mean(jac_ants[brain_mask] <= 0.0) * 100.0)
            min_jac_whole_ants = float(np.min(jac_ants))
            min_jac_brain_ants = float(np.min(jac_ants[brain_mask]))
            harm_ants = compute_harmonic_energy(warp_ants, fi.spacing)
            bend_ants = compute_bending_energy(warp_ants, fi.spacing)

        # 2. SyNGS with Balanced Sobolev
        t0 = time.time()
        res_syngs = syntx.syngs(
            fixed=fi, moving=mi, initial_transform=aff_0,
            backend='pytorch', device=device,
            flow_sigma=flow_sigma, total_sigma=total_sigma,
            alpha=alpha, regularizer='sobolev',
            transport_mode='transport',
            optimizer='reg_adam', optimizer_lr=1.2,
            max_step_norm=cfl,
            reg_iterations=list(reg_iterations),
            similarity_metric='cc2',
            bootstrap_mode=bootstrap_mode,
            bootstrap_orig_weight=bootstrap_orig_weight,
            bootstrap_jitter_scale=bootstrap_jitter_scale,
            n_steps=8,
            solver='euler',
            verbose=False
        )
        t_syngs = time.time() - t0

        which_inv = res_syngs.get("whichtoinvert_inv", [True, False])
        dfix_syngs, dmov_syngs, dice_syngs = compute_bidirectional_dice(
            fl, ml, fi, mi, res_syngs["fwdtransforms"], res_syngs["invtransforms"], which_inv
        )
        warp_syngs = res_syngs["fwdtransforms"][0]
        inv_warp_syngs = res_syngs["invtransforms"][-1]
        jac_syngs = ants.create_jacobian_determinant_image(fi, warp_syngs, do_log=False).numpy()
        fold_whole_syngs = float(np.mean(jac_syngs <= 0.0) * 100.0)
        fold_brain_syngs = float(np.mean(jac_syngs[brain_mask] <= 0.0) * 100.0)
        min_jac_whole_syngs = float(np.min(jac_syngs))
        min_jac_brain_syngs = float(np.min(jac_syngs[brain_mask]))
        harm_syngs = compute_harmonic_energy(warp_syngs, fi.spacing)
        bend_syngs = compute_bending_energy(warp_syngs, fi.spacing)

        inv_err = compute_inverse_identity_error(fi, warp_syngs, inv_warp_syngs)

        gain_pct = (dice_syngs - dice_ants) * 100.0
        speedup = t_ants / max(t_syngs, 1e-3)
        win = 1 if dice_syngs > dice_ants + 1e-4 else (0 if dice_syngs < dice_ants - 1e-4 else 0.5)
        win_str = "WIN" if win == 1 else ("LOSS" if win == 0 else "TIE")

        rec = {
            "pair": idx,
            "type": ptype,
            "subject1": sub1,
            "subject2": sub2,
            "model": "syngs_sobolev",
            "alpha": alpha,
            "cfl": cfl,
            "boot_mode": bootstrap_mode,
            "w_orig": bootstrap_orig_weight,
            "dice_affine": dice_aff,
            "dice_ants": dice_ants,
            "dice_ants_fixed": dfix_ants,
            "dice_ants_moving": dmov_ants,
            "dice_syngs": dice_syngs,
            "dice_syngs_fixed": dfix_syngs,
            "dice_syngs_moving": dmov_syngs,
            "dice_gain_pct": gain_pct,
            "win": win_str,
            "fold_brain_ants_pct": fold_brain_ants,
            "fold_brain_syngs_pct": fold_brain_syngs,
            "fold_whole_ants_pct": fold_whole_ants,
            "fold_whole_syngs_pct": fold_whole_syngs,
            "min_jac_brain_ants": min_jac_brain_ants,
            "min_jac_brain_syngs": min_jac_brain_syngs,
            "harm_ants": harm_ants,
            "harm_syngs": harm_syngs,
            "bend_ants": bend_ants,
            "bend_syngs": bend_syngs,
            "inv_err_mean_mm": inv_err["mean"],
            "inv_err_p95_mm": inv_err["p95"],
            "inv_err_max_mm": inv_err["max"],
            "time_affine_s": t_aff,
            "time_ants_s": t_ants,
            "time_syngs_s": t_syngs,
            "speedup": speedup,
            "time_total_s": time.time() - t_pair_start
        }
        results.append(rec)

        print(f"  * Pair {idx:02d} [SyNGS]: DICE = {dice_syngs:.4f} (Fix: {dfix_syngs:.4f}, Mov: {dmov_syngs:.4f}) vs ANTs = {dice_ants:.4f} ({gain_pct:+5.2f}% [{win_str}]) | Folds: {fold_brain_syngs:.5f}% | Time: {t_syngs:.1f}s ({speedup:.2f}x speedup)", flush=True)

        # Incrementally save CSV and JSON
        df_out = pd.DataFrame(results)
        df_out.to_csv(output_csv, index=False)
        with open(output_json, "w") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_completed": len(results),
                "total_planned": total_pairs,
                "model": "syngs_sobolev",
                "alpha": alpha,
                "cfl": cfl,
                "results": results
            }, f, indent=2)

        # Cleanup temporary files and GPU memory cache
        for tf in res_syngs.get("fwdtransforms", []) + res_syngs.get("invtransforms", []):
            if isinstance(tf, str) and os.path.exists(tf) and tf.startswith(("/tmp", "/var/folders")):
                try:
                    os.remove(tf)
                except Exception:
                    pass
        if os.path.exists(aff_0) and aff_0.startswith(("/tmp", "/var/folders")):
            try:
                os.remove(aff_0)
            except Exception:
                pass
        if os.path.exists(aff_inv) and aff_inv.startswith(("/tmp", "/var/folders")):
            try:
                os.remove(aff_inv)
            except Exception:
                pass

        if device == 'mps':
            torch.mps.empty_cache()
        elif device == 'cuda':
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    total_time_min = (time.time() - t0_all) / 60.0
    df_res = pd.DataFrame(results)
    
    # Statistical Metrology
    dice_syngs_vals = df_res['dice_syngs'].to_numpy()
    dice_ants_vals = df_res['dice_ants'].to_numpy()
    gains = df_res['dice_gain_pct'].to_numpy()
    n_wins = int((df_res['win'] == 'WIN').sum())
    n_losses = int((df_res['win'] == 'LOSS').sum())
    n_ties = int((df_res['win'] == 'TIE').sum())
    win_rate = (n_wins / len(df_res)) * 100.0

    from scipy import stats
    t_stat, p_val_t = stats.ttest_rel(dice_syngs_vals, dice_ants_vals)
    w_stat, p_val_w = stats.wilcoxon(dice_syngs_vals, dice_ants_vals)
    d_diff = dice_syngs_vals - dice_ants_vals
    cohen_d = float(np.mean(d_diff) / (np.std(d_diff, ddof=1) + 1e-8))

    print("\n" + "=" * 80)
    print(" === FULL 90-PAIR MINDBOGGLE BENCHMARK RESULTS: SyNGS BALANCED SOBOLEV ===")
    print("=" * 80)
    print(f" Total Evaluated Pairs   : {len(df_res)} / {total_pairs}")
    print(f" Mean SyNGS Cortical DICE: {np.mean(dice_syngs_vals):.4f} ± {np.std(dice_syngs_vals):.4f}")
    print(f" Mean ANTs Cortical DICE : {np.mean(dice_ants_vals):.4f} ± {np.std(dice_ants_vals):.4f}")
    print(f" Mean Symmetric Gain     : {np.mean(gains):+.2f}%")
    print(f" Cohort Win Rate         : {n_wins}W / {n_losses}L / {n_ties}T ({win_rate:.1f}%)")
    print(f" Paired t-test           : t = {t_stat:.4f}, p = {p_val_t:.2e}")
    print(f" Wilcoxon Signed-Rank    : W = {w_stat:.1f}, p = {p_val_w:.2e}")
    print(f" Cohen's d Effect Size   : d = {cohen_d:.2f}")
    print(f" Mean Brain Folding %    : {df_res['fold_brain_syngs_pct'].mean():.5f}%")
    print(f" Mean Runtime (SyNGS)    : {df_res['time_syngs_s'].mean():.1f}s")
    print(f" Mean GPU Speedup        : {df_res['speedup'].mean():.2f}x")
    print(f" Total Wall-Clock Time   : {total_time_min:.1f} minutes")
    print("=" * 80 + "\n", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete 90-pair SyNGS Balanced Sobolev cohort benchmark.")
    parser.add_argument("--alpha", type=float, default=0.35, help="Sobolev damping parameter alpha (default: 0.35)")
    parser.add_argument("--cfl", type=float, default=0.25, help="CFL step limit max_step_norm (default: 0.25)")
    parser.add_argument("--flow-sigma", type=float, default=3.0, help="Fluid sigma in mm (default: 3.0)")
    parser.add_argument("--output-csv", type=str, default="results/cohort_90pair_syngs_sobolev_summary.csv")
    parser.add_argument("--force", action="store_true", help="Force recomputation from scratch without resuming.")
    args = parser.parse_args()

    run_cohort_benchmark(
        alpha=args.alpha,
        cfl=args.cfl,
        flow_sigma=args.flow_sigma,
        output_csv=args.output_csv,
        force=args.force
    )
