#!/usr/bin/env python3
"""
scripts/explore_mbhard_tradeoff.py

Systematic exploration of the Dice-regularity trade-off for div_curl with total_sigma=0.0 on 3D Mindboggle 'mbhard'.
Reuses cached preprocessed images and the canonical affine baseline (0.3324) to execute Stage 2 on MPS.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import ants

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import syntx
from syntx.benchmark.data import load_mindboggle_pair
from syntx.benchmark.evaluate import normalize_intensity, clean_device_cache, AFFINE_BACKEND_KEY
from syntx.deformation_metrics import (
    compute_bidirectional_dice,
    compute_jacobian_metrics,
    compute_harmonic_energy,
    compute_bending_energy,
)
from syntx.core.inverse import calculate_inverse_identity_error
from syntx.viz import create_registration_report


def main():
    print("=" * 80)
    print("   EXPLORING DICE TRADE-OFF ON 3D MINDBOGGLE MBHARD (total_sigma=0.0, MPS)   ")
    print("=" * 80)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device.upper()}")

    # 1. Load cached preprocessed data and canonical affine
    print("\n[1/3] Loading cached pair 44 and canonical affine...", flush=True)
    t0_load = time.time()
    pair_data = load_mindboggle_pair(pair_idx=44, use_n4=True)
    fi = normalize_intensity(pair_data["fixed"])
    mi = normalize_intensity(pair_data["moving"])
    fl, ml = pair_data["fixed_label"], pair_data["moving_label"]
    fixed_id = pair_data["fixed_id"]
    moving_id = pair_data["moving_id"]

    aff_mat_path = f"results/canonical_affines/pair_044_{AFFINE_BACKEND_KEY}_affine.mat"
    assert os.path.exists(aff_mat_path), f"Missing {aff_mat_path}"
    print(f"  Loaded in {time.time() - t0_load:.2f}s | Using Affine: {aff_mat_path}")

    # Test configurations exploring envelope power, alpha, and beta/gamma ratios
    configs = [
        {
            "name": "H2_iso_a10_b10",
            "desc": "H^2 envelope (power=1.0), alpha=1.0, beta=1.0, gamma=1.0 (balanced compliant)",
            "params": {
                "regularizer": "div_curl",
                "beta": 1.0,
                "gamma": 1.0,
                "h3_envelope": True,
                "envelope_power": 1.0,
                "sobolev_alpha": 1.0,
                "flow_sigma": 3.0,
                "total_sigma": 0.0,
                "grad_step": 0.4,
            }
        },
        {
            "name": "H2_modest21_a10_b15_g075",
            "desc": "H^2 envelope (power=1.0), alpha=1.0, beta=1.5, gamma=0.75 (2:1 ratio, flexible shear)",
            "params": {
                "regularizer": "div_curl",
                "beta": 1.5,
                "gamma": 0.75,
                "h3_envelope": True,
                "envelope_power": 1.0,
                "sobolev_alpha": 1.0,
                "flow_sigma": 3.0,
                "total_sigma": 0.0,
                "grad_step": 0.4,
            }
        },
        {
            "name": "H25_iso_a12_b10",
            "desc": "H^2.5 envelope (power=1.5), alpha=1.2, beta=1.0, gamma=1.0 (intermediate damping)",
            "params": {
                "regularizer": "div_curl",
                "beta": 1.0,
                "gamma": 1.0,
                "h3_envelope": True,
                "envelope_power": 1.5,
                "sobolev_alpha": 1.2,
                "flow_sigma": 3.0,
                "total_sigma": 0.0,
                "grad_step": 0.4,
            }
        },
        {
            "name": "H2_iso_a08_b08_step05",
            "desc": "H^2 envelope (power=1.0), alpha=0.8, beta=0.8, gamma=0.8, grad_step=0.5 (high agility)",
            "params": {
                "regularizer": "div_curl",
                "beta": 0.8,
                "gamma": 0.8,
                "h3_envelope": True,
                "envelope_power": 1.0,
                "sobolev_alpha": 0.8,
                "flow_sigma": 3.0,
                "total_sigma": 0.0,
                "grad_step": 0.5,
            }
        }
    ]

    # Stored baseline from previous run for direct comparison
    results = [
        {
            "name": "Previous H3_iso_a225_b15_g15 (Reported Baseline)",
            "desc": "H^3 envelope (power=2.0), alpha=2.25, beta=1.5, gamma=1.5",
            "dice_sym": 0.5872,
            "dice_fixed": 0.5924,
            "dice_moving": 0.5820,
            "liouville_min": 0.0060,
            "liouville_folding": 0.0,
            "fd_folding": 0.0239,
            "mean_inv_err": 0.1135,
            "harmonic_energy": 0.2581,
            "runtime_s": 123.61,
        }
    ]

    best_cand = None
    best_dice = 0.5872

    print(f"\n[2/3] Evaluating {len(configs)} parameter configurations on MPS...", flush=True)

    for idx, cfg in enumerate(configs, 1):
        name = cfg["name"]
        p = cfg["params"]
        print(f"\n--- [{idx}/{len(configs)}] Running Configuration: {name} ---")
        print(f"    Description: {cfg['desc']}")
        print(f"    Params: beta={p['beta']}, gamma={p['gamma']}, power={p['envelope_power']}, alpha={p['sobolev_alpha']}, step={p['grad_step']}")

        clean_device_cache()
        t0 = time.time()
        outprefix = f"results/mbhard_sweep_{name}_"

        res = syntx.syn(
            fixed=fi,
            moving=mi,
            initial_transform=aff_mat_path,
            backend="pytorch",
            device=device,
            similarity_metric="cc2",
            reg_iterations=[100, 100, 20],
            in_loop_inv_steps=10,
            inverse_method="anderson",
            outprefix=outprefix,
            verbose=False,
            **p
        )
        runtime = time.time() - t0
        print(f"    SyN completed in {runtime:.2f}s")

        # Evaluate Metrics
        fwd_tx = res['fwdtransforms']
        inv_tx = res['invtransforms']
        which_inv = res.get('whichtoinvert_inv', [True, False])

        df, dm, dsym = compute_bidirectional_dice(fl, ml, fi, mi, fwd_tx, inv_tx, which_inv)
        fwd_warp_file = next((t for t in fwd_tx if t.endswith('.nii.gz')), fwd_tx[0])

        # Liouville determinant
        det_img = syntx.liouville_determinant(res, fi)
        det_arr = det_img.numpy()
        mask_eval = ants.get_mask(fi).numpy() > 0
        valid_det = det_arr[mask_eval & np.isfinite(det_arr)]
        liouv_min = float(np.min(valid_det)) if len(valid_det) > 0 else float('nan')
        liouv_fold = float(np.mean(valid_det <= 0.0) * 100.0) if len(valid_det) > 0 else float('nan')

        # Finite difference Jacobian
        jac = compute_jacobian_metrics(fi, fwd_warp_file)
        harm = compute_harmonic_energy(fwd_warp_file, spacing=fi.spacing)

        # Inverse identity error
        inv_dict = res.get('inverse_identity_errors', {}).get('phi_1', None)
        mean_inv = float(inv_dict.get('mean_error', 0.0)) if inv_dict else float('nan')

        rec = {
            "name": name,
            "desc": cfg["desc"],
            "params": p,
            "dice_sym": float(dsym),
            "dice_fixed": float(df),
            "dice_moving": float(dm),
            "liouville_min": liouv_min,
            "liouville_folding": liouv_fold,
            "fd_folding": float(jac['folding_pct']),
            "mean_inv_err": mean_inv,
            "harmonic_energy": float(harm),
            "runtime_s": float(runtime),
            "fwd_warp": fwd_warp_file,
            "fwd_tx": fwd_tx,
            "inv_tx": inv_tx,
            "which_inv": which_inv,
            "res": res,
            "det_img": det_img,
        }
        results.append(rec)

        print(f"    --> RESULT: Dice = {dsym:.4f} (Fixed: {df:.4f}, Moving: {dm:.4f})")
        print(f"        Liouville: min={liouv_min:+.4f}, folding={liouv_fold:.4f}% | FD folding={jac['folding_pct']:.4f}%")
        print(f"        Harmonic Energy: {harm:.4e} | Inverse Error: {mean_inv:.4f} mm")

        if dsym > best_dice:
            best_dice = dsym
            best_cand = rec
            print(f"    🌟 NEW BEST CANDIDATE! Dice improved to {dsym:.4f} (+{(dsym - 0.5872)*100:+.2f}%)")

    # 3. Save Sweep Results
    print("\n[3/3] Compiling Comparative Results...", flush=True)
    sweep_json = "docs/provenance/mbhard_div_curl_tradeoff_sweep.json"
    clean_results = [{k: v for k, v in r.items() if k not in ('res', 'det_img')} for r in results]
    with open(sweep_json, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"Sweep results saved to {sweep_json}")

    # Generate updated official HTML report for the winner
    if best_cand is not None and "res" in best_cand:
        print(f"\nGenerating Updated Interactive HTML Report for Winner: {best_cand['name']} (Dice={best_cand['dice_sym']:.4f})...")
        report_path = "docs/reports/mbhard_standard_div_curl_iso_tot0_report.html"

        prov = {
            "algorithm": f"syntx.syn (div_curl_iso_tot0: {best_cand['name']})",
            "regularizer": "div_curl",
            **best_cand["params"],
            "reg_iterations": [100, 100, 20],
            "backend": "PyTorch",
            "device": device,
            "runtime_total_sec": 8.83 + best_cand["runtime_s"],
            "runtime_affine_sec": 8.83,
            "runtime_syn_sec": best_cand["runtime_s"],
            "affine_sym_dice": 0.33235,
            "syn_sym_dice": best_cand["dice_sym"],
            "syn_fixed_dice": best_cand["dice_fixed"],
            "syn_moving_dice": best_cand["dice_moving"],
            "liouville_min_det": best_cand["liouville_min"],
            "liouville_folding_pct": best_cand["liouville_folding"],
            "folding_percentage": best_cand["fd_folding"],
            "harmonic_energy": best_cand["harmonic_energy"],
            "mean_inverse_error_mm": best_cand["mean_inv_err"],
        }

        # Apply transforms to label
        warped_ml = ants.apply_transforms(
            fixed=fi, moving=ml, transformlist=best_cand["fwd_tx"], interpolator="nearestNeighbor"
        )

        create_registration_report(
            fixed=fi,
            moving=mi,
            warped=best_cand["res"]["warpedmovout"],
            warp=best_cand["fwd_warp"],
            fixed_label=fl,
            moving_label=ml,
            warped_label=warped_ml,
            detJ=best_cand["det_img"],
            output_html=report_path,
            fixed_name=f"{fixed_id} (Fixed Target)",
            moving_name=f"{moving_id} (Moving Source)",
            reg=best_cand["res"],
            provenance=prov,
            title=f"Optimized 3D mbhard div_curl ({best_cand['name']}) Report (Dice={best_cand['dice_sym']:.4f})"
        )
        print(f"Report updated at: {report_path}")

    print("\n" + "=" * 80)
    print("                     TRADE-OFF EXPLORATION SCORECARD                    ")
    print("=" * 80)
    print(f"{'Configuration':<35} | {'Dice Sym':<8} | {'Fixed':<6} | {'Moving':<6} | {'Liouv Min':<9} | {'Fold %':<7}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<35} | {r['dice_sym']:<8.4f} | {r.get('dice_fixed', 0):<6.4f} | {r.get('dice_moving', 0):<6.4f} | {r.get('liouville_min', 0):<+9.4f} | {r.get('liouville_folding', 0):<7.4f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
