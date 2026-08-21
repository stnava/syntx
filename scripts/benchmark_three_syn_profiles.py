#!/usr/bin/env python3
"""
Benchmark Evaluation Suite: The 3 Canonical SyN Parameter Sets
==============================================================
Runs and compares the three officially named SyN parameter profiles:
1. `syn_energy_parity`   (Exact ANTs C++ kinetic energy & smoothness match)
2. `syn_balanced_peak`   (Peak accuracy Eulerian SyN standard)
3. `syn_sobolev_shield`  (Spectral Sobolev H^1.5 topology-preserving fluid flow)

Evaluates on representative Mindboggle benchmark pairs.
"""

import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
import ants

import syntx
from syntx.benchmark.data import load_mindboggle_pair
from syntx.deformation_metrics import compute_bidirectional_dice, compute_jacobian_metrics

def compute_displacement_energies(warp_file, spacing):
    """Computes exact Harmonic Energy ||∇u||² and Bending Energy ||∇²u||²."""
    disp_img = ants.image_read(warp_file)
    disp_np = disp_img.numpy()
    if disp_np.ndim == 4 and disp_np.shape[0] == 3:
        disp_np = np.moveaxis(disp_np, 0, -1)
    elif disp_np.ndim == 3 and disp_np.shape[0] == 2:
        disp_np = np.moveaxis(disp_np, 0, -1)

    sp_x = spacing[0]
    sp_y = spacing[1] if len(spacing) > 1 else 1.0
    sp_z = spacing[2] if len(spacing) > 2 else 1.0

    if disp_np.ndim == 4:  # 3D
        du_dx = (disp_np[1:, :, :] - disp_np[:-1, :, :]) / sp_x
        du_dy = (disp_np[:, 1:, :] - disp_np[:, :-1, :]) / sp_y
        du_dz = (disp_np[:, :, 1:] - disp_np[:, :, :-1]) / sp_z
        harmonic_energy = float(np.mean(du_dx**2) + np.mean(du_dy**2) + np.mean(du_dz**2))

        d2u_dx2 = (du_dx[1:, :, :] - du_dx[:-1, :, :]) / sp_x
        d2u_dy2 = (du_dy[:, 1:, :] - du_dy[:, :-1, :]) / sp_y
        d2u_dz2 = (du_dz[:, :, 1:] - du_dz[:, :, :-1]) / sp_z
        bending_energy = float(np.mean(d2u_dx2**2) + np.mean(d2u_dy2**2) + np.mean(d2u_dz2**2))

        disp_norm_mean = float(np.mean(np.linalg.norm(disp_np, axis=-1)))
        disp_norm_max = float(np.max(np.linalg.norm(disp_np, axis=-1)))
    else:
        du_dx = (disp_np[1:, :] - disp_np[:-1, :]) / sp_x
        du_dy = (disp_np[:, 1:] - disp_np[:, :-1]) / sp_y
        harmonic_energy = float(np.mean(du_dx**2) + np.mean(du_dy**2))
        d2u_dx2 = (du_dx[1:, :] - du_dx[:-1, :]) / sp_x
        d2u_dy2 = (du_dy[:, 1:] - du_dy[:, :-1]) / sp_y
        bending_energy = float(np.mean(d2u_dx2**2) + np.mean(d2u_dy2**2))
        disp_norm_mean = float(np.mean(np.linalg.norm(disp_np, axis=-1)))
        disp_norm_max = float(np.max(np.linalg.norm(disp_np, axis=-1)))

    return harmonic_energy, bending_energy, disp_norm_mean, disp_norm_max

def normalize_intensity(img: ants.ANTsImage) -> ants.ANTsImage:
    """Foreground 2nd-to-98th percentile intensity normalization."""
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

def main():
    print("=" * 125, flush=True)
    print(" BENCHMARK EVALUATION: THE 3 CANONICAL SyN PARAMETER SETS ")
    print("=" * 125, flush=True)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Compute Device: {device.upper()}", flush=True)

    # 3 Benchmark Pairs: Pair 00 (Intra-OASIS), Pair 08 (Intra-NKI), Pair 77 (Inter-mbhard)
    TEST_PAIRS = [0, 8, 77]

    PROFILES = {
        "1_syn_energy_parity": {
            "description": "Exact ANTs C++ Energy & Smoothness Parity",
            "model_type": "syn",
            "regularizer": "gaussian",
            "flow_sigma": 6.0,
            "total_sigma": 0.0,
            "grad_step": 0.25,
            "reg_iterations": [100, 100, 20]
        },
        "2_syn_balanced_peak": {
            "description": "Balanced Peak Accuracy Eulerian Standard",
            "model_type": "syn",
            "regularizer": "gaussian",
            "flow_sigma": 3.0,
            "total_sigma": 0.0,
            "grad_step": 0.25,
            "reg_iterations": [100, 100, 20]
        },
        "3_syn_sobolev_shield": {
            "description": "Spectral Sobolev Topology-Preserving Shield",
            "model_type": "syn",
            "regularizer": "sobolev",
            "sobolev_alpha": 1.5,
            "flow_sigma": 4.5,
            "total_sigma": 0.0,
            "grad_step": 0.35,
            "reg_iterations": [100, 100, 20]
        }
    }

    all_records = []

    for pair_idx in TEST_PAIRS:
        p = load_mindboggle_pair(pair_idx, "examples/pairs.csv")
        fi_raw, mi_raw = p['fixed'], p['moving']
        fl, ml = p['fixed_label'], p['moving_label']
        c1, s1 = p.get('fixed_cohort', ''), p.get('fixed_id', f'pair_{pair_idx}')
        c2, s2 = p.get('moving_cohort', ''), p.get('moving_id', '')

        pair_name = f"Pair_{pair_idx:02d} ({s1}->{s2})"
        print(f"\n{'='*35} Evaluating {pair_name} {'='*35}", flush=True)

        fi = normalize_intensity(fi_raw)
        mi = normalize_intensity(mi_raw)

        # Affine Init
        t0_aff = time.time()
        reg_aff = syntx.robust_affine(fi, mi, mode="auto", verbose=False)
        t_aff = time.time() - t0_aff
        aff_0 = reg_aff["fwdtransforms"][0]

        d_fix_aff, d_mov_aff, d_sym_aff = compute_bidirectional_dice(
            fl=fl, ml=ml, fi=fi, mi=mi,
            fwdtransforms=[aff_0], invtransforms=[aff_0],
            whichtoinvert_inv=[True]
        )
        print(f" [Affine Initializer] done in {t_aff:.2f}s | Baseline Sym DICE: {d_sym_aff:.4f}", flush=True)

        for prof_name, prof_cfg in PROFILES.items():
            print(f" -> Running {prof_name} ({prof_cfg['description']})...", flush=True)
            kw = dict(prof_cfg)
            kw.pop("description")
            kw.pop("model_type")

            t0 = time.time()
            res = syntx.syn(
                fixed=fi,
                moving=mi,
                initial_transform=aff_0,
                backend="pytorch",
                device=device,
                formulation="eulerian",
                inverse_method="anderson",
                in_loop_inv_steps=10,
                use_analytical_gradients=False,
                syn_metric="lncc",
                syn_sampling=2,
                levels=[4, 2, 1],
                verbose=False,
                **kw
            )
            t_run = time.time() - t0
            fwd_tx = res["fwdtransforms"]
            inv_tx = res["invtransforms"]
            which_inv = res.get("whichtoinvert_inv", [True, False])

            d_fix, d_mov, d_sym = compute_bidirectional_dice(
                fl=fl, ml=ml, fi=fi, mi=mi,
                fwdtransforms=fwd_tx, invtransforms=inv_tx,
                whichtoinvert_inv=which_inv
            )
            warp_file = next((tx for tx in fwd_tx if isinstance(tx, str) and tx.endswith(('.nii', '.nii.gz'))), None)
            jac = compute_jacobian_metrics(fi, warp_file) if warp_file else {}
            fold_pct = float(jac.get("folding_pct", 0.0))
            min_jac = float(jac.get("min", 0.0))
            harm_e, bend_e, disp_mean, disp_max = compute_displacement_energies(warp_file, fi.spacing) if warp_file else (0,0,0,0)

            rec = {
                "Pair": pair_name,
                "Pair_Idx": pair_idx,
                "Profile": prof_name,
                "Description": prof_cfg["description"],
                "Sym_DICE": d_sym,
                "Fix_DICE": d_fix,
                "Mov_DICE": d_mov,
                "Harmonic_Energy": harm_e,
                "Bending_Energy": bend_e,
                "Fold_Pct": fold_pct,
                "Min_detJ": min_jac,
                "Max_Disp_mm": disp_max,
                "Runtime_s": t_run
            }
            all_records.append(rec)
            print(f"    [DONE] Sym DICE={d_sym:.4f} | HarmEnergy={harm_e:.4f} | BendEnergy={bend_e:.4f} | Fold%={fold_pct:.4f}% | MinJ={min_jac:+.4f} | Time={t_run:.1f}s", flush=True)

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(all_records)
    out_csv = "results/benchmark_three_syn_profiles.csv"
    out_json = "results/benchmark_three_syn_profiles.json"
    os.makedirs("results", exist_ok=True)
    df.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\n[SUCCESS] Benchmark records saved to: {out_csv} and {out_json}", flush=True)

    print("\n" + "=" * 135, flush=True)
    print(" MASTER BENCHMARK SUMMARY: THE 3 CANONICAL SyN PARAMETER PROFILES ")
    print("=" * 135, flush=True)
    header = f"{'Pair':<28} {'Profile':<24} {'Sym DICE':>9} {'Harm Energy':>12} {'Bend Energy':>12} {'Fold %':>9} {'Min detJ':>10} {'Time':>7}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for _, r in df.iterrows():
        print(f"{r['Pair']:<28} {r['Profile']:<24} {r['Sym_DICE']:>9.4f} {r['Harmonic_Energy']:>12.4f} {r['Bending_Energy']:>12.4f} {r['Fold_Pct']:>9.4f}% {r['Min_detJ']:>10.4f} {r['Runtime_s']:>6.1f}s", flush=True)
    print("=" * 135 + "\n", flush=True)

    print("--- COHORT AVERAGES ACROSS PROFILES ---", flush=True)
    grp = df.groupby("Profile").agg({
        "Sym_DICE": "mean",
        "Harmonic_Energy": "mean",
        "Bending_Energy": "mean",
        "Fold_Pct": "mean",
        "Min_detJ": "mean",
        "Runtime_s": "mean"
    }).reset_index()
    for _, g in grp.iterrows():
        print(f" {g['Profile']:<24}: Mean Sym DICE={g['Sym_DICE']:.4f} | Mean HarmE={g['Harmonic_Energy']:.4f} | Mean BendE={g['Bending_Energy']:.4f} | Mean Fold%={g['Fold_Pct']:.4f}% | Mean MinJ={g['Min_detJ']:+.4f} | Mean Time={g['Runtime_s']:.1f}s", flush=True)
    print("=" * 135 + "\n", flush=True)

if __name__ == "__main__":
    main()
