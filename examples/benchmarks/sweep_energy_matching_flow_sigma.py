#!/usr/bin/env python3
"""
Energy-Matching Sweep: flow_sigma > 3.0 and grad_step in ANTs C++ vs syntx.syn
=============================================================================
Sweeps flow_sigma across [3.0, 4.5, 6.0, 8.0] and grad_step across [0.10, 0.15, 0.25, 0.35, 0.50]
to find exact iso-energy matching points (Harmonic & Bending Energy) between
ANTs C++ SyN and syntx.syn, measuring DICE and Folding at matched energies.
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
    """
    Computes exact Harmonic Energy ||∇u||² and Bending Energy ||∇²u||²
    from the displacement vector field file.
    """
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

def main():
    print("=" * 125, flush=True)
    print(" ENERGY-MATCHING PARAMETER SWEEP: flow_sigma > 3.0 & grad_step in ANTs C++ vs syntx.syn ")
    print("=" * 125, flush=True)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Compute Device: {device.upper()}", flush=True)

    # Use Pair 00 (OASIS-1 -> OASIS-2) as representative 3D Mindboggle pair
    p = load_mindboggle_pair(0, "examples/pairs.csv")
    fi_raw, mi_raw = p['fixed'], p['moving']
    fl, ml = p['fixed_label'], p['moving_label']

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

    fi = normalize_intensity(fi_raw)
    mi = normalize_intensity(mi_raw)

    print("\n--- Computing Locked Multi-Start Affine Initialization ---", flush=True)
    t0_aff = time.time()
    reg_aff = syntx.robust_affine(fi, mi, mode="auto", verbose=False)
    t_aff = time.time() - t0_aff
    aff_0 = reg_aff["fwdtransforms"][0]

    d_fix_aff, d_mov_aff, d_sym_aff = compute_bidirectional_dice(
        fl=fl, ml=ml, fi=fi, mi=mi,
        fwdtransforms=[aff_0], invtransforms=[aff_0],
        whichtoinvert_inv=[True]
    )
    print(f" [Affine Init] done in {t_aff:.2f}s | Baseline Sym DICE: {d_sym_aff:.4f}\n", flush=True)

    # Parameter Grids
    # 1. ANTs C++ Reference Grid
    ANTS_GRID = [
        (3.0, 0.15),
        (3.0, 0.25),  # Official Default
        (3.0, 0.35),
        (4.5, 0.25),
        (4.5, 0.35),
        (6.0, 0.25),
        (6.0, 0.50),
    ]

    # 2. Syntx Gaussian SyN Grid (Testing larger flow_sigma > 3.0)
    SYNTX_GAUSS_GRID = [
        # Standard flow_sigma=3.0
        (3.0, 0.10),
        (3.0, 0.15),
        (3.0, 0.25),
        (3.0, 0.35),
        (3.0, 0.50),
        # flow_sigma = 4.5
        (4.5, 0.15),
        (4.5, 0.25),
        (4.5, 0.35),
        (4.5, 0.50),
        # flow_sigma = 6.0
        (6.0, 0.25),
        (6.0, 0.35),
        (6.0, 0.50),
        (6.0, 0.75),
        # flow_sigma = 8.0
        (8.0, 0.35),
        (8.0, 0.50),
        (8.0, 0.75),
        (8.0, 1.00),
    ]

    # 3. Syntx Sobolev SyN Grid (flow_sigma + sobolev_alpha)
    SYNTX_SOBOLEV_GRID = [
        (3.0, 1.0, 0.25),
        (3.0, 1.0, 0.35),
        (4.5, 1.5, 0.35),
        (4.5, 1.5, 0.50),
        (6.0, 2.0, 0.50),
        (6.0, 2.0, 0.75),
    ]

    all_rows = []

    # Run ANTs C++
    print(f"\n{'='*40} Sweeping ANTs_CPP_SyN Grid {'='*40}", flush=True)
    for f_sigma, g_step in ANTS_GRID:
        t0 = time.time()
        res = ants.registration(
            fixed=fi,
            moving=mi,
            type_of_transform="SyN",
            initial_transform=aff_0,
            syn_metric="CC",
            syn_sampling=2,
            reg_iterations=(100, 50, 10),
            flow_sigma=f_sigma,
            total_sigma=0.0,
            grad_step=g_step,
            verbose=False
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

        row = {
            "Engine": "ANTs_CPP_SyN",
            "flow_sigma": f_sigma,
            "sobolev_alpha": 0.0,
            "grad_step": g_step,
            "Sym_DICE": d_sym,
            "Harmonic_Energy": harm_e,
            "Bending_Energy": bend_e,
            "Fold_Pct": fold_pct,
            "Min_detJ": min_jac,
            "Max_Disp_mm": disp_max,
            "Runtime_s": t_run
        }
        all_rows.append(row)
        print(f" >> ANTs_CPP  sigma={f_sigma:<4.1f} step={g_step:<4.2f} | Sym DICE={d_sym:.4f} | HarmE={harm_e:.4f} | BendE={bend_e:.4f} | Fold%={fold_pct:.4f}% | MinJ={min_jac:+.4f} | MaxDisp={disp_max:.2f}mm | Time={t_run:.1f}s", flush=True)

    # Run Syntx Gaussian
    print(f"\n{'='*40} Sweeping Syntx_Gaussian_SyN Grid (flow_sigma > 3.0) {'='*40}", flush=True)
    for f_sigma, g_step in SYNTX_GAUSS_GRID:
        t0 = time.time()
        res = syntx.syn(
            fixed=fi,
            moving=mi,
            initial_transform=aff_0,
            backend="pytorch",
            device=device,
            formulation="eulerian",
            regularizer="gaussian",
            optimizer="cfl",
            grad_step=g_step,
            flow_sigma=f_sigma,
            total_sigma=0.0,
            reg_iterations=[100, 50, 10],
            levels=[4, 2, 1],
            syn_metric="lncc",
            syn_sampling=2,
            inverse_method="anderson",
            in_loop_inv_steps=10,
            use_analytical_gradients=False,
            verbose=False
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

        row = {
            "Engine": "Syntx_Gaussian_SyN",
            "flow_sigma": f_sigma,
            "sobolev_alpha": 0.0,
            "grad_step": g_step,
            "Sym_DICE": d_sym,
            "Harmonic_Energy": harm_e,
            "Bending_Energy": bend_e,
            "Fold_Pct": fold_pct,
            "Min_detJ": min_jac,
            "Max_Disp_mm": disp_max,
            "Runtime_s": t_run
        }
        all_rows.append(row)
        print(f" >> Syntx_Gauss sigma={f_sigma:<4.1f} step={g_step:<4.2f} | Sym DICE={d_sym:.4f} | HarmE={harm_e:.4f} | BendE={bend_e:.4f} | Fold%={fold_pct:.4f}% | MinJ={min_jac:+.4f} | MaxDisp={disp_max:.2f}mm | Time={t_run:.1f}s", flush=True)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Run Syntx Sobolev
    print(f"\n{'='*40} Sweeping Syntx_Sobolev_SyN Grid {'='*40}", flush=True)
    for f_sigma, s_alpha, g_step in SYNTX_SOBOLEV_GRID:
        t0 = time.time()
        res = syntx.syn(
            fixed=fi,
            moving=mi,
            initial_transform=aff_0,
            backend="pytorch",
            device=device,
            formulation="eulerian",
            regularizer="sobolev",
            sobolev_alpha=s_alpha,
            optimizer="cfl",
            grad_step=g_step,
            flow_sigma=f_sigma,
            total_sigma=0.0,
            reg_iterations=[100, 50, 10],
            levels=[4, 2, 1],
            syn_metric="lncc",
            syn_sampling=2,
            inverse_method="anderson",
            in_loop_inv_steps=10,
            use_analytical_gradients=False,
            verbose=False
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

        row = {
            "Engine": "Syntx_Sobolev_SyN",
            "flow_sigma": f_sigma,
            "sobolev_alpha": s_alpha,
            "grad_step": g_step,
            "Sym_DICE": d_sym,
            "Harmonic_Energy": harm_e,
            "Bending_Energy": bend_e,
            "Fold_Pct": fold_pct,
            "Min_detJ": min_jac,
            "Max_Disp_mm": disp_max,
            "Runtime_s": t_run
        }
        all_rows.append(row)
        print(f" >> Syntx_Sobolev sigma={f_sigma:<4.1f} alpha={s_alpha:<4.1f} step={g_step:<4.2f} | Sym DICE={d_sym:.4f} | HarmE={harm_e:.4f} | BendE={bend_e:.4f} | Fold%={fold_pct:.4f}% | MinJ={min_jac:+.4f} | MaxDisp={disp_max:.2f}mm | Time={t_run:.1f}s", flush=True)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    out_csv = "results/sweep_energy_matching_flow_sigma.csv"
    os.makedirs("results", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Saved energy matching results to: {out_csv}", flush=True)

    print("\n" + "=" * 140, flush=True)
    print(" MASTER ISO-ENERGY COMPARISON TABLE: ANTs C++ vs syntx.syn across flow_sigma & grad_step ")
    print("=" * 140, flush=True)
    header = f"{'Engine':<20} {'flow_sigma':>10} {'alpha':>6} {'grad_step':>9} {'Sym DICE':>9} {'Harm Energy':>12} {'Bend Energy':>12} {'Fold %':>9} {'Min detJ':>10} {'Time':>7}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for _, r in df.iterrows():
        print(f"{r['Engine']:<20} {r['flow_sigma']:>10.1f} {r['sobolev_alpha']:>6.1f} {r['grad_step']:>9.2f} {r['Sym_DICE']:>9.4f} {r['Harmonic_Energy']:>12.4f} {r['Bending_Energy']:>12.4f} {r['Fold_Pct']:>9.4f}% {r['Min_detJ']:>10.4f} {r['Runtime_s']:>6.1f}s", flush=True)
    print("=" * 140 + "\n", flush=True)

if __name__ == "__main__":
    main()
