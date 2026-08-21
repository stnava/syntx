#!/usr/bin/env python3
"""
Empirical 3-Way Parameter Sweep: Deformation Energy, DICE, and Folding
======================================================================
Systematically sweeps `grad_step` across ANTs C++ SyN and syntx.syn
(Gaussian, Sobolev, and RegAdam) on 3D Mindboggle brain data to measure:
1. Harmonic & Bending Deformation Energy
2. Cortical DKT31 Label DICE
3. True Raw Jacobian Folding Rate (det(J) <= 0) and min det(J)
4. Displacement magnitude statistics
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
    else:  # 2D
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
    print("=" * 115, flush=True)
    print(" EMPIRICAL EXPERIMENT: DEFORMATION ENERGY vs DICE vs FOLDING AS A FUNCTION OF GRAD_STEP ")
    print("=" * 115, flush=True)

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

    # 1. Standard Locked Affine Initializer (Same for all methods)
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

    # 2. Grad Step Sweep Values
    GRAD_STEPS = [0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.75]
    
    ENGINES = [
        ("ANTs_CPP_SyN", "ants"),
        ("Syntx_Gaussian_SyN", "syntx_gaussian"),
        ("Syntx_Sobolev_SyN", "syntx_sobolev"),
        ("Syntx_RegAdam_SyN", "syntx_regadam"),
    ]

    all_records = []

    for engine_name, engine_type in ENGINES:
        print(f"\n{'='*35} Sweeping {engine_name} {'='*35}", flush=True)
        for step in GRAD_STEPS:
            t0 = time.time()

            if engine_type == "ants":
                # ANTs C++ SyN
                res = ants.registration(
                    fixed=fi,
                    moving=mi,
                    type_of_transform="SyN",
                    initial_transform=aff_0,
                    syn_metric="CC",
                    syn_sampling=2,
                    reg_iterations=(100, 50, 10),
                    flow_sigma=3.0,
                    total_sigma=0.0,
                    grad_step=step,
                    verbose=False
                )
            elif engine_type == "syntx_gaussian":
                # Syntx Gaussian SyN (Eulerian autograd)
                res = syntx.syn(
                    fixed=fi,
                    moving=mi,
                    initial_transform=aff_0,
                    backend="pytorch",
                    device=device,
                    formulation="eulerian",
                    regularizer="gaussian",
                    optimizer="cfl",
                    grad_step=step,
                    flow_sigma=3.0,
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
            elif engine_type == "syntx_sobolev":
                # Syntx Sobolev SyN
                res = syntx.syn(
                    fixed=fi,
                    moving=mi,
                    initial_transform=aff_0,
                    backend="pytorch",
                    device=device,
                    formulation="eulerian",
                    regularizer="sobolev",
                    sobolev_alpha=1.0,
                    optimizer="cfl",
                    grad_step=step,
                    flow_sigma=3.0,
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
            elif engine_type == "syntx_regadam":
                # Syntx RegAdam SyN
                res = syntx.syn(
                    fixed=fi,
                    moving=mi,
                    initial_transform=aff_0,
                    backend="pytorch",
                    device=device,
                    formulation="eulerian",
                    regularizer="gaussian",
                    optimizer="reg_adam",
                    optimizer_lr=step * 2.0,  # proportional LR
                    grad_step=step,
                    flow_sigma=3.0,
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

            # 1. DICE Overlap
            d_fix, d_mov, d_sym = compute_bidirectional_dice(
                fl=fl, ml=ml, fi=fi, mi=mi,
                fwdtransforms=fwd_tx, invtransforms=inv_tx,
                whichtoinvert_inv=which_inv
            )

            # 2. Raw Non-Log Jacobian Determinant (Exact Topological Fold Rate)
            warp_file = next((tx for tx in fwd_tx if isinstance(tx, str) and tx.endswith(('.nii', '.nii.gz'))), None)
            jac = compute_jacobian_metrics(fi, warp_file) if warp_file else {}
            fold_pct = float(jac.get("folding_pct", 0.0))
            min_jac = float(jac.get("min", 0.0))
            max_jac = float(jac.get("max", 0.0))

            # 3. Exact Spatial Deformation Energies
            if warp_file:
                harm_e, bend_e, disp_mean, disp_max = compute_displacement_energies(warp_file, fi.spacing)
            else:
                harm_e, bend_e, disp_mean, disp_max = 0.0, 0.0, 0.0, 0.0

            rec = {
                "Engine": engine_name,
                "grad_step": step,
                "Sym_DICE": d_sym,
                "Fix_DICE": d_fix,
                "Mov_DICE": d_mov,
                "Fold_Pct": fold_pct,
                "Min_detJ": min_jac,
                "Max_detJ": max_jac,
                "Harmonic_Energy": harm_e,
                "Bending_Energy": bend_e,
                "Mean_Displacement_mm": disp_mean,
                "Max_Displacement_mm": disp_max,
                "Runtime_s": t_run
            }
            all_records.append(rec)

            print(f" >> {engine_name:<20} step={step:<4.2f} | Sym DICE={d_sym:.4f} | HarmEnergy={harm_e:.4f} | BendEnergy={bend_e:.4f} | Fold%={fold_pct:.4f}% | MinJ={min_jac:+.4f} | MaxDisp={disp_max:.2f}mm | Time={t_run:.1f}s", flush=True)

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Create Master DataFrame
    df = pd.DataFrame(all_records)
    out_csv = "results/sweep_grad_step_energy_dice_folding.csv"
    os.makedirs("results", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Sweep data saved to: {out_csv}", flush=True)

    print("\n" + "=" * 130, flush=True)
    print(" MASTER EMPIRICAL TABLE: GRAD_STEP vs DEFORMATION ENERGY vs DICE vs FOLDING ")
    print("=" * 130, flush=True)
    header = f"{'Engine':<20} {'Step':<5} {'Sym DICE':>9} {'Harm Energy':>12} {'Bend Energy':>12} {'Fold %':>9} {'Min detJ':>10} {'Max Disp (mm)':>14} {'Time':>7}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for _, r in df.iterrows():
        print(f"{r['Engine']:<20} {r['grad_step']:<5.2f} {r['Sym_DICE']:>9.4f} {r['Harmonic_Energy']:>12.4f} {r['Bending_Energy']:>12.4f} {r['Fold_Pct']:>9.4f}% {r['Min_detJ']:>10.4f} {r['Max_Displacement_mm']:>14.2f} {r['Runtime_s']:>6.1f}s", flush=True)
    print("=" * 130 + "\n", flush=True)

if __name__ == "__main__":
    main()
