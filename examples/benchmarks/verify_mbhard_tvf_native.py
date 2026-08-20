#!/usr/bin/env python3
"""
Verification Script: mbhard Native Space (Pair 77: OASIS-TRT-20-8 -> NKI-TRT-20-3)
Evaluates peak CFL-bounded SobolevAdam TVF parameters on the native-space
anisotropic acquisitions and manual DKT31 segmentations.
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import ants

import syntx
from syntx.deformation_metrics import compute_bidirectional_dice, compute_jacobian_metrics
from syntx.viz.reports import create_registration_report

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
    print("=" * 90, flush=True)
    print(" VERIFYING mbhard NATIVE SPACE (Pair 77: OASIS-8 -> NKI-3) WITH BEST TVF PARAMETERS ")
    print("=" * 90, flush=True)

    # 1. Hardware Detection
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Compute Device: {device.upper()}", flush=True)

    # 2. File Paths for mbhard Native Space (Pair 77: OASIS-TRT-20-8 -> NKI-TRT-20-3)
    data_root = "/Users/stnava/data/mindboggle/volumes"
    fi_path = os.path.join(data_root, "NKI-TRT-20_volumes/NKI-TRT-20-3/t1weighted_brain.nii.gz")
    mi_path = os.path.join(data_root, "OASIS-TRT-20_volumes/OASIS-TRT-20-8/t1weighted_brain.nii.gz")
    fl_path = os.path.join(data_root, "NKI-TRT-20_volumes/NKI-TRT-20-3/labels.DKT31.manual.nii.gz")
    ml_path = os.path.join(data_root, "OASIS-TRT-20_volumes/OASIS-TRT-20-8/labels.DKT31.manual.nii.gz")

    print(f"Fixed Image (Target Native):   {fi_path}", flush=True)
    print(f"Moving Image (Source Native):  {mi_path}", flush=True)
    print(f"Fixed Labels (DKT31 Native):   {fl_path}", flush=True)
    print(f"Moving Labels (DKT31 Native):  {ml_path}\n", flush=True)

    fi_raw = ants.image_read(fi_path)
    mi_raw = ants.image_read(mi_path)
    fl = ants.image_read(fl_path)
    ml = ants.image_read(ml_path)

    fi = normalize_intensity(fi_raw)
    mi = normalize_intensity(mi_raw)

    print(f"Fixed Image Spacing:  {fi.spacing} | Shape: {fi.shape}", flush=True)
    print(f"Moving Image Spacing: {mi.spacing} | Shape: {mi.shape}\n", flush=True)

    # 3. Step 1: Multi-Start Robust Affine Initialization
    print("-" * 80, flush=True)
    print(" Step 1: Running Deterministic Multi-Start Robust Affine Alignment...", flush=True)
    print("-" * 80, flush=True)
    t0_aff = time.time()
    reg_aff = syntx.robust_affine(fi, mi, mode="auto", verbose=False)
    t_aff = time.time() - t0_aff
    aff_tx = reg_aff["fwdtransforms"][0]

    d_fix_aff, d_mov_aff, d_sym_aff = compute_bidirectional_dice(
        fl=fl,
        ml=ml,
        fi=fi,
        mi=mi,
        fwdtransforms=[aff_tx],
        invtransforms=[aff_tx],
        whichtoinvert_inv=[True]
    )
    print(f"   => Affine Initialization Complete in {t_aff:.2f}s")
    print(f"   => Baseline Affine Overlap: Sym DICE = {d_sym_aff:.4f} (Fixed: {d_fix_aff:.4f}, Moving: {d_mov_aff:.4f})\n", flush=True)

    # 4. Step 2: Continuous Time-Varying Velocity Field (TVF) Registration
    # Peak Provenance Parameters:
    schedule = [100, 50, 10]
    print("-" * 80, flush=True)
    print(f" Step 2: Optimizing TVF with Peak Provenance Schedule {schedule}...", flush=True)
    print("   * Optimizer:        SobolevAdam (lr=1.2, max_step_norm=0.35 voxels)")
    print("   * Regularization:   Physical Green Sobolev ((I - alpha Delta)^5, alpha=0.035 mm^-1)")
    print("   * Fluid / Elastic:  flow_sigma=1.0 (fluid), total_sigma=0.035 (elastic)")
    print("   * Trajectory Loss:  Multipoint LNCC at t in [0.0, 0.5, 1.0]")
    print("   * Trajectory Speed: Constant Speed Integration (relaxation=0.10, momentum=0.9)")
    print("   * Fast Filtering:   VRAM Fourier Transfer Function Caching")
    print("-" * 80, flush=True)

    t0_tvf = time.time()
    res = syntx.tvf(
        fixed=fi,
        moving=mi,
        initial_transform=aff_tx,
        backend="pytorch",
        device=device,
        reg_iterations=schedule,
        optimizer="sobolev_adam",
        optimizer_lr=1.2,
        max_step_norm=0.35,
        sobolev_alpha=0.035,
        flow_sigma=1.0,
        total_sigma=0.035,
        regularizer="sobolev",
        solver="euler",
        n_time_steps=3,
        multipoint_loss=[0.0, 0.5, 1.0],
        constant_speed=True,
        constant_speed_relaxation=0.10,
        cfl_momentum=0.9,
        fast_smooth=True,
        use_analytical_gradients=False,
        amp=False,
        verbose=True
    )
    t_tvf = time.time() - t0_tvf
    print(f"\n   => TVF Optimization Finished in {t_tvf:.2f}s ({t_tvf/60:.2f} min)\n", flush=True)

    # 5. Step 3: Compute Quantitative Metrology
    print("-" * 80, flush=True)
    print(" Step 3: Evaluating Standard Quantitative Metrology...", flush=True)
    print("-" * 80, flush=True)

    fwd_tx = res["fwdtransforms"]
    inv_tx = res["invtransforms"]
    which_inv = res.get("whichtoinvert_inv", [True, False])

    # 5a. Bidirectional Cortical DKT31 DICE
    d_fix, d_mov, d_sym = compute_bidirectional_dice(
        fl=fl,
        ml=ml,
        fi=fi,
        mi=mi,
        fwdtransforms=fwd_tx,
        invtransforms=inv_tx,
        whichtoinvert_inv=which_inv
    )

    # 5b. Jacobian Determinant & Manifold Regularity
    jac_metrics = compute_jacobian_metrics(fi, fwd_tx[0])
    fold_pct = float(jac_metrics.get("folding_pct", 0.0))
    min_jac = float(jac_metrics.get("min", 0.0))
    max_jac = float(jac_metrics.get("max", 0.0))
    std_jac = float(jac_metrics.get("std", 0.0))

    # 6. Print Comprehensive Summary
    print("\n" + "=" * 90, flush=True)
    print(" VERIFICATION RESULTS FOR mbhard NATIVE SPACE (OASIS-8 -> NKI-3) ")
    print("=" * 90, flush=True)
    print(f" * Multi-Scale Schedule:              {schedule}")
    print(f" * Mean Symmetric Cortical DICE:      {d_sym:.4f}  (+{d_sym - d_sym_aff:.4f} gain over affine)")
    print(f" * Fixed Space DICE (Target overlap): {d_fix:.4f}")
    print(f" * Moving Space DICE (Source overlap):{d_mov:.4f}")
    print(f" * Non-Invertible Grid Folds:         {fold_pct:.4f}% ({'STRICT 0% FOLDING - PASS' if fold_pct == 0.0 else 'FAIL'})")
    print(f" * Minimum Jacobian Determinant:      {min_jac:+.4f} ({'STRICTLY POSITIVE - PASS' if min_jac > 0 else 'FAIL'})")
    print(f" * Maximum Jacobian Determinant:      {max_jac:+.4f}")
    print(f" * Jacobian Determinant Std Dev:      {std_jac:.4f}")
    print(f" * Deformable TVF Execution Time:     {t_tvf:.2f} s ({t_tvf/60:.2f} min)")
    print(f" * Total Pipeline Execution Time:     {t_aff + t_tvf:.2f} s")
    print("=" * 90 + "\n", flush=True)

    # 7. Generate Visual Verification Suite and HTML Report
    out_dir = "/Users/stnava/data/syntx/results/verification_mbhard_native_best"
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, "mbhard_native_verification_report.html")

    print(f"Generating Interactive HTML Diagnostic Report at: {html_path} ...", flush=True)

    create_registration_report(
        fixed=fi,
        moving=mi,
        reg=res,
        fixed_label=fl,
        moving_label=ml,
        output_html=html_path,
        title="mbhard Native Space Verification Report (CFL-SobolevAdam Peak TVF)"
    )

    print(f"SUCCESS! Interactive HTML report generated at: {html_path}", flush=True)

if __name__ == "__main__":
    main()
