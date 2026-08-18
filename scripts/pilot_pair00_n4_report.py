"""
scripts/pilot_pair00_n4_report.py — Pilot Evaluation of Pair 00 with ANTsTorch N4 Preprocessing
=============================================================================================

Applies `antstorch.n4_bias_field_correction` as preprocessing on both fixed and moving
images for Mindboggle Pair 00, runs robust affine + deformable SyN registration, and
generates a standard 5-figure visual HTML diagnostic report.
"""

import os
import sys
import time
import json
import torch
import numpy as np
import ants
import antstorch

import syntx
from syntx.benchmark import load_mindboggle_pair, compute_bidirectional_dice, compute_jacobian_metrics
from syntx.viz import create_registration_report


def run_antstorch_n4(image: ants.ANTsImage, name: str = "image") -> ants.ANTsImage:
    """Applies differentiable ANTsTorch N4 bias field correction to an ANTsImage."""
    t0 = time.time()
    arr = image.numpy()  # ITK: [X, Y, Z]
    # PyTorch: [1, 1, Z, Y, X]
    tensor = torch.from_numpy(arr.transpose(2, 1, 0)).unsqueeze(0).unsqueeze(0).float()
    mask = (tensor > 0.01).float()

    print(f"[{name}] Applying antstorch.n4_bias_field_correction...", flush=True)
    corrected_tensor = antstorch.n4_bias_field_correction(
        tensor,
        mask=mask,
        shrink_factor=4,
        convergence={"iters": [50, 50, 50, 50], "tol": 1e-7}
    )
    corrected_arr = corrected_tensor.squeeze().detach().cpu().numpy().transpose(2, 1, 0)
    corrected_img = ants.from_numpy(
        corrected_arr,
        origin=image.origin,
        spacing=image.spacing,
        direction=image.direction
    )
    elapsed = time.time() - t0
    print(f"[{name}] N4 correction complete in {elapsed:.2f}s (Range: [{corrected_img.min():.1f}, {corrected_img.max():.1f}])", flush=True)
    return corrected_img


def main():
    print("=" * 80)
    print("      PILOT: MINDBOGGLE PAIR 00 WITH ANTSTORCH N4 PREPROCESSING")
    print("=" * 80, flush=True)

    # 1. Load Pair 00
    print("\n1. Loading Mindboggle Pair 00...")
    pair = load_mindboggle_pair(pair_idx=0)
    fixed_raw = pair["fixed"]
    moving_raw = pair["moving"]
    fixed_lbl = pair["fixed_label"]
    moving_lbl = pair["moving_label"]
    fixed_id = pair["fixed_id"]
    moving_id = pair["moving_id"]
    print(f"   Fixed Subject:  {fixed_id} ({fixed_raw.shape}, spacing: {fixed_raw.spacing})")
    print(f"   Moving Subject: {moving_id} ({moving_raw.shape}, spacing: {moving_raw.spacing})")

    # 2. Apply ANTsTorch N4 Preprocessing & Intensity Normalization
    print("\n2. Preprocessing images with antstorch.n4_bias_field_correction...", flush=True)
    t_n4_start = time.time()
    fixed_n4_raw = run_antstorch_n4(fixed_raw, name=f"Fixed ({fixed_id})")
    moving_n4_raw = run_antstorch_n4(moving_raw, name=f"Moving ({moving_id})")
    total_n4_time = time.time() - t_n4_start
    print(f"   Total N4 Preprocessing Time: {total_n4_time:.2f}s", flush=True)

    from syntx.benchmark.evaluate import normalize_intensity
    fixed_n4 = normalize_intensity(fixed_n4_raw)
    moving_n4 = normalize_intensity(moving_n4_raw)

    # 3. Robust Affine Registration on N4-corrected images
    print("\n3. Running Deterministic Robust Affine Registration...", flush=True)
    t_aff_start = time.time()
    reg_aff = syntx.robust_affine(
        fixed=fixed_n4,
        moving=moving_n4,
        mode="auto",
        verbose=False
    )
    aff_time = time.time() - t_aff_start

    # Evaluate Affine Dice on discrete DKT31 labels
    aff_fwd = reg_aff["fwdtransforms"]
    aff_inv = reg_aff["invtransforms"]
    aff_dice_f, aff_dice_m, aff_dice_s = compute_bidirectional_dice(
        fl=fixed_lbl,
        ml=moving_lbl,
        fi=fixed_n4,
        mi=moving_n4,
        fwdtransforms=aff_fwd,
        invtransforms=aff_inv
    )
    print(f"   Affine Complete in {aff_time:.2f}s", flush=True)
    print(f"   Affine Cortical Dice: Sym={aff_dice_s:.4f} (Fixed={aff_dice_f:.4f}, Moving={aff_dice_m:.4f})", flush=True)

    # 4. Deformable Gaussian SyN Registration (Exact Mindboggle Evaluation Parameters)
    print("\n4. Running Deformable SyN Registration (Gaussian Regularizer)...", flush=True)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using PyTorch backend on device: '{device}'", flush=True)
    t_syn_start = time.time()
    reg_syn = syntx.syn(
        fixed=fixed_n4,
        moving=moving_n4,
        initial_transform=aff_fwd[0],
        backend="pytorch",
        device=device,
        grad_step=0.25,
        flow_sigma=3.0,
        total_sigma=0.0,
        reg_iterations=[100, 100, 50],
        similarity_metric="cc2",
        use_analytical_gradients=False,
        syn_sampling=2,
        antisymmetric=True,
        inverse_method="anderson",
        formulation="eulerian",
        regularizer="gaussian",
        verbose=False
    )
    syn_time = time.time() - t_syn_start
    print(f"   Deformable SyN Complete in {syn_time:.2f}s", flush=True)

    # 5. Evaluate Deformable Dice & Metrics
    print("\n5. Computing Quantitative Registration Metrics...")
    syn_fwd = reg_syn["fwdtransforms"]
    syn_inv = reg_syn["invtransforms"]
    syn_dice_f, syn_dice_m, syn_dice_s = compute_bidirectional_dice(
        fl=fixed_lbl,
        ml=moving_lbl,
        fi=fixed_n4,
        mi=moving_n4,
        fwdtransforms=syn_fwd,
        invtransforms=syn_inv
    )

    # Jacobian determinant regularity
    warp_img = ants.image_read(syn_fwd[0])
    jac_metrics = compute_jacobian_metrics(fixed_n4, warp_img)
    folding_pct = jac_metrics.get("folding_pct", jac_metrics.get("folding_percentage", 0.0))
    min_jac = jac_metrics.get("min", jac_metrics.get("min_jacobian", 1.0))

    print(f"   Deformable Cortical Dice: Sym={syn_dice_s:.4f} (Fixed={syn_dice_f:.4f}, Moving={syn_dice_m:.4f})", flush=True)
    print(f"   Jacobian Folding Rate:    {folding_pct:.4f}% (Min det(J): {min_jac:.4f})", flush=True)

    # 6. Generate 5-Figure Interactive HTML Diagnostic Report
    print("\n6. Generating Standard 5-Figure HTML Verification Report...", flush=True)
    os.makedirs("docs/reports", exist_ok=True)
    report_html = "docs/reports/pair_00_n4_standard_report.html"

    report_dict = create_registration_report(
        fixed=fixed_n4,
        moving=moving_n4,
        reg=reg_syn,
        fixed_label=fixed_lbl,
        moving_label=moving_lbl,
        output_html=report_html,
        fixed_name=f"Fixed ({fixed_id}) [ANTsTorch N4]",
        moving_name=f"Moving ({moving_id}) [ANTsTorch N4]",
        title=f"Mindboggle Pair 00 ({fixed_id} -> {moving_id}) with ANTsTorch N4 Preprocessing",
        verbose=False
    )

    print(f"\n================================================================================", flush=True)
    print(f"                   PAIR 00 + ANTSTORCH N4 REPORT GENERATED", flush=True)
    print(f"================================================================================", flush=True)
    print(f"Report File: {report_dict['html_path']}", flush=True)
    print(f"Summary Metrics:", flush=True)
    print(f"  - Preprocessing:           ANTsTorch N4 Bias Field Correction ({total_n4_time:.2f}s)", flush=True)
    print(f"  - Affine Symmetric Dice:   {aff_dice_s:.4f} (Fixed: {aff_dice_f:.4f}, Moving: {aff_dice_m:.4f})", flush=True)
    print(f"  - Deform Symmetric Dice:   {syn_dice_s:.4f} (Fixed: {syn_dice_f:.4f}, Moving: {syn_dice_m:.4f})", flush=True)
    print(f"  - Jacobian Folding Rate:   {folding_pct:.4f}%", flush=True)
    print(f"  - Min Jacobian det(J):     {min_jac:.4f}", flush=True)
    print(f"  - SyN Execution Time:      {syn_time:.2f}s (Device: {device})", flush=True)
    print(f"================================================================================\n", flush=True)

    # Save metrics JSON
    results_json = "docs/reports/pair_00_n4_metrics.json"
    metrics_record = {
        "pair_idx": 0,
        "fixed_id": fixed_id,
        "moving_id": moving_id,
        "preprocessing": "antstorch.n4_bias_field_correction",
        "n4_time_sec": total_n4_time,
        "affine_time_sec": aff_time,
        "syn_time_sec": syn_time,
        "device": device,
        "affine_dice_sym": aff_dice_s,
        "affine_dice_fixed": aff_dice_f,
        "affine_dice_moving": aff_dice_m,
        "syntx_dice_sym": syn_dice_s,
        "syntx_dice_fixed": syn_dice_f,
        "syntx_dice_moving": syn_dice_m,
        "folding_percentage": folding_pct,
        "min_jacobian": min_jac,
        "report_html": os.path.abspath(report_html)
    }
    with open(results_json, "w") as f:
        json.dump(metrics_record, f, indent=2)


if __name__ == "__main__":
    main()
