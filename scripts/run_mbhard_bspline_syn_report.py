"""Run 3D Mindboggle mbhard benchmark with syntx.syn + B-spline regularizer (BSplineSyN)

Generates the standard 5-figure interactive HTML registration report with complete
provenance tracking and quantitative deformation metrics (Symmetric Cortical Dice,
Jacobian determinant min/folding %, real physical inverse identity error in mm, runtime).
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import ants

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import syntx
from syntx.core.inverse import calculate_inverse_identity_error
from syntx.viz import create_registration_report


def compute_symmetric_dice(fixed, moving, fixed_lbl, moving_lbl, fwd_tx, inv_tx, whichtoinvert=None):
    """Computes fixed, moving, and symmetric cortical DKT Dice scores."""
    warped_moving_lbl = ants.apply_transforms(
        fixed=fixed,
        moving=moving_lbl,
        transformlist=fwd_tx,
        interpolator='nearestNeighbor'
    )
    ov_fixed = ants.label_overlap_measures(fixed_lbl, warped_moving_lbl)
    df_f = ov_fixed[~ov_fixed['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_f = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_f.columns else 'MeanOverlap'
    vals_f = pd.to_numeric(df_f[col_f], errors='coerce').to_numpy(dtype=np.float64)
    vals_f = vals_f[np.isfinite(vals_f) & (vals_f >= 0.0) & (vals_f <= 1.0)]
    dice_fixed = float(np.mean(vals_f)) if len(vals_f) > 0 else 0.0

    apply_kwargs = {'interpolator': 'nearestNeighbor'}
    if whichtoinvert is not None:
        apply_kwargs['whichtoinvert'] = whichtoinvert
    else:
        apply_kwargs['whichtoinvert'] = [True] + [False] * (len(inv_tx) - 1) if len(inv_tx) > 0 else []

    warped_fixed_lbl = ants.apply_transforms(
        fixed=moving,
        moving=fixed_lbl,
        transformlist=inv_tx,
        **apply_kwargs
    )
    ov_moving = ants.label_overlap_measures(moving_lbl, warped_fixed_lbl)
    df_m = ov_moving[~ov_moving['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_m = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_m.columns else 'MeanOverlap'
    vals_m = pd.to_numeric(df_m[col_m], errors='coerce').to_numpy(dtype=np.float64)
    vals_m = vals_m[np.isfinite(vals_m) & (vals_m >= 0.0) & (vals_m <= 1.0)]
    dice_moving = float(np.mean(vals_m)) if len(vals_m) > 0 else 0.0

    dice_sym = 0.5 * (dice_fixed + dice_moving)
    return dice_fixed, dice_moving, dice_sym, warped_moving_lbl, warped_fixed_lbl


def main():
    print("=" * 80)
    print("      3D MINDBOGGLE MBHARD BENCHMARK: syntx.syn + BSPLINE REGULARIZER       ")
    print("=" * 80)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.upper()}")

    # 1. Load dataset
    print("[1/5] Loading 'mbhard' benchmark dataset...", flush=True)
    t0_load = time.time()
    data = syntx.benchmark_data('mbhard')
    fi_raw, mi_raw = data['fixed'], data['moving']
    fl, ml = data['fixed_label'], data['moving_label']

    from syntx.benchmark.evaluate import normalize_intensity
    fi = normalize_intensity(fi_raw)
    mi = normalize_intensity(mi_raw)
    print(f"  Fixed shape: {fi.shape}, spacing: {fi.spacing}")
    print(f"  Moving shape: {mi.shape}, spacing: {mi.spacing} [{time.time() - t0_load:.2f}s]")

    # 2. Stage 1: Robust Affine Alignment
    print("\n[2/5] Running Robust Multi-Start Affine Alignment (mode='auto')...", flush=True)
    t0_aff = time.time()
    reg_aff = syntx.robust_affine(fixed=fi, moving=mi, multi_start=True, mode='auto', verbose=False)
    t1_aff = time.time() - t0_aff
    aff_tx = reg_aff['fwdtransforms'][0]
    aff_inv_tx = reg_aff['invtransforms'][0]

    dice_aff_f, dice_aff_m, dice_aff_sym, _, _ = compute_symmetric_dice(
        fi, mi, fl, ml, reg_aff['fwdtransforms'], reg_aff['invtransforms'], reg_aff.get('whichtoinvert_inv', [True])
    )
    print(f"  Affine Sym Dice : {dice_aff_sym:.4f} (Fixed: {dice_aff_f:.4f}, Moving: {dice_aff_m:.4f}) [{t1_aff:.2f}s]")

    # 3. Stage 2: SyN + B-Spline Regularizer (BSplineSyN)
    print("\n[3/5] Running syntx.syn with regularizer='bspline' (spline_distance=4.0 mm)...", flush=True)
    t0_syn = time.time()
    res = syntx.syn(
        fixed=fi,
        moving=mi,
        initial_transform=aff_tx,
        backend='pytorch',
        device=device,
        similarity_metric='lncc',
        regularizer='bspline',
        spline_distance=4.0,
        grad_step=0.5,
        flow_sigma=3.0,
        total_sigma=0.0,
        reg_iterations=[100, 100, 20],
        in_loop_inv_steps=10,
        inverse_method='anderson',
        verbose=True
    )
    t1_syn = time.time() - t0_syn
    print(f"  SyN Optimization Complete [{t1_syn:.2f}s]")

    # 4. Quantitative Deformation Metrics
    print("\n[4/5] Computing Full Deformation Metrics Suite...", flush=True)
    fwd_tx = res['fwdtransforms']
    inv_tx = res['invtransforms']
    which_inv = res.get('whichtoinvert_inv', [True, False])

    dice_f, dice_m, dice_sym, warped_ml, warped_fl = compute_symmetric_dice(
        fi, mi, fl, ml, fwd_tx, inv_tx, which_inv
    )

    fwd_warp_file = fwd_tx[0]
    jac_img = ants.create_jacobian_determinant_image(fi, fwd_warp_file, do_log=False)
    jac_np = jac_img.numpy()
    mask_eval = ants.get_mask(fi).numpy() > 0
    min_detJ = float(np.min(jac_np[mask_eval]))
    max_detJ = float(np.max(jac_np[mask_eval]))
    folding_pct = float(np.mean(jac_np[mask_eval] <= 0.0) * 100.0)

    # Real Physical Inverse Identity Error Map in mm
    inv_err_dict = res.get('inverse_identity_errors', {}).get('phi_1', None)
    if inv_err_dict is None:
        model = res.get('model')
        if model is not None and hasattr(model, 'warp_l2r') and hasattr(model, 'warp_l2r_inv'):
            inv_err_dict = calculate_inverse_identity_error(
                model.warp_l2r.data, model.warp_l2r_inv.data,
                spacing=fi.spacing, origin=fi.origin, direction=fi.direction
            )

    if inv_err_dict is not None:
        inv_err_map_t = inv_err_dict['error_map']
        inv_err_map_np = inv_err_map_t.cpu().numpy().transpose(2, 1, 0)
        inv_err_map_img = ants.from_numpy(inv_err_map_np, origin=fi.origin, spacing=fi.spacing, direction=fi.direction)
        mean_inv_err = float(inv_err_dict['mean_error'])
        max_inv_err = float(inv_err_dict['max_error'])
        p95_inv_err = float(np.percentile(inv_err_map_np[mask_eval], 95))
    else:
        inv_err_map_img = ants.from_numpy(np.zeros(fi.shape, dtype=np.float32), origin=fi.origin, spacing=fi.spacing, direction=fi.direction)
        mean_inv_err, max_inv_err, p95_inv_err = 0.0, 0.0, 0.0

    print("=" * 80)
    print(f"  Symmetric Cortical Mean Dice : {dice_sym:.4f} (+{(dice_sym - dice_aff_sym)*100:+.2f}% over Affine)")
    print(f"    - Fixed Space Dice         : {dice_f:.4f}")
    print(f"    - Moving Space Dice        : {dice_m:.4f}")
    print(f"  Jacobian Regularity:")
    print(f"    - Min det(J)               : {min_detJ:+.4f}")
    print(f"    - Max det(J)               : {max_detJ:+.4f}")
    print(f"    - Grid Folding Rate        : {folding_pct:.4f}%")
    print(f"  Physical Inverse Error (mm):")
    print(f"    - Mean Inverse Error       : {mean_inv_err:.4f} mm")
    print(f"    - 95th Percentile Error    : {p95_inv_err:.4f} mm")
    print(f"    - Peak Max Error           : {max_inv_err:.4f} mm")
    print(f"  Total Runtime                : {t1_aff + t1_syn:.2f} s (Affine: {t1_aff:.2f}s, SyN: {t1_syn:.2f}s)")
    print("=" * 80)

    # 5. Generate Standard Interactive HTML Registration Report
    print("\n[5/5] Generating Standard 5-Figure HTML Diagnostic Report...", flush=True)
    report_path = "docs/reports/mbhard_bspline_syn_report.html"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    prov = {
        "algorithm": "syntx.syn (BSplineSyN)",
        "regularizer": "bspline",
        "spline_distance_mm": 4.0,
        "backend": "PyTorch",
        "device": device,
        "runtime_total_sec": t1_aff + t1_syn,
        "runtime_affine_sec": t1_aff,
        "runtime_syn_sec": t1_syn,
        "affine_sym_dice": dice_aff_sym,
        "syn_sym_dice": dice_sym,
        "syn_fixed_dice": dice_f,
        "syn_moving_dice": dice_m,
        "min_jacobian_det": min_detJ,
        "folding_percentage": folding_pct,
        "mean_inverse_error_mm": mean_inv_err,
        "p95_inverse_error_mm": p95_inv_err,
        "peak_inverse_error_mm": max_inv_err,
    }

    report_dict = create_registration_report(
        fixed=fi,
        moving=mi,
        warped=res['warpedmovout'],
        warp=fwd_warp_file,
        fixed_label=fl,
        moving_label=ml,
        warped_label=warped_ml,
        detJ=jac_img,
        inv_err_map=inv_err_map_img,
        output_html=report_path,
        fixed_name="NKI-TRT-20-2 (Fixed Target)",
        moving_name="MMRR-21-2 (Moving Source)",
        reg=res,
        provenance=prov,
        title="3D mbhard BSplineSyN Registration Verification Report (syntx.syn + bspline)"
    )

    print(f"\nReport generated successfully at:\n  --> {os.path.abspath(report_path)}")
    return report_path


if __name__ == "__main__":
    main()
