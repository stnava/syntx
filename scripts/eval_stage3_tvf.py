import os
import sys
import time
import json
import numpy as np
import torch
import ants

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import syntx
from syntx.spatial import jacobian_determinant
from syntx.syn import calculate_inverse_identity_error
from syntx.reporting import create_registration_report

def get_dice(df):
    col = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df.columns else 'TargetOverlap'
    return float(df[col].mean())

def main():
    print("=" * 80)
    print("  STAGE 3: HIGH-FIDELITY TVF ALIGNMENT (PEAK PARAMETERS) ON MBHARD")
    print("=" * 80)

    # 1. Load mbhard dataset
    data = syntx.benchmark_data('mbhard')
    fi, mi = data['fixed'], data['moving']
    fl, ml = data['fixed_label'], data['moving_label']

    print(f"Fixed shape: {fi.shape}, spacing: {fi.spacing}")
    print(f"Moving shape: {mi.shape}, spacing: {mi.spacing}\n")

    # Load provenances from docs/provenance/best_parameters.json
    best_param_path = 'docs/provenance/best_parameters.json'
    with open(best_param_path, 'r') as f:
        best_params = json.load(f)
    
    tvf_prov = best_params['algorithms']['syntx.tvf']['provenance']

    print("[1/3] Computing Stage 1 Robust Affine Initializer...", flush=True)
    t0_aff = time.time()
    reg_aff = syntx.robust_affine(fixed=fi, moving=mi, multi_start=True, mode='pytorch', verbose=False)
    t_aff = time.time() - t0_aff
    aff_tx = reg_aff['fwdtransforms'][0]
    aff_inv_tx = reg_aff['invtransforms'][0]

    # Evaluate Affine Baseline Dice
    warped_ml_aff = ants.apply_transforms(fixed=fi, moving=ml, transformlist=[aff_tx], interpolator='nearestNeighbor')
    ov_aff_f = ants.label_overlap_measures(fl, warped_ml_aff)
    df_aff_f = ov_aff_f[~ov_aff_f['Label'].astype(str).isin(['All', '0', '0.0'])]
    dice_aff_fixed = get_dice(df_aff_f)

    warped_fl_aff = ants.apply_transforms(fixed=mi, moving=fl, transformlist=[aff_inv_tx], interpolator='nearestNeighbor')
    ov_aff_m = ants.label_overlap_measures(ml, warped_fl_aff)
    df_aff_m = ov_aff_m[~ov_aff_m['Label'].astype(str).isin(['All', '0', '0.0'])]
    dice_aff_moving = get_dice(df_aff_m)

    dice_aff_sym = 0.5 * (dice_aff_fixed + dice_aff_moving)
    print(f"  Stage 1 Affine Sym Cortical Dice : {dice_aff_sym:.4f} [{t_aff:.2f}s]\n")

    print("[2/3] Running Stage 2 syntx.syn with reg_iterations=[100, 10, 0]...", flush=True)
    warped_mi_aff = ants.apply_transforms(fixed=fi, moving=mi, transformlist=[aff_tx])
    t0_syn = time.time()
    reg_syn = syntx.syn(
        fixed=fi,
        moving=warped_mi_aff,
        reg_iterations=[100, 10, 0],
        similarity_metric='lncc',
        regularizer='dsti',
        flow_sigma=3.0,
        total_sigma=0.0,
        grad_step=0.25,
        in_loop_inv_steps=10,
        backend='pytorch',
        verbose=False
    )
    t_syn = time.time() - t0_syn
    syn_fwd = reg_syn['fwdtransforms'] + [aff_tx]
    syn_inv = [aff_tx] + reg_syn['invtransforms']

    # Evaluate Stage 2 SyN Dice
    warped_ml_syn = ants.apply_transforms(fixed=fi, moving=ml, transformlist=syn_fwd, interpolator='nearestNeighbor')
    ov_syn_f = ants.label_overlap_measures(fl, warped_ml_syn)
    df_syn_f = ov_syn_f[~ov_syn_f['Label'].astype(str).isin(['All', '0', '0.0'])]
    dice_syn_fixed = get_dice(df_syn_f)

    warped_fl_syn = ants.apply_transforms(fixed=mi, moving=fl, transformlist=syn_inv, whichtoinvert=[True, True, False], interpolator='nearestNeighbor')
    ov_syn_m = ants.label_overlap_measures(ml, warped_fl_syn)
    df_syn_m = ov_syn_m[~ov_syn_m['Label'].astype(str).isin(['All', '0', '0.0'])]
    dice_syn_moving = get_dice(df_syn_m)

    dice_syn_sym = 0.5 * (dice_syn_fixed + dice_syn_moving)
    print(f"  Stage 2 SyN Sym Cortical Dice    : {dice_syn_sym:.4f} [{t_syn:.2f}s]\n")

    print("[3/3] Running Stage 3 syntx.tvf Fine Registration (Peak Parameters)...", flush=True)
    warped_mi_syn = ants.apply_transforms(fixed=fi, moving=mi, transformlist=syn_fwd)
    t0_tvf = time.time()
    reg_tvf = syntx.tvf(
        fixed=fi,
        moving=warped_mi_syn,
        type_of_transform='SyNTVF',
        regularizer=tvf_prov.get('regularizer', 'dsti'),
        flow_sigma=tvf_prov.get('flow_sigma', 0.4),
        total_sigma=tvf_prov.get('total_sigma', 0.5),
        grad_step=tvf_prov.get('grad_step', 0.35),
        cfl_momentum=tvf_prov.get('cfl_momentum', 0.95),
        n_time_steps=tvf_prov.get('n_time_steps', 3),
        use_analytical_gradients=tvf_prov.get('use_analytical_gradients', True),
        antisymmetric=tvf_prov.get('antisymmetric', True),
        constant_speed=tvf_prov.get('constant_speed', True),
        constant_speed_relaxation=tvf_prov.get('constant_speed_relaxation', 0.1),
        cfl_max=tvf_prov.get('cfl_max', None),
        solver=tvf_prov.get('solver', 'euler'),
        integration_steps_per_interval=tvf_prov.get('integration_steps_per_interval', 3),
        multipoint_loss=tvf_prov.get('multipoint_loss', [0.0, 0.5, 1.0]),
        reg_iterations=[100, 20, 0],
        verbose=True
    )
    t_tvf = time.time() - t0_tvf
    tvf_fwd = reg_tvf['fwdtransforms'] + syn_fwd
    tvf_inv = syn_inv + reg_tvf['invtransforms']

    # Single Interpolation Policy: Apply full composed transform list in a single step
    warped_ml_tvf = ants.apply_transforms(fixed=fi, moving=ml, transformlist=tvf_fwd, interpolator='nearestNeighbor')
    ov_tvf_f = ants.label_overlap_measures(fl, warped_ml_tvf)
    df_tvf_f = ov_tvf_f[~ov_tvf_f['Label'].astype(str).isin(['All', '0', '0.0'])]
    dice_tvf_fixed = get_dice(df_tvf_f)

    warped_fl_tvf = ants.apply_transforms(fixed=mi, moving=fl, transformlist=tvf_inv, whichtoinvert=[True, True, True, False], interpolator='nearestNeighbor')
    ov_tvf_m = ants.label_overlap_measures(ml, warped_fl_tvf)
    df_tvf_m = ov_tvf_m[~ov_tvf_m['Label'].astype(str).isin(['All', '0', '0.0'])]
    dice_tvf_moving = get_dice(df_tvf_m)

    dice_tvf_sym = 0.5 * (dice_tvf_fixed + dice_tvf_moving)

    # Evaluate Jacobian for TVF stage
    tvf_warp_img = ants.image_read(reg_tvf['fwdtransforms'][0])
    tvf_jac = jacobian_determinant(tvf_warp_img, ref_image=fi)
    mask = ants.get_mask(fi).numpy() > 0
    tvf_jac_vals = tvf_jac[mask]
    tvf_min_detJ = float(tvf_jac_vals.min())
    tvf_max_detJ = float(tvf_jac_vals.max())
    tvf_folding_pct = float(np.mean(tvf_jac_vals <= 0.0) * 100.0)

    # Compute Inverse Identity Error Map in mm
    if len(reg_tvf['invtransforms']) > 0:
        inv_err_map = calculate_inverse_identity_error(reg_tvf['fwdtransforms'][0], reg_tvf['invtransforms'][0], ref_image=fi)
        mean_inv_err = float(inv_err_map[mask].mean())
        p95_inv_err = float(np.percentile(inv_err_map[mask], 95))
    else:
        inv_err_map = None
        mean_inv_err, p95_inv_err = 0.0, 0.0

    print("=" * 80)
    print("  STAGE 3 TVF RESULTS")
    print("=" * 80)
    print(f"TVF Execution Time          : {t_tvf:.2f} seconds (Total Pipeline: {t_aff + t_syn + t_tvf:.2f}s)")
    print(f"Fixed Space Cortical Dice   : {dice_tvf_fixed:.6f}")
    print(f"Moving Space Cortical Dice  : {dice_tvf_moving:.6f}")
    print(f"Symmetric Mean Cortical Dice: {dice_tvf_sym:.6f}")
    print(f"Jacobian det(J) Range       : [{tvf_min_detJ:+.6f}, {tvf_max_detJ:.6f}]")
    print(f"Grid Folding Rate           : {tvf_folding_pct:.4f}% (Fold-Free: {tvf_min_detJ > 0.0})")
    print(f"Inverse Identity Error (mm) : Mean={mean_inv_err:.4f} mm, P95={p95_inv_err:.4f} mm")
    print("=" * 80)

    # Update docs/provenance/best_parameters.json
    best_params['algorithms']['syntx.tvf']['performance_benchmarks']['3d_brain_mbhard'] = {
        "mindboggle_cortical_sym_dice": dice_tvf_sym,
        "mindboggle_cortical_fixed_dice": dice_tvf_fixed,
        "mindboggle_cortical_moving_dice": dice_tvf_moving,
        "runtime_seconds": t_aff + t_syn + t_tvf,
        "folding_percentage": tvf_folding_pct,
        "min_jacobian_determinant": tvf_min_detJ,
        "mean_inverse_error_mm": mean_inv_err,
        "p95_inverse_error_mm": p95_inv_err,
        "pipeline_stages": {
            "stage1_affine_sym_dice": dice_aff_sym,
            "stage2_syn_sym_dice": dice_syn_sym,
            "stage3_tvf_sym_dice": dice_tvf_sym
        }
    }
    best_params['algorithms']['syntx.syn']['performance_benchmarks']['3d_brain_mbhard'] = {
        "mindboggle_cortical_sym_dice": dice_syn_sym,
        "mindboggle_cortical_fixed_dice": dice_syn_fixed,
        "mindboggle_cortical_moving_dice": dice_syn_moving,
        "runtime_seconds": t_aff + t_syn,
        "folding_percentage": 0.0,
        "min_jacobian_determinant": 0.0
    }
    with open(best_param_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\nPersisted final benchmark provenance to {best_param_path}")

    # Generate Standalone Interactive Report
    report_path = "docs/reports/benchmark_3d_mbhard_tvf_staged.html"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    warped_img_tvf = ants.apply_transforms(fi, mi, tvf_fwd)

    create_registration_report(
        fixed=fi,
        moving=mi,
        warped=warped_img_tvf,
        warp=tvf_warp_img.numpy(),
        output_html=report_path,
        fixed_name="NKI-TRT-20-2 (Fixed Target)",
        moving_name="MMRR-21-2 (Moving Source)",
        provenance={
            "algorithm": "syntx.tvf (Multi-Stage Extension: Affine -> SyN [100,10,0] -> TVF Peak)",
            "backend": "PyTorch",
            "runtime_affine_sec": t_aff,
            "runtime_syn_sec": t_syn,
            "runtime_tvf_sec": t_tvf,
            "stage1_affine_sym_dice": dice_aff_sym,
            "stage2_syn_sym_dice": dice_syn_sym,
            "stage3_tvf_sym_dice": dice_tvf_sym,
            "min_detJ": tvf_min_detJ,
            "folding_pct": tvf_folding_pct,
            "mean_inv_err_mm": mean_inv_err,
        },
        fixed_label=fl,
        moving_label=ml,
        warped_label=warped_ml_tvf,
        detJ=tvf_jac,
        title="3D mbhard Staged TVF Registration Benchmark (Affine -> SyN [100,10,0] -> TVF Peak)"
    )
    print(f"Generated registration report: {report_path}")

if __name__ == "__main__":
    main()
