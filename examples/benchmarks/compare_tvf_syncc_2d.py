#!/usr/bin/env python
"""
2D Registration Engine Comparison: syntx.tvf vs ANTsPy SyNCC Reference Baseline.

Evaluates Time-Varying Velocity Field (SyNTVF) registration against ANTsPy SyNCC
on 2D r16 (fixed) vs r64 (moving) benchmark images across both Otsu Label 2
(Primary Brain Parenchyma) and Otsu Label 3 (Outer Cortical Rim) segmentations.
"""

import os
import time
import argparse
import numpy as np
import torch
import ants

import syntx
from syntx.robust_affine import robust_affine
from syntx.tvf import tvf_registration
from syntx.syn import compute_jacobian_determinant_nd, compute_inverse_identity_error_nd
from syntx.viz import render_input_pair_figure, render_standard_4panel, plot_time_varying_velocity_grid


def run_tvf_syncc_comparison(output_dir="benchmark_vis"):
    os.makedirs(output_dir, exist_ok=True)

    # Pin random seeds for 100% deterministic execution reproducibility across environments
    torch.manual_seed(42)
    np.random.seed(42)

    fi = ants.image_read(ants.get_data('r16'))

    mi = ants.image_read(ants.get_data('r64'))

    # Otsu 3-class segmentation
    otsu_fi = ants.threshold_image(fi, 'Otsu', 3)
    otsu_mi = ants.threshold_image(mi, 'Otsu', 3)

    # Label 2: Primary Brain Parenchyma
    lbl_fi_2 = ants.threshold_image(otsu_fi, 2, 2)
    lbl_mi_2 = ants.threshold_image(otsu_mi, 2, 2)

    # Label 3: Outer Cortical Rim
    lbl_fi_3 = ants.threshold_image(otsu_fi, 3, 3)
    lbl_mi_3 = ants.threshold_image(otsu_mi, 3, 3)

    print("==========================================================================================")
    print("2D REGISTRATION ENGINE COMPARISON: syntx.tvf vs ANTsPy SyNCC")
    print("==========================================================================================")

    # Helper for bidirectional evaluation
    def evaluate_bidirectional_dice(lbl_fi, lbl_mi, target_img, source_img, fwdtransforms, invtransforms, whichtoinvert=None):
        lbl_w_fwd = ants.apply_transforms(fixed=target_img, moving=lbl_mi, transformlist=fwdtransforms, interpolator='nearestNeighbor')
        ov_fwd = ants.label_overlap_measures(lbl_fi, lbl_w_fwd)
        dice_fixed = float(ov_fwd[ov_fwd['Label'] == 1]['TotalOrTargetOverlap'].values[0]) if 1 in ov_fwd['Label'].values else 0.0

        lbl_w_inv = ants.apply_transforms(fixed=source_img, moving=lbl_fi, transformlist=invtransforms, whichtoinvert=whichtoinvert, interpolator='nearestNeighbor')
        ov_inv = ants.label_overlap_measures(lbl_mi, lbl_w_inv)
        dice_moving = float(ov_inv[ov_inv['Label'] == 1]['TotalOrTargetOverlap'].values[0]) if 1 in ov_inv['Label'].values else 0.0

        dice_sym = 0.5 * (dice_fixed + dice_moving)
        return dice_fixed, dice_moving, dice_sym

    # 1. ANTsPy SyNCC Reference Baseline
    print("\n[1/3] Running ANTsPy 'SyNCC' Reference Baseline...")
    t0 = time.time()
    res_syncc = ants.registration(fixed=fi, moving=mi, type_of_transform='SyNCC', random_seed=42)
    t_syncc = time.time() - t0

    dice_syncc_2_fix, dice_syncc_2_mov, dice_syncc_2_sym = evaluate_bidirectional_dice(lbl_fi_2, lbl_mi_2, fi, mi, res_syncc['fwdtransforms'], res_syncc['invtransforms'])
    dice_syncc_3_fix, dice_syncc_3_mov, dice_syncc_3_sym = evaluate_bidirectional_dice(lbl_fi_3, lbl_mi_3, fi, mi, res_syncc['fwdtransforms'], res_syncc['invtransforms'])

    warp_syncc_fwd = torch.from_numpy(ants.image_read(res_syncc['fwdtransforms'][0]).numpy()).unsqueeze(0).to(dtype=torch.float32)
    warp_syncc_inv = torch.from_numpy(ants.image_read(res_syncc['invtransforms'][1]).numpy()).unsqueeze(0).to(dtype=torch.float32)

    spacing_zyx = list(reversed(fi.spacing))
    detJ_syncc = compute_jacobian_determinant_nd(warp_syncc_fwd, physical_spacing=spacing_zyx).squeeze().detach().cpu().numpy()
    inv_err_syncc = compute_inverse_identity_error_nd(
        warp_syncc_fwd, warp_syncc_inv,
        spacing=list(fi.spacing), origin=list(fi.origin), direction=np.asarray(fi.direction),
        is_displacement=True
    ).squeeze().detach().cpu().numpy()

    # 2. syntx.tvf (Optimal Configuration: syn_sampling=3, total_sigma=0.04)
    print("\n[2/3] Running syntx.tvf (Optimal Diffeomorphic Configuration)...")
    res_aff = robust_affine(fi, mi, mode='pytorch', verbose=False)
    t0 = time.time()
    res_tvf = tvf_registration(
        fixed=fi, moving=mi, initial_transform=res_aff['fwdtransforms'],
        type_of_transform='SyNTVF', similarity_metric='lncc', regularizer='dsti',
        flow_sigma=0.4, total_sigma=0.04, grad_step=0.45, cfl_momentum=0.95,
        syn_sampling=3, n_time_steps=3, use_analytical_gradients=True,
        reg_iterations=[250, 250, 60], constant_speed=True, constant_speed_relaxation=0.10,
        antisymmetric=False, verbose=False
    )
    t_tvf = time.time() - t0

    dice_tvf_2_fix, dice_tvf_2_mov, dice_tvf_2_sym = evaluate_bidirectional_dice(lbl_fi_2, lbl_mi_2, fi, mi, res_tvf['fwdtransforms'], res_tvf['invtransforms'], res_tvf.get('whichtoinvert_inv'))
    dice_tvf_3_fix, dice_tvf_3_mov, dice_tvf_3_sym = evaluate_bidirectional_dice(lbl_fi_3, lbl_mi_3, fi, mi, res_tvf['fwdtransforms'], res_tvf['invtransforms'], res_tvf.get('whichtoinvert_inv'))

    warp_tvf_fwd = torch.from_numpy(ants.image_read(res_tvf['fwdtransforms'][0]).numpy()).unsqueeze(0).to(dtype=torch.float32)
    warp_tvf_inv = torch.from_numpy(ants.image_read(res_tvf['invtransforms'][1]).numpy()).unsqueeze(0).to(dtype=torch.float32)
    detJ_tvf = compute_jacobian_determinant_nd(warp_tvf_fwd, physical_spacing=spacing_zyx).squeeze().detach().cpu().numpy()
    inv_err_tvf = compute_inverse_identity_error_nd(
        warp_tvf_fwd, warp_tvf_inv,
        spacing=list(fi.spacing), origin=list(fi.origin), direction=np.asarray(fi.direction),
        is_displacement=True
    ).squeeze().detach().cpu().numpy()

    # 3. ANTsPy SyN (Mattes) Reference Baseline
    print("\n[3/3] Running ANTsPy 'SyN' (Mattes) Reference Baseline...")
    t0 = time.time()
    res_syn_mattes = ants.registration(fixed=fi, moving=mi, type_of_transform='SyN', syn_metric='mattes', random_seed=42)
    t_syn_mattes = time.time() - t0

    dice_syn_mattes_2_fix, dice_syn_mattes_2_mov, dice_syn_mattes_2_sym = evaluate_bidirectional_dice(lbl_fi_2, lbl_mi_2, fi, mi, res_syn_mattes['fwdtransforms'], res_syn_mattes['invtransforms'])
    dice_syn_mattes_3_fix, dice_syn_mattes_3_mov, dice_syn_mattes_3_sym = evaluate_bidirectional_dice(lbl_fi_3, lbl_mi_3, fi, mi, res_syn_mattes['fwdtransforms'], res_syn_mattes['invtransforms'])

    # Render Figures
    fig1_path = os.path.join(output_dir, "fig1_syncc_input_pair.png")
    render_input_pair_figure(fixed=fi, moving=mi, title="2D Input Pair (r16 Fixed vs r64 Moving)", output_path=fig1_path)

    warped_syncc = ants.apply_transforms(fixed=fi, moving=mi, transformlist=res_syncc['fwdtransforms'])
    fig2_syncc_path = os.path.join(output_dir, "fig2_4panel_antspy_syncc.png")
    render_standard_4panel(fixed=fi, warped=warped_syncc, moving=mi, detJ=detJ_syncc, inv_err_map=inv_err_syncc, warp=warp_syncc_fwd, title_prefix="ANTsPy SyNCC Reference Baseline", output_path=fig2_syncc_path)

    warped_tvf = ants.apply_transforms(fixed=fi, moving=mi, transformlist=res_tvf['fwdtransforms'])
    fig2_tvf_path = os.path.join(output_dir, "fig2_4panel_syntx_tvf_syncc_compare.png")
    render_standard_4panel(fixed=fi, warped=warped_tvf, moving=mi, detJ=detJ_tvf, inv_err_map=inv_err_tvf, warp=warp_tvf_fwd, title_prefix="syntx.tvf (total_sigma=0.04)", output_path=fig2_tvf_path)

    fig3_tvf_path = os.path.join(output_dir, "fig3_velocity_grid_tvf_syncc_compare.png")
    plot_time_varying_velocity_grid(
        tvf_model=res_tvf['model'], fixed_image=fi, subsample_step=8, mode="hybrid",
        title="Keyframe Velocity Fields (syntx.tvf total_sigma=0.04)", output_path=fig3_tvf_path
    )

    report_path = os.path.join(output_dir, "compare_tvf_syncc_report.html")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>2D Registration Comparison: syntx.tvf vs ANTsPy SyNCC</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        h1 {{ color: #38bdf8; border-bottom: 2px solid #334155; padding-bottom: 8px; }}
        h2 {{ color: #a855f7; margin-top: 24px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 13px; }}
        th, td {{ padding: 8px 10px; border: 1px solid #334155; text-align: left; }}
        th {{ background: #0f172a; color: #38bdf8; }}
        img {{ max-width: 100%; border-radius: 8px; border: 1px solid #334155; margin-top: 12px; }}
    </style>
</head>
<body>
    <h1>2D Registration Engine Comparison: syntx.tvf vs ANTsPy SyNCC</h1>
    
    <div class="card">
        <h2>Bidirectional Overlap Evaluation (Fixed Space vs Moving Space)</h2>
        <table>
            <thead>
                <tr>
                    <th rowspan="2">Registration Engine</th>
                    <th colspan="3">Label 2 (Brain Parenchyma)</th>
                    <th colspan="3">Label 3 (Outer Cortical Rim)</th>
                    <th rowspan="2">Symmetric Mean</th>
                    <th rowspan="2">Runtime</th>
                    <th rowspan="2">Min det(J) (Folding %)</th>
                    <th rowspan="2">Mean Inv Err</th>
                </tr>
                <tr>
                    <th>Fixed</th>
                    <th>Moving</th>
                    <th>Sym Mean</th>
                    <th>Fixed</th>
                    <th>Moving</th>
                    <th>Sym Mean</th>
                </tr>
            </thead>
            <tbody>
                <tr style="background: rgba(56, 189, 248, 0.1);">
                    <td><strong>syntx.tvf (syn_sampling=3)</strong> 🏆</td>
                    <td>{dice_tvf_2_fix:.6f}</td>
                    <td>{dice_tvf_2_mov:.6f}</td>
                    <td style="font-weight:bold; color:#38bdf8;">{dice_tvf_2_sym:.6f}</td>
                    <td>{dice_tvf_3_fix:.6f}</td>
                    <td>{dice_tvf_3_mov:.6f}</td>
                    <td style="font-weight:bold; color:#a855f7;">{dice_tvf_3_sym:.6f}</td>
                    <td style="font-weight:bold; color:#f43f5e;">{0.5*(dice_tvf_2_sym+dice_tvf_3_sym):.6f}</td>
                    <td>{t_tvf:.2f} s</td>
                    <td>{float(np.min(detJ_tvf)):+.6f} ({float(np.mean(detJ_tvf <= 0)*100):.4f}%)</td>
                    <td>{float(np.mean(inv_err_tvf)):.6f} mm</td>
                </tr>
                <tr>
                    <td><strong>ANTsPy SyNCC Reference</strong></td>
                    <td>{dice_syncc_2_fix:.6f}</td>
                    <td>{dice_syncc_2_mov:.6f}</td>
                    <td style="font-weight:bold; color:#38bdf8;">{dice_syncc_2_sym:.6f}</td>
                    <td>{dice_syncc_3_fix:.6f}</td>
                    <td>{dice_syncc_3_mov:.6f}</td>
                    <td style="font-weight:bold; color:#a855f7;">{dice_syncc_3_sym:.6f}</td>
                    <td style="font-weight:bold; color:#f43f5e;">{0.5*(dice_syncc_2_sym+dice_syncc_3_sym):.6f}</td>
                    <td>{t_syncc:.2f} s</td>
                    <td>{float(np.min(detJ_syncc)):+.6f} ({float(np.mean(detJ_syncc <= 0)*100):.4f}%)</td>
                    <td>{float(np.mean(inv_err_syncc)):.6f} mm</td>
                </tr>
                <tr>
                    <td><strong>ANTsPy SyN (Mattes)</strong></td>
                    <td>{dice_syn_mattes_2_fix:.6f}</td>
                    <td>{dice_syn_mattes_2_mov:.6f}</td>
                    <td>{dice_syn_mattes_2_sym:.6f}</td>
                    <td>{dice_syn_mattes_3_fix:.6f}</td>
                    <td>{dice_syn_mattes_3_mov:.6f}</td>
                    <td>{dice_syn_mattes_3_sym:.6f}</td>
                    <td>{0.5*(dice_syn_mattes_2_sym+dice_syn_mattes_3_sym):.6f}</td>
                    <td>{t_syn_mattes:.2f} s</td>
                    <td>—</td>
                    <td>—</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Figure 1: Input Image Pair</h2>
        <img src="fig1_syncc_input_pair.png" alt="Figure 1 Input Pair">
    </div>

    <div class="card">
        <h2>Figure 2: syntx.tvf 4-Panel Diagnostic Report</h2>
        <img src="fig2_4panel_syntx_tvf_syncc_compare.png" alt="Figure 2 syntx.tvf 4-Panel Report">
    </div>

    <div class="card">
        <h2>Figure 2 Reference: ANTsPy SyNCC 4-Panel Diagnostic Report</h2>
        <img src="fig2_4panel_antspy_syncc.png" alt="Figure 2 ANTsPy SyNCC 4-Panel Report">
    </div>

    <div class="card">
        <h2>Figure 3: Keyframe Velocity Field Flow Visualization (syntx.tvf)</h2>
        <img src="fig3_velocity_grid_tvf_syncc_compare.png" alt="Figure 3 Velocity Grid">
    </div>
</body>
</html>
"""

    with open(report_path, "w") as f:
        f.write(html_content)

    print("\n==========================================================================================")
    print("BIDIRECTIONAL BENCHMARK SUMMARY (Fixed Space | Moving Space | Symmetric Mean):")
    print(f"  syntx.tvf (syn_sampling=3):")
    print(f"    Label 2 (Parenchyma): Fixed={dice_tvf_2_fix:.6f} | Moving={dice_tvf_2_mov:.6f} | Sym={dice_tvf_2_sym:.6f}")
    print(f"    Label 3 (Outer Rim) : Fixed={dice_tvf_3_fix:.6f} | Moving={dice_tvf_3_mov:.6f} | Sym={dice_tvf_3_sym:.6f}")
    print(f"    Overall Symmetric Mean Dice: {0.5*(dice_tvf_2_sym+dice_tvf_3_sym):.6f}")
    print(f"  ANTsPy SyNCC Reference:")
    print(f"    Label 2 (Parenchyma): Fixed={dice_syncc_2_fix:.6f} | Moving={dice_syncc_2_mov:.6f} | Sym={dice_syncc_2_sym:.6f}")
    print(f"    Label 3 (Outer Rim) : Fixed={dice_syncc_3_fix:.6f} | Moving={dice_syncc_3_mov:.6f} | Sym={dice_syncc_3_sym:.6f}")
    print(f"    Overall Symmetric Mean Dice: {0.5*(dice_syncc_2_sym+dice_syncc_3_sym):.6f}")
    print(f"  ANTsPy SyN (Mattes):")
    print(f"    Label 2 (Parenchyma): Fixed={dice_syn_mattes_2_fix:.6f} | Moving={dice_syn_mattes_2_mov:.6f} | Sym={dice_syn_mattes_2_sym:.6f}")
    print(f"    Label 3 (Outer Rim) : Fixed={dice_syn_mattes_3_fix:.6f} | Moving={dice_syn_mattes_3_mov:.6f} | Sym={dice_syn_mattes_3_sym:.6f}")
    print(f"    Overall Symmetric Mean Dice: {0.5*(dice_syn_mattes_2_sym+dice_syn_mattes_3_sym):.6f}")
    print(f"Saved Interactive Report: {report_path}")
    print("==========================================================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="2D Registration Engine Comparison: syntx.tvf vs ANTsPy SyNCC")
    parser.add_argument("--output_dir", default="benchmark_vis", help="Directory to save report artifacts")
    args = parser.parse_args()
    run_tvf_syncc_comparison(output_dir=args.output_dir)
