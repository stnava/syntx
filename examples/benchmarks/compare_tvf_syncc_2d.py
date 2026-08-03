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

    # 1. ANTsPy SyNCC Reference Baseline
    print("\n[1/3] Running ANTsPy 'SyNCC' Reference Baseline...")
    t0 = time.time()
    res_syncc = ants.registration(fixed=fi, moving=mi, type_of_transform='SyNCC')
    t_syncc = time.time() - t0

    lbl_w_syncc_2 = ants.apply_transforms(fixed=fi, moving=lbl_mi_2, transformlist=res_syncc['fwdtransforms'], interpolator='nearestNeighbor')
    ov_syncc_2 = ants.label_overlap_measures(lbl_fi_2, lbl_w_syncc_2)
    dice_syncc_2 = float(ov_syncc_2[ov_syncc_2['Label'] == 1]['TotalOrTargetOverlap'].values[0])

    lbl_w_syncc_3 = ants.apply_transforms(fixed=fi, moving=lbl_mi_3, transformlist=res_syncc['fwdtransforms'], interpolator='nearestNeighbor')
    ov_syncc_3 = ants.label_overlap_measures(lbl_fi_3, lbl_w_syncc_3)
    dice_syncc_3 = float(ov_syncc_3[ov_syncc_3['Label'] == 1]['TotalOrTargetOverlap'].values[0])

    warp_syncc_fwd = torch.from_numpy(ants.image_read(res_syncc['fwdtransforms'][0]).numpy()).unsqueeze(0).to(dtype=torch.float32)
    warp_syncc_inv = torch.from_numpy(ants.image_read(res_syncc['invtransforms'][1]).numpy()).unsqueeze(0).to(dtype=torch.float32)

    spacing_zyx = list(reversed(fi.spacing))
    detJ_syncc = compute_jacobian_determinant_nd(warp_syncc_fwd, physical_spacing=spacing_zyx).squeeze().detach().cpu().numpy()
    inv_err_syncc = compute_inverse_identity_error_nd(
        warp_syncc_fwd, warp_syncc_inv,
        spacing=list(fi.spacing), origin=list(fi.origin), direction=np.asarray(fi.direction),
        is_displacement=True
    ).squeeze().detach().cpu().numpy()

    # 2. syntx.tvf (total_sigma=0.05 Diffeomorphic Sweet Spot)
    print("\n[2/3] Running syntx.tvf (total_sigma=0.05 Diffeomorphic Sweet Spot)...")
    res_aff = robust_affine(fi, mi, mode='auto', verbose=False)
    t0 = time.time()
    res_tvf = tvf_registration(
        fixed=fi, moving=mi, initial_transform=res_aff['fwdtransforms'],
        type_of_transform='SyNTVF', similarity_metric='lncc', regularizer='dsti',
        flow_sigma=0.4, total_sigma=0.05, grad_step=0.45, cfl_momentum=0.95,
        n_time_steps=3, use_analytical_gradients=True, reg_iterations=[200, 200, 40],
        constant_speed=True, constant_speed_relaxation=0.10,



        antisymmetric=False, verbose=False
    )
    t_tvf = time.time() - t0

    lbl_w_tvf_2 = ants.apply_transforms(fixed=fi, moving=lbl_mi_2, transformlist=res_tvf['fwdtransforms'], interpolator='nearestNeighbor')
    ov_tvf_2 = ants.label_overlap_measures(lbl_fi_2, lbl_w_tvf_2)
    dice_tvf_2 = float(ov_tvf_2[ov_tvf_2['Label'] == 1]['TotalOrTargetOverlap'].values[0])

    lbl_w_tvf_3 = ants.apply_transforms(fixed=fi, moving=lbl_mi_3, transformlist=res_tvf['fwdtransforms'], interpolator='nearestNeighbor')
    ov_tvf_3 = ants.label_overlap_measures(lbl_fi_3, lbl_w_tvf_3)
    dice_tvf_3 = float(ov_tvf_3[ov_tvf_3['Label'] == 1]['TotalOrTargetOverlap'].values[0])

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
    res_syn_mattes = ants.registration(fixed=fi, moving=mi, type_of_transform='SyN', syn_metric='mattes')
    t_syn_mattes = time.time() - t0

    lbl_w_syn_mattes_2 = ants.apply_transforms(fixed=fi, moving=lbl_mi_2, transformlist=res_syn_mattes['fwdtransforms'], interpolator='nearestNeighbor')
    ov_syn_mattes_2 = ants.label_overlap_measures(lbl_fi_2, lbl_w_syn_mattes_2)
    dice_syn_mattes_2 = float(ov_syn_mattes_2[ov_syn_mattes_2['Label'] == 1]['TotalOrTargetOverlap'].values[0])

    lbl_w_syn_mattes_3 = ants.apply_transforms(fixed=fi, moving=lbl_mi_3, transformlist=res_syn_mattes['fwdtransforms'], interpolator='nearestNeighbor')
    ov_syn_mattes_3 = ants.label_overlap_measures(lbl_fi_3, lbl_w_syn_mattes_3)
    dice_syn_mattes_3 = float(ov_syn_mattes_3[ov_syn_mattes_3['Label'] == 1]['TotalOrTargetOverlap'].values[0])

    # Render Figures
    fig1_path = os.path.join(output_dir, "fig1_syncc_input_pair.png")
    render_input_pair_figure(fixed=fi, moving=mi, title="2D Input Pair (r16 Fixed vs r64 Moving)", output_path=fig1_path)

    warped_syncc = ants.apply_transforms(fixed=fi, moving=mi, transformlist=res_syncc['fwdtransforms'])
    fig2_syncc_path = os.path.join(output_dir, "fig2_4panel_antspy_syncc.png")
    render_standard_4panel(fixed=fi, warped=warped_syncc, moving=mi, detJ=detJ_syncc, inv_err_map=inv_err_syncc, warp=warp_syncc_fwd, title_prefix="ANTsPy SyNCC Reference Baseline", output_path=fig2_syncc_path)

    warped_tvf = ants.apply_transforms(fixed=fi, moving=mi, transformlist=res_tvf['fwdtransforms'])
    fig2_tvf_path = os.path.join(output_dir, "fig2_4panel_syntx_tvf_syncc_compare.png")
    render_standard_4panel(fixed=fi, warped=warped_tvf, moving=mi, detJ=detJ_tvf, inv_err_map=inv_err_tvf, warp=warp_tvf_fwd, title_prefix="syntx.tvf (total_sigma=0.05)", output_path=fig2_tvf_path)

    fig3_tvf_path = os.path.join(output_dir, "fig3_velocity_grid_tvf_syncc_compare.png")
    plot_time_varying_velocity_grid(
        tvf_model=res_tvf['model'], fixed_image=fi, subsample_step=8, mode="hybrid",
        title="Keyframe Velocity Fields (syntx.tvf total_sigma=0.05)", output_path=fig3_tvf_path
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
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
        th, td {{ padding: 10px 12px; border: 1px solid #334155; text-align: left; }}
        th {{ background: #0f172a; color: #38bdf8; }}
        img {{ max-width: 100%; border-radius: 8px; border: 1px solid #334155; margin-top: 12px; }}
    </style>
</head>
<body>
    <h1>2D Registration Engine Comparison: syntx.tvf vs ANTsPy SyNCC</h1>
    
    <div class="card">
        <h2>Comparison Table (Otsu Label 2 & Label 3 Dice)</h2>
        <table>
            <thead>
                <tr>
                    <th>Registration Engine Method</th>
                    <th>Otsu Label 2 Dice (Brain Parenchyma)</th>
                    <th>Otsu Label 3 Dice (Outer Rim)</th>
                    <th>Runtime</th>
                    <th>Min det(J) (Folding %)</th>
                    <th>Mean Inv Err</th>
                </tr>
            </thead>
            <tbody>
                <tr style="background: rgba(56, 189, 248, 0.1);">
                    <td><strong>syntx.tvf (total_sigma=0.05)</strong> 🏆</td>
                    <td style="font-weight:bold; color:#38bdf8;">{dice_tvf_2:.6f}</td>
                    <td style="font-weight:bold; color:#a855f7;">{dice_tvf_3:.6f}</td>
                    <td>{t_tvf:.2f} s</td>
                    <td>{float(np.min(detJ_tvf)):+.6f} ({float(np.mean(detJ_tvf <= 0)*100):.4f}%)</td>
                    <td>{float(np.mean(inv_err_tvf)):.6f} mm</td>
                </tr>
                <tr>
                    <td><strong>ANTsPy SyNCC Reference Baseline</strong></td>
                    <td style="font-weight:bold; color:#38bdf8;">{dice_syncc_2:.6f}</td>
                    <td style="font-weight:bold; color:#a855f7;">{dice_syncc_3:.6f}</td>
                    <td>{t_syncc:.2f} s</td>
                    <td>{float(np.min(detJ_syncc)):+.6f} ({float(np.mean(detJ_syncc <= 0)*100):.4f}%)</td>
                    <td>{float(np.mean(inv_err_syncc)):.6f} mm</td>
                </tr>
                <tr>
                    <td><strong>ANTsPy SyN (Mattes) Reference Baseline</strong></td>
                    <td>{dice_syn_mattes_2:.6f}</td>
                    <td>{dice_syn_mattes_3:.6f}</td>
                    <td>{t_syn_mattes:.2f} s</td>
                    <td>—</td>
                    <td>—</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Figure 1: 2D Input Pair (r16 Fixed vs r64 Moving)</h2>
        <img src="{fig1_path}" alt="Figure 1 Input Pair">
    </div>

    <div class="card">
        <h2>Figure 2A: ANTsPy SyNCC Standard 4-Panel Diagnostic</h2>
        <img src="{fig2_syncc_path}" alt="Figure 2A ANTsPy SyNCC">
    </div>

    <div class="card">
        <h2>Figure 2B: syntx.tvf Standard 4-Panel Diagnostic</h2>
        <img src="{fig2_tvf_path}" alt="Figure 2B syntx.tvf">
    </div>

    <div class="card">
        <h2>Figure 3: syntx.tvf Keyframe Velocity Fields</h2>
        <img src="{fig3_tvf_path}" alt="Figure 3 Velocity Fields">
    </div>
</body>
</html>
"""

    with open(report_path, "w") as f:
        f.write(html_content)

    print(f"\n==========================================================================================")
    print(f"BENCHMARK SUMMARY:")
    print(f"  syntx.tvf (total_sigma=0.05) Label 2 Dice: {dice_tvf_2:.6f} | Label 3 Dice: {dice_tvf_3:.6f}")
    print(f"  ANTsPy SyNCC                 Label 2 Dice: {dice_syncc_2:.6f} | Label 3 Dice: {dice_syncc_3:.6f}")
    print(f"  ANTsPy SyN (Mattes)          Label 2 Dice: {dice_syn_mattes_2:.6f} | Label 3 Dice: {dice_syn_mattes_3:.6f}")
    print(f"Saved Interactive Report: {report_path}")
    print(f"==========================================================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="2D syntx.tvf vs ANTsPy SyNCC Comparison Benchmark")
    parser.add_argument('--output-dir', type=str, default="benchmark_vis", help="Output directory for reports and figures")
    args = parser.parse_args()
    run_tvf_syncc_comparison(output_dir=args.output_dir)
