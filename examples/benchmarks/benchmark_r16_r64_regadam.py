#!/usr/bin/env python3
"""
Benchmark: RegAdam vs CFL in syntx.syn on 2D r16 -> r64
Evaluating Parenchyma (Otsu 2+3) and Cortical (Otsu 2) Dice, folding, and timing.
"""

import time
import os
import torch
import numpy as np
import pandas as pd
import ants

import syntx
from syntx.deformation_metrics import compute_bidirectional_dice, compute_jacobian_metrics
from syntx.viz.reports import create_registration_report

def main():
    print("=" * 90, flush=True)
    print(" BENCHMARK: RegAdam vs CFL in syntx.syn on r16 -> r64 (2D Brain Slice) ")
    print("=" * 90, flush=True)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Compute Device: {device.upper()}", flush=True)

    fi = ants.image_read(ants.get_data("r16"))
    mi = ants.image_read(ants.get_data("r64"))

    # Generate Otsu Segmentations
    otsu_fi = ants.threshold_image(fi, "Otsu", 3)
    otsu_mi = ants.threshold_image(mi, "Otsu", 3)

    # Class 2: Cortical Gray Matter
    lbl_fi_cort = ants.threshold_image(otsu_fi, 2, 2)
    lbl_mi_cort = ants.threshold_image(otsu_mi, 2, 2)

    # Class 2+3: Parenchyma (Brain Tissue)
    lbl_fi_paren = ants.threshold_image(otsu_fi, 2, 3)
    lbl_mi_paren = ants.threshold_image(otsu_mi, 2, 3)

    print("\n--- Running Multi-Start Robust Affine Initialization ---", flush=True)
    t0_aff = time.time()
    res_aff = syntx.robust_affine(fi, mi, mode="auto", verbose=False)
    t_aff = time.time() - t0_aff
    aff_tx = res_aff["fwdtransforms"][0]

    # Baseline Dice
    d_f_paren_aff, d_m_paren_aff, d_sym_paren_aff = compute_bidirectional_dice(
        fl=lbl_fi_paren, ml=lbl_mi_paren, fi=fi, mi=mi,
        fwdtransforms=[aff_tx], invtransforms=[aff_tx],
        whichtoinvert_inv=[True]
    )
    d_f_cort_aff, d_m_cort_aff, d_sym_cort_aff = compute_bidirectional_dice(
        fl=lbl_fi_cort, ml=lbl_mi_cort, fi=fi, mi=mi,
        fwdtransforms=[aff_tx], invtransforms=[aff_tx],
        whichtoinvert_inv=[True]
    )
    print(f"   => Affine done in {t_aff:.3f}s", flush=True)
    print(f"   => Parenchyma Dice (2+3): Sym={d_sym_paren_aff:.4f} (Fix={d_f_paren_aff:.4f}, Mov={d_m_paren_aff:.4f})", flush=True)
    print(f"   => Cortical Dice (2):     Sym={d_sym_cort_aff:.4f} (Fix={d_f_cort_aff:.4f}, Mov={d_m_cort_aff:.4f})\n", flush=True)

    def run_case(name, **kwargs):
        base = dict(
            fixed=fi,
            moving=mi,
            initial_transform=aff_tx,
            backend="pytorch",
            formulation="eulerian",
            inverse_method="anderson",
            in_loop_inv_steps=10,
            use_analytical_gradients=False,
            flow_sigma=3.0,
            total_sigma=0.0,
            reg_iterations=[100, 100, 20],
            levels=[4, 2, 1],
            syn_metric="lncc",
            syn_sampling=2,
            verbose=False
        )
        base.update(kwargs)

        t0 = time.time()
        res = syntx.syn(**base)
        t_run = time.time() - t0

        fwd_tx = res["fwdtransforms"]
        inv_tx = res["invtransforms"]
        which_inv = res.get("whichtoinvert_inv", [True, False])

        d_f_p, d_m_p, d_sym_p = compute_bidirectional_dice(
            fl=lbl_fi_paren, ml=lbl_mi_paren, fi=fi, mi=mi,
            fwdtransforms=fwd_tx, invtransforms=inv_tx,
            whichtoinvert_inv=which_inv
        )
        d_f_c, d_m_c, d_sym_c = compute_bidirectional_dice(
            fl=lbl_fi_cort, ml=lbl_mi_cort, fi=fi, mi=mi,
            fwdtransforms=fwd_tx, invtransforms=inv_tx,
            whichtoinvert_inv=which_inv
        )

        jac = compute_jacobian_metrics(fi, fwd_tx[0])
        fold_pct = float(jac.get("folding_pct", 0.0))
        min_jac = float(jac.get("min", 0.0))

        row = {
            "Config": name,
            "Optimizer": base.get("optimizer", "cfl"),
            "LR": base.get("optimizer_lr", "default"),
            "Step": base.get("grad_step", 0.25),
            "Reg": base.get("regularizer", "gaussian"),
            "Paren Sym": d_sym_p,
            "Cort Sym": d_sym_c,
            "Fix Cort": d_f_c,
            "Mov Cort": d_m_c,
            "Fold %": fold_pct,
            "MinJ": min_jac,
            "Time (s)": t_run
        }
        print(f" >> {name:<32}: Paren={d_sym_p:.4f} | Cort={d_sym_c:.4f} (Fix={d_f_c:.4f}, Mov={d_m_c:.4f}) | Fold={fold_pct:.4f}% | MinJ={min_jac:+.4f} | Time={t_run:.2f}s", flush=True)
        return row, res

    CONFIGS = {
        # 1. Standard CFL Baselines
        "1_CFL_Gaussian_Step0.25": dict(optimizer="cfl", regularizer="gaussian", grad_step=0.25),
        "2_CFL_Gaussian_Step0.50": dict(optimizer="cfl", regularizer="gaussian", grad_step=0.50),
        "3_CFL_Sobolev_Dual_Step0.50": dict(optimizer="cfl", regularizer="sobolev", sobolev_alpha=1.0, fast_smooth=False, grad_step=0.50),
        "4_CFL_DSTI1_Dual_Step0.50": dict(optimizer="cfl", regularizer="dsti1", sobolev_alpha=1.0, fast_smooth=False, grad_step=0.50),
        
        # 2. RegAdam with Gaussian quotient smoothing
        "5_RegAdam_Gaussian_LR0.5_Step0.50": dict(optimizer="reg_adam", optimizer_lr=0.5, regularizer="gaussian", grad_step=0.50),
        "6_RegAdam_Gaussian_LR1.0_Step0.50": dict(optimizer="reg_adam", optimizer_lr=1.0, regularizer="gaussian", grad_step=0.50),
        "7_RegAdam_Gaussian_LR1.2_Step0.50": dict(optimizer="reg_adam", optimizer_lr=1.2, regularizer="gaussian", grad_step=0.50),
        
        # 3. RegAdam with Sobolev quotient smoothing
        "8_RegAdam_Sobolev_LR1.0_Step0.50": dict(optimizer="reg_adam", optimizer_lr=1.0, regularizer="sobolev", sobolev_alpha=1.0, fast_smooth=False, grad_step=0.50),
        "9_RegAdam_Sobolev_LR1.2_Step0.50": dict(optimizer="reg_adam", optimizer_lr=1.2, regularizer="sobolev", sobolev_alpha=1.0, fast_smooth=False, grad_step=0.50),
        
        # 4. RegAdam with DST-I1 quotient smoothing
        "10_RegAdam_DSTI1_LR1.0_Step0.50": dict(optimizer="reg_adam", optimizer_lr=1.0, regularizer="dsti1", sobolev_alpha=1.0, fast_smooth=False, grad_step=0.50),
    }

    results = []
    saved_outputs = {}

    print("--- Running 2D r16 -> r64 Parameter Evaluation ---", flush=True)
    for name, kw in CONFIGS.items():
        row, res = run_case(name, **kw)
        results.append(row)
        saved_outputs[name] = res

    # Master Table
    df = pd.DataFrame(results)
    print("\n" + "=" * 115, flush=True)
    print(" MASTER 2D R16 -> R64 RESULTS: RegAdam vs CFL in syntx.syn ")
    print("=" * 115, flush=True)
    header = f"{'Config':<34} {'Opt':<9} {'LR':<7} {'Step':<5} {'Reg':<9} {'Paren Sym':>10} {'Cort Sym':>9} {'Fix Cort':>9} {'Mov Cort':>9} {'Fold %':>8} {'Time':>6}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for _, r in df.iterrows():
        print(f"{r['Config']:<34} {r['Optimizer']:<9} {str(r['LR']):<7} {r['Step']:<5.2f} {r['Reg']:<9} {r['Paren Sym']:>10.4f} {r['Cort Sym']:>9.4f} {r['Fix Cort']:>9.4f} {r['Mov Cort']:>9.4f} {r['Fold %']:>8.4f}% {r['Time (s)']:>5.2f}s", flush=True)
    print("=" * 115 + "\n", flush=True)

    # Generate HTML Report for best config
    best_config_name = df.sort_values(by="Cort Sym", ascending=False).iloc[0]["Config"]
    best_res = saved_outputs[best_config_name]
    out_dir = "/Users/stnava/data/syntx/docs/reports"
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, "r16_r64_regadam_vs_cfl_report.html")

    print(f"Generating HTML report for best model ({best_config_name}) at: {html_path} ...", flush=True)
    create_registration_report(
        fixed=fi,
        moving=mi,
        reg=best_res,
        fixed_label=lbl_fi_cort,
        moving_label=lbl_mi_cort,
        output_html=html_path,
        title=f"r16 -> r64: RegAdam vs CFL Benchmark (Best: {best_config_name})"
    )
    print(f"Report available at: {html_path}\n", flush=True)

if __name__ == "__main__":
    main()
