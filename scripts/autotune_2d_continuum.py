"""
Autotune continuum regularizers on 2D r16_r64 slice pair with proper preprocessing.
"""
import os
import json
import time
import numpy as np
import torch
import ants
import syntx
from syntx.generators import benchmark_data
from syntx.core.pipeline import normalize_and_tensorize

def run_2d_autotune():
    print("=== Starting 2D Continuum Autotuning (r16_r64) ===")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    ds = benchmark_data('r16_r64')
    fixed = ds['fixed']
    moving = ds['moving']
    fixed_lbl = ds['fixed_label']
    moving_lbl = ds['moving_label']
    
    # Baseline ANTs registration
    print("Running 2D ANTs reference...")
    t0 = time.time()
    reg_ants = ants.registration(
        fixed=fixed, moving=moving,
        type_of_transform="SyN",
        syn_metric="CC",
        syn_sampling=2,
        reg_iterations=(100, 100, 20),
        flow_sigma=3.0,
        total_sigma=0.0,
        grad_step=0.25
    )
    t_ants = time.time() - t0
    
    # Evaluate ANTs Dice
    w_ants = ants.apply_transforms(fixed=fixed, moving=moving_lbl,
                                  transformlist=reg_ants['fwdtransforms'],
                                  interpolator='nearestNeighbor')
    ants_ov = ants.label_overlap_measures(fixed_lbl, w_ants)
    ants_dice = float(ants_ov[ants_ov['Label'] == 'MeanOverlap']['MeanOverlap'].iloc[0]) if 'MeanOverlap' in ants_ov['Label'].values else float(ants_ov['MeanOverlap'].iloc[0])
    print(f"ANTs 2D Baseline: Dice={ants_dice:.4f}, Time={t_ants:.2f}s")
    
    results = []
    
    # 1. Sobolev Baseline
    for step in [0.25, 0.35]:
        for alpha in [1.5, 2.0]:
            for fs in [True, False]:
                t0 = time.time()
                reg = syntx.syn(
                    fixed=fixed, moving=moving,
                    backend="pytorch", device=device,
                    grad_step=step,
                    sobolev_alpha=alpha,
                    fast_smooth=fs,
                    similarity_metric="cc2",
                    regularizer="sobolev",
                    reg_iterations=[100, 100, 20]
                )
                t_run = time.time() - t0
                w_lbl = ants.apply_transforms(fixed=fixed, moving=moving_lbl,
                                             transformlist=reg['fwdtransforms'],
                                             interpolator='nearestNeighbor')
                ov = ants.label_overlap_measures(fixed_lbl, w_lbl)
                d_val = float(ov[ov['Label'] == 'MeanOverlap']['MeanOverlap'].iloc[0]) if 'MeanOverlap' in ov['Label'].values else float(ov['MeanOverlap'].iloc[0])
                results.append({
                    'model': 'sobolev',
                    'params': {'grad_step': step, 'sobolev_alpha': alpha, 'fast_smooth': fs},
                    'dice': d_val,
                    'runtime': t_run,
                    'win_vs_ants': d_val > ants_dice
                })
                print(f"Sobolev (step={step}, alpha={alpha}, fs={fs}): Dice={d_val:.4f}, Time={t_run:.2f}s")
                
    # 2. Div-Curl
    for step in [0.25, 0.35]:
        for beta in [3.0, 5.0]:
            for gamma in [0.5, 1.0]:
                for env_p in [1.0, 2.0]:
                    t0 = time.time()
                    reg = syntx.syn(
                        fixed=fixed, moving=moving,
                        backend="pytorch", device=device,
                        grad_step=step,
                        sobolev_alpha=1.5,
                        regularizer="div_curl",
                        beta=beta,
                        gamma=gamma,
                        h3_envelope=True,
                        envelope_power=env_p,
                        similarity_metric="cc2",
                        reg_iterations=[100, 100, 20]
                    )
                    t_run = time.time() - t0
                    w_lbl = ants.apply_transforms(fixed=fixed, moving=moving_lbl,
                                                 transformlist=reg['fwdtransforms'],
                                                 interpolator='nearestNeighbor')
                    ov = ants.label_overlap_measures(fixed_lbl, w_lbl)
                    d_val = float(ov[ov['Label'] == 'MeanOverlap']['MeanOverlap'].iloc[0]) if 'MeanOverlap' in ov['Label'].values else float(ov['MeanOverlap'].iloc[0])
                    results.append({
                        'model': 'div_curl',
                        'params': {'grad_step': step, 'beta': beta, 'gamma': gamma, 'envelope_power': env_p},
                        'dice': d_val,
                        'runtime': t_run,
                        'win_vs_ants': d_val > ants_dice
                    })
                    print(f"Div-Curl (step={step}, beta={beta}, gamma={gamma}, env_p={env_p}): Dice={d_val:.4f}, Time={t_run:.2f}s")

    # 3. Navier
    for step in [0.25, 0.35]:
        for nu in [0.35, 0.45]:
            for s in [1.5, 2.0]:
                t0 = time.time()
                reg = syntx.syn(
                    fixed=fixed, moving=moving,
                    backend="pytorch", device=device,
                    grad_step=step,
                    sobolev_alpha=1.5,
                    regularizer="navier",
                    poisson_ratio=nu,
                    s=s,
                    similarity_metric="cc2",
                    reg_iterations=[100, 100, 20]
                )
                t_run = time.time() - t0
                w_lbl = ants.apply_transforms(fixed=fixed, moving=moving_lbl,
                                             transformlist=reg['fwdtransforms'],
                                             interpolator='nearestNeighbor')
                ov = ants.label_overlap_measures(fixed_lbl, w_lbl)
                d_val = float(ov[ov['Label'] == 'MeanOverlap']['MeanOverlap'].iloc[0]) if 'MeanOverlap' in ov['Label'].values else float(ov['MeanOverlap'].iloc[0])
                results.append({
                    'model': 'navier',
                    'params': {'grad_step': step, 'poisson_ratio': nu, 's': s},
                    'dice': d_val,
                    'runtime': t_run,
                    'win_vs_ants': d_val > ants_dice
                })
                print(f"Navier (step={step}, nu={nu}, s={s}): Dice={d_val:.4f}, Time={t_run:.2f}s")

    # 4. Hyperelastic
    for step in [0.25, 0.35]:
        for k in [3.0, 5.0, 10.0]:
            for env_p in [1.0, 2.0]:
                t0 = time.time()
                reg = syntx.syn(
                    fixed=fixed, moving=moving,
                    backend="pytorch", device=device,
                    grad_step=step,
                    sobolev_alpha=1.5,
                    regularizer="hyperelastic",
                    bulk_modulus=k,
                    h3_envelope=True,
                    envelope_power=env_p,
                    similarity_metric="cc2",
                    reg_iterations=[100, 100, 20]
                )
                t_run = time.time() - t0
                w_lbl = ants.apply_transforms(fixed=fixed, moving=moving_lbl,
                                             transformlist=reg['fwdtransforms'],
                                             interpolator='nearestNeighbor')
                ov = ants.label_overlap_measures(fixed_lbl, w_lbl)
                d_val = float(ov[ov['Label'] == 'MeanOverlap']['MeanOverlap'].iloc[0]) if 'MeanOverlap' in ov['Label'].values else float(ov['MeanOverlap'].iloc[0])
                results.append({
                    'model': 'hyperelastic',
                    'params': {'grad_step': step, 'bulk_modulus': k, 'envelope_power': env_p},
                    'dice': d_val,
                    'runtime': t_run,
                    'win_vs_ants': d_val > ants_dice
                })
                print(f"Hyperelastic (step={step}, K={k}, env_p={env_p}): Dice={d_val:.4f}, Time={t_run:.2f}s")

    # 5. Solenoidal
    for step in [0.25, 0.35]:
        for alpha in [1.5, 2.0]:
            for s in [1.5, 2.0]:
                t0 = time.time()
                reg = syntx.syn(
                    fixed=fixed, moving=moving,
                    backend="pytorch", device=device,
                    grad_step=step,
                    sobolev_alpha=alpha,
                    regularizer="solenoidal",
                    s=s,
                    similarity_metric="cc2",
                    reg_iterations=[100, 100, 20]
                )
                t_run = time.time() - t0
                w_lbl = ants.apply_transforms(fixed=fixed, moving=moving_lbl,
                                             transformlist=reg['fwdtransforms'],
                                             interpolator='nearestNeighbor')
                ov = ants.label_overlap_measures(fixed_lbl, w_lbl)
                d_val = float(ov[ov['Label'] == 'MeanOverlap']['MeanOverlap'].iloc[0]) if 'MeanOverlap' in ov['Label'].values else float(ov['MeanOverlap'].iloc[0])
                results.append({
                    'model': 'solenoidal',
                    'params': {'grad_step': step, 'sobolev_alpha': alpha, 's': s},
                    'dice': d_val,
                    'runtime': t_run,
                    'win_vs_ants': d_val > ants_dice
                })
                print(f"Solenoidal (step={step}, alpha={alpha}, s={s}): Dice={d_val:.4f}, Time={t_run:.2f}s")

    # Save results
    os.makedirs("docs/provenance/tuning", exist_ok=True)
    out_json = "docs/provenance/tuning/continuum_2d_autotune_verified.json"
    with open(out_json, "w") as f:
        json.dump({
            "ants_baseline": {"dice": ants_dice, "runtime": t_ants},
            "evaluations": results
        }, f, indent=2)
    print(f"Results saved to {out_json}")

if __name__ == "__main__":
    run_2d_autotune()
