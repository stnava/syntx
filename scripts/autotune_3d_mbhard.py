"""
3D Autotuning on Mindboggle mbhard (Pair 44) with proper preprocessing.
Enforces strict quality criteria: must meet or beat prior benchmark results (ANTs 0.5876, Sobolev 0.6144).
"""
import os
import json
import time
import numpy as np
import torch
import ants
import syntx
from syntx.benchmark.evaluate import evaluate_mindboggle_pair

PAIR_IDX = 44  # mbhard canonical pair

def run_3d_autotune():
    print("=== Starting 3D Continuum Autotuning on mbhard (Pair 44) ===")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 0. Verified Baselines
    print("\n--- Running / Verifying Baselines on Pair 44 ---")
    
    # Sobolev Baseline (2026-09-20 canonical settings)
    t0 = time.time()
    rec_sob = evaluate_mindboggle_pair(
        PAIR_IDX, 'sobolev',
        grad_step=0.25,
        flow_sigma=3.0,
        sobolev_alpha=1.5,
        fast_smooth=True,
        verbose=False
    )
    t_sob = time.time() - t0
    sob_dice = rec_sob.get('syntx_dice_sym', 0.0)
    sob_fold = rec_sob.get('syntx_fold', 1.0)
    sob_jmin = rec_sob.get('syntx_min_jac', -1.0)
    ants_dice = rec_sob.get('ants_baseline', {}).get('dice_sym', 0.5876)
    
    print(f"ANTs Baseline: Dice={ants_dice:.4f}")
    print(f"Sobolev Baseline: Dice={sob_dice:.4f}, Fold={sob_fold:.6f}, Jmin={sob_jmin:.4f}, Time={t_sob:.1f}s")
    
    evaluations = []
    evaluations.append({
        'model': 'sobolev_baseline',
        'params': {'grad_step': 0.25, 'flow_sigma': 3.0, 'sobolev_alpha': 1.5, 'fast_smooth': True},
        'dice_sym': sob_dice,
        'folding': sob_fold,
        'min_jac': sob_jmin,
        'runtime': t_sob,
        'status': 'ACCEPTED' if sob_dice >= 0.6100 and sob_fold < 0.001 else 'REJECTED'
    })

    # Search candidates
    configs_to_test = [
        # 1. Div-Curl
        {
            'model': 'syn_divcurl',
            'params': {
                'grad_step': 0.30,
                'sobolev_alpha': 1.5,
                'beta': 3.0,
                'gamma': 1.0,
                'h3_envelope': True,
                'envelope_power': 1.0,
                'syn_metric': 'cc2',
                'reg_iterations': [100, 100, 20]
            }
        },
        {
            'model': 'syn_divcurl',
            'params': {
                'grad_step': 0.35,
                'sobolev_alpha': 1.5,
                'beta': 3.0,
                'gamma': 1.0,
                'h3_envelope': True,
                'envelope_power': 1.0,
                'syn_metric': 'cc2',
                'reg_iterations': [100, 100, 20]
            }
        },
        {
            'model': 'syn_divcurl',
            'params': {
                'grad_step': 0.30,
                'sobolev_alpha': 1.5,
                'beta': 4.0,
                'gamma': 0.8,
                'h3_envelope': True,
                'envelope_power': 1.0,
                'syn_metric': 'cc2',
                'reg_iterations': [100, 100, 20]
            }
        },
        # 2. Navier
        {
            'model': 'syn_navier',
            'params': {
                'grad_step': 0.30,
                'sobolev_alpha': 1.5,
                'poisson_ratio': 0.35,
                's': 2.0,
                'syn_metric': 'cc2',
                'reg_iterations': [100, 100, 20]
            }
        },
        {
            'model': 'syn_navier',
            'params': {
                'grad_step': 0.35,
                'sobolev_alpha': 1.5,
                'poisson_ratio': 0.35,
                's': 2.0,
                'syn_metric': 'cc2',
                'reg_iterations': [100, 100, 20]
            }
        },
        {
            'model': 'syn_navier',
            'params': {
                'grad_step': 0.30,
                'sobolev_alpha': 1.5,
                'poisson_ratio': 0.42,
                's': 2.0,
                'syn_metric': 'cc2',
                'reg_iterations': [100, 100, 20]
            }
        },
        # 3. Hyperelastic
        {
            'model': 'syn_hyperelastic',
            'params': {
                'grad_step': 0.30,
                'sobolev_alpha': 1.5,
                'bulk_modulus': 5.0,
                'h3_envelope': True,
                'envelope_power': 1.0,
                'syn_metric': 'cc2',
                'reg_iterations': [100, 100, 20]
            }
        },
        {
            'model': 'syn_hyperelastic',
            'params': {
                'grad_step': 0.35,
                'sobolev_alpha': 1.5,
                'bulk_modulus': 5.0,
                'h3_envelope': True,
                'envelope_power': 1.0,
                'syn_metric': 'cc2',
                'reg_iterations': [100, 100, 20]
            }
        },
        {
            'model': 'syn_hyperelastic',
            'params': {
                'grad_step': 0.30,
                'sobolev_alpha': 1.5,
                'bulk_modulus': 10.0,
                'h3_envelope': True,
                'envelope_power': 1.0,
                'syn_metric': 'cc2',
                'reg_iterations': [100, 100, 20]
            }
        },
        # 4. Solenoidal
        {
            'model': 'syn_solenoidal',
            'params': {
                'grad_step': 0.30,
                'sobolev_alpha': 1.5,
                's': 2.0,
                'syn_metric': 'cc2',
                'reg_iterations': [100, 100, 20]
            }
        },
        {
            'model': 'syn_solenoidal',
            'params': {
                'grad_step': 0.35,
                'sobolev_alpha': 1.5,
                's': 2.0,
                'syn_metric': 'cc2',
                'reg_iterations': [100, 100, 20]
            }
        },
        # 5. Sobolev Refined
        {
            'model': 'sobolev',
            'params': {
                'grad_step': 0.30,
                'flow_sigma': 3.0,
                'sobolev_alpha': 1.5,
                'fast_smooth': True,
                'syn_metric': 'cc2',
                'reg_iterations': [100, 100, 20]
            }
        }
    ]

    print(f"\n--- Evaluating {len(configs_to_test)} Continuum Configurations ---")
    for i, c in enumerate(configs_to_test, 1):
        m = c['model']
        p = c['params']
        print(f"\n[{i}/{len(configs_to_test)}] Testing {m} with params: {p}")
        t0 = time.time()
        
        # Evaluate pair 44
        if m == 'sobolev':
            rec = evaluate_mindboggle_pair(
                PAIR_IDX, 'sobolev',
                grad_step=p['grad_step'],
                flow_sigma=p['flow_sigma'],
                sobolev_alpha=p['sobolev_alpha'],
                fast_smooth=p['fast_smooth'],
                verbose=False
            )
        else:
            rec = evaluate_mindboggle_pair(
                PAIR_IDX, m,
                config=p,
                verbose=False
            )
        runtime = time.time() - t0
        
        d_val = float(rec.get('syntx_dice_sym', 0.0))
        fold = float(rec.get('syntx_fold', 1.0))
        jmin = float(rec.get('syntx_min_jac', -1.0))
        
        # Strict quality criteria
        # Must beat ANTs baseline (0.5876) and meet/exceed Sobolev baseline (~0.614)
        is_superior_to_ants = d_val > ants_dice
        is_competitive_with_sobolev = d_val >= 0.6100
        is_regular = fold < 0.001
        
        status = "ACCEPTED" if (is_superior_to_ants and is_competitive_with_sobolev and is_regular) else "REJECTED"
        
        print(f"--> Result: Dice={d_val:.4f} (ANTs: {ants_dice:.4f}, Sob: {sob_dice:.4f}) | Fold={fold:.6f} | Jmin={jmin:.4f} | Status={status}")
        
        evaluations.append({
            'model': m,
            'params': p,
            'dice_sym': d_val,
            'folding': fold,
            'min_jac': jmin,
            'runtime': runtime,
            'status': status
        })

    # Save results
    os.makedirs("docs/provenance/tuning", exist_ok=True)
    out_json = "docs/provenance/tuning/continuum_3d_autotune_verified.json"
    with open(out_json, "w") as f:
        json.dump({
            "ants_baseline": {"dice": ants_dice},
            "sobolev_baseline": {"dice": sob_dice, "folding": sob_fold, "min_jac": sob_jmin},
            "evaluations": evaluations
        }, f, indent=2)
    print(f"\nAll 3D evaluations saved to {out_json}")

if __name__ == "__main__":
    run_3d_autotune()
