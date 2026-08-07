"""
Master Benchmarking Orchestrator for syntx (Phase 1, Phase 2, Phase 3).

Executes the complete protocol specified in docs/BENCHMARKING_GUIDE.md:
1. Phase 1: All 2D Benchmark Datasets ('r16_r64', 'c', 'ellipse') across 30 parameter combinations.
2. Phase 2: 3D Benchmark Dataset ('mbhard') across 30 parameter combinations -> selects Top 5 configurations.
3. Phase 3: 90-Pair Mindboggle Population Evaluation across Top 5 configurations.

Logs real-time progress and structured JSON results to docs/provenance/ and updates
the user-facing artifact BENCHMARKING_PROGRESS_REPORT.md.
"""

import os
import time
import json
import gc
import pandas as pd
import numpy as np
import torch
import ants
import syntx
from syntx.syn import calculate_inverse_identity_error


# --- Helper Functions ---

def get_mindboggle_paths(cohort, subject):
    """Resolves image and label file paths for Mindboggle subjects."""
    base_dir = f"/Users/stnava/data/mindboggle/volumes/{cohort}_volumes"
    img_path = os.path.join(base_dir, subject, "t1weighted.nii.gz")
    lbl_path = os.path.join(base_dir, subject, "labels.dkt31.mri.nii.gz")
    return img_path, lbl_path


def compute_bidirectional_dice(fl, ml, fi, mi, fwdtransforms, invtransforms, whichtoinvert_inv=None):
    """Computes bidirectional fixed, moving, and symmetric mean Dice scores."""
    if whichtoinvert_inv is None:
        whichtoinvert_inv = [True, False]

    # 1. Fixed Space Dice
    ml_warped = ants.apply_transforms(
        fixed=fi, moving=ml,
        transformlist=fwdtransforms,
        interpolator='nearestNeighbor'
    )
    ov_fixed = ants.label_overlap_measures(fl, ml_warped)
    df_fixed = ov_fixed[~ov_fixed['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_fixed = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_fixed.columns else 'TargetOverlap'
    dice_fixed = float(df_fixed[col_fixed].mean()) if len(df_fixed) > 0 else 0.0

    # 2. Moving Space Dice
    fl_warped = ants.apply_transforms(
        fixed=mi, moving=fl,
        transformlist=invtransforms,
        whichtoinvert=whichtoinvert_inv,
        interpolator='nearestNeighbor'
    )
    ov_moving = ants.label_overlap_measures(ml, fl_warped)
    df_moving = ov_moving[~ov_moving['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_moving = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_moving.columns else 'TargetOverlap'
    dice_moving = float(df_moving[col_moving].mean()) if len(df_moving) > 0 else 0.0

    dice_sym = 0.5 * (dice_fixed + dice_moving)
    return dice_fixed, dice_moving, dice_sym


def build_30_grid():
    """Builds the canonical 30-combination parameter grid."""
    grid = []
    syn_tuples = {
        'S1': {'flow_sigma': 1.0, 'grad_step': 0.25},
        'S2': {'flow_sigma': 3.0, 'grad_step': 0.25},
        'S3': {'flow_sigma': 3.0, 'grad_step': 0.50},
    }
    for tuple_name, params in syn_tuples.items():
        for reg in ['gaussian', 'sobolev', 'dsti']:
            for fast in [True, False]:
                grid.append({
                    'id': f"syn_{reg}_fast{fast}_{tuple_name}",
                    'model': 'syn',
                    'regularizer': reg,
                    'fast_smooth': fast,
                    'tuple_name': tuple_name,
                    'params': params
                })

    tvf_tuples = {
        'T1': {'flow_sigma': 1.5, 'grad_step': 0.90, 'total_sigma': 0.05},
        'T2': {'flow_sigma': 0.4, 'grad_step': 0.50, 'total_sigma': 0.05},
    }
    for tuple_name, params in tvf_tuples.items():
        for reg in ['gaussian', 'sobolev', 'dsti']:
            for fast in [True, False]:
                grid.append({
                    'id': f"tvf_{reg}_fast{fast}_{tuple_name}",
                    'model': 'tvf',
                    'regularizer': reg,
                    'fast_smooth': fast,
                    'tuple_name': tuple_name,
                    'params': params
                })

    return grid


# --- Phase 1: Complete 2D Grid Sweeps ---

def run_phase1_2d(grid):
    print("\n==================================================================", flush=True)
    print(" PHASE 1: COMPLETE 2D GRID SWEEP ('r16_r64', 'c', 'ellipse')", flush=True)
    print("==================================================================", flush=True)

    datasets = ['r16_r64', 'c', 'ellipse']
    all_2d_results = {}

    for ds_key in datasets:
        print(f"\n--- Sweeping 30 combinations on 2D dataset: '{ds_key}' ---", flush=True)
        data = syntx.benchmark_data(ds_key)
        fi, mi = data['fixed'], data['moving']
        fl, ml = data['fixed_label'], data['moving_label']

        reg_aff = syntx.robust_affine(fixed=fi, moving=mi, multi_start=True, mode='pytorch', verbose=False)
        aff_tx = reg_aff['fwdtransforms'][0]

        ds_records = []
        for idx, cfg in enumerate(grid, 1):
            config_id = cfg['id']
            t0 = time.time()
            try:
                if cfg['model'] == 'syn':
                    reg = syntx.syn(
                        fixed=fi, moving=mi, initial_transform=aff_tx,
                        backend='pytorch', device='mps' if torch.backends.mps.is_available() else 'cpu',
                        reg_iterations=[100, 100, 20], affine_iterations=[0, 0, 0],
                        similarity_metric='lncc', syn_sampling=2, inverse_method='anderson',
                        total_sigma=0.0, regularizer=cfg['regularizer'], fast_smooth=cfg['fast_smooth'],
                        antisymmetric=True, verbose=False, **cfg['params']
                    )
                else:
                    reg = syntx.tvf(
                        fixed=fi, moving=mi, initial_transform=aff_tx,
                        backend='pytorch', device='mps' if torch.backends.mps.is_available() else 'cpu',
                        reg_iterations=[100, 100, 20], affine_iterations=[0, 0, 0],
                        similarity_metric='lncc', syn_sampling=2, multipoint_loss=[0.0, 0.5, 1.0],
                        optimizer='lars', cfl_max=0.0, cfl_momentum=0.95, n_time_steps=3,
                        constant_speed=True, constant_speed_relaxation=0.10, use_analytical_gradients=True,
                        regularizer=cfg['regularizer'], fast_smooth=cfg['fast_smooth'],
                        antisymmetric=True, verbose=False, **cfg['params']
                    )

                elapsed = time.time() - t0
                dice_fixed, dice_moving, dice_sym = compute_bidirectional_dice(fl, ml, fi, mi, reg['fwdtransforms'], reg['invtransforms'], reg.get('whichtoinvert_inv'))

                # Specific per-class scores for r16_r64
                class_scores = {}
                if ds_key == 'r16_r64':
                    fl_c2, ml_c2 = data['fixed_labels']['class2'], data['moving_labels']['class2']
                    fl_c23, ml_c23 = data['fixed_labels']['class2_3'], data['moving_labels']['class2_3']
                    fl_c3 = ants.threshold_image(data['fixed_labels']['otsu'], 3, 3)
                    ml_c3 = ants.threshold_image(data['moving_labels']['otsu'], 3, 3)

                    c2_f, c2_m, c2_s = compute_bidirectional_dice(fl_c2, ml_c2, fi, mi, reg['fwdtransforms'], reg['invtransforms'], reg.get('whichtoinvert_inv'))
                    c3_f, c3_m, c3_s = compute_bidirectional_dice(fl_c3, ml_c3, fi, mi, reg['fwdtransforms'], reg['invtransforms'], reg.get('whichtoinvert_inv'))
                    c23_f, c23_m, c23_s = compute_bidirectional_dice(fl_c23, ml_c23, fi, mi, reg['fwdtransforms'], reg['invtransforms'], reg.get('whichtoinvert_inv'))

                    class_scores = {
                        'cortical_gm_c2_dice': c2_s,
                        'white_matter_c3_dice': c3_s,
                        'parenchyma_c23_dice': c23_s
                    }

                jac_ants = ants.create_jacobian_determinant_image(fi, reg['fwdtransforms'][0], do_log=False)
                jac_np = jac_ants.numpy()
                mask_eval = ants.get_mask(fi).numpy() > 0

                folding_pct = float(np.mean(jac_np[mask_eval] <= 0) * 100.0)
                min_j = float(jac_np[mask_eval].min())

                record = {
                    'dataset': ds_key,
                    'config_id': config_id,
                    'model': cfg['model'],
                    'regularizer': cfg['regularizer'],
                    'fast_smooth': cfg['fast_smooth'],
                    'tuple_name': cfg['tuple_name'],
                    'dice_fixed': dice_fixed,
                    'dice_moving': dice_moving,
                    'dice_sym': dice_sym,
                    'folding_pct': folding_pct,
                    'min_jacobian': min_j,
                    'runtime_seconds': elapsed,
                    'class_scores': class_scores,
                    'status': 'SUCCESS'
                }
                print(f"[{idx}/30] {ds_key:<8} | {config_id:<28} | Dice_sym: {dice_sym:.4f} | Folding: {folding_pct:.4f}% | Time: {elapsed:.2f}s", flush=True)

            except Exception as e:
                elapsed = time.time() - t0
                print(f"[{idx}/30] {ds_key:<8} | {config_id:<28} | FAILED: {e}", flush=True)
                record = {
                    'dataset': ds_key,
                    'config_id': config_id,
                    'error': str(e),
                    'runtime_seconds': elapsed,
                    'status': 'FAILED'
                }

            ds_records.append(record)
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        all_2d_results[ds_key] = ds_records

    out_path = 'docs/provenance/phase1_2d_complete_results.json'
    os.makedirs('docs/provenance', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_2d_results, f, indent=2)

    return all_2d_results


# --- Phase 2: Complete 3D Grid Sweep ('mbhard') ---

def run_phase2_3d(grid):
    print("\n==================================================================", flush=True)
    print(" PHASE 2: COMPLETE 3D GRID SWEEP ('mbhard')", flush=True)
    print("==================================================================", flush=True)

    data = syntx.benchmark_data('mbhard')
    fi, mi = data['fixed'], data['moving']
    fl, ml = data['fixed_label'], data['moving_label']

    print("Computing robust 3D affine initialization...", flush=True)
    reg_aff = syntx.robust_affine(fixed=fi, moving=mi, multi_start=True, mode='pytorch', verbose=False)
    aff_tx = reg_aff['fwdtransforms'][0]

    records_3d = []
    for idx, cfg in enumerate(grid, 1):
        config_id = cfg['id']
        t0 = time.time()
        try:
            if cfg['model'] == 'syn':
                reg = syntx.syn(
                    fixed=fi, moving=mi, initial_transform=aff_tx,
                    backend='pytorch', device='mps' if torch.backends.mps.is_available() else 'cpu',
                    reg_iterations=[100, 100, 20], affine_iterations=[0, 0, 0],
                    similarity_metric='lncc', syn_sampling=2, inverse_method='anderson',
                    total_sigma=0.0, regularizer=cfg['regularizer'], fast_smooth=cfg['fast_smooth'],
                    antisymmetric=True, verbose=False, **cfg['params']
                )
            else:
                reg = syntx.tvf(
                    fixed=fi, moving=mi, initial_transform=aff_tx,
                    backend='pytorch', device='mps' if torch.backends.mps.is_available() else 'cpu',
                    reg_iterations=[100, 100, 20], affine_iterations=[0, 0, 0],
                    similarity_metric='lncc', syn_sampling=2, multipoint_loss=[0.0, 0.5, 1.0],
                    optimizer='lars', cfl_max=0.0, cfl_momentum=0.95, n_time_steps=3,
                    constant_speed=True, constant_speed_relaxation=0.10, use_analytical_gradients=True,
                    regularizer=cfg['regularizer'], fast_smooth=cfg['fast_smooth'],
                    antisymmetric=True, verbose=False, **cfg['params']
                )

            elapsed = time.time() - t0
            dice_fixed, dice_moving, dice_sym = compute_bidirectional_dice(fl, ml, fi, mi, reg['fwdtransforms'], reg['invtransforms'], reg.get('whichtoinvert_inv'))

            jac_ants = ants.create_jacobian_determinant_image(fi, reg['fwdtransforms'][0], do_log=False)
            jac_np = jac_ants.numpy()
            mask_eval = ants.get_mask(fi).numpy() > 0

            folding_pct = float(np.mean(jac_np[mask_eval] <= 0) * 100.0)
            min_j = float(jac_np[mask_eval].min())

            record = {
                'dataset': 'mbhard',
                'config_id': config_id,
                'model': cfg['model'],
                'regularizer': cfg['regularizer'],
                'fast_smooth': cfg['fast_smooth'],
                'tuple_name': cfg['tuple_name'],
                'params': cfg['params'],
                'dice_fixed': dice_fixed,
                'dice_moving': dice_moving,
                'dice_sym': dice_sym,
                'folding_pct': folding_pct,
                'min_jacobian': min_j,
                'runtime_seconds': elapsed,
                'status': 'SUCCESS'
            }
            print(f"[{idx}/30] mbhard | {config_id:<28} | Dice_sym: {dice_sym:.4f} | Folding: {folding_pct:.4f}% | Time: {elapsed:.2f}s", flush=True)

        except Exception as e:
            elapsed = time.time() - t0
            print(f"[{idx}/30] mbhard | {config_id:<28} | FAILED: {e}", flush=True)
            record = {
                'dataset': 'mbhard',
                'config_id': config_id,
                'error': str(e),
                'runtime_seconds': elapsed,
                'status': 'FAILED'
            }

        records_3d.append(record)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    out_path = 'docs/provenance/phase2_3d_complete_results.json'
    with open(out_path, 'w') as f:
        json.dump(records_3d, f, indent=2)

    # Rank and select Top 5 configurations
    valid_records = [r for r in records_3d if r.get('status') == 'SUCCESS']
    valid_records.sort(key=lambda x: (x['dice_sym'], -x['folding_pct']), reverse=True)
    top_5_configs = valid_records[:5]

    print("\n--- TOP 5 CONFIGURATIONS FROM PHASE 2 ---", flush=True)
    for i, cfg in enumerate(top_5_configs, 1):
        print(f" #{i}: {cfg['config_id']:<28} | Dice_sym: {cfg['dice_sym']:.4f} | Folding: {cfg['folding_pct']:.4f}% | Time: {cfg['runtime_seconds']:.2f}s", flush=True)

    return records_3d, top_5_configs


# --- Phase 3: 90-Pair Mindboggle Population Benchmark ---

def run_phase3_90pair(top_5_configs):
    print("\n==================================================================", flush=True)
    print(" PHASE 3: 90-PAIR MINDBOGGLE POPULATION BENCHMARK", flush=True)
    print("==================================================================", flush=True)

    pairs_df = pd.read_csv("examples/pairs.csv")
    total_pairs = len(pairs_df)

    population_results = {}

    for cfg_idx, cfg in enumerate(top_5_configs, 1):
        config_id = cfg['config_id']
        model_type = cfg['model']
        reg_type = cfg['regularizer']
        fast_smooth = cfg['fast_smooth']
        params = cfg['params']

        print(f"\n--- [{cfg_idx}/5] Evaluating Config: {config_id} across {total_pairs} pairs ---", flush=True)

        pair_metrics = []

        for p_idx, row in pairs_df.iterrows():
            pair_type = row['type']
            c1, s1 = row['cohort1'], row['subject1']
            c2, s2 = row['cohort2'], row['subject2']

            fi_path, fl_path = get_mindboggle_paths(c1, s1)
            mi_path, ml_path = get_mindboggle_paths(c2, s2)

            if not os.path.exists(fi_path) or not os.path.exists(mi_path):
                print(f" [{p_idx+1}/{total_pairs}] SKIPPING {s1} vs {s2} (File missing)", flush=True)
                continue

            fi = ants.image_read(fi_path)
            mi = ants.image_read(mi_path)
            fl = ants.image_read(fl_path)
            ml = ants.image_read(ml_path)

            t0 = time.time()
            try:
                reg_aff = syntx.robust_affine(fixed=fi, moving=mi, multi_start=True, mode='pytorch', verbose=False)
                aff_tx = reg_aff['fwdtransforms'][0]

                if model_type == 'syn':
                    reg = syntx.syn(
                        fixed=fi, moving=mi, initial_transform=aff_tx,
                        backend='pytorch', device='mps' if torch.backends.mps.is_available() else 'cpu',
                        reg_iterations=[100, 100, 20], affine_iterations=[0, 0, 0],
                        similarity_metric='lncc', syn_sampling=2, inverse_method='anderson',
                        total_sigma=0.0, regularizer=reg_type, fast_smooth=fast_smooth,
                        antisymmetric=True, verbose=False, **params
                    )
                else:
                    reg = syntx.tvf(
                        fixed=fi, moving=mi, initial_transform=aff_tx,
                        backend='pytorch', device='mps' if torch.backends.mps.is_available() else 'cpu',
                        reg_iterations=[100, 100, 20], affine_iterations=[0, 0, 0],
                        similarity_metric='lncc', syn_sampling=2, multipoint_loss=[0.0, 0.5, 1.0],
                        optimizer='lars', cfl_max=0.0, cfl_momentum=0.95, n_time_steps=3,
                        constant_speed=True, constant_speed_relaxation=0.10, use_analytical_gradients=True,
                        regularizer=reg_type, fast_smooth=fast_smooth,
                        antisymmetric=True, verbose=False, **params
                    )

                elapsed = time.time() - t0
                dice_fixed, dice_moving, dice_sym = compute_bidirectional_dice(fl, ml, fi, mi, reg['fwdtransforms'], reg['invtransforms'], reg.get('whichtoinvert_inv'))

                jac_ants = ants.create_jacobian_determinant_image(fi, reg['fwdtransforms'][0], do_log=False)
                jac_np = jac_ants.numpy()
                mask_eval = ants.get_mask(fi).numpy() > 0
                folding_pct = float(np.mean(jac_np[mask_eval] <= 0) * 100.0)

                pair_record = {
                    'pair_index': p_idx,
                    'pair_type': pair_type,
                    'pair_id': f"{s1}_vs_{s2}",
                    'dice_fixed': dice_fixed,
                    'dice_moving': dice_moving,
                    'dice_sym': dice_sym,
                    'folding_pct': folding_pct,
                    'runtime_seconds': elapsed
                }
                print(f" [{p_idx+1}/{total_pairs}] Pair {s1} vs {s2} ({pair_type}) | Dice_sym: {dice_sym:.4f} | Folding: {folding_pct:.4f}% | Time: {elapsed:.2f}s", flush=True)

            except Exception as e:
                elapsed = time.time() - t0
                print(f" [{p_idx+1}/{total_pairs}] FAILED {s1} vs {s2}: {e}", flush=True)
                pair_record = {
                    'pair_index': p_idx,
                    'pair_type': pair_type,
                    'pair_id': f"{s1}_vs_{s2}",
                    'error': str(e),
                    'runtime_seconds': elapsed,
                    'status': 'FAILED'
                }

            pair_metrics.append(pair_record)
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Compute summary population statistics for this config
        successful = [m for m in pair_metrics if 'dice_sym' in m]
        dice_syms = [m['dice_sym'] for m in successful]
        fold_pcts = [m['folding_pct'] for m in successful]
        times = [m['runtime_seconds'] for m in successful]

        summary = {
            'config_id': config_id,
            'n_pairs_evaluated': len(successful),
            'mean_dice_sym': float(np.mean(dice_syms)) if dice_syms else 0.0,
            'std_dice_sym': float(np.std(dice_syms)) if dice_syms else 0.0,
            'min_dice_sym': float(np.min(dice_syms)) if dice_syms else 0.0,
            'max_dice_sym': float(np.max(dice_syms)) if dice_syms else 0.0,
            'mean_folding_pct': float(np.mean(fold_pcts)) if fold_pcts else 0.0,
            'mean_runtime_s': float(np.mean(times)) if times else 0.0,
            'pair_details': pair_metrics
        }

        population_results[config_id] = summary

    out_path = 'docs/provenance/phase3_90pair_population_results.json'
    with open(out_path, 'w') as f:
        json.dump(population_results, f, indent=2)

    return population_results


def update_progress_report(phase1_results=None, phase2_results=None, top_5_configs=None, phase3_results=None):
    """Dynamically updates BENCHMARKING_PROGRESS_REPORT.md with latest benchmark tables."""
    report_path = "/Users/stnava/.gemini/antigravity-cli/brain/e4be1d1c-0c23-4e02-90f9-700b0da93d22/BENCHMARKING_PROGRESS_REPORT.md"
    
    p1_status = "COMPLETED" if phase1_results else "IN PROGRESS"
    p2_status = "COMPLETED" if phase2_results else ("IN PROGRESS" if phase1_results else "QUEUED")
    p3_status = "COMPLETED" if phase3_results else ("IN PROGRESS" if phase2_results else "QUEUED")
    
    md = [
        "# Syntx Benchmarking Progress & Provenance Report",
        "",
        "> **Live Benchmarking Execution Artifact** — Updated in Real Time",
        "> Protocol: [`docs/BENCHMARKING_GUIDE.md`](file:///Users/stnava/code/syntx/docs/BENCHMARKING_GUIDE.md)",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "This report tracks the systematic execution of the 3-phase `syntx` benchmarking protocol:",
        "",
        "1. **Phase 1: Complete 2D Grid Sweep** — Sweeping 30 parameter combinations (18 SyN, 12 TVF) across all 3 canonical 2D datasets (`'r16_r64'`, `'c'`, `'ellipse'`).",
        "2. **Phase 2: Complete 3D Grid Sweep** — Sweeping 30 parameter combinations across `syntx.benchmark_data('mbhard')` to select Top 5 configurations.",
        "3. **Phase 3: 90-Pair Mindboggle Population Benchmark** — Population-scale evaluation across all 90 Mindboggle pairs (`examples/pairs.csv`).",
        "",
        "---",
        "",
        "## 1. Protocol Execution Status",
        "",
        "| Phase | Dataset / Target | Parameter Scope | Execution Status | Key Output File |",
        "|-------|------------------|-----------------|------------------|-----------------|",
        f"| **Phase 1** | 2D Datasets (`r16_r64`, `c`, `ellipse`) | 30 Combinations × 3 Datasets | **{p1_status}** | `docs/provenance/phase1_2d_complete_results.json` |",
        f"| **Phase 2** | 3D Mindboggle (`mbhard`) | 30 Combinations | **{p2_status}** | `docs/provenance/phase2_3d_complete_results.json` |",
        f"| **Phase 3** | 90-Pair Mindboggle Population | Top 5 Configs × 90 Pairs | **{p3_status}** | `docs/provenance/phase3_90pair_population_results.json` |",
        "",
        "---",
        "",
        "## 2. Phase 1: 2D Benchmark Characterization (`r16_r64`, `c`, `ellipse`)",
        ""
    ]
    
    if phase1_results:
        # 2.1 r16_r64
        if 'r16_r64' in phase1_results:
            md.append("### 2.1 Brain Slice Benchmark (`r16_r64`) — Full 30-Combination Grid")
            md.append("")
            md.append("| Config ID | Model | Regularizer | `fast_smooth` | Class 2 GM Dice | Class 3 WM Dice | Parenchyma Dice | Sym Mean Dice | Folding % | Time |")
            md.append("|-----------|-------|-------------|---------------|-----------------|-----------------|-----------------|---------------|-----------|------|")
            for r in phase1_results['r16_r64']:
                if r.get('status') == 'SUCCESS':
                    cs = r.get('class_scores', {})
                    gm = cs.get('cortical_gm_c2_dice', 0.0)
                    wm = cs.get('white_matter_c3_dice', 0.0)
                    par = cs.get('parenchyma_c23_dice', 0.0)
                    md.append(f"| `{r['config_id']}` | `{r['model']}` | `{r['regularizer']}` | `{r['fast_smooth']}` | {gm:.4f} | {wm:.4f} | {par:.4f} | {r['dice_sym']:.4f} | {r['folding_pct']:.4f}% | {r['runtime_seconds']:.2f}s |")
                else:
                    md.append(f"| `{r['config_id']}` | `{r.get('model','')}` | `{r.get('regularizer','')}` | `{r.get('fast_smooth','')}` | FAILED | FAILED | FAILED | FAILED | N/A | {r['runtime_seconds']:.2f}s |")
            md.append("")

        # 2.2 c & ellipse
        for ds in ['c', 'ellipse']:
            if ds in phase1_results:
                md.append(f"### 2.2 Phantom Benchmark (`{ds}`) — 30-Combination Grid")
                md.append("")
                md.append("| Config ID | Model | Regularizer | `fast_smooth` | Fixed Dice | Moving Dice | Sym Mean Dice | Folding % | Time |")
                md.append("|-----------|-------|-------------|---------------|------------|-------------|---------------|-----------|------|")
                for r in phase1_results[ds]:
                    if r.get('status') == 'SUCCESS':
                        md.append(f"| `{r['config_id']}` | `{r['model']}` | `{r['regularizer']}` | `{r['fast_smooth']}` | {r['dice_fixed']:.4f} | {r['dice_moving']:.4f} | {r['dice_sym']:.4f} | {r['folding_pct']:.4f}% | {r['runtime_seconds']:.2f}s |")
                    else:
                        md.append(f"| `{r['config_id']}` | `{r.get('model','')}` | `{r.get('regularizer','')}` | `{r.get('fast_smooth','')}` | FAILED | FAILED | FAILED | N/A | {r['runtime_seconds']:.2f}s |")
                md.append("")

    if phase2_results:
        md.append("---")
        md.append("")
        md.append("## 3. Phase 2: 3D Mindboggle Characterization (`mbhard`)")
        md.append("")
        if top_5_configs:
            md.append("### Top 5 Winning Parameter Configurations")
            md.append("")
            md.append("| Rank | Config ID | Model | Regularizer | `fast_smooth` | Parameters | DKT31 Sym Dice | Folding % | Time |")
            md.append("|------|-----------|-------|-------------|---------------|------------|----------------|-----------|------|")
            for i, cfg in enumerate(top_5_configs, 1):
                p_str = f"sigma={cfg['params'].get('flow_sigma')}, step={cfg['params'].get('grad_step')}"
                md.append(f"| #{i} | `{cfg['config_id']}` | `{cfg['model']}` | `{cfg['regularizer']}` | `{cfg['fast_smooth']}` | `{p_str}` | **{cfg['dice_sym']:.4f}** | {cfg['folding_pct']:.4f}% | {cfg['runtime_seconds']:.2f}s |")
            md.append("")

        md.append("### Complete 3D 30-Combination Grid Results")
        md.append("")
        md.append("| Config ID | Model | Regularizer | `fast_smooth` | Fixed Dice | Moving Dice | Sym Mean Dice | Folding % | Time |")
        md.append("|-----------|-------|-------------|---------------|------------|-------------|---------------|-----------|------|")
        for r in phase2_results:
            if r.get('status') == 'SUCCESS':
                md.append(f"| `{r['config_id']}` | `{r['model']}` | `{r['regularizer']}` | `{r['fast_smooth']}` | {r['dice_fixed']:.4f} | {r['dice_moving']:.4f} | {r['dice_sym']:.4f} | {r['folding_pct']:.4f}% | {r['runtime_seconds']:.2f}s |")
            else:
                md.append(f"| `{r['config_id']}` | `{r.get('model','')}` | `{r.get('regularizer','')}` | `{r.get('fast_smooth','')}` | FAILED | FAILED | FAILED | N/A | {r['runtime_seconds']:.2f}s |")
        md.append("")

    if phase3_results:
        md.append("---")
        md.append("")
        md.append("## 4. Phase 3: 90-Pair Mindboggle Population Evaluation")
        md.append("")
        md.append("| Config ID | Evaluated Pairs | Mean Sym Dice | Std Dice | Min Dice | Max Dice | Mean Folding % | Mean Time / Pair |")
        md.append("|-----------|-----------------|---------------|----------|----------|----------|----------------|-------------------|")
        for cfg_id, s in phase3_results.items():
            md.append(f"| `{cfg_id}` | {s['n_pairs_evaluated']} / 90 | **{s['mean_dice_sym']:.4f}** | ±{s['std_dice_sym']:.4f} | {s['min_dice_sym']:.4f} | {s['max_dice_sym']:.4f} | {s['mean_folding_pct']:.4f}% | {s['mean_runtime_s']:.2f}s |")
        md.append("")

    with open(report_path, 'w') as f:
        f.write("\n".join(md))


def main():
    grid = build_30_grid()

    # Initial report setup
    update_progress_report()

    # 1. Phase 1: 2D Benchmark Sweeps
    phase1_results = run_phase1_2d(grid)
    update_progress_report(phase1_results=phase1_results)

    # 2. Phase 2: 3D Benchmark Sweep & Top 5 Selection
    phase2_results, top_5_configs = run_phase2_3d(grid)
    update_progress_report(phase1_results=phase1_results, phase2_results=phase2_results, top_5_configs=top_5_configs)

    # 3. Phase 3: 90-Pair Population Evaluation
    phase3_results = run_phase3_90pair(top_5_configs)
    update_progress_report(phase1_results=phase1_results, phase2_results=phase2_results, top_5_configs=top_5_configs, phase3_results=phase3_results)

    print("\n==================================================================", flush=True)
    print(" ALL BENCHMARK PHASES (1, 2, 3) COMPLETED SUCCESSFULLY!", flush=True)
    print("==================================================================", flush=True)


if __name__ == '__main__':
    main()
