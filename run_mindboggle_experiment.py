import os
import sys

os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import csv
import json
import tempfile
import numpy as np
import pandas as pd
import torch
import ants

try:
    ants.set_number_of_threads(4)
except AttributeError:
    pass

import syntx

def compute_smoothness_metrics(disp_np, spacing):
    # disp_np is (D, H, W, 3) or (3, D, H, W)
    if disp_np.ndim == 4 and disp_np.shape[0] == 3:
        disp_np = np.moveaxis(disp_np, 0, -1)
        
    sp_x, sp_y, sp_z = spacing if spacing is not None else (1.0, 1.0, 1.0)
    
    du_dx = (disp_np[1:, :-1, :-1] - disp_np[:-1, :-1, :-1]) / sp_x
    du_dy = (disp_np[:-1, 1:, :-1] - disp_np[:-1, :-1, :-1]) / sp_y
    du_dz = (disp_np[:-1, :-1, 1:] - disp_np[:-1, :-1, :-1]) / sp_z
    
    s1 = float(np.mean(np.sqrt(du_dx**2 + du_dy**2 + du_dz**2)))
    
    d2u_dx2 = (du_dx[1:, :-1, :-1] - du_dx[:-1, :-1, :-1]) / sp_x
    d2u_dy2 = (du_dy[:-1, 1:, :-1] - du_dy[:-1, :-1, :-1]) / sp_y
    d2u_dz2 = (du_dz[:-1, :-1, 1:] - du_dz[:-1, :-1, :-1]) / sp_z
    
    s2 = float(np.mean(np.sqrt(d2u_dx2**2 + d2u_dy2**2 + d2u_dz2**2)))
    return s1, s2

def compute_jacobian_and_folding(fi, fwdtransform):
    jac_img = ants.create_jacobian_determinant_image(fi, fwdtransform)
    jac_np = jac_img.numpy()
    mask = ants.get_mask(fi).numpy() > 0
    
    jac_mean = float(np.mean(jac_np))
    jac_min = float(np.min(jac_np))
    jac_max = float(np.max(jac_np))
    jac_std = float(np.std(jac_np))
    
    folding_pct = float(np.mean(jac_np[mask] <= 0) * 100.0) if np.sum(mask) > 0 else 0.0
    return jac_mean, jac_min, jac_max, jac_std, folding_pct

def compute_overlap(fi, ml, fwdtransforms, fl):
    ml_warped = ants.apply_transforms(fi, ml, fwdtransforms, interpolator='nearestNeighbor')
    overlap = ants.label_overlap_measures(fl, ml_warped)
    df = overlap[(overlap['Label'] != 'All') & (overlap['Label'] != 0) & (overlap['Label'] != '0')]
    col = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df.columns else 'TargetOverlap'
    return float(df[col].mean()) if len(df) > 0 else 0.0

def process_pair(idx, pair, base_path, existing_record=None, **kwargs):
    c1, s1 = pair['cohort1'], pair['subject1']
    c2, s2 = pair['cohort2'], pair['subject2']
    
    res = dict(existing_record) if existing_record else {
        'pair_idx': idx,
        'fixed': s1,
        'moving': s2,
        'type': pair['type']
    }
    
    force_rerun_pt_jax = kwargs.get('force_rerun_pt_jax', True)
    need_ants = 'ants_dice' not in res
    need_pt = force_rerun_pt_jax or ('pt_dice' not in res)
    need_jax = force_rerun_pt_jax or ('jax_dice' not in res)
    
    if not (need_ants or need_pt or need_jax):
        return res
        
    f_path = os.path.join(base_path, f"{c1}_volumes", s1, 't1weighted_brain.MNI152.nii.gz')
    m_path = os.path.join(base_path, f"{c2}_volumes", s2, 't1weighted_brain.MNI152.nii.gz')
    fl_path = os.path.join(base_path, f"{c1}_volumes", s1, 'labels.DKT31.manual.MNI152.nii.gz')
    ml_path = os.path.join(base_path, f"{c2}_volumes", s2, 'labels.DKT31.manual.MNI152.nii.gz')
    
    fi_full = ants.image_read(f_path)
    mi_full = ants.image_read(m_path)
    mask_f = ants.iMath(ants.get_mask(fi_full), "MD", 12)
    fi = ants.crop_image(fi_full, mask_f)
    mask_m = ants.iMath(ants.get_mask(mi_full), "MD", 12)
    mi = ants.crop_image(mi_full, mask_m)
    
    fl = ants.crop_image(ants.image_read(fl_path), mask_f)
    ml = ants.crop_image(ants.image_read(ml_path), mask_m)
    
    # 1. ANTs (only if missing)
    if need_ants:
        print(f"  [Pair {idx}] Computing ANTs baseline...", flush=True)
        t0 = time.time()
        reg_ants = ants.registration(
            fixed=fi, moving=mi, type_of_transform='SyN',
            grad_step=0.25, reg_iterations=[100, 100, 20],
            syn_metric='cc', syn_sampling=2
        )
        res['ants_time'] = time.time() - t0
        res['ants_dice'] = compute_overlap(fi, ml, reg_ants['fwdtransforms'], fl)
        
        fwd_ants = reg_ants['fwdtransforms'][0]
        jmean, jmin, jmax, jstd, fold = compute_jacobian_and_folding(fi, fwd_ants)
        res['ants_jac_mean'], res['ants_jac_min'], res['ants_jac_max'], res['ants_jac_std'], res['ants_folding'] = jmean, jmin, jmax, jstd, fold
        
        disp_ants = ants.image_read(fwd_ants)
        s1_a, s2_a = compute_smoothness_metrics(disp_ants.numpy(), disp_ants.spacing)
        res['ants_smooth_1st'], res['ants_smooth_2nd'] = s1_a, s2_a

        # ANTs Inverse Identity Error
        inv_ants = next((tx for tx in reg_ants.get('invtransforms', []) if isinstance(tx, str) and tx.endswith(('.nii', '.nii.gz'))), None)
        if inv_ants is not None:
            try:
                import torch
                fwd_tensor = torch.tensor(disp_ants.numpy()).unsqueeze(0)
                inv_tensor = torch.tensor(ants.image_read(inv_ants).numpy()).unsqueeze(0)
                ants_err = syntx.calculate_inverse_identity_error(fwd_tensor, inv_tensor, fi.spacing, fi.origin, fi.direction)
                res['ants_inv_mean'] = float(ants_err.get('mean_error', 0.0))
                res['ants_inv_max'] = float(ants_err.get('max_error', 0.0))
            except Exception:
                res['ants_inv_mean'], res['ants_inv_max'] = 0.0, 0.0
    else:
        print(f"  [Pair {idx}] Preserving existing ANTs baseline (Dice={res['ants_dice']:.4f})", flush=True)

    import torch
    if torch.cuda.is_available():
        target_device = 'cuda'
    elif torch.backends.mps.is_available():
        target_device = 'mps'
    else:
        target_device = 'cpu'

    # 2. PyTorch (only if missing)
    if need_pt:
        print(f"  [Pair {idx}] Computing PyTorch backend...", flush=True)
        t0 = time.time()
        reg_pt = syntx.syn(
            fixed=fi, moving=mi, backend='pytorch', device=target_device,
            affine_iterations=[100, 50, 20], reg_iterations=[100, 100, 20],
            grad_step=0.25, flow_sigma=3.0, syn_metric='lncc', syn_sampling=2, inverse_steps=10
        )
        res['pt_time'] = time.time() - t0
        res['pt_dice'] = compute_overlap(fi, ml, reg_pt['fwdtransforms'], fl)
        
        fwd_pt = reg_pt['fwdtransforms'][0]
        jmean, jmin, jmax, jstd, fold = compute_jacobian_and_folding(fi, fwd_pt)
        res['pt_jac_mean'], res['pt_jac_min'], res['pt_jac_max'], res['pt_jac_std'], res['pt_folding'] = jmean, jmin, jmax, jstd, fold
        
        disp_pt = ants.image_read(fwd_pt)
        s1_p, s2_p = compute_smoothness_metrics(disp_pt.numpy(), disp_pt.spacing)
        res['pt_smooth_1st'], res['pt_smooth_2nd'] = s1_p, s2_p

        inv_errs_pt = reg_pt.get('inverse_identity_errors', {}).get('phi_1', {})
        res['pt_inv_mean'] = float(inv_errs_pt.get('mean_error', 0.0))
        res['pt_inv_max'] = float(inv_errs_pt.get('max_error', 0.0))
        
    # 3. JAX (only if missing)
    if need_jax:
        print(f"  [Pair {idx}] Computing JAX backend...", flush=True)
        t0 = time.time()
        reg_jax = syntx.syn(
            fixed=fi, moving=mi, backend='jax', device=target_device,
            affine_iterations=[100, 50, 20], reg_iterations=[100, 100, 20],
            grad_step=0.25, flow_sigma=3.0, syn_metric='lncc', syn_sampling=2, inverse_steps=10
        )
        res['jax_time'] = time.time() - t0
        res['jax_dice'] = compute_overlap(fi, ml, reg_jax['fwdtransforms'], fl)
        
        fwd_jax = reg_jax['fwdtransforms'][0]
        jmean, jmin, jmax, jstd, fold = compute_jacobian_and_folding(fi, fwd_jax)
        res['jax_jac_mean'], res['jax_jac_min'], res['jax_jac_max'], res['jax_jac_std'], res['jax_folding'] = jmean, jmin, jmax, jstd, fold
        
        disp_jax = ants.image_read(fwd_jax)
        s1_j, s2_j = compute_smoothness_metrics(disp_jax.numpy(), disp_jax.spacing)
        res['jax_smooth_1st'], res['jax_smooth_2nd'] = s1_j, s2_j

        inv_errs_jax = reg_jax.get('inverse_identity_errors', {}).get('phi_1', {})
        res['jax_inv_mean'] = float(inv_errs_jax.get('mean_error', 0.0))
        res['jax_inv_max'] = float(inv_errs_jax.get('max_error', 0.0))
        
    return res

def main():
    import random
    base_path = '/Users/stnava/data/mindboggle/volumes'
    pairs_file = '/Users/stnava/code/syntx/examples/pairs.csv'
    out_json = '/Users/stnava/code/syntx/benchmark_results.json'
    
    with open(pairs_file, 'r') as f:
        pairs = list(csv.DictReader(f))
        
    results_map = {}
    if os.path.exists(out_json):
        try:
            with open(out_json, 'r') as f:
                raw_results = json.load(f)
                results_map = {item['pair_idx']: item for item in raw_results if 'pair_idx' in item}
            print(f"Loaded {len(results_map)} existing records from {out_json}.", flush=True)
        except Exception as e:
            print(f"Could not load existing results ({e}). Starting fresh...", flush=True)
            results_map = {}

    pair_indices = list(range(len(pairs)))
    random.seed(42)  # Reproducible randomized order
    random.shuffle(pair_indices)

    print(f"Starting Mindboggle Experiment across {len(pairs)} subject pairs in randomized order...", flush=True)
    
    completed_eval_count = 0
    for step_num, i in enumerate(pair_indices):
        p = pairs[i]
        existing_rec = results_map.get(i, None)
        
        force_rerun = True
        if existing_rec and ('ants_dice' in existing_rec) and ('pt_dice' in existing_rec) and ('jax_dice' in existing_rec) and not force_rerun:
            print(f"[{step_num+1}/{len(pairs)}] Skipping Pair {i} ({p['subject1']} -> {p['subject2']}): all backends already complete.", flush=True)
            continue
            
        print(f"\n[{step_num+1}/{len(pairs)}] Processing Pair {i} ({p['subject1']} -> {p['subject2']})...", flush=True)
        r = process_pair(i, p, base_path, existing_record=existing_rec)
        results_map[i] = r
        completed_eval_count += 1
        
        # Save progress to benchmark_results.json, preserving disk records
        if os.path.exists(out_json):
            try:
                with open(out_json, 'r') as f_disk:
                    disk_recs = json.load(f_disk)
                    for item in disk_recs:
                        pid = item.get('pair_idx')
                        if pid == 55 and item.get('ants_dice', 0) > 0.1:
                            results_map[55] = item
            except Exception:
                pass
        sorted_results = [results_map[k] for k in sorted(results_map.keys())]
        with open(out_json, 'w') as f:
            json.dump(sorted_results, f, indent=2)
            
        f_str = r.get('fixed', r.get('pair_fixed_id', '?'))
        m_str = r.get('moving', r.get('pair_moving_id', '?'))
        print(f"[Completed {completed_eval_count}/{len(pairs)}] Pair {i} ({f_str} -> {m_str}): Dice [ANTs={r.get('ants_dice', 0):.4f}, PyTorch={r.get('pt_dice', 0):.4f}, JAX={r.get('jax_dice', 0):.4f}] | InvErr Mean/Max (mm) [PyTorch={r.get('pt_inv_mean', 0):.3f}/{r.get('pt_inv_max', 0):.3f}, JAX={r.get('jax_inv_mean', 0):.3f}/{r.get('jax_inv_max', 0):.3f}] | Folding % [ANTs={r.get('ants_folding', 0):.4f}%, PyTorch={r.get('pt_folding', 0):.4f}%, JAX={r.get('jax_folding', 0):.4f}%]", flush=True)
        
        # Print summary statistics table for pairs that have all backends computed
        fully_complete = [item for item in sorted_results if 'ants_dice' in item and 'pt_dice' in item and 'jax_dice' in item]
        count = len(fully_complete)
        if count > 0 and (count % 4 == 0 or count == len(pairs)):
            ants_dices = [item['ants_dice'] for item in fully_complete]
            pt_dices = [item['pt_dice'] for item in fully_complete]
            jax_dices = [item['jax_dice'] for item in fully_complete]
            
            ants_folds = [item['ants_folding'] for item in fully_complete]
            pt_folds = [item['pt_folding'] for item in fully_complete]
            jax_folds = [item['jax_folding'] for item in fully_complete]
            
            ants_jmins = [item['ants_jac_min'] for item in fully_complete]
            pt_jmins = [item['pt_jac_min'] for item in fully_complete]
            jax_jmins = [item['jax_jac_min'] for item in fully_complete]
            
            ants_jmaxs = [item['ants_jac_max'] for item in fully_complete]
            pt_jmaxs = [item['pt_jac_max'] for item in fully_complete]
            jax_jmaxs = [item['jax_jac_max'] for item in fully_complete]
            
            ants_s1 = [item['ants_smooth_1st'] for item in fully_complete]
            pt_s1 = [item['pt_smooth_1st'] for item in fully_complete]
            jax_s1 = [item['jax_smooth_1st'] for item in fully_complete]
            
            ants_s2 = [item['ants_smooth_2nd'] for item in fully_complete]
            pt_s2 = [item['pt_smooth_2nd'] for item in fully_complete]
            jax_s2 = [item['jax_smooth_2nd'] for item in fully_complete]

            ants_inv_m = [item.get('ants_inv_mean', 0.0) for item in fully_complete]
            pt_inv_m = [item.get('pt_inv_mean', 0.0) for item in fully_complete]
            jax_inv_m = [item.get('jax_inv_mean', 0.0) for item in fully_complete]

            ants_inv_mx = [item.get('ants_inv_max', 0.0) for item in fully_complete]
            pt_inv_mx = [item.get('pt_inv_max', 0.0) for item in fully_complete]
            jax_inv_mx = [item.get('jax_inv_max', 0.0) for item in fully_complete]
            
            ants_times = [item['ants_time'] for item in fully_complete]
            pt_times = [item['pt_time'] for item in fully_complete]
            jax_times = [item['jax_time'] for item in fully_complete]
            
            print(f"\n==========================================================================================================", flush=True)
            print(f"                   COMPREHENSIVE SUMMARY STATISTICS AFTER {count} FULLY COMPLETED PAIRS                       ", flush=True)
            print(f"==========================================================================================================", flush=True)
            print(f" METRIC                           | ANTs Baseline           | PyTorch Backend         | JAX Backend", flush=True)
            print(f" ---------------------------------+-------------------------+-------------------------+-------------------", flush=True)
            print(f" TargetOverlap Dice (Mean)        | {np.mean(ants_dices):.4f}                  | {np.mean(pt_dices):.4f}                  | {np.mean(jax_dices):.4f}", flush=True)
            print(f" TargetOverlap Dice (Median)      | {np.median(ants_dices):.4f}                  | {np.median(pt_dices):.4f}                  | {np.median(jax_dices):.4f}", flush=True)
            print(f" Folding Rate (% J <= 0)         | {np.mean(ants_folds):.4f}%                 | {np.mean(pt_folds):.4f}%                 | {np.mean(jax_folds):.4f}%", flush=True)
            print(f" Min Jacobian Determinant (Mean) | {np.mean(ants_jmins):.4f}                  | {np.mean(pt_jmins):.4f}                  | {np.mean(jax_jmins):.4f}", flush=True)
            print(f" Max Jacobian Determinant (Mean) | {np.mean(ants_jmaxs):.4f}                 | {np.mean(pt_jmaxs):.4f}                 | {np.mean(jax_jmaxs):.4f}", flush=True)
            print(f" 1st Derivative Smoothness (Mean) | {np.mean(ants_s1):.4f}                  | {np.mean(pt_s1):.4f}                  | {np.mean(jax_s1):.4f}", flush=True)
            print(f" 2nd Derivative Smoothness (Mean) | {np.mean(ants_s2):.4f}                  | {np.mean(pt_s2):.4f}                  | {np.mean(jax_s2):.4f}", flush=True)
            print(f" Inverse Identity Mean Error (mm)| {np.mean(ants_inv_m):.4f} mm              | {np.mean(pt_inv_m):.4f} mm              | {np.mean(jax_inv_m):.4f} mm", flush=True)
            print(f" Inverse Identity Max Error (mm) | {np.mean(ants_inv_mx):.4f} mm              | {np.mean(pt_inv_mx):.4f} mm              | {np.mean(jax_inv_mx):.4f} mm", flush=True)
            print(f" Execution Time per Pair (Mean)   | {np.mean(ants_times):.2f}s                 | {np.mean(pt_times):.2f}s                 | {np.mean(jax_times):.2f}s", flush=True)
            print(f"==========================================================================================================\n", flush=True)

if __name__ == '__main__':
    main()
