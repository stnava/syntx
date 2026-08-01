#!/usr/bin/env python3
"""
Mindboggle 90-pair benchmark: ANTs SyN (preserved), PyTorch SyN (MPS),
JAX SyN (CPU), TVF endpoint-loss (MPS), TVF midpoint-loss (MPS).

CPU tasks (ANTs, JAX SyN) run in parallel with MPS tasks (PyTorch SyN, TVF).
ANTs results are preserved from previous runs; only missing columns are computed.
"""
import os
import sys

os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import time
import csv
import json
import numpy as np
import torch
import ants
from concurrent.futures import ThreadPoolExecutor

try:
    ants.set_number_of_threads(4)
except AttributeError:
    pass

import syntx


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_smoothness_metrics(disp_np, spacing):
    """Compute 1st and 2nd derivative displacement smoothness."""
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
    """Compute Jacobian determinant statistics and folding rate."""
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
    """Compute mean DKT label Dice via nearest-neighbor interpolation."""
    ml_warped = ants.apply_transforms(fi, ml, fwdtransforms, interpolator='nearestNeighbor')
    overlap = ants.label_overlap_measures(fl, ml_warped)
    df = overlap[(overlap['Label'] != 'All') & (overlap['Label'] != 0) & (overlap['Label'] != '0')]
    col = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df.columns else 'TargetOverlap'
    return float(df[col].mean()) if len(df) > 0 else 0.0


def eval_registration(fi, ml, fl, reg, prefix):
    """Extract standard metrics from a registration result dict."""
    metrics = {}
    t_sec = reg.get('_elapsed', 0.0)
    fwdtransforms = reg['fwdtransforms']
    metrics[f'{prefix}_dice'] = compute_overlap(fi, ml, fwdtransforms, fl)
    metrics[f'{prefix}_time'] = t_sec

    fwd_tx = fwdtransforms[0]
    jmean, jmin, jmax, jstd, fold = compute_jacobian_and_folding(fi, fwd_tx)
    metrics[f'{prefix}_jac_mean'] = jmean
    metrics[f'{prefix}_jac_min'] = jmin
    metrics[f'{prefix}_jac_max'] = jmax
    metrics[f'{prefix}_jac_std'] = jstd
    metrics[f'{prefix}_folding'] = fold

    disp = ants.image_read(fwd_tx)
    s1, s2 = compute_smoothness_metrics(disp.numpy(), disp.spacing)
    metrics[f'{prefix}_smooth_1st'] = s1
    metrics[f'{prefix}_smooth_2nd'] = s2

    inv_errs = reg.get('inverse_identity_errors', {}).get('phi_1', {})
    metrics[f'{prefix}_inv_mean'] = float(inv_errs.get('mean_error', 0.0))
    metrics[f'{prefix}_inv_max'] = float(inv_errs.get('max_error', 0.0))

    return metrics


# ── CPU Tasks (run in ThreadPoolExecutor) ──────────────────────────────────────

def run_ants_task(fi, mi, fl, ml, affine_tx):
    """ANTs SyN registration (CPU), initialized with shared affine."""
    t0 = time.time()
    reg = ants.registration(
        fixed=fi, moving=mi, type_of_transform='SyN',
        initial_transform=affine_tx,
        grad_step=0.25, reg_iterations=[100, 100, 20],
        syn_metric='cc', syn_sampling=2
    )
    reg['_elapsed'] = time.time() - t0
    return eval_registration(fi, ml, fl, reg, 'ants')


def run_jax_syn_task(fi, mi, fl, ml, affine_tx):
    """JAX SyN registration (CPU), initialized with shared affine."""
    t0 = time.time()
    reg = syntx.syn(
        fixed=fi, moving=mi, backend='jax', device='cpu',
        initial_transform=affine_tx,
        affine_iterations=[100, 50, 20], reg_iterations=[100, 100, 20],
        grad_step=0.25, flow_sigma=3.0, syn_metric='lncc', syn_sampling=2, inverse_steps=10
    )
    reg['_elapsed'] = time.time() - t0
    return eval_registration(fi, ml, fl, reg, 'jax_syn')


# ── MPS Tasks (run sequentially on GPU) ───────────────────────────────────────

def run_pt_syn(fi, mi, fl, ml, device, affine_tx):
    """PyTorch SyN registration (MPS), initialized with shared affine."""
    t0 = time.time()
    reg = syntx.syn(
        fixed=fi, moving=mi, backend='pytorch', device=device,
        initial_transform=affine_tx,
        affine_iterations=[100, 50, 20], reg_iterations=[100, 100, 20],
        grad_step=0.25, flow_sigma=3.0, syn_metric='lncc', syn_sampling=2, inverse_steps=10
    )
    reg['_elapsed'] = time.time() - t0
    return eval_registration(fi, ml, fl, reg, 'pt_syn')


def run_tvf_endpoint(fi, mi, fl, ml, device, affine_tx):
    """TVF with endpoint loss multipoint_loss=[0.0, 1.0] (MPS), initialized with shared affine."""
    t0 = time.time()
    reg = syntx.tvf(
        fixed=fi, moving=mi, backend='pytorch', device=device,
        initial_transform=affine_tx,
        affine_iterations=[100, 50, 20], reg_iterations=[100, 100, 20],
        grad_step=0.20, flow_sigma=3.0, n_time_steps=4, syn_sampling=2,
        cfl_momentum=0.9, fast_smooth=True,
        multipoint_loss=[0.0, 1.0]
    )
    reg['_elapsed'] = time.time() - t0
    return eval_registration(fi, ml, fl, reg, 'tvf_ep')


def run_tvf_midpoint(fi, mi, fl, ml, device, affine_tx):
    """TVF with midpoint loss multipoint_loss=[0.5] (MPS), initialized with shared affine."""
    t0 = time.time()
    reg = syntx.tvf(
        fixed=fi, moving=mi, backend='pytorch', device=device,
        initial_transform=affine_tx,
        affine_iterations=[100, 50, 20], reg_iterations=[100, 100, 20],
        grad_step=0.20, flow_sigma=3.0, n_time_steps=4, syn_sampling=2,
        cfl_momentum=0.9, fast_smooth=True,
        multipoint_loss=[0.5]
    )
    reg['_elapsed'] = time.time() - t0
    return eval_registration(fi, ml, fl, reg, 'tvf_mp')


def run_tvf_noaff(fi, mi, fl, ml, device, affine_tx):
    """TVF endpoint loss pure deformable without internal affine refinement (affine_iterations=0)."""
    t0 = time.time()
    reg = syntx.tvf(
        fixed=fi, moving=mi, backend='pytorch', device=device,
        initial_transform=affine_tx,
        affine_iterations=0, reg_iterations=[100, 100, 20],
        grad_step=0.20, flow_sigma=3.0, n_time_steps=4, syn_sampling=2,
        cfl_momentum=0.9, fast_smooth=True,
        multipoint_loss=[0.0, 1.0]
    )
    reg['_elapsed'] = time.time() - t0
    return eval_registration(fi, ml, fl, reg, 'tvf_noaff')


# ── Per-Pair Processing ───────────────────────────────────────────────────────

def process_pair(idx, pair, base_path, existing_record=None):
    """Process a single Mindboggle pair. Skips columns that already exist."""
    c1, s1 = pair['cohort1'], pair['subject1']
    c2, s2 = pair['cohort2'], pair['subject2']

    res = dict(existing_record) if existing_record else {
        'pair_idx': idx,
        'fixed': s1,
        'moving': s2,
        'type': pair['type']
    }

    # Determine what needs to run
    need_ants = 'ants_dice' not in res
    need_pt_syn = 'pt_syn_dice' not in res
    need_jax_syn = 'jax_syn_dice' not in res
    need_tvf_ep = 'tvf_ep_dice' not in res
    need_tvf_mp = 'tvf_mp_dice' not in res
    need_tvf_noaff = 'tvf_noaff_dice' not in res

    if not (need_ants or need_pt_syn or need_jax_syn or need_tvf_ep or need_tvf_mp or need_tvf_noaff):
        return res

    # Load images
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

    # Select MPS device
    if torch.cuda.is_available():
        pt_device = 'cuda'
    elif torch.backends.mps.is_available():
        pt_device = 'mps'
    else:
        pt_device = 'cpu'

    print(f"\n{'='*60}")
    print(f"  [Pair {idx}] {c1}/{s1} vs {c2}/{s2} ({pair['type']})")
    print(f"  Need: ANTs={need_ants} PT_SyN={need_pt_syn} JAX_SyN={need_jax_syn} TVF_EP={need_tvf_ep} TVF_MP={need_tvf_mp}")
    print(f"{'='*60}")

    # ── Shared Affine Initialization ──
    # All methods start from the same ANTs affine to ensure fair deformable-only comparison.
    print(f"  [AFFINE] Computing shared ANTs Affine initialization...", flush=True)
    t0_aff = time.time()
    reg_affine = ants.registration(fixed=fi, moving=mi, type_of_transform='Affine')
    affine_time = time.time() - t0_aff
    affine_tx = reg_affine['fwdtransforms'][0]  # path to .mat file
    print(f"  [AFFINE] Done in {affine_time:.1f}s", flush=True)
    res['affine_init_time'] = affine_time

    # ── Run MPS tasks FIRST (no CPU contention for memory bandwidth) ──
    if need_pt_syn:
        print(f"  [MPS] Running PyTorch SyN (device={pt_device})...", flush=True)
        try:
            res.update(run_pt_syn(fi, mi, fl, ml, pt_device, affine_tx))
        except Exception as e:
            print(f"  [MPS] PyTorch SyN FAILED: {e}", flush=True)

    if need_tvf_ep:
        print(f"  [MPS] Running TVF endpoint (device={pt_device})...", flush=True)
        try:
            res.update(run_tvf_endpoint(fi, mi, fl, ml, pt_device, affine_tx))
        except Exception as e:
            print(f"  [MPS] TVF endpoint FAILED: {e}", flush=True)

    if need_tvf_mp:
        print(f"  [MPS] Running TVF midpoint (device={pt_device})...", flush=True)
        try:
            res.update(run_tvf_midpoint(fi, mi, fl, ml, pt_device, affine_tx))
        except Exception as e:
            print(f"  [MPS] TVF midpoint FAILED: {e}", flush=True)

    if need_tvf_noaff:
        print(f"  [MPS] Running TVF No-Affine (device={pt_device})...", flush=True)
        try:
            res.update(run_tvf_noaff(fi, mi, fl, ml, pt_device, affine_tx))
        except Exception as e:
            print(f"  [MPS] TVF No-Affine FAILED: {e}", flush=True)

    # ── Then run CPU tasks in parallel (after MPS is done) ──
    cpu_futures = {}
    executor = ThreadPoolExecutor(max_workers=2)

    if need_ants:
        print("  [CPU] Launching ANTs SyN...", flush=True)
        cpu_futures['ants'] = executor.submit(run_ants_task, fi, mi, fl, ml, affine_tx)

    if need_jax_syn:
        print("  [CPU] Launching JAX SyN...", flush=True)
        cpu_futures['jax_syn'] = executor.submit(run_jax_syn_task, fi, mi, fl, ml, affine_tx)

    for name, future in cpu_futures.items():
        try:
            res.update(future.result())
        except Exception as e:
            print(f"  [CPU] {name} FAILED: {e}", flush=True)

    executor.shutdown(wait=False)

    # ── Print summary ──
    print(f"\n--- [Pair {idx} Summary] ---")
    if 'ants_dice' in res:
        print(f"  ANTs SyN:      Dice={res['ants_dice']:.4f} ({res.get('ants_time', 0):.1f}s)")
    if 'pt_syn_dice' in res:
        print(f"  PyTorch SyN:   Dice={res['pt_syn_dice']:.4f} ({res.get('pt_syn_time', 0):.1f}s)")
    if 'jax_syn_dice' in res:
        print(f"  JAX SyN:       Dice={res['jax_syn_dice']:.4f} ({res.get('jax_syn_time', 0):.1f}s)")
    if 'tvf_ep_dice' in res:
        print(f"  TVF Endpoint:  Dice={res['tvf_ep_dice']:.4f} ({res.get('tvf_ep_time', 0):.1f}s)")
    if 'tvf_noaff_dice' in res:
        print(f"  TVF No-Affine: Dice={res['tvf_noaff_dice']:.4f} ({res.get('tvf_noaff_time', 0):.1f}s)")
    if 'tvf_mp_dice' in res:
        print(f"  TVF Midpoint:  Dice={res['tvf_mp_dice']:.4f} ({res.get('tvf_mp_time', 0):.1f}s)")

    return res


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import random
    base_path = '/Users/stnava/data/mindboggle/volumes'
    repo_root = os.path.dirname(os.path.abspath(__file__))
    pairs_file = os.path.join(repo_root, 'examples', 'pairs.csv')
    out_json = os.path.join(repo_root, 'benchmark_results.json')

    with open(pairs_file, 'r') as f:
        pairs = list(csv.DictReader(f))

    # Load existing results (ANTs data preserved from prior runs)
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

    # Randomized order for fair time distribution
    pair_indices = list(range(len(pairs)))
    random.seed(42)
    random.shuffle(pair_indices)

    print(f"\nStarting Mindboggle Benchmark: {len(pairs)} pairs")
    print(f"Methods: ANTs SyN | PyTorch SyN (MPS) | JAX SyN (CPU) | TVF Endpoint (MPS) | TVF No-Affine (MPS) | TVF Midpoint (MPS)")
    print(f"{'='*80}\n", flush=True)

    all_methods = ['ants_dice', 'pt_syn_dice', 'jax_syn_dice', 'tvf_ep_dice', 'tvf_noaff_dice', 'tvf_mp_dice']

    completed_count = 0
    for step_num, i in enumerate(pair_indices):
        p = pairs[i]
        existing_rec = results_map.get(i, None)

        # Skip if ALL methods are complete
        if existing_rec and all(k in existing_rec for k in all_methods):
            print(f"[{step_num+1}/{len(pairs)}] Skipping Pair {i}: all methods complete.", flush=True)
            continue

        print(f"\n[{step_num+1}/{len(pairs)}] Processing Pair {i}...", flush=True)
        r = process_pair(i, p, base_path, existing_record=existing_rec)
        results_map[i] = r
        completed_count += 1

        # Save progress after each pair
        sorted_results = [results_map[k] for k in sorted(results_map.keys())]
        with open(out_json, 'w') as f:
            json.dump(sorted_results, f, indent=2)

        # Print running statistics every 5 completed pairs
        fully_complete = [item for item in sorted_results
                          if all(k in item for k in all_methods)]
        count = len(fully_complete)
        if count > 0 and (count % 5 == 0 or count == len(pairs)):
            _print_summary_table(fully_complete, count)

    print(f"\n{'='*80}")
    print(f"Benchmark complete. {completed_count} pairs processed. Results: {out_json}")
    print(f"{'='*80}\n")


def _print_summary_table(records, count):
    """Print a formatted summary statistics table."""
    def _col(key):
        return [r[key] for r in records if key in r]

    print(f"\n{'='*115}")
    print(f"  SUMMARY STATISTICS AFTER {count} FULLY COMPLETED PAIRS")
    print(f"{'='*115}")
    print(f" {'METRIC':<35} | {'ANTs SyN':>12} | {'PT SyN':>12} | {'JAX SyN':>12} | {'TVF EP':>12} | {'TVF NoAff':>12} | {'TVF MP':>12}")
    print(f" {'-'*35}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for label, keys in [
        ('Dice (Mean)',      ['ants_dice', 'pt_syn_dice', 'jax_syn_dice', 'tvf_ep_dice', 'tvf_noaff_dice', 'tvf_mp_dice']),
        ('Dice (Median)',    ['ants_dice', 'pt_syn_dice', 'jax_syn_dice', 'tvf_ep_dice', 'tvf_noaff_dice', 'tvf_mp_dice']),
        ('Folding % (Mean)', ['ants_folding', 'pt_syn_folding', 'jax_syn_folding', 'tvf_ep_folding', 'tvf_noaff_folding', 'tvf_mp_folding']),
        ('Jac Min (Mean)',   ['ants_jac_min', 'pt_syn_jac_min', 'jax_syn_jac_min', 'tvf_ep_jac_min', 'tvf_noaff_jac_min', 'tvf_mp_jac_min']),
        ('Smooth 1st (Mean)',['ants_smooth_1st', 'pt_syn_smooth_1st', 'jax_syn_smooth_1st', 'tvf_ep_smooth_1st', 'tvf_noaff_smooth_1st', 'tvf_mp_smooth_1st']),
        ('Inv Err Mean (mm)',['ants_inv_mean', 'pt_syn_inv_mean', 'jax_syn_inv_mean', 'tvf_ep_inv_mean', 'tvf_noaff_inv_mean', 'tvf_mp_inv_mean']),
        ('Time (Mean s)',    ['ants_time', 'pt_syn_time', 'jax_syn_time', 'tvf_ep_time', 'tvf_noaff_time', 'tvf_mp_time']),
    ]:
        vals = []
        for key in keys:
            col = _col(key)
            if len(col) > 0:
                if 'Median' in label:
                    vals.append(f"{np.median(col):.4f}")
                elif 'Time' in label:
                    vals.append(f"{np.mean(col):.1f}")
                else:
                    vals.append(f"{np.mean(col):.4f}")
            else:
                vals.append("—")
        print(f" {label:<35} | {vals[0]:>12} | {vals[1]:>12} | {vals[2]:>12} | {vals[3]:>12} | {vals[4]:>12} | {vals[5]:>12}")

    print(f"{'='*115}\n")


if __name__ == '__main__':
    main()
