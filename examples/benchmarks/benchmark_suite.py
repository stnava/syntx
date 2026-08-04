import os

os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import argparse
import csv
import json
import tempfile
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import ants

try:
    ants.set_number_of_threads(1)
except AttributeError:
    pass

import torch
device_str = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

import syntx
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

def _ants_worker(fi_path, mi_path, outprefix, init_tx_path, queue):
    try:
        import os
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "4"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        import time
        import ants
        try:
            ants.set_number_of_threads(4)
        except AttributeError:
            pass

        fi = ants.image_read(fi_path)
        mi = ants.image_read(mi_path)

        t0 = time.time()
        kwargs = dict(
            fixed=fi,
            moving=mi,
            type_of_transform='SyN',
            grad_step=0.25,
            reg_iterations=[100, 100, 20],
            syn_metric='cc',
            syn_sampling=2,
            outprefix=outprefix
        )
        if init_tx_path and os.path.exists(init_tx_path):
            kwargs['initial_transform'] = init_tx_path

        reg_ants = ants.registration(**kwargs)
        elapsed = time.time() - t0
        queue.put(('ok', reg_ants['fwdtransforms'], reg_ants['invtransforms'], elapsed))
    except Exception as e:
        queue.put(('error', str(e), [], 0.0))

def run_ants_registration_isolated(fi_path, mi_path, outprefix, init_tx_path=None, timeout=600):
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    p = ctx.Process(target=_ants_worker, args=(fi_path, mi_path, outprefix, init_tx_path, queue))
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
            p.join()
        raise TimeoutError(f"ants.registration timed out after {timeout} seconds")

    if queue.empty():
        raise RuntimeError(f"ants.registration process exited unexpectedly with code {p.exitcode}")

    status, fwdtransforms, invtransforms, elapsed = queue.get()
    if status == 'error':
        raise RuntimeError(f"ants.registration failed in subprocess: {fwdtransforms}")

    return fwdtransforms, invtransforms, elapsed

def compute_smoothness_metrics_3d(disp_np, spacing):
    sp_x, sp_y, sp_z = spacing
    du_dx = (disp_np[1:, :-1, :-1] - disp_np[:-1, :-1, :-1]) / sp_x
    du_dy = (disp_np[:-1, 1:, :-1] - disp_np[:-1, :-1, :-1]) / sp_y
    du_dz = (disp_np[:-1, :-1, 1:] - disp_np[:-1, :-1, :-1]) / sp_z
    
    d2u_dx2 = (du_dx[1:, :-1, :-1] - du_dx[:-1, :-1, :-1]) / sp_x
    d2u_dy2 = (du_dy[:-1, 1:, :-1] - du_dy[:-1, :-1, :-1]) / sp_y
    d2u_dz2 = (du_dz[:-1, :-1, 1:] - du_dz[:-1, :-1, :-1]) / sp_z
    
    d2u_dxdy = (du_dx[:-1, 1:, :-1] - du_dx[:-1, :-1, :-1]) / sp_y
    d2u_dxdz = (du_dx[:-1, :-1, 1:] - du_dx[:-1, :-1, :-1]) / sp_z
    d2u_dydz = (du_dy[:-1, :-1, 1:] - du_dy[:-1, :-1, :-1]) / sp_z
    
    smooth_1st = np.mean(du_dx**2) + np.mean(du_dy**2) + np.mean(du_dz**2)
    smooth_2nd = np.mean(d2u_dx2**2) + np.mean(d2u_dy2**2) + np.mean(d2u_dz2**2) + 2 * (np.mean(d2u_dxdy**2) + np.mean(d2u_dxdz**2) + np.mean(d2u_dydz**2))
    return float(smooth_1st), float(smooth_2nd)

def plot_vector_grid(fi_img, disp_img, filename, step=10):
    mid_z = fi_img.shape[2] // 2
    f_slice = fi_img.numpy()[:, :, mid_z].T
    disp_np = disp_img.numpy()
    slice_disp = disp_np[:, :, mid_z, :]
    dx = slice_disp[:, :, 0] / fi_img.spacing[0]
    dy = slice_disp[:, :, 1] / fi_img.spacing[1]
    X, Y = dx.shape
    new_x = np.zeros((X, Y))
    new_y = np.zeros((X, Y))
    for i in range(X):
        for j in range(Y):
            new_x[i, j] = i + dx[i, j]
            new_y[i, j] = j + dy[i, j]
            
    plt.figure(figsize=(8, 8))
    plt.imshow(f_slice, cmap='gray', origin='lower')
    for i in range(0, X, step):
        plt.plot(new_x[i, :], new_y[i, :], color='red', alpha=0.8, linewidth=1.5)
    for j in range(0, Y, step):
        plt.plot(new_x[:, j], new_y[:, j], color='red', alpha=0.8, linewidth=1.5)
        
    plt.title('Vector Deformation Grid', color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, facecolor='#1e1e1e', bbox_inches='tight', dpi=150)
    plt.close()

def plot_jacobian(fi_img, jac_img, filename):
    mid_z = fi_img.shape[2] // 2
    fi_slice = fi_img.numpy()[:, :, mid_z].T
    jac_slice = jac_img.numpy()[:, :, mid_z].T
    plt.figure(figsize=(8, 8))
    plt.imshow(fi_slice, cmap='gray', origin='lower')
    H, W = jac_slice.shape
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    fold_mask = jac_slice <= 0
    contract_mask = (jac_slice > 0) & (jac_slice < 1)
    expand_mask = jac_slice >= 1
    safe_jac = np.clip(jac_slice, 1e-6, None)
    log_jac = np.log(safe_jac)
    max_log = 1.5
    red_intensity = np.clip(log_jac / max_log, 0, 1)
    blue_intensity = np.clip(-log_jac / max_log, 0, 1)
    overlay[expand_mask, 0] = 1.0
    overlay[expand_mask, 3] = red_intensity[expand_mask] * 0.8
    overlay[contract_mask, 2] = 1.0
    overlay[contract_mask, 3] = blue_intensity[contract_mask] * 0.8
    overlay[fold_mask, 1] = 1.0
    overlay[fold_mask, 3] = 1.0
    plt.imshow(overlay, origin='lower')
    plt.title('Jacobian', color='white')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, facecolor='#1e1e1e', bbox_inches='tight', dpi=150)
    plt.close()

def compute_bidirectional_dice(fl, ml, fi, mi, fwdtransforms, invtransforms, whichtoinvert_fwd=None, whichtoinvert_inv=None):
    if whichtoinvert_fwd is None:
        whichtoinvert_fwd = [False] * len(fwdtransforms)
    if whichtoinvert_inv is None:
        whichtoinvert_inv = [t.endswith('.mat') or 'GenericAffine' in t for t in invtransforms]

    # 1. Fixed Space Evaluation: warp moving labels to fixed space
    ml_warped = ants.apply_transforms(fixed=fi, moving=ml, transformlist=fwdtransforms, whichtoinvert=whichtoinvert_fwd, interpolator='nearestNeighbor')
    overlap_fixed = ants.label_overlap_measures(fl, ml_warped)
    df_fixed = overlap_fixed[~overlap_fixed['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_f = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_fixed.columns else 'TargetOverlap'
    dice_fixed = float(df_fixed[col_f].mean()) if len(df_fixed) > 0 else 0.0
    
    # 2. Moving Space Evaluation: warp fixed labels to moving space
    fl_warped = ants.apply_transforms(fixed=mi, moving=fl, transformlist=invtransforms, whichtoinvert=whichtoinvert_inv, interpolator='nearestNeighbor')
    overlap_moving = ants.label_overlap_measures(ml, fl_warped)
    df_moving = overlap_moving[~overlap_moving['Label'].astype(str).isin(['All', '0', '0.0'])]
    col_m = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df_moving.columns else 'TargetOverlap'
    dice_moving = float(df_moving[col_m].mean()) if len(df_moving) > 0 else 0.0
    
    # 3. Symmetric Mean
    dice_sym = 0.5 * (dice_fixed + dice_moving)
    
    return dice_fixed, dice_moving, dice_sym, overlap_fixed.to_dict('records')


def process_pair(args):
    idx, pair, base_path, out_dir, cached_ants = args
    c1, s1 = pair['cohort1'], pair['subject1']
    c2, s2 = pair['cohort2'], pair['subject2']
    
    f_path = os.path.join(base_path, f"{c1}_volumes", s1, 't1weighted_brain.MNI152.nii.gz')
    m_path = os.path.join(base_path, f"{c2}_volumes", s2, 't1weighted_brain.MNI152.nii.gz')
    
    fl_path = os.path.join(base_path, f"{c1}_volumes", s1, 'labels.DKT31.manual.MNI152.nii.gz')
    ml_path = os.path.join(base_path, f"{c2}_volumes", s2, 'labels.DKT31.manual.MNI152.nii.gz')
    
    fi_full = ants.image_read(f_path)
    mi_full = ants.image_read(m_path)
    
    # Crop fixed and moving images to save time and align physical boundaries
    mask_f = ants.iMath(ants.get_mask(fi_full), "MD", 20)
    fi = ants.crop_image(fi_full, mask_f)
    
    mask_m = ants.iMath(ants.get_mask(mi_full), "MD", 20)
    mi = ants.crop_image(mi_full, mask_m)
    
    has_labels = os.path.exists(fl_path) and os.path.exists(ml_path)
    if has_labels:
        fl_full = ants.image_read(fl_path)
        fl = ants.crop_image(fl_full, mask_f)
        ml_full = ants.image_read(ml_path)
        ml = ants.crop_image(ml_full, mask_m)
    else:
        fl = None
        ml = None
    
    results = dict(cached_ants) if cached_ants else {
        'pair_idx': idx,
        'fixed': s1,
        'moving': s2,
        'type': pair['type'],
    }
    
    # Pre-compute shared robust_affine transform (multi-start at low-res + Translation -> Rigid -> Similarity -> Affine)
    aff_tx = None
    try:
        reg_aff = syntx.robust_affine(fixed=fi, moving=mi, multi_start=True, mode='pytorch', verbose=False)
        aff_tx = reg_aff['fwdtransforms'][0]
    except Exception as e:
        print(f"[{idx}] Syntx robust_affine initialization failed: {e}", flush=True)
    
    # 1. ANTs Baseline
    if cached_ants is not None and cached_ants.get('ants_dice', 0.0) > 0:
        pass
    else:
        print(f"[{idx}] Running ANTs...", flush=True)
        temp_dir = tempfile.mkdtemp(prefix=f"ants_pair_{idx}_")
        try:
            fi_temp_path = os.path.join(temp_dir, "fi_cropped.nii.gz")
            mi_temp_path = os.path.join(temp_dir, "mi_cropped.nii.gz")
            ants.image_write(fi, fi_temp_path)
            ants.image_write(mi, mi_temp_path)
            outprefix = os.path.join(temp_dir, f"ants_pair_{idx}_")
            
            fwdtransforms, invtransforms, ants_time = run_ants_registration_isolated(
                fi_temp_path, mi_temp_path, outprefix, init_tx_path=aff_tx, timeout=600
            )
            results['ants_time'] = ants_time
            
            mi_ants = ants.apply_transforms(fi, mi, fwdtransforms)
            if has_labels:
                df_fixed, df_moving, df_sym, regional = compute_bidirectional_dice(fl, ml, fi, mi, fwdtransforms, invtransforms)
                results['ants_dice'] = df_sym
                results['ants_dice_fixed'] = df_fixed
                results['ants_dice_moving'] = df_moving
                results['ants_dice_sym'] = df_sym
                results['ants_regional_dice'] = regional
            else:
                results['ants_dice'] = 0.0
                results['ants_dice_fixed'] = 0.0
                results['ants_dice_moving'] = 0.0
                results['ants_dice_sym'] = 0.0
                results['ants_regional_dice'] = []
                
            jac_ants = ants.create_jacobian_determinant_image(fi, fwdtransforms[0])
            jac_ants_np = jac_ants.numpy()
            results['ants_jac_mean'] = float(jac_ants_np.mean())
            results['ants_jac_min'] = float(jac_ants_np.min())
            results['ants_jac_max'] = float(jac_ants_np.max())
            results['ants_jac_std'] = float(jac_ants_np.std())
            mask_ants = ants.get_mask(fi).numpy() > 0
            results['ants_folding'] = float(np.mean(jac_ants_np[mask_ants] <= 0) * 100)
            
            disp_ants = ants.image_read(fwdtransforms[0])
            s1_ants, s2_ants = compute_smoothness_metrics_3d(disp_ants.numpy(), disp_ants.spacing)
            results['ants_smooth_1st'] = s1_ants
            results['ants_smooth_2nd'] = s2_ants
        except Exception as e:
            print(f"[{idx}] ANTs registration failed or timed out: {e}", flush=True)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # 2. Syntx SyN (PyTorch Baseline + Sobolev Regularization)
    if cached_ants is not None and cached_ants.get('syn_dice', 0.0) > 0:
        pass
    else:
        print(f"[{idx}] Running Syntx (PyTorch SyN Sobolev alpha=2.5)...", flush=True)
        try:
            t0 = time.time()
            reg_syn = syntx.syn(
                fixed=fi, moving=mi,
                initial_transform=aff_tx,
                backend='pytorch', device=device_str,
                reg_iterations=[100, 100, 20], affine_iterations=0,
                similarity_metric='lncc', flow_sigma=3.0, total_sigma=0.0, grad_step=0.25
            )
            results['syn_time'] = time.time() - t0
            
            mi_syn = ants.apply_transforms(fi, mi, reg_syn['fwdtransforms'])
            if has_labels:
                df_fixed, df_moving, df_sym, regional = compute_bidirectional_dice(fl, ml, fi, mi, reg_syn['fwdtransforms'], reg_syn['invtransforms'], whichtoinvert_inv=reg_syn.get('whichtoinvert_inv'))
                results['syn_dice'] = df_sym
                results['syn_dice_fixed'] = df_fixed
                results['syn_dice_moving'] = df_moving
                results['syn_dice_sym'] = df_sym
                results['syn_regional_dice'] = regional
            else:
                results['syn_dice'] = 0.0
                results['syn_dice_fixed'] = 0.0
                results['syn_dice_moving'] = 0.0
                results['syn_dice_sym'] = 0.0
                results['syn_regional_dice'] = []
                
            jac_syn = ants.create_jacobian_determinant_image(fi, reg_syn['fwdtransforms'][0])
            jac_syn_np = jac_syn.numpy()
            results['syn_jac_mean'] = float(jac_syn_np.mean())
            results['syn_jac_min'] = float(jac_syn_np.min())
            results['syn_jac_max'] = float(jac_syn_np.max())
            results['syn_jac_std'] = float(jac_syn_np.std())
            mask_eval = ants.get_mask(fi).numpy() > 0
            results['syn_folding'] = float(np.mean(jac_syn_np[mask_eval] <= 0) * 100)
            
            disp_syn = ants.image_read(reg_syn['fwdtransforms'][0])
            s1_syn, s2_syn = compute_smoothness_metrics_3d(disp_syn.numpy(), disp_syn.spacing)
            results['syn_smooth_1st'] = s1_syn
            results['syn_smooth_2nd'] = s2_syn
            
            err_syn = reg_syn.get('inverse_identity_errors', {})
            results['syn_inv_err'] = float(max(err_syn.get('phi_1', {}).get('max_error', 0), err_syn.get('phi_2', {}).get('max_error', 0)))
        except Exception as e:
            print(f"[{idx}] Syntx (PyTorch SyN) failed: {e}", flush=True)

    # 3. TVF (With 100 Affine Refinement Iterations + Sobolev Regularization)
    if cached_ants is not None and cached_ants.get('tvf_dice', 0.0) > 0:
        pass
    else:
        print(f"[{idx}] Running Syntx (TVF 100 Affine Sobolev alpha=2.5)...", flush=True)
        try:
            t0 = time.time()
            reg_tvf = syntx.tvf(
                fixed=fi, moving=mi,
                initial_transform=aff_tx,
                backend='pytorch', device=device_str,
                reg_iterations=[100, 100, 20], affine_iterations=0,
                similarity_metric='lncc', multipoint_loss=[0.0, 0.5, 1.0],
                flow_sigma=0.4, total_sigma=0.05, grad_step=0.45,
                cfl_momentum=0.95, n_time_steps=3, constant_speed=True
            )
            results['tvf_time'] = time.time() - t0
            
            mi_tvf = ants.apply_transforms(fi, mi, reg_tvf['fwdtransforms'])
            if has_labels:
                df_fixed, df_moving, df_sym, regional = compute_bidirectional_dice(fl, ml, fi, mi, reg_tvf['fwdtransforms'], reg_tvf['invtransforms'], whichtoinvert_inv=reg_tvf.get('whichtoinvert_inv'))
                results['tvf_dice'] = df_sym
                results['tvf_dice_fixed'] = df_fixed
                results['tvf_dice_moving'] = df_moving
                results['tvf_dice_sym'] = df_sym
                results['tvf_regional_dice'] = regional
            else:
                results['tvf_dice'] = 0.0
                results['tvf_dice_fixed'] = 0.0
                results['tvf_dice_moving'] = 0.0
                results['tvf_dice_sym'] = 0.0
                results['tvf_regional_dice'] = []
                
            jac_tvf = ants.create_jacobian_determinant_image(fi, reg_tvf['fwdtransforms'][0])
            jac_tvf_np = jac_tvf.numpy()
            results['tvf_jac_mean'] = float(jac_tvf_np.mean())
            results['tvf_jac_min'] = float(jac_tvf_np.min())
            results['tvf_jac_max'] = float(jac_tvf_np.max())
            results['tvf_jac_std'] = float(jac_tvf_np.std())
            mask_eval = ants.get_mask(fi).numpy() > 0
            results['tvf_folding'] = float(np.mean(jac_tvf_np[mask_eval] <= 0) * 100)
            
            disp_tvf = ants.image_read(reg_tvf['fwdtransforms'][0])
            s1_tvf, s2_tvf = compute_smoothness_metrics_3d(disp_tvf.numpy(), disp_tvf.spacing)
            results['tvf_smooth_1st'] = s1_tvf
            results['tvf_smooth_2nd'] = s2_tvf
            
            err_tvf = reg_tvf.get('inverse_identity_errors', {})
            results['tvf_inv_err'] = float(max(err_tvf.get('phi_1', {}).get('max_error', 0), err_tvf.get('phi_2', {}).get('max_error', 0)))
        except Exception as e:
            print(f"[{idx}] Syntx (TVF 100 Affine) failed: {e}", flush=True)



    # In-loop GPU cache clearing and garbage collection safeguard
    import gc
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', '--num-pairs', type=int, default=None, help='Limit number of pairs')
    parser.add_argument('--pair-indices', type=str, default=None, help='Comma-separated list of specific pair indices to process (e.g. 14,44,53,55)')
    parser.add_argument('--force', action='store_true', help='Force re-evaluation of pairs even if cached')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--out-file', type=str, default='benchmark_barn.json', help='Output JSON filename')
    args = parser.parse_args()
    
    base_path = '/Users/stnava/data/mindboggle/volumes'
    pairs_file = os.path.join(os.path.dirname(__file__), '..', 'pairs.csv')
    if not os.path.exists(pairs_file):
        pairs_file = os.path.join(os.path.dirname(__file__), 'pairs.csv')
        
    if not os.path.exists(pairs_file):
        print("Run generate_benchmark_pairs.py first!")
        return
        
    with open(pairs_file, 'r') as f:
        pairs = list(csv.DictReader(f))
        
    target_indices = list(range(len(pairs)))
    if args.pair_indices:
        target_indices = [int(x.strip()) for x in args.pair_indices.split(',') if x.strip()]
    elif args.limit:
        target_indices = target_indices[:args.limit]
        
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'benchmark_vis')
    os.makedirs(out_dir, exist_ok=True)
    
    root_json = os.path.abspath(args.out_file)
    vis_json = os.path.join(out_dir, args.out_file)
    baseline_ref_json = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'benchmark_results.json'))
    
    # Load existing benchmark results cache if present
    ants_cache = {}
    for source in [baseline_ref_json, root_json]:
        if os.path.exists(source):
            try:
                with open(source, 'r') as f:
                    raw = json.load(f)
                    for item in raw:
                        if item.get('pair_idx') not in ants_cache:
                            ants_cache[item.get('pair_idx')] = item
                        else:
                            ants_cache[item.get('pair_idx')].update(item)
            except Exception:
                pass
            
    tasks = []
    for i in target_indices:
        cached = ants_cache.get(i)
        if args.force and cached:
            cached = None
        tasks.append((i, pairs[i], base_path, out_dir, cached))
        
    results = []
    
    print(f"Processing {len(tasks)} target pairs with {args.workers} workers...")

    def save_results():
        full_dict = {} if args.force else {k: dict(v) for k, v in ants_cache.items()}
        for r in results:
            idx = r['pair_idx']
            if idx in full_dict:
                full_dict[idx].update(r)
            else:
                full_dict[idx] = r
        full_output = [full_dict[k] for k in sorted(full_dict.keys())]
        with open(vis_json, 'w') as f:
            json.dump(full_output, f, indent=2)
        with open(root_json, 'w') as f:
            json.dump(full_output, f, indent=2)

    if args.workers > 1:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for r in executor.map(process_pair, tasks):
                results.append(r)
                save_results()
                print(f"[Progress] Completed pair {r['pair_idx']} ({len(results)}/{len(tasks)})", flush=True)
    else:
        import gc
        import torch
        for t in tasks:
            r = process_pair(t)
            results.append(r)
            save_results()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print(f"[Progress] Completed pair {r['pair_idx']} ({len(results)}/{len(tasks)})", flush=True)

    save_results()
    print(f"Benchmark complete. Results saved to {root_json} and {vis_json}", flush=True)
    
if __name__ == '__main__':
    main()
