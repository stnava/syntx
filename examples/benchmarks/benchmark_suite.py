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
import numpy as np
import matplotlib.pyplot as plt
import ants

try:
    ants.set_number_of_threads(1)
except AttributeError:
    pass

import syntx
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

def _ants_worker(fi_path, mi_path, outprefix, queue):
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
        reg_ants = ants.registration(
            fixed=fi,
            moving=mi,
            type_of_transform='SyN',
            grad_step=0.25,
            reg_iterations=[100, 100, 20],
            syn_metric='cc',
            syn_sampling=2,
            outprefix=outprefix
        )
        elapsed = time.time() - t0
        queue.put(('ok', reg_ants['fwdtransforms'], reg_ants['invtransforms'], elapsed))
    except Exception as e:
        queue.put(('error', str(e), [], 0.0))

def run_ants_registration_isolated(fi_path, mi_path, outprefix, timeout=600):
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    p = ctx.Process(target=_ants_worker, args=(fi_path, mi_path, outprefix, queue))
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

def process_pair(args):
    idx, pair, base_path, out_dir = args
    c1, s1 = pair['cohort1'], pair['subject1']
    c2, s2 = pair['cohort2'], pair['subject2']
    
    f_path = os.path.join(base_path, f"{c1}_volumes", s1, 't1weighted_brain.MNI152.nii.gz')
    m_path = os.path.join(base_path, f"{c2}_volumes", s2, 't1weighted_brain.MNI152.nii.gz')
    
    fl_path = os.path.join(base_path, f"{c1}_volumes", s1, 'labels.DKT31.manual.MNI152.nii.gz')
    ml_path = os.path.join(base_path, f"{c2}_volumes", s2, 'labels.DKT31.manual.MNI152.nii.gz')
    
    fi_full = ants.image_read(f_path)
    mi_full = ants.image_read(m_path)
    
    # Crop fixed and moving images to save time and align physical boundaries
    mask_f = ants.iMath(ants.get_mask(fi_full), "MD", 12)
    fi = ants.crop_image(fi_full, mask_f)
    
    mask_m = ants.iMath(ants.get_mask(mi_full), "MD", 12)
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
    
    results = {
        'pair_idx': idx,
        'fixed': s1,
        'moving': s2,
        'type': pair['type'],
        'ants_time': 0.0,
        'ants_dice': 0.0,
        'ants_smooth_1st': 0.0,
        'ants_smooth_2nd': 0.0,
        'ants_folding': 0.0,
        'pt_time': 0.0,
        'pt_dice': 0.0,
        'pt_smooth_1st': 0.0,
        'pt_smooth_2nd': 0.0,
        'pt_folding': 0.0,
        'jax_time': 0.0,
        'jax_dice': 0.0,
        'jax_smooth_1st': 0.0,
        'jax_smooth_2nd': 0.0,
        'jax_folding': 0.0,
    }
    
    # 1. ANTs
    print(f"[{idx}] Running ANTs...", flush=True)
    temp_dir = tempfile.mkdtemp(prefix=f"ants_pair_{idx}_")
    try:
        fi_temp_path = os.path.join(temp_dir, "fi_cropped.nii.gz")
        mi_temp_path = os.path.join(temp_dir, "mi_cropped.nii.gz")
        ants.image_write(fi, fi_temp_path)
        ants.image_write(mi, mi_temp_path)
        outprefix = os.path.join(temp_dir, f"ants_pair_{idx}_")
        
        fwdtransforms, invtransforms, ants_time = run_ants_registration_isolated(
            fi_temp_path, mi_temp_path, outprefix, timeout=600
        )
        results['ants_time'] = ants_time
        
        mi_ants = ants.apply_transforms(fi, mi, fwdtransforms)
        if has_labels:
            ml_ants = ants.apply_transforms(fi, ml, fwdtransforms, interpolator='nearestNeighbor')
            overlap = ants.label_overlap_measures(fl, ml_ants)
            df = overlap[(overlap['Label'] != 'All') & (overlap['Label'] != 0) & (overlap['Label'] != '0')]
            col = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df.columns else 'TargetOverlap'
            results['ants_dice'] = float(df[col].mean()) if len(df) > 0 else 0.0
            results['ants_regional_dice'] = overlap.to_dict('records') if overlap.shape[0] > 0 else []
        else:
            results['ants_dice'] = 0.0
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
        
        plot_jacobian(fi, jac_ants, os.path.join(out_dir, f'pair_{idx}_ants_jac.png'))
        plot_vector_grid(fi, disp_ants, os.path.join(out_dir, f'pair_{idx}_ants_grid.png'), step=12)
    except Exception as e:
        print(f"[{idx}] ANTs registration failed or timed out: {e}", flush=True)
        results['ants_time'] = 600.0
        results['ants_dice'] = 0.0
        results['ants_smooth_1st'] = 0.0
        results['ants_smooth_2nd'] = 0.0
        results['ants_folding'] = 0.0
        results['ants_regional_dice'] = []
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # 2. PyTorch
    print(f"[{idx}] Running Syntx (PyTorch)...", flush=True)
    try:
        t0 = time.time()
        reg_pt = syntx.syn(
            fixed=fi, moving=mi, backend='pytorch',
            affine_iterations=[100, 100, 50, 10], reg_iterations=[100, 100, 20],
            grad_step=0.25, syn_metric='lncc', lncc_radius=2, inverse_steps=10
        )
        results['pt_time'] = time.time() - t0
        
        mi_pt = ants.apply_transforms(fi, mi, reg_pt['fwdtransforms'])
        if has_labels:
            ml_pt = ants.apply_transforms(fi, ml, reg_pt['fwdtransforms'], interpolator='nearestNeighbor')
            overlap = ants.label_overlap_measures(fl, ml_pt)
            df = overlap[(overlap['Label'] != 'All') & (overlap['Label'] != 0) & (overlap['Label'] != '0')]
            col = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df.columns else 'TargetOverlap'
            results['pt_dice'] = float(df[col].mean()) if len(df) > 0 else 0.0
            results['pt_regional_dice'] = overlap.to_dict('records') if overlap.shape[0] > 0 else []
        else:
            results['pt_dice'] = 0.0
            results['pt_regional_dice'] = []
            
        jac_pt = ants.create_jacobian_determinant_image(fi, reg_pt['fwdtransforms'][0])
        jac_pt_np = jac_pt.numpy()
        results['pt_jac_mean'] = float(jac_pt_np.mean())
        results['pt_jac_min'] = float(jac_pt_np.min())
        results['pt_jac_max'] = float(jac_pt_np.max())
        results['pt_jac_std'] = float(jac_pt_np.std())
        mask_eval = ants.get_mask(fi).numpy() > 0
        results['pt_folding'] = float(np.mean(jac_pt_np[mask_eval] <= 0) * 100)
        
        disp_pt = ants.image_read(reg_pt['fwdtransforms'][0])
        s1_pt, s2_pt = compute_smoothness_metrics_3d(disp_pt.numpy(), disp_pt.spacing)
        results['pt_smooth_1st'] = s1_pt
        results['pt_smooth_2nd'] = s2_pt
        
        err_pt = reg_pt.get('inverse_identity_errors', {})
        results['pt_inv_err'] = float(max(err_pt.get('phi_1', {}).get('max_error', 0), err_pt.get('phi_2', {}).get('max_error', 0)))
        
        plot_jacobian(fi, jac_pt, os.path.join(out_dir, f'pair_{idx}_pt_jac.png'))
        plot_vector_grid(fi, disp_pt, os.path.join(out_dir, f'pair_{idx}_pt_grid.png'), step=12)
    except Exception as e:
        print(f"[{idx}] Syntx (PyTorch) registration failed: {e}", flush=True)
        results['pt_time'] = 0.0
        results['pt_dice'] = 0.0
        results['pt_smooth_1st'] = 0.0
        results['pt_smooth_2nd'] = 0.0
        results['pt_folding'] = 0.0
        results['pt_jac_mean'] = 0.0
        results['pt_jac_min'] = 0.0
        results['pt_jac_max'] = 0.0
        results['pt_jac_std'] = 0.0
        results['pt_inv_err'] = 0.0
        results['pt_regional_dice'] = []
    
    # 3. JAX
    print(f"[{idx}] Running Syntx (JAX)...", flush=True)
    try:
        t0 = time.time()
        reg_jax = syntx.syn(
            fixed=fi, moving=mi, backend='jax',
            affine_iterations=[100, 100, 50, 10], reg_iterations=[100, 100, 20],
            grad_step=0.25, syn_metric='lncc', lncc_radius=2, inverse_steps=10
        )
        results['jax_time'] = time.time() - t0
        
        mi_jax = ants.apply_transforms(fi, mi, reg_jax['fwdtransforms'])
        if has_labels:
            ml_jax = ants.apply_transforms(fi, ml, reg_jax['fwdtransforms'], interpolator='nearestNeighbor')
            overlap = ants.label_overlap_measures(fl, ml_jax)
            df = overlap[(overlap['Label'] != 'All') & (overlap['Label'] != 0) & (overlap['Label'] != '0')]
            col = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df.columns else 'TargetOverlap'
            results['jax_dice'] = float(df[col].mean()) if len(df) > 0 else 0.0
            results['jax_regional_dice'] = overlap.to_dict('records') if overlap.shape[0] > 0 else []
        else:
            results['jax_dice'] = 0.0
            results['jax_regional_dice'] = []
            
        jac_jax = ants.create_jacobian_determinant_image(fi, reg_jax['fwdtransforms'][0])
        jac_jax_np = jac_jax.numpy()
        results['jax_jac_mean'] = float(jac_jax_np.mean())
        results['jax_jac_min'] = float(jac_jax_np.min())
        results['jax_jac_max'] = float(jac_jax_np.max())
        results['jax_jac_std'] = float(jac_jax_np.std())
        mask_eval = ants.get_mask(fi).numpy() > 0
        results['jax_folding'] = float(np.mean(jac_jax_np[mask_eval] <= 0) * 100)
        
        disp_jax = ants.image_read(reg_jax['fwdtransforms'][0])
        s1_jax, s2_jax = compute_smoothness_metrics_3d(disp_jax.numpy(), disp_jax.spacing)
        results['jax_smooth_1st'] = s1_jax
        results['jax_smooth_2nd'] = s2_jax
        
        err_jax = reg_jax.get('inverse_identity_errors', {})
        results['jax_inv_err'] = float(max(err_jax.get('phi_1', {}).get('max_error', 0), err_jax.get('phi_2', {}).get('max_error', 0)))
        
        plot_jacobian(fi, jac_jax, os.path.join(out_dir, f'pair_{idx}_jax_jac.png'))
        plot_vector_grid(fi, disp_jax, os.path.join(out_dir, f'pair_{idx}_jax_grid.png'), step=12)
    except Exception as e:
        print(f"[{idx}] Syntx (JAX) registration failed: {e}", flush=True)
        results['jax_time'] = 0.0
        results['jax_dice'] = 0.0
        results['jax_smooth_1st'] = 0.0
        results['jax_smooth_2nd'] = 0.0
        results['jax_folding'] = 0.0
        results['jax_jac_mean'] = 0.0
        results['jax_jac_min'] = 0.0
        results['jax_jac_max'] = 0.0
        results['jax_jac_std'] = 0.0
        results['jax_inv_err'] = 0.0
        results['jax_regional_dice'] = []
    
    return results
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', '--num-pairs', '--pairs', type=int, default=None, help='Limit number of pairs')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    args = parser.parse_args()
    
    base_path = '/Users/stnava/data/mindboggle/volumes'
    pairs_file = os.path.join(os.path.dirname(__file__), 'pairs.csv')
    
    if not os.path.exists(pairs_file):
        print("Run generate_benchmark_pairs.py first!")
        return
        
    with open(pairs_file, 'r') as f:
        pairs = list(csv.DictReader(f))
        
    if args.limit:
        pairs = pairs[:args.limit]
        
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'benchmark_vis')
    os.makedirs(out_dir, exist_ok=True)
    
    tasks = [(i, p, base_path, out_dir) for i, p in enumerate(pairs)]
    results = []
    
    print(f"Processing {len(tasks)} pairs with {args.workers} workers...")
    
    root_json = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'benchmark_results.json'))
    vis_json = os.path.join(out_dir, 'benchmark_results.json')

    def save_results():
        with open(vis_json, 'w') as f:
            json.dump(results, f, indent=2)
        with open(root_json, 'w') as f:
            json.dump(results, f, indent=2)

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
        for t in tasks:
            r = process_pair(t)
            results.append(r)
            save_results()
            print(f"[Progress] Completed pair {r['pair_idx']} ({len(results)}/{len(tasks)})", flush=True)

    save_results()
    print(f"Benchmark complete. Results saved to {root_json} and {vis_json}", flush=True)
    
if __name__ == '__main__':
    main()
