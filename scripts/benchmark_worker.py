import os
import sys
import time
import csv
import json
import numpy as np
import torch
import ants
import syntx

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

idx = int(sys.argv[1])
base_path = '/Users/stnava/data/mindboggle/volumes'
pairs_file = '/Users/stnava/code/syntx/examples/pairs.csv'
json_path = '/Users/stnava/code/syntx/benchmark_results_tvf.json'

with open(pairs_file, 'r') as f:
    all_pairs = list(csv.DictReader(f))

def compute_overlap(fi, ml, fwdtransforms, fl):
    ml_warped = ants.apply_transforms(fi, ml, fwdtransforms, interpolator='nearestNeighbor')
    overlap = ants.label_overlap_measures(fl, ml_warped)
    df = overlap[(overlap['Label'] != 'All') & (overlap['Label'] != 0) & (overlap['Label'] != '0')]
    col = 'TotalOrTargetOverlap' if 'TotalOrTargetOverlap' in df.columns else 'TargetOverlap'
    return float(df[col].mean()) if len(df) > 0 else 0.0

def main():
    p = all_pairs[idx]
    c1, s1 = p['cohort1'], p['subject1']
    c2, s2 = p['cohort2'], p['subject2']
    
    # Load existing to update
    results_map = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            for item in json.load(f):
                if 'pair_idx' in item:
                    results_map[item['pair_idx']] = item

    res = results_map.get(idx, {})
    if res.get('tvf_dice', 0.0) > 0:
        print(f"Pair {idx} already processed.", flush=True)
        return

    res['pair_idx'] = idx
    res['fixed_id'] = s1
    res['moving_id'] = s2
    
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
    
    print(f"Running pair {idx}: {s1} <- {s2}", flush=True)
    aff_tx = syntx.robust_affine(fixed=fi, moving=mi, multi_start=True, mode='pytorch', verbose=False)['fwdtransforms'][0]
    
    t0 = time.time()
    reg_tvf = syntx.tvf(
        fixed=fi, moving=mi, initial_transform=aff_tx,
        backend='pytorch', device='mps' if torch.backends.mps.is_available() else 'cpu',
        similarity_metric='lncc',
        multipoint_loss=[0.0, 0.5, 1.0],
        flow_sigma=2.5, total_sigma=1.0, grad_step=0.5,
        cfl_momentum=0.95, n_time_steps=3, constant_speed=True, constant_speed_relaxation=1.0,
        use_analytical_gradients=True, reg_iterations=[100, 100, 20],
        regularizer='dsti', fast_smooth=False, antisymmetric=True,
        verbose=False
    )
    res['tvf_time'] = time.time() - t0
    res['tvf_dice'] = compute_overlap(fi, ml, reg_tvf['fwdtransforms'], fl)
    
    jac_img = ants.create_jacobian_determinant_image(fi, reg_tvf['fwdtransforms'][0], do_log=False)
    jac_np = jac_img.numpy()
    mask_eval = ants.get_mask(fi).numpy() > 0
    res['tvf_folding_pct'] = float(np.mean(jac_np[mask_eval] <= 0) * 100.0) if np.sum(mask_eval) > 0 else 0.0
    res['tvf_jac_min'] = float(np.min(jac_np))
    print(f"  -> TVF Dice: {res['tvf_dice']:.4f} | Fold: {res['tvf_folding_pct']:.4f}% | Time: {res['tvf_time']:.1f}s", flush=True)

    results_map[idx] = res
    with open(json_path, 'w') as f:
        json.dump([results_map[k] for k in sorted(results_map.keys())], f, indent=2)

if __name__ == '__main__':
    main()
