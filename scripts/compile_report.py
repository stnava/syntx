import os
import json
import ants
import numpy as np
from syntx.benchmark.data import load_mindboggle_pair
from syntx.deformation_metrics import compute_jacobian_metrics

pairs = [0, 1, 2, 41, 45, 57]
full_report = []

def get_inv_identity_stats(fi_img, fwd_warp_path, inv_warp_path):
    if not os.path.exists(fwd_warp_path) or not os.path.exists(inv_warp_path):
        return None
    try:
        fwd_warp = ants.image_read(fwd_warp_path)
        inv_warp = ants.image_read(inv_warp_path)
        inv_at_fwd = ants.apply_transforms(fixed=fi_img, moving=inv_warp, transformlist=[fwd_warp_path])
        err_vec = fwd_warp.numpy() + inv_at_fwd.numpy()
        err_mag = np.sqrt(np.sum(err_vec**2, axis=-1))
        mask = ants.get_mask(fi_img).numpy() > 0
        valid_err = err_mag[mask] if np.any(mask) else err_mag.flatten()
        return {
            'mean': float(np.mean(valid_err)),
            'p95': float(np.percentile(valid_err, 95)),
            'max': float(np.max(valid_err))
        }
    except Exception as e:
        print(f"Error computing inv identity: {e}")
        return None

for p_idx in pairs:
    print(f"Processing Pair {p_idx:02d}...")
    data = load_mindboggle_pair(p_idx)
    fi = data['fixed']

    syn_fwd = f'results/pair_{p_idx:03d}_syn_fwd_transform_0.nii.gz'
    syn_inv = f'results/pair_{p_idx:03d}_syn_inv_transform_1.nii.gz'
    syn_json_path = f'results/pair_{p_idx:03d}_syn.json'

    ants_fwd = f'results/pair_{p_idx:03d}_ants_syn_fwd_transform_0.nii.gz'
    ants_inv = f'results/pair_{p_idx:03d}_ants_syn_inv_transform_1.nii.gz'
    ants_json_path = f'results/pair_{p_idx:03d}_ants_syn.json'

    syn_json = json.load(open(syn_json_path)) if os.path.exists(syn_json_path) else {}
    ants_json = json.load(open(ants_json_path)) if os.path.exists(ants_json_path) else {}

    syn_inv_err = get_inv_identity_stats(fi, syn_fwd, syn_inv)
    ants_inv_err = get_inv_identity_stats(fi, ants_fwd, ants_inv)

    syn_jac = compute_jacobian_metrics(fi, syn_fwd) if os.path.exists(syn_fwd) else {}
    ants_jac = compute_jacobian_metrics(fi, ants_fwd) if os.path.exists(ants_fwd) else {}

    rec = {
        'pair': p_idx,
        'syntx': {
            'dice_sym': syn_json.get('dice_sym', float('nan')),
            'dice_fixed': syn_json.get('dice_fixed', float('nan')),
            'dice_moving': syn_json.get('dice_moving', float('nan')),
            'folding_pct': syn_jac.get('folding_pct', syn_json.get('folding_pct', 0.0)),
            'min_jacobian': syn_jac.get('min', syn_json.get('min_jacobian', 1.0)),
            'inv_err_mean': syn_inv_err['mean'] if syn_inv_err else float('nan'),
            'inv_err_p95': syn_inv_err['p95'] if syn_inv_err else float('nan'),
            'runtime_s': syn_json.get('runtime_seconds', float('nan'))
        },
        'ants': {
            'dice_sym': ants_json.get('dice_sym', float('nan')),
            'dice_fixed': ants_json.get('dice_fixed', float('nan')),
            'dice_moving': ants_json.get('dice_moving', float('nan')),
            'folding_pct': ants_jac.get('folding_pct', ants_json.get('folding_pct', 0.0)),
            'min_jacobian': ants_jac.get('min', ants_json.get('min_jacobian', 1.0)),
            'inv_err_mean': ants_inv_err['mean'] if ants_inv_err else float('nan'),
            'inv_err_p95': ants_inv_err['p95'] if ants_inv_err else float('nan'),
            'runtime_s': ants_json.get('runtime_seconds', float('nan'))
        }
    }
    full_report.append(rec)

os.makedirs('results', exist_ok=True)
with open('results/comprehensive_report.json', 'w') as f:
    json.dump(full_report, f, indent=2)

print("\n--- REPORT COMPILED SUCCESSFULLY ---")
