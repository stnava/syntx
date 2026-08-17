import os
import sys
import json
import ants
import numpy as np

pairs = [0, 1, 2, 41, 45, 57]
rows = []

for p in pairs:
    syn_f = f'results/pair_{p:03d}_syn.json'
    ants_f = f'results/pair_{p:03d}_ants_syn.json'
    
    sd = json.load(open(syn_f))
    ad = json.load(open(ants_f))
    
    # Calculate ANTs inverse error
    fwd_p = f'results/pair_{p:03d}_ants_syn_fwd_transform_0.nii.gz'
    inv_p = f'results/pair_{p:03d}_ants_syn_inv_transform_1.nii.gz'
    
    try:
        fwd = ants.image_read(fwd_p)
        inv = ants.image_read(inv_p)
        inv_w = ants.apply_transforms(fixed=fwd, moving=inv, transformlist=[fwd_p])
        err = fwd.numpy() + inv_w.numpy()
        err_mag = np.sqrt(np.sum(err**2, axis=-1))
        mask = ants.get_mask(fwd).numpy() > 0
        valid_err = err_mag[mask] if np.any(mask) else err_mag.flatten()
        a_inv_mean = float(np.mean(valid_err))
        a_inv_p95 = float(np.percentile(valid_err, 95))
    except Exception as e:
        a_inv_mean = float('nan')
        a_inv_p95 = float('nan')
        
    s_inv_mean = sd.get('inverse_error_mean', float('nan'))
    s_inv_p95 = sd.get('inverse_error_p95', float('nan'))
    
    row = {
        'pair': p,
        'syn_dice': sd['dice_sym'],
        'syn_df': sd.get('dice_fixed', float('nan')),
        'syn_dm': sd.get('dice_moving', float('nan')),
        'syn_fold': sd.get('folding_pct', 0.0),
        'syn_minj': sd.get('min_jacobian', 0.0),
        'syn_inv_mean': s_inv_mean,
        'syn_inv_p95': s_inv_p95,
        'syn_time': sd.get('runtime_seconds', 0.0),
        'ants_dice': ad['dice_sym'],
        'ants_df': ad.get('dice_fixed', float('nan')),
        'ants_dm': ad.get('dice_moving', float('nan')),
        'ants_fold': ad.get('folding_pct', 0.0),
        'ants_minj': ad.get('min_jacobian', 0.0),
        'ants_inv_mean': a_inv_mean,
        'ants_inv_p95': a_inv_p95,
        'ants_time': ad.get('runtime_seconds', 0.0),
    }
    rows.append(row)

with open('results/table_summary.json', 'w') as f:
    json.dump(rows, f, indent=2)

print("Saved table_summary.json successfully!")
