import os
import sys
import time
import json
import ants
import numpy as np
import torch
import syntx
from syntx.benchmark.data import load_mindboggle_pair
from syntx.deformation_metrics import compute_bidirectional_dice, compute_jacobian_metrics

pair_idx = int(sys.argv[1])
out_file = f"results/pair_{pair_idx:03d}_autograd_gaussian.json"
os.makedirs("results", exist_ok=True)

ants_baselines = {
    0: {"dice_sym": 0.6329, "dice_fixed": 0.6105, "dice_moving": 0.6553, "folding_pct": 0.000, "min_jacobian": 0.122, "runtime_seconds": 218.8},
    1: {"dice_sym": 0.6067, "dice_fixed": 0.6227, "dice_moving": 0.5907, "folding_pct": 0.000, "min_jacobian": 0.092, "runtime_seconds": 224.4},
    2: {"dice_sym": 0.6737, "dice_fixed": 0.6708, "dice_moving": 0.6765, "folding_pct": 0.000, "min_jacobian": 0.151, "runtime_seconds": 185.9},
    41: {"dice_sym": 0.6093, "dice_fixed": 0.5697, "dice_moving": 0.6488, "folding_pct": 0.000, "min_jacobian": 0.098, "runtime_seconds": 225.0},
    45: {"dice_sym": 0.6038, "dice_fixed": 0.5606, "dice_moving": 0.6470, "folding_pct": 0.000, "min_jacobian": 0.080, "runtime_seconds": 129.8},
    57: {"dice_sym": 0.6150, "dice_fixed": 0.6566, "dice_moving": 0.5734, "folding_pct": 0.000, "min_jacobian": 0.093, "runtime_seconds": 140.1},
}

print(f"\n=======================================================", flush=True)
print(f"  [ISOLATED PROCESS] Starting Pair {pair_idx:02d}", flush=True)
print(f"=======================================================", flush=True)

t0_start = time.time()
p = load_mindboggle_pair(pair_idx)
fi, mi, fl, ml = p['fixed'], p['moving'], p['fixed_label'], p['moving_label']

# Initial Affine Alignment
t0_aff = time.time()
res_aff = ants.registration(fixed=fi, moving=mi, type_of_transform='Affine', verbose=False)
aff_0 = res_aff['fwdtransforms'][0]
t_aff = time.time() - t0_aff

# Syntx SyN with Autograd + Gaussian Kernel
t0_syn = time.time()
res_syn = syntx.syn(
    fixed=fi, moving=mi, initial_transform=aff_0,
    backend='pytorch', device='mps',
    grad_step=0.25, flow_sigma=3.0, total_sigma=0.0,
    reg_iterations=[100, 100, 20], similarity_metric='cc2',
    use_ants_pseudo_gradient=False, use_analytical_gradients=False,
    dual_gradient=False,
    syn_sampling=2, fast_smooth=False, inverse_method='anderson',
    formulation='eulerian', kernel_type='gaussian',
    antisymmetric=True, verbose=False
)
t_syn = time.time() - t0_syn + t_aff

df_f_s, df_m_s, sym_s = compute_bidirectional_dice(fl, ml, fi, mi, res_syn['fwdtransforms'], res_syn['invtransforms'], res_syn['whichtoinvert_inv'])
fwd_warp_s = next(x for x in res_syn['fwdtransforms'] if isinstance(x, str) and x.endswith('.nii.gz'))
jac_s = compute_jacobian_metrics(fi, fwd_warp_s)

inv_errs = res_syn.get('inverse_identity_errors', {})
if 'phi_1' in inv_errs:
    inv_mean_s = float(inv_errs['phi_1'].get('mean', float('nan')))
    inv_p95_s = float(inv_errs['phi_1'].get('p95', float('nan')))
else:
    inv_mean_s = float(inv_errs.get('mean', float('nan')))
    inv_p95_s = float(inv_errs.get('p95', float('nan')))

rec = {
    'pair_idx': pair_idx,
    'status': 'SUCCESS',
    'syntx_dice_sym': float(sym_s), 'syntx_dice_fixed': float(df_f_s), 'syntx_dice_moving': float(df_m_s),
    'syntx_fold': float(jac_s['folding_pct']), 'syntx_min_jac': float(jac_s['min']),
    'syntx_inv_mean': float(inv_mean_s), 'syntx_inv_p95': float(inv_p95_s), 'syntx_time': float(t_syn),
    'ants_baseline': ants_baselines.get(pair_idx, {})
}
with open(out_file, "w") as f:
    json.dump(rec, f, indent=2)

ants_ref = ants_baselines.get(pair_idx, {})
ants_dice = ants_ref.get('dice_sym', 0.0)
print(f"CASE_COMPLETE: Pair {pair_idx:02d} | Autograd Sym: {sym_s:.4f} (F: {df_f_s:.4f}, M: {df_m_s:.4f}) | Fold: {jac_s['folding_pct']:.4f}% | MinJac: {jac_s['min']:.4f} | InvErr: {inv_mean_s:.2f}mm (P95: {inv_p95_s:.2f}mm) | Time: {t_syn:.1f}s | (ANTs C++ Baseline: {ants_dice:.4f})", flush=True)
