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

print("Loading Mindboggle Pair 002...", flush=True)
p = load_mindboggle_pair(2)
fi, mi, fl, ml = p['fixed'], p['moving'], p['fixed_label'], p['moving_label']

# Initial Affine Alignment
print("Computing Initial Affine Alignment...", flush=True)
t0_aff = time.time()
res_aff = ants.registration(fixed=fi, moving=mi, type_of_transform='Affine', verbose=False)
aff_0 = res_aff['fwdtransforms'][0]
t_aff = time.time() - t0_aff
print(f"Affine Alignment finished in {t_aff:.1f}s", flush=True)

def eval_inverse_error(fi_img, fwd_warp, inv_warp):
    try:
        fwd_w = ants.image_read(fwd_warp)
        inv_w = ants.image_read(inv_warp)
        inv_at_fwd = ants.apply_transforms(fixed=fi_img, moving=inv_w, transformlist=[fwd_warp])
        err_vec = fwd_w.numpy() + inv_at_fwd.numpy()
        err_mag = np.sqrt(np.sum(err_vec**2, axis=-1))
        mask = ants.get_mask(fi_img).numpy() > 0
        valid_err = err_mag[mask] if np.any(mask) else err_mag.flatten()
        return float(np.mean(valid_err)), float(np.percentile(valid_err, 95))
    except Exception as e:
        return float('nan'), float('nan')

# 1. Standard Peak Analytical SyN (Eulerian + ITK Gaussian Kernel)
print("\n--- Running 1: Syntx Analytical SyN (kernel_type='gaussian') ---", flush=True)
t0 = time.time()
res_analytic = syntx.syn(
    fixed=fi, moving=mi, initial_transform=aff_0,
    backend='pytorch', device='mps',
    grad_step=0.25, flow_sigma=3.0, total_sigma=0.0,
    reg_iterations=[100, 100, 20], similarity_metric='cc2',
    use_ants_pseudo_gradient=True, use_analytical_gradients=True,
    dual_gradient=False,
    syn_sampling=2, fast_smooth=False, inverse_method='anderson',
    formulation='eulerian', kernel_type='gaussian',
    antisymmetric=True, verbose=False
)
t_analytic = time.time() - t0 + t_aff
df_f_1, df_m_1, sym_1 = compute_bidirectional_dice(fl, ml, fi, mi, res_analytic['fwdtransforms'], res_analytic['invtransforms'], res_analytic['whichtoinvert_inv'])
fwd_w1 = next(x for x in res_analytic['fwdtransforms'] if x.endswith('.nii.gz'))
inv_w1 = next(x for x in res_analytic['invtransforms'] if x.endswith('.nii.gz'))
jac_1 = compute_jacobian_metrics(fi, fwd_w1)
inv_m1, inv_p1 = eval_inverse_error(fi, fwd_w1, inv_w1)
print(f"Result 1 (Analytical): SymDice={sym_1:.4f} (F:{df_f_1:.4f}, M:{df_m_1:.4f}), Fold={jac_1['folding_pct']:.4f}%, MinJac={jac_1['min']:.4f}, InvErr={inv_m1:.2f}mm, Time={t_analytic:.1f}s", flush=True)

# 2. Dual-Gradient SyN (50% Analytical + 50% Autograd)
print("\n--- Running 2: Syntx Dual-Gradient SyN (50/50 Analytic + Autograd) ---", flush=True)
t0 = time.time()
res_dual = syntx.syn(
    fixed=fi, moving=mi, initial_transform=aff_0,
    backend='pytorch', device='mps',
    grad_step=0.25, flow_sigma=3.0, total_sigma=0.0,
    reg_iterations=[100, 100, 20], similarity_metric='cc2',
    dual_gradient=True, dual_gradient_weight=0.5,
    syn_sampling=2, fast_smooth=False, inverse_method='anderson',
    formulation='eulerian', kernel_type='gaussian',
    antisymmetric=True, verbose=False
)
t_dual = time.time() - t0 + t_aff
df_f_2, df_m_2, sym_2 = compute_bidirectional_dice(fl, ml, fi, mi, res_dual['fwdtransforms'], res_dual['invtransforms'], res_dual['whichtoinvert_inv'])
fwd_w2 = next(x for x in res_dual['fwdtransforms'] if x.endswith('.nii.gz'))
inv_w2 = next(x for x in res_dual['invtransforms'] if x.endswith('.nii.gz'))
jac_2 = compute_jacobian_metrics(fi, fwd_w2)
inv_m2, inv_p2 = eval_inverse_error(fi, fwd_w2, inv_w2)
print(f"Result 2 (Dual-Gradient): SymDice={sym_2:.4f} (F:{df_f_2:.4f}, M:{df_m_2:.4f}), Fold={jac_2['folding_pct']:.4f}%, MinJac={jac_2['min']:.4f}, InvErr={inv_m2:.2f}mm, Time={t_dual:.1f}s", flush=True)

out = {
    'pair': 2,
    'ants_cpp_baseline': {'dice_sym': 0.6737, 'folding_pct': 0.000, 'min_jacobian': 0.151, 'time': 185.9},
    'legacy_syntx': {'dice_sym': 0.6260, 'folding_pct': 0.028, 'min_jacobian': 0.000, 'time': 72.7},
    'analytic_gaussian': {'dice_sym': float(sym_1), 'dice_fixed': float(df_f_1), 'dice_moving': float(df_m_1), 'folding_pct': float(jac_1['folding_pct']), 'min_jacobian': float(jac_1['min']), 'inv_err_mean': float(inv_m1), 'inv_err_p95': float(inv_p1), 'time': float(t_analytic)},
    'dual_gradient': {'dice_sym': float(sym_2), 'dice_fixed': float(df_f_2), 'dice_moving': float(df_m_2), 'folding_pct': float(jac_2['folding_pct']), 'min_jacobian': float(jac_2['min']), 'inv_err_mean': float(inv_m2), 'inv_err_p95': float(inv_p2), 'time': float(t_dual)},
}

os.makedirs('results', exist_ok=True)
with open('results/pair002_dual_gradient_experiment.json', 'w') as f:
    json.dump(out, f, indent=2)

print("\n--- ALL PAIR 002 EXPERIMENTS COMPLETED SUCCESSFULLY! ---", flush=True)
