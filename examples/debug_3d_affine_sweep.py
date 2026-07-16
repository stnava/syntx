#!/usr/bin/env python
"""
Phase 2-3: Detailed bug hunting and optimization sweeps for 3D affine registration.
Uses SyNTo model directly to control all parameters.
"""
import sys
import os
import numpy as np
import ants
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from syntx.syn import SyNTo, grid_to_physical_affine, registration
import tempfile

# ---- Load images ----
fi_full = ants.image_read(ants.get_ants_data('mni'))
mi_full = ants.image_read(ants.get_ants_data('ch2'))
fi = ants.resample_image(fi_full, (2, 2, 2), use_voxels=False, interp_type=0)
mi = ants.resample_image(mi_full, (2, 2, 2), use_voxels=False, interp_type=0)

print(f"Fixed:  shape={fi.shape}, spacing={fi.spacing}, origin={fi.origin}")
print(f"Moving: shape={mi.shape}, spacing={mi.spacing}, origin={mi.origin}")

dim = fi.dimension
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

def compute_multilabel_dice(img1, img2, num_classes=3):
    t1 = ants.threshold_image(img1, 'Otsu', num_classes)
    t2 = ants.threshold_image(img2, 'Otsu', num_classes)
    arr1 = t1.numpy().astype(int)
    arr2 = t2.numpy().astype(int)
    dices = []
    for label in range(1, num_classes + 1):
        m1 = (arr1 == label).astype(np.float32)
        m2 = (arr2 == label).astype(np.float32)
        intersection = np.sum(m1 * m2)
        total = np.sum(m1) + np.sum(m2)
        if total > 0:
            dices.append(2.0 * intersection / total)
    return np.mean(dices) if dices else 0.0

def apply_syntx_affine_and_get_dice(model, fi, mi):
    """Apply learned affine using physical conversion and measure DICE."""
    with torch.no_grad():
        T_grid = model.affine.get_matrix().cpu().numpy()
    M_phys, t_phys = grid_to_physical_affine(T_grid, fi, mi)
    
    # Create ANTs affine transform
    tx = ants.new_ants_transform(precision='float', dimension=dim, transform_type='AffineTransform')
    tx.set_parameters(np.concatenate([M_phys.ravel(), t_phys]))
    tx.set_fixed_parameters(np.zeros(dim))
    
    tx_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False).name
    ants.write_transform(tx, tx_file)
    
    warped = ants.apply_transforms(fixed=fi, moving=mi, transformlist=[tx_file])
    dice = compute_multilabel_dice(fi, warped)
    os.unlink(tx_file)
    return dice, warped

def run_affine_model(fi, mi, levels, affine_iters, lr, smoothing_sigmas=None, verbose=False):
    """Run affine-only registration using SyNTo directly."""
    grid_shape = fi.shape
    spacing = fi.spacing
    sp_ordered = tuple(reversed(spacing))
    direction = fi.direction
    
    fi_np = fi.numpy()
    mi_np = mi.numpy()
    fi_norm = (fi_np - fi_np.mean()) / (fi_np.std() + 1e-8)
    mi_norm = (mi_np - mi_np.mean()) / (mi_np.std() + 1e-8)
    
    I_tensor = torch.tensor(fi_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    J_tensor = torch.tensor(mi_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    model = SyNTo(
        dim=dim, grid_shape=grid_shape, spacing=sp_ordered,
        direction=direction, transform_type='Affine'
    ).to(device)
    
    syn_iters = [0] * len(levels)
    
    model.fit(
        I_tensor, J_tensor,
        levels=levels,
        epochs_per_level=syn_iters,
        affine_epochs=affine_iters,
        affine_lr=lr,
        similarity_metric='mattes_mi',
        mattes_bins=32,
        smoothing_sigmas=smoothing_sigmas
    )
    
    dice, warped = apply_syntx_affine_and_get_dice(model, fi, mi)
    losses = []
    for l in model.affine_losses:
        if hasattr(l, 'item'):
            losses.append(l.item())
        else:
            losses.append(float(l))
    
    return dice, warped, model, losses

# ---- ANTs baseline ----
print("\n" + "=" * 80)
ants_result = ants.registration(fixed=fi, moving=mi, type_of_transform='Affine')
ants_dice = compute_multilabel_dice(fi, ants_result['warpedmovout'])
baseline_dice = compute_multilabel_dice(fi, mi)
print(f"ANTs Affine DICE:  {ants_dice:.6f}")
print(f"Baseline DICE:     {baseline_dice:.6f}")

# ---- Phase 2: Trace CoM init ----
print("\n" + "=" * 80)
print("PHASE 2: CoM INITIALIZATION ANALYSIS")
print("=" * 80)

# Check what happens without CoM init (manual test)
# The CoM init is in fit(), so we need to test with and without
# For now, let's first check the default behavior

# ---- Phase 3: Sweep ----
print("\n" + "=" * 80)
print("PHASE 3: OPTIMIZATION SWEEP")
print("=" * 80)

configs = [
    # (name, levels, affine_iters, lr, smoothing_sigmas)
    ("defaults",                [4, 2, 1],    [100, 50, 20],     1e-2,  None),
    ("more_iters",              [4, 2, 1],    [300, 200, 100],   1e-2,  None),
    ("8_4_2_1",                 [8, 4, 2, 1], [200, 100, 50, 20], 1e-2, None),
    ("2_1",                     [2, 1],       [300, 100],         1e-2,  None),
    ("single_1",                [1],          [500],              1e-2,  None),
    ("lr_1e-1",                 [4, 2, 1],    [200, 100, 50],    1e-1,  None),
    ("lr_5e-2",                 [4, 2, 1],    [200, 100, 50],    5e-2,  None),
    ("lr_5e-3",                 [4, 2, 1],    [200, 100, 50],    5e-3,  None),
    ("lr_1e-3",                 [4, 2, 1],    [200, 100, 50],    1e-3,  None),
    ("smooth_2_1_0",            [4, 2, 1],    [200, 100, 50],    1e-2,  [2.0, 1.0, 0.0]),
    ("smooth_3_2_0",            [4, 2, 1],    [200, 100, 50],    1e-2,  [3.0, 2.0, 0.0]),
    ("1k_single",               [1],          [1000],             1e-2,  None),
    ("lr_5e-2_more",            [4, 2, 1],    [300, 200, 100],   5e-2,  None),
    ("lr_5e-2_smooth",          [4, 2, 1],    [200, 100, 50],    5e-2,  [2.0, 1.0, 0.0]),
    ("4_2_1_big",               [4, 2, 1],    [500, 300, 200],   1e-2,  None),
    ("lr_2e-2",                 [4, 2, 1],    [300, 200, 100],   2e-2,  None),
    ("lr_3e-2",                 [4, 2, 1],    [300, 200, 100],   3e-2,  None),
]

print(f"{'Config':<30} {'DICE':>10} {'Time':>8} {'#Iters':>8} {'Init Loss':>11} {'Final Loss':>11}")
print("-" * 85)

results = []
for name, levels, aff_iters, lr, sigmas in configs:
    try:
        t0 = time.time()
        dice, warped, model, losses = run_affine_model(fi, mi, levels, aff_iters, lr, sigmas)
        elapsed = time.time() - t0
        init_loss = losses[0] if losses else float('nan')
        final_loss = losses[-1] if losses else float('nan')
        
        print(f"{name:<30} {dice:>10.6f} {elapsed:>7.1f}s {len(losses):>8} {init_loss:>11.6f} {final_loss:>11.6f}")
        results.append((name, dice, elapsed, len(losses), init_loss, final_loss))
    except Exception as e:
        import traceback
        print(f"{name:<30} ERROR: {e}")
        traceback.print_exc()
        results.append((name, 0.0, 0.0, 0, float('nan'), float('nan')))

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"ANTs Affine DICE:  {ants_dice:.6f}")
print(f"Baseline DICE:     {baseline_dice:.6f}")
if results:
    best_idx = max(range(len(results)), key=lambda i: results[i][1])
    print(f"Best syntx DICE:   {results[best_idx][1]:.6f} ({results[best_idx][0]})")
    print(f"DICE Gap (ANTs - best syntx): {ants_dice - results[best_idx][1]:.6f}")
    
    # Sort by DICE descending
    print("\nRanked by DICE:")
    for name, dice, elapsed, n, init_l, final_l in sorted(results, key=lambda x: -x[1]):
        gap = ants_dice - dice
        print(f"  {name:<30} DICE={dice:.6f}  gap={gap:+.6f}  time={elapsed:.1f}s")
