#!/usr/bin/env python
"""
Debug 3D Affine Registration: ANTs vs syntx comprehensive comparison.
Tests grid_to_physical_affine, CoM init, optimization convergence, and multi-resolution strategy.
"""
import sys
import os
import numpy as np
import ants
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from syntx.syn import registration, SyNTo, grid_to_physical_affine, mattes_mi_loss_nd
import tempfile

# ---- Load images ----
fi = ants.resample_image(ants.image_read(ants.get_ants_data('mni')), (2, 2, 2), use_voxels=False, interp_type=0)
mi = ants.resample_image(ants.image_read(ants.get_ants_data('ch2')), (2, 2, 2), use_voxels=False, interp_type=0)

print(f"Fixed:  shape={fi.shape}, spacing={fi.spacing}, origin={fi.origin}")
print(f"Moving: shape={mi.shape}, spacing={mi.spacing}, origin={mi.origin}")
print(f"Fixed dir:  {np.array(fi.direction).diagonal()}")
print(f"Moving dir: {np.array(mi.direction).diagonal()}")

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

# ---- Phase 1: Baseline ----
print("\n" + "=" * 60)
print("PHASE 1: BASELINE COMPARISON")
print("=" * 60)

baseline_dice = compute_multilabel_dice(fi, mi)
print(f"Baseline (no reg): {baseline_dice:.6f}")

ants_result = ants.registration(fixed=fi, moving=mi, type_of_transform='Affine')
ants_dice = compute_multilabel_dice(fi, ants_result['warpedmovout'])
print(f"ANTs Affine:       {ants_dice:.6f}")

t0 = time.time()
syntx_result = registration(
    fixed=fi, moving=mi,
    type_of_transform='Affine',
    backend='pytorch',
    syn_metric='mattes_mi'
)
syntx_time = time.time() - t0
syntx_dice = compute_multilabel_dice(fi, syntx_result['warpedmovout'])
print(f"syntx Affine:      {syntx_dice:.6f} ({syntx_time:.1f}s)")
print(f"DICE Gap:          {ants_dice - syntx_dice:.6f}")
n_losses = len(syntx_result.get('affine_losses', []))
print(f"Affine iterations: {n_losses}")

# ---- Phase 2: Bug verification ----
print("\n" + "=" * 60)
print("PHASE 2: BUG VERIFICATION")
print("=" * 60)

# Test 1: Identity grid_to_physical
print("\n--- grid_to_physical_affine identity test ---")
T_id = np.eye(dim + 1)
M_id, t_id = grid_to_physical_affine(T_id, fi, mi)
print(f"M_phys diagonal: {M_id.diagonal()}")
print(f"t_phys: {t_id}")
# M[0,0] should be -1 because fixed has direction[0,0]=+1 but moving has -1
expected_x_flip = np.sign(np.array(fi.direction).reshape(3,3)[0,0]) * np.sign(np.array(mi.direction).reshape(3,3)[0,0])
print(f"Expected M[0,0] sign: {expected_x_flip} (actual: {np.sign(M_id[0,0])})")

# Test 2: CoM initialization
print("\n--- CoM initialization test ---")
fi_np = fi.numpy()
mi_np = mi.numpy()
fi_norm = (fi_np - fi_np.mean()) / (fi_np.std() + 1e-8)
mi_norm = (mi_np - mi_np.mean()) / (mi_np.std() + 1e-8)
I_t = torch.tensor(fi_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
J_t = torch.tensor(mi_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    fp = torch.clamp(I_t, min=0.0)
    mp = torch.clamp(J_t, min=0.0)
    theta = torch.eye(dim, dim+1, device=device).unsqueeze(0)
    gf = F.affine_grid(theta, size=I_t.shape, align_corners=True)
    gm = F.affine_grid(theta, size=J_t.shape, align_corners=True)
    cf = torch.stack([torch.sum(fp[0,0]*gf[0,...,k])/fp.sum() for k in range(dim)])
    cm = torch.stack([torch.sum(mp[0,0]*gm[0,...,k])/mp.sum() for k in range(dim)])
    com_diff = (cm - cf).cpu().numpy()
print(f"CoM translation init: {com_diff}")
print(f"CoM magnitude: {np.linalg.norm(com_diff):.6f}")

# Test 3: Identity warp error
print("\n--- Identity warp test ---")
sp_ordered = tuple(reversed(fi.spacing))
model = SyNTo(dim=dim, grid_shape=fi.shape, spacing=sp_ordered,
              direction=fi.direction, transform_type='Affine').to(device)
with torch.no_grad():
    grid = model.get_affine_grid(fi.shape, device)
    warped = F.grid_sample(I_t, grid, padding_mode='border', align_corners=True)
    err = (warped - I_t).abs().mean().item()
print(f"Identity warp error: {err:.10f}")

# ---- Phase 3: Multi-resolution sweep ----
print("\n" + "=" * 60)
print("PHASE 3: OPTIMIZATION SWEEP")
print("=" * 60)

configs = [
    # via registration() wrapper
    ("syntx defaults",          dict(type_of_transform='Affine', backend='pytorch', syn_metric='mattes_mi')),
    ("more_iters",              dict(type_of_transform='Affine', backend='pytorch', syn_metric='mattes_mi',
                                     affine_iterations=[300, 200, 100])),
    ("8_4_2_1",                 dict(type_of_transform='Affine', backend='pytorch', syn_metric='mattes_mi',
                                     levels=[8, 4, 2, 1], affine_iterations=[200, 100, 50, 20])),
    ("2_1",                     dict(type_of_transform='Affine', backend='pytorch', syn_metric='mattes_mi',
                                     levels=[2, 1], affine_iterations=[300, 100])),
    ("smooth_2_1_0",            dict(type_of_transform='Affine', backend='pytorch', syn_metric='mattes_mi',
                                     smoothing_sigmas=[2.0, 1.0, 0.0])),
    ("smooth_3_2_0",            dict(type_of_transform='Affine', backend='pytorch', syn_metric='mattes_mi',
                                     smoothing_sigmas=[3.0, 2.0, 0.0])),
]

print(f"{'Config':<25} {'DICE':>10} {'Time':>8} {'#Iters':>8} {'Gap':>10}")
print("-" * 65)
print(f"{'ANTs Affine':<25} {ants_dice:>10.6f}")

results = []
for name, kwargs in configs:
    try:
        t0 = time.time()
        result = registration(fixed=fi, moving=mi, **kwargs)
        elapsed = time.time() - t0
        dice = compute_multilabel_dice(fi, result['warpedmovout'])
        n = len(result.get('affine_losses', []))
        gap = ants_dice - dice
        print(f"{name:<25} {dice:>10.6f} {elapsed:>7.1f}s {n:>8} {gap:>+10.6f}")
        results.append((name, dice, elapsed, n))
    except Exception as e:
        print(f"{name:<25} ERROR: {e}")
        results.append((name, 0.0, 0.0, 0))

# ---- Summary ----
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
best = max(results, key=lambda x: x[1])
print(f"ANTs reference:    {ants_dice:.6f}")
print(f"Baseline (no reg): {baseline_dice:.6f}")
print(f"Best syntx:        {best[1]:.6f} ({best[0]})")
print(f"Gap to ANTs:       {ants_dice - best[1]:.6f}")

if ants_dice - best[1] < 0.01:
    print("\n✅ syntx Affine is within 1% DICE of ANTs — COMPETITIVE")
else:
    print("\n⚠️  Gap exceeds 1% — further optimization needed")
