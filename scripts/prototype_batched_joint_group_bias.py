#!/usr/bin/env python
"""
scripts/prototype_batched_joint_group_bias.py
================================================
Extends the working batched single-group rigid solver
(scripts/prototype_batched_motion_correction.py) to jointly estimate:

  - per-frame rigid jitter for EVERY b0 and DWI frame (as before), AND
  - ONE shared group-bias rigid transform, applied only to the DWI group's frames,

all in a SINGLE batched optimization, rather than the previous two-stage pipeline
(build mean_b0, build mean_dwi separately, THEN cross-register the two means as a
downstream step). That two-stage approach is where the ground-truth test earlier this
session found ~2.3mm/2.2deg recovery error on a true 3.24mm group bias -- noise
comparable to the effect being measured, because the group-bias estimate only sees TWO
noisy, independently-built mean images, not all the individual frames.

Joint estimation should do better: every one of the N_dwi frames directly constrains the
shared group-bias parameter through its own per-frame loss term, rather than that signal
being filtered through an intermediate mean-image estimate first.

Model
-----
  b0 frame i:  warped_i = grid_sample(frame_i; R_i, t_i)               -> compared to mean_b0
  dwi frame j: warped_j = grid_sample(frame_j; R_group @ R_j, ...)     -> compared to mean_b0

  i.e. a dwi frame's total transform composes its own per-frame jitter with the ONE shared
  group-bias transform, and ALL frames (both groups) are registered against the SAME fixed
  reference (mean_b0) in one batched objective -- no separate mean_dwi needed at all.

Usage
-----
    python scripts/prototype_batched_joint_group_bias.py
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import ants

_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_repo, 'src'))

from syntx.robust_affine import compute_center_of_mass
from syntx.core.losses import parzen_weights

sys.path.insert(0, os.path.dirname(__file__))
from prototype_batched_motion_correction import (
    batched_rodrigues, batched_mattes_mi_loss, batched_parzen_weights, rigid_error,
)

CACHE_ROOT = '/tmp/syntx_motion_bench_cache'


def joint_register(fixed_img, b0_imgs, dwi_imgs, device='mps', num_bins=32, verbose=False,
                    lambda_smooth_t=0.0, lambda_smooth_r=0.0):
    dev = torch.device(device)
    Nb0, Ndwi = len(b0_imgs), len(dwi_imgs)
    N = Nb0 + Ndwi

    t0 = time.time()
    f32 = dict(dtype=torch.float32, device=dev)
    sp_xyz = torch.tensor(fixed_img.spacing, **f32)
    orig_xyz = torch.tensor(fixed_img.origin, **f32)
    dir_xyz = torch.tensor(fixed_img.direction, **f32)
    com_f = np.asarray(compute_center_of_mass(fixed_img, weighted=True), dtype=np.float64)
    C_phys = torch.tensor(com_f, **f32)

    def to_tensor(img):
        arr = np.transpose(img.numpy().astype(np.float32), (2, 1, 0))
        return torch.from_numpy(np.ascontiguousarray(arr)).to(dev)

    fixed_t_zyx = to_tensor(fixed_img)
    all_imgs = list(b0_imgs) + list(dwi_imgs)
    moving_stack = torch.stack([to_tensor(m) for m in all_imgs], dim=0)  # [N,Z,Y,X]
    fixed_t = fixed_t_zyx.permute(2, 1, 0)
    shape_xyz = torch.tensor(list(fixed_img.shape), **f32)

    fg = fixed_t.reshape(-1) > 0.01
    idx_fg = torch.nonzero(fg, as_tuple=False).squeeze(1)
    rng = torch.Generator(device='cpu').manual_seed(42)
    n_sample = max(2000, int(0.15 * idx_fg.numel()))
    sel = idx_fg[torch.randperm(idx_fg.numel(), generator=rng)[:n_sample]].to(dev)
    vox_idx = torch.stack(torch.unravel_index(sel, fixed_t.shape), dim=-1).float()
    phys_X = orig_xyz + (vox_idx * sp_xyz) @ dir_xyz.t()
    fixed_vals = fixed_t.reshape(-1)[sel]

    lo, hi = 0.0, float(fixed_t.max())
    fixed_scaled = (fixed_vals - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
    w_y = parzen_weights(fixed_scaled, num_bins=num_bins)

    def warp_to_moving_norm(y_phys):
        y_vox = (y_phys - orig_xyz) @ torch.inverse(dir_xyz).t() / sp_xyz
        return 2.0 * (y_vox / (shape_xyz - 1.0)) - 1.0

    # Per-frame CoM offsets (all N frames, each vs. the fixed/b0 reference's own CoM).
    # For b0 frames this IS the right init for t_param directly (group component is a
    # no-op there). For dwi frames this offset conflates the shared group bias with each
    # frame's own jitter -- initializing t_param from it directly (with t_group at zero)
    # forces the optimizer to discover that split purely through gradients, which the
    # coarse Adam stage apparently doesn't reliably do within budget. Instead: seed
    # t_group from the MEAN dwi-frame CoM offset (the common component), and seed each
    # dwi frame's own t_param from its RESIDUAL after subtracting that mean (the
    # individual component) -- starting near the right decomposition instead of needing
    # to find it from scratch.
    com_offsets = np.zeros((N, 3), dtype=np.float32)
    for b, m in enumerate(all_imgs):
        com_m = np.asarray(compute_center_of_mass(m, weighted=True), dtype=np.float64)
        com_offsets[b] = (com_m - com_f).astype(np.float32)

    t_group_init = com_offsets[Nb0:].mean(axis=0) if Ndwi > 0 else np.zeros(3, dtype=np.float32)
    t_init = com_offsets.copy()
    t_init[Nb0:] -= t_group_init

    omega = torch.zeros(N, 3, device=dev, requires_grad=True)
    t_param = torch.tensor(t_init, device=dev, requires_grad=True)
    # Shared group-bias parameter (applies to the DWI frames only).
    omega_group = torch.zeros(3, device=dev, requires_grad=True)
    t_group = torch.tensor(t_group_init, device=dev, requires_grad=True)

    dwi_mask = torch.zeros(N, dtype=torch.bool, device=dev)
    dwi_mask[Nb0:] = True

    def compute_loss(X_stage, w_y_stage):
        R_frame = batched_rodrigues(omega)                       # [N,3,3]
        R_group = batched_rodrigues(omega_group.unsqueeze(0))[0]  # [3,3], shared

        Xc = X_stage - C_phys
        # Total per-frame transform y = T_total(x) = T_group(T_frame(x)) -- group applied
        # SECOND (outer), matching how the cache was built: frames were warped via
        # apply_transforms(transformlist=[group_tx, frame_tx]) i.e. group_tx applied first
        # (listed first) then frame_tx second when mapping reference->native space; the
        # recovered fwd transform approximates the INVERSE of that composition, which
        # reverses the order: T_hat = T_group^-1 . T_frame^-1, i.e. T_frame^-1 (my R_frame/
        # t_param, per-frame) applied first, T_group^-1 (my R_group/t_group, shared) applied
        # second. Both t_frame and t_group already share the same center C_phys and use the
        # canonical y=A(x-c)+c+t form, so composing is just:
        #   R_total = R_group @ R_frame,  t_total = R_group @ t_frame + t_group
        # (b0 frames use R_group=I, t_group=0 via the mask below, leaving them unchanged.)
        R_group_b = R_group.unsqueeze(0).expand(N, -1, -1)
        t_group_b = t_group.unsqueeze(0).expand(N, -1)
        apply_group = dwi_mask.float().view(N, 1, 1)
        R_eff_group = apply_group * R_group_b + (1 - apply_group) * torch.eye(3, device=dev).unsqueeze(0)
        t_eff_group = dwi_mask.float().view(N, 1) * t_group_b

        R_total = torch.bmm(R_eff_group, R_frame)                                     # [N,3,3]
        t_total = torch.einsum('nij,nj->ni', R_eff_group, t_param) + t_eff_group      # [N,3]

        y_phys = torch.einsum('nij,sj->nsi', R_total, Xc) + C_phys + t_total.unsqueeze(1)
        grid_norm = warp_to_moving_norm(y_phys)
        grid = grid_norm.view(N, 1, 1, -1, 3)
        warped = F.grid_sample(moving_stack.unsqueeze(1), grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
        warped = warped.reshape(N, -1)
        w_scaled = (warped - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
        w_x = batched_parzen_weights(w_scaled, num_bins=num_bins)
        return batched_mattes_mi_loss(w_x, w_y_stage)

    def smoothness_penalty():
        # Temporal smoothness prior: penalize frame-to-frame CHANGE in rigid parameters
        # WITHIN each group (real head motion drifts smoothly; jumping straight to the
        # global optimum for each frame independently is noisier than exploiting that
        # correlation). Never penalizes across the b0/dwi boundary -- those are separate
        # acquisition blocks with no reason to be temporally close to each other, that's
        # exactly what the group-bias parameter is for.
        if lambda_smooth_t == 0.0 and lambda_smooth_r == 0.0:
            return torch.tensor(0.0, device=dev)
        pen = torch.tensor(0.0, device=dev)
        if Nb0 > 1:
            pen = pen + lambda_smooth_t * torch.sum((t_param[1:Nb0] - t_param[0:Nb0 - 1]) ** 2)
            pen = pen + lambda_smooth_r * torch.sum((omega[1:Nb0] - omega[0:Nb0 - 1]) ** 2)
        if Ndwi > 1:
            pen = pen + lambda_smooth_t * torch.sum((t_param[Nb0 + 1:] - t_param[Nb0:-1]) ** 2)
            pen = pen + lambda_smooth_r * torch.sum((omega[Nb0 + 1:] - omega[Nb0:-1]) ** 2)
        return pen

    schedule = [
        dict(optimizer='adam', iters=130, lr_t=0.08, lr_r=0.04, sampling=0.18),
        dict(optimizer='adam', iters=50, lr_t=0.02, lr_r=0.006, sampling=0.18),
        dict(optimizer='lbfgs', iters=18, lr_t=0.2, lr_r=0.05, sampling=0.15),
        dict(optimizer='lbfgs', iters=15, lr_t=0.05, lr_r=0.01, sampling=0.3),
    ]

    for si, stage in enumerate(schedule):
        samp = stage.get('sampling', 1.0)
        if samp < 1.0:
            k = max(500, int(samp * phys_X.shape[0]))
            sub = torch.randperm(phys_X.shape[0], generator=rng)[:k].to(dev)
            X_stage, w_y_stage = phys_X[sub], w_y[sub]
        else:
            X_stage, w_y_stage = phys_X, w_y

        params = [t_param, omega, t_group, omega_group]
        if stage.get('optimizer', 'adam') == 'lbfgs':
            opt = torch.optim.LBFGS(params, lr=1.0, max_iter=stage['iters'], history_size=10,
                                     tolerance_grad=1e-9, tolerance_change=1e-11,
                                     line_search_fn='strong_wolfe')

            def closure():
                opt.zero_grad(set_to_none=True)
                l = compute_loss(X_stage, w_y_stage).sum() + smoothness_penalty()
                l.backward()
                return l

            loss_val = opt.step(closure)
            if verbose:
                print(f"    stage {si} (lbfgs): total loss = {float(loss_val):.4f}, "
                      f"t_group={t_group.detach().cpu().numpy()}")
        else:
            opt = torch.optim.Adam([
                {'params': [t_param, t_group], 'lr': stage['lr_t']},
                {'params': [omega, omega_group], 'lr': stage['lr_r']},
            ])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=stage['iters'], eta_min=stage['lr_t'] * 0.02)
            for it in range(stage['iters']):
                opt.zero_grad(set_to_none=True)
                per_frame_loss = compute_loss(X_stage, w_y_stage)
                loss = per_frame_loss.sum() + smoothness_penalty()
                loss.backward()
                opt.step()
                sched.step()
            if verbose:
                print(f"    stage {si} (adam) done: t_group={t_group.detach().cpu().numpy()}")

    elapsed = time.time() - t0
    R_frame = batched_rodrigues(omega).detach().cpu().numpy()
    t_frame = t_param.detach().cpu().numpy()
    R_group = batched_rodrigues(omega_group.unsqueeze(0))[0].detach().cpu().numpy()
    t_group_np = t_group.detach().cpu().numpy()
    return R_frame, t_frame, R_group, t_group_np, elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache-dir', default=None)
    p.add_argument('--device', default='mps')
    p.add_argument('--lambda-smooth-t', type=float, default=0.0,
                    help='Temporal smoothness weight on translation (0 = off).')
    p.add_argument('--lambda-smooth-r', type=float, default=0.0,
                    help='Temporal smoothness weight on rotation (0 = off).')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    cache_dir = args.cache_dir
    if cache_dir is None:
        subdirs = [os.path.join(CACHE_ROOT, d) for d in os.listdir(CACHE_ROOT)]
        cache_dir = max(subdirs, key=os.path.getmtime)
    print(f"[cache] {cache_dir}")

    with open(os.path.join(cache_dir, 'meta.json')) as f:
        meta = json.load(f)
    fixed_img = ants.image_read(os.path.join(cache_dir, 'mean_b0.nii.gz'))
    b0_meta = [r for r in meta['frames'] if r['group'] == 'b0']
    dwi_meta = [r for r in meta['frames'] if r['group'] == 'dwi']
    b0_imgs = [ants.image_read(r['frame_path']) for r in b0_meta]
    dwi_imgs = [ants.image_read(r['frame_path']) for r in dwi_meta]
    print(f"[data] {len(b0_imgs)} b0 + {len(dwi_imgs)} dwi frames, shape={fixed_img.shape}")

    # Ground truth for the injected GROUP bias is only present if the cache was built with
    # `bench_motion_recovery_methods.py --build --group-trans-mm ... --group-rot-deg ...`.
    # Fall back gracefully otherwise: report per-frame recovery and the estimated (but not
    # ground-truth-checked) group bias.
    group_truth_path = os.path.join(cache_dir, 'tx_group_true.mat')
    has_group_truth = os.path.exists(group_truth_path)

    print(f"\n[joint] registering {len(b0_imgs)+len(dwi_imgs)} frames + 1 shared group-bias "
          f"parameter in ONE batched optimization (device={args.device}) ...")
    R_frame, t_frame, R_group, t_group, elapsed = joint_register(
        fixed_img, b0_imgs, dwi_imgs, device=args.device, verbose=args.verbose,
        lambda_smooth_t=args.lambda_smooth_t, lambda_smooth_r=args.lambda_smooth_r)
    n_total = len(b0_imgs) + len(dwi_imgs)
    print(f"[joint] total elapsed: {elapsed:.2f}s ({elapsed/n_total:.3f}s/frame amortized)")

    print("\n-- Per-frame jitter recovery --")
    trans_errs, rot_errs = [], []
    for b, rec in enumerate(b0_meta + dwi_meta):
        R_true, t_true = np.array(rec['R_inv']), np.array(rec['t_inv'])
        te, ae = rigid_error(R_true, t_true, R_frame[b], t_frame[b])
        trans_errs.append(te); rot_errs.append(ae)
    print(f"  b0  MEAN: trans={np.mean(trans_errs[:len(b0_imgs)]):.3f}mm "
          f"rot={np.mean(rot_errs[:len(b0_imgs)]):.3f}deg")
    print(f"  dwi MEAN: trans={np.mean(trans_errs[len(b0_imgs):]):.3f}mm "
          f"rot={np.mean(rot_errs[len(b0_imgs):]):.3f}deg")

    print(f"\nEstimated group-bias: t={np.round(t_group, 3)}, |t|={np.linalg.norm(t_group):.3f} mm")
    if has_group_truth:
        tx_true = ants.read_transform(group_truth_path)
        tx_true_inv = tx_true.invert()
        params_inv = np.array(tx_true_inv.parameters)
        dim = 3
        R_true_g = params_inv[:9].reshape(3, 3)
        t_true_g = params_inv[9:12]
        te_g, ae_g = rigid_error(R_true_g, t_true_g, R_group, t_group)
        print(f"TRUE group-bias:      t={np.round(t_true_g, 3)}, |t|={np.linalg.norm(t_true_g):.3f} mm")
        print(f"Joint recovery error: translation={te_g:.3f} mm, rotation={ae_g:.3f} deg")
        print("(compare: two-stage sequential pipeline error was 2.322 mm / 2.242 deg on a "
              "true |t|=3.243mm bias -- see ground_truth_v2.log)")
    else:
        print("(no ground-truth group-bias file in this cache -- rebuild with "
              "validate_dwi_motion_ground_truth.py's cache layout to check recovery error)")


if __name__ == '__main__':
    main()
