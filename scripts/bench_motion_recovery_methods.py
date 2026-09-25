#!/usr/bin/env python
"""
scripts/bench_motion_recovery_methods.py
=========================================
Fast, cached benchmark harness for comparing rigid-registration METHODS on the ground-truth
recovery task from `scripts/validate_dwi_motion_ground_truth.py`. Not part of the default
pytest suite; run manually.

Design goal: separate the expensive, one-time cost (building a large, cached set of known-
truth corrupted frames from real clinical b0/DWI mean images) from the cheap, per-method cost
(a handful of single-pair rigid registrations), so new registration methods/parameters can be
evaluated in seconds, not minutes, without regenerating data.

Usage
-----
  # Step 1 (slow, run once -- or whenever N/corruption params change): build & cache the
  # ground-truth ("gt") frame set.
  python scripts/bench_motion_recovery_methods.py --build --n-b0 20 --n-dwi 20

  # Step 2 (fast, run as often as you like): evaluate one or more methods against the cache.
  python scripts/bench_motion_recovery_methods.py --methods ants_rigid,pytorch_rigid,pytorch_rigid_sift

  # List available methods.
  python scripts/bench_motion_recovery_methods.py --list-methods
"""

import os
import sys
import json
import time
import hashlib
import argparse
import numpy as np
import ants

_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_repo, 'src'))

from syntx.motion import _extract_rigid_parameters
from syntx.robust_affine import robust_affine

CACHE_ROOT = '/tmp/syntx_motion_bench_cache'


# --------------------------------------------------------------------------------------
# Ground-truth frame generation (cached)
# --------------------------------------------------------------------------------------

def _rigid_from_vec(dim, center, t_vec, omega_vec):
    """Build an ANTsTransform (and its inverse's R,t) from explicit translation + so(3)/so(2)
    rotation-vector parameters, rather than drawing them randomly -- used by both the i.i.d.
    sampler and the smooth-walk trajectory generator below."""
    if dim == 3:
        theta = np.linalg.norm(omega_vec)
        if theta < 1e-8:
            R = np.eye(3)
        else:
            axis = omega_vec / theta
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    else:
        angle = float(omega_vec[0])
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    tx = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=dim)
    tx.set_parameters(np.concatenate([R.flatten(), t_vec]))
    tx.set_fixed_parameters(np.asarray(center, dtype=np.float64))
    tx_inv = tx.invert()
    params_inv = np.array(tx_inv.parameters)
    R_inv = params_inv[:dim * dim].reshape(dim, dim)
    t_inv = params_inv[dim * dim:dim * dim + dim]
    return tx, R_inv, t_inv


def random_rigid_transform(dim, center, max_trans_mm, max_rot_deg, rng):
    t = rng.uniform(-max_trans_mm, max_trans_mm, size=dim)
    n_rot = 3 if dim == 3 else 1
    if dim == 3:
        axis = rng.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        angle = np.radians(rng.uniform(-max_rot_deg, max_rot_deg))
        omega_vec = axis * angle
    else:
        omega_vec = np.array([np.radians(rng.uniform(-max_rot_deg, max_rot_deg))])
    return _rigid_from_vec(dim, center, t, omega_vec)


def smooth_walk_rigid_transforms(dim, center, n_frames, step_trans_mm, step_rot_deg, rho, rng):
    """Ornstein-Uhlenbeck (mean-reverting random walk) trajectory of rigid transforms --
    temporally CORRELATED motion (each frame close to the previous one), unlike
    `random_rigid_transform`'s i.i.d. draws. Matches real head motion, which drifts smoothly
    rather than jumping independently frame to frame. `rho` in (0,1): closer to 1 = more
    persistent drift, closer to 0 = faster mean reversion (snappier, less smooth).
    Returns a list of (tx, R_inv, t_inv), one per frame, in temporal order."""
    n_rot = 3 if dim == 3 else 1
    step_rot = np.radians(step_rot_deg)
    t_state = np.zeros(dim)
    omega_state = np.zeros(n_rot)
    out = []
    for _ in range(n_frames):
        t_state = rho * t_state + rng.normal(scale=step_trans_mm, size=dim)
        omega_state = rho * omega_state + rng.normal(scale=step_rot, size=n_rot)
        out.append(_rigid_from_vec(dim, center, t_state.copy(), omega_state.copy()))
    return out


def jump_rigid_transforms(dim, center, n_frames, jump_frame, jump_trans_mm, jump_rot_deg,
                           jitter_trans_mm, jitter_rot_deg, rng):
    """Step-function trajectory: small i.i.d. jitter around a baseline pose for frames
    [0, jump_frame), then ONE large abrupt displacement (a single random direction/axis,
    magnitude jump_trans_mm/jump_rot_deg) at `jump_frame`, then small jitter around that NEW
    pose for the rest of the series -- e.g. "the participant coughs and shifts, then holds
    still." This specifically stress-tests capture range (can a solver's coarse stage find a
    large, sudden offset for the jump frame itself?) independent of the smooth-drift case,
    which never actually requires a large single-step jump."""
    n_rot = 3 if dim == 3 else 1
    jump_t_axis = rng.normal(size=dim)
    jump_t_axis /= np.linalg.norm(jump_t_axis)
    jump_t = jump_t_axis * jump_trans_mm
    if dim == 3:
        jump_r_axis = rng.normal(size=3)
        jump_r_axis /= np.linalg.norm(jump_r_axis)
        jump_omega = jump_r_axis * np.radians(jump_rot_deg)
    else:
        jump_omega = np.array([np.radians(jump_rot_deg) * rng.choice([-1.0, 1.0])])

    out = []
    for i in range(n_frames):
        base_t = jump_t if i >= jump_frame else np.zeros(dim)
        base_omega = jump_omega if i >= jump_frame else np.zeros(n_rot)
        t_state = base_t + rng.normal(scale=jitter_trans_mm, size=dim)
        omega_state = base_omega + rng.normal(scale=np.radians(jitter_rot_deg), size=n_rot)
        out.append(_rigid_from_vec(dim, center, t_state.copy(), omega_state.copy()))
    return out


def corrupt(img, rng, bias_sd, noise_sd):
    log_field = ants.simulate_bias_field(img, number_of_points=10, sd_bias_field=bias_sd,
                                          number_of_fitting_levels=3)
    field = ants.from_numpy(np.exp(log_field.numpy()).astype(np.float32))
    ants.copy_image_info(log_field, field)
    biased = img * field
    noisy = ants.add_noise_to_image(biased, noise_model='additivegaussian',
                                     noise_parameters=(0.0, noise_sd * float(img.numpy().max())))
    return noisy


def cache_key(args):
    payload = dict(n_b0=args.n_b0, n_dwi=args.n_dwi, max_trans_mm=args.max_trans_mm,
                   max_rot_deg=args.max_rot_deg, bias_sd=args.bias_sd, noise_sd=args.noise_sd,
                   seed=args.seed, base_dir=args.base_dir,
                   group_trans_mm=getattr(args, 'group_trans_mm', 0.0),
                   group_rot_deg=getattr(args, 'group_rot_deg', 0.0),
                   smooth=getattr(args, 'smooth', False),
                   step_trans_mm=getattr(args, 'step_trans_mm', None),
                   step_rot_deg=getattr(args, 'step_rot_deg', None),
                   rho=getattr(args, 'rho', None),
                   jump=getattr(args, 'jump', False),
                   jump_frame_frac=getattr(args, 'jump_frame_frac', None),
                   jump_trans_mm=getattr(args, 'jump_trans_mm', None),
                   jump_rot_deg=getattr(args, 'jump_rot_deg', None),
                   jitter_trans_mm=getattr(args, 'jitter_trans_mm', None),
                   jitter_rot_deg=getattr(args, 'jitter_rot_deg', None))
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return h


def build_cache(args):
    key = cache_key(args)
    cache_dir = os.path.join(CACHE_ROOT, key)
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, 'meta.json')
    if os.path.exists(meta_path) and not args.force_rebuild:
        print(f"[build] cache already exists at {cache_dir} (use --force-rebuild to regenerate)")
        return cache_dir

    print(f"[build] generating ground-truth frame set -> {cache_dir}")
    mean_b0 = ants.image_read(os.path.join(args.base_dir, 'mean_b0.nii.gz'))
    mean_dwi = ants.image_read(os.path.join(args.base_dir, 'mean_dwi_aligned.nii.gz'))
    ants.image_write(mean_b0, os.path.join(cache_dir, 'mean_b0.nii.gz'))
    ants.image_write(mean_dwi, os.path.join(cache_dir, 'mean_dwi_aligned.nii.gz'))

    dim = mean_b0.dimension
    center = np.array(mean_b0.get_center_of_mass())
    rng = np.random.default_rng(args.seed)

    meta = {'n_b0': args.n_b0, 'n_dwi': args.n_dwi, 'dim': dim, 'center': center.tolist(),
            'frames': []}

    group_trans_mm = getattr(args, 'group_trans_mm', 0.0)
    group_rot_deg = getattr(args, 'group_rot_deg', 0.0)
    group_tx_path = None
    if group_trans_mm > 0 or group_rot_deg > 0:
        tx_group, R_group_inv, t_group_inv = random_rigid_transform(
            dim, center, group_trans_mm, group_rot_deg, rng)
        group_tx_path = os.path.join(cache_dir, 'tx_group_true.mat')
        ants.write_transform(tx_group, group_tx_path)
        meta['group_bias_R_inv'] = R_group_inv.tolist()
        meta['group_bias_t_inv'] = t_group_inv.tolist()
        print(f"    injected DWI group bias: |t_inv|={np.linalg.norm(t_group_inv):.3f} mm")

    smooth = getattr(args, 'smooth', False)
    jump = getattr(args, 'jump', False)
    assert not (smooth and jump), "--smooth and --jump are mutually exclusive trajectory modes"
    walk_trajectory = {}
    if smooth:
        walk_trajectory['b0'] = smooth_walk_rigid_transforms(
            dim, center, args.n_b0, args.step_trans_mm, args.step_rot_deg, args.rho, rng)
        walk_trajectory['dwi'] = smooth_walk_rigid_transforms(
            dim, center, args.n_dwi, args.step_trans_mm, args.step_rot_deg, args.rho, rng)
    elif jump:
        jump_frame_b0 = max(1, int(round(args.jump_frame_frac * args.n_b0)))
        jump_frame_dwi = max(1, int(round(args.jump_frame_frac * args.n_dwi)))
        walk_trajectory['b0'] = jump_rigid_transforms(
            dim, center, args.n_b0, jump_frame_b0, args.jump_trans_mm, args.jump_rot_deg,
            args.jitter_trans_mm, args.jitter_rot_deg, rng)
        walk_trajectory['dwi'] = jump_rigid_transforms(
            dim, center, args.n_dwi, jump_frame_dwi, args.jump_trans_mm, args.jump_rot_deg,
            args.jitter_trans_mm, args.jitter_rot_deg, rng)
        print(f"    jump at b0 frame {jump_frame_b0}/{args.n_b0}, dwi frame "
              f"{jump_frame_dwi}/{args.n_dwi}: {args.jump_trans_mm}mm / {args.jump_rot_deg}deg")

    def simulate_group(base_img, n_frames, group_label, apply_group_tx=False):
        for k in range(n_frames):
            t0 = time.time()
            if smooth or jump:
                tx_i, R_inv, t_inv = walk_trajectory[group_label][k]
            else:
                tx_i, R_inv, t_inv = random_rigid_transform(dim, center, args.max_trans_mm,
                                                              args.max_rot_deg, rng)
            tx_path = os.path.join(cache_dir, f'tx_{group_label}_{k:03d}.mat')
            ants.write_transform(tx_i, tx_path)
            tlist = ([group_tx_path] if (apply_group_tx and group_tx_path) else []) + [tx_path]
            warped = ants.apply_transforms(fixed=base_img, moving=base_img,
                                            transformlist=tlist, interpolator='linear')
            warped = corrupt(warped, rng, args.bias_sd, args.noise_sd)
            frame_path = os.path.join(cache_dir, f'frame_{group_label}_{k:03d}.nii.gz')
            ants.image_write(warped, frame_path)
            meta['frames'].append({
                'group': group_label, 'idx': k,
                'frame_path': frame_path, 'tx_path': tx_path,
                'R_inv': R_inv.tolist(), 't_inv': t_inv.tolist(),
            })
            print(f"    {group_label} frame {k}: built in {time.time()-t0:.1f}s")

    simulate_group(mean_b0, args.n_b0, 'b0', apply_group_tx=False)
    simulate_group(mean_dwi, args.n_dwi, 'dwi', apply_group_tx=True)

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[build] done. {len(meta['frames'])} frames cached at {cache_dir}")
    return cache_dir


def load_cache(cache_dir):
    with open(os.path.join(cache_dir, 'meta.json')) as f:
        meta = json.load(f)
    mean_b0 = ants.image_read(os.path.join(cache_dir, 'mean_b0.nii.gz'))
    mean_dwi = ants.image_read(os.path.join(cache_dir, 'mean_dwi_aligned.nii.gz'))
    for rec in meta['frames']:
        rec['frame'] = ants.image_read(rec['frame_path'])
    return meta, mean_b0, mean_dwi


# --------------------------------------------------------------------------------------
# Candidate registration methods. Each takes (fixed, moving) and returns
# (fwdtransform_path, elapsed_seconds).
# --------------------------------------------------------------------------------------

def _method_ants(type_of_transform, **kwargs):
    def fn(fixed, moving):
        t0 = time.time()
        res = ants.registration(fixed=fixed, moving=moving, type_of_transform=type_of_transform,
                                 **kwargs)
        return res['fwdtransforms'][0], time.time() - t0
    return fn


def _method_robust_affine(**kwargs):
    def fn(fixed, moving):
        t0 = time.time()
        res = robust_affine(fixed=fixed, moving=moving, backend='pytorch', **kwargs)
        return res['fwdtransforms'][0], time.time() - t0
    return fn


def _schedule_single_stage(level, iters, optimizer, lr, sampling, full_grid=False):
    # NOTE: full_grid=True at level=1 (full resolution, no subsampling) is expensive --
    # especially with optimizer='lbfgs', whose strong-Wolfe line search re-evaluates the
    # loss (and its backward pass) several times per reported "iteration". Point-sampling
    # (sampling<1.0, full_grid=False) is both far cheaper AND already gives more samples
    # than needed for a precise gradient at typical sampling fractions (e.g. 5-10% of a
    # ~2M-voxel volume is still 100-200k points).
    return [dict(level=level, iters=iters, dof='rigid', lr=lr, eta_min=lr[0] * 0.01,
                 select=False, sampling=sampling, full_grid=full_grid, optimizer=optimizer)]


METHODS = {
    'ants_rigid': _method_ants('Rigid'),
    'ants_quickrigid': _method_ants('QuickRigid'),
    'ants_boldrigid': _method_ants('BOLDRigid'),
    'pytorch_rigid': _method_robust_affine(mode='auto', dof='rigid'),
    'pytorch_rigid_sift': _method_robust_affine(mode='auto', dof='rigid', enable_landmarks=True),
    'pytorch_rigid_accurate': _method_robust_affine(mode='auto', dof='rigid', n_starts=6, num_rotations=10),
    'pytorch_rigid_fullgrid': _method_robust_affine(mode='auto', dof='rigid', sampling_percentage=0.5),
    'pytorch_tournament_rigid': _method_robust_affine(mode='tournament', dof='rigid'),
    'pytorch_affine_baseline': _method_robust_affine(mode='auto', dof='affine'),
    # -- Motion-correction-tuned candidates: single-start (no cone search / tournament).
    #    All use point-sampling (not full_grid) for speed -- 10-20% of a ~2M-voxel volume
    #    is still 200k-400k points, far more than needed for a precise gradient, and much
    #    cheaper per step than the whole volume (especially with LBFGS's line search, which
    #    re-evaluates the loss+backward several times per reported "iteration").
    'pytorch_rigid_lbfgs_fine': _method_robust_affine(
        mode='pytorch', dof='rigid', multi_start=False,
        schedule=_schedule_single_stage(level=1, iters=25, optimizer='lbfgs',
                                         lr=(0.5, 0.1, 0.0, 0.0), sampling=0.1)),
    'pytorch_rigid_lbfgs_coarse_then_fine': _method_robust_affine(
        mode='pytorch', dof='rigid', multi_start=False,
        schedule=[
            dict(level=2, iters=20, dof='rigid', lr=(0.5, 0.1, 0.0, 0.0), eta_min=0.005,
                 select=False, sampling=0.2, optimizer='lbfgs'),
            dict(level=1, iters=15, dof='rigid', lr=(0.2, 0.05, 0.0, 0.0), eta_min=0.002,
                 select=False, sampling=0.1, optimizer='lbfgs'),
        ]),
    'pytorch_rigid_more_bins': _method_robust_affine(
        mode='pytorch', dof='rigid', multi_start=False, num_bins=64,
        schedule=_schedule_single_stage(level=1, iters=25, optimizer='lbfgs',
                                         lr=(0.5, 0.1, 0.0, 0.0), sampling=0.1)),
    'pytorch_rigid_lbfgs_3stage': _method_robust_affine(
        mode='pytorch', dof='rigid', multi_start=False,
        schedule=[
            dict(level=4, iters=20, dof='rigid', lr=(1.0, 0.2, 0.0, 0.0), eta_min=0.02,
                 select=False, sampling=1.0, optimizer='lbfgs'),
            dict(level=2, iters=20, dof='rigid', lr=(0.5, 0.1, 0.0, 0.0), eta_min=0.005,
                 select=False, sampling=0.2, optimizer='lbfgs'),
            dict(level=1, iters=20, dof='rigid', lr=(0.2, 0.05, 0.0, 0.0), eta_min=0.002,
                 select=False, sampling=0.15, optimizer='lbfgs'),
        ]),
    'pytorch_rigid_lbfgs_3stage_morebins': _method_robust_affine(
        mode='pytorch', dof='rigid', multi_start=False, num_bins=48,
        schedule=[
            dict(level=4, iters=20, dof='rigid', lr=(1.0, 0.2, 0.0, 0.0), eta_min=0.02,
                 select=False, sampling=1.0, optimizer='lbfgs'),
            dict(level=2, iters=20, dof='rigid', lr=(0.5, 0.1, 0.0, 0.0), eta_min=0.005,
                 select=False, sampling=0.2, optimizer='lbfgs'),
            dict(level=1, iters=25, dof='rigid', lr=(0.15, 0.03, 0.0, 0.0), eta_min=0.001,
                 select=False, sampling=0.2, optimizer='lbfgs'),
        ]),
    'pytorch_rigid_lbfgs_3stage_fast': _method_robust_affine(
        mode='pytorch', dof='rigid', multi_start=False,
        schedule=[
            dict(level=4, iters=10, dof='rigid', lr=(1.0, 0.2, 0.0, 0.0), eta_min=0.02,
                 select=False, sampling=1.0, optimizer='lbfgs'),
            dict(level=2, iters=10, dof='rigid', lr=(0.5, 0.1, 0.0, 0.0), eta_min=0.005,
                 select=False, sampling=0.15, optimizer='lbfgs'),
            dict(level=1, iters=10, dof='rigid', lr=(0.2, 0.05, 0.0, 0.0), eta_min=0.002,
                 select=False, sampling=0.1, optimizer='lbfgs'),
        ]),
    'pytorch_rigid_lbfgs_3stage_cpu': _method_robust_affine(
        mode='pytorch', dof='rigid', multi_start=False, device='cpu',
        schedule=[
            dict(level=4, iters=20, dof='rigid', lr=(1.0, 0.2, 0.0, 0.0), eta_min=0.02,
                 select=False, sampling=1.0, optimizer='lbfgs'),
            dict(level=2, iters=20, dof='rigid', lr=(0.5, 0.1, 0.0, 0.0), eta_min=0.005,
                 select=False, sampling=0.2, optimizer='lbfgs'),
            dict(level=1, iters=20, dof='rigid', lr=(0.2, 0.05, 0.0, 0.0), eta_min=0.002,
                 select=False, sampling=0.15, optimizer='lbfgs'),
        ]),
    'pytorch_rigid_lbfgs_3stage_mid': _method_robust_affine(
        mode='pytorch', dof='rigid', multi_start=False,
        schedule=[
            dict(level=4, iters=15, dof='rigid', lr=(1.0, 0.2, 0.0, 0.0), eta_min=0.02,
                 select=False, sampling=1.0, optimizer='lbfgs'),
            dict(level=2, iters=15, dof='rigid', lr=(0.5, 0.1, 0.0, 0.0), eta_min=0.005,
                 select=False, sampling=0.2, optimizer='lbfgs'),
            dict(level=1, iters=15, dof='rigid', lr=(0.2, 0.05, 0.0, 0.0), eta_min=0.002,
                 select=False, sampling=0.15, optimizer='lbfgs'),
        ]),
    'pytorch_rigid_lbfgs_2stage_mid': _method_robust_affine(
        mode='pytorch', dof='rigid', multi_start=False,
        schedule=[
            dict(level=2, iters=18, dof='rigid', lr=(0.5, 0.1, 0.0, 0.0), eta_min=0.005,
                 select=False, sampling=0.2, optimizer='lbfgs'),
            dict(level=1, iters=15, dof='rigid', lr=(0.2, 0.05, 0.0, 0.0), eta_min=0.002,
                 select=False, sampling=0.15, optimizer='lbfgs'),
        ]),
    'pytorch_rigid_lbfgs_2stage_fast': _method_robust_affine(
        mode='pytorch', dof='rigid', multi_start=False,
        schedule=[
            dict(level=2, iters=12, dof='rigid', lr=(0.5, 0.1, 0.0, 0.0), eta_min=0.005,
                 select=False, sampling=0.15, optimizer='lbfgs'),
            dict(level=1, iters=10, dof='rigid', lr=(0.2, 0.05, 0.0, 0.0), eta_min=0.002,
                 select=False, sampling=0.1, optimizer='lbfgs'),
        ]),
    'pytorch_rigid_lbfgs_4stage': _method_robust_affine(
        mode='pytorch', dof='rigid', multi_start=False,
        schedule=[
            dict(level=4, iters=20, dof='rigid', lr=(1.0, 0.2, 0.0, 0.0), eta_min=0.02,
                 select=False, sampling=1.0, optimizer='lbfgs'),
            dict(level=2, iters=20, dof='rigid', lr=(0.5, 0.1, 0.0, 0.0), eta_min=0.005,
                 select=False, sampling=0.2, optimizer='lbfgs'),
            dict(level=1, iters=20, dof='rigid', lr=(0.15, 0.03, 0.0, 0.0), eta_min=0.002,
                 select=False, sampling=0.15, optimizer='lbfgs'),
            dict(level=1, iters=20, dof='rigid', lr=(0.03, 0.006, 0.0, 0.0), eta_min=0.0003,
                 select=False, sampling=0.3, optimizer='lbfgs'),
        ]),
    'pytorch_rigid_lbfgs_4stage_finegrid': _method_robust_affine(
        mode='pytorch', dof='rigid', multi_start=False,
        schedule=[
            dict(level=4, iters=20, dof='rigid', lr=(1.0, 0.2, 0.0, 0.0), eta_min=0.02,
                 select=False, sampling=1.0, optimizer='lbfgs'),
            dict(level=2, iters=20, dof='rigid', lr=(0.5, 0.1, 0.0, 0.0), eta_min=0.005,
                 select=False, sampling=0.2, optimizer='lbfgs'),
            dict(level=1, iters=20, dof='rigid', lr=(0.15, 0.03, 0.0, 0.0), eta_min=0.002,
                 select=False, sampling=0.15, optimizer='lbfgs'),
            dict(level=1, iters=25, dof='rigid', lr=(0.02, 0.004, 0.0, 0.0), eta_min=0.0001,
                 select=False, sampling=0.5, optimizer='lbfgs'),
        ]),
    'pytorch_rigid_lbfgs_coarse_then_fine_x2': _method_robust_affine(
        mode='pytorch', dof='rigid', multi_start=False,
        schedule=[
            dict(level=2, iters=20, dof='rigid', lr=(0.5, 0.1, 0.0, 0.0), eta_min=0.005,
                 select=False, sampling=0.2, optimizer='lbfgs'),
            dict(level=1, iters=15, dof='rigid', lr=(0.2, 0.05, 0.0, 0.0), eta_min=0.002,
                 select=False, sampling=0.1, optimizer='lbfgs'),
            dict(level=1, iters=15, dof='rigid', lr=(0.05, 0.01, 0.0, 0.0), eta_min=0.0005,
                 select=False, sampling=0.25, optimizer='lbfgs'),
        ]),
    # Coarse Adam (broad capture -- robust to the large real-world jumps a rigid motion
    # series can have, since Adam doesn't rely on a good local quadratic approximation
    # early on the way LBFGS does) followed by fine LBFGS polish (precise convergence near
    # the optimum LBFGS finds quickly once it's actually near one). Still single-start
    # (Identity_CoM) -- no cone search / tournament.
    'pytorch_rigid_adam_then_lbfgs': _method_robust_affine(
        mode='pytorch', dof='rigid', multi_start=False,
        schedule=[
            dict(level=2, iters=60, dof='rigid', lr=(0.06, 0.012, 0.0, 0.0), eta_min=0.002,
                 select=False, sampling=0.2, optimizer='adam'),
            dict(level=1, iters=15, dof='rigid', lr=(0.2, 0.05, 0.0, 0.0), eta_min=0.002,
                 select=False, sampling=0.1, optimizer='lbfgs'),
        ]),
    # Wide-capture variant: coarse Adam starts at level=4 (heavy downsample, cheap, large
    # basin of attraction) for genuinely large real-world jumps, then narrows.
    'pytorch_rigid_adam_then_lbfgs_wide': _method_robust_affine(
        mode='pytorch', dof='rigid', multi_start=False,
        schedule=[
            dict(level=4, iters=80, dof='rigid', lr=(0.08, 0.02, 0.0, 0.0), eta_min=0.002,
                 select=False, sampling=1.0, optimizer='adam'),
            dict(level=2, iters=40, dof='rigid', lr=(0.03, 0.008, 0.0, 0.0), eta_min=0.001,
                 select=False, sampling=0.2, optimizer='adam'),
            dict(level=1, iters=15, dof='rigid', lr=(0.2, 0.05, 0.0, 0.0), eta_min=0.002,
                 select=False, sampling=0.1, optimizer='lbfgs'),
        ]),
}


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------

def rigid_error(R_true, t_true, tx_path, dim):
    tx = ants.read_transform(tx_path)
    params = np.array(tx.parameters)
    A = params[:dim * dim].reshape(dim, dim)
    t_hat = params[dim * dim:dim * dim + dim]
    U, _, Vt = np.linalg.svd(A)
    R_hat = U @ Vt
    trans_err = float(np.linalg.norm(t_hat - t_true))
    R_diff = np.asarray(R_true).T @ R_hat
    ang_err = float(np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))))
    return trans_err, ang_err


def evaluate(cache_dir, method_names, group_filter=None):
    meta, mean_b0, mean_dwi = load_cache(cache_dir)
    dim = meta['dim']
    frames = meta['frames']
    if group_filter:
        frames = [f for f in frames if f['group'] == group_filter]

    print(f"\nEvaluating {len(method_names)} method(s) on {len(frames)} cached frame(s) "
          f"({cache_dir})\n")
    print(f"{'method':28s} {'group':5s} {'n':>3s} {'trans_err(mm)':>16s} {'rot_err(deg)':>14s} "
          f"{'time/frame(s)':>14s}")
    print("-" * 90)

    results = {}
    for name in method_names:
        fn = METHODS[name]
        trans_errs, rot_errs, times = [], [], []
        for rec in frames:
            fixed = mean_b0 if rec['group'] == 'b0' else mean_dwi
            moving = rec['frame']
            try:
                tx_path, elapsed = fn(fixed, moving)
            except Exception as e:
                print(f"{name:28s} FRAME {rec['group']}{rec['idx']} FAILED: {e}")
                continue
            te, ae = rigid_error(np.array(rec['R_inv']), np.array(rec['t_inv']), tx_path, dim)
            trans_errs.append(te); rot_errs.append(ae); times.append(elapsed)
        for grp in (['b0', 'dwi'] if group_filter is None else [group_filter]):
            idxs = [i for i, r in enumerate(frames) if r['group'] == grp]
            if not idxs:
                continue
            te_g = [trans_errs[i] for i in idxs if i < len(trans_errs)]
            ae_g = [rot_errs[i] for i in idxs if i < len(rot_errs)]
            t_g = [times[i] for i in idxs if i < len(times)]
            if not te_g:
                continue
            print(f"{name:28s} {grp:5s} {len(te_g):3d} {np.mean(te_g):16.3f} "
                  f"{np.mean(ae_g):14.3f} {np.mean(t_g):14.2f}")
        results[name] = {'trans_errs': trans_errs, 'rot_errs': rot_errs, 'times': times}
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--build', action='store_true', help='(Re)build the cached ground-truth frame set.')
    p.add_argument('--force-rebuild', action='store_true')
    p.add_argument('--base-dir', default='/tmp/syntx_dwi_ground_truth',
                    help='Directory containing mean_b0.nii.gz / mean_dwi_aligned.nii.gz.')
    p.add_argument('--n-b0', type=int, default=10)
    p.add_argument('--n-dwi', type=int, default=10)
    p.add_argument('--max-trans-mm', type=float, default=2.5)
    p.add_argument('--max-rot-deg', type=float, default=3.0)
    p.add_argument('--bias-sd', type=float, default=0.3)
    p.add_argument('--noise-sd', type=float, default=0.02)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--group-trans-mm', type=float, default=0.0,
                    help='If >0, inject a fixed known rigid bias on top of every DWI frame '
                         '(on top of its own jitter), for testing group-bias recovery.')
    p.add_argument('--group-rot-deg', type=float, default=0.0)
    p.add_argument('--smooth', action='store_true',
                    help='Generate a temporally-correlated (Ornstein-Uhlenbeck random-walk) '
                         'motion trajectory instead of i.i.d. per-frame draws.')
    p.add_argument('--step-trans-mm', type=float, default=0.4,
                    help='(--smooth) per-step translation noise std (mm).')
    p.add_argument('--step-rot-deg', type=float, default=0.5,
                    help='(--smooth) per-step rotation noise std (deg).')
    p.add_argument('--rho', type=float, default=0.9,
                    help='(--smooth) OU mean-reversion persistence, in (0,1); closer to 1 = '
                         'smoother/more persistent drift.')
    p.add_argument('--jump', action='store_true',
                    help='Generate a step-function trajectory: small jitter, ONE large abrupt '
                         'displacement partway through, small jitter around the new pose after '
                         '-- e.g. "participant coughs and shifts, then holds still." Mutually '
                         'exclusive with --smooth.')
    p.add_argument('--jump-frame-frac', type=float, default=0.5,
                    help='(--jump) fraction of the way through each group where the jump occurs.')
    p.add_argument('--jump-trans-mm', type=float, default=15.0,
                    help='(--jump) magnitude of the single large translation jump (mm).')
    p.add_argument('--jump-rot-deg', type=float, default=15.0,
                    help='(--jump) magnitude of the single large rotation jump (deg).')
    p.add_argument('--jitter-trans-mm', type=float, default=0.3,
                    help='(--jump) baseline small-jitter translation std (mm), before/after the jump.')
    p.add_argument('--jitter-rot-deg', type=float, default=0.4,
                    help='(--jump) baseline small-jitter rotation std (deg), before/after the jump.')
    p.add_argument('--methods', default='ants_rigid,pytorch_rigid')
    p.add_argument('--group', default=None, choices=[None, 'b0', 'dwi'])
    p.add_argument('--list-methods', action='store_true')
    args = p.parse_args()

    if args.list_methods:
        print("Available methods:")
        for name in METHODS:
            print(f"  {name}")
        return

    cache_dir = os.path.join(CACHE_ROOT, cache_key(args))
    if args.build:
        cache_dir = build_cache(args)
    elif not os.path.exists(os.path.join(cache_dir, 'meta.json')):
        print(f"No cache found at {cache_dir}. Run with --build first.")
        sys.exit(1)

    method_names = args.methods.split(',')
    unknown = [m for m in method_names if m not in METHODS]
    if unknown:
        print(f"Unknown method(s): {unknown}. Use --list-methods to see options.")
        sys.exit(1)

    evaluate(cache_dir, method_names, group_filter=args.group)


if __name__ == '__main__':
    main()
