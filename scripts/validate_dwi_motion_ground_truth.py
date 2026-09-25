#!/usr/bin/env python
"""
scripts/validate_dwi_motion_ground_truth.py
============================================
SLOW, manual-only ground-truth recovery test for the b0/DWI grouped motion-correction +
cross-registration design. NOT part of the default pytest suite (real 3D rigid-registration
recovery testing at realistic volume sizes takes minutes, not seconds); run manually:

    python scripts/validate_dwi_motion_ground_truth.py

Why ground truth instead of a similarity/overlap metric on real data
----------------------------------------------------------------------
Real DWI b0-vs-high-b data has no ground truth: any validation metric computed on the real
acquired frames (image similarity, thresholded-outline Dice/COM, edge correlation, ...) is
confounded by genuine diffusion-contrast differences between b0 and DWI tissue, which look
identical to real misalignment to any generic metric. Masking doesn't sidestep this: a mask
built from the same confounded contrast just inherits the confound (if thresholding worked as
an alignment signal, you'd register the masks themselves and be done). The only way to get an
unambiguous answer is to know the true motion by construction.

Design
------
Start from two REAL, already cross-aligned mean images (`mean_b0`, `mean_dwi`, themselves
built from real clinical data via `syntx.motion.motion_correction(reference='mean',
two_pass=True)` per group, then cross-registered) so the simulated frames have realistic
contrast/texture/SNR -- not idealized synthetic phantoms. For each simulated frame:

1. Apply a known, random small rigid jitter transform (independent per frame, both groups)
   -- ordinary within-scan head motion, recovered by each group's own two-pass
   `motion_correction`.
2. For the DWI group only, ALSO apply one FIXED known rigid transform to every simulated DWI
   frame (on top of its own jitter) -- a systematic average-pose difference between the b0
   and DWI acquisition blocks, exactly the effect the cross-registration step
   (`mean_dwi -> mean_b0` via `robust_affine`) exists to correct.
3. Corrupt with a simulated low-frequency multiplicative bias field
   (`ants.simulate_bias_field`) and additive Gaussian noise (`ants.add_noise_to_image`),
   independently per frame, approximating real scanner inhomogeneity / thermal noise.

Then run the actual proposed pipeline and compare recovered transforms against the known
injected ones directly -- no proxy metric, no contrast confound, exact ground truth.
"""

import os
import sys
import time
import argparse
import numpy as np
import ants

_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(_repo, 'src'))

from syntx.motion import motion_correction, _extract_rigid_parameters
from syntx.robust_affine import robust_affine


def random_rigid_transform(dim, center, max_trans_mm, max_rot_deg, rng):
    """Build a random small ANTsTransform (AffineTransform, exactly orthogonal rotation part).

    Returns (tx, R_inv, t_inv): `tx` is the transform used to WARP the ground-truth image
    (`apply_transforms(fixed=base, moving=base, transformlist=[tx])` samples `base` at
    `tx(point)` for each output point -- a pull-back). Registering the resulting frame back
    to a reference recovers approximately `tx`'s INVERSE, not `tx` itself, so `R_inv`/`t_inv`
    (from `tx.invert()`) -- not `R`/`t` -- are the correct ground truth to compare a
    recovered `fwdtransforms` result against.
    """
    t = rng.uniform(-max_trans_mm, max_trans_mm, size=dim)
    if dim == 3:
        axis = rng.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        angle = np.radians(rng.uniform(-max_rot_deg, max_rot_deg))
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    else:
        angle = np.radians(rng.uniform(-max_rot_deg, max_rot_deg))
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    tx = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=dim)
    tx.set_parameters(np.concatenate([R.flatten(), t]))
    tx.set_fixed_parameters(np.asarray(center, dtype=np.float64))
    tx_inv = tx.invert()
    params_inv = np.array(tx_inv.parameters)
    R_inv = params_inv[:dim * dim].reshape(dim, dim)
    t_inv = params_inv[dim * dim:dim * dim + dim]
    return tx, R_inv, t_inv


def write_tx(tx, path):
    ants.write_transform(tx, path)
    return path


def corrupt(img, rng, bias_sd=0.3, noise_sd=0.02):
    log_field = ants.simulate_bias_field(img, number_of_points=10, sd_bias_field=bias_sd,
                                          number_of_fitting_levels=3)
    field = ants.from_numpy(np.exp(log_field.numpy()).astype(np.float32))
    ants.copy_image_info(log_field, field)
    biased = img * field
    noisy = ants.add_noise_to_image(biased, noise_model='additivegaussian',
                                     noise_parameters=(0.0, noise_sd * float(img.numpy().max())))
    return noisy


def rigid_error(R_true, t_true, tx_recovered, dim):
    params = np.array(tx_recovered.parameters)
    A = params[:dim * dim].reshape(dim, dim)
    t_hat = params[dim * dim:dim * dim + dim]
    U, _, Vt = np.linalg.svd(A)
    R_hat = U @ Vt
    trans_err = float(np.linalg.norm(t_hat - t_true))
    R_diff = R_true.T @ R_hat
    ang_err = float(np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))))
    return trans_err, ang_err


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dwi-path', default='/Users/stnava/data/blast_cohorts/BIDS/SOCOM/sub-Blast-01/ses-01/dwi/sub-Blast-01_ses-01_run-001_dir-AP_dwi.nii.gz')
    p.add_argument('--bval-path', default='/Users/stnava/data/blast_cohorts/BIDS/SOCOM/sub-Blast-01/ses-01/dwi/sub-Blast-01_ses-01_run-001_dir-AP_dwi.bval')
    p.add_argument('--n-b0', type=int, default=5)
    p.add_argument('--n-dwi', type=int, default=5)
    p.add_argument('--max-trans-mm', type=float, default=2.5)
    p.add_argument('--max-rot-deg', type=float, default=3.0)
    p.add_argument('--group-trans-mm', type=float, default=2.5)
    p.add_argument('--group-rot-deg', type=float, default=1.5)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--work-dir', default='/tmp/syntx_dwi_ground_truth')
    args = p.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("[1/5] Building real, cross-aligned mean_b0 / mean_dwi from real clinical data ...")
    full_dwi = ants.image_read(args.dwi_path)
    bvals = np.loadtxt(args.bval_path)
    n = len(bvals)

    def frame(i):
        return ants.slice_image(full_dwi, axis=3, idx=int(i))

    def make_series(indices):
        slices = [frame(i) for i in indices]
        arr = np.stack([s.numpy() for s in slices], axis=-1)
        return ants.from_numpy(arr, origin=slices[0].origin + (0.0,),
                                spacing=slices[0].spacing + (full_dwi.spacing[3],),
                                direction=full_dwi.direction)

    b0_idx = np.where(bvals < 50)[0].tolist()
    bounds = b0_idx + [n]
    dwi_idx = [(a + b) // 2 for a, b in zip(bounds[:-1], bounds[1:]) if b - a > 1]

    res_b0 = motion_correction(make_series(b0_idx), reference='mean', two_pass=True,
                                backend='pytorch', type_of_transform='Rigid', verbose=False)
    mean_b0 = res_b0.reference
    res_dwi = motion_correction(make_series(dwi_idx), reference='mean', two_pass=True,
                                 backend='pytorch', type_of_transform='Rigid', verbose=False)
    mean_dwi = res_dwi.reference
    cross0 = robust_affine(fixed=mean_b0, moving=mean_dwi, mode='auto', backend='pytorch',
                            dof='rigid', verbose=False)
    mean_dwi_aligned = cross0['warpedmovout']
    print(f"    mean_b0 {mean_b0.shape}, mean_dwi_aligned {mean_dwi_aligned.shape} ready.")

    dim = mean_b0.dimension
    center = np.array(mean_b0.get_center_of_mass())

    print(f"\n[2/5] Simulating {args.n_b0} b0 + {args.n_dwi} DWI frames with KNOWN rigid "
          f"jitter (+/-{args.max_trans_mm}mm, +/-{args.max_rot_deg}deg per frame) and a KNOWN "
          f"fixed group-level bias on the DWI block "
          f"({args.group_trans_mm}mm/{args.group_rot_deg}deg), plus simulated bias field + noise.")

    tx_group, R_group, t_group = random_rigid_transform(
        dim, center, args.group_trans_mm, args.group_rot_deg, rng)
    group_path = write_tx(tx_group, os.path.join(args.work_dir, 'tx_group_true.mat'))
    print(f"    TRUE group bias transform: t={np.round(t_group, 3)} mm, "
          f"|t|={np.linalg.norm(t_group):.3f} mm")

    def simulate_group(base_img, n_frames, group_tx_path=None, label=''):
        frames = []
        truths = []
        for k in range(n_frames):
            tx_i, R_i, t_i = random_rigid_transform(dim, center, args.max_trans_mm,
                                                      args.max_rot_deg, rng)
            tx_i_path = write_tx(tx_i, os.path.join(args.work_dir, f'tx_{label}{k:02d}_true.mat'))
            tlist = ([group_tx_path] if group_tx_path else []) + [tx_i_path]
            warped = ants.apply_transforms(fixed=base_img, moving=base_img, transformlist=tlist,
                                            interpolator='linear')
            warped = corrupt(warped, rng)
            frames.append(warped)
            truths.append((R_i, t_i))
        return frames, truths

    b0_frames, b0_truth = simulate_group(mean_b0, args.n_b0, group_tx_path=None, label='b0_')
    dwi_frames, dwi_truth = simulate_group(mean_dwi_aligned, args.n_dwi, group_tx_path=group_path, label='dwi_')

    def stack(frames):
        arr = np.stack([f.numpy() for f in frames], axis=-1)
        dir3 = np.asarray(frames[0].direction)
        dir4 = np.eye(dim + 1)
        dir4[:dim, :dim] = dir3
        return ants.from_numpy(arr, origin=frames[0].origin + (0.0,),
                                spacing=frames[0].spacing + (1.0,),
                                direction=dir4)

    sim_b0_series = stack(b0_frames)
    sim_dwi_series = stack(dwi_frames)

    print("\n[3/5] Running the real pipeline: within-group two-pass motion_correction per group ...")
    t0 = time.time()
    rec_b0 = motion_correction(sim_b0_series, reference='mean', two_pass=True, backend='pytorch',
                                type_of_transform='Rigid', verbose=False)
    rec_dwi = motion_correction(sim_dwi_series, reference='mean', two_pass=True, backend='pytorch',
                                 type_of_transform='Rigid', verbose=False)
    print(f"    done in {time.time()-t0:.0f}s")

    print("\n[4/5] Cross-registering recovered mean_dwi -> recovered mean_b0 ...")
    t0 = time.time()
    cross_rec = robust_affine(fixed=rec_b0.reference, moving=rec_dwi.reference, mode='auto',
                               backend='pytorch', dof='rigid', verbose=False)
    print(f"    done in {time.time()-t0:.1f}s")

    print("\n[5/5] Comparing recovered transforms against KNOWN injected ground truth:\n")

    print("  -- Per-frame jitter recovery (b0 group) --")
    b0_trans_errs, b0_rot_errs = [], []
    for k in range(args.n_b0):
        tx_hat = ants.read_transform(rec_b0.fwdtransforms[k][0])
        R_true, t_true = b0_truth[k]
        te, ae = rigid_error(R_true, t_true, tx_hat, dim)
        b0_trans_errs.append(te); b0_rot_errs.append(ae)
        print(f"    frame {k}: translation error={te:.3f} mm, rotation error={ae:.3f} deg")
    print(f"    MEAN: translation error={np.mean(b0_trans_errs):.3f} mm, "
          f"rotation error={np.mean(b0_rot_errs):.3f} deg")

    print("\n  -- Per-frame jitter recovery (DWI group, group bias present but not yet removed) --")
    dwi_trans_errs, dwi_rot_errs = [], []
    for k in range(args.n_dwi):
        tx_hat = ants.read_transform(rec_dwi.fwdtransforms[k][0])
        R_true, t_true = dwi_truth[k]
        te, ae = rigid_error(R_true, t_true, tx_hat, dim)
        dwi_trans_errs.append(te); dwi_rot_errs.append(ae)
        print(f"    frame {k}: translation error={te:.3f} mm, rotation error={ae:.3f} deg "
              f"(this is expected to be large -- it does not yet include the group bias)")
    print(f"    MEAN: translation error={np.mean(dwi_trans_errs):.3f} mm, "
          f"rotation error={np.mean(dwi_rot_errs):.3f} deg")

    print("\n  -- GROUP-LEVEL BIAS RECOVERY (the headline number) --")
    tx_cross_hat = ants.read_transform(cross_rec['fwdtransforms'][0])
    te_group, ae_group = rigid_error(R_group, t_group, tx_cross_hat, dim)
    print(f"    TRUE:      t={np.round(t_group, 3)} mm, |t|={np.linalg.norm(t_group):.3f} mm")
    params_hat = np.array(tx_cross_hat.parameters)
    t_hat = params_hat[dim*dim:dim*dim+dim]
    print(f"    RECOVERED: t={np.round(t_hat, 3)} mm, |t|={np.linalg.norm(t_hat):.3f} mm")
    print(f"    Recovery error: translation={te_group:.3f} mm, rotation={ae_group:.3f} deg")

    print("\n  -- Composed (per-frame + group) DWI recovery: does full recovery match truth? --")
    # Composition order: apply_transforms(fixed=mean_b0, moving=frame_k,
    # transformlist=[T_cross, T_within_k]) applies T_cross FIRST (listed first) then
    # T_within_k SECOND when pulling a point from mean_b0-space back to frame_k's native
    # space -- i.e. y = T_within_k(T_cross(x)), so as matrices (column-vector convention)
    # Th_total = Th_within @ Th_cross (NOT Th_cross @ Th_within). Ground truth must use the
    # same order: the simulated frame applied group bias first, then per-frame jitter
    # (transformlist=[group_tx, tx_i]), so Th_true_total = Th_true_i @ Th_true_group.
    composed_trans_errs, composed_rot_errs = [], []
    for k in range(args.n_dwi):
        tx_within = ants.read_transform(rec_dwi.fwdtransforms[k][0])
        _, _, Th_within = _extract_rigid_parameters(tx_within, dim)
        _, _, Th_cross = _extract_rigid_parameters(tx_cross_hat, dim)
        Th_total = Th_within @ Th_cross
        A_total = Th_total[:dim, :dim]
        R_true_i, t_true_i = dwi_truth[k]
        Th_true_group = np.eye(dim + 1)
        Th_true_group[:dim, :dim] = R_group
        Th_true_group[:dim, dim] = t_group + center - R_group @ center
        Th_true_i = np.eye(dim + 1)
        Th_true_i[:dim, :dim] = R_true_i
        Th_true_i[:dim, dim] = t_true_i + center - R_true_i @ center
        Th_true_total = Th_true_i @ Th_true_group
        A_true_total = Th_true_total[:dim, :dim]
        diff_R = A_true_total.T @ A_total
        ang_err = float(np.degrees(np.arccos(np.clip((np.trace(diff_R) - 1) / 2, -1, 1))))
        b_hat = Th_total[:dim, dim]
        b_true = Th_true_total[:dim, dim]
        trans_err = float(np.linalg.norm(b_hat - b_true))
        composed_trans_errs.append(trans_err); composed_rot_errs.append(ang_err)
        print(f"    frame {k}: composed translation error={trans_err:.3f} mm, "
              f"rotation error={ang_err:.3f} deg")
    print(f"    MEAN: translation error={np.mean(composed_trans_errs):.3f} mm, "
          f"rotation error={np.mean(composed_rot_errs):.3f} deg")

    ants.image_write(mean_b0, os.path.join(args.work_dir, 'mean_b0.nii.gz'))
    ants.image_write(mean_dwi_aligned, os.path.join(args.work_dir, 'mean_dwi_aligned.nii.gz'))
    print(f"\nArtifacts written to {args.work_dir}/")


if __name__ == '__main__':
    main()
