#!/usr/bin/env python
"""
scripts/prototype_batched_motion_correction.py
================================================
PROTOTYPE: batched rigid registration of an entire time series to a single reference in
ONE GPU forward/backward pass per optimization step, instead of N sequential single-pair
calls (what `motion_correction`/`robust_affine`/`ants.registration` all do today).

Why: this session established that per-frame sequential registration on this hardware never
beats `ants.registration('Rigid')` on wall-clock (best tuned pytorch single-pair candidate:
~1.0-1.3s/frame vs ants's ~0.8s/frame), because a large, roughly-constant per-call overhead
(Python dispatch, autograd graph construction, MPS kernel-launch latency -- all established
earlier this session as the dominant cost for problems this small) is paid ONCE PER FRAME.
`ants.registration` is a sequential C++ tool and structurally cannot amortize that overhead
across frames. A batched torch solver can: N frames' rigid parameters optimized jointly, one
`F.grid_sample` call samples all N moving volumes against the (single, shared) reference in
parallel, one batched Mattes-MI evaluation, one backward pass, one optimizer step -- the fixed
per-step overhead is paid once for the whole series, not N times.

Status: prototype only, not integrated into `syntx.motion.motion_correction`. Reuses the same
physical-coordinate conventions as `robust_affine._run_pytorch_affine_solver` (adapted/copied,
not imported, since that function has no batched form) to avoid introducing a new axis-order
bug. Uses a single-start (Identity_CoM-only) coarse-to-fine Adam schedule -- no cone search,
consistent with this session's finding that capture range is not the bottleneck for
motion-correction-scale offsets, but retains a coarse stage for the "large real jumps" case.

Usage
-----
    python scripts/prototype_batched_motion_correction.py --group b0
    python scripts/prototype_batched_motion_correction.py --group dwi
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
from syntx.core.losses import parzen_weights, _parzen_joint_histogram
from syntx.motion import _extract_rigid_parameters, MotionParameters, calculate_framewise_displacement

CACHE_ROOT = '/tmp/syntx_motion_bench_cache'


def fd_from_rigid_params(R_list, t_list, center, radius=50.0):
    """Standard evaluation step: compute Power/Jenkinson FD from a sequence of (R, t) rigid
    parameters (temporal order), reusing `syntx.motion`'s OWN `_extract_rigid_parameters` /
    `calculate_framewise_displacement` directly -- not reimplemented -- so ground-truth and
    recovered FD are computed by the exact same code path `motion_correction` itself uses,
    guaranteeing this check means what it says it means. `R_list`/`t_list` are in the
    'y = R(x-c)+c+t' convention (c=center) used throughout this session's ground-truth
    tooling and by `robust_affine`'s own output transforms.
    """
    dim = 3
    n = len(R_list)
    translations = np.zeros((n, dim if dim == 2 else 6))
    homog = []
    for i in range(n):
        tx = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=dim)
        tx.set_parameters(np.concatenate([np.asarray(R_list[i]).flatten(), np.asarray(t_list[i])]))
        tx.set_fixed_parameters(np.asarray(center, dtype=np.float64))
        trans, rot, T_h = _extract_rigid_parameters(tx, dim)
        translations[i] = np.concatenate([trans, rot])
        homog.append(T_h)
    mp = MotionParameters(translations, columns=["tx", "ty", "tz", "rx", "ry", "rz"],
                           rad_to_deg_cols=["rx", "ry", "rz"], spatial_dim=dim)
    fd_power, fd_jenkinson = calculate_framewise_displacement(
        motion_parameters=mp, homogeneous_matrices=homog, radius=radius,
        center_of_sphere=np.asarray(center))
    return fd_power, fd_jenkinson, mp


def fd_agreement_report(fd_true, fd_hat, label=''):
    """Pearson r, Lin's CCC, and MAE between ground-truth and recovered FD trajectories --
    the standard trio used elsewhere in this session (scripts/benchmark_motion_parity.py)
    for real-data backend agreement, applied here to ground-truth recovery instead."""
    a, b = np.asarray(fd_true[1:]), np.asarray(fd_hat[1:])  # frame 0 FD is always 0 by definition
    if len(a) < 2:
        return dict(r=float('nan'), ccc=float('nan'), mae=float('nan'))
    r = float(np.corrcoef(a, b)[0, 1])
    mean_a, mean_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = np.mean((a - mean_a) * (b - mean_b))
    ccc = float((2 * cov) / (var_a + var_b + (mean_a - mean_b) ** 2 + 1e-12))
    mae = float(np.mean(np.abs(a - b)))
    print(f"  FD agreement {label}: r={r:.4f}  CCC={ccc:.4f}  MAE={mae:.4f} mm")
    return dict(r=r, ccc=ccc, mae=mae)


def batched_rodrigues(omega):
    """omega: [B,3] so(3) vectors -> R: [B,3,3] rotation matrices. Vectorized, no branch
    for the near-zero angle case (clamp_min keeps it numerically safe without one)."""
    theta = torch.norm(omega, dim=1, keepdim=True).clamp_min(1e-8)  # [B,1]
    u = omega / theta
    zeros = torch.zeros_like(u[:, 0])
    K = torch.stack([
        torch.stack([zeros, -u[:, 2], u[:, 1]], dim=1),
        torch.stack([u[:, 2], zeros, -u[:, 0]], dim=1),
        torch.stack([-u[:, 1], u[:, 0], zeros], dim=1),
    ], dim=1)  # [B,3,3]
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0)
    sin_t = torch.sin(theta).unsqueeze(-1)   # [B,1,1]
    cos_t = torch.cos(theta).unsqueeze(-1)
    return I + sin_t * K + (1.0 - cos_t) * torch.bmm(K, K)


def batched_mattes_mi_loss(w_x_batch, w_y, chunk=4096):
    """w_x_batch: [B, S, nbins] per-frame Parzen weights of warped samples.
    w_y: [S, nbins] Parzen weights of the (shared, precomputed once) fixed samples.
    Returns per-frame negative MI, shape [B]."""
    B, S, nb = w_x_batch.shape
    pad = (-S) % chunk
    if pad:
        w_x_batch = F.pad(w_x_batch, (0, 0, 0, pad))
        w_y_p = F.pad(w_y, (0, 0, 0, pad))
    else:
        w_y_p = w_y
    Sp = w_x_batch.shape[1]
    a = w_x_batch.view(B, -1, chunk, nb).permute(0, 1, 3, 2)          # [B, M, nb, chunk]
    b = w_y_p.view(-1, chunk, nb).unsqueeze(0).expand(B, -1, -1, -1)  # [B, M, chunk, nb]
    joint = torch.matmul(a, b).sum(dim=1)                             # [B, nb, nb]
    pxy = joint / (joint.sum(dim=(1, 2), keepdim=True) + 1e-8)
    px = pxy.sum(dim=2, keepdim=True)
    py = pxy.sum(dim=1, keepdim=True)
    ratio = pxy / (px * py + 1e-8)
    return -torch.sum(pxy * torch.log(torch.clamp(ratio, min=1e-8)), dim=(1, 2))  # [B]


def batched_correlation_loss(warped, fixed_vals):
    """Negative Pearson correlation between each frame's warped samples and the (shared)
    fixed samples. warped: [B,S], fixed_vals: [S]. Returns per-frame loss, shape [B].

    DUAL-METRIC RATIONALE (motion-correction specific, not general-purpose): Mattes MI's
    soft-histogram landscape is well-suited to cross-contrast/cross-modality alignment
    (arbitrary, non-monotonic intensity relationships), but that flexibility is exactly
    what makes it prone to local optima far from the true alignment -- useful generality
    elsewhere, wasted robustness here. Intra-subject motion tracking is the OPPOSITE
    case: consecutive frames of the same subject/same modality share near-identical
    contrast, so a simple correlation assumption (roughly linear intensity relationship)
    is actually satisfied, not an approximation. Correlation's loss surface is far
    smoother/more globally well-behaved as a result (no joint-histogram binning, no
    Parzen kernel discretization) -- and it's genuinely cheap: just mean/variance/
    covariance of the same point samples already gathered for MI, no extra grid_sample
    call, no extra histogram. Used as a coarse-stage CAPTURE aid (added to, not replacing,
    MI), not for final precision -- MI still drives the fine LBFGS polish stage alone.
    """
    w_mean = warped.mean(dim=1, keepdim=True)
    f_mean = fixed_vals.mean()
    w_c = warped - w_mean
    f_c = fixed_vals - f_mean
    cov = (w_c * f_c).sum(dim=1)
    w_std = torch.sqrt((w_c ** 2).sum(dim=1) + 1e-8)
    f_std = torch.sqrt((f_c ** 2).sum() + 1e-8)
    corr = cov / (w_std * f_std + 1e-8)
    return -corr  # [B], minimized => correlation maximized


def batched_parzen_weights(v_batch, num_bins=32, min_val=-1.0, max_val=1.0, pad=2.0):
    """v_batch: [B, S] -> [B, S, num_bins]. Vectorized form of core.losses.parzen_weights."""
    v = torch.nan_to_num(torch.clamp(v_batch.float(), min_val, max_val), nan=0.0)
    u_min, u_max = pad, float(num_bins - 1) - pad
    scale = (u_max - u_min) / (max_val - min_val)
    bins = torch.arange(num_bins, device=v.device, dtype=torch.float32).view(1, 1, -1)
    u = u_min + (v.unsqueeze(-1) - min_val) * scale - bins
    abs_x = torch.abs(u)
    x2 = abs_x * abs_x
    y1 = (2.0 / 3.0) - x2 + 0.5 * x2 * abs_x
    rem = 2.0 - abs_x
    y2 = (1.0 / 6.0) * (rem * rem * rem)
    return torch.where(abs_x < 1.0, y1, torch.where(abs_x < 2.0, y2, torch.zeros_like(u)))


def batched_rigid_register(fixed_img, moving_imgs, device='mps', n_starts_dummy=None,
                            schedule=None, num_bins=32, verbose=False):
    """Register ALL moving_imgs to fixed_img in one batched optimization (no per-frame
    sequential loop). Returns a list of (R, t) numpy arrays, one per frame, in the same
    ITK 'y = R(x-c)+c+t' convention as the rest of syntx (c = fixed image's own CoM),
    plus total elapsed seconds."""
    dev = torch.device(device)
    dim = fixed_img.dimension
    assert dim == 3, "prototype only implements the 3D path"
    B = len(moving_imgs)

    t0 = time.time()

    # -- physical grid / normalisation helpers, copied from robust_affine._run_pytorch_affine_solver
    # to reuse its already-validated XYZ physical convention rather than risk a new one. --
    f32 = dict(dtype=torch.float32, device=dev)
    sp_xyz = torch.tensor(fixed_img.spacing, **f32)
    orig_xyz = torch.tensor(fixed_img.origin, **f32)
    dir_xyz = torch.tensor(fixed_img.direction, **f32)
    com_f = np.asarray(compute_center_of_mass(fixed_img, weighted=True), dtype=np.float64)
    C_phys = torch.tensor(com_f, **f32)

    def to_tensor(img):
        # F.grid_sample expects [N,C,D,H,W] with grid's last-dim (x,y,z) mapped to (W,H,D).
        # ants' .numpy() is natively (X,Y,Z)-shaped, so transpose to (Z,Y,X) here -- matching
        # spatial.image_to_tensor(..., to_zyx=True)'s convention -- so grid_sample's (D,H,W)
        # correctly correspond to (Z,Y,X) and a plain (x,y,z)-ordered grid (no axis flip
        # needed) lines up with it.
        arr = np.transpose(img.numpy().astype(np.float32), (2, 1, 0))
        return torch.from_numpy(np.ascontiguousarray(arr)).to(dev)  # [Z,Y,X]

    fixed_t_zyx = to_tensor(fixed_img)
    moving_stack = torch.stack([to_tensor(m) for m in moving_imgs], dim=0)  # [B,Z,Y,X]
    fixed_t = fixed_t_zyx.permute(2, 1, 0)  # back to [X,Y,Z] for the foreground-sampling logic below
    shape_xyz = torch.tensor(list(fixed_img.shape), **f32)

    # Fixed-domain foreground point sample, shared across ALL frames and ALL iterations --
    # this is what makes w_y (fixed weights) reusable across the whole batch and every step.
    fg = fixed_t.reshape(-1) > 0.01
    idx_fg = torch.nonzero(fg, as_tuple=False).squeeze(1)
    rng = torch.Generator(device='cpu').manual_seed(42)
    n_sample = max(2000, int(0.15 * idx_fg.numel()))
    sel = idx_fg[torch.randperm(idx_fg.numel(), generator=rng)[:n_sample]].to(dev)
    vox_idx = torch.stack(torch.unravel_index(sel, fixed_t.shape), dim=-1).float()  # [S,3] in x,y,z voxel order
    phys_X = orig_xyz + (vox_idx * sp_xyz) @ dir_xyz.t()                            # [S,3]
    fixed_vals = fixed_t.reshape(-1)[sel]                                            # [S]

    # NOTE: tried robust p2/p98 percentile clipping here (matching normalize_image's default,
    # used by the single-frame solver) as a hypothesis for the remaining accuracy gap vs ants
    # -- it made things WORSE (0.242mm vs 0.159mm mean error on the same 40-frame b0 set), so
    # reverted to raw max. Recorded so this isn't retried as if untested.
    lo, hi = float(fixed_t.max()) * 0.0, float(fixed_t.max())
    fixed_scaled = (fixed_vals - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
    w_y = parzen_weights(fixed_scaled, num_bins=num_bins)  # [S, num_bins], computed ONCE

    def warp_to_moving_norm(y_phys, shape_xyz_level=None, level=1):
        # y_phys: [B,S,3] physical -> normalized grid coords in moving's own (shared) grid.
        # BUG (found and fixed): converting physical points to voxel-index units must use
        # the LEVEL-APPROPRIATE spacing/origin (a pooled grid's voxel i sits at full-res
        # index level*i+(level-1)/2, i.e. pooled spacing = sp_xyz*level, pooled origin
        # shifted by (level-1)/2 voxels) -- using full-res sp_xyz/orig_xyz to compute a
        # voxel index and then normalizing by the POOLED grid's (smaller) shape silently
        # samples the wrong location for level>1, worse the coarser the level. This was
        # the actual cause of the catastrophic (~20mm/40deg) coarse-level-only errors, not
        # the bias-field-corruption confound hypothesized earlier -- confirmed by testing
        # on uncorrupted frames (still bad) and isolating the exact physical-coordinate
        # formula (verified correct in isolation; only this normalization step was wrong).
        sxyz = shape_xyz if shape_xyz_level is None else shape_xyz_level
        sp_lvl = sp_xyz * level
        orig_lvl = orig_xyz + dir_xyz @ (((level - 1) / 2.0) * sp_xyz)
        y_vox = (y_phys - orig_lvl) @ torch.inverse(dir_xyz).t() / sp_lvl  # [B,S,3]
        return 2.0 * (y_vox / (sxyz - 1.0)) - 1.0

    # -- TRUE multi-resolution pyramid (avg-pool downsampling, not just point-subsampling of
    # the fine volume) for coarse stages, matching ants' own pyramid and the ORIGINAL
    # single-frame solver's `level` mechanism (which the batched prototype had dropped in
    # favor of point-subsampling for simplicity). A genuinely downsampled coarse level
    # smooths/denoises the loss landscape (removes small-scale local optima) and is cheap
    # (level=4 downsampling cuts voxel count by 64x), unlike point-subsampling at full
    # resolution, which just gives a noisier estimate of the SAME fine, multi-modal
    # landscape. Dense (every pooled voxel, no further subsampling) since pooled volumes
    # are already small.
    _pyramid_cache = {}

    def get_pyramid_level(level, point_sample_frac=None):
        # point_sample_frac: if given, foreground-mask AND point-sample within this pyramid
        # level (combining true multi-resolution downsampling with point-sampling), instead
        # of using every pooled voxel densely. Two real, separate benefits vs the
        # dense-coarse-level attempt: (1) matches the fine level's foreground-only
        # restriction, which the dense version omitted entirely -- diluting the coarse MI
        # histogram with mostly-uninformative background/air voxel pairs; (2) cheaper.
        cache_key = (level, point_sample_frac)
        if cache_key in _pyramid_cache:
            return _pyramid_cache[cache_key]
        if level == 1:
            if point_sample_frac is None:
                entry = dict(X=phys_X, w_y=w_y, moving=moving_stack, shape=shape_xyz,
                              fixed_vals=fixed_vals)
                _pyramid_cache[cache_key] = entry
                return entry
            phys_X_lvl, fixed_vals_lvl, moving_lvl, shape_lvl = phys_X, fixed_vals, moving_stack, shape_xyz
        else:
            fixed_pool = F.avg_pool3d(fixed_t_zyx.unsqueeze(0).unsqueeze(0),
                                       kernel_size=level, stride=level)[0, 0]      # [Z',Y',X']
            moving_pool = F.avg_pool3d(moving_stack.unsqueeze(1),
                                        kernel_size=level, stride=level)[:, 0]     # [B,Z',Y',X']
            axes = [torch.arange(n, device=dev, dtype=torch.float32) for n in fixed_pool.shape]
            mesh = torch.meshgrid(*axes, indexing='ij')                            # z,y,x
            vox_xyz_pooled = torch.stack(list(reversed(mesh)), dim=-1).reshape(-1, 3)  # x,y,z
            # a pooled voxel i averages full-res voxels [level*i, level*i+level-1]; its centre
            # is level*i + (level-1)/2 in full-res continuous voxel-index space.
            vox_xyz_full = vox_xyz_pooled * level + (level - 1) / 2.0
            phys_X_lvl = orig_xyz + (vox_xyz_full * sp_xyz) @ dir_xyz.t()
            fixed_vals_lvl = fixed_pool.reshape(-1)
            moving_lvl = moving_pool
            shape_lvl = torch.tensor([fixed_pool.shape[2], fixed_pool.shape[1],
                                       fixed_pool.shape[0]], **f32)             # x,y,z sizes

        if point_sample_frac is not None:
            # Same foreground threshold convention as the fine level's `fg` mask.
            fg_lvl = fixed_vals_lvl > 0.01
            idx_fg_lvl = torch.nonzero(fg_lvl, as_tuple=False).squeeze(1)
            n_lvl = max(300, int(point_sample_frac * idx_fg_lvl.numel()))
            sel_lvl = idx_fg_lvl[torch.randperm(idx_fg_lvl.numel(), generator=rng)[:n_lvl]].to(dev)
            phys_X_lvl = phys_X_lvl[sel_lvl]
            fixed_vals_lvl = fixed_vals_lvl[sel_lvl]

        fixed_scaled_lvl = (fixed_vals_lvl - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
        w_y_lvl = parzen_weights(fixed_scaled_lvl, num_bins=num_bins)
        entry = dict(X=phys_X_lvl, w_y=w_y_lvl, moving=moving_lvl, shape=shape_lvl,
                      fixed_vals=fixed_vals_lvl)
        _pyramid_cache[cache_key] = entry
        return entry

    # -- per-frame center-of-mass translation init (cheap, done once, not batched-optimized) --
    # -- Translation init: CoM offset alone was diagnosed as the actual root cause of the
    # large-jump outlier frames (NOT the MI-vs-correlation loss shape, which made no
    # difference when tested) -- weighted CoM is off by 7-11mm for some frames (vs 1.4mm
    # for well-behaved ones), likely from bias-field corruption skewing the apparent
    # centroid, and no fixed-iteration-budget coarse optimization reliably escapes a
    # starting point that far off. Cross-check it against FFT-based phase correlation, a
    # classic, GLOBALLY exact (not iterative/local-optima-prone) translation estimator --
    # one small real FFT per frame at the coarsest pooled resolution, genuinely cheap --
    # and use whichever of the two starting points scores better under Mattes MI at the
    # coarsest level. This is the same "dual metric, cheap, safe capture-range" idea
    # applied to INITIALIZATION rather than to the optimization objective itself.
    t_init_com = np.zeros((B, dim), dtype=np.float32)
    for b, m in enumerate(moving_imgs):
        com_m = np.asarray(compute_center_of_mass(m, weighted=True), dtype=np.float64)
        t_init_com[b] = (com_m - com_f).astype(np.float32)

    def phase_correlation_translation_init(level=4, topk=5, min_sep_vox=2):
        # get_pyramid_level doesn't expose the dense pooled tensor directly when it was
        # built via the point-sampled path; recompute the pooled dense tensors directly
        # here (cheap: level=4 downsampling is <1% of full voxel count).
        fixed_pool = F.avg_pool3d(fixed_t_zyx.unsqueeze(0).unsqueeze(0), kernel_size=level,
                                   stride=level)[0, 0]                      # [Z',Y',X']
        moving_pool = F.avg_pool3d(moving_stack.unsqueeze(1), kernel_size=level,
                                    stride=level)[:, 0]                     # [B,Z',Y',X']
        shape = fixed_pool.shape
        # Hann window to reduce circular-convolution edge artifacts (images aren't
        # actually periodic).
        def hann(n):
            return torch.hann_window(n, periodic=False, device=dev, dtype=torch.float32)
        wz, wy_, wx = hann(shape[0]), hann(shape[1]), hann(shape[2])
        win = wz.view(-1, 1, 1) * wy_.view(1, -1, 1) * wx.view(1, 1, -1)
        F_fixed = torch.fft.rfftn(fixed_pool * win)
        F_moving = torch.fft.rfftn(moving_pool * win.unsqueeze(0), dim=(-3, -2, -1))
        R = F_fixed.unsqueeze(0) * torch.conj(F_moving)
        R = R / (R.abs() + 1e-8)
        r = torch.fft.irfftn(R, s=shape, dim=(-3, -2, -1))                 # [B,Z',Y',X']

        def unwrap(idx, size):
            return torch.where(idx > size / 2, idx - size, idx)

        # A single best peak is fragile: real anatomy + windowing can produce a strong but
        # WRONG competing peak (observed directly: several frames converged on the exact
        # same spurious candidate). Standard fix -- extract the top-`topk` local maxima
        # (simple non-max suppression: zero out a small cube around each found peak before
        # finding the next) and let the MI scoring step below disambiguate, rather than
        # trusting the raw correlation-surface argmax alone.
        work = r.reshape(B, shape[0], shape[1], shape[2]).clone()
        candidates = []
        for _ in range(topk):
            flat = work.reshape(B, -1)
            peak = flat.argmax(dim=1)
            pz = (peak // (shape[1] * shape[2]))
            rem = peak % (shape[1] * shape[2])
            py = (rem // shape[2])
            px = (rem % shape[2])
            pz_s = unwrap(pz.float(), shape[0]) * level
            py_s = unwrap(py.float(), shape[1]) * level
            px_s = unwrap(px.float(), shape[2]) * level
            vox_shift_xyz = torch.stack([px_s, py_s, pz_s], dim=-1)        # [B,3], full-res voxel units
            phys_shift = (vox_shift_xyz * sp_xyz) @ dir_xyz.t()            # [B,3]
            candidates.append(-phys_shift.cpu().numpy())
            for b in range(B):
                zc, yc, xc = int(pz[b]), int(py[b]), int(px[b])
                z0, z1 = max(0, zc - min_sep_vox), min(shape[0], zc + min_sep_vox + 1)
                y0, y1 = max(0, yc - min_sep_vox), min(shape[1], yc + min_sep_vox + 1)
                x0, x1 = max(0, xc - min_sep_vox), min(shape[2], xc + min_sep_vox + 1)
                work[b, z0:z1, y0:y1, x0:x1] = -1e9
        return candidates  # list of `topk` [B,3] arrays

    pc_candidates = phase_correlation_translation_init()

    # For frames with a real rotation on top of the translation jump, phase correlation's
    # pure-translation assumption breaks down -- diagnosed directly: candidates scored fine
    # in isolation, but none of them were near truth for frames whose jump ALSO included a
    # large rotation, because moving content isn't just a shifted copy of fixed anymore.
    # Fix: give each (translation, rotation) SEED pair a fair chance by fitting a SHORT
    # (cheap, coarse-level, correlation-objective) JOINT refinement -- both t and omega
    # free, not rotation-only with translation frozen. Freezing translation was diagnosed
    # as its own trap: a still-wrong (e.g. 7mm-off) translation makes every candidate
    # rotation look bad, so rotation-only fitting can't discover the right rotation either.
    # Letting both drift together breaks that chicken-and-egg coupling.
    def fit_pair(t_seed, omega_seed, iters=30, lr_t=0.15, lr_r=0.06):
        lvl = get_pyramid_level(4)
        Xc = lvl['X'] - C_phys
        t_cand = torch.as_tensor(np.broadcast_to(t_seed, (B, 3)).copy(), **f32).clone()
        omega_cand = torch.as_tensor(np.broadcast_to(omega_seed, (B, 3)).copy(), **f32).clone()
        t_cand.requires_grad_(True); omega_cand.requires_grad_(True)
        opt = torch.optim.Adam([{'params': [t_cand], 'lr': lr_t}, {'params': [omega_cand], 'lr': lr_r}])

        def eval_loss():
            R_cand = batched_rodrigues(omega_cand)
            y_phys = torch.einsum('bij,sj->bsi', R_cand, Xc) + C_phys + t_cand.unsqueeze(1)
            grid = warp_to_moving_norm(y_phys, lvl['shape'], level=4).view(B, 1, 1, -1, 3)
            warped = F.grid_sample(lvl['moving'].unsqueeze(1), grid, mode='bilinear',
                                    padding_mode='zeros', align_corners=True).reshape(B, -1)
            return batched_correlation_loss(warped, lvl['fixed_vals'])

        for _ in range(iters):
            opt.zero_grad(set_to_none=True)
            eval_loss().sum().backward()
            opt.step()
        with torch.no_grad():
            final_score = eval_loss().cpu().numpy()
        return t_cand.detach().cpu().numpy(), omega_cand.detach().cpu().numpy(), final_score

    # Rotation itself needs SOME capture range too (diagnosed: a 20-iteration rotation-only
    # fit starting from identity did not reliably find a genuine ~15deg rotation either --
    # same local-optimum problem one level up). Pair each translation candidate with a
    # small set of rotation SEEDS (identity + moderate rotations about each axis), not just
    # identity -- a narrow, cheap version of the classic cone-search idea, used only for
    # this init-disambiguation step (NOT reintroducing a full multi-start tournament into
    # the main optimization).
    # NOTE: also tried adding face-diagonal seeds (19 total instead of 7) to chase the last
    # ~1mm DWI-under-jump residual -- cost roughly doubled (0.64s -> 1.28s/frame, exceeding
    # ants) for negligible accuracy gain. That residual matches DWI's own established
    # baseline floor (different gradient directions have genuinely different contrast/SNR,
    # not a jump-capture failure), so reverted to the cheaper set rather than pay 2x for it.
    rot_seeds = [np.zeros(3)]
    for axis in range(3):
        for deg in (20.0, -20.0):
            v = np.zeros(3); v[axis] = np.radians(deg)
            rot_seeds.append(v)

    all_candidates = [t_init_com] + pc_candidates
    fit_results = []   # list of (t_fit[B,3], omega_fit[B,3], score[B])
    for t_cand in all_candidates:
        for seed in rot_seeds:
            t_fit, omega_fit, score = fit_pair(t_cand, seed)
            fit_results.append((t_fit, omega_fit, score))

    all_scores = np.stack([s for _, _, s in fit_results], axis=0)   # [n_combos, B]
    best_combo = all_scores.argmin(axis=0)                          # [B]
    t_init = np.stack([fit_results[best_combo[b]][0][b] for b in range(B)], axis=0).astype(np.float32)
    omega_init = np.stack([fit_results[best_combo[b]][1][b] for b in range(B)], axis=0).astype(np.float32)
    if verbose:
        print(f"    translation+rotation init: searched {len(all_candidates)} translation "
              f"candidates x {len(rot_seeds)} rotation seeds = {len(fit_results)} combos/frame")
        for b in range(B):
            print(f"      frame {b}: t={np.round(t_init[b], 2)} omega={np.round(omega_init[b], 3)} "
                  f"best_score={all_scores[best_combo[b], b]:.4f}")

    omega = torch.tensor(omega_init, device=dev, requires_grad=True)
    t_param = torch.tensor(t_init, device=dev, requires_grad=True)

    if schedule is None:
        # TRUE multi-resolution pyramid (level=4,2: genuine avg_pool3d downsampling, dense)
        # for coarse capture, then point-sampled level=1 LBFGS polish for precision -- the
        # combination this session established works, now with a real coarse landscape
        # instead of noisy point-subsamples of the fine one.
        schedule = [
            dict(optimizer='adam', iters=80, lr_t=0.08, lr_r=0.04, level=4, sampling=0.2, corr_weight=8.0),
            dict(optimizer='adam', iters=40, lr_t=0.03, lr_r=0.015, level=2, sampling=0.2, corr_weight=3.0),
            dict(optimizer='lbfgs', iters=18, lr_t=0.2, lr_r=0.05, sampling=0.2, level=1),
            dict(optimizer='lbfgs', iters=15, lr_t=0.05, lr_r=0.01, sampling=0.2, level=1),
        ]
        # Tried and discarded: extra iters/sampling/corr_weight on a 5th level=1 LBFGS stage
        # (up to sampling=0.8, corr_weight=3.0) gave IDENTICAL 0.092mm translation error and
        # slightly WORSE rotation error (0.078 vs 0.061-0.067 deg) on cache 78f2c2ce4910/b0.
        # This is a genuine convergence floor, not a compute/sampling budget problem -- the
        # remaining ~25-30% gap vs ants (0.067-0.072mm) needs a different optimizer/loss
        # structure (e.g. analytic Gauss-Newton step), not more iterations of this schedule.

    def compute_loss(X_stage, w_y_stage, moving_tensor, shape_stage, level_stage=1,
                      fixed_vals_stage=None, corr_weight=0.0):
        R = batched_rodrigues(omega)                                   # [B,3,3]
        Xc = X_stage - C_phys                                          # [S,3]
        y_phys = torch.einsum('bij,sj->bsi', R, Xc) + C_phys + t_param.unsqueeze(1)  # [B,S,3]
        grid_norm = warp_to_moving_norm(y_phys, shape_stage, level=level_stage)  # [B,S,3]
        grid = grid_norm.view(B, 1, 1, -1, 3)
        warped = F.grid_sample(moving_tensor.unsqueeze(1), grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
        warped = warped.reshape(B, -1)                                 # [B,S]
        w_scaled = (warped - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
        w_x = batched_parzen_weights(w_scaled, num_bins=num_bins)      # [B,S,nb]
        per_frame_loss = batched_mattes_mi_loss(w_x, w_y_stage)        # [B]
        if corr_weight > 0.0 and fixed_vals_stage is not None:
            # Dual-metric: add correlation as a coarse-stage capture aid (see
            # batched_correlation_loss's docstring). Reuses the SAME `warped` samples
            # already gathered for MI -- no extra grid_sample call, negligible extra cost
            # (mean/var/covariance of vectors already in hand).
            per_frame_loss = per_frame_loss + corr_weight * batched_correlation_loss(warped, fixed_vals_stage)
        return per_frame_loss

    for si, stage in enumerate(schedule):
        level = stage.get('level', 1)
        samp = stage.get('sampling', 1.0)
        if level == 1:
            # Fine level: base data is already the once-computed foreground point sample
            # (phys_X/w_y); `sampling` further subsets that, as before.
            lvl_data = get_pyramid_level(1)
            if samp < 1.0:
                k = max(500, int(samp * lvl_data['X'].shape[0]))
                sub = torch.randperm(lvl_data['X'].shape[0], generator=rng)[:k].to(dev)
                X_stage, w_y_stage = lvl_data['X'][sub], lvl_data['w_y'][sub]
                fixed_vals_stage = lvl_data['fixed_vals'][sub]
            else:
                X_stage, w_y_stage = lvl_data['X'], lvl_data['w_y']
                fixed_vals_stage = lvl_data['fixed_vals']
        else:
            # Coarse level: combine TRUE avg-pool downsampling with foreground-masked
            # point-sampling (not the dense-every-pooled-voxel approach, which diluted the
            # coarse MI histogram with uninformative background/air voxel pairs).
            lvl_data = get_pyramid_level(level, point_sample_frac=min(samp, 1.0))
            X_stage, w_y_stage = lvl_data['X'], lvl_data['w_y']
            fixed_vals_stage = lvl_data['fixed_vals']
        moving_tensor, shape_stage = lvl_data['moving'], lvl_data['shape']
        corr_weight = stage.get('corr_weight', 0.0)

        if stage.get('optimizer', 'adam') == 'lbfgs':
            opt = torch.optim.LBFGS([t_param, omega], lr=1.0, max_iter=stage['iters'],
                                     history_size=10, tolerance_grad=1e-9,
                                     tolerance_change=1e-11, line_search_fn='strong_wolfe')

            def closure():
                opt.zero_grad(set_to_none=True)
                l = compute_loss(X_stage, w_y_stage, moving_tensor, shape_stage, level_stage=level,
                                  fixed_vals_stage=fixed_vals_stage, corr_weight=corr_weight).sum()
                l.backward()
                return l

            loss_val = opt.step(closure)
            if verbose:
                print(f"    stage {si} (lbfgs, level={level}, iters={stage['iters']}, "
                      f"sampling={samp}, corr_weight={corr_weight}): total loss = {float(loss_val):.4f}")
        else:
            opt = torch.optim.Adam([
                {'params': [t_param], 'lr': stage['lr_t']},
                {'params': [omega], 'lr': stage['lr_r']},
            ])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=stage['iters'], eta_min=stage['lr_t'] * 0.02)
            for it in range(stage['iters']):
                opt.zero_grad(set_to_none=True)
                per_frame_loss = compute_loss(X_stage, w_y_stage, moving_tensor, shape_stage,
                                               level_stage=level, fixed_vals_stage=fixed_vals_stage,
                                               corr_weight=corr_weight)
                loss = per_frame_loss.sum()
                loss.backward()
                opt.step()
                sched.step()
                if verbose and it % 20 == 0:
                    print(f"    stage {si} (adam, level={level}, corr_weight={corr_weight}) it={it}: "
                          f"mean per-frame loss = {per_frame_loss.mean().item():.4f}")

    elapsed = time.time() - t0

    R_final = batched_rodrigues(omega).detach().cpu().numpy()
    t_final = t_param.detach().cpu().numpy()
    return [(R_final[b], t_final[b]) for b in range(B)], elapsed, com_f


def rigid_error(R_true, t_true, R_hat, t_hat):
    U, _, Vt = np.linalg.svd(R_hat)
    R_hat_o = U @ Vt
    trans_err = float(np.linalg.norm(t_hat - t_true))
    R_diff = np.asarray(R_true).T @ R_hat_o
    ang_err = float(np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))))
    return trans_err, ang_err


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache-dir', default=None, help='defaults to the most recent bench cache')
    p.add_argument('--group', default='b0', choices=['b0', 'dwi'])
    p.add_argument('--device', default='mps')
    p.add_argument('--num-bins', type=int, default=32)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    cache_dir = args.cache_dir
    if cache_dir is None:
        subdirs = [os.path.join(CACHE_ROOT, d) for d in os.listdir(CACHE_ROOT)]
        cache_dir = max(subdirs, key=os.path.getmtime)
    print(f"[cache] {cache_dir}")

    with open(os.path.join(cache_dir, 'meta.json')) as f:
        meta = json.load(f)
    fixed_img = ants.image_read(os.path.join(
        cache_dir, 'mean_b0.nii.gz' if args.group == 'b0' else 'mean_dwi_aligned.nii.gz'))
    frames_meta = [r for r in meta['frames'] if r['group'] == args.group]
    moving_imgs = [ants.image_read(r['frame_path']) for r in frames_meta]
    print(f"[data] {len(moving_imgs)} '{args.group}' frames, shape={fixed_img.shape}")

    print(f"\n[batched] registering ALL {len(moving_imgs)} frames in one batched optimization "
          f"(device={args.device}) ...")
    results, elapsed, com_f = batched_rigid_register(fixed_img, moving_imgs, device=args.device,
                                                       num_bins=args.num_bins, verbose=args.verbose)
    print(f"[batched] total elapsed: {elapsed:.2f}s "
          f"({elapsed/len(moving_imgs):.3f}s/frame amortized)")

    print("\nframe | trans_err(mm) | rot_err(deg)")
    trans_errs, rot_errs = [], []
    for b, rec in enumerate(frames_meta):
        R_true = np.array(rec['R_inv'])
        t_true = np.array(rec['t_inv'])
        R_hat, t_hat = results[b]
        te, ae = rigid_error(R_true, t_true, R_hat, t_hat)
        trans_errs.append(te); rot_errs.append(ae)
        print(f"  {b:3d} | {te:14.3f} | {ae:12.3f}")
    print(f"\nMEAN: trans_err={np.mean(trans_errs):.3f} mm, rot_err={np.mean(rot_errs):.3f} deg")
    print(f"Amortized time/frame: {elapsed/len(moving_imgs):.3f}s "
          f"(ants_rigid baseline: ~0.8s/frame, always sequential)")

    # -- STANDARD check: ground-truth FD recovery. Per-frame rigid-parameter error doesn't
    # tell you whether the actual downstream QC number (FD, what real motion_correction
    # users read directly) is right -- e.g. an error that's consistent/correlated across
    # nearby frames could largely cancel in the FRAME-TO-FRAME difference FD measures, or
    # conversely a small per-frame error could still corrupt FD if it's poorly correlated
    # frame-to-frame. Always compute and report both, using syntx.motion's own FD code.
    print("\n-- Ground-truth FD recovery (standard check) --")
    center = np.array(fixed_img.get_center_of_mass())
    R_true_list = [np.array(rec['R_inv']) for rec in frames_meta]
    t_true_list = [np.array(rec['t_inv']) for rec in frames_meta]
    R_hat_list = [results[b][0] for b in range(len(frames_meta))]
    t_hat_list = [results[b][1] for b in range(len(frames_meta))]
    fd_power_true, fd_jenk_true, _ = fd_from_rigid_params(R_true_list, t_true_list, center)
    fd_power_hat, fd_jenk_hat, _ = fd_from_rigid_params(R_hat_list, t_hat_list, center)
    fd_agreement_report(fd_power_true, fd_power_hat, label='(Power FD)')
    fd_agreement_report(fd_jenk_true, fd_jenk_hat, label='(Jenkinson FD)')
    print(f"  true FD(power) mean={np.mean(fd_power_true[1:]):.4f} mm, "
          f"recovered FD(power) mean={np.mean(fd_power_hat[1:]):.4f} mm")


if __name__ == '__main__':
    main()
