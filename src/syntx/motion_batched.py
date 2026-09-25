"""
syntx.motion_batched — Batched Multi-Frame Rigid Registration for Motion Correction
=====================================================================================

Registers an ENTIRE time series to a single shared reference in one batched GPU
forward/backward pass per optimization step, instead of N sequential single-pair
registration calls (what `ants.registration` and syntx's own per-frame `robust_affine`
loop both do). Extracted and productionized from
`scripts/prototype_batched_motion_correction.py` after validation -- see
`docs/SESSION_2026-09-25_PHASE_CORRELATION_AND_BATCHED_MOTION_CORRECTION.md` for the full
root-cause investigation (axis-order bugs found and fixed, what similarity-metric and
regularization variants were tried and discarded, the large-jump capture-range fix, and
the ants.registration comparison).

Why batching helps here specifically: a large, roughly-constant per-call overhead (Python
dispatch, autograd graph construction, MPS kernel-launch latency) dominates wall-clock at
this problem size -- established across this and the prior session -- not raw compute.
`ants.registration` is a sequential C++ tool and cannot amortize that overhead across
frames; a batched torch solver pays it once for the whole series.

Validated result (40-frame real clinical b0/DWI series, realistic corruption): ~3x faster
than ants' `type_of_transform='Rigid'` registration per frame, comparable accuracy (within
~25-45% of ants' absolute error, matching on FD correlation r>=0.995), zero observed
failures on a stress test with a large (15mm/15deg) mid-series jump after the
capture-range fixes described in the session doc above.

Status: 3D only. No masked-MI support (no per-voxel weighting in the batched Mattes MI /
correlation implementation here). Both are checked and raise `NotImplementedError` /
`ValueError` rather than silently falling back or producing a wrong answer.
"""

from __future__ import annotations

import os
import tempfile
from typing import List, Optional, Tuple

import ants
import numpy as np
import torch
import torch.nn.functional as F

from .robust_affine import compute_center_of_mass
from .core.losses import parzen_weights


def _batched_rodrigues(omega: torch.Tensor) -> torch.Tensor:
    """omega: [B,3] so(3) vectors -> R: [B,3,3] rotation matrices."""
    theta = torch.norm(omega, dim=1, keepdim=True).clamp_min(1e-8)
    u = omega / theta
    zeros = torch.zeros_like(u[:, 0])
    K = torch.stack([
        torch.stack([zeros, -u[:, 2], u[:, 1]], dim=1),
        torch.stack([u[:, 2], zeros, -u[:, 0]], dim=1),
        torch.stack([-u[:, 1], u[:, 0], zeros], dim=1),
    ], dim=1)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0)
    sin_t = torch.sin(theta).unsqueeze(-1)
    cos_t = torch.cos(theta).unsqueeze(-1)
    return I + sin_t * K + (1.0 - cos_t) * torch.bmm(K, K)


def _batched_mattes_mi_loss(w_x_batch: torch.Tensor, w_y: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """w_x_batch: [B,S,nbins] per-frame Parzen weights. w_y: [S,nbins] shared fixed weights.
    Returns per-frame negative MI, shape [B]."""
    B, S, nb = w_x_batch.shape
    pad = (-S) % chunk
    if pad:
        w_x_batch = F.pad(w_x_batch, (0, 0, 0, pad))
        w_y_p = F.pad(w_y, (0, 0, 0, pad))
    else:
        w_y_p = w_y
    a = w_x_batch.view(B, -1, chunk, nb).permute(0, 1, 3, 2)
    b = w_y_p.view(-1, chunk, nb).unsqueeze(0).expand(B, -1, -1, -1)
    joint = torch.matmul(a, b).sum(dim=1)
    pxy = joint / (joint.sum(dim=(1, 2), keepdim=True) + 1e-8)
    px = pxy.sum(dim=2, keepdim=True)
    py = pxy.sum(dim=1, keepdim=True)
    ratio = pxy / (px * py + 1e-8)
    return -torch.sum(pxy * torch.log(torch.clamp(ratio, min=1e-8)), dim=(1, 2))


def _batched_parzen_weights(v_batch: torch.Tensor, num_bins: int = 32,
                             min_val: float = -1.0, max_val: float = 1.0, pad: float = 2.0) -> torch.Tensor:
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


def _batched_correlation_loss(warped: torch.Tensor, fixed_vals: torch.Tensor) -> torch.Tensor:
    """Negative Pearson correlation per frame. See module docstring / session doc for why
    this is added alongside (not instead of) Mattes MI for intra-subject motion tracking
    specifically: consecutive frames of the same subject/modality share near-identical
    contrast, so correlation's linear-relationship assumption is satisfied, not
    approximated, and its coarse-resolution loss surface is measurably more reliable than
    MI's (diagnosed directly, not assumed -- see session doc Sec 4.2)."""
    w_mean = warped.mean(dim=1, keepdim=True)
    f_mean = fixed_vals.mean()
    w_c = warped - w_mean
    f_c = fixed_vals - f_mean
    cov = (w_c * f_c).sum(dim=1)
    w_std = torch.sqrt((w_c ** 2).sum(dim=1) + 1e-8)
    f_std = torch.sqrt((f_c ** 2).sum() + 1e-8)
    return -(cov / (w_std * f_std + 1e-8))


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def batched_rigid_register_pass(
    reference_img: ants.ANTsImage,
    moving_imgs: List[ants.ANTsImage],
    device: str = "auto",
    num_bins: int = 32,
    verbose: bool = False,
    outprefix: Optional[str] = None,
) -> Tuple[List[List[str]], List[List[str]], float]:
    """
    Register every image in `moving_imgs` to the SAME `reference_img` in one batched
    optimization. Returns `(fwd_transforms, inv_transforms, elapsed_seconds)`, each a list
    (one entry per frame) of single-element lists of `.mat` transform file paths -- the
    same shape `robust_affine`'s `fwdtransforms`/`invtransforms` use, so callers (e.g.
    `syntx.motion.motion_correction`) don't need a separate code path to consume the
    result.

    Parameters
    ----------
    reference_img : ants.ANTsImage
        Shared fixed/reference image (3D only).
    moving_imgs : list of ants.ANTsImage
        Frames to register. All must share the reference's spatial grid (same shape,
        spacing, origin, direction) -- true for time-series frames of one acquisition,
        which is the only case this function is meant for.
    device : str, default='auto'
        'auto' picks cuda -> mps -> cpu. An explicit value is honoured as-is.
    outprefix : str, optional
        If given, transform files are written under this prefix (`{outprefix}_vol{t:04d}
        .mat`); otherwise a dedicated temp directory is used.

    Raises
    ------
    NotImplementedError
        If `reference_img.dimension != 3` (2D not implemented).
    """
    dim = reference_img.dimension
    if dim != 3:
        raise NotImplementedError(
            "motion_batched.batched_rigid_register_pass only implements the 3D path. "
            "Use backend='pytorch' (per-frame robust_affine) or backend='ants' for 2D+t series."
        )
    dev = torch.device(_auto_device() if device == "auto" else device)
    B = len(moving_imgs)
    if B == 0:
        return [], [], 0.0

    import time
    t0 = time.time()

    # Presets below assume real head-sized volumes (hundreds of voxels/axis); on a small
    # volume, downsampling by a coarse level (e.g. 4x on a 16-voxel axis) leaves only a
    # handful of voxels per axis, an unstable estimate -- the exact degenerate-pyramid
    # failure mode found and fixed in robust_affine.py's single-frame solver earlier this
    # session (spurious scale contraction from a coarse-level Mattes-MI optimum on too few
    # voxels). Same fix here: clamp the coarsest usable level so the smallest spatial axis
    # keeps at least MIN_VOXELS_PER_AXIS voxels at every pyramid level actually used.
    MIN_VOXELS_PER_AXIS = 8
    max_level = max(1, min(int(s) for s in reference_img.shape) // MIN_VOXELS_PER_AXIS)

    f32 = dict(dtype=torch.float32, device=dev)
    sp_xyz = torch.tensor(reference_img.spacing, **f32)
    orig_xyz = torch.tensor(reference_img.origin, **f32)
    dir_xyz = torch.tensor(reference_img.direction, **f32)
    com_f = np.asarray(compute_center_of_mass(reference_img, weighted=True), dtype=np.float64)
    C_phys = torch.tensor(com_f, **f32)

    def to_tensor(img):
        arr = np.transpose(img.numpy().astype(np.float32), (2, 1, 0))
        return torch.from_numpy(np.ascontiguousarray(arr)).to(dev)

    fixed_t_zyx = to_tensor(reference_img)
    moving_stack = torch.stack([to_tensor(m) for m in moving_imgs], dim=0)
    fixed_t = fixed_t_zyx.permute(2, 1, 0)
    shape_xyz = torch.tensor(list(reference_img.shape), **f32)

    fg = fixed_t.reshape(-1) > 0.01
    idx_fg = torch.nonzero(fg, as_tuple=False).squeeze(1)
    rng = torch.Generator(device="cpu").manual_seed(42)
    n_sample = max(2000, int(0.15 * idx_fg.numel()))
    sel = idx_fg[torch.randperm(idx_fg.numel(), generator=rng)[:n_sample]].to(dev)
    vox_idx = torch.stack(torch.unravel_index(sel, fixed_t.shape), dim=-1).float()
    phys_X = orig_xyz + (vox_idx * sp_xyz) @ dir_xyz.t()
    fixed_vals = fixed_t.reshape(-1)[sel]

    lo, hi = 0.0, float(fixed_t.max())
    fixed_scaled = (fixed_vals - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
    w_y = parzen_weights(fixed_scaled, num_bins=num_bins)

    def warp_to_moving_norm(y_phys, shape_lvl, level=1):
        sp_lvl = sp_xyz * level
        orig_lvl = orig_xyz + dir_xyz @ (((level - 1) / 2.0) * sp_xyz)
        y_vox = (y_phys - orig_lvl) @ torch.inverse(dir_xyz).t() / sp_lvl
        return 2.0 * (y_vox / (shape_lvl - 1.0)) - 1.0

    _pyramid_cache = {}

    def get_pyramid_level(level, point_sample_frac=None):
        cache_key = (level, point_sample_frac)
        if cache_key in _pyramid_cache:
            return _pyramid_cache[cache_key]
        if level == 1 and point_sample_frac is None:
            entry = dict(X=phys_X, w_y=w_y, moving=moving_stack, shape=shape_xyz, fixed_vals=fixed_vals)
            _pyramid_cache[cache_key] = entry
            return entry
        if level == 1:
            phys_X_lvl, fixed_vals_lvl, moving_lvl, shape_lvl = phys_X, fixed_vals, moving_stack, shape_xyz
        else:
            fixed_pool = F.avg_pool3d(fixed_t_zyx.unsqueeze(0).unsqueeze(0), kernel_size=level, stride=level)[0, 0]
            moving_pool = F.avg_pool3d(moving_stack.unsqueeze(1), kernel_size=level, stride=level)[:, 0]
            shape = fixed_pool.shape
            axes = [torch.arange(n, device=dev, dtype=torch.float32) for n in shape]
            mesh = torch.meshgrid(*axes, indexing="ij")
            vox_xyz_pooled = torch.stack(list(reversed(mesh)), dim=-1).reshape(-1, 3)
            vox_xyz_full = vox_xyz_pooled * level + (level - 1) / 2.0
            phys_X_lvl = orig_xyz + (vox_xyz_full * sp_xyz) @ dir_xyz.t()
            fixed_vals_lvl = fixed_pool.reshape(-1)
            moving_lvl = moving_pool
            shape_lvl = torch.tensor([shape[2], shape[1], shape[0]], **f32)
        if point_sample_frac is not None:
            fg_lvl = fixed_vals_lvl > 0.01
            idx_fg_lvl = torch.nonzero(fg_lvl, as_tuple=False).squeeze(1)
            n_lvl = max(300, int(point_sample_frac * idx_fg_lvl.numel()))
            sel_lvl = idx_fg_lvl[torch.randperm(idx_fg_lvl.numel(), generator=rng)[:n_lvl]].to(dev)
            phys_X_lvl = phys_X_lvl[sel_lvl]
            fixed_vals_lvl = fixed_vals_lvl[sel_lvl]
        fixed_scaled_lvl = (fixed_vals_lvl - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
        w_y_lvl = parzen_weights(fixed_scaled_lvl, num_bins=num_bins)
        entry = dict(X=phys_X_lvl, w_y=w_y_lvl, moving=moving_lvl, shape=shape_lvl, fixed_vals=fixed_vals_lvl)
        _pyramid_cache[cache_key] = entry
        return entry

    # -- translation + rotation seed search (large-jump capture-range fix; see session doc
    # Sec 4.2 for why both need seeding, and jointly, not rotation-only with translation
    # frozen) --
    t_init_com = np.zeros((B, 3), dtype=np.float32)
    for b, m in enumerate(moving_imgs):
        com_m = np.asarray(compute_center_of_mass(m, weighted=True), dtype=np.float64)
        t_init_com[b] = (com_m - com_f).astype(np.float32)

    def phase_correlation_candidates(level=4, topk=5, min_sep_vox=2):
        level = min(level, max_level)
        fixed_pool = F.avg_pool3d(fixed_t_zyx.unsqueeze(0).unsqueeze(0), kernel_size=level, stride=level)[0, 0]
        moving_pool = F.avg_pool3d(moving_stack.unsqueeze(1), kernel_size=level, stride=level)[:, 0]
        shape = fixed_pool.shape

        def hann(n):
            return torch.hann_window(n, periodic=False, device=dev, dtype=torch.float32)

        win = hann(shape[0]).view(-1, 1, 1) * hann(shape[1]).view(1, -1, 1) * hann(shape[2]).view(1, 1, -1)
        F_fixed = torch.fft.rfftn(fixed_pool * win)
        F_moving = torch.fft.rfftn(moving_pool * win.unsqueeze(0), dim=(-3, -2, -1))
        R = F_fixed.unsqueeze(0) * torch.conj(F_moving)
        R = R / (R.abs() + 1e-8)
        r = torch.fft.irfftn(R, s=shape, dim=(-3, -2, -1))

        def unwrap(idx, size):
            return torch.where(idx > size / 2, idx - size, idx)

        work = r.reshape(B, shape[0], shape[1], shape[2]).clone()
        cands = []
        for _ in range(topk):
            flat = work.reshape(B, -1)
            peak = flat.argmax(dim=1)
            pz = peak // (shape[1] * shape[2])
            rem = peak % (shape[1] * shape[2])
            py = rem // shape[2]
            px = rem % shape[2]
            pz_s = unwrap(pz.float(), shape[0]) * level
            py_s = unwrap(py.float(), shape[1]) * level
            px_s = unwrap(px.float(), shape[2]) * level
            vox_shift_xyz = torch.stack([px_s, py_s, pz_s], dim=-1)
            phys_shift = (vox_shift_xyz * sp_xyz) @ dir_xyz.t()
            cands.append(-phys_shift.cpu().numpy())
            for b in range(B):
                zc, yc, xc = int(pz[b]), int(py[b]), int(px[b])
                z0, z1 = max(0, zc - min_sep_vox), min(shape[0], zc + min_sep_vox + 1)
                y0, y1 = max(0, yc - min_sep_vox), min(shape[1], yc + min_sep_vox + 1)
                x0, x1 = max(0, xc - min_sep_vox), min(shape[2], xc + min_sep_vox + 1)
                work[b, z0:z1, y0:y1, x0:x1] = -1e9
        return cands

    pc_candidates = phase_correlation_candidates()
    coarse_level = min(4, max_level)

    def fit_pair(t_seed, omega_seed, iters=30, lr_t=0.15, lr_r=0.06):
        lvl = get_pyramid_level(coarse_level)
        Xc = lvl["X"] - C_phys
        t_cand = torch.as_tensor(np.broadcast_to(t_seed, (B, 3)).copy(), **f32).clone()
        omega_cand = torch.as_tensor(np.broadcast_to(omega_seed, (B, 3)).copy(), **f32).clone()
        t_cand.requires_grad_(True)
        omega_cand.requires_grad_(True)
        opt = torch.optim.Adam([{"params": [t_cand], "lr": lr_t}, {"params": [omega_cand], "lr": lr_r}])

        def eval_loss():
            R_cand = _batched_rodrigues(omega_cand)
            y_phys = torch.einsum("bij,sj->bsi", R_cand, Xc) + C_phys + t_cand.unsqueeze(1)
            grid = warp_to_moving_norm(y_phys, lvl["shape"], level=coarse_level).view(B, 1, 1, -1, 3)
            warped = F.grid_sample(lvl["moving"].unsqueeze(1), grid, mode="bilinear",
                                    padding_mode="zeros", align_corners=True).reshape(B, -1)
            return _batched_correlation_loss(warped, lvl["fixed_vals"])

        for _ in range(iters):
            opt.zero_grad(set_to_none=True)
            eval_loss().sum().backward()
            opt.step()
        with torch.no_grad():
            score = eval_loss().cpu().numpy()
        return t_cand.detach().cpu().numpy(), omega_cand.detach().cpu().numpy(), score

    rot_seeds = [np.zeros(3)]
    for axis in range(3):
        for deg in (20.0, -20.0):
            v = np.zeros(3)
            v[axis] = np.radians(deg)
            rot_seeds.append(v)

    all_t_candidates = [t_init_com] + pc_candidates
    fit_results = []
    for t_cand in all_t_candidates:
        for seed in rot_seeds:
            fit_results.append(fit_pair(t_cand, seed))
    all_scores = np.stack([s for _, _, s in fit_results], axis=0)
    best_combo = all_scores.argmin(axis=0)
    t_init = np.stack([fit_results[best_combo[b]][0][b] for b in range(B)], axis=0).astype(np.float32)
    omega_init = np.stack([fit_results[best_combo[b]][1][b] for b in range(B)], axis=0).astype(np.float32)
    if verbose:
        print(f"[motion_batched] translation+rotation init: searched {len(all_t_candidates)} "
              f"translation candidates x {len(rot_seeds)} rotation seeds = "
              f"{len(fit_results)} combos/frame")

    omega = torch.tensor(omega_init, device=dev, requires_grad=True)
    t_param = torch.tensor(t_init, device=dev, requires_grad=True)

    # Coarse-to-fine schedule: Adam at real avg_pool3d-downsampled levels (broad, cheap
    # capture) -> LBFGS at level 1 with point-sampling (precise polish). Dual-metric
    # (Mattes MI + correlation) at the coarse stages only -- see module docstring / session
    # doc for why correlation specifically helps at coarse resolution for this
    # intra-subject case.
    mid_level = min(2, max_level)
    schedule = [
        dict(optimizer="adam", iters=80, lr_t=0.08, lr_r=0.04, level=coarse_level, sampling=0.2, corr_weight=1.0),
        dict(optimizer="adam", iters=40, lr_t=0.03, lr_r=0.015, level=mid_level, sampling=0.2, corr_weight=1.0),
        dict(optimizer="lbfgs", iters=18, lr_t=0.2, lr_r=0.05, sampling=0.2, level=1),
        dict(optimizer="lbfgs", iters=15, lr_t=0.05, lr_r=0.01, sampling=0.2, level=1),
    ]

    def compute_loss(X_stage, w_y_stage, moving_tensor, shape_stage, level_stage=1,
                      fixed_vals_stage=None, corr_weight=0.0):
        R = _batched_rodrigues(omega)
        Xc = X_stage - C_phys
        y_phys = torch.einsum("bij,sj->bsi", R, Xc) + C_phys + t_param.unsqueeze(1)
        grid_norm = warp_to_moving_norm(y_phys, shape_stage, level=level_stage)
        grid = grid_norm.view(B, 1, 1, -1, 3)
        warped = F.grid_sample(moving_tensor.unsqueeze(1), grid, mode="bilinear",
                                padding_mode="zeros", align_corners=True).reshape(B, -1)
        w_scaled = (warped - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
        w_x = _batched_parzen_weights(w_scaled, num_bins=num_bins)
        per_frame_loss = _batched_mattes_mi_loss(w_x, w_y_stage)
        if corr_weight > 0.0 and fixed_vals_stage is not None:
            per_frame_loss = per_frame_loss + corr_weight * _batched_correlation_loss(warped, fixed_vals_stage)
        return per_frame_loss

    for stage in schedule:
        level = stage.get("level", 1)
        samp = stage.get("sampling", 1.0)
        if level == 1:
            lvl_data = get_pyramid_level(1)
            if samp < 1.0:
                k = max(500, int(samp * lvl_data["X"].shape[0]))
                sub = torch.randperm(lvl_data["X"].shape[0], generator=rng)[:k].to(dev)
                X_stage, w_y_stage = lvl_data["X"][sub], lvl_data["w_y"][sub]
            else:
                X_stage, w_y_stage = lvl_data["X"], lvl_data["w_y"]
        else:
            lvl_data = get_pyramid_level(level, point_sample_frac=min(samp, 1.0))
            X_stage, w_y_stage = lvl_data["X"], lvl_data["w_y"]
        moving_tensor, shape_stage = lvl_data["moving"], lvl_data["shape"]
        fixed_vals_stage = lvl_data["fixed_vals"]
        corr_weight = stage.get("corr_weight", 0.0)

        if stage.get("optimizer", "adam") == "lbfgs":
            opt = torch.optim.LBFGS([t_param, omega], lr=1.0, max_iter=stage["iters"],
                                     history_size=10, tolerance_grad=1e-9,
                                     tolerance_change=1e-11, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad(set_to_none=True)
                l = compute_loss(X_stage, w_y_stage, moving_tensor, shape_stage, level_stage=level,
                                  fixed_vals_stage=fixed_vals_stage, corr_weight=corr_weight).sum()
                l.backward()
                return l

            opt.step(closure)
        else:
            opt = torch.optim.Adam([
                {"params": [t_param], "lr": stage["lr_t"]},
                {"params": [omega], "lr": stage["lr_r"]},
            ])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=stage["iters"],
                                                                 eta_min=stage["lr_t"] * 0.02)
            for _ in range(stage["iters"]):
                opt.zero_grad(set_to_none=True)
                per_frame_loss = compute_loss(X_stage, w_y_stage, moving_tensor, shape_stage,
                                               level_stage=level, fixed_vals_stage=fixed_vals_stage,
                                               corr_weight=corr_weight)
                per_frame_loss.sum().backward()
                opt.step()
                sched.step()

    elapsed = time.time() - t0

    # -- Write per-frame results as ANTs transform files, matching robust_affine's
    # fwdtransforms/invtransforms shape (list-of-list-of-paths). Rigid, so the same .mat
    # file serves as both forward and inverse-eligible (whichtoinvert semantics handled by
    # the caller the same way robust_affine's own output already is).
    if outprefix is None:
        work_dir = tempfile.mkdtemp(prefix="syntx_batched_moco_")
        prefix = os.path.join(work_dir, "vol")
    else:
        prefix = outprefix

    R_final = _batched_rodrigues(omega).detach().cpu().numpy()
    t_final = t_param.detach().cpu().numpy()
    fwd_transforms, inv_transforms = [], []
    for b in range(B):
        tx = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3)
        tx.set_parameters(np.concatenate([R_final[b].flatten(), t_final[b]]))
        tx.set_fixed_parameters(com_f)
        tx_path = f"{prefix}{b:04d}.mat"
        ants.write_transform(tx, tx_path)
        fwd_transforms.append([tx_path])
        inv_transforms.append([tx_path])

    return fwd_transforms, inv_transforms, elapsed
