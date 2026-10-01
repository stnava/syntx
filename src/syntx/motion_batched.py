"""
Batched rigid registration of many 3-D frames to one shared reference, for motion correction.

All frames are optimised together: one forward / backward pass per optimisation step covers
the whole series (per-frame rotation and translation parameters, per-frame losses summed),
instead of one registration call per frame. Used by ``syntx.motion.motion_correction`` with
``backend='pytorch_batched'``.

- ``batched_rigid_register_pass``: independent rigid transform per frame.
- ``batched_group_bias_register_pass``: per-frame rigid transforms plus one shared rigid
  "group bias" transform applied to a subset of frames (e.g. DWI frames vs b0 frames).

Similarity: Mattes mutual information (cubic B-spline Parzen windows) on a fixed random
sample of foreground reference voxels, optionally plus negative Pearson correlation at the
coarse stages. Rotations are axis-angle (Rodrigues) about the reference's intensity-weighted
centre of mass. Results are written as ANTs ``AffineTransform`` ``.mat`` files.

Limitations: 3-D only (``NotImplementedError`` otherwise); all frames must share the
reference's grid (not checked); no mask support. Background and benchmark results (speed and
accuracy versus ``ants.registration``) are in
``docs/SESSION_2026-09-25_PHASE_CORRELATION_AND_BATCHED_MOTION_CORRECTION.md`` and the
prototypes ``scripts/prototype_batched_motion_correction.py`` /
``scripts/prototype_batched_joint_group_bias.py``.
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
from .core.grid import grid_sample_nd, _image_spatial_gradient


def _batched_rodrigues(omega: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula: axis-angle vectors omega (B, 3), radians, -> rotation matrices (B, 3, 3).
    The angle is clamped to >= 1e-8, so omega = 0 gives the identity."""
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
    """
    Per-frame negative mutual information from Parzen weights.

    ``w_x_batch`` (B, S, nbins) are the warped frames' weights, ``w_y`` (S, nbins) the shared
    reference weights for the same S sample points. The joint histograms are accumulated in
    zero-padded blocks of ``chunk`` samples. Returns a tensor of shape (B,): -MI per frame.
    """
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
    """Cubic B-spline Parzen weights for a batch, same mapping as ``core.losses.parzen_weights``:
    values clamped to [min_val, max_val] (NaN -> 0) and mapped onto bins [pad, num_bins-1-pad].
    ``v_batch`` (B, S) -> (B, S, num_bins)."""
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
    """Negative Pearson correlation per frame: ``warped`` (B, S) against ``fixed_vals`` (S,) ->
    (B,), in [-1, 1]. Used for the seed search and, added to Mattes MI, at the coarse stages
    (frames of one subject and contrast are close to linearly related)."""
    w_mean = warped.mean(dim=1, keepdim=True)
    f_mean = fixed_vals.mean()
    w_c = warped - w_mean
    f_c = fixed_vals - f_mean
    cov = (w_c * f_c).sum(dim=1)
    w_std = torch.sqrt((w_c ** 2).sum(dim=1) + 1e-8)
    f_std = torch.sqrt((f_c ** 2).sum() + 1e-8)
    return -(cov / (w_std * f_std + 1e-8))


def _auto_device() -> str:
    """'cuda' if available, else 'mps' if available, else 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def batched_rigid_register_pass(
    reference_img: ants.ANTsImage,
    moving_imgs: List[ants.ANTsImage],
    device: str = "auto",
    num_bins: int = 18,
    verbose: bool = False,
    outprefix: Optional[str] = None,
) -> Tuple[List[List[str]], List[List[str]], float]:
    """
    Rigidly register every frame in ``moving_imgs`` to the same ``reference_img`` in one
    batched optimisation.

    Algorithm:

    1. Sample ``max(2000, 15 %)`` of the reference voxels with intensity > 0.01 (seeded,
       deterministic). Intensities are scaled with the reference range [0, max(reference)]
       for the histograms (assumes non-negative images).
    2. Seed search at the coarse level ``min(4, max_level)`` (``max_level`` keeps at least 8
       voxels on the smallest axis): translation candidates = centre-of-mass offset plus the
       top 5 phase-correlation peaks; rotation seeds = 0 and +-20 degrees about each axis. Each
       of the 6 x 7 combinations is refined with 30 Adam steps on the correlation loss; the
       best one is kept per frame.
    3. Schedule: Adam (80 iterations) at the coarse level and Adam (40) at level
       ``min(2, max_level)``, both on MI + correlation with 20 % point sampling and cosine
       learning-rate decay; two joint LBFGS stages (18, 15 iterations, strong-Wolfe line
       search, 20 % sampling) at full resolution; then translation-only and rotation-only
       LBFGS stages (15 each) on all sampled points. Coarse levels are ``avg_pool3d``
       reductions; sampling uses the analytic-gradient bilinear path of
       ``core.grid.grid_sample_nd``.

    Parameters
    ----------
    reference_img : ants.ANTsImage
        Shared fixed image (3-D).
    moving_imgs : list of ants.ANTsImage
        Frames to register; must have the reference's shape, spacing, origin and direction
        (not checked).
    device : str, default 'auto'
        'auto' picks cuda, then mps, then cpu; any other value is used as given.
    num_bins : int, default 18
        Mattes MI histogram bins (the session doc reports 18-20 bins working better than 32 on
        b0/DWI data). ``syntx.motion.motion_correction`` passes ``kwargs.get('num_bins', 32)``,
        so through that caller the default is 32.
    verbose : bool, default False
        Prints one line about the seed search.
    outprefix : str, optional
        Transform files are written as ``f"{outprefix}{t:04d}.mat"`` (t = index in
        ``moving_imgs``); without it, as ``vol{t:04d}.mat`` in a new temporary directory
        (not deleted).

    Returns
    -------
    fwd_transforms : list of list of str
        One single-element list per frame: the path of an ANTs ``AffineTransform`` (rotation
        about the reference's centre of mass, then translation) mapping reference-space
        points into the frame, i.e. usable directly in ``ants.apply_transforms``.
    inv_transforms : list of list of str
        The same paths (callers invert them with ``whichtoinvert``).
    elapsed : float
        Seconds spent from set-up to the end of the optimisation (file writing excluded).
        ``([], [], 0.0)`` for an empty ``moving_imgs``.

    Raises
    ------
    NotImplementedError
        If ``reference_img.dimension != 3``.
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
    _grad_I_cache = {}

    def get_grad_I(moving_tensor):
        # Cached by tensor identity: `moving_tensor` objects here are always one of the few
        # pyramid-level tensors created ONCE by get_pyramid_level and reused across every
        # stage/iteration that references that level -- never recreated per-call. Recomputing
        # this on every LBFGS/Adam iteration (as the analytic-gradient grid_sample path would
        # otherwise force) was the dominant new per-iteration cost from switching to it;
        # caching restores the original amortized-cost design this module is built around.
        key = id(moving_tensor)
        if key not in _grad_I_cache:
            _grad_I_cache[key] = _image_spatial_gradient(moving_tensor.unsqueeze(1))
        return _grad_I_cache[key]


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
        # Validated final polish (see docs/SESSION_2026-09-25_..._MOTION_CORRECTION.md
        # Sec 8/10/13 for the full investigation this came out of): alternating
        # (coordinate-descent) translation-only / rotation-only LBFGS instead of one joint
        # stage -- jointly optimizing both at this precision was found to trade them off
        # against each other (an intervention that helped one measurably hurt the other),
        # where alternating lets each converge without disturbing the other.
        #
        # Every stage here (this one and the coarse ones above) uses `grid_sample_nd`'s
        # analytic-gradient bilinear mode (`syntx.core.grid`) rather than plain
        # `F.grid_sample`: forward pass is identical (still trilinear), but the backward
        # pass is an exact image-spatial-gradient computation instead of autograd's default
        # through the discrete sampling op. This turned out to be THE fix for rotation
        # precision specifically (a hand-rolled tricubic/Catmull-Rom interpolator was tried
        # first and only partly helped translation; switching the FORWARD interpolation
        # scheme was the wrong lever -- the gradient computation was) -- validated on two
        # independent ground-truth caches, EXCEEDING ants on translation (0.0626-0.0638mm
        # vs ants' 0.0645-0.0716mm) AND rotation (0.0214-0.0240deg vs ants' 0.0332-0.0374deg)
        # simultaneously. See Sec 13 for the full before/after and the other approaches
        # (ensembling, oracle/label-free best-of-N, Gaussian smoothing, Gauss-Newton,
        # radius-biased sampling, a hand-rolled tricubic interpolator) tried before this.
        #
        # Only ONE round (2 stages, not 4): confirmed directly that a 2nd round gives
        # IDENTICAL accuracy (already fully converged after 1) while costing ~40% more
        # wall-clock -- LBFGS's strong_wolfe line search evaluates the closure (a full
        # forward+backward) many times per "iter", so each of these stages is markedly
        # more expensive than the coarse Adam stages above at the same iters count.
        dict(optimizer="lbfgs", iters=15, lr_t=0.01, lr_r=0.002, sampling=1.0, level=1, freeze="r"),
        dict(optimizer="lbfgs", iters=15, lr_t=0.01, lr_r=0.002, sampling=1.0, level=1, freeze="t"),
    ]

    def compute_loss(X_stage, w_y_stage, moving_tensor, shape_stage, level_stage=1,
                      fixed_vals_stage=None, corr_weight=0.0):
        R = _batched_rodrigues(omega)
        Xc = X_stage - C_phys
        y_phys = torch.einsum("bij,sj->bsi", R, Xc) + C_phys + t_param.unsqueeze(1)
        grid_norm = warp_to_moving_norm(y_phys, shape_stage, level=level_stage)
        grid = grid_norm.view(B, 1, 1, -1, 3)
        warped = grid_sample_nd(moving_tensor.unsqueeze(1), grid, mode="bilinear",
                                 padding_mode="zeros", align_corners=True,
                                 interpolator="linear", use_analytical_gradients=True,
                                 precomputed_grad_I=get_grad_I(moving_tensor)).reshape(B, -1)
        w_scaled = (warped - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
        w_x = _batched_parzen_weights(w_scaled, num_bins=num_bins)
        per_frame_loss = _batched_mattes_mi_loss(w_x, w_y_stage)
        if corr_weight > 0.0 and fixed_vals_stage is not None:
            per_frame_loss = per_frame_loss + corr_weight * _batched_correlation_loss(warped, fixed_vals_stage)
        return per_frame_loss

    for stage in schedule:
        level = stage.get("level", 1)
        samp = stage.get("sampling", 1.0)
        freeze = stage.get("freeze", None)  # 'r' freezes omega (translation-only update);
                                             # 't' freezes t_param (rotation-only update).
        if level == 1:
            lvl_data = get_pyramid_level(1)
            if samp < 1.0:
                k = max(500, int(samp * lvl_data["X"].shape[0]))
                sub = torch.randperm(lvl_data["X"].shape[0], generator=rng)[:k].to(dev)
                X_stage, w_y_stage = lvl_data["X"][sub], lvl_data["w_y"][sub]
                fixed_vals_stage = lvl_data["fixed_vals"][sub]
            else:
                X_stage, w_y_stage = lvl_data["X"], lvl_data["w_y"]
                fixed_vals_stage = lvl_data["fixed_vals"]
        else:
            lvl_data = get_pyramid_level(level, point_sample_frac=min(samp, 1.0))
            X_stage, w_y_stage = lvl_data["X"], lvl_data["w_y"]
            fixed_vals_stage = lvl_data["fixed_vals"]
        moving_tensor, shape_stage = lvl_data["moving"], lvl_data["shape"]
        corr_weight = stage.get("corr_weight", 0.0)

        if freeze == "r":
            opt_params = [t_param]
        elif freeze == "t":
            opt_params = [omega]
        else:
            opt_params = [t_param, omega]

        if stage.get("optimizer", "adam") == "lbfgs":
            opt = torch.optim.LBFGS(opt_params, lr=1.0, max_iter=stage["iters"],
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
            param_groups = []
            if freeze != "t":
                param_groups.append({"params": [t_param], "lr": stage["lr_t"]})
            if freeze != "r":
                param_groups.append({"params": [omega], "lr": stage["lr_r"]})
            opt = torch.optim.Adam(param_groups)
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


def batched_group_bias_register_pass(
    reference_img: ants.ANTsImage,
    moving_imgs: List[ants.ANTsImage],
    group_mask: List[bool],
    device: str = "auto",
    num_bins: int = 32,
    verbose: bool = False,
    outprefix: Optional[str] = None,
) -> Tuple[List[List[str]], List[List[str]], np.ndarray, np.ndarray, float]:
    """
    Like ``batched_rigid_register_pass``, but for a series from two acquisition groups that
    differ by a systematic rigid offset (e.g. b0 and DWI frames of one session).

    Jointly estimates, in one batched optimisation, a rigid transform per frame and one
    shared rigid group-bias transform (R_group, t_group) applied only to frames with
    ``group_mask[b] == True``, composed as the outer transform:
    ``R_total = R_group @ R_frame``, ``t_total = R_group @ t_frame + t_group`` (rotations about
    the reference's centre of mass). Frames with ``group_mask[b] == False`` are plain
    per-frame rigid registrations.

    Differences from ``batched_rigid_register_pass``: no phase-correlation / rotation seed
    search (translations start from centre-of-mass offsets, split into the mean over group
    frames for ``t_group`` and per-frame residuals; all rotations start at zero); no pyramid
    (all stages at full resolution on the sampled points); MI only (no correlation term);
    ``num_bins`` default 32. Schedule: Adam 130 and 50 iterations (18 % sampling), then
    LBFGS 18 (15 %) and 15 (30 %) iterations.

    Parameters
    ----------
    reference_img : ants.ANTsImage
        Shared fixed image (3-D), typically the mean of the ``group_mask == False`` frames.
    moving_imgs : list of ants.ANTsImage
        All frames of both groups, on the reference's grid (not checked).
    group_mask : list of bool
        One entry per frame; True where the shared group-bias transform applies.
    device : str, default 'auto'
        'auto' picks cuda, then mps, then cpu.
    num_bins : int, default 32
        Mattes MI histogram bins.
    verbose : bool, default False
        Unused.
    outprefix : str, optional
        Files are ``f"{outprefix}{t:04d}.mat"``; default ``vol{t:04d}.mat`` in a new temporary
        directory.

    Returns
    -------
    fwd_transforms, inv_transforms : list of list of str
        As in ``batched_rigid_register_pass``; each file holds the total per-frame transform
        (frame transform composed with the group bias where it applies). Both lists hold the
        same paths.
    R_group : np.ndarray (3, 3)
        Estimated group-bias rotation (identity if ``group_mask`` is all False).
    t_group : np.ndarray (3,)
        Estimated group-bias translation (physical units).
    elapsed : float
        Seconds from set-up to the end of the optimisation. An empty ``moving_imgs`` returns
        ``([], [], np.eye(3), np.zeros(3), 0.0)``.

    Raises
    ------
    NotImplementedError
        If ``reference_img.dimension != 3``.
    ValueError
        If ``len(group_mask) != len(moving_imgs)``.
    """
    dim = reference_img.dimension
    if dim != 3:
        raise NotImplementedError(
            "motion_batched.batched_group_bias_register_pass only implements the 3D path."
        )
    if len(group_mask) != len(moving_imgs):
        raise ValueError(
            f"group_mask must have one entry per moving image, got {len(group_mask)} "
            f"for {len(moving_imgs)} images."
        )
    dev = torch.device(_auto_device() if device == "auto" else device)
    B = len(moving_imgs)
    if B == 0:
        return [], [], np.eye(3), np.zeros(3), 0.0

    import time
    t0 = time.time()

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

    dwi_mask_np = np.asarray(group_mask, dtype=bool)
    dwi_mask = torch.tensor(dwi_mask_np, device=dev)
    Ndwi = int(dwi_mask_np.sum())

    # Per-frame CoM offsets vs. the reference's own CoM. For non-group (b0-like) frames
    # this is the direct t_param init (group component is a no-op there via the mask).
    # For group frames, seed the SHARED t_group from the mean group-frame CoM offset (the
    # common component) and each group frame's own t_param from its RESIDUAL after
    # subtracting that mean -- starting near the right per-frame/group decomposition
    # instead of leaving the optimizer to discover the split from scratch.
    com_offsets = np.zeros((B, 3), dtype=np.float32)
    for b, m in enumerate(moving_imgs):
        com_m = np.asarray(compute_center_of_mass(m, weighted=True), dtype=np.float64)
        com_offsets[b] = (com_m - com_f).astype(np.float32)
    t_group_init = com_offsets[dwi_mask_np].mean(axis=0) if Ndwi > 0 else np.zeros(3, dtype=np.float32)
    t_init = com_offsets.copy()
    t_init[dwi_mask_np] -= t_group_init

    # Deliberately NOT using the single-group solver's phase-correlation + fit_pair coarse
    # candidate search here: that machinery fits each frame as an ISOLATED rigid transform
    # against the reference (no group concept), so for group-masked frames it converges near
    # the TOTAL offset (frame jitter + shared group bias combined). Assigning that directly
    # as t_param (meant to hold only the frame-LOCAL component) while t_group is ALSO
    # separately nonzero double-counts the shared component for every group frame --
    # confirmed empirically: adding that seeding step here made recovery WORSE (1.5-2.7mm
    # group-bias error) than the simpler CoM-decomposed init below (1.238mm, matching
    # `scripts/prototype_batched_joint_group_bias.py`'s validated result), even after
    # attempting to correct the double-count by subtracting t_group_init back out. The
    # decomposed CoM init plus the joint schedule's own coarse Adam stage is what's
    # validated to work for THIS (group-bias) model; omega starts at zero (real head
    # rotations are small enough that Adam's coarse stage captures them directly).
    omega = torch.zeros(B, 3, device=dev, requires_grad=True)
    t_param = torch.tensor(t_init, device=dev, requires_grad=True)
    omega_group = torch.zeros(3, device=dev, requires_grad=True)
    t_group = torch.tensor(t_group_init, device=dev, requires_grad=True)

    # Point-sampled at FULL resolution throughout (no avg-pool coarse pyramid levels): the
    # true multi-res pyramid used by the single-group solver was tried here and confirmed
    # to HURT this decomposition task specifically (worse group-bias recovery, 1.4-2.7mm
    # vs 1.238mm) -- averaging pools the frame-local and shared-group signals together at
    # very few effective voxels/frame, exactly when the optimizer most needs to tell them
    # apart. This matches `scripts/prototype_batched_joint_group_bias.py`'s validated
    # schedule shape (point-sampling fraction only, never true downsampling).
    schedule = [
        dict(optimizer="adam", iters=130, lr_t=0.08, lr_r=0.04, level=1, sampling=0.18),
        dict(optimizer="adam", iters=50, lr_t=0.02, lr_r=0.006, level=1, sampling=0.18),
        dict(optimizer="lbfgs", iters=18, lr_t=0.2, lr_r=0.05, level=1, sampling=0.15),
        dict(optimizer="lbfgs", iters=15, lr_t=0.05, lr_r=0.01, level=1, sampling=0.3),
    ]

    def compute_loss(X_stage, w_y_stage, moving_tensor, shape_stage, level_stage=1,
                      fixed_vals_stage=None, corr_weight=0.0):
        R_frame = _batched_rodrigues(omega)
        R_group = _batched_rodrigues(omega_group.unsqueeze(0))[0]
        Xc = X_stage - C_phys

        # Total per-frame transform composes the shared group-bias transform SECOND
        # (outer): R_total = R_group @ R_frame, t_total = R_group @ t_frame + t_group --
        # see the module-level docstring on `batched_group_bias_register_pass` and
        # `scripts/prototype_batched_joint_group_bias.py` for the pull-back convention
        # this matches. Non-group frames get R_group=I, t_group=0 via the mask, leaving
        # them as plain per-frame rigid registration.
        R_group_b = R_group.unsqueeze(0).expand(B, -1, -1)
        t_group_b = t_group.unsqueeze(0).expand(B, -1)
        apply_group = dwi_mask.float().view(B, 1, 1)
        eye3 = torch.eye(3, device=dev).unsqueeze(0)
        R_eff_group = apply_group * R_group_b + (1 - apply_group) * eye3
        t_eff_group = dwi_mask.float().view(B, 1) * t_group_b

        R_total = torch.bmm(R_eff_group, R_frame)
        t_total = torch.einsum("bij,bj->bi", R_eff_group, t_param) + t_eff_group

        y_phys = torch.einsum("bij,sj->bsi", R_total, Xc) + C_phys + t_total.unsqueeze(1)
        grid_norm = warp_to_moving_norm(y_phys, shape_stage, level=level_stage)
        grid = grid_norm.view(B, 1, 1, -1, 3)
        warped = grid_sample_nd(moving_tensor.unsqueeze(1), grid, mode="bilinear",
                                 padding_mode="zeros", align_corners=True,
                                 interpolator="linear", use_analytical_gradients=True).reshape(B, -1)
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
                fixed_vals_stage = lvl_data["fixed_vals"][sub]
            else:
                X_stage, w_y_stage = lvl_data["X"], lvl_data["w_y"]
                fixed_vals_stage = lvl_data["fixed_vals"]
        else:
            lvl_data = get_pyramid_level(level, point_sample_frac=min(samp, 1.0))
            X_stage, w_y_stage = lvl_data["X"], lvl_data["w_y"]
            fixed_vals_stage = lvl_data["fixed_vals"]
        moving_tensor, shape_stage = lvl_data["moving"], lvl_data["shape"]
        corr_weight = stage.get("corr_weight", 0.0)
        params = [t_param, omega, t_group, omega_group]

        if stage.get("optimizer", "adam") == "lbfgs":
            opt = torch.optim.LBFGS(params, lr=1.0, max_iter=stage["iters"], history_size=10,
                                     tolerance_grad=1e-9, tolerance_change=1e-11,
                                     line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad(set_to_none=True)
                l = compute_loss(X_stage, w_y_stage, moving_tensor, shape_stage, level_stage=level,
                                  fixed_vals_stage=fixed_vals_stage, corr_weight=corr_weight).sum()
                l.backward()
                return l

            opt.step(closure)
        else:
            opt = torch.optim.Adam([
                {"params": [t_param, t_group], "lr": stage["lr_t"]},
                {"params": [omega, omega_group], "lr": stage["lr_r"]},
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

    R_frame_final = _batched_rodrigues(omega).detach().cpu().numpy()
    t_frame_final = t_param.detach().cpu().numpy()
    R_group_final = _batched_rodrigues(omega_group.unsqueeze(0))[0].detach().cpu().numpy()
    t_group_final = t_group.detach().cpu().numpy()

    if outprefix is None:
        work_dir = tempfile.mkdtemp(prefix="syntx_batched_moco_group_")
        prefix = os.path.join(work_dir, "vol")
    else:
        prefix = outprefix

    # Write the TOTAL per-frame transform (jitter composed with group bias where the
    # mask applies), so callers use this exactly like `batched_rigid_register_pass`'s
    # output -- no group-bias-aware logic needed downstream.
    fwd_transforms, inv_transforms = [], []
    for b in range(B):
        if dwi_mask_np[b]:
            R_b = R_group_final @ R_frame_final[b]
            t_b = R_group_final @ t_frame_final[b] + t_group_final
        else:
            R_b = R_frame_final[b]
            t_b = t_frame_final[b]
        tx = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3)
        tx.set_parameters(np.concatenate([R_b.flatten(), t_b]))
        tx.set_fixed_parameters(com_f)
        tx_path = f"{prefix}{b:04d}.mat"
        ants.write_transform(tx, tx_path)
        fwd_transforms.append([tx_path])
        inv_transforms.append([tx_path])

    return fwd_transforms, inv_transforms, R_group_final, t_group_final, elapsed
