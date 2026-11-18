"""Physical-space-first adapters for external deformable registration baselines.

Provides standardized ``syntx`` registration-dictionary wrappers for:

- ``run_voxelmorph_pretrained``: Official DalcaLab PyTorch ``voxelmorph`` 3-D
  ``VxmPairwise`` brain weights fetched via ``antsxdata``
  (``syntx_features/voxelmorph_pairwise_3d_brain``), with lossless signed axis
  permutation into the pretrained canonical anatomical frame and optional
  full-resolution SVF velocity refinement (``polish_steps > 0``).
- ``run_voxelmorph_multiscale``: ``[4, 2, 1]`` multi-scale ``VxmPairwise`` SVF
  registration built on ``syntx.pyramid.build_pyramid(..., policy="deformable")``
  and sync-free ``5^3`` LNCC (``syntx.core.losses.cross_correlation.local_ncc_loss_nd``).
- ``run_fireants_greedy``: Repaired ``FireANTs`` ``GreedyRegistration`` adapter
  with decoupled ``[1Warp.nii.gz, initial_transform]`` ITK LPS export.

All adapters return standard ``syntx`` registration dictionaries
(``fwdtransforms``, ``invtransforms``, ``whichtoinvert_inv``, ``warpedmovout``,
``provenance``) with single-interpolation physical-space transform composition.
"""

from __future__ import annotations

import importlib.metadata
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ants
import numpy as np
import torch

from ..core.grid import resize_field
from ..core.losses.cross_correlation import local_ncc_loss_nd
from ..core.pipeline import cleanup_gpu
from ..core.smoothing import separable_gaussian_filter
from ..device import resolve_device
from ..pyramid import build_pyramid
from ..transforms import apply_transforms as syntx_apply_transforms


_I3 = np.eye(3, dtype=np.float64)
_LPS_AXIS_VECTORS: Dict[str, np.ndarray] = {
    "L": _I3[0],
    "R": -_I3[0],
    "P": _I3[1],
    "A": -_I3[1],
    "S": _I3[2],
    "I": -_I3[2],
}

_AXIS_PAIR_FAMILY: Dict[str, int] = {
    "L": 0,
    "R": 0,
    "P": 1,
    "A": 1,
    "S": 2,
    "I": 2,
}


def _sync_device(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _release_device_cache(device: str) -> None:
    cleanup_gpu(device)


def canonical_orientation_transform(
    direction: np.ndarray,
    target_orientation: str = "PIR",
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Derive lossless signed axis permutation mapping ``direction`` to ``target_orientation``.

    Parameters
    ----------
    direction : ndarray of shape (3, 3)
        ITK LPS direction cosine matrix of the native image (column ``j`` is the
        LPS physical direction of native voxel axis ``j``).
    target_orientation : str, default 'PIR'
        3-character anatomical orientation code (one axis from each of
        ``{L, R}``, ``{P, A}``, ``{S, I}``) naming the positive direction of
        each output array axis in ITK LPS coordinates.

    Returns
    -------
    perm : tuple of 3 ints
        Axis permutation such that output axis ``i`` corresponds to native axis
        ``perm[i]``.
    flips : tuple of 3 ints (each +1 or -1)
        Sign flip applied along each permuted axis ``i`` so output axis ``i``
        aligns positively with ``target_orientation[i]``.
    """
    code = str(target_orientation).upper().strip()
    if len(code) != 3 or any(ch not in _LPS_AXIS_VECTORS for ch in code):
        raise ValueError(
            f"target_orientation must be a 3-letter anatomical code from L/R/P/A/S/I, got {target_orientation!r}"
        )
    if len({_AXIS_PAIR_FAMILY[ch] for ch in code}) != 3:
        raise ValueError(
            f"target_orientation must span all 3 anatomical axes, got {target_orientation!r}"
        )

    dir_mat = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    target_basis = np.column_stack([_LPS_AXIS_VECTORS[ch] for ch in code])
    overlap = dir_mat.T @ target_basis  # overlap[j, i] = D[:, j] . C[:, i]

    perm: List[int] = []
    flips: List[int] = []
    used_axes: set[int] = set()
    for out_ax in range(3):
        scores = np.abs(overlap[:, out_ax]).copy()
        for prev in used_axes:
            scores[prev] = -1.0
        src_ax = int(np.argmax(scores))
        used_axes.add(src_ax)
        perm.append(src_ax)
        flips.append(1 if overlap[src_ax, out_ax] >= 0.0 else -1)
    return (perm[0], perm[1], perm[2]), (flips[0], flips[1], flips[2])


def apply_axis_permutation(
    tensor: torch.Tensor,
    perm: Sequence[int],
    flips: Sequence[int],
) -> torch.Tensor:
    """Apply lossless signed axis permutation to a ``(B, C, *spatial)`` tensor."""
    permuted = tensor.permute(0, 1, *(2 + int(p) for p in perm))
    flip_dims = [2 + i for i, s in enumerate(flips) if int(s) < 0]
    if flip_dims:
        permuted = torch.flip(permuted, dims=flip_dims)
    return permuted.contiguous()


def invert_displacement_permutation(
    disp: torch.Tensor,
    perm: Sequence[int],
    flips: Sequence[int],
) -> torch.Tensor:
    """Invert ``apply_axis_permutation`` on a voxel-unit displacement/velocity field."""
    flip_dims = [2 + i for i, s in enumerate(flips) if int(s) < 0]
    out = torch.flip(disp, dims=flip_dims) if flip_dims else disp
    signs = torch.tensor(
        [float(s) for s in flips], dtype=out.dtype, device=out.device
    ).view(1, 3, 1, 1, 1)
    out = out * signs

    inv_perm = [0, 0, 0]
    for i, p in enumerate(perm):
        inv_perm[int(p)] = i
    out = out[:, inv_perm, ...]
    out = out.permute(0, 1, *(2 + p for p in inv_perm))
    return out.contiguous()


def _pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int = 16,
) -> Tuple[torch.Tensor, Tuple[Tuple[int, int], ...]]:
    """Symmetrically zero-pad a 5-D tensor to multiples of ``multiple`` via slice copy.

    Avoids ``torch.nn.functional.pad`` on 5-D MPS tensors (``GEMINI.md`` §4).
    """
    spatial = tensor.shape[2:]
    pads: List[Tuple[int, int]] = []
    target_shape: List[int] = []
    needs_pad = False
    for size in spatial:
        rem = size % multiple
        total_pad = (multiple - rem) if rem != 0 else 0
        p0 = total_pad // 2
        p1 = total_pad - p0
        pads.append((p0, p1))
        target_shape.append(size + total_pad)
        if total_pad > 0:
            needs_pad = True
    if not needs_pad:
        return tensor, tuple(pads)

    out = torch.zeros(
        (tensor.shape[0], tensor.shape[1], *target_shape),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    slices = (
        slice(None),
        slice(None),
        *(slice(p0, p0 + s) for (p0, _), s in zip(pads, spatial)),
    )
    out[slices] = tensor
    return out, tuple(pads)


def _crop_from_pads(
    tensor: torch.Tensor,
    pads: Tuple[Tuple[int, int], ...],
) -> torch.Tensor:
    """Remove symmetric padding added by ``_pad_to_multiple``."""
    spatial = tensor.shape[2:]
    slices = (
        slice(None),
        slice(None),
        *(slice(p0, s - p1) for (p0, p1), s in zip(pads, spatial)),
    )
    return tensor[slices].contiguous()


def voxel_disp_to_itk_lps(
    disp_vox: torch.Tensor,
    ref_image: ants.ANTsImage,
) -> ants.ANTsImage:
    """Convert a ``(1, 3, nx, ny, nz)`` voxel-index displacement tensor to ITK LPS ``ANTsImage``."""
    d_np = disp_vox.detach().cpu().numpy()[0]
    d_xyz = np.moveaxis(d_np, 0, -1)
    jac_vox2lps = np.asarray(ref_image.direction, dtype=np.float64) @ np.diag(
        np.asarray(ref_image.spacing, dtype=np.float64)
    )
    disp_lps = np.einsum("ij,...j->...i", jac_vox2lps, d_xyz).astype(np.float32)
    return ants.from_numpy(
        np.ascontiguousarray(disp_lps),
        origin=ref_image.origin,
        spacing=ref_image.spacing,
        direction=ref_image.direction,
        has_components=True,
    )


def ants_mat_to_4x4(mat_path: str) -> np.ndarray:
    """Convert an ITK ``AffineTransform`` ``.mat`` file to a 4x4 physical LPS matrix."""
    tx_obj = ants.read_transform(mat_path)
    mat_a = np.asarray(tx_obj.parameters[:9], dtype=np.float64).reshape(3, 3)
    trans_t = np.asarray(tx_obj.parameters[9:12], dtype=np.float64)
    center_c = np.asarray(tx_obj.fixed_parameters[:3], dtype=np.float64)
    offset = trans_t + center_c - mat_a @ center_c
    mat_4x4 = np.eye(4, dtype=np.float64)
    mat_4x4[:3, :3] = mat_a
    mat_4x4[:3, 3] = offset
    return mat_4x4


def _smooth_cf(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """Isotropic voxel-unit Gaussian smoothing of a ``(B, C, *spatial)`` tensor."""
    if sigma <= 0.0:
        return tensor
    sig_tuple = [float(sigma)] * (tensor.dim() - 2)
    with torch.no_grad():
        return separable_gaussian_filter(
            tensor.movedim(1, -1), sig_tuple, kernel_type="bessel"
        ).movedim(-1, 1)


def _import_voxelmorph():
    try:
        import neurite as ne
        import voxelmorph as vxm
    except ImportError as exc:
        raise ImportError(
            "voxelmorph (>=0.3) and neurite (>=0.3) are required for VoxelMorph external baselines. "
            "See docs/EXTERNAL_BENCHMARK_BASELINES.md for --no-deps installation instructions."
        ) from exc
    return vxm, ne


def _prepare_moving_aligned(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    initial_transform: Optional[str],
) -> ants.ANTsImage:
    """Resample ``moving`` onto ``fixed``'s physical grid once in memory via ``initial_transform``."""
    tx_list = [initial_transform] if initial_transform else []
    return syntx_apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=tx_list,
        interpolator="linear",
    )


def _resolve_output_dir(out_dir: Optional[str], prefix: str) -> str:
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        return out_dir
    return tempfile.mkdtemp(prefix=prefix)


def _build_return_dict(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    fwd_warp_path: str,
    inv_warp_path: Optional[str],
    initial_transform: Optional[str],
    provenance: Dict[str, Any],
) -> Dict[str, Any]:
    fwd_tx = [fwd_warp_path, initial_transform] if initial_transform else [fwd_warp_path]
    if inv_warp_path is not None:
        inv_tx = [initial_transform, inv_warp_path] if initial_transform else [inv_warp_path]
        which_inv = [True, False] if initial_transform else [False]
    else:
        inv_tx = []
        which_inv = []
    warped = syntx_apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=fwd_tx,
        interpolator="linear",
    )
    return {
        "fwdtransforms": fwd_tx,
        "invtransforms": inv_tx,
        "whichtoinvert_inv": which_inv,
        "warpedmovout": warped,
        "provenance": provenance,
    }


def run_voxelmorph_pretrained(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    initial_transform: Optional[str] = None,
    polish_steps: int = 0,
    polish_lr: float = 0.12,
    grad_weight: float = 0.30,
    lncc_radius: int = 2,
    canonical_orientation: Optional[str] = "PIR",
    int_steps: int = 7,
    vel_smooth_sigma: float = 0.40,
    return_inverse: bool = False,
    device: Optional[str] = None,
    seed: int = 42,
    asset_ref: str = "syntx_features/voxelmorph_pairwise_3d_brain",
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run official DalcaLab pretrained 3-D ``VxmPairwise`` brain registration.

    Fetches ``asset_ref`` via ``antsxdata.fetch``, applies a lossless discrete
    signed axis permutation into ``canonical_orientation`` (default ``'PIR'``)
    so 3x3x3 convolutional filters operate in a consistent anatomical frame
    without extra resampling, and exports a single-interpolation ITK LPS
    ``1Warp.nii.gz`` composed with ``initial_transform``.

    When ``polish_steps == 0``, performs a pure zero-shot forward pass
    (``< 1 s`` on MPS). When ``polish_steps > 0``, refines the predicted SVF
    velocity field at native resolution using sync-free ``(2*lncc_radius+1)^3``
    LNCC (``local_ncc_loss_nd(..., squared=True)``) + L2 velocity-gradient
    regularization + Gaussian velocity smoothing.
    """
    if fixed.dimension != 3 or moving.dimension != 3:
        raise ValueError(
            f"run_voxelmorph_pretrained supports 3-D images only, got {fixed.dimension}-D / {moving.dimension}-D"
        )
    import antsxdata

    vxm, ne = _import_voxelmorph()
    dev = resolve_device(device, task="reg")
    _release_device_cache(dev)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    weights_path = str(antsxdata.fetch(asset_ref))
    asset_prov = antsxdata.provenance(asset_ref)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    t0 = time.perf_counter()
    mi_aff = _prepare_moving_aligned(fixed, moving, initial_transform)

    src_nat = torch.from_numpy(mi_aff.numpy()).float()[None, None].to(dev)
    tgt_nat = torch.from_numpy(fixed.numpy()).float()[None, None].to(dev)

    if canonical_orientation is not None:
        perm, flips = canonical_orientation_transform(
            np.asarray(fixed.direction, dtype=np.float64), canonical_orientation
        )
        src_can = apply_axis_permutation(src_nat, perm, flips)
        tgt_can = apply_axis_permutation(tgt_nat, perm, flips)
    else:
        perm, flips = (0, 1, 2), (1, 1, 1)
        src_can, tgt_can = src_nat, tgt_nat

    nb_features = (16, 16, 16, 16, 16)
    pad_mult = 2 ** len(nb_features)
    model = vxm.nn.models.VxmPairwise(
        ndim=3,
        source_channels=1,
        target_channels=1,
        nb_features=nb_features,
        integration_steps=int(int_steps),
        device=dev,
    ).to(dev)
    model.load_state_dict(ckpt)
    model.eval()

    src_pad, pads = _pad_to_multiple(src_can, pad_mult)
    tgt_pad, _ = _pad_to_multiple(tgt_can, pad_mult)

    with torch.no_grad():
        vel_can = _crop_from_pads(
            model(src_pad, tgt_pad, return_field_type="velocity"), pads
        )

    if int(polish_steps) > 0:
        vel_init = _smooth_cf(vel_can, float(vel_smooth_sigma))
        vel_opt = vel_init.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([vel_opt], lr=float(polish_lr))
        grad_fn = ne.nn.modules.SpatialGradient("l2")
        fg_mask = (tgt_can > 0.02).float()
        win_size = 2 * int(lncc_radius) + 1
        for _ in range(int(polish_steps)):
            opt.zero_grad()
            disp_step = model.velocity_field_integrator(vel_opt)
            warped_step = model.spatial_transformer(src_can, disp_step)
            loss_cc = local_ncc_loss_nd(
                tgt_can,
                warped_step,
                mask=fg_mask,
                window_size=win_size,
                squared=True,
                single_pass=False,
            )
            loss = loss_cc + float(grad_weight) * grad_fn(vel_opt)
            loss.backward()
            opt.step()
            if vel_smooth_sigma > 0.0:
                with torch.no_grad():
                    vel_opt.copy_(_smooth_cf(vel_opt, float(vel_smooth_sigma)))
        vel_can = vel_opt.detach()

    with torch.no_grad():
        disp_fwd_can = model.velocity_field_integrator(vel_can)
        disp_fwd_nat = invert_displacement_permutation(disp_fwd_can, perm, flips)
        disp_inv_nat = None
        if return_inverse:
            disp_inv_can = model.velocity_field_integrator(-vel_can)
            disp_inv_nat = invert_displacement_permutation(disp_inv_can, perm, flips)

    _sync_device(dev)
    runtime_s = time.perf_counter() - t0

    save_dir = _resolve_output_dir(out_dir, "syntx_vxm_pretrained_")
    fwd_warp_path = os.path.join(save_dir, "1Warp.nii.gz")
    ants.image_write(voxel_disp_to_itk_lps(disp_fwd_nat, fixed), fwd_warp_path)

    inv_warp_path: Optional[str] = None
    if disp_inv_nat is not None:
        inv_warp_path = os.path.join(save_dir, "1InverseWarp.nii.gz")
        ants.image_write(voxel_disp_to_itk_lps(disp_inv_nat, fixed), inv_warp_path)

    del model, src_nat, tgt_nat, src_can, tgt_can, src_pad, tgt_pad, vel_can, disp_fwd_nat
    _release_device_cache(dev)

    provenance = {
        "method": "voxelmorph_pretrained",
        "device": dev,
        "voxelmorph_version": importlib.metadata.version("voxelmorph"),
        "asset_ref": asset_prov.get("ref", asset_ref),
        "asset_sha256": asset_prov.get("sha256"),
        "canonical_orientation": canonical_orientation,
        "axis_perm": list(perm),
        "axis_flips": list(flips),
        "polish_steps": int(polish_steps),
        "polish_lr": float(polish_lr),
        "grad_weight": float(grad_weight),
        "lncc_radius": int(lncc_radius),
        "vel_smooth_sigma": float(vel_smooth_sigma),
        "int_steps": int(int_steps),
        "return_inverse": bool(return_inverse),
        "runtime_seconds": float(runtime_s),
    }
    return _build_return_dict(
        fixed, moving, fwd_warp_path, inv_warp_path, initial_transform, provenance
    )


def run_voxelmorph_multiscale(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    initial_transform: Optional[str] = None,
    scales: Sequence[int] = (4, 2, 1),
    reg_iterations: Sequence[int] = (100, 100, 50),
    lr: float = 2e-3,
    svf_lr: float = 0.12,
    grad_weight: float = 0.05,
    svf_grad_weight: float = 10.0,
    lncc_radius: int = 2,
    nb_features: Sequence[int] = (16, 32, 32, 32),
    int_steps: int = 7,
    final_smooth_sigma: float = 0.45,
    return_inverse: bool = False,
    device: Optional[str] = None,
    seed: int = 42,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run ``[4, 2, 1]`` multi-scale VoxelMorph SVF optimization on native physical grids.

    Uses ``syntx.pyramid.build_pyramid(..., policy='deformable')`` for anti-aliased
    scale-space construction and sync-free foreground-masked ``(2*lncc_radius+1)^3``
    squared LNCC (``local_ncc_loss_nd``). The coarsest level trains ``VxmPairwise``
    with constant ``lr``; finer levels resize the stationary velocity field via
    ``syntx.core.grid.resize_field`` (scaled by voxel ratio) and refine in SVF
    space with cosine LR decay on the finest level.
    """
    if fixed.dimension != 3 or moving.dimension != 3:
        raise ValueError(
            f"run_voxelmorph_multiscale supports 3-D images only, got {fixed.dimension}-D / {moving.dimension}-D"
        )
    scales_tup = tuple(int(s) for s in scales)
    iters_tup = tuple(int(k) for k in reg_iterations)
    if len(scales_tup) == 0 or len(scales_tup) != len(iters_tup):
        raise ValueError(
            f"scales ({scales_tup}) and reg_iterations ({iters_tup}) must be non-empty and match in length"
        )

    vxm, ne = _import_voxelmorph()
    dev = resolve_device(device, task="reg")
    _release_device_cache(dev)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    t0 = time.perf_counter()
    mi_aff = _prepare_moving_aligned(fixed, moving, initial_transform)

    # build_pyramid takes tensor (B, C, z, y, x) with ANTs spacing (x, y, z)
    src_zyx = torch.from_numpy(mi_aff.numpy()).float()[None, None].to(dev).permute(0, 1, 4, 3, 2).contiguous()
    tgt_zyx = torch.from_numpy(fixed.numpy()).float()[None, None].to(dev).permute(0, 1, 4, 3, 2).contiguous()
    pyr_src = build_pyramid(src_zyx, fixed.spacing, levels=scales_tup, policy="deformable")
    pyr_tgt = build_pyramid(tgt_zyx, fixed.spacing, levels=scales_tup, policy="deformable")

    # Convert level tensors back to ANTs (x, y, z) spatial ordering for VxmPairwise
    levels_xyz = [
        (
            ls.tensor.permute(0, 1, 4, 3, 2).contiguous(),
            lt.tensor.permute(0, 1, 4, 3, 2).contiguous(),
        )
        for ls, lt in zip(pyr_src, pyr_tgt)
    ]

    nb_tup = tuple(int(f) for f in nb_features)
    pad_mult = 2 ** len(nb_tup)
    model = vxm.nn.models.VxmPairwise(
        ndim=3,
        source_channels=1,
        target_channels=1,
        nb_features=nb_tup,
        integration_steps=int(int_steps),
        device=dev,
    ).to(dev)
    grad_fn = ne.nn.modules.SpatialGradient("l2")
    win_size = 2 * int(lncc_radius) + 1

    # Stage 0 (coarsest): train VxmPairwise weights with constant lr
    sx0, tx0 = levels_xyz[0]
    s0_pad, pads0 = _pad_to_multiple(sx0, pad_mult)
    t0_pad, _ = _pad_to_multiple(tx0, pad_mult)
    mask0 = (tx0 > 0.02).float()
    opt_net = torch.optim.Adam(model.parameters(), lr=float(lr))
    for _ in range(iters_tup[0]):
        opt_net.zero_grad()
        vel0 = _crop_from_pads(
            model(s0_pad, t0_pad, return_field_type="velocity"), pads0
        )
        disp0 = model.velocity_field_integrator(vel0)
        warped0 = model.spatial_transformer(sx0, disp0)
        loss0 = local_ncc_loss_nd(
            tx0,
            warped0,
            mask=mask0,
            window_size=win_size,
            squared=True,
            single_pass=False,
        ) + float(grad_weight) * grad_fn(vel0)
        loss0.backward()
        opt_net.step()

    with torch.no_grad():
        vel_curr = _crop_from_pads(
            model(s0_pad, t0_pad, return_field_type="velocity"), pads0
        )

    # Finer stages: upsample velocity via syntx.core.grid.resize_field and refine SVF
    n_stages = len(levels_xyz)
    for lvl_idx in range(1, n_stages):
        sx, tx = levels_xyz[lvl_idx]
        n_it = iters_tup[lvl_idx]
        scale_vec = torch.tensor(
            [float(sx.shape[2 + d]) / float(vel_curr.shape[2 + d]) for d in range(3)],
            dtype=vel_curr.dtype,
            device=dev,
        ).view(1, 3, 1, 1, 1)
        vel_lvl = (
            resize_field(vel_curr.movedim(1, -1), sx.shape[2:]).movedim(-1, 1)
            * scale_vec
        ).detach().requires_grad_(True)
        opt_svf = torch.optim.Adam([vel_lvl], lr=float(svf_lr))
        is_finest = lvl_idx == (n_stages - 1)
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_svf, T_max=max(1, n_it), eta_min=float(svf_lr) * 0.15
            )
            if is_finest and n_it > 0
            else None
        )
        mask_lvl = (tx > 0.02).float()
        # Scale gradient penalty smoothly toward finest level
        stage_gw = float(svf_grad_weight) if is_finest else 0.5 * float(svf_grad_weight)
        for _ in range(n_it):
            opt_svf.zero_grad()
            disp_lvl = model.velocity_field_integrator(vel_lvl)
            warped_lvl = model.spatial_transformer(sx, disp_lvl)
            loss_lvl = local_ncc_loss_nd(
                tx,
                warped_lvl,
                mask=mask_lvl,
                window_size=win_size,
                squared=True,
                single_pass=False,
            ) + stage_gw * grad_fn(vel_lvl)
            loss_lvl.backward()
            opt_svf.step()
            if scheduler is not None:
                scheduler.step()
        vel_curr = vel_lvl.detach()

    with torch.no_grad():
        if final_smooth_sigma > 0.0:
            vel_curr = _smooth_cf(vel_curr, float(final_smooth_sigma))
        disp_fwd = model.velocity_field_integrator(vel_curr)
        disp_inv = model.velocity_field_integrator(-vel_curr) if return_inverse else None

    _sync_device(dev)
    runtime_s = time.perf_counter() - t0

    save_dir = _resolve_output_dir(out_dir, "syntx_vxm_ms_")
    fwd_warp_path = os.path.join(save_dir, "1Warp.nii.gz")
    ants.image_write(voxel_disp_to_itk_lps(disp_fwd, fixed), fwd_warp_path)

    inv_warp_path: Optional[str] = None
    if disp_inv is not None:
        inv_warp_path = os.path.join(save_dir, "1InverseWarp.nii.gz")
        ants.image_write(voxel_disp_to_itk_lps(disp_inv, fixed), inv_warp_path)

    del model, src_zyx, tgt_zyx, pyr_src, pyr_tgt, levels_xyz, vel_curr, disp_fwd
    _release_device_cache(dev)

    provenance = {
        "method": "voxelmorph_multiscale",
        "device": dev,
        "voxelmorph_version": importlib.metadata.version("voxelmorph"),
        "scales": list(scales_tup),
        "reg_iterations": list(iters_tup),
        "lr": float(lr),
        "svf_lr": float(svf_lr),
        "grad_weight": float(grad_weight),
        "svf_grad_weight": float(svf_grad_weight),
        "lncc_radius": int(lncc_radius),
        "nb_features": list(nb_tup),
        "int_steps": int(int_steps),
        "final_smooth_sigma": float(final_smooth_sigma),
        "return_inverse": bool(return_inverse),
        "runtime_seconds": float(runtime_s),
    }
    return _build_return_dict(
        fixed, moving, fwd_warp_path, inv_warp_path, initial_transform, provenance
    )


def run_fireants_greedy(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    initial_transform: Optional[str] = None,
    scales: Sequence[int] = (4, 2, 1),
    reg_iterations: Sequence[int] = (100, 100, 50),
    optimizer: str = "regadam",
    optimizer_lr: float = 0.5,
    smooth_grad_sigma: float = 1.0,
    smooth_warp_sigma: float = 0.25,
    smooth_step_sigma: float = 0.5,
    cc_kernel_size: int = 5,
    return_inverse: bool = False,
    device: Optional[str] = None,
    seed: int = 42,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run repaired ``FireANTs`` ``GreedyRegistration`` with decoupled ITK LPS export."""
    try:
        from fireants.io import BatchedImages as FABatchedImages
        from fireants.io import Image as FAImage
        from fireants.registration.greedy import GreedyRegistration as FireANTsGreedy
    except ImportError as exc:
        raise ImportError(
            "fireants (>=1.5.1 editable remediation) is required for FireANTs external baselines. "
            "See docs/EXTERNAL_BENCHMARK_BASELINES.md."
        ) from exc

    dev = resolve_device(device, task="reg")
    _release_device_cache(dev)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    init_affine_t: Optional[torch.Tensor] = None
    if initial_transform is not None:
        m_np = ants_mat_to_4x4(initial_transform)
        init_affine_t = torch.from_numpy(m_np).float().unsqueeze(0).to(dev)

    save_dir = _resolve_output_dir(out_dir, "syntx_fireants_")
    f_tmp = os.path.join(save_dir, "_tmp_fixed.nii.gz")
    m_tmp = os.path.join(save_dir, "_tmp_moving.nii.gz")
    ants.image_write(fixed, f_tmp)
    ants.image_write(moving, m_tmp)

    t0 = time.perf_counter()
    try:
        b_f = FABatchedImages([FAImage.load_file(f_tmp, device=dev)])
        b_m = FABatchedImages([FAImage.load_file(m_tmp, device=dev)])
    finally:
        for p_tmp in (f_tmp, m_tmp):
            if os.path.exists(p_tmp):
                try:
                    os.remove(p_tmp)
                except OSError:
                    pass

    fa_reg = FireANTsGreedy(
        scales=[int(s) for s in scales],
        iterations=[int(k) for k in reg_iterations],
        fixed_images=b_f,
        moving_images=b_m,
        loss_type="cc",
        cc_kernel_size=int(cc_kernel_size),
        deformation_type="compositive",
        optimizer=str(optimizer),
        optimizer_lr=float(optimizer_lr),
        smooth_grad_sigma=float(smooth_grad_sigma),
        smooth_warp_sigma=float(smooth_warp_sigma),
        smooth_step_sigma=float(smooth_step_sigma),
        init_affine=init_affine_t,
    )
    fa_reg.optimize()
    _sync_device(dev)
    opt_s = time.perf_counter() - t0

    t_exp0 = time.perf_counter()
    fwd_warp_path = os.path.join(save_dir, "1Warp.nii.gz")
    fa_reg.save_as_ants_transforms(
        fwd_warp_path, save_inverse=False, include_affine=False
    )
    inv_warp_path: Optional[str] = None
    if return_inverse:
        inv_warp_path = os.path.join(save_dir, "1InverseWarp.nii.gz")
        fa_reg.save_as_ants_transforms(
            inv_warp_path, save_inverse=True, include_affine=False
        )
    _sync_device(dev)
    exp_s = time.perf_counter() - t_exp0

    del fa_reg, b_f, b_m, init_affine_t
    _release_device_cache(dev)

    provenance = {
        "method": "fireants_greedy",
        "device": dev,
        "fireants_version": importlib.metadata.version("fireants"),
        "scales": [int(s) for s in scales],
        "reg_iterations": [int(k) for k in reg_iterations],
        "optimizer": str(optimizer),
        "optimizer_lr": float(optimizer_lr),
        "smooth_grad_sigma": float(smooth_grad_sigma),
        "smooth_warp_sigma": float(smooth_warp_sigma),
        "smooth_step_sigma": float(smooth_step_sigma),
        "cc_kernel_size": int(cc_kernel_size),
        "include_affine": False,
        "return_inverse": bool(return_inverse),
        "optimize_seconds": float(opt_s),
        "export_seconds": float(exp_s),
        "runtime_seconds": float(opt_s + exp_s),
    }
    return _build_return_dict(
        fixed, moving, fwd_warp_path, inv_warp_path, initial_transform, provenance
    )
