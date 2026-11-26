"""Hybrid Neural-Variational Diffeomorphic Registration: ``hybrid_greedy`` and ``hybrid_syn``.

Pairs a GPU-cached 3-D ``VxmPairwise`` stationary-velocity-field (SVF) neural initializer
(fetched via ``antsxdata`` with lossless discrete ``PIR`` orientation permutation on GPU and
zero intermediate disk I/O) with fine-level ``syntx`` Navier continuum refinement and
``enforce_digital_diffeomorphism_nd`` (guaranteeing ``0.0000%`` folding and exact single-interpolation
ITK LPS ``TransformChain`` composition).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import ants
import numpy as np
import torch
import torch.nn.functional as F

from .benchmark.evaluate import normalize_intensity
from .benchmark.external import (
    _crop_from_pads,
    _import_voxelmorph,
    _load_voxelmorph_weights,
    _pad_to_multiple,
    _smooth_cf,
    _voxel_disp_nat_to_norm_zyx,
    apply_axis_permutation,
    canonical_orientation_transform,
    invert_displacement_permutation,
)
from .cohort import FixedTarget
from .core.affine import parse_ants_affine
from .core.utils import require_finite_images
from .device import resolve_device
from .greedy import _Conv1x1AffineGrid, _build_torch_affine_matrix, greedy_registration
from .robust_affine import robust_affine
from .syn import syn
from .transforms import apply_transforms as _apply_transforms

_VXM_MODEL_CACHE: Dict[Tuple[str, str, int], Tuple[Any, Dict[str, Any]]] = {}


def get_cached_voxelmorph_3d(
    asset_ref: str = "syntx_features/voxelmorph_pairwise_3d_brain",
    device: Union[str, torch.device] = "mps",
    int_steps: int = 7,
) -> Tuple[Any, Dict[str, Any]]:
    """Return a GPU-resident ``.eval()`` 3-D ``VxmPairwise`` model and its ``antsxdata`` provenance."""
    dev_str = str(torch.device(device))
    key = (str(asset_ref), dev_str, int(int_steps))
    cached = _VXM_MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    vxm, _ = _import_voxelmorph()
    _, asset_prov, ckpt = _load_voxelmorph_weights(asset_ref)
    nb_features = (16, 16, 16, 16, 16)
    model = vxm.nn.models.VxmPairwise(
        ndim=3,
        source_channels=1,
        target_channels=1,
        nb_features=nb_features,
        integration_steps=int(int_steps),
        device=dev_str,
    ).to(dev_str)
    model.load_state_dict(ckpt)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _VXM_MODEL_CACHE[key] = (model, asset_prov)
    return model, asset_prov


def _predict_cached_vxm_initial_warps(
    fixed_target_or_img: Union[ants.ANTsImage, FixedTarget],
    moving: ants.ANTsImage,
    *,
    initial_transform: Any,
    need_inverse: bool,
    dev_str: str,
    asset_ref: str,
    canonical_orientation: Optional[str],
    int_steps: int,
    vel_smooth_sigma: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    """Predict normalized ``(1, nz, ny, nx, 3)`` ``(u0_norm, u0_inv_norm)`` directly on GPU."""
    dev_t = torch.device(dev_str)
    is_ft = bool(getattr(fixed_target_or_img, "_is_syntx_fixed_target", False))
    fi_norm = fixed_target_or_img.norm_image if is_ft else normalize_intensity(fixed_target_or_img)
    fi_zyx = (
        fixed_target_or_img.tensor_zyx.to(device=dev_t, dtype=torch.float32)
        if is_ft
        else torch.from_numpy(np.ascontiguousarray(fi_norm.numpy(), dtype=np.float32))
        .to(device=dev_t, dtype=torch.float32)
        .permute(2, 1, 0)
        .contiguous()
        .unsqueeze(0)
        .unsqueeze(0)
    )
    mi_norm = normalize_intensity(moving)
    mi_zyx = (
        torch.from_numpy(np.ascontiguousarray(mi_norm.numpy(), dtype=np.float32))
        .to(device=dev_t, dtype=torch.float32)
        .permute(2, 1, 0)
        .contiguous()
        .unsqueeze(0)
        .unsqueeze(0)
    )

    t0 = time.perf_counter()
    model, asset_prov = get_cached_voxelmorph_3d(asset_ref=asset_ref, device=dev_str, int_steps=int_steps)

    with torch.no_grad():
        if initial_transform is None or (
            isinstance(initial_transform, str) and initial_transform.lower() == "identity"
        ):
            M_phys = np.eye(3, dtype=np.float64)
            t_phys = np.zeros(3, dtype=np.float64)
        else:
            M_phys, t_phys = parse_ants_affine(initial_transform, 3, allow_nonlinear=True)

        if M_phys is None:
            mi_aff = _apply_transforms(
                fixed=fi_norm,
                moving=mi_norm,
                transformlist=initial_transform if isinstance(initial_transform, (list, tuple)) else [initial_transform],
                interpolator="linear",
                device=dev_str,
                return_tensor=True,
            )
            mi_aff_zyx = mi_aff.to(device=dev_t, dtype=torch.float32)
        else:
            same_id = (
                np.allclose(M_phys, np.eye(3), atol=1e-6)
                and np.allclose(t_phys, 0.0, atol=1e-6)
                and tuple(fi_norm.shape) == tuple(mi_norm.shape)
                and np.allclose(fi_norm.spacing, mi_norm.spacing, atol=1e-4)
                and np.allclose(fi_norm.origin, mi_norm.origin, atol=1e-4)
                and np.allclose(fi_norm.direction, mi_norm.direction, atol=1e-4)
            )
            if same_id:
                mi_aff_zyx = mi_zyx
            else:
                theta, _, _ = _build_torch_affine_matrix(fi_norm, mi_norm, M_phys, t_phys, dev_t)
                id_grid = F.affine_grid(
                    torch.eye(3, 4, device=dev_t, dtype=torch.float32).unsqueeze(0),
                    [1, 1, *fi_zyx.shape[2:]],
                    align_corners=True,
                )
                sample_grid = _Conv1x1AffineGrid.apply(
                    id_grid, theta[0, :, :3].contiguous(), theta[0, :, 3].contiguous()
                )
                mi_aff_zyx = F.grid_sample(
                    mi_zyx, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True
                )
                del id_grid, sample_grid
        del mi_zyx

        src_nat = mi_aff_zyx.permute(0, 1, 4, 3, 2).contiguous()
        tgt_nat = fi_zyx.permute(0, 1, 4, 3, 2).contiguous()
        del mi_aff_zyx

        if canonical_orientation is not None:
            perm, flips = canonical_orientation_transform(
                np.asarray(fi_norm.direction, dtype=np.float64), canonical_orientation
            )
            src_can = apply_axis_permutation(src_nat, perm, flips)
            tgt_can = apply_axis_permutation(tgt_nat, perm, flips)
        else:
            perm = tuple(range(3))
            flips = (1, 1, 1)
            src_can, tgt_can = src_nat, tgt_nat
        del src_nat, tgt_nat

        src_pad, pads = _pad_to_multiple(src_can, 32)
        tgt_pad, _ = _pad_to_multiple(tgt_can, 32)
        del src_can, tgt_can

        vel_can = _crop_from_pads(model(src_pad, tgt_pad, return_field_type="velocity"), pads)
        del src_pad, tgt_pad
        if float(vel_smooth_sigma) > 0.0:
            vel_can = _smooth_cf(vel_can, float(vel_smooth_sigma))

        disp_fwd_can = model.velocity_field_integrator(vel_can)
        disp_fwd_nat = invert_displacement_permutation(disp_fwd_can, perm, flips)
        del disp_fwd_can
        u0_norm = _voxel_disp_nat_to_norm_zyx(disp_fwd_nat)
        del disp_fwd_nat

        u0_inv_norm = None
        if need_inverse:
            disp_inv_can = model.velocity_field_integrator(-vel_can)
            disp_inv_nat = invert_displacement_permutation(disp_inv_can, perm, flips)
            del disp_inv_can
            u0_inv_norm = _voxel_disp_nat_to_norm_zyx(disp_inv_nat)
            del disp_inv_nat
        del vel_can

    vxm_sec = time.perf_counter() - t0
    meta = {
        "asset_ref": asset_prov.get("ref", asset_ref),
        "asset_sha256": asset_prov.get("sha256"),
        "canonical_orientation": canonical_orientation,
        "int_steps": int(int_steps),
        "vel_smooth_sigma": float(vel_smooth_sigma),
        "vxm_seconds": round(float(vxm_sec), 4),
    }
    return u0_norm, u0_inv_norm, meta


def hybrid_greedy(
    fixed: Union[ants.ANTsImage, FixedTarget],
    moving: ants.ANTsImage,
    *,
    initial_transform: Optional[Any] = None,
    scales: Sequence[int] = (2, 1),
    reg_iterations: Sequence[int] = (15, 8),
    learning_rate: float = 1.2,
    flow_sigma: float = 1.62,
    regularizer: str = "navier",
    sobolev_alpha: Optional[float] = 2.625,
    return_inverse: bool = False,
    return_warped: bool = True,
    write_transforms: bool = True,
    sync_free: bool = True,
    canonical_orientation: Optional[str] = "PIR",
    int_steps: int = 7,
    vel_smooth_sigma: float = 0.40,
    asset_ref: str = "syntx_features/voxelmorph_pairwise_3d_brain",
    affine_mode: str = "auto",
    device: Union[str, torch.device, None] = "auto",
    seed: int = 42,
    verbose: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """3-D Hybrid Neural-Variational Greedy registration (cached ``VxmPairwise`` SVF + ``[15, 8]`` Navier polish)."""
    fixed_img = fixed.image if bool(getattr(fixed, "_is_syntx_fixed_target", False)) else fixed
    require_finite_images(fixed=fixed_img, moving=moving)
    if int(fixed_img.dimension) != 3 or int(moving.dimension) != 3:
        raise ValueError("syntx.hybrid_greedy requires 3-D images.")

    dev_req = None if (device is None or str(device).lower() == "auto") else str(device)
    dev_str = resolve_device(dev_req, task="reg")

    outprefix = kwargs.get("outprefix", None)
    do_write = bool(write_transforms or outprefix is not None)

    t0_aff = time.perf_counter()
    if initial_transform is None:
        aff_res = robust_affine(
            fixed_img,
            moving,
            mode=affine_mode,
            device=dev_str,
            seed=seed,
            return_warped=False,
            write_transforms=do_write,
            sync_free=sync_free,
            verbose=verbose,
        )
        aff_tx: Any = aff_res["fwdtransforms"] if do_write else aff_res["fwd_chain"]
    else:
        aff_tx = initial_transform
    t_aff = time.perf_counter() - t0_aff

    u0_norm, u0_inv_norm, vxm_meta = _predict_cached_vxm_initial_warps(
        fixed,
        moving,
        initial_transform=aff_tx,
        need_inverse=return_inverse,
        dev_str=dev_str,
        asset_ref=asset_ref,
        canonical_orientation=canonical_orientation,
        int_steps=int_steps,
        vel_smooth_sigma=vel_smooth_sigma,
    )

    res = greedy_registration(
        fixed=fixed,
        moving=moving,
        initial_transform=aff_tx,
        initial_warp=u0_norm,
        initial_inverse_warp=u0_inv_norm,
        scales=list(scales),
        reg_iterations=list(reg_iterations),
        learning_rate=learning_rate,
        flow_sigma=flow_sigma,
        regularizer=regularizer,
        sobolev_alpha=sobolev_alpha,
        return_inverse=return_inverse,
        return_warped=return_warped,
        write_transforms=write_transforms,
        sync_free=sync_free,
        device=dev_str,
        verbose=verbose,
        **kwargs,
    )
    prov = dict(res.get("provenance", {}))
    prov.update(
        {
            "algorithm": "syntx.hybrid_greedy",
            "runtime_affine_sec": round(float(t_aff), 3),
            **vxm_meta,
        }
    )
    res["provenance"] = prov
    return res


def hybrid_syn(
    fixed: Union[ants.ANTsImage, FixedTarget],
    moving: ants.ANTsImage,
    *,
    initial_transform: Optional[Any] = None,
    type_of_transform: str = "SyN",
    levels: Sequence[int] = (2, 1),
    reg_iterations: Sequence[int] = (15, 8),
    grad_step: float = 0.5,
    flow_sigma: float = 2.4,
    regularizer: str = "navier",
    sobolev_alpha: Optional[float] = 1.5,
    return_warped: bool = True,
    write_transforms: bool = True,
    export_midpoint: bool = False,
    sync_free: bool = True,
    canonical_orientation: Optional[str] = "PIR",
    int_steps: int = 7,
    vel_smooth_sigma: float = 0.40,
    asset_ref: str = "syntx_features/voxelmorph_pairwise_3d_brain",
    affine_mode: str = "auto",
    device: Union[str, torch.device, None] = "auto",
    seed: int = 42,
    verbose: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """3-D Hybrid Neural-Variational SyN registration (cached ``VxmPairwise`` SVF + ``[15, 8]`` symmetric Navier polish)."""
    fixed_img = fixed.image if bool(getattr(fixed, "_is_syntx_fixed_target", False)) else fixed
    require_finite_images(fixed=fixed_img, moving=moving)
    if int(fixed_img.dimension) != 3 or int(moving.dimension) != 3:
        raise ValueError("syntx.hybrid_syn requires 3-D images.")

    dev_req = None if (device is None or str(device).lower() == "auto") else str(device)
    dev_str = resolve_device(dev_req, task="syn")
    outprefix = kwargs.get("outprefix", None)
    do_write = bool(write_transforms or outprefix is not None)
    is_syn_only = str(type_of_transform).upper() == "SYNONLY"

    t0_aff = time.perf_counter()
    if initial_transform is None:
        if is_syn_only:
            aff_tx: Any = "identity"
        else:
            aff_res = robust_affine(
                fixed_img,
                moving,
                mode=affine_mode,
                device=dev_str,
                seed=seed,
                return_warped=False,
                write_transforms=do_write,
                sync_free=sync_free,
                verbose=verbose,
            )
            aff_tx = aff_res["fwdtransforms"] if do_write else aff_res["fwd_chain"]
    else:
        aff_tx = initial_transform
    t_aff = time.perf_counter() - t0_aff

    u0_norm, u0_inv_norm, vxm_meta = _predict_cached_vxm_initial_warps(
        fixed,
        moving,
        initial_transform=aff_tx,
        need_inverse=True,
        dev_str=dev_str,
        asset_ref=asset_ref,
        canonical_orientation=canonical_orientation,
        int_steps=int_steps,
        vel_smooth_sigma=vel_smooth_sigma,
    )

    res = syn(
        fixed=fixed_img,
        moving=moving,
        type_of_transform=type_of_transform,
        initial_transform=aff_tx,
        initial_warp=u0_norm,
        initial_inverse_warp=u0_inv_norm,
        levels=list(levels),
        reg_iterations=list(reg_iterations),
        grad_step=grad_step,
        flow_sigma=flow_sigma,
        regularizer=regularizer,
        sobolev_alpha=sobolev_alpha,
        return_warped=return_warped,
        write_transforms=write_transforms,
        export_midpoint=export_midpoint,
        sync_free=sync_free,
        device=dev_str,
        seed=seed,
        verbose=verbose,
        **kwargs,
    )
    prov = dict(res.get("provenance", {}))
    prov.update(
        {
            "algorithm": "syntx.hybrid_syn",
            "runtime_affine_sec": round(float(t_aff), 3),
            **vxm_meta,
        }
    )
    res["provenance"] = prov
    return res
