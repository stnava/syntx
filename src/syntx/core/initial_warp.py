"""Centralized initial-warp resolution, sampling-grid construction, and endpoint composition.

Provides the single shared contract used by ``syntx.greedy``, ``syntx.syn`` (PyTorch & JAX),
``syntx.tvf`` (PyTorch & JAX), ``syntx.syngs`` (PyTorch & JAX), and ``syntx.auto_reg`` to
accept:

1. ``initial_warp``: normalized ``[-1, 1]`` displacement tensor/array ``(1, *spatial, dim)``
   in tensor ``(z, y, x)`` spatial order and ``(x, y, z)`` component order, an ITK LPS
   vector ``ANTsImage`` (``has_components=True``), or a ``.nii`` / ``.nii.gz`` warp path.
2. ``initial_inverse_warp``: optional inverse of ``initial_warp`` in the same formats; when
   omitted and an inverse field is required, automatically computed via
   ``update_inverse_field_nd_anderson``.
3. ``initial_transform=[warp_path, affine_path, ...]``: decomposes a leading displacement
   field followed by linear transform(s) into ``(u_0, M_phys, t_phys)`` representing
   ``T_init(x) = M_phys @ (x + u_0(x)) + t_phys``.

Composition law at export preserves a single non-linear ``1Warp.nii.gz`` and
``1InverseWarp.nii.gz`` alongside ``0GenericAffine.mat``:

- Forward composed displacement on ``fixed``: ``u_total(x) = w_fwd(x) + u_0(x + w_fwd(x))``
- Inverse composed displacement on ``fixed``: ``u_total_inv(z) = u_0_inv(z) + w_inv(z + u_0_inv(z))``
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import ants
import numpy as np
import torch

from ..spatial import (
    disp_itk_to_tensor,
    get_physical_grid_torch,
    normalized_to_physical_disp,
    physical_to_normalized_disp,
    physical_to_normalized_torch_cached,
    reverse_metadata,
)
from .affine import parse_ants_affine
from .grid import resize_field, sample_field_cf
from .inverse import update_inverse_field_nd_anderson


@dataclass
class ResolvedInitialWarp:
    """Resolved linear affine + optional dense initial warp on ``fixed``'s grid."""

    aff_tx: Optional[List[Any]]
    M_phys: np.ndarray
    t_phys: np.ndarray
    u0_norm: Optional[torch.Tensor]
    u0_phys: Optional[torch.Tensor]
    u0_inv_phys: Optional[torch.Tensor]
    initial_grid_zyx: Optional[torch.Tensor]
    has_initial_warp: bool
    initial_warp_kind: str
    has_linear_affine: bool


def is_displacement_field_item(item: Any, dim: int) -> bool:
    """Return True if ``item`` is a dense ITK displacement field (ANTsImage, TransformElement, or NIfTI path)."""
    from ..transforms import TransformElement

    if isinstance(item, TransformElement):
        return item.kind in ("displacement", "inverse_displacement")
    if hasattr(item, "numpy") and hasattr(item, "spacing"):
        return int(getattr(item, "components", 1)) == dim
    if isinstance(item, (str, os.PathLike)):
        p = str(item)
        if not os.path.exists(p):
            return False
        if p.lower().endswith((".nii", ".nii.gz")):
            return True
        try:
            tx = ants.read_transform(p)
            return "Displacement" in str(getattr(tx, "transform_type", ""))
        except Exception:
            return False
    return False


def classify_initial_warp_kind(
    initial_warp: Any = None,
    initial_transform: Any = None,
    dim: int = 3,
) -> Tuple[bool, str]:
    """Return ``(has_initial_warp, initial_warp_kind)`` in
    ``{"none", "tensor", "ants_image", "file", "transform_list"}``."""
    if initial_warp is not None:
        if isinstance(initial_warp, (str, os.PathLike)):
            return True, "file"
        if hasattr(initial_warp, "numpy") and hasattr(initial_warp, "spacing"):
            return True, "ants_image"
        return True, "tensor"
    if initial_transform is not None and not isinstance(initial_transform, bool):
        if isinstance(initial_transform, str) and initial_transform.lower() == "identity":
            return False, "none"
        tx_list = list(initial_transform) if isinstance(initial_transform, (list, tuple)) else [initial_transform]
        if tx_list and is_displacement_field_item(tx_list[0], dim):
            return True, "transform_list"
    return False, "none"


def _warp_to_norm_and_phys(
    warp_source: Any,
    fixed: ants.ANTsImage,
    fixed_zyx: Tuple[int, ...],
    torch_device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a normalized tensor/array, ANTsImage, or file path into
    ``(u0_norm, u0_phys)`` of shape ``(1, *fixed_zyx, dim)`` on ``torch_device``."""
    dim = fixed.dimension
    mode = "trilinear" if dim == 3 else "bilinear"

    is_array_like = isinstance(warp_source, (torch.Tensor, np.ndarray)) or (
        hasattr(warp_source, "__array__") and not hasattr(warp_source, "spacing")
    )
    if is_array_like:
        iw = (
            warp_source
            if isinstance(warp_source, torch.Tensor)
            else torch.from_numpy(np.asarray(warp_source, dtype=np.float32))
        )
        iw = iw.to(device=torch_device, dtype=torch.float32)
        if iw.ndim == dim + 1:
            iw = iw.unsqueeze(0)
        if iw.ndim != dim + 2 or iw.shape[-1] != dim:
            raise ValueError(
                f"initial_warp tensor must have shape (1, *spatial, {dim}) or (*spatial, {dim}), "
                f"got {tuple(iw.shape)}"
            )
        if tuple(iw.shape[1:-1]) != tuple(fixed_zyx):
            iw = resize_field(iw, list(fixed_zyx), mode=mode)
        u0_norm = iw.contiguous()
        u0_phys = normalized_to_physical_disp(
            u0_norm,
            fixed_zyx,
            fixed.spacing,
            fixed.direction,
            device=torch_device,
            dtype=torch.float32,
        )
        return u0_norm, u0_phys

    from ..transforms import TransformElement

    if isinstance(warp_source, TransformElement):
        if warp_source.field is not None:
            same_grid_el = (
                tuple(warp_source.field.spatial_shape) == tuple(fixed.shape[:dim])
                and np.allclose(np.asarray(warp_source.field.spacing[:dim]), np.asarray(fixed.spacing[:dim]), atol=1e-4)
                and np.allclose(np.asarray(warp_source.field.origin[:dim]), np.asarray(fixed.origin[:dim]), atol=1e-4)
                and np.allclose(
                    np.asarray(warp_source.field.direction)[:dim, :dim],
                    np.asarray(fixed.direction)[:dim, :dim],
                    atol=1e-4,
                )
            )
            if same_grid_el:
                u0_phys = warp_source.disp_tensor_zyx(device=torch_device)
                u0_norm = physical_to_normalized_disp(
                    u0_phys,
                    fixed_zyx,
                    fixed.spacing,
                    fixed.direction,
                    device=torch_device,
                    dtype=torch.float32,
                )
                return u0_norm, u0_phys
        warp_source = warp_source.to_ants_image()

    if isinstance(warp_source, (str, os.PathLike)):
        warp_path = str(warp_source)
        if not os.path.exists(warp_path):
            raise FileNotFoundError(f"initial_warp file not found: {warp_path!r}")
        warp_img = ants.image_read(warp_path)
    elif hasattr(warp_source, "numpy") and hasattr(warp_source, "spacing"):
        warp_img = warp_source
    else:
        raise TypeError(
            f"initial_warp must be a tensor/array, ANTsImage, or warp file path; "
            f"got {type(warp_source).__name__}"
        )

    if int(getattr(warp_img, "components", 1)) != dim:
        raise ValueError(
            f"initial_warp image must have {dim} vector components, "
            f"got {getattr(warp_img, 'components', 1)}"
        )

    same_grid = (
        tuple(warp_img.shape[:dim]) == tuple(fixed.shape[:dim])
        and np.allclose(np.asarray(warp_img.spacing[:dim]), np.asarray(fixed.spacing[:dim]), atol=1e-4)
        and np.allclose(np.asarray(warp_img.origin[:dim]), np.asarray(fixed.origin[:dim]), atol=1e-4)
        and np.allclose(
            np.asarray(warp_img.direction)[:dim, :dim],
            np.asarray(fixed.direction)[:dim, :dim],
            atol=1e-4,
        )
    )
    if not same_grid:
        from ..transforms import apply_transforms as _apply_transforms

        warp_img = _apply_transforms(
            fixed=fixed,
            moving=warp_img,
            transformlist=["identity"],
            interpolator="linear",
            imagetype=1,
            device=str(torch_device),
        )

    u0_phys = disp_itk_to_tensor(warp_img, device=torch_device).to(torch.float32)
    u0_norm = physical_to_normalized_disp(
        u0_phys,
        fixed_zyx,
        fixed.spacing,
        fixed.direction,
        device=torch_device,
        dtype=torch.float32,
    )
    return u0_norm, u0_phys


def compute_unclamped_initial_grid_zyx(
    u0_norm: torch.Tensor,
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    M_phys: np.ndarray,
    t_phys: np.ndarray,
    torch_device: torch.device,
) -> torch.Tensor:
    """Compute unclamped normalized moving-image coordinates ``(1, *fixed_zyx, dim)``
    for ``T_init(x) = M_phys @ (x + u_0(x)) + t_phys`` directly on ``torch_device``."""
    dim = fixed.dimension
    d_f = np.asarray(fixed.direction, dtype=np.float64).reshape(dim, dim)
    s_f = np.asarray(fixed.spacing, dtype=np.float64)
    o_f = np.asarray(fixed.origin, dtype=np.float64)
    n_f = np.asarray(fixed.shape[:dim], dtype=np.float64)
    p_f = np.eye(dim + 1, dtype=np.float64)
    p_f[:dim, :dim] = d_f @ np.diag(s_f) @ np.diag(np.maximum(n_f - 1.0, 1.0) / 2.0)
    p_f[:dim, dim] = d_f @ (s_f * np.maximum(n_f - 1.0, 1.0) / 2.0) + o_f

    d_m = np.asarray(moving.direction, dtype=np.float64).reshape(dim, dim)
    s_m = np.asarray(moving.spacing, dtype=np.float64)
    o_m = np.asarray(moving.origin, dtype=np.float64)
    n_m = np.asarray(moving.shape[:dim], dtype=np.float64)
    p_m = np.eye(dim + 1, dtype=np.float64)
    p_m[:dim, :dim] = d_m @ np.diag(s_m) @ np.diag(np.maximum(n_m - 1.0, 1.0) / 2.0)
    p_m[:dim, dim] = d_m @ (s_m * np.maximum(n_m - 1.0, 1.0) / 2.0) + o_m

    t_mat = np.eye(dim + 1, dtype=np.float64)
    t_mat[:dim, :dim] = M_phys
    t_mat[:dim, dim] = t_phys
    t_norm = np.linalg.inv(p_m) @ t_mat @ p_f
    m_th = torch.from_numpy(t_norm[:dim, :dim]).to(device=torch_device, dtype=torch.float32)
    b_th = torch.from_numpy(t_norm[:dim, dim]).to(device=torch_device, dtype=torch.float32)

    fixed_zyx = tuple(int(x) for x in u0_norm.shape[1:-1])
    eye_theta = torch.eye(dim, dim + 1, device=torch_device, dtype=torch.float32).unsqueeze(0)
    id_grid = torch.nn.functional.affine_grid(
        eye_theta, [1, 1, *fixed_zyx], align_corners=True
    )
    coords = id_grid + u0_norm.to(device=torch_device, dtype=torch.float32)
    return torch.addmm(b_th.view(1, dim), coords.reshape(-1, dim), m_th.t()).view_as(coords)


def resolve_initial_transform_and_warp(
    fixed: ants.ANTsImage,
    moving: ants.ANTsImage,
    initial_transform: Any = None,
    initial_warp: Any = None,
    initial_inverse_warp: Any = None,
    *,
    need_inverse: bool = False,
    device: Optional[Union[str, torch.device]] = None,
) -> ResolvedInitialWarp:
    """Decompose ``initial_transform`` / ``initial_warp`` / ``initial_inverse_warp``
    into linear affine ``(M_phys, t_phys)`` and dense initial displacement fields on
    ``fixed``'s grid.

    Caller must resolve ``initial_transform is None`` (automatic ``robust_affine``)
    before calling if automatic affine initialization is desired; ``None`` here is
    treated as identity linear alignment unless ``initial_transform`` was provided.
    """
    dim = fixed.dimension
    fixed_zyx = tuple(int(x) for x in fixed.shape[:dim])[::-1]
    if device is None:
        from ..device import resolve_device

        torch_device = torch.device(resolve_device(None, task="reg"))
    else:
        torch_device = device if isinstance(device, torch.device) else torch.device(str(device))

    aff_tx: Optional[List[Any]] = None
    lead_warp_source = initial_warp
    warp_kind = "none"

    if initial_warp is not None:
        _, warp_kind = classify_initial_warp_kind(initial_warp=initial_warp, dim=dim)

    if initial_transform is False or (
        isinstance(initial_transform, str) and initial_transform.lower() == "identity"
    ):
        aff_tx = None
    elif initial_transform is not None:
        from ..transforms import TransformChain

        if isinstance(initial_transform, TransformChain):
            tx_list = list(initial_transform.elements)
        elif isinstance(initial_transform, (list, tuple)):
            tx_list = list(initial_transform)
        else:
            tx_list = [initial_transform]
        M_check, _ = parse_ants_affine(tx_list, dim, allow_nonlinear=True)
        if M_check is not None:
            aff_tx = tx_list
        elif tx_list and is_displacement_field_item(tx_list[0], dim):
            if initial_warp is not None:
                raise ValueError(
                    "Cannot specify both initial_warp and a non-linear leading transform "
                    "in initial_transform."
                )
            lead_warp_source = tx_list[0]
            warp_kind = "transform_list"
            tail_tx = tx_list[1:]
            if tail_tx:
                M_tail, _ = parse_ants_affine(tail_tx, dim, allow_nonlinear=True)
                if M_tail is None:
                    raise NotImplementedError(
                        "initial_transform contains multiple non-linear transforms; only a "
                        "leading displacement field followed by linear transform(s) is "
                        "supported as a dense initial grid."
                    )
                aff_tx = tail_tx
            else:
                aff_tx = None
        else:
            raise NotImplementedError(
                "initial_transform contains a non-linear transform that is not a leading "
                "displacement field followed by linear transform(s); pass [warp_file, "
                "affine_file], initial_warp=..., a linear transform, False, or 'identity' "
                "instead (dense initial grid)."
            )

    if aff_tx is not None:
        M_phys, t_phys = parse_ants_affine(aff_tx, dim, allow_nonlinear=False, return_float64=True)
        has_linear_affine = True
    else:
        M_phys = np.eye(dim, dtype=np.float64)
        t_phys = np.zeros(dim, dtype=np.float64)
        has_linear_affine = False

    if lead_warp_source is None:
        return ResolvedInitialWarp(
            aff_tx=aff_tx,
            M_phys=M_phys,
            t_phys=t_phys,
            u0_norm=None,
            u0_phys=None,
            u0_inv_phys=None,
            initial_grid_zyx=None,
            has_initial_warp=False,
            initial_warp_kind="none",
            has_linear_affine=has_linear_affine,
        )

    u0_norm, u0_phys = _warp_to_norm_and_phys(lead_warp_source, fixed, fixed_zyx, torch_device)
    initial_grid_zyx = compute_unclamped_initial_grid_zyx(
        u0_norm, fixed, moving, M_phys, t_phys, torch_device
    )

    u0_inv_phys: Optional[torch.Tensor] = None
    if initial_inverse_warp is not None:
        _, u0_inv_phys = _warp_to_norm_and_phys(
            initial_inverse_warp, fixed, fixed_zyx, torch_device
        )
    elif need_inverse:
        u0_inv_phys = update_inverse_field_nd_anderson(
            W_disp=u0_phys,
            W_inv_disp=None,
            steps=15,
            m=5,
            spacing=fixed.spacing,
            origin=fixed.origin,
            direction=fixed.direction,
        )

    return ResolvedInitialWarp(
        aff_tx=aff_tx,
        M_phys=M_phys,
        t_phys=t_phys,
        u0_norm=u0_norm,
        u0_phys=u0_phys,
        u0_inv_phys=u0_inv_phys,
        initial_grid_zyx=initial_grid_zyx,
        has_initial_warp=True,
        initial_warp_kind=warp_kind,
        has_linear_affine=has_linear_affine,
    )


def compose_initial_and_residual_warps(
    w_fwd: Optional[Union[torch.Tensor, np.ndarray]],
    w_inv: Optional[Union[torch.Tensor, np.ndarray]],
    u0_phys: torch.Tensor,
    u0_inv_phys: Optional[torch.Tensor],
    spacing: Sequence[float],
    origin: Sequence[float],
    direction: Any,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compose residual solver displacement fields ``(w_fwd, w_inv)`` with initial
    dense displacements ``(u0_phys, u0_inv_phys)`` in physical mm on ``fixed``'s grid:

    - ``u_total_fwd(x) = w_fwd(x) + u0(x + w_fwd(x))``
    - ``u_total_inv(z) = u0_inv(z) + w_inv(z + u0_inv(z))``
    """
    dev = u0_phys.device if device is None else (
        device if isinstance(device, torch.device) else torch.device(str(device))
    )
    u0_d = u0_phys.to(device=dev, dtype=torch.float32)
    spatial_zyx = tuple(int(s) for s in u0_d.shape[1:-1])
    dim = int(u0_d.shape[-1])

    sp_rev, org_rev, dir_rev = reverse_metadata(spacing, origin, direction)
    shape_t = torch.tensor(list(spatial_zyx), device=dev, dtype=torch.float32)
    sp_t = torch.tensor(sp_rev, device=dev, dtype=torch.float32)
    org_t = torch.tensor(org_rev, device=dev, dtype=torch.float32)
    dir_t = torch.tensor(dir_rev, device=dev, dtype=torch.float32)

    X_phys = get_physical_grid_torch(
        spatial_zyx, spacing, origin, direction, device=dev, dtype=torch.float32
    )

    u_tot_fwd: Optional[torch.Tensor] = None
    if w_fwd is not None:
        wf = (
            w_fwd.to(device=dev, dtype=torch.float32)
            if isinstance(w_fwd, torch.Tensor)
            else torch.from_numpy(np.asarray(w_fwd, dtype=np.float32)).to(device=dev)
        )
        if wf.ndim == dim + 1:
            wf = wf.unsqueeze(0)
        coords_fwd = physical_to_normalized_torch_cached(
            X_phys + wf, shape_t, sp_t, org_t, dir_t
        )
        u0_sampled = sample_field_cf(
            torch.movedim(u0_d, -1, 1), coords_fwd, mode="bilinear", padding_mode="border"
        )
        u_tot_fwd = wf + u0_sampled

    u_tot_inv: Optional[torch.Tensor] = None
    if w_inv is not None:
        u0_i_d = (
            u0_inv_phys.to(device=dev, dtype=torch.float32)
            if u0_inv_phys is not None
            else update_inverse_field_nd_anderson(
                W_disp=u0_d,
                W_inv_disp=None,
                steps=15,
                m=5,
                spacing=spacing,
                origin=origin,
                direction=direction,
            )
        )
        wi = (
            w_inv.to(device=dev, dtype=torch.float32)
            if isinstance(w_inv, torch.Tensor)
            else torch.from_numpy(np.asarray(w_inv, dtype=np.float32)).to(device=dev)
        )
        if wi.ndim == dim + 1:
            wi = wi.unsqueeze(0)
        coords_inv = physical_to_normalized_torch_cached(
            X_phys + u0_i_d, shape_t, sp_t, org_t, dir_t
        )
        wi_sampled = sample_field_cf(
            torch.movedim(wi, -1, 1), coords_inv, mode="bilinear", padding_mode="border"
        )
        u_tot_inv = u0_i_d + wi_sampled

    return u_tot_fwd, u_tot_inv
