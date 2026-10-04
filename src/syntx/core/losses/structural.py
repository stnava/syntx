"""
Canonical structural, label, and distance-transform similarity losses.

Includes:
- soft_dice_loss_nd (Milletari squared denominator with AMP float16 overflow protection)
- compute_soft_distance_transform (differentiable Gaussian + log potential)
- _soft_signed_distance (differentiable signed boundary distance)
- compute_image_distance_transform (exact Euclidean distance transform via SciPy)
- distance_transform_loss (multi-mode distance field similarity metric)
- _soft_signed_distance (differentiable signed boundary distance)
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import scipy.ndimage as ndi
import torch

from ..smoothing import separable_gaussian_filter
from .cross_correlation import local_ncc_loss_nd
from .pointwise import mae_loss_nd, mse_loss_nd


def soft_dice_loss_nd(
    I: torch.Tensor,
    J: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Soft Dice loss of continuous probability / membership maps, any spatial dimension.

    Per (batch, channel): dice = 2 sum(I J) / (sum(I^2 + J^2) + eps) (squared-denominator form);
    loss = 1 - mean of dice over batch and channels.

    Parameters
    ----------
    I : Tensor (B, C, *spatial)
        Fixed / target maps.
    J : Tensor (B, C, *spatial)
        Moving / deformed maps. Symmetric in I and J.
    mask : Tensor, optional
        (B, 1, *spatial) or (B, C, *spatial); I and J are multiplied by it first.
    eps : float, default 1e-6
        Added to denominator.

    Returns
    -------
    torch.Tensor
        Scalar in [0, 1] for non-negative inputs; 0 = perfect overlap.
    """
    work_dtype = torch.float32 if I.dtype in (torch.float16, torch.bfloat16) else I.dtype
    dev_type = "cuda" if I.is_cuda else ("mps" if I.device.type == "mps" else "cpu")

    with torch.amp.autocast(device_type=dev_type, enabled=False):
        I_w = I.to(work_dtype)
        J_w = J.to(work_dtype)
        if mask is not None:
            m = mask.to(work_dtype)
            I_w = I_w * m
            J_w = J_w * m

        spatial_dims = tuple(range(2, I_w.ndim))
        intersection = 2.0 * torch.sum(I_w * J_w, dim=spatial_dims)
        cardinality = torch.sum(I_w**2 + J_w**2, dim=spatial_dims) + eps

        dice_per_channel = intersection / cardinality
        loss = 1.0 - torch.mean(dice_per_channel)

    return loss if I.dtype in (torch.float16, torch.bfloat16) else loss.to(I.dtype)


def compute_soft_distance_transform(
    image: torch.Tensor,
    sigma: float = 3.0,
    threshold: float | None = None,
    spacing: Sequence[float] | None = None,
) -> torch.Tensor:
    """Differentiable soft "distance from the object" map: D = -sigma * log(G_sigma * m)."""
    orig_shape = image.shape
    work_dtype = torch.float32 if image.dtype in (torch.float16, torch.bfloat16) else image.dtype
    if image.dim() == 2:
        img_nd = image.unsqueeze(0).unsqueeze(0).to(work_dtype)
    elif image.dim() == 3:
        if spacing is not None and len(spacing) == 2:
            img_nd = image.unsqueeze(0).to(work_dtype)
        else:
            img_nd = image.unsqueeze(0).unsqueeze(0).to(work_dtype)
    elif image.dim() in (4, 5):
        img_nd = image.to(work_dtype)
    else:
        raise ValueError(f"Unsupported image dimension: {image.dim()}")

    if threshold is not None:
        mask = torch.sigmoid((img_nd - threshold) * 10.0)
    else:
        max_val = torch.amax(img_nd.abs(), dim=tuple(range(2, img_nd.dim())), keepdim=True).clamp_min(1e-6)
        mask = torch.clamp(img_nd.abs() / max_val, 0.0, 1.0)

    mask_last = mask.movedim(1, -1)
    if spacing is not None:
        smoothed_last = separable_gaussian_filter(mask_last, sigma=sigma, spacing=spacing, sigma_mode="physical")
    else:
        smoothed_last = separable_gaussian_filter(mask_last, sigma=sigma)
    smoothed = smoothed_last.movedim(-1, 1).clamp_min(1e-8)

    dist = -float(sigma) * torch.log(smoothed)
    return dist.view(orig_shape)


def _soft_signed_distance(
    image: torch.Tensor,
    sigma: float,
    threshold: float | None = None,
    spacing: Sequence[float] | None = None,
) -> torch.Tensor:
    """Differentiable soft signed distance: object minus complement."""
    work_dtype = torch.float32 if image.dtype in (torch.float16, torch.bfloat16) else image.dtype
    img_w = image.to(work_dtype)
    if threshold is not None:
        comp = 2.0 * threshold - img_w
    else:
        max_val = (
            torch.amax(img_w.abs(), dim=tuple(range(2, img_w.dim())), keepdim=True)
            if img_w.dim() >= 4
            else img_w.abs().max()
        )
        comp = max_val - img_w.abs()
    return compute_soft_distance_transform(
        img_w, sigma=sigma, threshold=threshold, spacing=spacing
    ) - compute_soft_distance_transform(comp, sigma=sigma, threshold=threshold, spacing=spacing)


def compute_image_distance_transform(
    image: torch.Tensor | np.ndarray | Any,
    threshold: float | None = None,
    tau: float | None = None,
    signed: bool = False,
    sampling_spacing: Sequence[float] | None = None,
    return_ants: bool = False,
) -> torch.Tensor | Any:
    """Exact Euclidean distance transform (SciPy distance_transform_edt), non-differentiable."""
    is_ants = hasattr(image, "spacing") and hasattr(image, "numpy")
    ref_image = image if is_ants else None

    if is_ants:
        if sampling_spacing is None:
            sampling_spacing = tuple(float(s) for s in image.spacing)
        img_np_raw = image.numpy().astype(np.float32)
        device = torch.device("cpu")
        dtype = torch.float32
        orig_shape = img_np_raw.shape
        img_nd = torch.from_numpy(img_np_raw).unsqueeze(0).unsqueeze(0)
        sampling = sampling_spacing
    else:
        if not isinstance(image, torch.Tensor):
            image_t = torch.as_tensor(image, dtype=torch.float32)
        else:
            image_t = image
        device = image_t.device
        dtype = image_t.dtype if image_t.is_floating_point() else torch.float32
        orig_shape = image_t.shape

        if image_t.dim() == 2:
            img_nd = image_t.unsqueeze(0).unsqueeze(0)
        elif image_t.dim() == 3:
            if sampling_spacing is not None and len(sampling_spacing) == 2:
                img_nd = image_t.unsqueeze(0)
            else:
                img_nd = image_t.unsqueeze(0).unsqueeze(0)
        elif image_t.dim() in (4, 5):
            img_nd = image_t
        else:
            raise ValueError(f"Unsupported image dimension: {image_t.dim()}")

        if sampling_spacing is not None:
            sampling = tuple(reversed(sampling_spacing))
        else:
            sampling = None

    B, C = img_nd.shape[:2]
    spatial_shape = img_nd.shape[2:]

    img_np = img_nd.detach().cpu().numpy()
    out_np = np.zeros_like(img_np, dtype=np.float32)

    for b in range(B):
        for c in range(C):
            sl = img_np[b, c]
            if threshold is not None:
                fg = sl > threshold
            else:
                max_v = float(np.max(np.abs(sl)))
                fg = np.abs(sl) > (1e-4 * max_v if max_v > 0 else 1e-4)

            if not np.any(fg):
                diag_span = float(
                    np.sqrt(
                        sum(
                            (dim_sz * (sp if sp else 1.0)) ** 2
                            for dim_sz, sp in zip(spatial_shape, sampling or [1.0] * len(spatial_shape))
                        )
                    )
                )
                edt = np.ones_like(sl, dtype=np.float32) * diag_span
            elif np.all(fg):
                diag_span = float(
                    np.sqrt(
                        sum(
                            (dim_sz * (sp if sp else 1.0)) ** 2
                            for dim_sz, sp in zip(spatial_shape, sampling or [1.0] * len(spatial_shape))
                        )
                    )
                )
                edt = (-diag_span * np.ones_like(sl, dtype=np.float32)) if signed else np.zeros_like(sl, dtype=np.float32)
            else:
                if signed:
                    d_out = ndi.distance_transform_edt(~fg, sampling=sampling)
                    d_in = ndi.distance_transform_edt(fg, sampling=sampling)
                    edt = d_out - d_in
                else:
                    edt = ndi.distance_transform_edt(~fg, sampling=sampling)
            out_np[b, c] = edt

    if is_ants and return_ants:
        import ants

        out_res = out_np.reshape(orig_shape)
        if tau is not None:
            if tau <= 0.0:
                raise ValueError(f"tau must be strictly positive, got {tau}")
            out_res = np.exp(-np.abs(out_res) / float(tau))
        return ants.from_numpy(
            out_res.astype(np.float32),
            origin=ref_image.origin,
            spacing=ref_image.spacing,
            direction=ref_image.direction,
        )

    out_t = torch.from_numpy(out_np).to(device=device, dtype=dtype)
    if tau is not None:
        if tau <= 0.0:
            raise ValueError(f"tau must be strictly positive, got {tau}")
        out_t = torch.exp(-torch.abs(out_t) / float(tau))
    return out_t.view(orig_shape)


def distance_transform_loss(
    I: torch.Tensor,
    J: torch.Tensor,
    mode: Literal["potential_lncc", "potential_mse", "edt_mse", "edt_l1", "sdf_mse"] = "potential_lncc",
    tau: float = 0.10,
    window_size: int = 9,
    mask: torch.Tensor | None = None,
    is_distance_field: bool = False,
    threshold: float | None = None,
    spacing: Sequence[float] | None = None,
) -> torch.Tensor:
    """Loss comparing two images through distance maps (or distance potentials)."""
    if tau is None or not float(tau) > 0:
        raise ValueError(f"distance_transform_loss: tau must be > 0, got {tau!r}")
    if not is_distance_field and mode == "sdf_mse":
        D_I = _soft_signed_distance(I, sigma=tau * 10.0, threshold=threshold, spacing=spacing)
        D_J = _soft_signed_distance(J, sigma=tau * 10.0, threshold=threshold, spacing=spacing)
    elif not is_distance_field:
        D_I = compute_soft_distance_transform(I, sigma=tau * 10.0, threshold=threshold, spacing=spacing)
        D_J = compute_soft_distance_transform(J, sigma=tau * 10.0, threshold=threshold, spacing=spacing)
    else:
        D_I = I
        D_J = J

    if mode == "potential_lncc":
        P_I = torch.exp(-torch.abs(D_I) / tau) if is_distance_field else torch.exp(-D_I / tau)
        P_J = torch.exp(-torch.abs(D_J) / tau) if is_distance_field else torch.exp(-D_J / tau)
        return local_ncc_loss_nd(P_I, P_J, mask=mask, window_size=window_size)
    elif mode == "potential_mse":
        P_I = torch.exp(-torch.abs(D_I) / tau) if is_distance_field else torch.exp(-D_I / tau)
        P_J = torch.exp(-torch.abs(D_J) / tau) if is_distance_field else torch.exp(-D_J / tau)
        return mse_loss_nd(P_I, P_J, mask=mask)
    elif mode in ("edt_mse", "sdf_mse"):
        return mse_loss_nd(D_I, D_J, mask=mask)
    elif mode == "edt_l1":
        return mae_loss_nd(D_I, D_J, mask=mask)
    else:
        raise ValueError(f"Unknown distance transform loss mode: '{mode}'")
