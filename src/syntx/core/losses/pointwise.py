"""
Canonical pointwise and pairwise similarity losses (MSE, L2, MAE, Huber).

Supports arbitrary spatial dimensions (2-D, 3-D, N-D), broadcastable masks,
precision-aware computation (float64 gradcheck compliance, float16 AMP overflow protection),
and division-by-zero protection.
"""

from __future__ import annotations

import torch


def mse_loss_nd(
    I: torch.Tensor,
    J: torch.Tensor,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Mean squared error loss between images I and J with optional masking.

    Parameters
    ----------
    I, J : torch.Tensor
        Input images of identical shape (*spatial or B, C, *spatial).
    mask : torch.Tensor, optional
        Weight/selection mask broadcastable to I and J. Non-negative weights.
    reduction : {'mean', 'sum', 'none'}, default 'mean'
        Reduction mode applied over spatial and batch elements.

    Returns
    -------
    torch.Tensor
        Scalar loss if reduction is 'mean' or 'sum', otherwise elementwise loss tensor.
    """
    # Prevent float16 squared intensity overflow (max 65504) under AMP while preserving float64 for gradcheck
    work_dtype = torch.float32 if I.dtype in (torch.float16, torch.bfloat16) else I.dtype
    dev_type = "cuda" if I.is_cuda else ("mps" if I.device.type == "mps" else "cpu")

    with torch.amp.autocast(device_type=dev_type, enabled=False):
        diff = I.to(work_dtype) - J.to(work_dtype)
        sq_diff = diff * diff

        if mask is not None:
            m = mask.to(work_dtype)
            weighted_sq = sq_diff * m
            if reduction == "mean":
                sum_m = torch.clamp_min(torch.sum(m), 1e-8)
                loss = torch.sum(weighted_sq) / sum_m
            elif reduction == "sum":
                loss = torch.sum(weighted_sq)
            elif reduction == "none":
                loss = weighted_sq
            else:
                raise ValueError(f"Unknown reduction: {reduction!r}")
        else:
            if reduction == "mean":
                loss = torch.mean(sq_diff)
            elif reduction == "sum":
                loss = torch.sum(sq_diff)
            elif reduction == "none":
                loss = sq_diff
            else:
                raise ValueError(f"Unknown reduction: {reduction!r}")

    return loss if I.dtype in (torch.float16, torch.bfloat16) else loss.to(I.dtype)


def l2_loss_nd(
    I: torch.Tensor,
    J: torch.Tensor,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
    squared: bool = True,
) -> torch.Tensor:
    """
    L2 Euclidean distance loss between images I and J.

    When ``squared=True`` (default), equivalent to ``mse_loss_nd``.
    When ``squared=False``, computes root mean squared error (RMSE) or Euclidean norm.

    Parameters
    ----------
    I, J : torch.Tensor
        Input images of identical shape.
    mask : torch.Tensor, optional
        Weight/selection mask broadcastable to I and J.
    reduction : {'mean', 'sum', 'none'}, default 'mean'
    squared : bool, default True
        If True, returns squared L2 / MSE. If False, returns square root.

    Returns
    -------
    torch.Tensor
        Computed L2 loss.
    """
    mse = mse_loss_nd(I, J, mask=mask, reduction=reduction)
    if squared:
        return mse
    return torch.sqrt(torch.clamp_min(mse, 1e-12))


def mae_loss_nd(
    I: torch.Tensor,
    J: torch.Tensor,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Mean absolute error (L1) loss between images I and J with optional masking.

    Parameters
    ----------
    I, J : torch.Tensor
        Input images of identical shape.
    mask : torch.Tensor, optional
        Weight/selection mask broadcastable to I and J.
    reduction : {'mean', 'sum', 'none'}, default 'mean'

    Returns
    -------
    torch.Tensor
        Computed MAE loss.
    """
    work_dtype = torch.float32 if I.dtype in (torch.float16, torch.bfloat16) else I.dtype
    dev_type = "cuda" if I.is_cuda else ("mps" if I.device.type == "mps" else "cpu")

    with torch.amp.autocast(device_type=dev_type, enabled=False):
        abs_diff = torch.abs(I.to(work_dtype) - J.to(work_dtype))

        if mask is not None:
            m = mask.to(work_dtype)
            weighted_abs = abs_diff * m
            if reduction == "mean":
                sum_m = torch.clamp_min(torch.sum(m), 1e-8)
                loss = torch.sum(weighted_abs) / sum_m
            elif reduction == "sum":
                loss = torch.sum(weighted_abs)
            elif reduction == "none":
                loss = weighted_abs
            else:
                raise ValueError(f"Unknown reduction: {reduction!r}")
        else:
            if reduction == "mean":
                loss = torch.mean(abs_diff)
            elif reduction == "sum":
                loss = torch.sum(abs_diff)
            elif reduction == "none":
                loss = abs_diff
            else:
                raise ValueError(f"Unknown reduction: {reduction!r}")

    return loss if I.dtype in (torch.float16, torch.bfloat16) else loss.to(I.dtype)


def huber_loss_nd(
    I: torch.Tensor,
    J: torch.Tensor,
    mask: torch.Tensor | None = None,
    delta: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Huber loss between images I and J with optional masking.

    Quadratic for residuals ``|r| <= delta`` (``r**2 / 2``) and linear beyond (``delta * (|r| - delta / 2)``), so
    voxels whose residual is dominated by noise or by content that no rigid map can match (signal dropout, pile-up,
    partial-volume at the brain edge) contribute at most a bounded gradient instead of the unbounded pull of MSE.
    Continuous first derivative everywhere (unlike MAE), hence usable with L-BFGS. ``delta`` is in the images'
    intensity units; choose it at the noise level of the (normalised) residual.

    Parameters
    ----------
    I, J : torch.Tensor
        Input images of identical shape (*spatial or B, C, *spatial).
    mask : torch.Tensor, optional
        Weight/selection mask broadcastable to I and J. Non-negative weights.
    delta : float, default 0.1
        Transition point between the quadratic and the linear regime (intensity units). Must be > 0.
    reduction : {'mean', 'sum', 'none'}, default 'mean'
        Reduction mode applied over spatial and batch elements.

    Returns
    -------
    torch.Tensor
        Scalar loss if reduction is 'mean' or 'sum', otherwise elementwise loss tensor.
    """
    if not delta > 0:
        raise ValueError(f"delta must be > 0, got {delta!r}")
    work_dtype = torch.float32 if I.dtype in (torch.float16, torch.bfloat16) else I.dtype
    dev_type = "cuda" if I.is_cuda else ("mps" if I.device.type == "mps" else "cpu")

    with torch.amp.autocast(device_type=dev_type, enabled=False):
        abs_diff = torch.abs(I.to(work_dtype) - J.to(work_dtype))
        d = torch.as_tensor(delta, dtype=work_dtype, device=I.device)
        elementwise = torch.where(abs_diff <= d, 0.5 * abs_diff * abs_diff, d * (abs_diff - 0.5 * d))

        if mask is not None:
            m = mask.to(work_dtype)
            weighted = elementwise * m
            if reduction == "mean":
                sum_m = torch.clamp_min(torch.sum(m), 1e-8)
                loss = torch.sum(weighted) / sum_m
            elif reduction == "sum":
                loss = torch.sum(weighted)
            elif reduction == "none":
                loss = weighted
            else:
                raise ValueError(f"Unknown reduction: {reduction!r}")
        else:
            if reduction == "mean":
                loss = torch.mean(elementwise)
            elif reduction == "sum":
                loss = torch.sum(elementwise)
            elif reduction == "none":
                loss = elementwise
            else:
                raise ValueError(f"Unknown reduction: {reduction!r}")

    return loss if I.dtype in (torch.float16, torch.bfloat16) else loss.to(I.dtype)
