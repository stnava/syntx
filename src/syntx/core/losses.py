"""
Similarity losses for registration (torch, 2-D and 3-D, tensors (B, C, *spatial) in tensor order
(z, y, x) unless stated otherwise). All losses are "lower is better".

- Local normalised cross-correlation: ``local_ncc_loss_nd`` (autograd path, or the hand-written
  ``AnalyticalLNCC`` / ``ANTsPseudoLNCC`` backward passes), ``BoxLNCCLoss`` /
  ``box_lncc_loss_nd`` / ``box_cc2_loss_nd`` (box sums by separable convolution, smoothing
  constants instead of a variance floor). ``box_mean_nd`` is the local box mean with a
  reproducible GPU / MPS gradient.
- Mattes mutual information with cubic B-spline Parzen windows: ``mattes_mi_loss_nd`` (scales to
  [-1, 1]), ``mattes_mi_loss_core``, ``parzen_weights``, ``mattes_mi_from_weights``.
- Distance-transform losses: ``compute_soft_distance_transform`` (differentiable, Gaussian
  smoothing + log), ``compute_image_distance_transform`` (exact EDT via SciPy, not
  differentiable), ``distance_transform_loss``.
- ``soft_dice_loss_nd`` for probability / label-membership maps.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union, Sequence, Literal, Any
import numpy as np
import scipy.ndimage as ndi


def _box_pool(x, window_size, count_include_pad):
    """Stride-1 ``avg_pool2d`` / ``avg_pool3d`` (chosen by ``x.dim()``) with zero padding
    ``window_size // 2``. Output has the input shape only for odd ``window_size``."""
    pool_fn = F.avg_pool2d if x.dim() == 4 else F.avg_pool3d
    return pool_fn(x, kernel_size=window_size, stride=1, padding=window_size // 2,
                   count_include_pad=count_include_pad)


class _DeterministicBoxMean(torch.autograd.Function):
    """Local box mean (stride 1, zero padding, count_include_pad=False) whose backward is
    run-to-run reproducible.

    The GPU/MPS ``avg_pool*_backward`` kernels scatter with float atomics, so identical calls give
    gradients differing by ~1e-7, which iterative registration amplifies. For an odd window the
    zero-padded stride-1 box sum S is self-adjoint, so with y = S(x) / count the exact gradient is
    S(g / count) -- computed with the (deterministic, gather-based) forward pooling kernel.
    """

    @staticmethod
    def forward(ctx, x, window_size):
        y = _box_pool(x, window_size, count_include_pad=False)
        ctx.window_size = window_size
        return y

    @staticmethod
    def backward(ctx, g):
        k = ctx.window_size
        n = k ** (g.dim() - 2)
        # count = S(1) = n * avg_pool(1, include_pad); S(z) = n * avg_pool(z, include_pad)
        ones = torch.ones((1, 1) + tuple(g.shape[2:]), dtype=g.dtype, device=g.device)
        count = _box_pool(ones, k, count_include_pad=True)          # = S(1) / n
        return _box_pool(g / count, k, count_include_pad=True), None


# Devices on which box_mean_nd uses the deterministic backward (CPU pooling is already deterministic).
DETERMINISTIC_POOL_DEVICES = ('mps', 'cuda')


def box_mean_nd(x, window_size):
    """Local box mean of ``x`` (B, C, *spatial), 2-D or 3-D: ``avg_pool{2,3}d(x, window_size,
    stride=1, padding=window_size//2, count_include_pad=False)``, i.e. near the border the mean
    is over the in-image voxels only.

    The deterministic backward of ``_DeterministicBoxMean`` is used only when ``window_size`` is
    odd, ``x`` requires grad, grad mode is on and ``x`` is on a device in
    ``DETERMINISTIC_POOL_DEVICES`` ('mps', 'cuda'); otherwise plain pooling. Returns a tensor of
    the input shape (odd ``window_size``; an even one gives one extra voxel per axis).
    """
    if (window_size % 2 == 1 and x.requires_grad and torch.is_grad_enabled()
            and x.device.type in DETERMINISTIC_POOL_DEVICES):
        return _DeterministicBoxMean.apply(x, window_size)
    return _box_pool(x, window_size, count_include_pad=False)


def _window_counts(x, window_size):
    """Number of in-image voxels of each box window (zero padding excluded), shape of ``x``."""
    dim = x.dim() - 2
    pool = F.avg_pool2d if dim == 2 else F.avg_pool3d
    ones = torch.ones_like(x[:, :1])
    pad = window_size // 2
    return pool(ones, kernel_size=window_size, stride=1, padding=pad, count_include_pad=True) * (window_size ** dim)


class AnalyticalLNCC(torch.autograd.Function):
    """Local CC loss (not CC^2) with a hand-written, approximate backward pass.

    ``AnalyticalLNCC.apply(I, J, mask, window_size)``; I, J (B, 1, *spatial), 2-D or 3-D;
    ``mask`` None or (B, 1, *spatial); ``window_size`` odd. Called by
    ``local_ncc_loss_nd(..., use_ants_pseudo_gradient=True, squared=False)``.

    Forward: the same value as the autograd path of ``local_ncc_loss_nd(squared=False)``:
    box means / variances / covariance (``avg_pool``, ``count_include_pad=False``), variances
    floored at 1e-6, CC = cov / (sqrt(var_I var_J) + 1e-6) clamped to [-1, 1], loss = -mean(CC)
    (over voxels with mask > 0.5 and both raw variances > 1e-6 when a mask is given, else over
    all voxels). Returns a scalar.

    Backward: no autograd graph through the pooling. Each voxel c gets only the derivative of its
    own window's CC with respect to its own (centre) intensity, in the ANTs style:

        dCC/dJ_c ~ (1/N) / sqrt(var_I var_J) * (F_c - CC * M_c)
        dCC/dI_c ~ (1/N) / sqrt(var_I var_J) * (M_c - CC * F_c)

    F_c, M_c are the mean-subtracted centre intensities of I, J and N the number of in-image
    voxels of the window (window_size ** dim in the interior, fewer at the border). This
    is not the exact gradient of the forward value: contributions of the neighbouring windows
    that also contain c are ignored, and the exact centre-pixel term has CC * sqrt(var_I / var_J)
    in place of CC (they agree only when the local variances are equal). The gradient is divided
    by the number of averaged voxels and zeroed outside the active set when a mask is given.
    No gradient is returned for ``mask`` or ``window_size``.
    """

    @staticmethod
    def forward(ctx, I, J, mask, window_size):
        dim = I.dim() - 2
        pad = window_size // 2

        if dim == 2:
            pool_fn = F.avg_pool2d
        elif dim == 3:
            pool_fn = F.avg_pool3d
        else:
            raise ValueError(f"Only 2D and 3D images are supported, got {dim}D.")

        def box_filter(x):
            return pool_fn(x, kernel_size=window_size, stride=1, padding=pad, count_include_pad=False)

        I_mean = box_filter(I)
        J_mean = box_filter(J)

        F_centered = I - I_mean
        M_centered = J - J_mean

        I_var = torch.clamp(box_filter(F_centered ** 2), min=0.0)
        J_var = torch.clamp(box_filter(M_centered ** 2), min=0.0)
        IJ_cov = box_filter(F_centered * M_centered)

        var_floor = 1e-6
        safe_I_var = torch.clamp(I_var, min=var_floor)
        safe_J_var = torch.clamp(J_var, min=var_floor)

        denom = torch.sqrt(safe_I_var * safe_J_var) + 1e-6
        cc_raw = IJ_cov / denom
        cc = torch.clamp(cc_raw, min=-1.0, max=1.0)

        ctx.save_for_backward(F_centered, M_centered, cc, safe_I_var, safe_J_var, _window_counts(I, window_size))

        if mask is not None:
            active = ((I_var > 1e-6) & (J_var > 1e-6) & (mask > 0.5)).to(I.dtype)
            loss = -torch.sum(cc * active) / (torch.sum(active) + 1e-8)
            ctx.active = active
        else:
            loss = -torch.mean(cc)
            ctx.active = None

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        F_centered, M_centered, cc, safe_I_var, safe_J_var, n_window = ctx.saved_tensors

        inv_denom = 1.0 / (torch.sqrt(safe_I_var * safe_J_var) + 1e-6)

        # Analytical derivative of CC w.r.t. center pixel:
        #   dCC/dJ_c = (1/N) / sqrt(sFF * sMM) * (F_c - CC * M_c)
        #   dCC/dI_c = (1/N) / sqrt(sFF * sMM) * (M_c - CC * F_c)
        # Loss is -CC, so negate:
        scale = -(1.0 / n_window) * inv_denom

        grad_J = scale * (F_centered - cc * M_centered)
        grad_I = scale * (M_centered - cc * F_centered)

        if ctx.active is not None:
            N_spatial = torch.sum(ctx.active) + 1e-8
            grad_J = grad_J * ctx.active / N_spatial
            grad_I = grad_I * ctx.active / N_spatial
        else:
            N_spatial = F_centered.numel() / F_centered.shape[0]
            grad_J = grad_J / N_spatial
            grad_I = grad_I / N_spatial

        return grad_I * grad_output, grad_J * grad_output, None, None


class ANTsPseudoLNCC(torch.autograd.Function):
    """Local CC^2 loss with the ITK / ANTs pseudo-derivative as its backward pass.

    ``ANTsPseudoLNCC.apply(I, J, mask, window_size)``; arguments as for ``AnalyticalLNCC``.
    Called by ``local_ncc_loss_nd(..., use_ants_pseudo_gradient=True, squared=True)``.

    Forward: CC^2 = cov^2 / (var_I var_J + 1e-8) (variances floored at 1e-6) clamped to [0, 1];
    loss = -mean(CC^2) over all voxels, or over voxels with mask > 0.5 and both raw variances
    > 1e-6 when a mask is given. Returns a scalar in [-1, 0].

    Backward (per voxel c, own window only, as in ITK's
    ``ANTSNeighborhoodCorrelationImageToImageMetricv4``):

        d(CC^2)/dJ_c ~ 2/N * cov / (var_I var_J) * (F_c - cov / var_J * M_c)

    and symmetrically for I_c, negated for the loss, divided by the number of averaged voxels and
    zeroed outside the active set. N is the number of in-image voxels of the window (fewer at
    the border). Contributions from neighbouring windows are ignored, so this is a
    pseudo-gradient, not the exact gradient of the forward value.
    """
    @staticmethod
    def forward(ctx, I, J, mask, window_size):
        dim = I.dim() - 2
        pad = window_size // 2

        if dim == 2:
            pool_fn = F.avg_pool2d
        elif dim == 3:
            pool_fn = F.avg_pool3d
        else:
            raise ValueError(f"Only 2D and 3D images are supported, got {dim}D.")
            
        def box_filter(x):
            return pool_fn(x, kernel_size=window_size, stride=1, padding=pad, count_include_pad=False)
            
        I_mean = box_filter(I)
        J_mean = box_filter(J)
        
        F_centered = I - I_mean
        M_centered = J - J_mean
        
        I_var = torch.clamp(box_filter(F_centered**2), min=0.0)
        J_var = torch.clamp(box_filter(M_centered**2), min=0.0)
        IJ_cov = box_filter(F_centered * M_centered)
        
        var_floor = 1e-6
        safe_I_var = torch.clamp(I_var, min=var_floor)
        safe_J_var = torch.clamp(J_var, min=var_floor)
        
        # ITK uses CC^2: localCC = sFixedMoving * sFixedMoving / (sFixedFixed * sMovingMoving)
        cc2_raw = (IJ_cov ** 2) / (safe_I_var * safe_J_var + 1e-8)
        cc2 = torch.clamp(cc2_raw, min=0.0, max=1.0)
        
        ctx.save_for_backward(F_centered, M_centered, IJ_cov, safe_I_var, safe_J_var, _window_counts(I, window_size))
        
        if mask is not None:
            active = ((I_var > 1e-6) & (J_var > 1e-6) & (mask > 0.5)).to(I.dtype)
            loss = -torch.sum(cc2 * active) / (torch.sum(active) + 1e-8)
            ctx.active = active
        else:
            loss = -torch.mean(cc2)
            ctx.active = None
            
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        F_centered, M_centered, IJ_cov, safe_I_var, safe_J_var, n_window = ctx.saved_tensors
        
        s_FM = IJ_cov
        s_FF = safe_I_var
        s_MM = safe_J_var
        
        sFF_sMM = s_FF * s_MM + 1e-8
        
        # ITK's pseudo-derivative for +CC^2 wrt moving center pixel M_c is:
        # 2/N * cov / (var_F * var_M) * (F_c - cov / var_M * M_c)
        # By symmetry, wrt fixed center pixel F_c is:
        # 2/N * cov / (var_F * var_M) * (M_c - cov / var_F * F_c)
        
        # Since our loss is -CC^2, the gradient of the loss is the negative of this.
        grad_factor = -2.0 * (1.0 / n_window) * (s_FM / sFF_sMM)
        
        grad_J = grad_factor * (F_centered - (s_FM / (s_MM + 1e-8)) * M_centered)
        grad_I = grad_factor * (M_centered - (s_FM / (s_FF + 1e-8)) * F_centered)
        
        # Scale by the spatial reduction (mean over all pixels)
        if ctx.active is not None:
            N_spatial = torch.sum(ctx.active) + 1e-8
            grad_J = grad_J * ctx.active / N_spatial
            grad_I = grad_I * ctx.active / N_spatial
        else:
            N_spatial = F_centered.numel() / F_centered.shape[0]
            grad_J = grad_J / N_spatial
            grad_I = grad_I / N_spatial
            
        return grad_I * grad_output, grad_J * grad_output, None, None


def local_ncc_loss_nd(
    I: torch.Tensor,
    J: torch.Tensor,
    mask: torch.Tensor = None,
    window_size: int = 9,
    use_ants_pseudo_gradient: bool = False,
    squared: bool = False
) -> torch.Tensor:
    """
    Negative local normalised cross-correlation (LNCC) of 2-D or 3-D images I and J.

    In a sliding box window (zero padding, border windows average the in-image voxels only):
    local means, variances var_I, var_J (clamped >= 0, then floored at 1e-6) and covariance.
    Per voxel, CC = cov / (sqrt(var_I var_J) + 1e-6) clamped to [-1, 1], or with ``squared``
    CC^2 = cov^2 / (var_I var_J + 1e-8) clamped to [0, 1]. The loss is minus the mean over
    voxels. The floor keeps 1/var finite in flat regions; there CC is ~0.

    Parameters
    ----------
    I, J : torch.Tensor
        Images (B, C, *spatial), same shape, usually C = 1 (channels are pooled separately
        and averaged). Symmetric in I and J.
    mask : torch.Tensor, optional
        (B, 1, *spatial). If given, the mean is over voxels with mask > 0.5 AND both unfloored
        local variances > 1e-6 (flat regions are excluded). If None, over all voxels.
    window_size : int, default 9
        Box width in voxels; must be odd (ValueError). If larger than the smallest spatial size
        it is reduced to that size (minus 1 if even).
    use_ants_pseudo_gradient : bool, default False
        True: use a hand-written backward without an autograd graph through the pooling --
        ``ANTsPseudoLNCC`` (ITK CC^2 pseudo-derivative) when ``squared``, else
        ``AnalyticalLNCC`` (approximate CC gradient). Same forward value; see those classes.
        False: autograd through ``box_mean_nd`` (exact gradient, reproducible on GPU / MPS).
    squared : bool, default False
        Use CC^2 instead of CC (insensitive to the sign of the correlation).

    Returns
    -------
    torch.Tensor
        Scalar. In [-1, 1] for CC (-1 = perfect positive correlation), [-1, 0] for CC^2.

    Raises
    ------
    ValueError
        If the images are not 2-D or 3-D (autograd path and the custom functions).
    """
    dim = I.dim() - 2
    if dim not in (2, 3):
        raise ValueError(f"Only 2D and 3D images are supported, got {dim}D.")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(f"local_ncc_loss_nd: window_size must be a positive odd integer, got {window_size}")

    # Adapt window size dynamically if input is smaller than kernel
    min_spatial = min(I.shape[2:])
    if window_size > min_spatial:
        window_size = min_spatial
        if window_size % 2 == 0:
            window_size = max(1, window_size - 1)
            
    if use_ants_pseudo_gradient and squared:
        # ANTsPseudoLNCC optimizes CC^2 using the ITK pseudo-derivative formula.
        return ANTsPseudoLNCC.apply(I, J, mask, window_size)
    elif use_ants_pseudo_gradient and not squared:
        # AnalyticalLNCC: CC with an approximate (centre-window, ANTs-style) analytical gradient;
        # same speed as ANTsPseudoLNCC (no autograd graph through avg_pool3d).
        return AnalyticalLNCC.apply(I, J, mask, window_size)
            
    def box_filter(x):
        return box_mean_nd(x, window_size)

    I_mean = box_filter(I)
    J_mean = box_filter(J)

    # 1. Non-negative variance enforcement
    I_var = torch.clamp(box_filter((I - I_mean)**2), min=0.0)
    J_var = torch.clamp(box_filter((J - J_mean)**2), min=0.0)
    IJ_cov = box_filter((I - I_mean) * (J - J_mean))
    
    # 2. Variance floor to prevent 1/var derivative explosion
    var_floor = 1e-6
    safe_I_var = torch.clamp(I_var, min=var_floor)
    safe_J_var = torch.clamp(J_var, min=var_floor)
    
    if squared:
        cc_metric = (IJ_cov ** 2) / (safe_I_var * safe_J_var + 1e-8)
        cc_metric = torch.clamp(cc_metric, min=0.0, max=1.0)
    else:
        cc_raw = IJ_cov / (torch.sqrt(safe_I_var * safe_J_var) + 1e-6)
        cc_metric = torch.clamp(cc_raw, min=-1.0, max=1.0)
    
    if mask is not None:
        active_mask_float = ((I_var > 1e-6) & (J_var > 1e-6) & (mask > 0.5)).to(dtype=I.dtype)
        return -torch.sum(cc_metric * active_mask_float) / (torch.sum(active_mask_float) + 1e-8)
    else:
        return -torch.mean(cc_metric)


class BoxLNCCLoss(torch.nn.Module):
    """
    Box-window local cross-correlation loss (CC^2 by default, or CC), computed with box sums.

    With S(.) the sum over a ``kernel_size`` box (separable ones-kernel convolution, zero
    padding) and n = kernel_size ** dim:
    cross = S(t p) - S(p) S(t) / n, var_t = max(S(t^2) - S(t)^2 / n, smooth_dr), likewise var_p;
    ``squared``: ncc = (cross^2 + smooth_nr) / (var_t var_p + smooth_dr); else
    ncc = cross / (sqrt(var_t var_p) + smooth_dr). Loss = -mean(ncc) over all voxels (no mask).

    In flat or zero background the squared form gives ~ smooth_nr / smooth_dr = 1 (treated as
    perfectly correlated, so it contributes no gradient); the linear form gives ~0 there. n is
    used unchanged near the image border, where the zero padding enters the sums as data.
    The squared ratio is not clamped. Computed in float32 with autocast disabled.

    Parameters
    ----------
    kernel_size : int, default 5
        Box width in voxels; must be odd (an even width breaks the reshape in
        ``separable_1d_filter``).
    smooth_nr, smooth_dr : float, default 1e-5
        Numerator constant (squared form only) and variance floor / denominator constant.
    squared : bool, default True
        CC^2 (True) or CC (False).

    Notes
    -----
    Box sums of ``target`` are cached per module instance when ``target`` does not require
    grad, keyed by ``id(target)`` and its in-place version counter (at most 4 entries; a
    reference is kept so the id stays unique). Reusing one instance with a fixed target saves
    two of the five convolutions.
    """
    def __init__(self, kernel_size: int = 5, smooth_nr: float = 1e-5, smooth_dr: float = 1e-5, squared: bool = True):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"BoxLNCCLoss: kernel_size must be a positive odd integer, got {kernel_size}")
        self.kernel_size = kernel_size
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.squared = squared
        self._target_cache = {}
        self._target_refs = {}

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss of ``pred`` (moving) against ``target`` (fixed), both (B, C, *spatial), 2-D or
        3-D. Returns a float32 scalar (-1 at perfect correlation for either form)."""
        # Crucial: Under AMP float16, sum of squares over 3D volumes (125 voxels * intensity^2)
        # can easily overflow float16 (max 65504) or underflow in variance/gradients, causing NaNs.
        # Disabling autocast inside BoxLNCC ensures robust, overflow-free float32 computation.
        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            pred_f = pred.float()
            target_f = target.float()
            from .smoothing import separable_1d_filter
            dim = pred_f.dim() - 2
            kernel_vol = float(self.kernel_size ** dim)
            k = torch.ones(self.kernel_size, dtype=torch.float32, device=pred_f.device)
            kernels = [k] * dim

            target_id = id(target)
            version = getattr(target, '_version', 0)
            if not target.requires_grad and target_id in self._target_cache and self._target_cache[target_id][0] == version:
                _, t_sum, t_var = self._target_cache[target_id]
            else:
                t_sum = separable_1d_filter(target_f, kernels)
                t2_sum = separable_1d_filter(target_f * target_f, kernels)
                t_var = torch.clamp(t2_sum - t_sum * t_sum / kernel_vol, min=self.smooth_dr)
                if not target.requires_grad:
                    if len(self._target_cache) >= 4:
                        self._target_cache.clear()
                        self._target_refs.clear()
                    self._target_cache[target_id] = (version, t_sum, t_var)
                    self._target_refs[target_id] = target

            p_sum = separable_1d_filter(pred_f, kernels)
            p2_sum = separable_1d_filter(pred_f * pred_f, kernels)
            tp_sum = separable_1d_filter(target_f * pred_f, kernels)

            cross = tp_sum - p_sum * t_sum / kernel_vol
            p_var = torch.clamp(p2_sum - p_sum * p_sum / kernel_vol, min=self.smooth_dr)

            if self.squared:
                ncc = (cross * cross + self.smooth_nr) / (t_var * p_var + self.smooth_dr)
            else:
                ncc = cross / (torch.sqrt(t_var * p_var) + self.smooth_dr)
            return -torch.mean(ncc)


def box_lncc_loss_nd(
    I: torch.Tensor,
    J: torch.Tensor,
    window_size: int = 5,
    smooth_nr: float = 1e-5,
    smooth_dr: float = 1e-5,
    squared: bool = False,
) -> torch.Tensor:
    """``BoxLNCCLoss(window_size, smooth_nr, smooth_dr, squared)(I, J)``: I is the moving
    (``pred``), J the fixed (``target``) image. Note the default here is linear CC
    (``squared=False``), unlike the class. A new module per call, so nothing is cached."""
    loss_fn = BoxLNCCLoss(kernel_size=window_size, smooth_nr=smooth_nr, smooth_dr=smooth_dr, squared=squared)
    return loss_fn(I, J)


def box_cc2_loss_nd(
    I: torch.Tensor,
    J: torch.Tensor,
    window_size: int = 5,
    smooth_nr: float = 1e-5,
    smooth_dr: float = 1e-5,
) -> torch.Tensor:
    """``box_lncc_loss_nd(..., squared=True)``: the box CC^2 loss of ``BoxLNCCLoss``."""
    return box_lncc_loss_nd(I, J, window_size=window_size, smooth_nr=smooth_nr, smooth_dr=smooth_dr, squared=True)


def b_spline_3(x):
    """Cubic B-spline B3(x), elementwise: 2/3 - x^2 + |x|^3 / 2 for |x| < 1,
    (2 - |x|)^3 / 6 for 1 <= |x| < 2, else 0. Written with multiplications instead of ``pow``.
    """
    abs_x = torch.abs(x)
    x2 = abs_x * abs_x
    y1 = (2.0 / 3.0) - x2 + 0.5 * x2 * abs_x
    rem = 2.0 - abs_x
    y2 = (1.0 / 6.0) * (rem * rem * rem)
    return torch.where(abs_x < 1.0, y1, torch.where(abs_x < 2.0, y2, torch.zeros_like(x)))


def _parzen_joint_histogram(w_x: torch.Tensor, w_y: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """
    Joint Parzen histogram  H = w_xᵀ w_y  for [N, B] B-spline weight matrices, accumulated as a
    batch of ``chunk``-row blocks (``torch.bmm``) followed by a fixed-order sum.

    Why not one big matmul: on Apple MPS (torch 2.13) ``w_x.t() @ w_y`` with N ≳ 1e5 is
    non-deterministic and, for concentrated histograms, wrong by up to ~15 % (the affine solver
    produced different transforms on every first call). Blocked bmm with K = 4096 was measured
    to be bitwise deterministic across allocations and ~30x closer to a float64 reference.
    Rows are zero-padded to a multiple of ``chunk``. Returns [B, B]. Differentiable.
    """
    n, nb = w_x.shape
    pad = (-n) % chunk
    if pad:
        z = torch.zeros(pad, nb, dtype=w_x.dtype, device=w_x.device)
        w_x = torch.cat([w_x, z]); w_y = torch.cat([w_y, z])
    a = w_x.view(-1, chunk, nb).transpose(1, 2)          # [M, B, chunk]
    b = w_y.view(-1, chunk, nb)                          # [M, chunk, B]
    return torch.bmm(a, b).sum(dim=0)                    # [B, B]


def parzen_weights(v: torch.Tensor, num_bins: int = 32, min_val: float = -1.0, max_val: float = 1.0, pad: float = 2.0) -> torch.Tensor:
    """Cubic B-spline Parzen weights of intensity samples, shape [N, num_bins], float32.

    ``v`` (any shape, flattened) is clamped to [min_val, max_val] and mapped linearly to bin
    coordinates [pad, num_bins - 1 - pad]; row i holds B3(u_i - k) for bins k = 0 .. num_bins - 1.
    With pad = 2 every sample's full spline support lies inside the bins, so each row sums to 1;
    rows of non-finite samples are zero (they add nothing to the histogram). Cache the result for
    an image whose samples do not change.
    """
    v = v.float().reshape(-1)
    finite = torch.isfinite(v)
    v = torch.clamp(torch.nan_to_num(v, nan=0.0), min_val, max_val)
    u_min, u_max = pad, float(num_bins - 1) - pad
    scale = (u_max - u_min) / (max_val - min_val)
    bins = torch.arange(num_bins, device=v.device, dtype=torch.float32).unsqueeze(0)
    return b_spline_3(u_min + (v.view(-1, 1) - min_val) * scale - bins) * finite.unsqueeze(1).float()


def mattes_mi_from_weights(w_x: torch.Tensor, w_y: torch.Tensor) -> torch.Tensor:
    """Negative mutual information from two [N, B] Parzen weight matrices (same N).

    Joint histogram H = w_x^T w_y (``_parzen_joint_histogram``), p_xy = H / sum(H), marginals by
    row / column sums; returns -sum p_xy log(p_xy / (p_x p_y)) (in nats, 1e-8 guards) as a
    scalar tensor, differentiable in both inputs.
    """
    joint_hist = _parzen_joint_histogram(w_x, w_y)
    pxy = joint_hist / (joint_hist.sum() + 1e-8)
    px = pxy.sum(dim=1, keepdim=True)
    py = pxy.sum(dim=0, keepdim=True)
    ratio = pxy / (px * py + 1e-8)
    return -torch.sum(pxy * torch.log(torch.clamp(ratio, min=1e-8)))


def mattes_mi_loss_core(I, J, mask=None, num_bins=32, min_val=-1.0, max_val=1.0, sampling_percentage=None,
                        fixed_weights: torch.Tensor = None):
    """
    Negative Mattes mutual information of intensities already scaled to [min_val, max_val].

    Both images get cubic B-spline Parzen weights (``parzen_weights``, pad = 2 bins, so every
    sample's weights sum to 1); the joint histogram is accumulated with
    ``_parzen_joint_histogram`` (blocked, deterministic). Autocast is disabled for this part.

    Parameters
    ----------
    I, J : torch.Tensor
        Samples of the moving (I) and fixed (J) image, any matching shape. Values outside
        [min_val, max_val] are clamped into the edge bins; non-finite samples get zero weight.
    mask : torch.Tensor, optional
        Boolean / float mask of the same shape; voxels with mask > 0.5 are used. None: all.
    num_bins : int, default 32
    min_val, max_val : float, default -1, 1
        Histogram range of both images.
    sampling_percentage : float, optional
        If < 1, a regular subsample of round(p * N) of the flattened, masked samples (evenly
        spaced indices, deterministic).
    fixed_weights : torch.Tensor, optional
        Pre-computed ``parzen_weights`` [N, num_bins] of J's (masked, subsampled) samples; N must
        equal the sample count (ValueError otherwise).

    Returns
    -------
    torch.Tensor
        Scalar -MI (nats). With no samples, zero (still connected to I and J, zero gradient).
    """
    if mask is not None:
        valid = mask > 0.5
        x = I[valid]
        y = J[valid]
    else:
        x = I.flatten()
        y = J.flatten()

    if sampling_percentage is not None and sampling_percentage < 1.0 and x.numel() > 0:
        k = max(1, int(round(float(sampling_percentage) * x.numel())))
        idx = torch.linspace(0, x.numel() - 1, k, device=x.device).round().long()
        x = x[idx]
        y = y[idx]

    if x.numel() == 0:
        return I.sum() * 0.0 + J.sum() * 0.0
    if fixed_weights is not None and fixed_weights.shape[0] != x.numel():
        raise ValueError(f"fixed_weights has {fixed_weights.shape[0]} rows for {x.numel()} samples")

    # Disable AMP autocast specifically for joint histogram accumulation and entropy
    # to prevent float16 overflow (max 65504) in N-voxel sum and matmul
    dev_type = 'cuda' if I.is_cuda else ('mps' if I.device.type == 'mps' else 'cpu')
    with torch.amp.autocast(device_type=dev_type, enabled=False):
        w_x = parzen_weights(x, num_bins, min_val, max_val)
        # J is the fixed image in registration: its weights may be supplied pre-computed
        w_y = fixed_weights if fixed_weights is not None else parzen_weights(y, num_bins, min_val, max_val)
        return mattes_mi_from_weights(w_x, w_y)


def mattes_mi_loss_nd(I, J, mask=None, num_bins=32, sampling_percentage=None, auto_mask=True,
                      fixed_range=None, fixed_weights=None):
    """
    Negative Mattes mutual information of images I (moving) and J (fixed), any dimension.

    Selects the voxels to use, scales each image's samples to [-1, 1] with its histogram bounds,
    and calls ``mattes_mi_loss_core`` (cubic B-spline Parzen windows).

    Parameters
    ----------
    I, J : torch.Tensor
        Images of the same shape (any layout; only voxel-wise pairing matters).
    mask : torch.Tensor, optional
        Same shape; voxels with mask > 0.5 are used.
    num_bins : int, default 32
    sampling_percentage : float, optional
        Regular subsampling of round(p * N) samples, see ``mattes_mi_loss_core``.
    auto_mask : bool, default True
        Also restrict to the foreground |I| > 1 % of max |I| OR |J| > 1 % of max |J| (relative,
        so independent of the intensity scale; ANDed with ``mask`` if given). The set changes as
        I is warped.
    fixed_range : None | (min, max) | ((min_I, max_I), (min_J, max_J))
        Histogram bounds. None recomputes them from the current selected voxels of each image
        (detached), so the axes move with the transform and MI values are not comparable across
        candidates. Fixed bounds (ITK behaviour) for optimisation / candidate scoring, e.g.
        ``(0.0, 1.0)`` for foreground-normalised images; (min, max) applies to both images and
        must be Python numbers. Samples outside the bounds fall in the edge bins.
    fixed_weights : torch.Tensor [N, num_bins], optional
        Pre-computed ``parzen_weights`` of J's selected, subsampled, scaled samples. Requires
        ``fixed_range`` and ``auto_mask=False`` (otherwise the sample set changes with I) and N
        equal to the sample count; ValueError otherwise. Halves the cost.

    Returns
    -------
    torch.Tensor
        Scalar -MI (nats). With no selected voxels, zero (still connected to I and J).
    """
    if fixed_weights is not None and (fixed_range is None or auto_mask):
        raise ValueError("fixed_weights needs fixed_range and auto_mask=False (a fixed sample set)")
    if auto_mask:
        fg_mask = (I.abs() > 0.01 * I.detach().abs().max()) | (J.abs() > 0.01 * J.detach().abs().max())
        if mask is not None:
            mask = (mask > 0.5) & fg_mask
        else:
            mask = fg_mask

    if mask is not None:
        valid = mask > 0.5
        x = I[valid]
        y = J[valid]
    else:
        x = I.flatten()
        y = J.flatten()

    if x.numel() == 0:
        return I.sum() * 0.0 + J.sum() * 0.0

    if fixed_range is None:
        min_i, max_i = x.min().detach(), x.max().detach()
        min_j, max_j = y.min().detach(), y.max().detach()
    else:
        fr = fixed_range
        if isinstance(fr[0], (int, float)):
            fr = (fr, fr)
        min_i, max_i = float(fr[0][0]), float(fr[0][1])
        min_j, max_j = float(fr[1][0]), float(fr[1][1])

    x_scaled = (x - min_i) / (max_i - min_i + 1e-8) * 2.0 - 1.0
    y_scaled = (y - min_j) / (max_j - min_j + 1e-8) * 2.0 - 1.0
    
    return mattes_mi_loss_core(x_scaled, y_scaled, mask=None, num_bins=num_bins, min_val=-1.0, max_val=1.0,
                               sampling_percentage=sampling_percentage,
                               fixed_weights=fixed_weights)


def compute_soft_distance_transform(
    image: torch.Tensor,
    sigma: float = 3.0,
    threshold: Optional[float] = None,
    spacing: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Differentiable soft "distance from the object" map: D = -sigma * log(G_sigma * m).

    m is a soft foreground map, G_sigma a Gaussian (``separable_gaussian_filter``, Bessel kernel,
    replicate padding), and G_sigma * m is clamped to >= 1e-8. D is ~0 inside the object and
    grows away from it, but it is not a Euclidean distance: at distance d from a straight edge
    G_sigma * m ~ Phi(-d / sigma), so D grows roughly like d^2 / (2 sigma) far from the object
    and saturates at -sigma log(1e-8) ~ 18.4 sigma. Values are in the units of ``sigma``.

    Parameters
    ----------
    image : Tensor
        (B, C, *spatial) for 4-D / 5-D input, or (*spatial) for 2-D / 3-D input. A 3-D tensor
        with a 2-element ``spacing`` is treated as (C, H, W). Tensor order (z, y, x).
    sigma : float, default 3.0
        Gaussian width: voxels when ``spacing`` is None; else physical units, converted per axis
        to sigma / spacing voxels and clipped to [0.5, 10] voxels.
    threshold : float, optional
        m = sigmoid(10 * (image - threshold)). None: m = |image| / max|image| per (B, C).
    spacing : sequence of float, optional
        Voxel spacing in ANTs (x, y, z) order.

    Returns
    -------
    Tensor of the input shape, differentiable with respect to ``image``.

    Raises
    ------
    ValueError
        If ``image`` is not 2-D to 5-D.
    """
    orig_shape = image.shape
    if image.dim() == 2:
        img_nd = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        if spacing is not None and len(spacing) == 2:
            img_nd = image.unsqueeze(0)
        else:
            img_nd = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() in (4, 5):
        img_nd = image
    else:
        raise ValueError(f"Unsupported image dimension: {image.dim()}")

    if threshold is not None:
        mask = torch.sigmoid((img_nd - threshold) * 10.0)
    else:
        max_val = torch.amax(img_nd.abs(), dim=tuple(range(2, img_nd.dim())), keepdim=True).clamp_min(1e-6)
        mask = torch.clamp(img_nd.abs() / max_val, 0.0, 1.0)

    from .smoothing import separable_gaussian_filter
    mask_last = mask.movedim(1, -1)
    if spacing is not None:
        smoothed_last = separable_gaussian_filter(mask_last, sigma=sigma, spacing=spacing, sigma_mode='physical')
    else:
        smoothed_last = separable_gaussian_filter(mask_last, sigma=sigma)
    smoothed = smoothed_last.movedim(-1, 1).clamp_min(1e-8)

    dist = -float(sigma) * torch.log(smoothed)
    return dist.view(orig_shape)


def _soft_signed_distance(image: torch.Tensor, sigma: float, threshold: Optional[float] = None,
                          spacing: Optional[Sequence[float]] = None) -> torch.Tensor:
    """Differentiable soft signed distance: ``compute_soft_distance_transform`` of the object
    minus that of its complement (negative inside, positive outside, ~0 on the boundary)."""
    if threshold is not None:
        comp = 2.0 * threshold - image             # m(comp) = sigmoid(10 (threshold - image)) = 1 - m
    else:
        max_val = torch.amax(image.abs(), dim=tuple(range(2, image.dim())), keepdim=True) if image.dim() >= 4 \
            else image.abs().max()
        comp = max_val - image.abs()               # m(comp) = 1 - m
    return (compute_soft_distance_transform(image, sigma=sigma, threshold=threshold, spacing=spacing)
            - compute_soft_distance_transform(comp, sigma=sigma, threshold=threshold, spacing=spacing))


def compute_image_distance_transform(
    image: Union[torch.Tensor, np.ndarray, Any],
    threshold: Optional[float] = None,
    tau: Optional[float] = None,
    signed: bool = False,
    sampling_spacing: Optional[Sequence[float]] = None,
    return_ants: bool = False,
) -> Union[torch.Tensor, Any]:
    """Exact Euclidean distance transform (SciPy ``distance_transform_edt``) of a foreground,
    optionally turned into the potential exp(-|D| / tau). Not differentiable (NumPy, CPU).

    Each (B, C) slice is binarised; D is the distance of each voxel to the nearest foreground
    voxel (0 on the foreground), in the units of the spacing (voxels if none). An empty
    foreground gives the grid diagonal length everywhere; an all-foreground slice gives 0
    (with ``signed``: minus the diagonal).

    Parameters
    ----------
    image : ANTsImage, Tensor or array-like
        Tensor / array: (B, C, *spatial) for 4-D / 5-D, (*spatial) for 2-D / 3-D, in tensor
        order (z, y, x); a 3-D input with a 2-element ``sampling_spacing`` is treated as
        (C, H, W). ANTsImage: its ``.numpy()`` array, in ANTs (x, y, z) order, as one slice.
    threshold : float, optional
        Foreground = value > threshold. None: |value| > 1e-4 * max|value| of the slice.
    tau : float, optional
        If given (must be > 0, else ValueError), return exp(-|D| / tau) instead of D.
    signed : bool, default False
        D = distance outside - distance inside: negative inside (foreground voxels next to the
        boundary get -1 voxel, not 0), positive outside.
    sampling_spacing : sequence of float, optional
        Voxel spacing in ANTs (x, y, z) order (reversed internally for tensors). For an
        ANTsImage, defaults to ``image.spacing``.
    return_ants : bool, default False
        For ANTsImage input only: return an ANTsImage with the input's origin / spacing /
        direction. Ignored for other inputs.

    Returns
    -------
    torch.Tensor or ANTsImage
        Same shape as the input, on the input tensor's device, in its dtype if floating point
        else float32. ANTsImage input without
        ``return_ants``: a CPU float32 tensor in ANTs (x, y, z) array order.

    Raises
    ------
    ValueError
        Unsupported dimensionality, or ``tau <= 0``.
    """
    is_ants = hasattr(image, 'spacing') and hasattr(image, 'numpy')
    ref_image = image if is_ants else None

    if is_ants:
        if sampling_spacing is None:
            sampling_spacing = tuple(float(s) for s in image.spacing)
        img_np_raw = image.numpy().astype(np.float32)
        device = torch.device('cpu')
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

        # In PyTorch tensors, spatial dimensions are ordered (Z, Y, X) or (Y, X).
        # ITK sampling_spacing is ordered (sx, sy, sz) or (sx, sy).
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
                diag_span = float(np.sqrt(sum((dim_sz * (sp if sp else 1.0))**2 for dim_sz, sp in zip(spatial_shape, sampling or [1.0]*len(spatial_shape)))))
                edt = np.ones_like(sl, dtype=np.float32) * diag_span
            elif np.all(fg):
                # no background: 0 unsigned; signed = -(grid diagonal), mirroring the empty case
                diag_span = float(np.sqrt(sum((dim_sz * (sp if sp else 1.0))**2 for dim_sz, sp in zip(spatial_shape, sampling or [1.0]*len(spatial_shape)))))
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
    mode: Literal['potential_lncc', 'potential_mse', 'edt_mse', 'edt_l1', 'sdf_mse'] = 'potential_lncc',
    tau: float = 0.10,
    window_size: int = 9,
    mask: Optional[torch.Tensor] = None,
    is_distance_field: bool = False,
    threshold: Optional[float] = None,
    spacing: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Loss comparing two images through distance maps (or distance potentials).

    If ``is_distance_field`` is False, I and J are first converted with
    ``compute_soft_distance_transform(sigma=10 * tau, threshold, spacing)`` (signed variant for
    'sdf_mse'). That map is not a Euclidean distance (see its docstring); with it the potential
    exp(-D / tau) equals (G_sigma * m) ** 10. If True, I and J are used as given (e.g. from
    ``compute_image_distance_transform``) and potentials use |D|.

    Parameters
    ----------
    I, J : Tensor (B, C, *spatial)
        Images / segmentations / distance fields of the same shape.
    mode : str, default 'potential_lncc'
        - 'potential_lncc': ``local_ncc_loss_nd`` (CC, autograd) of the potentials.
        - 'potential_mse': mean squared difference of the potentials.
        - 'edt_mse': mean squared difference of the distance maps.
        - 'sdf_mse': mean squared difference of soft *signed* distance maps (``_soft_signed_
          distance``: object minus complement) when ``is_distance_field`` is False; with
          distance fields given, the same as 'edt_mse' on them (pass signed fields).
        - 'edt_l1': mean absolute difference of the distance maps.
        Any other value raises ValueError.
    tau : float, default 0.10
        Potential decay length, in the units of the distance maps; must be > 0 (ValueError).
        Also sets the soft-transform sigma (10 * tau).
    window_size : int, default 9
        LNCC window ('potential_lncc' only).
    mask : Tensor, optional
        Broadcastable to I. 'potential_lncc': passed to ``local_ncc_loss_nd``; other modes:
        weighted mean sum(err * mask) / sum(mask).
    is_distance_field : bool, default False
        See above.
    threshold : float, optional
        Soft binarisation threshold for ``compute_soft_distance_transform``; only used when
        ``is_distance_field`` is False.
    spacing : sequence of float, optional
        Voxel spacing in ANTs (x, y, z) order for the soft transform (sigma then in physical
        units); only used when ``is_distance_field`` is False.

    Returns
    -------
    torch.Tensor
        Scalar loss (lower is better).
    """
    if tau is None or not float(tau) > 0:
        raise ValueError(f"distance_transform_loss: tau must be > 0, got {tau!r}")
    if not is_distance_field and mode == 'sdf_mse':
        D_I = _soft_signed_distance(I, sigma=tau * 10.0, threshold=threshold, spacing=spacing)
        D_J = _soft_signed_distance(J, sigma=tau * 10.0, threshold=threshold, spacing=spacing)
    elif not is_distance_field:
        D_I = compute_soft_distance_transform(I, sigma=tau * 10.0, threshold=threshold, spacing=spacing)
        D_J = compute_soft_distance_transform(J, sigma=tau * 10.0, threshold=threshold, spacing=spacing)
    else:
        D_I = I
        D_J = J

    if mode == 'potential_lncc':
        P_I = torch.exp(-torch.abs(D_I) / tau) if is_distance_field else torch.exp(-D_I / tau)
        P_J = torch.exp(-torch.abs(D_J) / tau) if is_distance_field else torch.exp(-D_J / tau)
        return local_ncc_loss_nd(P_I, P_J, mask=mask, window_size=window_size)
    elif mode == 'potential_mse':
        P_I = torch.exp(-torch.abs(D_I) / tau) if is_distance_field else torch.exp(-D_I / tau)
        P_J = torch.exp(-torch.abs(D_J) / tau) if is_distance_field else torch.exp(-D_J / tau)
        if mask is not None:
            return torch.sum(((P_I - P_J) ** 2) * mask) / (mask.sum() + 1e-8)
        return F.mse_loss(P_I, P_J)
    elif mode in ('edt_mse', 'sdf_mse'):
        if mask is not None:
            return torch.sum(((D_I - D_J) ** 2) * mask) / (mask.sum() + 1e-8)
        return F.mse_loss(D_I, D_J)
    elif mode == 'edt_l1':
        if mask is not None:
            return torch.sum(torch.abs(D_I - D_J) * mask) / (mask.sum() + 1e-8)
        return F.l1_loss(D_I, D_J)
    else:
        raise ValueError(f"Unknown distance transform loss mode: '{mode}'")


def soft_dice_loss_nd(
    I: torch.Tensor,
    J: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6
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
        Moving / deformed maps. The loss is symmetric in I and J.
    mask : Tensor, optional
        (B, 1, *spatial) or (B, C, *spatial); I and J are multiplied by it first.
    eps : float, default 1e-6
        Added to the denominator (an empty channel gives dice 0).

    Returns
    -------
    torch.Tensor
        Scalar in [0, 1] for non-negative inputs; 0 = perfect overlap.
    """
    if mask is not None:
        I = I * mask
        J = J * mask

    spatial_dims = tuple(range(2, I.ndim))
    intersection = 2.0 * torch.sum(I * J, dim=spatial_dims)
    cardinality = torch.sum(I ** 2 + J ** 2, dim=spatial_dims) + eps

    dice_per_channel = intersection / cardinality
    return 1.0 - torch.mean(dice_per_channel)


