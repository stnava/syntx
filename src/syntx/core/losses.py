import torch
import torch.nn.functional as F
from typing import Optional, Union, Sequence, Literal, Any
import numpy as np
import scipy.ndimage as ndi


class AnalyticalLNCC(torch.autograd.Function):
    """Analytically-differentiated Local NCC (CC, not CC²).

    Computes forward CC = cov(I,J) / sqrt(var(I)*var(J)) identical to
    the autograd path in ``local_ncc_loss_nd(..., squared=False)``, but
    manually implements backward() so that PyTorch never builds a memory-
    heavy autograd graph through ``F.avg_pool3d``.  This makes it as fast
    as ``ANTsPseudoLNCC`` on Apple MPS while optimising the true CC loss
    landscape instead of the CC² pseudo-derivative.

    Analytical gradient of -mean(CC) w.r.t. center pixel J_c (analogous
    for I_c by symmetry):

        dCC/dJ_c = (1/N) * 1/sqrt(var_F * var_M) * (F_c - CC * M_c)

    where F_c, M_c are mean-subtracted center-pixel intensities and N is
    the window volume.
    """

    @staticmethod
    def forward(ctx, I, J, mask, window_size):
        dim = I.dim() - 2
        pad = window_size // 2
        N_window = window_size ** dim

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

        ctx.save_for_backward(F_centered, M_centered, cc, safe_I_var, safe_J_var, mask)
        ctx.N_window = N_window

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
        F_centered, M_centered, cc, safe_I_var, safe_J_var, mask = ctx.saved_tensors

        inv_denom = 1.0 / (torch.sqrt(safe_I_var * safe_J_var) + 1e-6)

        # Analytical derivative of CC w.r.t. center pixel:
        #   dCC/dJ_c = (1/N) / sqrt(sFF * sMM) * (F_c - CC * M_c)
        #   dCC/dI_c = (1/N) / sqrt(sFF * sMM) * (M_c - CC * F_c)
        # Loss is -CC, so negate:
        scale = -(1.0 / ctx.N_window) * inv_denom

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
    @staticmethod
    def forward(ctx, I, J, mask, window_size):
        dim = I.dim() - 2
        pad = window_size // 2
        N_window = window_size ** dim
        
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
        
        ctx.save_for_backward(F_centered, M_centered, IJ_cov, safe_I_var, safe_J_var, mask)
        ctx.N_window = N_window
        
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
        F_centered, M_centered, IJ_cov, safe_I_var, safe_J_var, mask = ctx.saved_tensors
        
        s_FM = IJ_cov
        s_FF = safe_I_var
        s_MM = safe_J_var
        
        sFF_sMM = s_FF * s_MM + 1e-8
        
        # ITK's pseudo-derivative for +CC^2 wrt moving center pixel M_c is:
        # 2/N * cov / (var_F * var_M) * (F_c - cov / var_M * M_c)
        # By symmetry, wrt fixed center pixel F_c is:
        # 2/N * cov / (var_F * var_M) * (M_c - cov / var_F * F_c)
        
        # Since our loss is -CC^2, the gradient of the loss is the negative of this.
        grad_factor = -2.0 * (1.0 / ctx.N_window) * (s_FM / sFF_sMM)
        
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
    r"""
    Computes Local Normalized Cross-Correlation (LNCC) Loss between N-D images $I$ and $J$.

    Formulation & Rule Guardrails (GEMINI.md Rule 2):
    - Sliding Box Filter: Evaluates local mean $\mu_I, \mu_J$, local variance $\text{Var}(I), \text{Var}(J)$,
      and covariance $\text{Cov}(I, J)$ over a window of size `window_size`.
    - Variance Floor (Singularity Prevention): Enforces a variance floor $\text{Var}_{\text{safe}}(I) = \max(\text{Var}(I), 10^{-6})$
      to prevent $\frac{1}{\text{Var}(I)}$ analytical autograd derivative spikes in flat intensity or zero-padded background regions.
    - Cauchy-Schwarz $[-1.0, 1.0]$ Clamping: Enforces strictly bounded correlation coefficient
      $\text{CC} = \text{clamp}\left(\frac{\text{Cov}(I, J)}{\sqrt{\text{Var}_{\text{safe}}(I) \text{Var}_{\text{safe}}(J)}}, -1.0, 1.0\right)$
      to eliminate 32-bit floating-point roundoff overflow near sharp boundary edges.

    Parameters
    ----------
    I : torch.Tensor
        First image tensor of shape `(B, 1, *spatial)`.
    J : torch.Tensor
        Second image tensor of shape `(B, 1, *spatial)`.
    mask : torch.Tensor, optional
        Binary mask tensor of shape `(B, 1, *spatial)` identifying active evaluation voxels.
    window_size : int, default=9
        Sliding box-filter window size in voxels.
    use_ants_pseudo_gradient : bool, default=False
        If True, uses ANTs C++ style analytical pseudo-gradient autograd function (`ANTsPseudoLNCC`). Implicitly optimizes CC^2.
    squared : bool, default=False
        If True, optimizes squared LNCC (CC^2) instead of CC. This acts as a multi-modal metric.

    Returns
    -------
    torch.Tensor
        Scalar negative LNCC loss tensor (range `[-1.0, 0.0]`, where `-1.0` indicates perfect alignment).
    """
    device = I.device
    dim = I.dim() - 2
    
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
        # AnalyticalLNCC optimizes true CC with exact analytical gradients.
        # Same speed as ANTsPseudoLNCC (no autograd graph through avg_pool3d).
        return AnalyticalLNCC.apply(I, J, mask, window_size)
            
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
    Native sliding box-filter squared zero-normalized cross-correlation loss.
    
    Features dual variance floors (smooth_nr=1e-5, smooth_dr=1e-5):
    In zero-padded or uniform background regions, this evaluates to:
        (0 + 1e-5) / (0 + 1e-5) = 1.0
    treating flat background as perfectly correlated and completely suppressing
    peripheral boundary gradient artifacts at image edges.
    """
    def __init__(self, kernel_size: int = 5, smooth_nr: float = 1e-5, smooth_dr: float = 1e-5, squared: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.squared = squared
        self._target_cache = {}
        self._target_refs = {}

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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
) -> torch.Tensor:
    """Functional interface for BoxLNCCLoss."""
    loss_fn = BoxLNCCLoss(kernel_size=window_size, smooth_nr=smooth_nr, smooth_dr=smooth_dr)
    return loss_fn(I, J)


def b_spline_3(x):
    """3rd-order B-spline kernel for Parzen windowing.
    
    Evaluates cubic B-spline polynomials using direct multiplications
    to avoid slow pow kernels on GPU/MPS.
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
    produced different transforms on every first call). Blocked bmm with K = 4096 is bitwise
    deterministic across allocations and ~30x closer to a float64 reference.  Differentiable.
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
    """Cubic B-spline Parzen weights [N, num_bins] of a flat intensity vector scaled to [min_val, max_val]
    (boundary-padded partition of unity).  Cache this for an image whose samples do not change."""
    v = torch.nan_to_num(torch.clamp(v.float(), min_val, max_val), nan=0.0)
    u_min, u_max = pad, float(num_bins - 1) - pad
    scale = (u_max - u_min) / (max_val - min_val)
    bins = torch.arange(num_bins, device=v.device, dtype=torch.float32).unsqueeze(0)
    return b_spline_3(u_min + (v.view(-1, 1) - min_val) * scale - bins)


def mattes_mi_from_weights(w_x: torch.Tensor, w_y: torch.Tensor) -> torch.Tensor:
    """Negative Mattes MI from Parzen weight matrices (deterministic blocked histogram)."""
    joint_hist = _parzen_joint_histogram(w_x, w_y)
    pxy = joint_hist / (joint_hist.sum() + 1e-8)
    px = pxy.sum(dim=1, keepdim=True)
    py = pxy.sum(dim=0, keepdim=True)
    ratio = pxy / (px * py + 1e-8)
    return -torch.sum(pxy * torch.log(torch.clamp(ratio, min=1e-8)))


def mattes_mi_loss_core(I, J, mask=None, num_bins=32, min_val=-1.0, max_val=1.0, sampling_percentage=None,
                        fixed_weights: torch.Tensor = None):
    """
    Differentiable Mattes Mutual Information (Parzen window using 3rd-order B-spline).
    Returns Negative Mutual Information (for minimization).

    Enforces strict partition of unity by padding boundary bins (pad=2.0 bins),
    preventing artificial boundary forces and gradient spikes.  The joint histogram is
    accumulated with ``_parzen_joint_histogram`` (deterministic on MPS/CUDA/CPU).
    """
    if mask is not None:
        valid = mask > 0.5
        x = I[valid]
        y = J[valid]
    else:
        x = I.flatten()
        y = J.flatten()

    if sampling_percentage is not None and sampling_percentage < 1.0:
        stride = max(1, int(1.0 / sampling_percentage))
        # .contiguous(): strided views feed MPS kernels inconsistently; make the sample explicit
        x = x[::stride].contiguous()
        y = y[::stride].contiguous()

    if x.numel() == 0:
        return torch.tensor(0.0, device=I.device, requires_grad=True)

    # Disable AMP autocast specifically for joint histogram accumulation and entropy
    # to prevent float16 overflow (max 65504) in N-voxel sum and matmul
    dev_type = 'cuda' if I.is_cuda else ('mps' if I.device.type == 'mps' else 'cpu')
    with torch.amp.autocast(device_type=dev_type, enabled=False):
        w_x = parzen_weights(x, num_bins, min_val, max_val)
        # J is the fixed image in registration: its weights may be supplied pre-computed
        w_y = fixed_weights if (fixed_weights is not None and fixed_weights.shape[0] == w_x.shape[0]) \
            else parzen_weights(y, num_bins, min_val, max_val)
        return mattes_mi_from_weights(w_x, w_y)


def mattes_mi_loss_nd(I, J, mask=None, num_bins=32, sampling_percentage=None, auto_mask=True,
                      fixed_range=None, fixed_weights=None):
    """
    N-dimensional Mattes Mutual Information loss wrapper.
    Scale images to [-1, 1] internally.
    Extracts foreground voxels prior to scaling to conserve memory and preserve
    tissue dynamic range across histogram bins.

    fixed_range : None | ((min_I, max_I), (min_J, max_J)) | (min, max)
        Intensity bounds used for the histogram axes.  Default (None) recomputes the
        bounds from the *current* masked voxels of each image, so the axes move with
        the transform (the warped image's range changes) and MI values are not
        comparable across candidates.  Pass fixed bounds (ITK behaviour) for
        optimisation / candidate scoring, e.g. ``fixed_range=(0.0, 1.0)`` for
        foreground-normalised images.
    fixed_weights : Tensor [N, num_bins], optional
        Pre-computed ``parzen_weights`` of the (masked, subsampled, scaled) fixed image J —
        valid only with ``fixed_range`` and an unchanging fixed sample; halves the cost.
    """
    if auto_mask:
        fg_mask = (I.abs() > 0.01) | (J.abs() > 0.01)
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
        return torch.tensor(0.0, device=I.device, requires_grad=True)

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
                               fixed_weights=fixed_weights if fixed_range is not None else None)


def compute_soft_distance_transform(
    image: torch.Tensor,
    sigma: float = 3.0,
    threshold: Optional[float] = None,
    spacing: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Compute soft differentiable distance transform using heat diffusion.

    Approximates the Euclidean distance transform differentiably in PyTorch using
    the heat method D_sigma(x) = -sigma * log(G_sigma * mask). When physical spacing
    is provided, the diffusion filter scales by physical voxel spacing, yielding
    distances scaled in physical millimeter space.

    Parameters
    ----------
    image : Tensor of shape (B, C, *spatial) or (*spatial)
        Input image or segmentation.
    sigma : float, default 3.0
        Diffusion scale in physical units (mm) if spacing is provided, or voxel units otherwise.
    threshold : float, optional
        Foreground binarization threshold.
    spacing : sequence of float, optional
        Physical voxel spacing in ITK (sx, sy, sz) convention.

    Returns
    -------
    Tensor of shape matching input
        Smooth, autograd-differentiable distance field in physical units (mm).
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


def compute_image_distance_transform(
    image: Union[torch.Tensor, np.ndarray, Any],
    threshold: Optional[float] = None,
    tau: Optional[float] = None,
    signed: bool = False,
    sampling_spacing: Optional[Sequence[float]] = None,
    return_ants: bool = False,
) -> Union[torch.Tensor, Any]:
    """Compute exact Euclidean Distance Transform (or smooth potential) from an image or mask.

    Computes distances in physical space (mm). If image is an ANTsImage, its physical spacing
    is automatically extracted and respected.

    Parameters
    ----------
    image : ANTsImage, Tensor, or ndarray of shape (B, C, *spatial), (B, 1, *spatial), or (*spatial)
        Input image, segmentation, or edge map.
    threshold : float, optional
        Foreground binarization threshold. If None, uses non-zero voxels.
    tau : float, optional
        Bandwidth for exponential potential P(x) = exp(-D(x) / tau). If None, returns raw distance.
    signed : bool, default False
        If True, returns signed distance transform (negative inside, positive outside).
    sampling_spacing : sequence of float, optional
        Physical voxel spacing in ITK convention (sx, sy, sz).
        If image is an ANTsImage and sampling_spacing is None, image.spacing is used automatically.
    return_ants : bool, default False
        If True and input is an ANTsImage, returns an ANTsImage with matching geometry.

    Returns
    -------
    torch.Tensor or ANTsImage
        Distance transform in physical units (mm).
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
        dtype = image_t.dtype
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
                edt = np.zeros_like(sl, dtype=np.float32)
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
    """General Distance Transform Similarity Loss in Physical Space.

    Computes distance transform alignment between two images, segmentation masks,
    or precomputed distance fields in physical space (mm). Applicable across any
    registration model in syntx.

    Parameters
    ----------
    I : Tensor of shape (B, C, *spatial)
        First input image, segmentation, or distance field.
    J : Tensor of shape (B, C, *spatial)
        Second input image, segmentation, or distance field.
    mode : {'potential_lncc', 'potential_mse', 'edt_mse', 'edt_l1', 'sdf_mse'}, default 'potential_lncc'
        Loss formulation:
        - 'potential_lncc': Negative LNCC on exponential distance potentials exp(-D / tau).
        - 'potential_mse': Mean squared error on exponential distance potentials.
        - 'edt_mse': Mean squared error on Euclidean distance fields.
        - 'edt_l1': L1 absolute error on Euclidean distance fields.
        - 'sdf_mse': Mean squared error on signed distance fields.
    tau : float, default 0.10
        Decay bandwidth for exponential potentials in physical units (mm).
    window_size : int, default 9
        Window size for LNCC when mode='potential_lncc'.
    mask : Tensor, optional
        Spatial domain mask.
    is_distance_field : bool, default False
        If True, I and J are already distance fields. If False, computes soft distance transforms.
    threshold : float, optional
        Threshold for binarizing I and J when is_distance_field=False.
    spacing : sequence of float, optional
        Physical voxel spacing in ITK (sx, sy, sz) convention.

    Returns
    -------
    torch.Tensor
        Scalar similarity loss.
    """
    if not is_distance_field:
        D_I = compute_soft_distance_transform(I, sigma=tau * 10.0 if tau else 3.0, threshold=threshold, spacing=spacing)
        D_J = compute_soft_distance_transform(J, sigma=tau * 10.0 if tau else 3.0, threshold=threshold, spacing=spacing)
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
    Computes smooth, differentiable Soft Dice loss between continuous probability
    or membership tensors I and J in 2D or 3D. Supports single or multi-channel tensors.

    Parameters
    ----------
    I : Tensor of shape (B, C, *spatial)
        Fixed target probability / membership tensor.
    J : Tensor of shape (B, C, *spatial)
        Moving / deformed probability / membership tensor.
    mask : Tensor, optional
        Spatial domain foreground mask of shape (B, 1, *spatial) or (B, C, *spatial).
    eps : float, default 1e-6
        Safety epsilon to prevent division by zero.

    Returns
    -------
    torch.Tensor
        Scalar Soft Dice loss (1.0 - mean_dice).
    """
    if mask is not None:
        I = I * mask
        J = J * mask

    spatial_dims = tuple(range(2, I.ndim))
    intersection = 2.0 * torch.sum(I * J, dim=spatial_dims)
    cardinality = torch.sum(I ** 2 + J ** 2, dim=spatial_dims) + eps

    dice_per_channel = intersection / cardinality
    return 1.0 - torch.mean(dice_per_channel)


