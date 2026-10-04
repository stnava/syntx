"""
Continuum-mechanics, fluid-dynamics and incompressibility-inspired regularization operators.

This module provides a unified, plug-and-play regularization system for velocity and displacement
fields in syntx. In addition to classical Sobolev and Gaussian smoothing, it implements:

1. Solenoidal / Leray Projector (``project_solenoidal``):
   Exact orthogonal projection onto divergence-free vector fields (div v = 0) in Fourier space,
   guaranteeing det(J) = 1.0000 and 0.0000% coordinate folding under Liouville flow integration.
2. Solenoidal Sobolev Green's Operator (``apply_solenoidal_sobolev_operator``):
   Combines Sobolev spatial smoothing (H^s) with exact divergence-free solenoidal projection in a
   single vectorized spectral multiplication.
3. Decoupled Div-Curl / Helmholtz Green's Operator (``apply_div_curl_green_operator``):
   Independently parameterizes resistance to volumetric dilatation (div v) and rotational shear
   (curl v), allowing sulcal banks to slide with minimal shear penalty while heavily penalizing
   volume changes.
4. Navier-Cauchy / Stokes Continuum Elastic Operator (``apply_navier_green_operator``):
   Continuum mechanics Green's operator parameterized by Poisson's ratio nu. Smoothly approaches
   the incompressible Stokes limit as nu -> 0.5.
5. Mask-Gated / Spatially-Varying Incompressibility (``apply_masked_incompressible_filter``):
   Enforces near-zero divergence within soft anatomical parenchyma (brain tissue, myocardium)
   while leaving fluid cavities (ventricles, CSF spaces) free to expand or contract.
6. Unified Registry (``get_regularizer``, ``register_regularizer``, ``list_regularizers``):
   Plug-and-play factory mapping regularizer names to callables that accept channel-last fields.
"""

from __future__ import annotations

import collections
import math
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Spectral grid cache for rfftn
# -----------------------------------------------------------------------------
_K_MESH_CACHE: collections.OrderedDict = collections.OrderedDict()
_K_MESH_LOCK = threading.Lock()
_MAX_K_MESH_CACHE_SIZE = 64


def _get_k_mesh_rfft(
    spatial_shape: Sequence[int],
    spacing: Optional[Sequence[float]],
    device: torch.device,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Cached frequency coordinate mesh on the rfftn grid.

    Parameters
    ----------
    spatial_shape : sequence of int
        Spatial shape in tensor order (H, W) or (D, H, W).
    spacing : sequence of float, optional
        Voxel spacing in ANTs (x, y, z) order (reversed internally to tensor order).
    device : torch.device

    Returns
    -------
    k_mesh : list of Tensor
        List of coordinate tensors [k_0, ..., k_{dim-1}] on the rfftn grid.
    k_sq : Tensor
        Squared frequency norm sum(k_d^2).
    k_sq_safe : Tensor
        k_sq with 0.0 replaced by 1.0 to prevent division by zero at DC.
    """
    dim = len(spatial_shape)
    sp_tuple = tuple(float(x) for x in spacing) if spacing is not None else None
    cache_key = (tuple(spatial_shape), sp_tuple, str(device))

    with _K_MESH_LOCK:
        if cache_key in _K_MESH_CACHE:
            _K_MESH_CACHE.move_to_end(cache_key)
            return _K_MESH_CACHE[cache_key]

    spacing_zyx = tuple(reversed(spacing)) if spacing is not None else (1.0,) * dim
    k_axes = []
    for d in range(dim):
        n_d = spatial_shape[d]
        sp_d = float(spacing_zyx[d]) if d < len(spacing_zyx) else 1.0
        if d == dim - 1:
            k_d = torch.fft.rfftfreq(n_d, d=max(sp_d, 1e-6), device=device) * (2.0 * math.pi)
            if n_d % 2 == 0:
                k_d[-1] = 0.0
        else:
            k_d = torch.fft.fftfreq(n_d, d=max(sp_d, 1e-6), device=device) * (2.0 * math.pi)
            if n_d % 2 == 0:
                k_d[n_d // 2] = 0.0
        k_axes.append(k_d)

    k_mesh = list(torch.meshgrid(*k_axes, indexing='ij'))
    k_sq = torch.zeros_like(k_mesh[0])
    for d in range(dim):
        k_sq = k_sq + k_mesh[d] ** 2

    mask_zero = (k_sq == 0.0)
    k_sq_safe = torch.where(mask_zero, torch.ones_like(k_sq), k_sq)

    result = (k_mesh, k_sq, k_sq_safe)
    with _K_MESH_LOCK:
        _K_MESH_CACHE[cache_key] = result
        if len(_K_MESH_CACHE) > _MAX_K_MESH_CACHE_SIZE:
            _K_MESH_CACHE.popitem(last=False)

    return result


def clear_k_mesh_cache() -> None:
    """Clear the frequency grid cache (thread-safe)."""
    with _K_MESH_LOCK:
        _K_MESH_CACHE.clear()


# -----------------------------------------------------------------------------
# Differential Operators (Divergence & Curl)
# -----------------------------------------------------------------------------
def compute_divergence_nd(
    v: torch.Tensor,
    spacing: Optional[Sequence[float]] = None,
    method: str = 'central',
) -> torch.Tensor:
    """
    Compute divergence div(v) = sum_d d v_d / d x_d of a vector field.

    Parameters
    ----------
    v : Tensor (B, *spatial, dim) or (*spatial, dim)
        Vector field with spatial axes and components in tensor order.
    spacing : sequence of float, optional
        Voxel spacing in ANTs (x, y, z) order.
    method : {'central', 'spectral'}, default 'central'
        - 'central': finite difference via torch.gradient with correct spacing.
        - 'spectral': exact spectral differentiation via FFT (periodic boundary).

    Returns
    -------
    Tensor (B, *spatial) or (*spatial,) of scalar divergence values.
    """
    unbatched = (v.dim() == v.shape[-1] + 1)
    if unbatched:
        v = v.unsqueeze(0)

    dim = v.shape[-1]
    spatial_shape = v.shape[1:-1]
    spacing_zyx = tuple(reversed(spacing)) if spacing is not None else [1.0] * dim

    if method == 'spectral':
        device = v.device
        v_cf = torch.movedim(v, -1, 1).to(torch.float32)
        spatial_dims = tuple(range(2, 2 + dim))
        v_fft = torch.fft.rfftn(v_cf, dim=spatial_dims)
        k_mesh, _, _ = _get_k_mesh_rfft(spatial_shape, spacing, device)

        k_dot_v = torch.zeros_like(v_fft[:, 0, ...])
        for d in range(dim):
            k_dot_v = k_dot_v + k_mesh[d] * v_fft[:, d, ...]

        div_dims = tuple(range(1, 1 + dim))
        div_val = torch.fft.irfftn(1j * k_dot_v, s=spatial_shape, dim=div_dims).to(dtype=v.dtype)
        return div_val[0] if unbatched else div_val

    # Standard central difference
    div_sum = torch.zeros(v.shape[:-1], device=v.device, dtype=v.dtype)
    for d in range(dim):
        sp_d = float(spacing_zyx[d])
        grad_d = torch.gradient(v[..., d], spacing=sp_d, dim=1 + d)[0]
        div_sum = div_sum + grad_d

    return div_sum[0] if unbatched else div_sum


def compute_curl_nd(
    v: torch.Tensor,
    spacing: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """
    Compute vorticity / curl of a 2-D or 3-D vector field.

    - In 2-D: returns scalar curl omega = d v_x / d y - d v_y / d x.
    - In 3-D: returns vector curl curl(v) in tensor order (curl_z, curl_y, curl_x).

    Parameters
    ----------
    v : Tensor (B, *spatial, dim) or (*spatial, dim)
    spacing : sequence of float, optional
        Voxel spacing in ANTs (x, y, z) order.

    Returns
    -------
    Tensor: scalar (B, *spatial) in 2-D, or vector (B, *spatial, 3) in 3-D.
    """
    unbatched = (v.dim() == v.shape[-1] + 1)
    if unbatched:
        v = v.unsqueeze(0)

    dim = v.shape[-1]
    if dim not in (2, 3):
        raise ValueError(f"compute_curl_nd supports dim=2 or dim=3, got dim={dim}")

    spacing_zyx = tuple(reversed(spacing)) if spacing is not None else [1.0] * dim

    if dim == 2:
        # Spatial axis 0 is y (step spacing_zyx[0]), comp 0 is v_y
        # Spatial axis 1 is x (step spacing_zyx[1]), comp 1 is v_x
        # curl = d v_x / d y - d v_y / d x
        dvx_dy = torch.gradient(v[..., 1], spacing=float(spacing_zyx[0]), dim=1)[0]
        dvy_dx = torch.gradient(v[..., 0], spacing=float(spacing_zyx[1]), dim=2)[0]
        curl_2d = dvx_dy - dvy_dx
        return curl_2d[0] if unbatched else curl_2d

    # dim == 3: axes 0: z, 1: y, 2: x; comps 0: v_z, 1: v_y, 2: v_x
    # curl_z = d v_x / d y - d v_y / d x = d v_2 / d x_1 - d v_1 / d x_2
    # curl_y = d v_z / d x - d v_x / d z = d v_0 / d x_2 - d v_2 / d x_0
    # curl_x = d v_y / d z - d v_z / d y = d v_1 / d x_0 - d v_0 / d x_1
    sp_z, sp_y, sp_x = float(spacing_zyx[0]), float(spacing_zyx[1]), float(spacing_zyx[2])
    dv2_dy = torch.gradient(v[..., 2], spacing=sp_y, dim=2)[0]
    dv1_dx = torch.gradient(v[..., 1], spacing=sp_x, dim=3)[0]
    curl_z = dv2_dy - dv1_dx

    dv0_dx = torch.gradient(v[..., 0], spacing=sp_x, dim=3)[0]
    dv2_dz = torch.gradient(v[..., 2], spacing=sp_z, dim=1)[0]
    curl_y = dv0_dx - dv2_dz

    dv1_dz = torch.gradient(v[..., 1], spacing=sp_z, dim=1)[0]
    dv0_dy = torch.gradient(v[..., 0], spacing=sp_y, dim=2)[0]
    curl_x = dv1_dz - dv0_dy

    curl_3d = torch.stack([curl_z, curl_y, curl_x], dim=-1)
    return curl_3d[0] if unbatched else curl_3d


# -----------------------------------------------------------------------------
# Incompressible & Continuum Mechanics Operators
# -----------------------------------------------------------------------------
def project_solenoidal(
    v: torch.Tensor,
    spacing: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """
    Project vector field v onto the divergence-free (solenoidal) subspace: P v = v - grad Delta^-1 div v.

    In Fourier space, this applies the Leray projector P(k) = I - (k k^T) / |k|^2 for |k| > 0,
    leaving the zero-frequency DC translation mode (k = 0) strictly preserved.

    Parameters
    ----------
    v : Tensor (B, *spatial, dim) or (*spatial, dim)
        Vector field in tensor order.
    spacing : sequence of float, optional
        Voxel spacing in ANTs (x, y, z) order.

    Returns
    -------
    Tensor of the same shape and dtype as v, with spectral divergence analytically zero.
    """
    unbatched = (v.dim() == v.shape[-1] + 1)
    if unbatched:
        v = v.unsqueeze(0)

    dim = v.shape[-1]
    spatial_shape = v.shape[1:-1]
    device, dtype = v.device, v.dtype
    spatial_dims = tuple(range(2, 2 + dim))

    v_cf = torch.movedim(v, -1, 1).to(torch.float32)
    v_fft = torch.fft.rfftn(v_cf, dim=spatial_dims)

    k_mesh, k_sq, k_sq_safe = _get_k_mesh_rfft(spatial_shape, spacing, device)

    # Compute longitudinal component: (k . v_hat) / |k|^2
    k_dot_v = torch.zeros_like(v_fft[:, 0, ...])
    for d in range(dim):
        k_dot_v = k_dot_v + k_mesh[d] * v_fft[:, d, ...]

    mask_zero = (k_sq == 0.0)
    factor = torch.where(mask_zero, torch.zeros_like(k_dot_v), k_dot_v / k_sq_safe)

    # Subtract longitudinal component along each coordinate axis
    v_sol_fft = torch.stack([
        v_fft[:, d, ...] - factor * k_mesh[d] for d in range(dim)
    ], dim=1)

    v_sol_cf = torch.fft.irfftn(v_sol_fft, s=spatial_shape, dim=spatial_dims).to(dtype=dtype)
    v_sol = torch.movedim(v_sol_cf, 1, -1)
    return v_sol[0] if unbatched else v_sol


def apply_solenoidal_sobolev_operator(
    m: torch.Tensor,
    alpha: Optional[float] = None,
    fluid_sigma: float = 3.0,
    spacing: Optional[Sequence[float]] = None,
    s: float = 2.0,
) -> torch.Tensor:
    """
    Smooth field m with the Sobolev Green's operator K = (1 + alpha |k|^2)^-s AND project onto
    the divergence-free solenoidal subspace in a single vectorized Fourier multiplication.

    Parameters
    ----------
    m : Tensor (B, *spatial, dim) or (*spatial, dim)
    alpha : float, optional
        Kernel width (squared spacing units). If None, defaults to fluid_sigma / 2.0.
    fluid_sigma : float, default 3.0
    spacing : sequence of float, optional
    s : float, default 2.0
        Sobolev operator power.

    Returns
    -------
    Tensor of the same shape and dtype as m.
    """
    unbatched = (m.dim() == m.shape[-1] + 1)
    if unbatched:
        m = m.unsqueeze(0)

    dim = m.shape[-1]
    spatial_shape = m.shape[1:-1]
    device, dtype = m.device, m.dtype
    spatial_dims = tuple(range(2, 2 + dim))

    alpha_val = float(alpha) if alpha is not None else float(fluid_sigma) / 2.0
    if alpha_val <= 0:
        v_sol = project_solenoidal(m, spacing=spacing)
        return v_sol[0] if unbatched else v_sol

    m_cf = torch.movedim(m, -1, 1).to(torch.float32)
    m_fft = torch.fft.rfftn(m_cf, dim=spatial_dims)

    k_mesh, k_sq, k_sq_safe = _get_k_mesh_rfft(spatial_shape, spacing, device)

    # Sobolev filter
    K_sobolev = 1.0 / ((1.0 + alpha_val * k_sq) ** float(s))

    # Solenoidal projection factor
    k_dot_m = torch.zeros_like(m_fft[:, 0, ...])
    for d in range(dim):
        k_dot_m = k_dot_m + k_mesh[d] * m_fft[:, d, ...]

    mask_zero = (k_sq == 0.0)
    factor = torch.where(mask_zero, torch.zeros_like(k_dot_m), k_dot_m / k_sq_safe)

    v_sol_fft = torch.stack([
        K_sobolev * (m_fft[:, d, ...] - factor * k_mesh[d]) for d in range(dim)
    ], dim=1)

    v_sol_cf = torch.fft.irfftn(v_sol_fft, s=spatial_shape, dim=spatial_dims).to(dtype=dtype)
    v_sol = torch.movedim(v_sol_cf, 1, -1)
    return v_sol[0] if unbatched else v_sol


def apply_div_curl_green_operator(
    m: torch.Tensor,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    fluid_sigma: float = 3.0,
    total_sigma: Optional[float] = None,
    elastic_sigma: Optional[float] = None,
    spacing: Optional[Sequence[float]] = None,
    h3_envelope: bool = True,
    envelope_power: float = 2.0,
) -> torch.Tensor:
    """
    Decoupled Helmholtz Green's operator independently parameterizing dilatation (div) and shear (curl):
    G(k) = (1 / (alpha + beta |k|^2)) (k k^T / |k|^2) + (1 / (alpha + gamma |k|^2)) (I - k k^T / |k|^2).

    Parameters
    ----------
    m : Tensor (B, *spatial, dim) or (*spatial, dim)
    alpha : float, optional
        Mass damping term. Defaults to fluid_sigma / 2.0 (or 1.0).
    beta : float, optional
        Stiffness against volumetric dilatation (div v). Higher beta suppresses volume changes.
    gamma : float, optional
        Stiffness against rotational shear (curl v).
    fluid_sigma : float, default 3.0
    total_sigma, elastic_sigma : float, optional
        If provided and beta is not explicitly set, scales the dilatation-to-shear resistance ratio.
    spacing : sequence of float, optional
    h3_envelope : bool, default True
        If True, applies an isotropic Sobolev multiscale envelope K_sobolev = (1 + alpha |k|^2)^(-envelope_power)
        to the decoupled operator, imposing O(|k|^-6) high-frequency decay to eliminate non-physical shear
        micro-striations (ribs) and guarantee positive Liouville Jacobians under pure fluid SyN (total_sigma=0.0).
    envelope_power : float, default 2.0
        Power of the Sobolev envelope filter (2.0 = H^3 tri-harmonic total decay).

    Returns
    -------
    Tensor of the same shape and dtype as m.
    """
    unbatched = (m.dim() == m.shape[-1] + 1)
    if unbatched:
        m = m.unsqueeze(0)

    dim = m.shape[-1]
    spatial_shape = v_shape = m.shape[1:-1]
    device, dtype = m.device, m.dtype
    spatial_dims = tuple(range(2, 2 + dim))

    alpha_val = float(alpha) if alpha is not None else max(float(fluid_sigma) / 2.0, 1e-4)

    # Shear stiffness (gamma) defaults to alpha
    gamma_val = float(gamma) if gamma is not None else alpha_val

    # Dilatation stiffness (beta): if not provided, default to a high ratio (e.g. 5x) to resist volume change
    if beta is not None:
        beta_val = float(beta)
    else:
        sig = total_sigma if total_sigma is not None else elastic_sigma
        ratio = 1.0 + (float(sig) if sig is not None and sig > 0 else 4.0)
        beta_val = alpha_val * ratio

    m_cf = torch.movedim(m, -1, 1).to(torch.float32)
    m_fft = torch.fft.rfftn(m_cf, dim=spatial_dims)

    k_mesh, k_sq, k_sq_safe = _get_k_mesh_rfft(spatial_shape, spacing, device)

    # Eigenvalue filters: filt_div for longitudinal (k-parallel), filt_curl for transverse (k-perp)
    filt_div = 1.0 / (alpha_val + beta_val * k_sq_safe)
    filt_curl = 1.0 / (alpha_val + gamma_val * k_sq_safe)

    # Apply multiscale Sobolev envelope if requested
    if h3_envelope and envelope_power > 0:
        K_sobolev = 1.0 / ((1.0 + alpha_val * k_sq) ** float(envelope_power))
        filt_div = filt_div * K_sobolev
        filt_curl = filt_curl * K_sobolev

    k_dot_m = torch.zeros_like(m_fft[:, 0, ...])
    for d in range(dim):
        k_dot_m = k_dot_m + k_mesh[d] * m_fft[:, d, ...]

    mask_zero = (k_sq == 0.0)
    factor = torch.where(mask_zero, torch.zeros_like(k_dot_m), k_dot_m / k_sq_safe)

    # At DC (k=0), both filters give 1 / alpha_val
    dc_val = 1.0 / alpha_val
    filt_curl_safe = torch.where(mask_zero, torch.full_like(filt_curl, dc_val), filt_curl)
    filt_div_safe = torch.where(mask_zero, torch.full_like(filt_div, dc_val), filt_div)

    v_fft = torch.stack([
        filt_curl_safe * m_fft[:, d, ...] + (filt_div_safe - filt_curl_safe) * factor * k_mesh[d]
        for d in range(dim)
    ], dim=1)

    v_cf = torch.fft.irfftn(v_fft, s=spatial_shape, dim=spatial_dims).to(dtype=dtype)
    v_out = torch.movedim(v_cf, 1, -1)
    return v_out[0] if unbatched else v_out


def apply_navier_green_operator(
    m: torch.Tensor,
    alpha: Optional[float] = None,
    fluid_sigma: float = 3.0,
    poisson_ratio: float = 0.49,
    total_sigma: Optional[float] = None,
    elastic_sigma: Optional[float] = None,
    spacing: Optional[Sequence[float]] = None,
    s: float = 1.0,
) -> torch.Tensor:
    """
    Navier-Cauchy continuum elastic Green's operator:
    G(k) = (1 + alpha |k|^2)^-s * [ I - kappa * (k k^T / |k|^2) ], where kappa = 1 / (2 (1 - nu)).

    - nu = 0.30: standard compressible engineering solid (kappa = 0.714).
    - nu = 0.49: nearly incompressible brain parenchyma (kappa = 0.980).
    - nu = 0.50: strictly incompressible Stokes fluid (kappa = 1.000).

    Parameters
    ----------
    m : Tensor (B, *spatial, dim) or (*spatial, dim)
    alpha : float, optional
    fluid_sigma : float, default 3.0
    poisson_ratio : float, default 0.49
        Poisson's ratio in [0.0, 0.50].
    total_sigma, elastic_sigma : float, optional
    spacing : sequence of float, optional
    s : float, default 1.0

    Returns
    -------
    Tensor of the same shape and dtype as m.
    """
    unbatched = (m.dim() == m.shape[-1] + 1)
    if unbatched:
        m = m.unsqueeze(0)

    dim = m.shape[-1]
    spatial_shape = m.shape[1:-1]
    device, dtype = m.device, m.dtype
    spatial_dims = tuple(range(2, 2 + dim))

    alpha_val = float(alpha) if alpha is not None else max(float(fluid_sigma) / 2.0, 1e-4)

    nu = float(poisson_ratio)
    # Clamp Poisson's ratio to physically valid isotropic range [0.0, 0.50]
    nu = max(0.0, min(0.50, nu))
    kappa = 1.0 if nu >= 0.4999 else (1.0 / (2.0 * (1.0 - nu)))

    m_cf = torch.movedim(m, -1, 1).to(torch.float32)
    m_fft = torch.fft.rfftn(m_cf, dim=spatial_dims)

    k_mesh, k_sq, k_sq_safe = _get_k_mesh_rfft(spatial_shape, spacing, device)

    K_base = 1.0 / ((1.0 + alpha_val * k_sq) ** float(s))

    k_dot_m = torch.zeros_like(m_fft[:, 0, ...])
    for d in range(dim):
        k_dot_m = k_dot_m + k_mesh[d] * m_fft[:, d, ...]

    mask_zero = (k_sq == 0.0)
    factor = torch.where(mask_zero, torch.zeros_like(k_dot_m), k_dot_m / k_sq_safe)

    v_fft = torch.stack([
        K_base * (m_fft[:, d, ...] - kappa * factor * k_mesh[d]) for d in range(dim)
    ], dim=1)

    v_cf = torch.fft.irfftn(v_fft, s=spatial_shape, dim=spatial_dims).to(dtype=dtype)
    v_out = torch.movedim(v_cf, 1, -1)
    return v_out[0] if unbatched else v_out


def _prepare_mask_tensor(
    mask: Any,
    spatial_shape: Sequence[int],
    device: torch.device,
    batch_size: int = 1,
) -> Optional[torch.Tensor]:
    """
    Prepare anatomical mask tensor, converting from ANTsImage/numpy/tensor and
    resampling to `spatial_shape` if needed. Returns tensor of shape (B, 1, *spatial_shape).
    """
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        if hasattr(mask, 'numpy'):
            mask_t = torch.from_numpy(np.asarray(mask.numpy(), dtype=np.float32))
        else:
            mask_t = torch.as_tensor(mask, dtype=torch.float32)
    else:
        mask_t = mask.to(dtype=torch.float32)
    mask_t = mask_t.to(device=device)

    dim = len(spatial_shape)
    while mask_t.ndim < dim + 2:
        mask_t = mask_t.unsqueeze(0)
    while mask_t.ndim > dim + 2:
        mask_t = mask_t.squeeze(0)

    if tuple(mask_t.shape[2:]) != tuple(spatial_shape):
        mode = 'bilinear' if dim == 2 else ('trilinear' if dim == 3 else 'nearest')
        mask_t = F.interpolate(mask_t, size=tuple(spatial_shape), mode=mode, align_corners=False)

    if mask_t.shape[0] != batch_size and mask_t.shape[0] == 1:
        mask_t = mask_t.expand(batch_size, 1, *spatial_shape)

    return mask_t


def apply_masked_incompressible_filter(
    v: torch.Tensor,
    mask: Optional[Any] = None,
    alpha: Optional[float] = None,
    spacing: Optional[Sequence[float]] = None,
    num_iters: int = 3,
) -> torch.Tensor:
    """
    Tissue-specific incompressibility projection (Chorin projection method):
    Enforces div(v) ~ 0 within parenchymal tissue (mask = 1) while permitting volume expansion
    and contraction in unmasked regions (ventricles, fluid cavities, background, mask = 0).

    Parameters
    ----------
    v : Tensor (B, *spatial, dim) or (*spatial, dim)
    mask : Tensor, ANTsImage, or numpy array, optional
        Anatomical tissue mask with values in [0, 1]. If None, reduces to global solenoidal projection.
    alpha : float, optional
    spacing : sequence of float, optional
    num_iters : int, default 3
        Number of projection correction iterations.

    Returns
    -------
    Tensor of the same shape and dtype as v.
    """
    if mask is None:
        return project_solenoidal(v, spacing=spacing)

    unbatched = (v.dim() == v.shape[-1] + 1)
    if unbatched:
        v = v.unsqueeze(0)

    dim = v.shape[-1]
    spatial_shape = v.shape[1:-1]
    device, dtype = v.device, v.dtype
    spatial_dims = tuple(range(2, 2 + dim))

    mask_t = _prepare_mask_tensor(mask, spatial_shape, device, batch_size=v.shape[0])
    mask_sp = mask_t[:, 0, ...]

    k_mesh, k_sq, k_sq_safe = _get_k_mesh_rfft(spatial_shape, spacing, device)
    mask_zero = (k_sq == 0.0)

    curr_v = v.clone()

    for _ in range(max(1, int(num_iters))):
        # 1. Compute spectral divergence of current velocity
        v_cf = torch.movedim(curr_v, -1, 1).to(torch.float32)
        v_fft = torch.fft.rfftn(v_cf, dim=spatial_dims)

        k_dot_v = torch.zeros_like(v_fft[:, 0, ...])
        for d in range(dim):
            k_dot_v = k_dot_v + k_mesh[d] * v_fft[:, d, ...]

        div_dims = tuple(range(1, 1 + dim))
        div_v = torch.fft.irfftn(1j * k_dot_v, s=spatial_shape, dim=div_dims)

        # 2. Extract masked divergence to be eliminated: d_target = mask * div(v)
        d_target = mask_sp * div_v

        # 3. Solve Poisson equation Delta psi = d_target in Fourier domain: psi_hat = - d_hat / |k|^2
        d_fft = torch.fft.rfftn(d_target, dim=div_dims)
        psi_fft = torch.where(mask_zero, torch.zeros_like(d_fft), -d_fft / k_sq_safe)

        # 4. Correction field: delta_v = grad psi -> delta_v_hat = i k psi_hat
        if alpha is not None and float(alpha) > 0:
            K_smooth = 1.0 / (1.0 + float(alpha) * k_sq)
            psi_curr = psi_fft * K_smooth
        else:
            psi_curr = psi_fft

        delta_v_fft = torch.stack([
            1j * k_mesh[d] * psi_curr for d in range(dim)
        ], dim=1)

        delta_v = torch.fft.irfftn(delta_v_fft, s=spatial_shape, dim=spatial_dims)
        curr_v = curr_v - torch.movedim(delta_v, 1, -1).to(dtype=dtype)

    return curr_v[0] if unbatched else curr_v


def apply_beltrami_regularizer(
    v: torch.Tensor,
    alpha: Optional[float] = None,
    fluid_sigma: float = 3.0,
    dilatation_weight: Optional[float] = None,
    spacing: Optional[Sequence[float]] = None,
    s: float = 2.0,
) -> torch.Tensor:
    """
    Quasiconformal / Beltrami distortion operator.

    Inverts the conformal Killing differential operator:
        D_conf(v) = 0.5 * (grad v + (grad v)^T) - (1/d) * (div v) * I
    which penalizes deviatoric shear strain and maximal dilatation (sigma_max / sigma_min)
    to eliminate needle-like slivers and aspect-ratio distortion while preserving local angles.

    In Fourier space, the Beltrami Green operator acts as:
        G_Beltrami(k) = (1 / (1 + alpha * |k|^2))^s * [ I - ((d - 2) / (2*(d - 1))) * (k k^T / |k|^2) ]
    For d=2: factor = 0 (isotropic Cauchy-Riemann conformal smoothing).
    For d=3: factor = 1/4 (optimal deviatoric shear damping).

    Parameters
    ----------
    v : Tensor (B, *spatial, dim) or (*spatial, dim)
    alpha : float, optional
    fluid_sigma : float, default 3.0
    dilatation_weight : float, optional
    spacing : sequence of float, optional
    s : float, default 1.0

    Returns
    -------
    Tensor of the same shape and dtype as v.
    """
    unbatched = (v.dim() == v.shape[-1] + 1)
    if unbatched:
        v = v.unsqueeze(0)

    dim = v.shape[-1]
    spatial_shape = v.shape[1:-1]
    device, dtype = v.device, v.dtype
    spatial_dims = tuple(range(2, 2 + dim))

    if alpha is None:
        alpha = float(fluid_sigma) ** 2 / 2.0

    k_mesh, k_sq, k_sq_safe = _get_k_mesh_rfft(spatial_shape, spacing, device)

    # Conformal shear factor: eta = (d - 2) / (2 * (d - 1))
    # d=2 -> 0.0, d=3 -> 0.25
    if dilatation_weight is not None:
        eta = float(dilatation_weight)
    else:
        eta = (float(dim) - 2.0) / (2.0 * (float(dim) - 1.0)) if dim > 1 else 0.0

    K_base = (1.0 / (1.0 + alpha * k_sq)).pow(s)

    v_cf = torch.movedim(v, -1, 1).to(torch.float32)
    v_fft = torch.fft.rfftn(v_cf, dim=spatial_dims)

    k_dot_v = torch.zeros_like(v_fft[:, 0, ...])
    for d in range(dim):
        k_dot_v = k_dot_v + k_mesh[d] * v_fft[:, d, ...]

    mask_zero = (k_sq == 0.0)
    factor = torch.where(mask_zero, torch.zeros_like(k_dot_v), k_dot_v / k_sq_safe)

    v_out_fft = torch.stack([
        K_base * (v_fft[:, d, ...] - eta * factor * k_mesh[d]) for d in range(dim)
    ], dim=1)

    v_out_cf = torch.fft.irfftn(v_out_fft, s=spatial_shape, dim=spatial_dims).to(dtype=dtype)
    v_out = torch.movedim(v_out_cf, 1, -1)
    return v_out[0] if unbatched else v_out


def apply_poroelastic_filter(
    v: torch.Tensor,
    alpha: Optional[float] = None,
    darcy_permeability: Optional[float] = None,
    total_sigma: Optional[float] = None,
    elastic_sigma: Optional[float] = None,
    mask: Optional[torch.Tensor] = None,
    fluid_sigma: float = 3.0,
    spacing: Optional[Sequence[float]] = None,
    s: float = 2.0,
) -> torch.Tensor:
    """
    Two-Phase Poroelastic / Biot consolidation filter.

    Decomposes the deformation velocity into an incompressible solid skeleton
    v_solid (solenoidal, div v_solid = 0) and an interstitial fluid drainage flux
    v_fluid = -kappa * grad p (Darcy flow down pressure gradients):
        v = v_solid + v_fluid

    Applies Sobolev stiffness (alpha) to the solid matrix and Darcy dissipation (kappa)
    to the fluid phase:
        v_poro = G_solid(v_solid; alpha) + kappa * G_fluid(v_fluid; alpha)

    If `mask` is provided, Darcy fluid volume loss is restricted to edematous/cavity
    regions (where mask < 1), while the solid parenchymal matrix is preserved.

    Parameters
    ----------
    v : Tensor (B, *spatial, dim) or (*spatial, dim)
    alpha : float, optional
    darcy_permeability : float, optional
    total_sigma, elastic_sigma : float, optional
    mask : Tensor, optional
    fluid_sigma : float, default 3.0
    spacing : sequence of float, optional
    s : float, default 2.0
        Sobolev operator power (2.0 = H^2 biharmonic regularity).

    Returns
    -------
    Tensor of the same shape and dtype as v.
    """
    unbatched = (v.dim() == v.shape[-1] + 1)
    if unbatched:
        v = v.unsqueeze(0)

    dim = v.shape[-1]
    spatial_shape = v.shape[1:-1]
    device, dtype = v.device, v.dtype
    spatial_dims = tuple(range(2, 2 + dim))

    if alpha is None:
        alpha = float(fluid_sigma) ** 2 / 2.0

    if darcy_permeability is not None:
        kappa = float(darcy_permeability)
    elif total_sigma is not None:
        kappa = min(float(total_sigma), 1.0)
    elif elastic_sigma is not None:
        kappa = min(float(elastic_sigma), 1.0)
    else:
        kappa = 0.20

    # 1. Spectral Helmholtz decomposition: v = v_solid + v_fluid
    k_mesh, k_sq, k_sq_safe = _get_k_mesh_rfft(spatial_shape, spacing, device)
    mask_zero = (k_sq == 0.0)

    v_cf = torch.movedim(v, -1, 1).to(torch.float32)
    v_fft = torch.fft.rfftn(v_cf, dim=spatial_dims)

    k_dot_v = torch.zeros_like(v_fft[:, 0, ...])
    for d in range(dim):
        k_dot_v = k_dot_v + k_mesh[d] * v_fft[:, d, ...]

    factor = torch.where(mask_zero, torch.zeros_like(k_dot_v), k_dot_v / k_sq_safe)

    # v_fluid_fft is the irrotational potential: (k k^T / |k|^2) v
    # v_solid_fft is the solenoidal projection: (I - k k^T / |k|^2) v
    K_solid = (1.0 / (1.0 + alpha * k_sq)).pow(s)
    K_fluid = (1.0 / (1.0 + (alpha * 0.5) * k_sq)).pow(s)

    v_solid_fft = torch.stack([
        K_solid * (v_fft[:, d, ...] - factor * k_mesh[d]) for d in range(dim)
    ], dim=1)

    v_fluid_fft = torch.stack([
        K_fluid * (factor * k_mesh[d]) for d in range(dim)
    ], dim=1)

    v_solid_cf = torch.fft.irfftn(v_solid_fft, s=spatial_shape, dim=spatial_dims)
    v_fluid_cf = torch.fft.irfftn(v_fluid_fft, s=spatial_shape, dim=spatial_dims)

    if mask is not None:
        mask_t = _prepare_mask_tensor(mask, spatial_shape, device, batch_size=v.shape[0])
        fluid_weight = (1.0 - mask_t.clamp(0.0, 1.0))
        v_fluid_cf = v_fluid_cf * fluid_weight

    v_out_cf = (v_solid_cf + kappa * v_fluid_cf).to(dtype=dtype)
    v_out = torch.movedim(v_out_cf, 1, -1)
    return v_out[0] if unbatched else v_out


def apply_hyperelastic_regularizer(
    v: torch.Tensor,
    alpha: Optional[float] = None,
    bulk_modulus: Optional[float] = None,
    total_sigma: Optional[float] = None,
    elastic_sigma: Optional[float] = None,
    fluid_sigma: float = 3.0,
    spacing: Optional[Sequence[float]] = None,
    s: float = 2.0,
    h3_envelope: bool = True,
    envelope_power: Optional[float] = None,
) -> torch.Tensor:
    """
    Hyperelastic volumetric strain regularizer.

    In the increment limit where volumetric strain J ~ 1 + div(v),
    this filter penalizes non-unit Jacobian strain based on the Simo-Pister potential,
    damping divergent expansion/compression modes before coordinate composition.

    Parameters
    ----------
    v : Tensor (B, *spatial, dim) or (*spatial, dim)
    alpha : float, optional
    bulk_modulus : float, optional
        Bulk modulus K relative to shear modulus. Default 10.0 (near incompressibility).
    total_sigma, elastic_sigma : float, optional
    fluid_sigma : float, default 3.0
    spacing : sequence of float, optional
    s : float, default 2.0
    h3_envelope : bool, default True
    envelope_power : float, optional

    Returns
    -------
    Tensor of the same shape and dtype as v.
    """
    if alpha is None:
        alpha = float(fluid_sigma) ** 2 / 2.0

    if bulk_modulus is not None:
        K_b = float(bulk_modulus)
    elif total_sigma is not None and total_sigma > 0:
        K_b = float(total_sigma) * 10.0
    elif elastic_sigma is not None and elastic_sigma > 0:
        K_b = float(elastic_sigma) * 10.0
    else:
        K_b = 10.0

    p = envelope_power if envelope_power is not None else s
    return apply_div_curl_green_operator(
        v,
        alpha=alpha,
        beta=alpha * K_b,
        gamma=alpha,
        fluid_sigma=fluid_sigma,
        spacing=spacing,
        h3_envelope=h3_envelope,
        envelope_power=p,
    )


# -----------------------------------------------------------------------------
# Unified Regularizer Registry & Factory
# -----------------------------------------------------------------------------
_REGISTRY: Dict[str, Callable[..., Callable[[torch.Tensor], torch.Tensor]]] = {}


def register_regularizer(name: Union[str, Sequence[str]]):
    """
    Decorator to register a regularizer builder function.

    The builder must accept arbitrary keyword arguments and return a callable
    `reg_fn(v: torch.Tensor) -> torch.Tensor`.
    """
    def decorator(fn):
        names = [name] if isinstance(name, str) else list(name)
        for n in names:
            _REGISTRY[n.lower()] = fn
        return fn
    return decorator


def list_regularizers() -> List[str]:
    """Return sorted list of all registered regularizer names."""
    return sorted(_REGISTRY.keys())


def _build_sobolev(
    alpha: Optional[float] = None,
    sobolev_alpha: Optional[float] = None,
    fluid_sigma: float = 3.0,
    spacing: Optional[Sequence[float]] = None,
    s: float = 2.0,
    **kwargs,
) -> Callable[[torch.Tensor], torch.Tensor]:
    from .smoothing import apply_sobolev_green_operator
    a = sobolev_alpha if sobolev_alpha is not None else alpha
    return lambda v: apply_sobolev_green_operator(v, fluid_sigma=fluid_sigma, alpha=a, spacing=spacing)


def _build_gaussian(
    gaussian_sigma: Optional[float] = None,
    sigma: Optional[float] = None,
    fluid_sigma: Optional[float] = None,
    spacing: Optional[Sequence[float]] = None,
    **kwargs,
) -> Callable[[torch.Tensor], torch.Tensor]:
    from .smoothing import separable_gaussian_filter
    sig = gaussian_sigma if gaussian_sigma is not None else (sigma if sigma is not None else fluid_sigma)
    if sig is None or sig <= 0:
        return lambda v: v
    return lambda v: separable_gaussian_filter(v, sigma=sig, spacing=spacing)


def _build_compact_gaussian(
    gaussian_sigma: Optional[float] = None,
    sigma: Optional[float] = None,
    fluid_sigma: Optional[float] = None,
    **kwargs,
) -> Callable[[torch.Tensor], torch.Tensor]:
    from .smoothing import fast_separable_gaussian_filter
    sig = gaussian_sigma if gaussian_sigma is not None else (sigma if sigma is not None else fluid_sigma)
    if sig is None or sig <= 0:
        return lambda v: v
    return lambda v: fast_separable_gaussian_filter(v, sigma=sig)


def _build_dsti(
    alpha: Optional[float] = None,
    dsti_alpha: Optional[float] = None,
    fluid_sigma: float = 3.0,
    spacing: Optional[Sequence[float]] = None,
    s: float = 2.0,
    **kwargs,
) -> Callable[[torch.Tensor], torch.Tensor]:
    from .smoothing import apply_dsti_green_operator
    a = dsti_alpha if dsti_alpha is not None else alpha
    return lambda v: apply_dsti_green_operator(v, fluid_sigma=fluid_sigma, alpha=a, spacing=spacing, s=s)


def _build_bspline(
    spacing: Optional[Sequence[float]] = None,
    mesh_size: Optional[Union[int, Sequence[int]]] = None,
    spline_distance: Optional[float] = None,
    fluid_sigma: float = 3.0,
    **kwargs,
) -> Callable[[torch.Tensor], torch.Tensor]:
    from .smoothing import smooth_displacement_field_bspline
    return lambda v: smooth_displacement_field_bspline(
        v, spacing=spacing, mesh_size=mesh_size, spline_distance=spline_distance, fluid_sigma=fluid_sigma
    )


def _build_solenoidal(
    alpha: Optional[float] = None,
    sobolev_alpha: Optional[float] = None,
    fluid_sigma: float = 3.0,
    spacing: Optional[Sequence[float]] = None,
    s: float = 2.0,
    sobolev_envelope: bool = True,
    **kwargs,
) -> Callable[[torch.Tensor], torch.Tensor]:
    a = sobolev_alpha if sobolev_alpha is not None else alpha
    if not sobolev_envelope:
        return lambda v: project_solenoidal(v, spacing=spacing)
    return lambda v: apply_solenoidal_sobolev_operator(v, alpha=a, fluid_sigma=fluid_sigma, spacing=spacing, s=s)


def _build_leray_pure(
    spacing: Optional[Sequence[float]] = None,
    **kwargs,
) -> Callable[[torch.Tensor], torch.Tensor]:
    return lambda v: project_solenoidal(v, spacing=spacing)


def _build_solenoidal_sobolev(
    alpha: Optional[float] = None,
    sobolev_alpha: Optional[float] = None,
    fluid_sigma: float = 3.0,
    spacing: Optional[Sequence[float]] = None,
    s: float = 2.0,
    **kwargs,
) -> Callable[[torch.Tensor], torch.Tensor]:
    a = sobolev_alpha if sobolev_alpha is not None else alpha
    return lambda v: apply_solenoidal_sobolev_operator(v, alpha=a, fluid_sigma=fluid_sigma, spacing=spacing, s=s)


def _build_div_curl(
    alpha: Optional[float] = None,
    sobolev_alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    fluid_sigma: float = 3.0,
    total_sigma: Optional[float] = None,
    elastic_sigma: Optional[float] = None,
    spacing: Optional[Sequence[float]] = None,
    h3_envelope: bool = True,
    envelope_power: float = 2.0,
    s: Optional[float] = None,
    **kwargs,
) -> Callable[[torch.Tensor], torch.Tensor]:
    a = sobolev_alpha if sobolev_alpha is not None else alpha
    p = envelope_power if s is None else s
    return lambda v: apply_div_curl_green_operator(
        v, alpha=a, beta=beta, gamma=gamma, fluid_sigma=fluid_sigma,
        total_sigma=total_sigma, elastic_sigma=elastic_sigma, spacing=spacing,
        h3_envelope=h3_envelope, envelope_power=p
    )


def _build_navier(
    alpha: Optional[float] = None,
    sobolev_alpha: Optional[float] = None,
    poisson_ratio: float = 0.49,
    fluid_sigma: float = 3.0,
    total_sigma: Optional[float] = None,
    elastic_sigma: Optional[float] = None,
    spacing: Optional[Sequence[float]] = None,
    s: float = 1.0,
    **kwargs,
) -> Callable[[torch.Tensor], torch.Tensor]:
    a = sobolev_alpha if sobolev_alpha is not None else alpha
    return lambda v: apply_navier_green_operator(
        v, alpha=a, fluid_sigma=fluid_sigma, poisson_ratio=poisson_ratio,
        total_sigma=total_sigma, elastic_sigma=elastic_sigma, spacing=spacing, s=s
    )


def _build_masked_incompressible(
    mask: Optional[torch.Tensor] = None,
    fixed_mask: Optional[torch.Tensor] = None,
    alpha: Optional[float] = None,
    sobolev_alpha: Optional[float] = None,
    spacing: Optional[Sequence[float]] = None,
    num_iters: int = 3,
    fluid_sigma: float = 3.0,
    **kwargs,
) -> Callable[[torch.Tensor], torch.Tensor]:
    m = mask if mask is not None else fixed_mask
    a = sobolev_alpha if sobolev_alpha is not None else alpha
    return lambda v: apply_masked_incompressible_filter(
        v, mask=m, alpha=a, spacing=spacing, num_iters=num_iters
    )


def _build_beltrami(
    alpha: Optional[float] = None,
    sobolev_alpha: Optional[float] = None,
    fluid_sigma: float = 3.0,
    dilatation_weight: Optional[float] = None,
    spacing: Optional[Sequence[float]] = None,
    s: float = 2.0,
    **kwargs,
) -> Callable[[torch.Tensor], torch.Tensor]:
    a = sobolev_alpha if sobolev_alpha is not None else alpha
    return lambda v: apply_beltrami_regularizer(
        v, alpha=a, fluid_sigma=fluid_sigma, dilatation_weight=dilatation_weight,
        spacing=spacing, s=s
    )


def _build_poroelastic(
    alpha: Optional[float] = None,
    sobolev_alpha: Optional[float] = None,
    darcy_permeability: Optional[float] = None,
    total_sigma: Optional[float] = None,
    elastic_sigma: Optional[float] = None,
    mask: Optional[torch.Tensor] = None,
    fixed_mask: Optional[torch.Tensor] = None,
    fluid_sigma: float = 3.0,
    spacing: Optional[Sequence[float]] = None,
    s: float = 2.0,
    **kwargs,
) -> Callable[[torch.Tensor], torch.Tensor]:
    a = sobolev_alpha if sobolev_alpha is not None else alpha
    m = mask if mask is not None else fixed_mask
    return lambda v: apply_poroelastic_filter(
        v, alpha=a, darcy_permeability=darcy_permeability, total_sigma=total_sigma,
        elastic_sigma=elastic_sigma, mask=m, fluid_sigma=fluid_sigma,
        spacing=spacing, s=s
    )


def _build_hyperelastic(
    alpha: Optional[float] = None,
    sobolev_alpha: Optional[float] = None,
    bulk_modulus: Optional[float] = None,
    total_sigma: Optional[float] = None,
    elastic_sigma: Optional[float] = None,
    fluid_sigma: float = 3.0,
    spacing: Optional[Sequence[float]] = None,
    s: float = 2.0,
    h3_envelope: bool = True,
    envelope_power: Optional[float] = None,
    **kwargs,
) -> Callable[[torch.Tensor], torch.Tensor]:
    a = sobolev_alpha if sobolev_alpha is not None else alpha
    p = envelope_power if envelope_power is not None else s
    return lambda v: apply_hyperelastic_regularizer(
        v, alpha=a, bulk_modulus=bulk_modulus, total_sigma=total_sigma,
        elastic_sigma=elastic_sigma, fluid_sigma=fluid_sigma,
        spacing=spacing, s=p, h3_envelope=h3_envelope, envelope_power=p
    )


def _build_identity(**kwargs) -> Callable[[torch.Tensor], torch.Tensor]:
    return lambda v: v


# Register all canonical regularizers
for _k, _fn in [
    ('sobolev', _build_sobolev),
    ('gaussian', _build_gaussian),
    ('gauss', _build_gaussian),
    ('compact_gaussian', _build_compact_gaussian),
    ('compact', _build_compact_gaussian),
    ('fast_gaussian', _build_compact_gaussian),
    ('erf', _build_compact_gaussian),
    ('dsti', _build_dsti),
    ('dsti1', _build_dsti),
    ('dirichlet', _build_dsti),
    ('bspline', _build_bspline),
    ('solenoidal', _build_solenoidal),
    ('leray', _build_solenoidal),
    ('leray_pure', _build_leray_pure),
    ('incompressible', _build_solenoidal),
    ('solenoidal_sobolev', _build_solenoidal_sobolev),
    ('div_curl', _build_div_curl),
    ('helmholtz', _build_div_curl),
    ('navier', _build_navier),
    ('stokes', _build_navier),
    ('elastic', _build_navier),
    ('masked_incompressible', _build_masked_incompressible),
    ('beltrami', _build_beltrami),
    ('quasiconformal', _build_beltrami),
    ('conformal', _build_beltrami),
    ('poroelastic', _build_poroelastic),
    ('darcy_stokes', _build_poroelastic),
    ('biot', _build_poroelastic),
    ('hyperelastic', _build_hyperelastic),
    ('simo_pister', _build_hyperelastic),
    ('log_jacobian', _build_hyperelastic),
    ('none', _build_identity),
    ('identity', _build_identity),
]:
    _REGISTRY[_k] = _fn


def get_regularizer(
    name: Union[str, Callable[[torch.Tensor], torch.Tensor]],
    **kwargs,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Obtain a plug-and-play field regularizer callable.

    Parameters
    ----------
    name : str or callable
        Name of registered regularizer (e.g. 'sobolev', 'solenoidal', 'div_curl', 'navier'),
        or a custom callable.
    **kwargs :
        Passed to the regularizer builder (e.g. alpha, spacing, fluid_sigma, poisson_ratio).

    Returns
    -------
    Callable `reg_fn(v) -> smoothed_v`. Handles (B, *spatial, dim), (*spatial, dim) or
    (B, 1, *spatial, dim) tensors cleanly.
    """
    if callable(name) and not isinstance(name, str):
        return name

    key = str(name).lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"unknown regularizer {name!r}; expected one of {sorted(_REGISTRY.keys())}"
        )

    base_reg = _REGISTRY[key](**kwargs)

    # Wrap in a robust shape adapter that handles (B, 1, *spatial, dim) or unbatched inputs
    def adapter(v: torch.Tensor) -> torch.Tensor:
        if v.ndim in (5, 6) and v.shape[1] == 1:
            # Singleton channel/depth dimension (e.g. RegAdam parameter group)
            sq = v.squeeze(1)
            out = base_reg(sq)
            return out.unsqueeze(1)
        return base_reg(v)

    return adapter
