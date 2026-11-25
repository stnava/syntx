"""
Small helpers shared by the registration front ends: device choice, input normalisation /
tensor conversion, and GPU cache clean-up.
"""
import numpy as np
import gc
import tempfile
import ants

def auto_detect_device(backend='pytorch', requested_device=None):
    """
    Pick a compute device string.

    Parameters
    ----------
    backend : str, default 'pytorch'
        'pytorch': central ``syntx.device.resolve_device`` (explicit > env > cuda > mps > cpu; an unavailable explicit device raises). 'jax': always
        returns 'jax' (JAX chooses its own device). Anything else: 'cpu'.
    requested_device : str or torch.device, optional
        pytorch: validated by the central resolver; other backends: returned lower-cased unchecked.

    Returns
    -------
    str
    """
    if backend == 'pytorch':
        from ..device import resolve_device
        return resolve_device(requested_device, task='reg')
    if requested_device is not None:
        return str(requested_device).lower()
    if backend == 'jax':
        # JAX automatically uses the best available backend
        return 'jax'
    return 'cpu'


def _same_device(a, b) -> bool:
    return a.type == b.type and ((a.index or 0) == (b.index or 0) if a.type != 'cpu' else True)


def is_unclipped_unit_image(img) -> bool:
    """Return True when ``img`` is 3-D float32 with values already in ``[0.0, 1.0]`` and ``max >= 0.5``,
    so the normalized optimization tensor equals ``ContinuousField(img).tensor`` bitwise."""
    if not hasattr(img, "numpy"):
        return False
    arr = img.numpy()
    if arr.dtype != np.float32 or arr.ndim != 3:
        return False
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    return bool(arr_min >= 0.0 and arr_max <= 1.0 and arr_max >= 0.5)


def upload_raw_image_zyx(img, device):
    """Upload a scalar ``ants.ANTsImage`` once to ``device`` in contiguous ``(1, 1, *zyx)`` float32 layout."""
    import torch

    arr = np.ascontiguousarray(img.numpy(), dtype=np.float32)
    rev_spatial = tuple(range(arr.ndim - 1, -1, -1))
    t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
    if arr.ndim >= 2:
        t = t.permute(*rev_spatial).contiguous()
    return t.unsqueeze(0).unsqueeze(0)


def _compute_fg_norm_params(arr, q_lo: float, q_hi: float):
    """Return ``(p02, denom, is_id_clamp)`` matching ``normalize_and_tensorize``'s foreground 2–98% policy."""
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_min >= -1e-4 and arr_max <= 1.0 + 1e-4 and arr_max >= 0.5:
        return 0.0, 1.0, True

    is_hu = bool(arr_min <= -500.0 and arr_max >= 100.0)
    has_negative = bool((arr < -1e-4).any())
    if is_hu:
        fg = arr[arr > -900.0]
    elif has_negative:
        fg = arr[np.abs(arr) > 1e-4]
    else:
        fg = arr[arr > 0]

    if len(fg) > 0:
        fg_q = fg[:: max(1, len(fg) // 2_000_000)] if arr.size > 16_777_216 and len(fg) > 4_000_000 else fg
        p02 = -1000.0 if is_hu else float(np.percentile(fg_q, 100.0 * q_lo))
        p98 = float(np.percentile(fg_q, 100.0 * q_hi))
        if is_hu:
            p98 = max(p98, 300.0)
        if p98 <= p02 + 1e-4:
            p02 = float(min(0.0, fg.min()))
            p98 = float(fg.max())
    else:
        p02 = float(arr_min)
        p98 = float(arr_max)
    return float(p02), float(p98 - p02 + 1e-6), False


def normalize_and_tensorize(
    fixed,
    moving,
    winsorize_quantiles=None,
    backend='pytorch',
    device='cpu',
    _preuploaded_zyx=None,
):
    """
    Rescale fixed / moving ANTsImages to [0, 1] and stack them as (1, C, *spatial) tensors.

    Each image is normalised on its own:

    - if its values already lie in [0, 1] (within 1e-4) and its maximum is >= 0.5, it is only
      clipped to [0, 1];
    - otherwise the ``winsorize_quantiles`` (default 0.02 / 0.98) quantiles of the foreground
      (voxels > 0, or voxels with ``|v| > 1e-4`` when the image has negative values) are mapped
      to 0 and 1 and the result is clipped to [0, 1]. If those quantiles coincide,
      ``min(0, fg.min())`` and ``fg.max()`` are used; with no foreground, the image min and max.

    Parameters
    ----------
    fixed, moving : ANTsImage or list / tuple of ANTsImage
        A list gives one channel per image; fixed and moving need the same channel count
        (ValueError otherwise).
    winsorize_quantiles : (lo, hi), optional
        Foreground quantile fractions mapped to 0 / 1, ``0 <= lo < hi <= 1`` (ValueError
        otherwise). None: (0.02, 0.98).
    backend : str, default 'pytorch'
        'pytorch' or 'jax'. Anything else raises ValueError.
    device : str, default 'cpu'
        Torch device for the outputs. 'jax' arrays live on the JAX default device, so only
        None / 'cpu' are accepted there (ValueError otherwise).

    Returns
    -------
    (I_tensor, J_tensor)
        float32 torch tensors (or ``jax.numpy`` arrays) of shape (1, C, *spatial), spatial axes
        in tensor order (z, y, x) (the ANTs (x, y, z) axes reversed).
    """
    is_multi = isinstance(fixed, (list, tuple))
    fixed_list = list(fixed) if is_multi else [fixed]
    moving_list = list(moving) if isinstance(moving, (list, tuple)) else [moving]
    if len(fixed_list) != len(moving_list):
        raise ValueError(f"fixed has {len(fixed_list)} channel image(s), moving {len(moving_list)}")
    if winsorize_quantiles is None:
        q_lo, q_hi = 0.02, 0.98
    else:
        try:
            q_lo, q_hi = (float(q) for q in winsorize_quantiles)
        except (TypeError, ValueError):
            raise ValueError(f"winsorize_quantiles must be (lo, hi), got {winsorize_quantiles!r}") from None
        if not 0.0 <= q_lo < q_hi <= 1.0:
            raise ValueError(f"winsorize_quantiles needs 0 <= lo < hi <= 1, got {winsorize_quantiles!r}")
    if backend == 'jax' and device not in (None, 'cpu'):
        raise ValueError(f"device={device!r} is not used by the jax backend (JAX default device)")
    
    def _norm_fg(arr):
        p02, denom, is_id = _compute_fg_norm_params(arr, q_lo, q_hi)
        if is_id:
            return np.clip(arr, 0.0, 1.0).astype(np.float32)
        return np.clip((arr - p02) / denom, 0.0, 1.0).astype(np.float32)

    dim = fixed_list[0].dimension
    perm = [0, 1] + list(range(dim + 1, 1, -1))
    rev_spatial = tuple(range(dim - 1, -1, -1))

    if backend == 'pytorch':
        import torch
        dev_obj = torch.device(device) if device is not None else torch.device('cpu')
        is_accel = dev_obj.type != 'cpu'
        if (
            _preuploaded_zyx is not None
            and len(fixed_list) == 1
            and isinstance(_preuploaded_zyx, tuple)
            and len(_preuploaded_zyx) == 2
            and _preuploaded_zyx[0] is not None
            and _preuploaded_zyx[1] is not None
            and _same_device(_preuploaded_zyx[0].device, dev_obj)
        ):
            f_t, m_t = _preuploaded_zyx
            f_p02, f_denom, f_id = _compute_fg_norm_params(fixed_list[0].numpy(), q_lo, q_hi)
            m_p02, m_denom, m_id = _compute_fg_norm_params(moving_list[0].numpy(), q_lo, q_hi)
            with torch.no_grad():
                if not f_id:
                    f_t.sub_(float(f_p02)).div_(float(f_denom))
                f_t.clamp_(0.0, 1.0)
                if not m_id:
                    m_t.sub_(float(m_p02)).div_(float(m_denom))
                m_t.clamp_(0.0, 1.0)
            return f_t, m_t

        I_channels = []
        J_channels = []
        for f, m in zip(fixed_list, moving_list):
            if is_accel:
                f_arr = f.numpy()
                m_arr = m.numpy()
                f_p02, f_denom, f_id = _compute_fg_norm_params(f_arr, q_lo, q_hi)
                m_p02, m_denom, m_id = _compute_fg_norm_params(m_arr, q_lo, q_hi)
                f_t = upload_raw_image_zyx(f, dev_obj)
                m_t = upload_raw_image_zyx(m, dev_obj)
                with torch.no_grad():
                    if not f_id:
                        f_t.sub_(float(f_p02)).div_(float(f_denom))
                    f_t.clamp_(0.0, 1.0)
                    if not m_id:
                        m_t.sub_(float(m_p02)).div_(float(m_denom))
                    m_t.clamp_(0.0, 1.0)
                I_channels.append(f_t)
                J_channels.append(m_t)
            else:
                f_norm = _norm_fg(f.numpy())
                m_norm = _norm_fg(m.numpy())
                f_t = torch.from_numpy(f_norm).to(device=dev_obj, dtype=torch.float32).permute(*rev_spatial).contiguous()
                m_t = torch.from_numpy(m_norm).to(device=dev_obj, dtype=torch.float32).permute(*rev_spatial).contiguous()
                I_channels.append(f_t.unsqueeze(0).unsqueeze(0))
                J_channels.append(m_t.unsqueeze(0).unsqueeze(0))
        I_tensor = I_channels[0] if len(I_channels) == 1 else torch.cat(I_channels, dim=1)
        J_tensor = J_channels[0] if len(J_channels) == 1 else torch.cat(J_channels, dim=1)
    elif backend == 'jax':
        import jax.numpy as jnp
        I_channels = []
        J_channels = []
        for f, m in zip(fixed_list, moving_list):
            f_norm = _norm_fg(f.numpy())
            m_norm = _norm_fg(m.numpy())
            I_channels.append(jnp.array(f_norm).reshape(1, 1, *f_norm.shape).transpose(perm))
            J_channels.append(jnp.array(m_norm).reshape(1, 1, *m_norm.shape).transpose(perm))
        I_tensor = jnp.concatenate(I_channels, axis=1)
        J_tensor = jnp.concatenate(J_channels, axis=1)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return I_tensor, J_tensor


def cleanup_gpu(device, backend='pytorch'):
    """
    Empty the torch CUDA or MPS cache and trim oversized cached frequency grids.

    On CPU or non-'pytorch' backends, returns immediately without forcing full-heap
    Python ``gc.collect()`` scans. On MPS/CUDA, runs a single ``gc.collect()`` only when
    allocated GPU memory exceeds 64 MiB before releasing cached blocks.
    """
    if backend == 'pytorch':
        import torch
        from .regularizers import clear_k_mesh_cache
        dev_str = str(device).lower() if device is not None else ''
        is_mps = 'mps' in dev_str
        is_cuda = 'cuda' in dev_str
        if not (is_mps or is_cuda):
            return
        clear_k_mesh_cache(max_voxels=16_777_216)
        if is_mps:
            alloc = torch.mps.current_allocated_memory() if hasattr(torch.mps, 'current_allocated_memory') else 0
            if alloc > 64 * 1024 * 1024:
                gc.collect()
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        elif is_cuda:
            alloc = torch.cuda.memory_allocated() if hasattr(torch.cuda, 'memory_allocated') else 0
            if alloc > 64 * 1024 * 1024:
                gc.collect()
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
