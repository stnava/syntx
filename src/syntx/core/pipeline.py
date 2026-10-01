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
        'pytorch': 'cuda' if available, else 'mps' if available, else 'cpu'. 'jax': always
        returns 'jax' (JAX chooses its own device). Anything else: 'cpu'.
    requested_device : str or torch.device, optional
        If given it is returned as ``str(requested_device).lower()`` without any check.

    Returns
    -------
    str
    """
    if requested_device is not None:
        return str(requested_device).lower()
        
    if backend == 'pytorch':
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    elif backend == 'jax':
        # JAX automatically uses the best available backend
        return 'jax'
    return 'cpu'


def normalize_and_tensorize(fixed, moving, winsorize_quantiles=None, backend='pytorch', device='cpu'):
    """
    Rescale fixed / moving ANTsImages to [0, 1] and stack them as (1, C, *spatial) tensors.

    Each image is normalised on its own:

    - if its values already lie in [0, 1] (within 1e-4) and its maximum is >= 0.5, it is only
      clipped to [0, 1];
    - otherwise the 2nd and 98th percentiles of the foreground (voxels > 0, or voxels with
      ``|v| > 1e-4`` when the image has negative values) are mapped to 0 and 1 and the result
      is clipped to [0, 1]. If those percentiles coincide, ``min(0, fg.min())`` and
      ``fg.max()`` are used; with no foreground, the image min and max.

    Parameters
    ----------
    fixed, moving : ANTsImage or list / tuple of ANTsImage
        A list gives one channel per image. Pairs are formed with ``zip``, so extra images in
        the longer list are dropped silently (a single moving image with a fixed list gives
        one channel).
    winsorize_quantiles : optional
        Ignored; the 2 / 98 percentiles are fixed.
    backend : str, default 'pytorch'
        'pytorch' or 'jax'. Anything else raises ValueError.
    device : str, default 'cpu'
        Torch device for the outputs. Ignored for 'jax'.

    Returns
    -------
    (I_tensor, J_tensor)
        float32 torch tensors (or ``jax.numpy`` arrays) of shape (1, C, *spatial), spatial axes
        in tensor order (z, y, x) (the ANTs (x, y, z) axes reversed).
    """
    is_multi = isinstance(fixed, (list, tuple))
    fixed_list = list(fixed) if is_multi else [fixed]
    moving_list = list(moving) if isinstance(moving, (list, tuple)) else [moving]
    
    def _norm_fg(arr):
        arr_min = float(arr.min())
        arr_max = float(arr.max())
        # Idempotency check: if already normalized to [0, 1] with active range, avoid re-clipping
        if arr_min >= -1e-4 and arr_max <= 1.0 + 1e-4 and arr_max >= 0.5:
            return np.clip(arr, 0.0, 1.0).astype(np.float32)

        has_negative = bool((arr < -1e-4).any())
        if has_negative:
            fg = arr[np.abs(arr) > 1e-4]
        else:
            fg = arr[arr > 0]
            
        if len(fg) > 0:
            p02 = float(np.percentile(fg, 2.0))
            p98 = float(np.percentile(fg, 98.0))
            if p98 <= p02 + 1e-4:
                p02 = float(min(0.0, fg.min()))
                p98 = float(fg.max())
        else:
            p02 = float(arr_min)
            p98 = float(arr_max)
        return np.clip((arr - p02) / (p98 - p02 + 1e-6), 0.0, 1.0).astype(np.float32)
        
    dim = fixed_list[0].dimension
    perm = [0, 1] + list(range(dim + 1, 1, -1))
    
    if backend == 'pytorch':
        import torch
        I_channels = []
        J_channels = []
        for f, m in zip(fixed_list, moving_list):
            f_norm = _norm_fg(f.numpy())
            m_norm = _norm_fg(m.numpy())
            I_channels.append(torch.tensor(f_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).permute(perm))
            J_channels.append(torch.tensor(m_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).permute(perm))
        I_tensor = torch.cat(I_channels, dim=1)
        J_tensor = torch.cat(J_channels, dim=1)
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
    Run ``gc.collect()`` and empty the torch CUDA or MPS cache, chosen from ``str(device)``.

    For a CPU device only garbage collection runs. For a ``backend`` other than 'pytorch'
    nothing is done. Returns None.
    """
    if backend == 'pytorch':
        import torch
        dev_str = str(device).lower() if device is not None else ''
        gc.collect()
        if 'mps' in dev_str and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        elif 'cuda' in dev_str and hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        gc.collect()
