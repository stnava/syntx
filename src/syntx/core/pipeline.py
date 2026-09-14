import numpy as np
import gc
import tempfile
import ants

def auto_detect_device(backend='pytorch', requested_device=None):
    """
    Auto-detects the optimal compute device.
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
    Winsorizes, normalizes, and tensorizes the input images using foreground 2nd-98th percentiles.
    Supports single ANTsImage or list/tuple of ANTsImages for multi-channel registration.
    Returns (I_tensor, J_tensor).
    """
    is_multi = isinstance(fixed, (list, tuple))
    fixed_list = list(fixed) if is_multi else [fixed]
    moving_list = list(moving) if isinstance(moving, (list, tuple)) else [moving]
    
    def _norm_fg(arr):
        has_negative = bool((arr < -1e-4).any())
        if has_negative:
            fg = arr[np.abs(arr) > 1e-4]
        else:
            fg = arr[arr > 0]
            
        if len(fg) > 0:
            p02 = float(np.percentile(fg, 2.0))
            p98 = float(np.percentile(fg, 98.0))
            if p98 <= p02 + 1e-4:
                p02 = float(fg.min())
                p98 = float(fg.max())
        else:
            p02 = float(arr.min())
            p98 = float(arr.max())
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
    Frees GPU/MPS memory to prevent OOM errors in loops.
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
