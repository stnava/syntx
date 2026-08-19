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
    Returns (I_tensor, J_tensor).
    """
    fi_np = fixed.numpy()
    mi_np = moving.numpy()
    
    def _norm_fg(arr):
        pos = arr[arr > 0]
        if len(pos) > 0:
            p02 = float(np.percentile(pos, 2.0))
            p98 = float(np.percentile(pos, 98.0))
            if p98 <= p02 + 1e-4:
                p02 = 0.0
                p98 = float(pos.max())
        else:
            p02 = float(arr.min())
            p98 = float(arr.max())
        return np.clip((arr - p02) / (p98 - p02 + 1e-6), 0.0, 1.0).astype(np.float32)
        
    fi_norm = _norm_fg(fi_np)
    mi_norm = _norm_fg(mi_np)
    
    dim = fixed.dimension
    perm = [0, 1] + list(range(dim + 1, 1, -1))
    
    if backend == 'pytorch':
        import torch
        I_tensor = torch.tensor(fi_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).permute(perm)
        J_tensor = torch.tensor(mi_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).permute(perm)
    elif backend == 'jax':
        import jax.numpy as jnp
        I_tensor = jnp.array(fi_norm).reshape(1, 1, *fi_np.shape).transpose(perm)
        J_tensor = jnp.array(mi_norm).reshape(1, 1, *mi_np.shape).transpose(perm)
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
