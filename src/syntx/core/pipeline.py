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
    Winsorizes, normalizes, and tensorizes the input images.
    Returns (I_tensor, J_tensor, fixed_np, moving_np).
    """
    fi_np = fixed.numpy()
    mi_np = moving.numpy()
    
    if winsorize_quantiles is not None:
        lo_f, hi_f = np.quantile(fi_np[fi_np > 0], winsorize_quantiles) if (fi_np > 0).any() else (fi_np.min(), fi_np.max())
        fi_np = np.clip(fi_np, lo_f, hi_f)
        lo_m, hi_m = np.quantile(mi_np[mi_np > 0], winsorize_quantiles) if (mi_np > 0).any() else (mi_np.min(), mi_np.max())
        mi_np = np.clip(mi_np, lo_m, hi_m)
        
    fi_norm = (fi_np - fi_np.mean()) / (fi_np.std() + 1e-8)
    mi_norm = (mi_np - mi_np.mean()) / (mi_np.std() + 1e-8)
    
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
