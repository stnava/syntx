import math
import numpy as np
import torch
import torch.nn.functional as F
import ants
import jax.numpy as jnp
from syntx.syn import SyNTo as SyNToPy, compute_jacobian_determinant_nd
from syntx.syn_jax import SyNTo as SyNToJax, compute_jacobian_determinant_nd_jax

def get_test_image_3d():
    vol = np.zeros((32, 32, 32), dtype=float)
    z, y, x = np.ogrid[:32, :32, :32]
    mask1 = (x - 16)**2 + (y - 16)**2 + (z - 16)**2 < 8**2
    vol[mask1] = 15
    img = ants.from_numpy(vol, spacing=(1, 1, 1), origin=(0, 0, 0))
    img = ants.smooth_image(img, 1.0)
    return img

def main():
    fixed_img = get_test_image_3d()
    fixed_np = fixed_img.numpy()
    fixed_tensor = torch.tensor(fixed_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    fi_mean = fixed_np.mean()
    fi_std = fixed_np.std() + 1e-8
    fi_norm = (fixed_np - fi_mean) / fi_std
    
    theta = math.radians(8.0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    A = torch.tensor([
        [cos_t, -sin_t, 0.0, 0.04],
        [sin_t,  cos_t, 0.0, -0.02],
        [0.0,    0.0,   1.0, 0.0]
    ], dtype=torch.float32).unsqueeze(0)
    
    grid = F.affine_grid(A, size=[1, 1, 32, 32, 32], align_corners=True)
    moving_tensor = F.grid_sample(fixed_tensor, grid, mode='bilinear', padding_mode='border', align_corners=True)
    moving_np = moving_tensor.squeeze().numpy()
    
    mi_mean = moving_np.mean()
    mi_std = moving_np.std() + 1e-8
    mi_norm = (moving_np - mi_mean) / mi_std
    
    fi_norm_tensor = torch.tensor(fi_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mi_norm_tensor = torch.tensor(mi_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # 1. Run PyTorch
    print("--- Running PyTorch 3D SyNTo ---")
    model_py = SyNToPy(dim=3, grid_shape=(32, 32, 32), fluid_sigma=2.0, elastic_sigma=1.0)
    model_py.fit(
        fi_norm_tensor, 
        mi_norm_tensor, 
        levels=[2, 1], 
        epochs_per_level=30, 
        cfl_voxels=0.15,
        affine_epochs=30, 
        affine_lr=1e-2,
        similarity_metric='lncc'
    )
    jac_py = compute_jacobian_determinant_nd(model_py.warp_l2r)
    print("PyTorch Jacobian min (entire grid):", float(torch.min(jac_py)))
    print("PyTorch Jacobian min (excluding boundary):", float(torch.min(jac_py[..., 1:-1, 1:-1, 1:-1])))
    
    # 2. Run JAX
    print("\n--- Running JAX 3D SyNTo ---")
    model_jax = SyNToJax(dim=3, grid_shape=(32, 32, 32), fluid_sigma=2.0, elastic_sigma=1.0)
    model_jax.fit(
        fi_norm_tensor, 
        mi_norm_tensor, 
        levels=[2, 1], 
        epochs_per_level=30, 
        cfl_voxels=0.15,
        affine_epochs=30, 
        affine_lr=1e-2,
        similarity_metric='lncc'
    )
    jac_jax = compute_jacobian_determinant_nd_jax(model_jax.warp_l2r)
    print("JAX Jacobian min (entire grid):", float(jnp.min(jac_jax)))
    print("JAX Jacobian min (excluding boundary):", float(jnp.min(jac_jax[..., 1:-1, 1:-1, 1:-1])))

if __name__ == '__main__':
    main()
