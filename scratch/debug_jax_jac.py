import math
import numpy as np
import torch
import torch.nn.functional as F
import ants
import jax.numpy as jnp
from syntx.syn_jax import SyNTo, compute_jacobian_determinant_nd_jax

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
    
    model = SyNTo(dim=3, grid_shape=(32, 32, 32), fluid_sigma=2.0, elastic_sigma=1.0)
    model.fit(
        fi_norm_tensor, 
        mi_norm_tensor, 
        levels=[2, 1], 
        epochs_per_level=30, 
        cfl_voxels=0.15,
        affine_epochs=30, 
        affine_lr=1e-2,
        similarity_metric='lncc'
    )
    
    warp = model.warp_l2r
    print("Warp field shape:", warp.shape)
    print("Warp field max abs value:", float(jnp.max(jnp.abs(warp))))
    print("Auto-detected is_physical:", bool(jnp.max(jnp.abs(warp)) > 2.0))
    
    jac = compute_jacobian_determinant_nd_jax(warp)
    print("Jacobian min value:", float(jnp.min(jac)))
    print("Jacobian max value:", float(jnp.max(jac)))

if __name__ == '__main__':
    main()
