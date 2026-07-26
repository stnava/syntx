import pytest
import torch
import numpy as np
from syntx.tvf import TVFModel

def test_tvf_model_2d_forward_and_warp():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    shape = (32, 32)
    
    # Create simple synthetic box images
    fi_np = np.zeros(shape, dtype=np.float32)
    fi_np[8:24, 8:24] = 1.0
    
    mi_np = np.zeros(shape, dtype=np.float32)
    mi_np[10:26, 8:24] = 1.0  # Shifted along Y
    
    fi_t = torch.tensor(fi_np, device=device).unsqueeze(0).unsqueeze(0)
    mi_t = torch.tensor(mi_np, device=device).unsqueeze(0).unsqueeze(0)
    
    model = TVFModel(
        dim=2, image_shape=shape, velocity_shape=(16, 16), n_time_steps=4,
        spacing=[1.0, 1.0], origin=[0.0, 0.0], direction=np.eye(2).tolist(),
        solver='euler', transform_type='Translation'
    )
    model.to(device)
    
    # Initial loss
    loss_init = model.forward(fi_t, mi_t).item()
    
    # Optimize velocity for 20 steps
    optimizer = torch.optim.Adam([model.velocity], lr=0.1)
    for _ in range(20):
        optimizer.zero_grad()
        loss = model.forward(fi_t, mi_t)
        loss.backward()
        optimizer.step()
        
    loss_opt = loss.item()
    assert loss_opt < loss_init, f"TVF loss did not decrease: {loss_init:.4f} -> {loss_opt:.4f}"
    
    with torch.no_grad():
        phi_fwd = model.get_forward_warp()
        phi_inv = model.get_inverse_warp()
        
    assert phi_fwd.shape == (1, 32, 32, 2)
    assert phi_inv.shape == (1, 32, 32, 2)
    
    # Inverse identity error should be sub-voxel
    sym_err = (phi_fwd + phi_inv).abs().mean().item()
    assert sym_err < 0.5, f"High symmetry error: {sym_err:.4f}"

def test_tvf_model_3d_forward_and_warp():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    shape = (16, 16, 16)
    
    fi_t = torch.randn(1, 1, *shape, device=device)
    mi_t = torch.randn(1, 1, *shape, device=device)
    
    model = TVFModel(
        dim=3, image_shape=shape, velocity_shape=(8, 8, 8), n_time_steps=4,
        spacing=[1.0, 1.0, 1.0], origin=[0.0, 0.0, 0.0], direction=np.eye(3).tolist(),
        solver='rk4', transform_type='Translation'
    )
    model.to(device)
    
    loss = model.forward(fi_t, mi_t)
    loss.backward()
    assert model.velocity.grad is not None
    
    with torch.no_grad():
        phi_fwd = model.get_forward_warp()
        phi_inv = model.get_inverse_warp()
        
    assert phi_fwd.shape == (1, 16, 16, 16, 3)
    assert phi_inv.shape == (1, 16, 16, 16, 3)
