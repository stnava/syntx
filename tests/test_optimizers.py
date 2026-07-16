import pytest
import numpy as np
import torch
import ants
from syntx.syn import registration as registration_py

def get_dummy_images():
    # Simple 2D dummy images
    fixed_np = np.ones((16, 16), dtype=np.float32)
    fixed_np[4:12, 4:12] = 2.0
    moving_np = np.ones((16, 16), dtype=np.float32)
    moving_np[5:13, 5:13] = 2.0
    
    fixed_img = ants.from_numpy(fixed_np)
    moving_img = ants.from_numpy(moving_np)
    return fixed_img, moving_img

@pytest.mark.parametrize("optimizer_type", ["cfl", "adam", "sgd", "lbfgs"])
def test_pytorch_optimizers(optimizer_type):
    fixed_img, moving_img = get_dummy_images()
    
    # Run registration with PyTorch backend
    res = registration_py(
        fixed=fixed_img,
        moving=moving_img,
        reg_iterations=[5],
        affine_iterations=[0],
        backend='pytorch',
        optimizer_type=optimizer_type,
        optimizer_lr=1e-2 if optimizer_type != "lbfgs" else 1.0
    )
    
    assert res is not None
    assert 'warpedmovout' in res
    assert 'fwdtransforms' in res

@pytest.mark.parametrize("optimizer_type", ["cfl", "adam", "sgd", "lbfgs"])
def test_jax_optimizers(optimizer_type):
    fixed_img, moving_img = get_dummy_images()
    
    # Run registration with JAX backend
    res = registration_py(
        fixed=fixed_img,
        moving=moving_img,
        reg_iterations=[5],
        affine_iterations=[0],
        backend='jax',
        optimizer_type=optimizer_type,
        optimizer_lr=1e-2 if optimizer_type != "lbfgs" else 1.0
    )
    
    assert res is not None
    assert 'warpedmovout' in res
    assert 'fwdtransforms' in res
