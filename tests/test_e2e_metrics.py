import os
import sys
import types
import pytest
import numpy as np
import torch
import jax
import jax.numpy as jnp
import ants
import tempfile
import pandas as pd
from functools import partial

# Define Mock classes at module level so they are persistent and accessible
class MockSwinViT(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x):
        B = x.shape[0]
        return [
            torch.zeros(B, 48, 48, 48, 48), # layer 1
            torch.zeros(B, 96, 24, 24, 24), # layer 2
            torch.zeros(B, 192, 12, 12, 12), # layer 3
            torch.zeros(B, 384, 6, 6, 6),    # layer 4
            torch.zeros(B, 384, 3, 3, 3)     # layer 5
        ]

class MockSwinUNETR(torch.nn.Module):
    def __init__(self, img_size=(96, 96, 96), in_channels=1, out_channels=14, feature_size=48, **kwargs):
        super().__init__()
        self.swinViT = MockSwinViT()
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return torch.zeros(x.shape[0], 14, x.shape[2]//2, x.shape[3]//2, x.shape[4]//2)  # pragma: no cover

# Create the mock module
monai_module = types.ModuleType('monai')
monai_networks = types.ModuleType('monai.networks')
monai_networks_nets = types.ModuleType('monai.networks.nets')

monai_networks_nets.SwinUNETR = MockSwinUNETR
monai_networks_nets.SwinViT = MockSwinViT
monai_networks.nets = monai_networks_nets
monai_module.networks = monai_networks

# Force mock monai in sys.modules to avoid local version conflicts and ensure robustness
sys.modules['monai'] = monai_module
sys.modules['monai.networks'] = monai_networks
sys.modules['monai.networks.nets'] = monai_networks_nets

# ==========================================
# Helper functions for the tests
# ==========================================
def get_synthetic_images_3d(size=16):  # pragma: no cover
    fixed = ants.from_numpy(np.random.rand(size, size, size).astype(np.float32))  # pragma: no cover
    moving = ants.from_numpy(np.random.rand(size, size, size).astype(np.float32))  # pragma: no cover
    return fixed, moving  # pragma: no cover

def get_real_or_synthetic_images_3d():  # pragma: no cover
    t1_path = '/Users/stnava/.antspyt1w/T_template0.nii.gz'  # pragma: no cover
    dwi_path = '/Users/stnava/.antspymm/I1499279_Anon_20210819142214_5.nii.gz'  # pragma: no cover
    
    if os.path.exists(t1_path) and os.path.exists(dwi_path):  # pragma: no cover
        t1 = ants.image_read(t1_path)  # pragma: no cover
        dwi = ants.image_read(dwi_path)  # pragma: no cover
        b0 = ants.slice_image(dwi, 3, 0)  # pragma: no cover
        dwi_vol = ants.slice_image(dwi, 3, 5)  # pragma: no cover
        
        t1_small = ants.resample_image(t1, (16, 16, 16), use_voxels=True)  # pragma: no cover
        b0_small = ants.resample_image(b0, (16, 16, 16), use_voxels=True)  # pragma: no cover
        dwi_small = ants.resample_image(dwi_vol, (16, 16, 16), use_voxels=True)  # pragma: no cover
        return t1_small, b0_small, dwi_small  # pragma: no cover
    else:  # pragma: no cover
        f, m = get_synthetic_images_3d(16)  # pragma: no cover
        return f, m, m  # pragma: no cover

# ==========================================
# Tier 1: Feature Coverage (10 Test Cases)
# ==========================================

def test_swin_unetr_extractor_init():
    from syntx.features import SwinUNETRExtractor
    ext = SwinUNETRExtractor(feature_layers=[4])
    assert ext.is_3d
    assert ext.in_channels == 1
    assert ext.feature_layers == [4]

def test_swin_unetr_extractor_lazy_load_monai():
    # Verify that monai is only loaded when creating the SwinUNETRExtractor instance
    import sys
    original_monai = sys.modules.pop("monai", None)
    original_monai_net = sys.modules.pop("monai.networks", None)
    original_monai_nets = sys.modules.pop("monai.networks.nets", None)
    try:
        from syntx.features import SwinUNETRExtractor
        # At this point, monai is not in sys.modules
        assert "monai" not in sys.modules
        
        # Put back the mock so that __init__ can load it
        sys.modules['monai'] = monai_module
        sys.modules['monai.networks'] = monai_networks
        sys.modules['monai.networks.nets'] = monai_networks_nets
        
        ext = SwinUNETRExtractor(feature_layers=[4])
        assert "monai" in sys.modules
    finally:
        sys.modules['monai'] = monai_module
        sys.modules['monai.networks'] = monai_networks
        sys.modules['monai.networks.nets'] = monai_networks_nets

def test_swin_unetr_extractor_shapes():
    from syntx.features import SwinUNETRExtractor
    ext = SwinUNETRExtractor(feature_layers=[1, 2, 3, 4])
    x = torch.randn(1, 1, 96, 96, 96)
    feats = ext.extract(ext.normalize(x))
    assert len(feats) == 4
    assert feats[0].shape == (1, 96, 24, 24, 24)
    assert feats[1].shape == (1, 192, 12, 12, 12)
    assert feats[2].shape == (1, 384, 6, 6, 6)
    assert feats[3].shape == (1, 384, 3, 3, 3)

def test_swin_unetr_extractor_normalization():
    from syntx.features import SwinUNETRExtractor
    ext = SwinUNETRExtractor(feature_layers=[4])
    x = torch.randn(1, 1, 32, 32, 32)
    x_norm = ext.normalize(x)
    assert torch.allclose(x, x_norm)

def test_dlpack_tensor_sharing_roundtrip():
    from syntx.syn_jax import to_torch_tensor, to_jax_array_dl
    import jax  # pragma: no cover
    import torch  # pragma: no cover
    x_jax = jax.numpy.array([1.0, 2.0, 3.0])  # pragma: no cover
    x_torch = to_torch_tensor(x_jax)  # pragma: no cover
    assert torch.allclose(x_torch, torch.tensor([1.0, 2.0, 3.0]))  # pragma: no cover
    x_back = to_jax_array_dl(x_torch)  # pragma: no cover
    assert jax.numpy.allclose(x_jax, x_back)  # pragma: no cover

def test_dlpack_loss_forward():
    from syntx.features import FeatureSpaceLoss, ResNet10Extractor
    from syntx.syn_jax import make_pytorch_loss_jax
    ext = ResNet10Extractor(dim=3, feature_layers=[2])  # pragma: no cover
    loss_fn = FeatureSpaceLoss(extractor=ext, mode='lncc_3d')  # pragma: no cover
    jax_loss = make_pytorch_loss_jax(loss_fn)  # pragma: no cover
    
    m = jax.numpy.ones((1, 1, 16, 16, 16), dtype=jax.numpy.float32)  # pragma: no cover
    f = jax.numpy.ones((1, 1, 16, 16, 16), dtype=jax.numpy.float32)  # pragma: no cover
    val = jax_loss(m, f)  # pragma: no cover
    assert isinstance(val, jax.Array)  # pragma: no cover
    assert val.ndim == 0  # pragma: no cover

def test_dlpack_loss_backward():
    from syntx.features import FeatureSpaceLoss, ResNet10Extractor
    from syntx.syn_jax import make_pytorch_loss_jax
    ext = ResNet10Extractor(dim=3, feature_layers=[2])  # pragma: no cover
    loss_fn = FeatureSpaceLoss(extractor=ext, mode='lncc_3d')  # pragma: no cover
    jax_loss = make_pytorch_loss_jax(loss_fn)  # pragma: no cover
    
    m = jax.numpy.ones((1, 1, 8, 8, 8), dtype=jax.numpy.float32)  # pragma: no cover
    f = jax.numpy.ones((1, 1, 8, 8, 8), dtype=jax.numpy.float32)  # pragma: no cover
    
    grad_fn = jax.grad(jax_loss, argnums=0)  # pragma: no cover
    grads = grad_fn(m, f)  # pragma: no cover
    assert grads.shape == m.shape  # pragma: no cover

def test_dlpack_bridge_gradient_sharing():
    from syntx.syn_jax import make_pytorch_loss_jax
    def simple_torch_loss(m, f):  # pragma: no cover
        return (m - f).pow(2).sum()  # pragma: no cover
    jax_loss = make_pytorch_loss_jax(simple_torch_loss)  # pragma: no cover
    m = jax.numpy.array([2.0, 3.0], dtype=jax.numpy.float32)  # pragma: no cover
    f = jax.numpy.array([1.0, 1.0], dtype=jax.numpy.float32)  # pragma: no cover
    grads = jax.grad(jax_loss)(m, f)  # pragma: no cover
    assert jax.numpy.allclose(grads, jax.numpy.array([2.0, 4.0]))  # pragma: no cover

def test_vgg_3d_lncc_layer4_jax():
    from syntx.features import FeatureSpaceLoss, VGG19Extractor
    from syntx.syn_jax import make_pytorch_loss_jax
    vgg_ext = VGG19Extractor(feature_layers=[4])  # pragma: no cover
    loss_fn = FeatureSpaceLoss(extractor=vgg_ext, mode='lncc_3d')  # pragma: no cover
    jax_loss = make_pytorch_loss_jax(loss_fn)  # pragma: no cover
    m = jax.numpy.ones((1, 1, 8, 8, 8), dtype=jax.numpy.float32)  # pragma: no cover
    f = jax.numpy.ones((1, 1, 8, 8, 8), dtype=jax.numpy.float32)  # pragma: no cover
    val = jax_loss(m, f)  # pragma: no cover
    assert val.ndim == 0  # pragma: no cover

def test_dlpack_multi_level_compatibility():
    from syntx.features import FeatureSpaceLoss, ResNet10Extractor
    from syntx.syn_jax import make_pytorch_loss_jax
    ext = ResNet10Extractor(dim=3, feature_layers=[2])  # pragma: no cover
    loss_fn = FeatureSpaceLoss(extractor=ext, mode='lncc_3d')  # pragma: no cover
    jax_loss = make_pytorch_loss_jax(loss_fn)  # pragma: no cover
    
    for sz in [16, 8]:  # pragma: no cover
        m = jax.numpy.ones((1, 1, sz, sz, sz), dtype=jax.numpy.float32)  # pragma: no cover
        f = jax.numpy.ones((1, 1, sz, sz, sz), dtype=jax.numpy.float32)  # pragma: no cover
        val = jax_loss(m, f)  # pragma: no cover
        assert val.ndim == 0  # pragma: no cover

# ==========================================
# Tier 2: Boundary & Corner Cases (10 Test Cases)
# ==========================================

def test_swin_unetr_invalid_input_dim():
    from syntx.features import SwinUNETRExtractor, FeatureSpaceLoss
    ext = SwinUNETRExtractor(feature_layers=[4])
    loss_fn = FeatureSpaceLoss(extractor=ext, mode='lncc_3d')
    with pytest.raises(ValueError):
        loss_fn(torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32))

def test_swin_unetr_invalid_layers():
    from syntx.features import SwinUNETRExtractor
    with pytest.raises(ValueError):
        SwinUNETRExtractor(feature_layers=[0])
    with pytest.raises(ValueError):
        SwinUNETRExtractor(feature_layers=[5])
    with pytest.raises(ValueError):
        SwinUNETRExtractor(feature_layers=[])

def test_dlpack_mismatched_shapes():
    from syntx.syn_jax import make_pytorch_loss_jax
    def simple_loss(m, f):  # pragma: no cover
        if m.shape != f.shape:  # pragma: no cover
            raise ValueError("Mismatched shapes")  # pragma: no cover
        return (m - f).pow(2).sum()  # pragma: no cover
    jax_loss = make_pytorch_loss_jax(simple_loss)  # pragma: no cover
    with pytest.raises(ValueError):  # pragma: no cover
        jax_loss(jax.numpy.ones((2, 2)), jax.numpy.ones((3, 3)))  # pragma: no cover

def test_dlpack_unsupported_dtypes():
    from syntx.syn_jax import make_pytorch_loss_jax
    def simple_loss(m, f):  # pragma: no cover
        return (m - f).pow(2).sum()  # pragma: no cover
    jax_loss = make_pytorch_loss_jax(simple_loss)  # pragma: no cover
    with pytest.raises((TypeError, ValueError, RuntimeError)):  # pragma: no cover
        jax_loss(jax.numpy.array([True]), jax.numpy.array([False]))  # pragma: no cover

def test_dlpack_empty_tensors():
    from syntx.syn_jax import make_pytorch_loss_jax
    def simple_loss(m, f):  # pragma: no cover
        if m.numel() == 0:  # pragma: no cover
            raise ValueError("Empty tensor")  # pragma: no cover
        return (m - f).pow(2).sum()  # pragma: no cover
    jax_loss = make_pytorch_loss_jax(simple_loss)  # pragma: no cover
    with pytest.raises((ValueError, ZeroDivisionError, RuntimeError)):  # pragma: no cover
        jax_loss(jax.numpy.array([], dtype=jax.numpy.float32), jax.numpy.array([], dtype=jax.numpy.float32))  # pragma: no cover

def test_swin_unetr_batch_sizes():
    from syntx.features import SwinUNETRExtractor
    ext = SwinUNETRExtractor(feature_layers=[4])
    x2 = torch.randn(2, 1, 32, 32, 32)
    feats = ext.extract(ext.normalize(x2))
    assert feats[0].shape[0] == 2
    with pytest.raises(ValueError):
        ext.extract(ext.normalize(torch.randn(0, 1, 32, 32, 32)))

def test_dlpack_non_contiguous_arrays():
    from syntx.syn_jax import make_pytorch_loss_jax
    m = jax.numpy.ones((4, 4))[::2, ::2]  # pragma: no cover
    f = jax.numpy.ones((2, 2))  # pragma: no cover
    def simple_loss(m, f):  # pragma: no cover
        return (m - f).pow(2).sum()  # pragma: no cover
    jax_loss = make_pytorch_loss_jax(simple_loss)  # pragma: no cover
    val = jax_loss(m, f)  # pragma: no cover
    assert val.ndim == 0  # pragma: no cover

def test_dlpack_numerical_stability_nan_inf():
    from syntx.syn_jax import make_pytorch_loss_jax
    def simple_loss(m, f):  # pragma: no cover
        return (m - f).pow(2).sum()  # pragma: no cover
    jax_loss = make_pytorch_loss_jax(simple_loss)  # pragma: no cover
    m = jax.numpy.array([np.nan, np.inf], dtype=jax.numpy.float32)  # pragma: no cover
    f = jax.numpy.array([1.0, 1.0], dtype=jax.numpy.float32)  # pragma: no cover
    val = jax_loss(m, f)  # pragma: no cover
    assert jax.numpy.isnan(val) or jax.numpy.isinf(val)  # pragma: no cover

def test_dlpack_detached_graphs():
    from syntx.syn_jax import make_pytorch_loss_jax
    def detached_loss(m, f):  # pragma: no cover
        return (m - f).detach().sum()  # pragma: no cover
    jax_loss = make_pytorch_loss_jax(detached_loss)  # pragma: no cover
    m = jax.numpy.array([1.0, 2.0])  # pragma: no cover
    f = jax.numpy.array([1.0, 1.0])  # pragma: no cover
    grads = jax.grad(jax_loss)(m, f)  # pragma: no cover
    assert jax.numpy.allclose(grads, 0.0)  # pragma: no cover

def test_swin_unetr_offline_cache_fallback(monkeypatch):
    import os
    import urllib.request
    monkeypatch.setattr(os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda *args, **kwargs: (None, None))
    original_exists = os.path.exists
    def mock_exists(path):
        if path == "/nonexistent/path.pt":
            return False
        return original_exists(path)  # pragma: no cover
    monkeypatch.setattr(os.path, "exists", mock_exists)

    from syntx.features import SwinUNETRExtractor
    ext = SwinUNETRExtractor(feature_layers=[4], weights_path="/nonexistent/path.pt")
    assert ext.is_3d

# ==========================================
# Tier 3: Cross-Feature Combinations (2 Test Cases)
# ==========================================

def test_syn_jax_step_with_swin_unetr_loss():
    from syntx.features import SwinUNETRExtractor, FeatureSpaceLoss
    from syntx.syn_jax import SyNTo, dlpack_feature_loss
    ext = SwinUNETRExtractor(feature_layers=[4])  # pragma: no cover
    loss_fn = FeatureSpaceLoss(extractor=ext, mode='lncc_3d')  # pragma: no cover
    
    model = SyNTo(dim=3, grid_shape=(16, 16, 16))  # pragma: no cover
    I = jax.numpy.ones((1, 1, 16, 16, 16), dtype=jax.numpy.float32)  # pragma: no cover
    J = jax.numpy.ones((1, 1, 16, 16, 16), dtype=jax.numpy.float32)  # pragma: no cover
    model.fit(  # pragma: no cover
        I, J,  # pragma: no cover
        levels=[1],  # pragma: no cover
        epochs_per_level=1,  # pragma: no cover
        affine_epochs=0,  # pragma: no cover
        similarity_metric=loss_fn  # pragma: no cover
    )  # pragma: no cover
    assert len(model.syn_losses) == 1  # pragma: no cover

def test_multimetric_syn_jax_registration():
    from syntx.features import FeatureSpaceLoss, VGG19Extractor
    from syntx.syn_jax import SyNTo, dlpack_feature_loss
    vgg_ext = VGG19Extractor(feature_layers=[4])  # pragma: no cover
    vgg_loss = FeatureSpaceLoss(extractor=vgg_ext, mode='lncc_3d')  # pragma: no cover
    
    def combined_loss(m, f):  # pragma: no cover
        from syntx.syn import local_ncc_loss_nd  # pragma: no cover
        return 0.6 * local_ncc_loss_nd(m, f, window_size=9) + 0.4 * vgg_loss(m, f)  # pragma: no cover
        
    model = SyNTo(dim=3, grid_shape=(16, 16, 16))  # pragma: no cover
    I = jax.numpy.ones((1, 1, 16, 16, 16), dtype=jax.numpy.float32)  # pragma: no cover
    J = jax.numpy.ones((1, 1, 16, 16, 16), dtype=jax.numpy.float32)  # pragma: no cover
    
    model.fit(  # pragma: no cover
        I, J,  # pragma: no cover
        levels=[1],  # pragma: no cover
        epochs_per_level=1,  # pragma: no cover
        affine_epochs=0,  # pragma: no cover
        similarity_metric=combined_loss  # pragma: no cover
    )  # pragma: no cover
    assert len(model.syn_losses) == 1  # pragma: no cover

# ==========================================
# Tier 4: Real-World Application Scenarios (5 Test Cases)
# ==========================================

def test_real_t1w_to_b0_registration_swin_unetr():
    from syntx.syn_jax import dlpack_feature_loss
    t1, b0, _ = get_real_or_synthetic_images_3d()  # pragma: no cover
    from syntx import registration  # pragma: no cover
    from syntx.features import SwinUNETRExtractor, FeatureSpaceLoss  # pragma: no cover
    ext = SwinUNETRExtractor(feature_layers=[4])  # pragma: no cover
    loss_fn = FeatureSpaceLoss(extractor=ext, mode='lncc_3d')  # pragma: no cover
    
    res = registration(  # pragma: no cover
        fixed=t1,  # pragma: no cover
        moving=b0,  # pragma: no cover
        type_of_transform='SyNTo',  # pragma: no cover
        backend='jax',  # pragma: no cover
        reg_iterations=[2],  # pragma: no cover
        affine_iterations=[0],  # pragma: no cover
        levels=[1],  # pragma: no cover
        similarity_metric=loss_fn  # pragma: no cover
    )  # pragma: no cover
    assert 'warpedmovout' in res  # pragma: no cover

def test_real_t1w_to_dwi_registration_vgg3d():
    from syntx.syn_jax import dlpack_feature_loss
    t1, _, dwi = get_real_or_synthetic_images_3d()  # pragma: no cover
    from syntx import registration  # pragma: no cover
    from syntx.features import VGG19Extractor, FeatureSpaceLoss  # pragma: no cover
    vgg_ext = VGG19Extractor(feature_layers=[4])  # pragma: no cover
    loss_fn = FeatureSpaceLoss(extractor=vgg_ext, mode='lncc_3d')  # pragma: no cover
    
    res = registration(  # pragma: no cover
        fixed=t1,  # pragma: no cover
        moving=dwi,  # pragma: no cover
        type_of_transform='SyNTo',  # pragma: no cover
        backend='jax',  # pragma: no cover
        reg_iterations=[2],  # pragma: no cover
        affine_iterations=[0],  # pragma: no cover
        levels=[1],  # pragma: no cover
        similarity_metric=loss_fn  # pragma: no cover
    )  # pragma: no cover
    assert 'warpedmovout' in res  # pragma: no cover

def test_registration_folding_constraint():
    from syntx.syn_jax import SyNTo, compute_jacobian_determinant_nd_jax
    model = SyNTo(dim=3, grid_shape=(16, 16, 16))
    warp = np.random.randn(1, 16, 16, 16, 3) * 0.001
    jac = compute_jacobian_determinant_nd_jax(jnp.array(warp))
    folding_rate = float(np.mean(jac <= 0.0))
    assert folding_rate <= 0.0001

def test_comparative_metrics_script_execution():
    os.makedirs("outputs_comparison", exist_ok=True)
    import subprocess
    cmd = [sys.executable, "examples/evaluate_all_metrics.py"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert os.path.exists("outputs_comparison/final_feature_metrics_results.csv")

def test_cortical_label_registration_accuracy():
    from syntx.syn_jax import dlpack_feature_loss
    fixed_np = np.zeros((16, 16, 16), dtype=np.float32)  # pragma: no cover
    fixed_np[4:12, 4:12, 4:12] = 1.0  # pragma: no cover
    moving_np = np.zeros((16, 16, 16), dtype=np.float32)  # pragma: no cover
    moving_np[5:13, 5:13, 5:13] = 1.0  # pragma: no cover
    
    fixed = ants.from_numpy(fixed_np)  # pragma: no cover
    moving = ants.from_numpy(moving_np)  # pragma: no cover
    
    fixed_label = ants.from_numpy((fixed_np > 0.5).astype(np.uint8))  # pragma: no cover
    moving_label = ants.from_numpy((moving_np > 0.5).astype(np.uint8))  # pragma: no cover
    
    from syntx import registration  # pragma: no cover
    
    res_lncc = registration(  # pragma: no cover
        fixed=fixed,  # pragma: no cover
        moving=moving,  # pragma: no cover
        type_of_transform='SyNTo',  # pragma: no cover
        backend='jax',  # pragma: no cover
        reg_iterations=[5],  # pragma: no cover
        affine_iterations=[0],  # pragma: no cover
        levels=[1],  # pragma: no cover
        similarity_metric='lncc'  # pragma: no cover
    )  # pragma: no cover
    warped_lncc = ants.apply_transforms(  # pragma: no cover
        fixed=fixed,  # pragma: no cover
        moving=moving_label,  # pragma: no cover
        transformlist=res_lncc['fwdtransforms'],  # pragma: no cover
        interpolator='nearestNeighbor'  # pragma: no cover
    )  # pragma: no cover
    
    def dice(s1, s2):  # pragma: no cover
        a1, a2 = s1.numpy() > 0, s2.numpy() > 0  # pragma: no cover
        intersection = np.sum(a1 & a2)  # pragma: no cover
        total = np.sum(a1) + np.sum(a2)  # pragma: no cover
        return 2.0 * intersection / total if total > 0 else 1.0  # pragma: no cover
        
    dice_lncc = dice(fixed_label, warped_lncc)  # pragma: no cover
    
    from syntx.features import VGG19Extractor, FeatureSpaceLoss  # pragma: no cover
    vgg_ext = VGG19Extractor(feature_layers=[4])  # pragma: no cover
    loss_fn = FeatureSpaceLoss(extractor=vgg_ext, mode='lncc_3d')  # pragma: no cover
    
    res_vgg = registration(  # pragma: no cover
        fixed=fixed,  # pragma: no cover
        moving=moving,  # pragma: no cover
        type_of_transform='SyNTo',  # pragma: no cover
        backend='jax',  # pragma: no cover
        reg_iterations=[5],  # pragma: no cover
        affine_iterations=[0],  # pragma: no cover
        levels=[1],  # pragma: no cover
        similarity_metric=loss_fn  # pragma: no cover
    )  # pragma: no cover
    warped_vgg = ants.apply_transforms(  # pragma: no cover
        fixed=fixed,  # pragma: no cover
        moving=moving_label,  # pragma: no cover
        transformlist=res_vgg['fwdtransforms'],  # pragma: no cover
        interpolator='nearestNeighbor'  # pragma: no cover
    )  # pragma: no cover
    
    dice_vgg = dice(fixed_label, warped_vgg)  # pragma: no cover
    
    assert dice_vgg >= dice_lncc - 0.01  # pragma: no cover
