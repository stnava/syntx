import sys
import os
import pytest
import torch
import torch.nn.functional as F
from unittest.mock import MagicMock, patch

# Define the mock classes
class MockSwinViT(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        return [
            torch.zeros(B, 48, D // 2, H // 2, W // 2, device=x.device),
            torch.zeros(B, 96, D // 4, H // 4, W // 4, device=x.device),
            torch.zeros(B, 192, D // 8, H // 8, W // 8, device=x.device),
            torch.zeros(B, 384, D // 16, H // 16, W // 16, device=x.device),
            torch.zeros(B, 384, D // 32, H // 32, W // 32, device=x.device),
        ]

class MockSwinUNETR(torch.nn.Module):
    def __init__(self, img_size=(96, 96, 96), in_channels=1, out_channels=14, feature_size=48, spatial_dims=3, *args, **kwargs):
        super().__init__()
        self.swinViT = MockSwinViT()

# Use a pytest fixture with module scope to mock monai during this module's test execution
@pytest.fixture(scope="module", autouse=True)
def mock_monai():
    original_modules = {}
    mocked_keys = ['monai', 'monai.networks', 'monai.networks.nets']
    
    for k in mocked_keys:
        if k in sys.modules:
            original_modules[k] = sys.modules[k]
            
    sys.modules['monai'] = MagicMock()
    sys.modules['monai.networks'] = MagicMock()
    sys.modules['monai.networks.nets'] = MagicMock()
    sys.modules['monai.networks.nets'].SwinUNETR = MockSwinUNETR
    
    yield
    
    for k in mocked_keys:
        if k in original_modules:
            sys.modules[k] = original_modules[k]
        else:
            sys.modules.pop(k, None)

# Now import the extractor
from syntx.features import SwinUNETRExtractor

def test_swin_unetr_extractor_empirical_interpolation():
    extractor = SwinUNETRExtractor(feature_layers=[1, 2, 3, 4], weights_path="random", img_size=(96, 96, 96))
    x_96 = torch.zeros(1, 1, 96, 96, 96)
    feats_96 = extractor.extract(x_96)
    
    assert feats_96[0].shape == (1, 96, 24, 24, 24)
    assert feats_96[1].shape == (1, 192, 12, 12, 12)
    assert feats_96[2].shape == (1, 384, 6, 6, 6)
    assert feats_96[3].shape == (1, 384, 3, 3, 3)

    x_64 = torch.zeros(1, 1, 64, 64, 64)
    feats_64 = extractor.extract(x_64)
    
    print("\nEmpirical Shape Analysis:")
    print(f"Input: 96^3 -> Output Layer 1 shape: {feats_96[0].shape} (downsampled 4x)")
    print(f"Input: 96^3 -> Output Layer 4 shape: {feats_96[3].shape} (downsampled 32x)")
    print(f"Input: 64^3 -> Output Layer 1 shape: {feats_64[0].shape} (downsampled {64 / feats_64[0].shape[2]:.1f}x)")
    print(f"Input: 64^3 -> Output Layer 4 shape: {feats_64[3].shape} (downsampled {64 / feats_64[3].shape[2]:.1f}x)")
    
    assert feats_64[0].shape == (1, 96, 16, 16, 16)
    assert feats_64[3].shape == (1, 384, 2, 2, 2)


def test_swin_unetr_extractor_layer_indexing():
    extractor = SwinUNETRExtractor(feature_layers=[1, 4], weights_path="random", img_size=(96, 96, 96))
    x = torch.zeros(1, 1, 96, 96, 96)
    with patch.object(extractor.model.swinViT, 'forward', return_value=[
        torch.zeros(1, 96, 24, 24, 24),
        torch.zeros(1, 192, 12, 12, 12),
        torch.zeros(1, 384, 6, 6, 6),
        torch.zeros(1, 384, 3, 3, 3)
    ]):
        feats = extractor.extract(x)
        assert feats[0].shape == (1, 96, 24, 24, 24)
        assert feats[1].shape == (1, 384, 3, 3, 3)


def test_offline_behavior_and_download_failures():
    with patch("os.path.exists", side_effect=[False, False]), \
         patch("urllib.request.urlretrieve", side_effect=Exception("Network unreachable")), \
         patch("os.makedirs"), \
         pytest.warns(UserWarning, match="Failed to download Swin ViT weights"):
        
        extractor = SwinUNETRExtractor(feature_layers=[4], weights_path=None)
        assert extractor.model is not None


def test_invalid_configurations():
    with pytest.raises(ValueError, match="feature_layers cannot be empty"):
        SwinUNETRExtractor(feature_layers=[])
        
    with pytest.raises(ValueError, match="Invalid layer index"):
        SwinUNETRExtractor(feature_layers=[0])
        
    with pytest.raises(ValueError, match="Invalid layer index"):
        SwinUNETRExtractor(feature_layers=[5])

    extractor = SwinUNETRExtractor(feature_layers=[4], weights_path="random")
    with pytest.raises(ValueError, match="Input must be a 5D tensor"):
        extractor.extract(torch.zeros(1, 1, 96, 96))
        
    with pytest.raises(ValueError, match="Batch size cannot be 0"):
        extractor.extract(torch.zeros(0, 1, 96, 96, 96))


def test_img_size_int_and_non_isotropic():
    extractor_int = SwinUNETRExtractor(feature_layers=[4], weights_path="random", img_size=96)
    x = torch.zeros(1, 1, 64, 64, 64)
    feats_int = extractor_int.extract(x)
    assert feats_int[0].shape == (1, 384, 2, 2, 2)

    extractor_non_iso = SwinUNETRExtractor(feature_layers=[4], weights_path="random", img_size=(96, 128, 64))
    feats = extractor_non_iso.extract(x)
    assert feats[0].shape == (1, 384, 2, 2, 2)
