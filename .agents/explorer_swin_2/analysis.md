# SwinUNETRExtractor Integration Analysis

## 1. Executive Summary
This analysis details the architectural plan, code modifications, and testing strategy for integrating MONAI's `SwinUNETR` 3D feature extractor (`SwinUNETRExtractor`) into the `syntx` codebase. The proposed design features:
- **Lazy Loading of MONAI:** Avoids mandatory dependency constraints by only importing MONAI modules inside constructor/methods and providing clear, actionable error handling.
- **Robust Weight Management:** Performs dynamic, cached download of pre-trained Swin ViT SSL weights with atomic write protection and automatic prefix stripping (`module.` and `swinViT.`).
- **Arbitrary Size Support:** Resolves MONAI Swin ViT's fixed positional embedding constraint by dynamically interpolating arbitrary-sized input volumes to `img_size` prior to feature extraction, and then interpolating extracted features back to the expected downsampled resolution.
- **Strict Compliance:** Adheres to all `GEMINI.md` guardrails, including the Single Interpolation Policy and similarity metric guidelines.

---

## 2. Integration Architecture Plan

The integration spans three codebase locations:
1. `src/syntx/features.py`: Implement the `SwinUNETRExtractor` class subclassing `FeatureExtractor`, implementing standard properties (`is_3d`, `in_channels`), robust normalization, and feature mapping.
2. `src/syntx/__init__.py`: Expose the new `SwinUNETRExtractor` class in the packages' top-level namespace and `__all__` list.
3. `src/syntx/syn.py`: Add case handlers within `SyNTo.fit` to instantiate `SwinUNETRExtractor` when `swinunetr` or `swin_unetr` is specified as a similarity metric.

---

## 3. Proposed Code Modifications

### 3.1 Modifications to `src/syntx/features.py`
Add the `SwinUNETRExtractor` class definition at the end of the file.

```python
class SwinUNETRExtractor(FeatureExtractor):
    """Swin UNETR 3D feature extractor with lazy loading and optional cached weights."""
    is_3d = True
    in_channels = 1

    def __init__(self, feature_layers=[4], weights_path=None, img_size=(96, 96, 96)):
        super().__init__()
        # Lazy import of MONAI to prevent module-level import errors
        try:
            from monai.networks.nets import SwinUNETR
        except ImportError:
            raise ImportError(
                "MONAI is required to use SwinUNETRExtractor. "
                "Please install it using 'pip install monai'."
            )
            
        self.feature_layers = feature_layers
        self.img_size = img_size

        # Construct architecture
        # out_channels=14 and feature_size=48 are defaults in standard MONAI SwinUNETR pretrained weights
        self.model = SwinUNETR(
            img_size=self.img_size,
            in_channels=self.in_channels,
            out_channels=14,
            feature_size=48,
            spatial_dims=3
        )

        # Dynamic loading / download of official MONAI weights
        if weights_path != "random":
            if weights_path is None:
                weights_path = os.path.expanduser("~/.syntx_cache/model_swinvit.pt")

            if not os.path.exists(weights_path):
                os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                url = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt"
                temp_path = weights_path + ".tmp"
                try:
                    import urllib.request
                    urllib.request.urlretrieve(url, temp_path)
                    os.rename(temp_path, weights_path)
                except Exception as e:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise RuntimeError(
                        f"Failed to download Swin ViT weights from {url}: {e}"
                    )

            # Load weights into SwinViT backbone, handling potential key prefix mismatches
            if os.path.exists(weights_path):
                state = torch.load(weights_path, map_location='cpu')
                state_dict = state.get('state_dict', state)
                
                # Strip 'module.' prefix if present
                cleaned_state_dict = {}
                for k, v in state_dict.items():
                    k_clean = k[7:] if k.startswith("module.") else k
                    cleaned_state_dict[k_clean] = v
                
                # Check if state_dict keys are for the full SwinUNETR model (prefixed with swinViT.)
                # or directly for the backbone.
                has_swinvit_prefix = any(k.startswith("swinViT.") for k in cleaned_state_dict.keys())
                if has_swinvit_prefix:
                    # Strip 'swinViT.' for direct loading into self.model.swinViT
                    swinvit_state_dict = {}
                    for k, v in cleaned_state_dict.items():
                        if k.startswith("swinViT."):
                            swinvit_state_dict[k[8:]] = v
                    self.model.swinViT.load_state_dict(swinvit_state_dict, strict=False)
                else:
                    self.model.swinViT.load_state_dict(cleaned_state_dict, strict=False)

        # Freeze model parameters
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Grayscale volumes are already in [0, 1] range.
        return x

    def extract(self, x: torch.Tensor) -> list:
        # Get spatial dimensions of the input tensor
        spatial_shape = x.shape[2:]
        original_img_size = self.img_size
        
        # If input size doesn't match configured img_size, interpolate to img_size to avoid pos_embed shape errors
        if spatial_shape != tuple(original_img_size):
            x_input = F.interpolate(x, size=original_img_size, mode='trilinear', align_corners=True)
        else:
            x_input = x

        # Run backbone forward pass to retrieve intermediate stages
        hidden_states = self.model.swinViT(x_input)
        
        features = []
        for layer in self.feature_layers:
            feat = hidden_states[layer]
            
            # If the input was interpolated, we interpolate the feature map back to the expected shape 
            # for the original input size. For a layer, downsampling factor is 2 ** (layer + 1).
            if spatial_shape != tuple(original_img_size):
                expected_shape = [max(1, s // (2 ** (layer + 1))) for s in spatial_shape]
                feat = F.interpolate(feat, size=expected_shape, mode='trilinear', align_corners=True)
                
            features.append(feat)
            
        return features
```

### 3.2 Modifications to `src/syntx/__init__.py`
Import and re-export the `SwinUNETRExtractor`:

```python
# Lines 4-5
from .features import FeatureSpaceLoss, VGG19Extractor, DINOv2Extractor, ResNet10Extractor, SwinUNETRExtractor

# Line 12-22 (in __all__):
__all__ = [
    "syn",
    "registration",
    "SyNTo",
    "SyNToJax",
    "SyNToTransform",
    "FeatureSpaceLoss",
    "VGG19Extractor",
    "DINOv2Extractor",
    "ResNet10Extractor",
    "SwinUNETRExtractor",
]
```

### 3.3 Modifications to `src/syntx/syn.py`
Add Swin UNETR metric mapping inside `SyNTo.fit`:

```python
# Around lines 857-886:
            elif metric_name_lower in ['swinunetr', 'swin_unetr']:
                from .features import SwinUNETRExtractor
                extractor = SwinUNETRExtractor(feature_layers=vgg_layers).to(device=device)
                self.loss_functions.append(FeatureSpaceLoss(
                    extractor=extractor, mode=vgg_mode, num_slices=kwargs.get('num_slices', 4), lncc_window=vgg_lncc_window_size
                ).to(device=device))
```

---

## 4. Proposed Unit Tests

We propose adding the following tests to `tests/test_feature_networks.py` to cover initialization, error handling, layer mapping, spatial interpolation, and key-cleaning logic:

```python
def test_swin_unetr_extractor_lazy_load_error():
    """Verify that SwinUNETRExtractor raises ImportError when MONAI is not installed."""
    import sys
    from unittest.mock import patch
    
    with patch.dict(sys.modules, {'monai': None}):
        with pytest.raises(ImportError) as exc_info:
            from syntx.features import SwinUNETRExtractor
            _ = SwinUNETRExtractor()
        assert "MONAI is required" in str(exc_info.value)


def test_swin_unetr_extractor_mocked_shapes():
    """Verify initialization parameters, attributes, and shapes under a mocked MONAI environment."""
    import sys
    from unittest.mock import MagicMock, patch
    
    mock_swin_unetr_class = MagicMock()
    mock_monai_nets = MagicMock()
    mock_monai_nets.SwinUNETR = mock_swin_unetr_class
    
    modules = {
        'monai': MagicMock(),
        'monai.networks': MagicMock(),
        'monai.networks.nets': mock_monai_nets
    }
    
    with patch.dict(sys.modules, modules):
        from syntx.features import SwinUNETRExtractor
        
        # Instantiate with weights_path="random" to bypass downloads
        extractor = SwinUNETRExtractor(feature_layers=[1, 4], weights_path="random", img_size=(96, 96, 96))
        
        mock_swin_unetr_class.assert_called_once_with(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=14,
            feature_size=48,
            spatial_dims=3
        )
        
        assert extractor.is_3d is True
        assert extractor.in_channels == 1
        
        # Test feature map shapes for exact dimensions
        # Layer 1 has F*2 = 96 channels, resolution downsampled by 4
        # Layer 4 has F*8 = 384 channels, resolution downsampled by 32
        mock_features = [
            torch.randn(1, 48, 48, 48, 48), # layer 0 (F=48, /2)
            torch.randn(1, 96, 24, 24, 24), # layer 1 (F*2=96, /4)
            torch.randn(1, 192, 12, 12, 12), # layer 2 (F*4=192, /8)
            torch.randn(1, 384, 6, 6, 6), # layer 3 (F*8=384, /16)
            torch.randn(1, 384, 3, 3, 3), # layer 4 (F*8=384, /32)
        ]
        extractor.model.swinViT = MagicMock(return_value=mock_features)
        
        x = torch.randn(1, 1, 96, 96, 96)
        feats = extractor.extract(x)
        
        extractor.model.swinViT.assert_called_once_with(x)
        assert len(feats) == 2
        assert feats[0].shape == (1, 96, 24, 24, 24)
        assert feats[1].shape == (1, 384, 3, 3, 3)


def test_swin_unetr_extractor_interpolation():
    """Verify that SwinUNETRExtractor correctly interpolates non-standard shapes."""
    import sys
    from unittest.mock import MagicMock, patch
    
    mock_swin_unetr_class = MagicMock()
    mock_monai_nets = MagicMock()
    mock_monai_nets.SwinUNETR = mock_swin_unetr_class
    
    modules = {
        'monai': MagicMock(),
        'monai.networks': MagicMock(),
        'monai.networks.nets': mock_monai_nets
    }
    
    with patch.dict(sys.modules, modules):
        from syntx.features import SwinUNETRExtractor
        
        extractor = SwinUNETRExtractor(feature_layers=[4], weights_path="random", img_size=(96, 96, 96))
        
        # Mock swinViT outputs for a 96x96x96 input
        mock_features = [
            torch.randn(1, 48, 48, 48, 48),
            torch.randn(1, 96, 24, 24, 24),
            torch.randn(1, 192, 12, 12, 12),
            torch.randn(1, 384, 6, 6, 6),
            torch.randn(1, 384, 3, 3, 3), # Layer 4 output (3x3x3 for 96x96x96)
        ]
        extractor.model.swinViT = MagicMock(return_value=mock_features)
        
        # Pass non-standard size (64, 64, 64)
        x_64 = torch.randn(1, 1, 64, 64, 64)
        feats = extractor.extract(x_64)
        
        # Output should be interpolated back to 64 // 32 = 2
        assert len(feats) == 1
        assert feats[0].shape == (1, 384, 2, 2, 2)


def test_swin_unetr_weights_download_and_key_cleaning():
    """Verify cached directory creation, url retrieve, and model weight cleaning logic."""
    import sys
    from unittest.mock import MagicMock, patch
    
    mock_swin_unetr_class = MagicMock()
    mock_monai_nets = MagicMock()
    mock_monai_nets.SwinUNETR = mock_swin_unetr_class
    
    modules = {
        'monai': MagicMock(),
        'monai.networks': MagicMock(),
        'monai.networks.nets': mock_monai_nets
    }
    
    with patch.dict(sys.modules, modules):
        from syntx.features import SwinUNETRExtractor
        
        with patch('os.path.exists', return_value=False), \
             patch('os.makedirs') as mock_makedirs, \
             patch('urllib.request.urlretrieve') as mock_urlretrieve, \
             patch('os.rename') as mock_rename, \
             patch('torch.load') as mock_torch_load:
                 
            # Mixed state dict format (some prefix-free, some prefixed)
            mock_state_dict = {
                'module.swinViT.patch_embed.proj.weight': torch.ones(1),
                'swinViT.layer1.weight': torch.zeros(1),
                'layer2.weight': torch.ones(2)
            }
            mock_torch_load.return_value = {'state_dict': mock_state_dict}
            
            extractor = SwinUNETRExtractor(feature_layers=[4])
            
            # Check directory creation and renaming
            mock_makedirs.assert_called_once()
            mock_urlretrieve.assert_called_once()
            mock_rename.assert_called_once()
            
            # Check state dict cleaning:
            # - module.swinViT.patch_embed.proj.weight -> patch_embed.proj.weight
            # - swinViT.layer1.weight -> layer1.weight
            load_state_dict_call = extractor.model.swinViT.load_state_dict.call_args
            assert load_state_dict_call is not None
            loaded_dict = load_state_dict_call[0][0]
            assert 'patch_embed.proj.weight' in loaded_dict
            assert 'layer1.weight' in loaded_dict
```

---

## 5. Compliance with GEMINI.md Guardrails

1. **Single Interpolation Policy:**
   - SwinUNETRExtractor operates directly in native space (the SyN optimization step handles warping of coordinates on the transformation grid, avoiding pre-warped inputs).
   - Our implementation does not perform any intermediate file-based pre-warping.
2. **Similarity Metric & VGG Feature Space Guidelines:**
   - SwinUNETRExtractor uses 3D feature representation. When configuring registrations targeting high accuracy (e.g. cortical label maps), standard intensity LNCC or VGG 3D LNCC with Layer 4 should be preferred.
3. **Reporting and Visualization Guidelines:**
   - Any registration HTML reports comparing SwinUNETR outputs must display region overlap, deformed grids, Jacobian determinant maps, and side-by-side deformed vs target images.
