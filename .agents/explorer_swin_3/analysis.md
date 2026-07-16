# SwinUNETRExtractor Integration Analysis

## 1. Executive Summary
This report presents a detailed plan, proposed code changes, and unit tests to integrate MONAI's `SwinUNETR` 3D self-supervised transformer encoder into the `syntx` registration framework. Since `monai` is not a default package dependency, the integration utilizes dynamic lazy importing and local weight caching to prevent module import errors for users without `monai`.

---

## 2. Integration Location & Architecture

The integration spans four main files in the `syntx` repository:
1. **`src/syntx/features.py`**: Define `SwinUNETRExtractor` subclass of `FeatureExtractor`.
2. **`src/syntx/__init__.py`**: Expose `SwinUNETRExtractor` to package-level imports.
3. **`src/syntx/syn.py`**: Parse similarity metric strings (`'swin_unetr'`, `'swinunetr'`) to instantiate and configure `SwinUNETRExtractor`.
4. **`tests/test_feature_networks.py`**: Add mock-based unit tests to verify shapes, lazy loading, and error handling without requiring active network downloads or large local assets during test execution.

---

## 3. SwinUNETRExtractor Design Specifications

### 3.1 Lazy Importing of MONAI
To prevent startup crashes for users who do not use SwinUNETR, we import `SwinUNETR` dynamically within `SwinUNETRExtractor.__init__`:
```python
try:
    from monai.networks.nets import SwinUNETR
except ImportError:
    raise ImportError(
        "MONAI is required to use SwinUNETRExtractor. "
        "Please install it using 'pip install monai'."
    )
```

### 3.2 Weight Caching & Local Offline Resilience
The extractor attempts to check if weights exist in the local cache directory `~/.syntx_cache/model_swinvit.pt`. If missing, it downloads the pre-trained weights from the official MONAI Model Zoo. If the download fails (e.g. in offline environments), it raises a clear error instructing the user how to place the weights manually:
```python
url = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt"
# ... download handling ...
except Exception as e:
    raise RuntimeError(
        f"Failed to download Swin ViT weights from {url}: {e}. "
        f"If you are in an offline or restricted network environment, "
        f"please manually download the weights from {url} and place them at '{weights_path}'."
    )
```

### 3.3 Dynamic Key Cleansing
Checkpoints from MONAI's self-supervised learning can contain prefixes like `module.` or `swinViT.` depending on the packaging. To ensure maximum compatibility when loading weights into the backbone:
```python
state = torch.load(weights_path, map_location='cpu')
state_dict = state.get('state_dict', state)

swinvit_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        k = k[7:]
    if k.startswith("swinViT."):
        k = k[8:]
    swinvit_state_dict[k] = v
    
self.model.swinViT.load_state_dict(swinvit_state_dict, strict=False)
```

### 3.4 Feature Map Resolution & Layer Mapping
The Swin Transformer backbone (`self.model.swinViT`) outputs five hidden states representing progressively downsampled representations of the input volume (from resolution scale factor of 2 down to 32):
* `hidden_states[0]`: shape `(B, feature_size, H/2, W/2, D/2)`
* `hidden_states[1]`: shape `(B, feature_size*2, H/4, W/4, D/4)`
* `hidden_states[2]`: shape `(B, feature_size*4, H/8, W/8, D/8)`
* `hidden_states[3]`: shape `(B, feature_size*8, H/16, W/16, D/16)`
* `hidden_states[4]`: shape `(B, feature_size*8, H/32, W/32, D/32)`

Our `SwinUNETRExtractor.extract(x)` will accept `feature_layers` (e.g. `[4]`) and return the matching hidden state tensors.

---

## 4. Proposed Code Changes

The complete code changes have been written to the unified patch file at:
`/Users/stnava/code/syntx/.agents/explorer_swin_3/proposed_changes.patch`

### 4.1 Modifications in `src/syntx/features.py`
Add `SwinUNETRExtractor` class:
```python
class SwinUNETRExtractor(FeatureExtractor):
    """SwinUNETR 3D self-supervised encoder feature extractor."""
    is_3d = True
    in_channels = 1

    def __init__(self, feature_layers=[4], weights_path=None, img_size=(96, 96, 96)):
        super().__init__()
        try:
            from monai.networks.nets import SwinUNETR
        except ImportError:
            raise ImportError(
                "MONAI is required to use SwinUNETRExtractor. "
                "Please install it using 'pip install monai'."
            )
            
        self.feature_layers = feature_layers
        self.img_size = img_size

        self.model = SwinUNETR(
            img_size=self.img_size,
            in_channels=self.in_channels,
            out_channels=14,
            feature_size=48,
            spatial_dims=3
        )

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
                    f"Failed to download Swin ViT weights from {url}: {e}. "
                    f"If you are in an offline or restricted network environment, "
                    f"please manually download the weights from {url} and place them at '{weights_path}'."
                )

        state = torch.load(weights_path, map_location='cpu')
        state_dict = state.get('state_dict', state)
        
        swinvit_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            if k.startswith("swinViT."):
                k = k[8:]
            swinvit_state_dict[k] = v
            
        self.model.swinViT.load_state_dict(swinvit_state_dict, strict=False)

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def extract(self, x: torch.Tensor) -> list:
        hidden_states = self.model.swinViT(x)
        features = []
        for layer in self.feature_layers:
            if 0 <= layer < len(hidden_states):
                features.append(hidden_states[layer])
            else:
                raise ValueError(f"SwinUNETRExtractor layer {layer} out of bounds. Valid layers: 0-{len(hidden_states)-1}")
        return features
```

### 4.2 Modifications in `src/syntx/syn.py`
Add SwinUNETR to list of metric choices inside `SyNTo.__init__` / registration loss parsing:
```python
            elif metric_name_lower in ['swin_unetr', 'swinunetr']:
                extractor = SwinUNETRExtractor(feature_layers=vgg_layers).to(device=device)
                self.loss_functions.append(FeatureSpaceLoss(
                    extractor=extractor, mode=vgg_mode, num_slices=kwargs.get('num_slices', 4), lncc_window=vgg_lncc_window_size
                ).to(device=device))
```

---

## 5. Propose Unit Tests

The proposed unit tests have been written to the unified patch file to append to `tests/test_feature_networks.py`.

### 5.1 Test 1: Lazy Import & Failure Gracefulness
This test simulates an environment without `monai` to verify `SwinUNETRExtractor` throws the expected `ImportError`.
```python
def test_swin_unetr_extractor_lazy_import():
    import sys
    monai_backup = sys.modules.get('monai')
    if 'monai' in sys.modules:
        del sys.modules['monai']
    try:
        with pytest.raises(ImportError) as exc_info:
            SwinUNETRExtractor(feature_layers=[4])
        assert "MONAI is required" in str(exc_info.value)
    finally:
        if monai_backup is not None:
            sys.modules['monai'] = monai_backup
```

### 5.2 Test 2: Shape and Output Dimension Testing
This test verifies `SwinUNETRExtractor` properties and feature extract tensor resolutions under mock inputs.
```python
def test_swin_unetr_extractor_shapes():
    monai = pytest.importorskip("monai")
    import unittest.mock as mock
    
    with mock.patch("urllib.request.urlretrieve"), \
         mock.patch("torch.load", return_value={"state_dict": {}}), \
         mock.patch("os.path.exists", return_value=True), \
         mock.patch("os.makedirs"):
        
        extractor = SwinUNETRExtractor(feature_layers=[2, 4], img_size=(96, 96, 96))
        assert extractor.is_3d
        assert extractor.in_channels == 1
        
        x = torch.randn(1, 1, 96, 96, 96)
        dummy_hidden_states = [
            torch.randn(1, 48, 48, 48, 48),
            torch.randn(1, 96, 24, 24, 24),
            torch.randn(1, 192, 12, 12, 12),
            torch.randn(1, 384, 6, 6, 6),
            torch.randn(1, 384, 3, 3, 3),
        ]
        
        with mock.patch.object(extractor.model, "swinViT", return_value=dummy_hidden_states):
            feats = extractor.extract(x)
            assert len(feats) == 2
            assert feats[0].shape == (1, 192, 12, 12, 12)
            assert feats[1].shape == (1, 384, 3, 3, 3)
```

### 5.3 Test 3: Loss Function Compatibility
This test verifies that `SwinUNETRExtractor` integrates smoothly with `FeatureSpaceLoss` in `lncc_3d` mode.
```python
def test_swin_unetr_feature_space_loss():
    monai = pytest.importorskip("monai")
    import unittest.mock as mock
    
    with mock.patch("urllib.request.urlretrieve"), \
         mock.patch("torch.load", return_value={"state_dict": {}}), \
         mock.patch("os.path.exists", return_value=True), \
         mock.patch("os.makedirs"):
        
        extractor = SwinUNETRExtractor(feature_layers=[4], img_size=(32, 32, 32))
        loss_fn = FeatureSpaceLoss(extractor=extractor, mode='lncc_3d')
        
        I_3d = torch.rand(1, 1, 32, 32, 32)
        J_3d = torch.rand(1, 1, 32, 32, 32)
        
        dummy_feat_in = [torch.randn(1, 384, 1, 1, 1)]
        dummy_feat_tg = [torch.randn(1, 384, 1, 1, 1)]
        
        with mock.patch.object(extractor, "extract") as mock_extract:
            mock_extract.side_effect = [dummy_feat_in, dummy_feat_tg]
            val = loss_fn(I_3d, J_3d)
            assert isinstance(val, torch.Tensor)
            assert val.ndim == 0
```

---

## 6. Guardrail and Guidelines Compliance

1. **Single Interpolation Policy Compliance**: SwinUNETRExtractor receives native-space images (or dynamically deformed native-space arrays within PyTorch) rather than file-based pre-warped inputs, preserving high-frequency boundary details as mandated by `GEMINI.md`.
2. **Similarity Metric Guidelines**: Consistent with requirements, SwinUNETRExtractor supports native 3D LNCC extraction, which aligns with standard intensity-based LNCC and VGG 3D LNCC with Layer 4 (the preferred high-accuracy modes).
3. **Visualization & Reporting Guidelines**: Future comparative reporting (e.g., in `examples/evaluate_all_metrics.py`) must display the specified structural images (edge overlap, deformed grids, Jacobian determinant maps, side-by-side images).
