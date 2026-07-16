# SwinUNETRExtractor Integration Analysis & Synthesis

## 1. Executive Summary
This analysis details the architectural design and code proposal for integrating MONAI's `SwinUNETR` 3D self-supervised transformer encoder into the `syntx` registration framework. Since `monai` is not a mandatory project dependency, the integration uses lazy loading and cached weights with graceful offline fallback. Crucially, we synthesize previous explorer findings to implement dynamic input/output interpolation, allowing SwinUNETR's fixed positional embeddings to handle arbitrary spatial resolutions in multi-resolution registration pyramids. All proposed changes comply with the Single Interpolation Policy and similarity metric guidelines in `GEMINI.md`.

---

## 2. Synthesis of Inputs
We catalog and synthesize the findings from codebase inspection and peer investigations (from `explorer_swin_2` and `explorer_swin_3`):

### 2.1 Input Catalog
* **Source A: `explorer_swin_2/analysis.md` (Confidence: High)**
  - Identified the positional embedding shape mismatch problem in MONAI's Swin ViT when input shapes vary (e.g., during pyramid scaling).
  - Proposed dynamic input/output shape interpolation to work around the fixed positional embedding constraint.
  - Proposed the `weights_path="random"` option to bypass downloads/loads during unit tests.
* **Source B: `explorer_swin_3/analysis.md` & `proposed_changes.patch` (Confidence: High)**
  - Described the sequential prefix stripping for state dict checkpoints (`module.` and `swinViT.`).
  - Proposed basic mock-based unit tests for environments where `monai` is not installed or networks are restricted.

### 2.2 Consensus Findings
1. **Inheritance & Properties:** The new `SwinUNETRExtractor` class must inherit from `FeatureExtractor` in `src/syntx/features.py`, setting `is_3d = True` and `in_channels = 1`.
2. **Lazy Loading:** `SwinUNETR` should be imported dynamically inside `SwinUNETRExtractor.__init__`. If `ImportError` is raised, it should output a clear instruction to install `monai`.
3. **Weight Cache & Zoo:** Pre-trained weights are cached locally at `~/.syntx_cache/model_swinvit.pt`. If missing, the model dynamically downloads the official MONAI SSL checkpoint.
4. **Integration Points:**
   - Expose class in `src/syntx/__init__.py`.
   - Add metric parser cases for `swinunetr` and `swin_unetr` in `src/syntx/syn.py`'s `SyNTo.fit`.

### 2.3 Resolved Conflicts
* **Fixed Positional Embeddings vs Variable Input Sizes:**
  - *Conflict:* `SwinUNETR`'s positional embeddings are fixed to `img_size` (e.g. `96x96x96`) during initialization. If input dimensions change during multi-resolution registration (e.g., coarse levels at `32x32x32`), passing the image directly will crash PyTorch due to a tensor shape mismatch.
  - *Resolution:* Adopt `explorer_swin_2`'s approach. In the `extract` function, if the input shape does not match `img_size`, we interpolate the input volume to `img_size` prior to `self.model.swinViT`, and then interpolate the output feature maps back to the expected downsampled resolution (factor of $2^{layer + 1}$).
* **Prefix Stripping Logic:**
  - *Conflict:* Checkpoints from MONAI SSL may contain key prefixes like `module.` or `swinViT.` depending on the training setup.
  - *Resolution:* Implement a sequential cleanup loop that checks and strips both prefixes if present, mapping weights cleanly onto `self.model.swinViT`.

### 2.4 Dissenting Views
No unresolved conflicts or dissenting views exist.

### 2.5 Gaps
* **Online Dependency during tests:** Tests must run without making external network calls (especially under our `CODE_ONLY` network constraint). Thus, our proposed unit tests must rely on mocking network calls and file lookups.

---

## 3. Precise Integration Plan
1. **`src/syntx/features.py`**:
   - Add `SwinUNETRExtractor` subclassing `FeatureExtractor`.
   - Perform lazy imports of `SwinUNETR` from `monai.networks.nets`.
   - Incorporate the dynamic weight download, cache path, and sequential prefix cleaning.
   - Implement `extract` with dynamic trilinear interpolation for non-standard input sizes.
2. **`src/syntx/__init__.py`**:
   - Import and append `SwinUNETRExtractor` to `__all__`.
3. **`src/syntx/syn.py`**:
   - Register the metric keys `'swinunetr'` and `'swin_unetr'` to instantiate `SwinUNETRExtractor` inside `SyNTo.fit`.
4. **`tests/test_feature_networks.py`**:
   - Add unit tests verifying mock-based shape handling, interpolation logic, lazy loading error raising, and weight loading/prefix stripping.

---

## 4. Proposed Code Changes

### 4.1 Changes to `src/syntx/features.py`
```python
class SwinUNETRExtractor(FeatureExtractor):
    """SwinUNETR 3D self-supervised encoder feature extractor with lazy loading and dynamic size resizing."""
    is_3d = True
    in_channels = 1

    def __init__(self, feature_layers=[4], weights_path=None, img_size=(96, 96, 96)):
        super().__init__()
        # Lazy import to avoid hard dependency on MONAI
        try:
            from monai.networks.nets import SwinUNETR
        except ImportError:
            raise ImportError(
                "MONAI is required to use SwinUNETRExtractor. "
                "Please install it using 'pip install monai'."
            )

        self.feature_layers = feature_layers
        self.img_size = img_size

        # Default SwinUNETR configuration for pre-trained weights
        self.model = SwinUNETR(
            img_size=self.img_size,
            in_channels=self.in_channels,
            out_channels=14,  # default out channels in SSL pretrained zoo
            feature_size=48,
            spatial_dims=3
        )

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
                        f"Failed to download Swin ViT weights from {url}: {e}. "
                        f"If you are in an offline or restricted network environment, "
                        f"please manually download the weights from {url} and place them at '{weights_path}'."
                    )

            # Load checkpoint, strip nested keys, and load into backbone
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

        # Freeze model parameters for extraction
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Grayscale volumes are already scaled.
        return x

    def extract(self, x: torch.Tensor) -> list:
        spatial_shape = x.shape[2:]
        original_img_size = self.img_size

        # Interpolate input if dimensions mismatch the configured img_size
        if spatial_shape != tuple(original_img_size):
            x_input = F.interpolate(x, size=original_img_size, mode='trilinear', align_corners=True)
        else:
            x_input = x

        hidden_states = self.model.swinViT(x_input)
        features = []
        for layer in self.feature_layers:
            if not (0 <= layer < len(hidden_states)):
                raise ValueError(f"SwinUNETRExtractor layer {layer} out of bounds. Valid layers: 0-{len(hidden_states)-1}")
            
            feat = hidden_states[layer]

            # If input was interpolated, interpolate the output feature map back to expected scale
            if spatial_shape != tuple(original_img_size):
                expected_shape = [max(1, s // (2 ** (layer + 1))) for s in spatial_shape]
                feat = F.interpolate(feat, size=expected_shape, mode='trilinear', align_corners=True)

            features.append(feat)

        return features
```

### 4.2 Changes to `src/syntx/__init__.py`
```python
# Import at top
from .features import FeatureSpaceLoss, VGG19Extractor, DINOv2Extractor, ResNet10Extractor, SwinUNETRExtractor

# Update __all__
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

### 4.3 Changes to `src/syntx/syn.py`
```python
            elif metric_name_lower in ['swin_unetr', 'swinunetr']:
                extractor = SwinUNETRExtractor(feature_layers=vgg_layers).to(device=device)
                self.loss_functions.append(FeatureSpaceLoss(
                    extractor=extractor, mode=vgg_mode, num_slices=kwargs.get('num_slices', 4), lncc_window=vgg_lncc_window_size
                ).to(device=device))
```

---

## 5. Proposed Unit Tests

```python
def test_swin_unetr_extractor_lazy_import():
    # If monai is not installed, initializing SwinUNETRExtractor must raise ImportError.
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


def test_swin_unetr_extractor_shapes():
    monai = pytest.importorskip("monai")
    import unittest.mock as mock
    
    with mock.patch("urllib.request.urlretrieve") as mock_retrieve, \
         mock.patch("torch.load") as mock_load, \
         mock.patch("os.path.exists", return_value=True), \
         mock.patch("os.makedirs"):
        
        mock_load.return_value = {"state_dict": {}}
        
        # Initialize extractor using weights_path="random" to bypass loading logic
        extractor = SwinUNETRExtractor(feature_layers=[2, 4], weights_path="random", img_size=(96, 96, 96))
        assert extractor.is_3d
        assert extractor.in_channels == 1
        
        x = torch.randn(1, 1, 96, 96, 96)
        
        dummy_hidden_states = [
            torch.randn(1, 48, 48, 48, 48), # layer 0 (/2)
            torch.randn(1, 96, 24, 24, 24), # layer 1 (/4)
            torch.randn(1, 192, 12, 12, 12), # layer 2 (/8)
            torch.randn(1, 384, 6, 6, 6),   # layer 3 (/16)
            torch.randn(1, 384, 3, 3, 3),   # layer 4 (/32)
        ]
        
        with mock.patch.object(extractor.model, "swinViT", return_value=dummy_hidden_states):
            feats = extractor.extract(x)
            assert len(feats) == 2
            assert feats[0].shape == (1, 192, 12, 12, 12)
            assert feats[1].shape == (1, 384, 3, 3, 3)


def test_swin_unetr_extractor_interpolation():
    monai = pytest.importorskip("monai")
    import unittest.mock as mock
    
    with mock.patch("urllib.request.urlretrieve"), \
         mock.patch("torch.load", return_value={"state_dict": {}}), \
         mock.patch("os.path.exists", return_value=True), \
         mock.patch("os.makedirs"):
        
        extractor = SwinUNETRExtractor(feature_layers=[4], weights_path="random", img_size=(96, 96, 96))
        
        dummy_hidden_states = [
            torch.randn(1, 48, 48, 48, 48),
            torch.randn(1, 96, 24, 24, 24),
            torch.randn(1, 192, 12, 12, 12),
            torch.randn(1, 384, 6, 6, 6),
            torch.randn(1, 384, 3, 3, 3), # Layer 4 output (3x3x3 for 96x96x96 input)
        ]
        
        with mock.patch.object(extractor.model, "swinViT", return_value=dummy_hidden_states) as mock_vit:
            # Pass input size of 64x64x64. Extractor should scale to 96x96x96 and output back to 64//32 = 2
            x_64 = torch.randn(1, 1, 64, 64, 64)
            feats = extractor.extract(x_64)
            
            assert len(feats) == 1
            assert feats[0].shape == (1, 384, 2, 2, 2)


def test_swin_unetr_weights_download_and_key_cleaning():
    monai = pytest.importorskip("monai")
    import unittest.mock as mock
    
    with mock.patch("os.path.exists", return_value=False), \
         mock.patch("os.makedirs") as mock_makedirs, \
         mock.patch("urllib.request.urlretrieve") as mock_urlretrieve, \
         mock.patch("os.rename") as mock_rename, \
         mock.patch("torch.load") as mock_torch_load:
             
        mock_state_dict = {
            'module.swinViT.patch_embed.proj.weight': torch.ones(1),
            'swinViT.layer1.weight': torch.zeros(1),
            'layer2.weight': torch.ones(2)
        }
        mock_torch_load.return_value = {'state_dict': mock_state_dict}
        
        extractor = SwinUNETRExtractor(feature_layers=[4])
        
        mock_makedirs.assert_called_once()
        mock_urlretrieve.assert_called_once()
        mock_rename.assert_called_once()
        
        # Verify that loaded state dict keys are successfully cleaned of "module." and "swinViT."
        load_state_dict_call = extractor.model.swinViT.load_state_dict.call_args
        assert load_state_dict_call is not None
        loaded_dict = load_state_dict_call[0][0]
        assert 'patch_embed.proj.weight' in loaded_dict
        assert 'layer1.weight' in loaded_dict
        assert 'layer2.weight' in loaded_dict
```

---

## 6. Compliance with GEMINI.md Guardrails

1. **Single Interpolation Policy:**
   - SwinUNETRExtractor operates directly in native space without generating intermediate file-based pre-warping during registration steps. The image resampling during pyramid registration optimization happens directly on the coordinate warping grids.
2. **Similarity Metric & VGG Feature Space Guidelines:**
   - Cortical registration tasks targeting high-accuracy require 3D LNCC with Layer 4. SwinUNETRExtractor supports native 3D LNCC metric operations, aligning perfectly with this requirement.
3. **Reporting and Visualization Guidelines:**
   - Comparative reports comparing SwinUNETRExtractor with VGG19, DINOv2, or ResNet-10 must show edge/region overlap, deformed coordinate grids, Jacobian determinant maps, and side-by-side deformed vs target images.
