# SwinUNETRExtractor Handoff Report

## 1. Observation
1. **Existing Baseline pytest execution output:**
   ```
   TOTAL                     2010    155    92%
   ================== 41 passed, 6 skipped, 4 warnings in 54.02s ==================
   ```
2. **Interface contracts in `PROJECT.md` (lines 25-29):**
   ```
   ### `src/syntx/features.py` ↔ `src/syntx/syn.py`
   - **FeatureSpaceLoss JAX Integration**: `syn_jax.py` requires a DLPack wrapper class/function that accepts `FeatureSpaceLoss` instance, performs forward/backward passes in PyTorch, and returns JAX scalar losses and gradients.
   - **Swin UNETR Extractor**: A PyTorch subclass of `FeatureExtractor` in `features.py`. Signature: `SwinUNETRExtractor(feature_layers=[4])`. Must support `is_3d = True` and output features matching the specified layers.
   ```
3. **Peer Analysis findings for Swin ViT Input Shape Sensitivity (`explorer_swin_2/analysis.md` lines 110-135):**
   ```python
       def extract(self, x: torch.Tensor) -> list:
           # Get spatial dimensions of the input tensor
           spatial_shape = x.shape[2:]
           original_img_size = self.img_size
           
           # If input size doesn't match configured img_size, interpolate to img_size to avoid pos_embed shape errors
           if spatial_shape != tuple(original_img_size):
               x_input = F.interpolate(x, size=original_img_size, mode='trilinear', align_corners=True)
           else:
               x_input = x
   ```
4. **Peer Analysis findings for Checkpoint Prefix Cleaning (`explorer_swin_3/proposed_changes.patch` lines 83-91):**
   ```python
           # Strip "module." or "swinViT." prefix if loading directly into swinViT
           swinvit_state_dict = {}
           for k, v in state_dict.items():
               if k.startswith("module."):
                   k = k[7:]
               if k.startswith("swinViT."):
                   k = k[8:]
               swinvit_state_dict[k] = v
   ```

## 2. Logic Chain
1. `PROJECT.md` contracts require `SwinUNETRExtractor` to support `is_3d = True` and output features matching specified layers.
2. In MONAI's Swin ViT architecture, positional embeddings are sized to a static shape (`img_size` in the constructor). If an input shape differs (e.g. from downsampling inside PyTorch multi-resolution registration pyramids), PyTorch will crash due to shape mismatch during adding positional embeddings (Observation 3).
3. To support dynamic image sizes without crashing or modifying MONAI's core network structure, we must interpolate input volumes to the configured `img_size` prior to calling the backbone, and interpolate the extracted feature maps back to match the original resolution scale factor ($2^{layer + 1}$).
4. MONAI Model Zoo weights might contain nesting prefixes (`module.` or `swinViT.`) depending on pre-training and packaging. A sequential stripping check (Observation 4) cleans keys dynamically, ensuring they map cleanly to the backbone model parameters.
5. All codebase feature extractors subclass `FeatureExtractor` in `features.py`. We can implement `SwinUNETRExtractor` as a subclass with lazy loading, cached weight checking, dynamic shape handling, and sequential key cleansing to guarantee robust operation.

## 3. Caveats
- Since `monai` is not installed in the standard environment, all proposed unit tests rely on mocking. Shape and interpolation tests mock the underlying `swinViT` method call, which is suitable for unit testing but assumes MONAI's backbone returns 5 hidden states as documented.
- Downsampling scale factors are assumed to match $2^{layer + 1}$ for `hidden_states[layer]`.

## 4. Conclusion
Integrating MONAI's `SwinUNETRExtractor` requires a subclass of `FeatureExtractor` in `src/syntx/features.py` that utilizes:
- **Lazy Importing:** to avoid startup crashes if MONAI is missing.
- **Dynamic Input Resizing:** to bypass positional embedding shape crashes on non-standard volumes.
- **Sequential Key Cleansing:** to handle multiple nested checkpoint prefixes.
- **Unit Tests:** using standard mock structures to test without downloading large local checkpoint assets or requiring an active internet connection.

## 5. Verification Method
1. Inspect the implementation proposal in `analysis.md`.
2. Verify that mock-based tests are correctly registered and do not make external HTTP requests.
3. Run the unit tests via `pytest` to assert that they execute and pass:
   ```bash
   pytest tests/test_feature_networks.py
   ```
