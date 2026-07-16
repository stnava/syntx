# Syntx Codebase Exploration and Integration Analysis

## Executive Summary
This report analyzes the `syntx` codebase to determine how to integrate MONAI's `SwinUNETR` 3D feature extractor and how to bridge PyTorch-based losses into the JAX SyN registration loop using DLPack for zero-copy tensor/gradient sharing. In addition, we present the baseline status of the existing test suite.

---

## 1. Existing Test Suite Baseline
We ran the existing test suite via `pytest` to establish a performance and coverage baseline.

* **Execution Command:** `pytest` (using Conda `base` environment with Python 3.13.2)
* **Results:**
  - **Total Tests:** 47 items collected, **41 passed, 6 skipped, 4 warnings** in 46.38 seconds.
  - **Code Coverage:**
    | Module | Statements | Missed | Coverage |
    |---|---|---|---|
    | `src/syntx/__init__.py` | 7 | 0 | 100% |
    | `src/syntx/features.py` | 232 | 13 | 94% |
    | `src/syntx/resnet.py` | 75 | 0 | 100% |
    | `src/syntx/syn.py` | 933 | 72 | 92% |
    | `src/syntx/syn_jax.py` | 667 | 70 | 90% |
    | `src/syntx/transform.py` | 96 | 0 | 100% |
    | **TOTAL** | **2010** | **155** | **92%** |

---

## 2. MONAI SwinUNETR Integration Analysis
To integrate MONAI's SwinUNETR 3D feature extractor, we inspect `src/syntx/features.py` and outline the design.

### 2.1 Integration Location
In `src/syntx/features.py`, we should define a class `SwinUNETRExtractor` that inherits from `FeatureExtractor`.

### 2.2 MONAI APIs Available
The `monai.networks.nets.SwinUNETR` class is a U-Net-like network with a Swin Transformer backbone (`SwinViT`) for 3D medical image segmentation.
* **Constructor Signature:**
  ```python
  SwinUNETR(
      img_size: Sequence[int] | int,
      in_channels: int,
      out_channels: int,
      feature_size: int = 48,
      use_checkpoint: bool = False,
      spatial_dims: int = 3,
      downsample: str = "merging",
      use_v2: bool = False,
  )
  ```
* **Backbone Feature Extraction:**
  The `SwinUNETR` instance stores its encoder backbone in `self.swinViT`. Calling `self.swinViT(x)` returns a list of 5 hidden states (outputs of the transformer stages at different resolutions):
  - `hidden_states[0]` (patch embedding output / resolution downsampled by 2)
  - `hidden_states[1]` (stage 1 output / resolution downsampled by 4)
  - `hidden_states[2]` (stage 2 output / resolution downsampled by 8)
  - `hidden_states[3]` (stage 3 output / resolution downsampled by 16)
  - `hidden_states[4]` (stage 4 output / resolution downsampled by 32)
  We can map `feature_layers=[1, 2, 3, 4]` directly to the corresponding hidden states (`hidden_states[1]`, `hidden_states[2]`, `hidden_states[3]`, `hidden_states[4]`).

### 2.3 Lazy and Cached Weight Loading Design
Since `monai` is not a mandatory dependency and may not be installed in all environments, we must implement a lazy and cached loading strategy:
1. **Lazy Importing:** Import `monai.networks.nets.SwinUNETR` inside `SwinUNETRExtractor.__init__` rather than globally at the top of `src/syntx/features.py`. If the import fails, raise a helpful `ImportError`.
2. **Weight Caching:** Check if the pre-trained weights file exists in the cache folder, defaulting to `~/.syntx_cache/model_swinvit.pt`.
3. **Dynamic Downloading:** If missing, download the official pre-trained Swin ViT SSL weights from the Project-MONAI Model Zoo (e.g. `https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt`) via `urllib.request.urlretrieve`.
4. **Backbone Loading:** Use `torch.load(..., map_location='cpu')` and load parameters into `self.model.swinViT` with `strict=False` (since SSL weights cover the SwinViT backbone but not the full SwinUNETR segmentation decoder).

### 2.4 Proposed Implementation Sketch
```python
class SwinUNETRExtractor(FeatureExtractor):
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
        self.model = SwinUNETR(
            img_size=self.img_size,
            in_channels=self.in_channels,
            out_channels=14,  # default out channels in SSL pretrained zoo
            feature_size=48,
            spatial_dims=3
        )

        # Cache path design
        if weights_path is None:
            weights_path = os.path.expanduser("~/.syntx_cache/model_swinvit.pt")

        # Dynamic download if weights do not exist
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
                raise RuntimeError(f"Failed to download Swin ViT weights from {url}: {e}")

        # Load weights into SwinViT backbone
        state = torch.load(weights_path, map_location='cpu')
        state_dict = state.get('state_dict', state)
        self.model.swinViT.load_state_dict(state_dict, strict=False)

        # Freeze model parameters
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Grayscale volumes are already scaled.
        return x

    def extract(self, x: torch.Tensor) -> list:
        # Run backbone forward pass to retrieve intermediate stages
        hidden_states = self.model.swinViT(x)
        features = []
        for layer in self.feature_layers:
            # Map layer 1 -> hidden_states[1], 2 -> [2], 3 -> [3], 4 -> [4]
            features.append(hidden_states[layer])
        return features
```

---

## 3. JAX Registration Loop and DLPack Bridge Analysis

### 3.1 JAX Registration Loop Structure
In `src/syntx/syn_jax.py`, the SyN registration is coordinated in `SyNTo.fit`:
1. **Multi-resolution Pyramids:** The inputs `fixed_image` and `moving_image` are downsampled using JAX image scaling (`interpolate_jax`) into lists `I_pyr` and `J_pyr` representing each pyramid level.
2. **Affine Optimization:** A JIT-compiled function `affine_step_jax` optimizes translation, rotation (`omega`), scale, and shear parameters via Rprop updates over `affine_epochs`.
3. **SyN Deformable Registration:**
   - Moving image is pre-aligned using the affine parameters to obtain `moving_affine`.
   - Pyramid resolution levels are iterated from coarse to fine.
   - Displacement fields (`warp_l2r`, `warp_r2l`, `warp_l2r_inv`, `warp_r2l_inv`) are upscaled to the current resolution (`upscale_field_jax`).
   - The loop runs `syn_step_jax` for each epoch.

### 3.2 Injecting FeatureSpaceLoss
In `syn_step_jax`, similarity loss is evaluated. Depending on `use_analytical_gradients`:
* **If `use_analytical_gradients=True`:**
  Analytical image gradients are computed and warped. The loss function `loss_mid_fn(im, jm)` calculates similarity w.r.t the warped images `I_mid` and `J_mid`:
  ```python
  def loss_mid_fn(im, jm):
      if similarity_metric == 'mattes_mi':
          return mattes_mi_loss_nd_jax(jm, im, num_bins=mattes_bins)
      elif similarity_metric == 'lncc':
          return local_ncc_loss_nd_jax(jm, im, window_size=window_size)
      else:
          # Inject PyTorch-based FeatureSpaceLoss wrapper here
          return feature_space_loss_jax_wrapped(im, jm)
  ```
* **If `use_analytical_gradients=False`:**
  Similarity loss is computed directly in `loss_warp_fn` w.r.t the warped displacement grids, allowing JAX AD to propagate gradients through the grid sampler:
  ```python
  def loss_warp_fn(wl, wr):
      ...
      im = jax_grid_sample(I_curr, phi_l, mode='bilinear', padding_mode='border')
      jm = jax_grid_sample(J_curr, phi_r, mode='bilinear', padding_mode='border')
      if similarity_metric == 'mattes_mi':
          return mattes_mi_loss_nd_jax(jm, im, num_bins=mattes_bins)
      elif similarity_metric == 'lncc':
          return local_ncc_loss_nd_jax(jm, im, window_size=window_size)
      else:
          # Inject PyTorch-based FeatureSpaceLoss wrapper here
          return feature_space_loss_jax_wrapped(im, jm)
  ```

### 3.3 Zero-Copy Tensor and Gradient Sharing via DLPack and custom VJP
Because PyTorch FeatureSpaceLoss uses PyTorch networks, it cannot be run directly within JAX JIT compilation. However, we can use `jax.pure_callback` combined with `jax.custom_vjp` to run the PyTorch code on host/device, while utilizing the standard Python array API DLPack protocol (`__dlpack__` / `from_dlpack`) for zero-copy memory access:

1. **JAX to PyTorch:** `torch.utils.dlpack.from_dlpack(x_jax)` provides a zero-copy PyTorch view of a JAX array.
2. **PyTorch to JAX:** `jax.dlpack.from_dlpack(x_torch)` (or falling back to numpy if DLPack encounters compatibility issues) returns a JAX array.
3. **Custom VJP Design:**
   - **Forward Pass:** Accepts JAX arrays, converts them zero-copy to PyTorch tensors, runs PyTorch feature extraction and loss computation, and returns the loss as a JAX array.
   - **Backward Pass:** Takes JAX gradients w.r.t the loss, converts them to PyTorch tensors, runs `.backward(gradient=g_torch)` in PyTorch, converts the resulting gradients w.r.t inputs (`im.grad` and `jm.grad`) back to JAX, and returns them.

#### Bridge Code Design:
```python
import jax
import jax.numpy as jnp
import torch
import torch.utils.dlpack
import numpy as np

def make_pytorch_loss_jax(pytorch_loss_fn):
    """Wraps a PyTorch module loss function into a JAX differentiable function."""
    
    def to_torch(x):
        if hasattr(x, '__dlpack__'):
            try:
                return torch.utils.dlpack.from_dlpack(x)
            except Exception:
                pass
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(np.array(x))

    def to_jax(t):
        try:
            return jax.dlpack.from_dlpack(t.detach())
        except Exception:
            return jnp.array(t.detach().cpu().numpy())

    def py_forward(im_jax, jm_jax):
        im_torch = to_torch(im_jax)
        jm_torch = to_torch(jm_jax)
        loss_torch = pytorch_loss_fn(im_torch, jm_torch)
        return to_jax(loss_torch)

    def py_backward(im_jax, jm_jax, g_jax):
        im_torch = to_torch(im_jax).detach().requires_grad_(True)
        jm_torch = to_torch(jm_jax).detach().requires_grad_(True)
        g_torch = to_torch(g_jax)
        
        loss_torch = pytorch_loss_fn(im_torch, jm_torch)
        loss_torch.backward(gradient=g_torch)
        
        grad_im = im_torch.grad if im_torch.grad is not None else torch.zeros_like(im_torch)
        grad_jm = jm_torch.grad if jm_torch.grad is not None else torch.zeros_like(jm_torch)
        
        return to_jax(grad_im), to_jax(grad_jm)

    @jax.custom_vjp
    def loss_jax(im, jm):
        return jax.pure_callback(
            py_forward,
            jax.ShapeDtypeStruct((), jnp.float32),
            im, jm
        )

    def loss_jax_fwd(im, jm):
        # Return primal output and inputs as residuals for backpropagation
        val = loss_jax(im, jm)
        return val, (im, jm)

    def loss_jax_bwd(res, g_jax):
        im, jm = res
        grad_im, grad_jm = jax.pure_callback(
            py_backward,
            (jax.ShapeDtypeStruct(im.shape, im.dtype), jax.ShapeDtypeStruct(jm.shape, jm.dtype)),
            im, jm, g_jax
        )
        return grad_im, grad_jm

    loss_jax.defvjp(loss_jax_fwd, loss_jax_bwd)
    return loss_jax
```

---

## 4. Guardrail Verification and Guidelines Compliance
1. **Single Interpolation Policy:** We verified that `src/syntx/syn_jax.py` does not perform intermediate file-based pre-warping during registration. All transformations are composed in middle space and applied in a single step using JAX grid sampling.
2. **Similarity Metric & VGG Feature Space Guidelines:** 
   - VGG 3D LNCC with Layer 4 must be prioritized for high-accuracy cortical label map registration tasks.
   - VGG 2D mode should not be recommended or defaulted to for high-accuracy registration.
3. **Reporting and Visualization Guidelines:**
   - Any registration reports or summaries must generate visualizations containing overlap, deformed grids, Jacobian determinant maps, and side-by-side comparisons of target/deformed images.
