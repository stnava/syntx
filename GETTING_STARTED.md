# Getting Started with Syntx

Syntx is a high-performance framework for 2D and 3D medical image registration with deep-learning-based feature-space similarity metrics. It supports both PyTorch and JAX optimization backends, allowing users to leverage pretrained models for accurate, multi-modal registration.

---

## Installation

```bash
pip install syntx
```

### Dependencies
Syntx requires:
- `numpy`, `scipy`, `pandas`
- `ants-python` (for core image processing and data loading)
- `torch` & `torchvision` (for PyTorch models and extractors)
- `jax` & `jaxlib` (for high-performance JAX acceleration)
- `monai` (optional, required to use the 3D transformer `SwinUNETRExtractor`)

---

## Core Features

### 1. Multi-Modal Feature Extractors
Syntx provides modular wrapper classes (subclassing `FeatureExtractor`) to extract multi-scale deep features from both 2D and 3D medical images:

- **`VGG19Extractor`**: 2D VGG19 features, with optional orthogonal slicing (`lncc` mode) to handle 3D inputs without native 3D convolutional weights.
- **`DINOv2Extractor`**: Pretrained self-supervised vision transformer (ViT-S/14 or ViT-B/14) features.
- **`ResNet10Extractor`**: Lightweight 2D/3D ResNet backbone, ideal for fast optimization.
- **`SwinUNETRExtractor`**: 3D self-supervised transformer encoder pretrained on massive clinical datasets (from MONAI Model Zoo), featuring dynamic scale interpolation.

### 2. PyTorch and JAX Co-Optimization (DLPack Bridge)
When using the JAX backend (`backend='jax'`), Syntx dynamically wraps PyTorch-based feature extractors with a zero-copy memory bridge using **DLPack** and JAX's **custom VJP (Vector-Jacobian Product)** autograd framework. Tensors and loss gradients are shared directly in memory without CPU roundtrips, allowing PyTorch feature backbones to guide JAX deformable SyN updates.

---

## Quick Start Examples

### PyTorch Deformable Registration with VGG19
To run a deformable registration using the PyTorch optimizer and VGG19-based LNCC similarity:

```python
import ants
import torch
from syntx import registration
from syntx.features import VGG19Extractor, FeatureSpaceLoss

# 1. Load Fixed and Moving scans
fixed_img = ants.image_read("fixed_t1w.nii.gz")
moving_img = ants.image_read("moving_t2w.nii.gz")

# 2. Instantiate VGG19 Extractor & Loss
# Using feature layer 4 (downsampled 16x) under 3D LNCC mode
vgg_ext = VGG19Extractor(feature_layers=[4])
similarity_metric = FeatureSpaceLoss(extractor=vgg_ext, mode='lncc_3d', lncc_window=9)

# 3. Execute Registration
output = registration(
    fixed=fixed_img,
    moving=moving_img,
    type_of_transform='SyNTo',
    backend='pytorch',
    reg_iterations=[40, 20, 0],
    levels=[4, 2, 1],
    similarity_metric=similarity_metric
)

# 4. Access registered outputs
warped_img = output['warpedmovout']
warp_field = output['warp_l2r']
```

### JAX Deformable Registration with Swin UNETR (DLPack)
To run a deformable registration on JAX, guided by a 3D Swin UNETR self-supervised transformer:

```python
import ants
from syntx import registration
from syntx.features import SwinUNETRExtractor, FeatureSpaceLoss

# 1. Load Scans
fixed_img = ants.image_read("fixed_t1w.nii.gz")
moving_img = ants.image_read("moving_dmri.nii.gz")

# 2. Instantiate Swin UNETR Extractor & Loss
swin_ext = SwinUNETRExtractor(feature_layers=[4], img_size=(96, 96, 96))
similarity_metric = FeatureSpaceLoss(extractor=swin_ext, mode='lncc_3d', lncc_window=5)

# 3. Execute Registration (JAX automatically uses zero-copy DLPack bridge)
output = registration(
    fixed=fixed_img,
    moving=moving_img,
    type_of_transform='SyNTo',
    backend='jax',
    reg_iterations=[20, 10, 0],
    levels=[4, 2, 1],
    similarity_metric=similarity_metric
)
```

### Composite/Multi-Metric Weighting
You can pass multiple metrics and weights to optimize both global intensity and deep structural features simultaneously:

```python
# Pass a list of similarity metrics and a corresponding list of weights
output = registration(
    fixed=fixed_img,
    moving=moving_img,
    type_of_transform='SyNTo',
    backend='jax',
    similarity_metric=['lncc', 'resnet10'],
    syn_metric_weights=[0.6, 0.4]
)
```

---

## Developer Guidelines & Rules

To maintain registration accuracy and numerical stability, all Syntx workflows adhere to the following guardrails:
1. **Single Interpolation Policy**: To prevent spatial blurring and loss of high-frequency detail, all transformation steps (initial affine + deformable SyN fields) are composed and applied to native space images in a single resampling step.
2. **VGG 3D Mode Requirement**: For high-accuracy brain label mapping, use **VGG 3D LNCC with Layer 4** (`vgg_mode='lncc_3d'`, `vgg_layers=[4]`). Coarser layer settings or orthogonal 2D slicing can incur up to a 4% drop in Dice score.
3. **No Intermediate Warping**: Do not pre-warp inputs or intermediates during registration optimization loop iterations.
