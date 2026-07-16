# Mindboggle Evaluation Reference

## Introduction
This document serves as a reference for evaluating registration performance using the Mindboggle dataset. The Mindboggle dataset contains structural T1-weighted images and manually annotated DKT31 cortical labels, which provides a rigorous ground truth for measuring alignment accuracy (via label overlap).

## The Data Challenge: Disparate Physical Spaces
When evaluating `syntx` against classical `ants.registration`, we use pairs of images with significant physical header discrepancies. The Mindboggle dataset consists of 5 sub-datasets (Extra, MMRR, NKI-RS, NKI-TRT, OASIS-TRT). The two datasets with the largest spatial discrepancy are:

1. **OASIS-TRT-20** (Origin: `[-80, 128, -128]`, Spacing: `[1.0, 1.0, 1.0]`)
2. **MMRR-21** (Origin: `[202.8, 0, 0]`, Spacing: `[1.2, 1.0, 1.0]`)

Because these images reside hundreds of millimeters apart in physical space and possess different voxel resolutions, they serve as a critical stress test for the registration algorithm's spatial awareness and coordinate grid mapping. 

## Standard Evaluation Protocol
To benchmark `syntx` against the Mindboggle data, follow this standard protocol:

### 1. Load the Images and Ground-Truth Labels
The data is located at `/Users/stnava/data/mindboggle/volumes`.

```python
import os
import ants

base_path = '/Users/stnava/data/mindboggle/volumes'

# Fixed Image: OASIS-TRT-20-1
fixed_img_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 't1weighted_brain.nii.gz')
fixed_lbl_path = os.path.join(base_path, 'OASIS-TRT-20_volumes', 'OASIS-TRT-20-1', 'labels.DKT31.manual.nii.gz')

# Moving Image: MMRR-21-1
moving_img_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 't1weighted_brain.nii.gz')
moving_lbl_path = os.path.join(base_path, 'MMRR-21_volumes', 'MMRR-21-1', 'labels.DKT31.manual.nii.gz')

fixed_image = ants.image_read(fixed_img_path)
fixed_labels = ants.image_read(fixed_lbl_path)

moving_image = ants.image_read(moving_img_path)
moving_labels = ants.image_read(moving_lbl_path)
```

### 2. Run Registration
Execute the registration using `ants.registration` (baseline) or `syntx.syn.registration`. Ensure that the moving image is properly mapped into the fixed image's coordinate space.

### 3. Warp the Labels (Single Interpolation)
Apply the resulting forward transforms to the moving labels. You **must** use nearest neighbor interpolation (`nearestNeighbor`) to prevent floating-point blending of the integer DKT labels.

```python
warped_moving_labels = ants.apply_transforms(
    fixed=fixed_image,
    moving=moving_labels,
    transformlist=registration_result['fwdtransforms'],
    interpolator='nearestNeighbor'
)
```

### 4. Evaluate Label Overlap (DICE)
Use `ants.label_overlap_measures` to compute the Mean Overlap / Target Overlap (DICE score) between the ground-truth fixed labels and the warped moving labels.

```python
overlap_measures = ants.label_overlap_measures(
    source_image=fixed_labels, 
    target_image=warped_moving_labels
)

# You can aggregate the 'TargetOverlap' (DICE) across all valid structural labels 
# to acquire the final mean DICE score for the registration run.
```
