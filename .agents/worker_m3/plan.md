# Implementation Plan: 2D Generative Cross-Product Space

This plan outlines the design and implementation steps for building the 2D generative cross-product space under `src/syntx/generators.py` and verifying it with unit tests in `tests/test_generators.py`.

## Step 1: Design the `CrossProductGenerator` API
- The generator will take a 2D image (either PyTorch tensor `(1, 1, H, W)` or a utility that creates a default 2D phantom if none is provided).
- Support 6 intensity changes:
  - `noise`: Additive Gaussian or Rician noise.
  - `bias`: Multiplicative low-frequency spatial inhomogeneity.
  - `inhomogeneity`: Local Gaussian blob (hyper/hypo-intense).
  - `modality`: Non-linear contrast mapping.
  - `step`: Quantized intensity step mapping (binning).
  - `missing`: Local masked region set to 0.
- Support 4 shape changes:
  - `translation`: Simple shift.
  - `rotation`: SO(2) rotation.
  - `affine`: Affine transformation.
  - `deformation`: Smooth non-rigid warping.
- Return a tuple of:
  - `fixed_image`: The source image (can have optional intensity change).
  - `moving_image`: The warped source image with shape change and intensity change applied.
  - `displacement_field`: The ground truth normalized displacement field `(1, H, W, 2)`.
  - `magnitude`: The physical L2 norm of the displacement field.

## Step 2: Implement Physical L2 Norm Computation
- Implement the formula:
  - Convert normalized offsets `u_norm` to voxel units: `u_vox = u_norm * (N - 1) / 2`
  - Convert to physical displacement `u_phys` using spacing (in mm) and direction: `u_phys = D * (u_vox * spacing)` (computed as matrix multiplication: `(u_vox * spacing) @ D.T`)
  - Compute physical L2 norm: `sqrt( delta_V * sum( ||u_phys(x)||^2 ) )` where `delta_V` is the product of spacing in each dimension.

## Step 3: Ensure >= 80% Spatial Overlap Constraint
- Define spatial overlap as the Dice coefficient of a foreground mask.
- A foreground mask can be the non-zero region of the base image, or a central region (e.g., a circle/square of size 0.5 times the image width).
- Bounds on translation and deformation magnitudes:
  - Limit translation: maximum shift of 0.05 (5% of the normalized [-1, 1] range).
  - Limit rotation: maximum angle of 5-10 degrees (approx. 0.1-0.15 radians).
  - Limit affine scaling/shearing: scale in [0.95, 1.05], shear in [-0.05, 0.05].
  - Limit deformation: low-resolution grid (e.g., 5x5) with random displacements bounded by 0.05, smoothed with a Gaussian filter.
- Write a helper to verify Dice overlap >= 80%.

## Step 4: Write `src/syntx/generators.py`
- Put all logic for generating coordinate grids, applying transformations, applying intensity maps, and calculating metrics in this file.
- Handle spacing and direction metadata (default to spacing=(1.0, 1.0), direction=identity).

## Step 5: Expose `CrossProductGenerator` in package
- Edit `src/syntx/__init__.py` to export `CrossProductGenerator`.

## Step 6: Write unit tests in `tests/test_generators.py`
- Assert that the combinations work correctly.
- Assert that every generated pair maintains >= 80% spatial overlap.
- Assert that ground-truth physical L2 norm magnitudes of the displacement fields are returned and correctly computed.

## Step 7: Run pytest and check for lint
- Run `pytest` on all tests.
- Address any issues or warnings.
