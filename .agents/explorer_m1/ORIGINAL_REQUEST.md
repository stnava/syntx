## 2026-07-15T13:13:43Z
Investigate the codebase of `syntx` (under `src/`, `tests/`, etc.) to:
1. Find any existing registration, metrics, and image comparison code.
2. Check the existing testing framework and how tests are run.
3. Understand what libraries/packages (like ANTs Py, PyTorch, JAX, scipy, numpy, etc.) are available.
4. Locate the VGG implementation if it exists, to see how it's used.
5. Propose a design for:
   - A suite of at least 64 unique, valid image comparison metrics in `syntx.image_compare` supporting both 2D and 3D images.
   - The 2D generative cross-product space of intensity changes (noise, bias, inhomogeneity, modality change, step function, missing data) and shape changes (translation, rotation, affine, deformation).
   - Grenander's metric deformation representation and how the ground-truth displacement field L2 norm is calculated.
   - The evaluation, HTML report generation (with edge/region overlap, deformed grids, Jacobian determinant maps, and side-by-side images), and documentation/example scripts.
6. Verify compliance with all constraints in GEMINI.md.

Write your findings and proposed design to `/Users/stnava/code/syntx/.agents/explorer_m1/handoff.md`.
