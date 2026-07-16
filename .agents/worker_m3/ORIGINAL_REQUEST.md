## 2026-07-15T13:21:28Z
Implement the 2D generative cross-product space of intensity and shape changes under `src/syntx/generators.py` (or as appropriate).
Specifically:
1. Create `src/syntx/generators.py` (or other appropriate module) implementing a generator/pipeline (e.g. `CrossProductGenerator` or a functional equivalent) that generates image pairs with combinations of:
   - 6 intensity changes:
     1. Noise (additive Gaussian or Rician)
     2. Bias field (multiplicative low-frequency spatial inhomogeneity)
     3. Inhomogeneity (hyper/hypo-intense local Gaussian blob)
     4. Modality change (non-linear contrast mapping)
     5. Step function (quantized intensity step mapping)
     6. Missing data (local masked region set to 0)
   - 4 shape changes:
     1. Translation
     2. Rotation
     3. Affine
     4. Deformation (smooth non-rigid coordinate warping)
2. Ensure that for each shape change, the pipeline returns the generated moving/warped image and the ground truth displacement field.
3. Compute and return the ground-truth shape change magnitude defined by the L2 norm of the physical displacement field (using Grenander's metric deformation representation). Use the formula:
   - Convert normalized offsets to voxel units: u_vox = u_norm * (N - 1) / 2
   - Convert to physical displacement using spacing (in mm) and direction: u_phys = D * (u_vox * spacing)
   - Compute physical L2 norm: sqrt( delta_V * sum( ||u_phys(x)||^2 ) )
     where delta_V is the product of spacing in each dimension.
4. Ensure that the generative pipeline maintains >= 80% spatial overlap for all generated image pairs. (Hint: Keep deformation and translation magnitudes bounded to ensure this overlap).
5. Expose the generator appropriately in the package.
6. Write unit tests in `tests/test_generators.py` to assert:
   - The generative pipeline outputs a cross-product of the specified intensity and shape changes.
   - Every generated pair maintains >= 80% spatial overlap.
   - Ground truth magnitudes (physical L2 norm of the displacement field) are explicitly returned.
7. Run `pytest` to ensure all tests pass (existing ones plus the new test file).
8. Write your plans, progress, and handoffs in `/Users/stnava/code/syntx/.agents/worker_m3/`.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.
