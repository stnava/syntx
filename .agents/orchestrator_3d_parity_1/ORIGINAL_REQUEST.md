# Original User Request

## 2026-07-15T15:34:06Z

Achieve 3D registration parity (including DKT label overlap) between `syntx` and `ants.registration` for the Mindboggle dataset, while maintaining the existing 2D parity. The fix must ensure that the SyN optimization operates correctly in fixed physical space and composes with the affine mapping properly, strictly adhering to the single interpolation policy.

Working directory: /Users/stnava/code/syntx
Integrity mode: development

## Requirements

### R1. Native Physical Space Optimization
Rewrite the PyTorch and JAX SyN optimization loops so that the forward (`phi_1`) and inverse (`phi_2`) displacement fields operate natively in physical millimeter coordinates within the fixed image's domain, mirroring the ITK C++ reference implementation.

### R2. GPU Performance Balance
The implementation must remain blazing fast. Strike a balance between mathematical correctness in physical space and the efficiency of PyTorch/JAX tensor operations, avoiding heavy per-iteration overhead or CPU round-trips.

### R3. Affine Coordinate Composition
Correctly compose the initial affine transform to account for disparate fixed/moving physical spaces. The mapping to moving space must be strictly implemented as `y = A(phi_2_inv(phi_1(x)))`.

## Verification Resources
- Use `scratch/test_internal_dice.py` to evaluate the raw coordinate-mapping accuracy internally.
- `ants.label_overlap_measures` against the Mindboggle DKT manual labels.

## Acceptance Criteria

### Parity & Accuracy
- [ ] Programmatic DICE Verification: Running `synto` with an equivalent set of parameters must yield a label overlap metric (DICE score) that meets, exceeds, or is at least within 0.5% (0.005) of the standard `ants.registration` C++ baseline in both the 2D and 3D parity tests.

### Runtime Profiling
- [ ] Profiling Report: The agent must generate a profiling breakdown report confirming that the newly implemented physical space conversions do not dominate the GPU runtime during the optimization loop.
