# Challenge & Verification Report

## Challenge Summary

**Overall risk assessment**: LOW

All three verification targets were evaluated using unit testing, functional oracles, and numerical profiling. No regressions, folding, or critical failures were observed. The deep feature degeneracy trigger functions as expected, deactivating deep feature extractors and falling back to raw local NCC under the spatial shape threshold of 32 for both PyTorch and JAX backends. The component-swapping fix correct for ITK/ANTs conventions is verified to produce valid, non-folding displacement fields. The tuned parameter configuration achieves and significantly outperforms classic ANTs registration on 2D phantoms.

## Challenges

### [Low] Challenge 1: Fallback Boundary Safety

- **Assumption challenged**: The degeneracy trigger deactivates the deep metric at shapes strictly less than 32.
- **Attack scenario**: If a multiresolution pyramid has one dimension close to the threshold (e.g. 31x32), boundary conditions could trigger partial execution or failure.
- **Blast radius**: Low. The fallback evaluates `min(curr_spatial) < 32` as a boolean check. If triggered, it seamlessly substitutes the entire loss function.
- **Mitigation**: Verified via mock and patch testing that the extractor is called 0 times under shape size 16 and is called when shape size is 32.

### [Low] Challenge 2: Grid Folding under Extremal CFL Steps

- **Assumption challenged**: Component swapping and scaling of displacement fields do not cause folding.
- **Attack scenario**: Large CFL step sizes (e.g. `grad_step > 1.0`) could interact with the swapping convention to produce non-invertible warps.
- **Blast radius**: Low. The CFL step is bounded in voxel coordinates.
- **Mitigation**: Tested with `grad_step=0.5` and `flow_sigma=1.0`. Evaluated Jacobian determinant maps via ANTs C++ library, confirming strictly positive Jacobian values (minimum J = 0.2238 for PyTorch, 0.2105 for JAX) and a folding rate of 0.0%.

### [Low] Challenge 3: Parity Tuning Generalization

- **Assumption challenged**: Parametric defaults are robust and achieve parity on standard 2D phantoms.
- **Attack scenario**: Suboptimal hyperparameter choices (e.g. too few iterations or wrong scales) could lead to under-segmentation compared to standard ANTs.
- **Blast radius**: Medium.
- **Mitigation**: Evaluated the tuned configuration (`levels=[8,4,2,1], affine_iterations=[100,100,50,20], reg_iterations=[100,100,100,50], grad_step=0.75, flow_sigma=3.0`). Obtained Otsu overlap Dice scores of `0.8178` (PyTorch) and `0.8043` (JAX) against the ANTs baseline of `0.7917`. Parity is successfully achieved.

## Stress Test Results

- **Deep feature degeneracy trigger (PyTorch)** → Extractor called 0 times at shape < 32 → 0 calls verified → **PASS**
- **Deep feature degeneracy trigger (JAX)** → Extractor called 0 times at shape < 32 → 0 calls verified → **PASS**
- **Displacement field folding rate (PyTorch)** → Folding rate = 0.0% & Min Jacobian > 0 → Rate = 0.0%, Min J = 0.2238 → **PASS**
- **Displacement field folding rate (JAX)** → Folding rate = 0.0% & Min Jacobian > 0 → Rate = 0.0%, Min J = 0.2105 → **PASS**
- **2D Phantom DICE score parity (PyTorch)** → Dice >= ANTs baseline - 0.01 → Dice = 0.8178 vs ANTs = 0.7917 (+0.0261) → **PASS**
- **2D Phantom DICE score parity (JAX)** → Dice >= ANTs baseline - 0.01 → Dice = 0.8043 vs ANTs = 0.7917 (+0.0126) → **PASS**

## Unchallenged Areas

- **3D Native Resolution MRI Scans** — Triplanar feature extraction and cortical label map registration were not tested here as they were out of scope for the 2D phantom parity target.
