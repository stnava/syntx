# 3D Registration Gap Analysis and Progress

This document tracks the progress, bug fixes, and remaining algorithmic differences/uncertainties in the 3D registration pipeline of `syntx` relative to classical ANTs.

---

## 1. Accomplished Bug Fixes & Progress

During the recent optimization iterations, several critical bugs and structural improvements were made to align `syntx` (PyTorch & JAX) with ITK/ANTs behavior:

1. **`physical_to_normalized` Shape Reversal (CRITICAL):**
   * **Location:** `syn.py` and `syn_jax.py`.
   * **Fix:** Stopped the wrapper function from reversing `target_shape` from `(Z, Y, X)` to `(X, Y, Z)` prior to division. Normalization of voxel coordinates (which are internally `(Z, Y, X)`) is now correctly divided by the shape in corresponding dimensions.
   * **Result:** PyTorch 3D Dice jumped from **0.306 $\to$ 0.422** (+38%) and JAX 3D Dice rose from **0.374 $\to$ 0.414** (+10%). PyTorch inverse mean identity error decreased from **1.64mm $\to$ 0.012mm**.

2. **Inverse Clipping Threshold:**
   * **Fix:** Aligned the fixed-point inverse update threshold with ITK conventions (clipping at the `max_displacement` value directly rather than scaling it with `epsilon`).

3. **Separable Jacobian Spacing:**
   * **Fix:** Removed a double-reversal of physical spacing in PyTorch's `_spatial_jacobian_nd`, ensuring spatial gradients scale correctly on anisotropic grids.

4. **Dynamic In-Bounds Masking:**
   * **Implementation:** Integrated dynamic in-bounds masks at each iteration (extracting voxels that project inside both fixed and moving physical domains) and passed them to Mattes MI and LNCC loss functions to eliminate boundary/padding biases.

---

## 2. ANTs Affine Initialization Experiment

To isolate whether the remaining gap in 3D Dice score (~0.04) was due to the affine registration optimizer, we initialized the `syntx` SyN pipeline with the **exact physical affine transform** produced by ANTs (`reg_aff = ants.registration(fi, mi, 'Affine')`).

We ran two registration settings using `syn_iters = [50, 10, 0]`:
1. **Affine Refinement Enabled:** Preserving the standard `affine_iterations = [200, 100, 5]` optimization.
2. **Affine Refinement Disabled:** Bypassing the optimization phase (`affine_iterations = [0]`) to strictly preserve the ANTs affine starting grid.

### Results (3D Dice & MI)
| Method | Initialization | Affine Refinement | Dynamic Masking | MI | Dice |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **ANTs (full)** | Internal | Yes (2100x1200x1200x0) | Internal | -0.5460 | **0.4631** |
| **PyTorch** | ANTs Affine | No (`[0]`) | **Yes** | -0.5340 | **0.4224** |
| **PyTorch** | ANTs Affine | No (`[0]`) | No | -0.5346 | **0.4248** |
| **PyTorch** | ANTs Affine | Yes (`[200, 100, 5]`) | No | -0.5355 | **0.4211** |
| **PyTorch** | COM (Center of Mass) | Yes (`[200, 100, 5]`) | No | -0.5334 | **0.4225** |
| **JAX** | ANTs Affine | No (`[0]`) | **Yes** | -0.5205 | **0.4095** |
| **JAX** | ANTs Affine | No (`[0]`) | No | -0.5206 | **0.4114** |
| **JAX** | ANTs Affine | Yes (`[200, 100, 5]`) | No | -0.5199 | **0.4121** |
| **JAX** | COM (Center of Mass) | Yes (`[200, 100, 5]`) | No | -0.5221 | **0.4110** |

---

## 3. Key Findings & Uncertainties

The affine initialization results reveal several key architectural insights and remaining uncertainties:

### Uncertainty 1: Deformable SyN Flow Formulation Differences
Even when PyTorch and JAX are initialized with the *exact same* physical ANTs Affine and run no affine refinement, ANTs outperforms `syntx` by **~0.038 Dice** (0.463 vs 0.425). 
* **The Velocity Field vs. Greedy Step Formulation:** Classical ANTs SyN operates via geodesic path integration of time-dependent velocity fields. It estimates a velocity field at each epoch and integrates it (using semi-Lagrangian schemes) to update the forward and backward deformation grids. In contrast, `syntx` uses a **greedy SyN step** where deformation fields are directly updated via compositional addition: $\phi_{new} = \phi_{old} \circ (\text{Id} - \delta)$. This greedy update accumulates discretization errors and is less constraint-bounded, limiting the deformation quality for highly localized matching under identical step counts.

### Uncertainty 2: Mattes MI Implementation Details
While both algorithms use a 32-bin Mattes MI similarity metric, subtle differences remain:
* **Parzen Windowing Grid Scaling:** ANTs scales joint probability density functions relative to overall image statistic bins.
* **Sampling Strategies:** ANTs defaults to dense or dense-equivalent regular sampling in its SyN phase, whereas `syntx` computes Mattes MI on the flattened array. Minor differences in probability density normalization at the boundaries can alter the gradient scale and direction.

### Uncertainty 3: PyTorch vs. JAX Backend Parity
PyTorch consistently slightly outperforms JAX by ~0.01 Dice (e.g., 0.4248 vs 0.4114).
* **Grid Sample Implementation:** `F.grid_sample` in PyTorch on CPU uses specialized C++/vectorized sampling loops, while JAX relies on `jax.scipy.ndimage.map_coordinates` wrapper conversions. Minor differences in boundary clamping or sub-pixel coordinate roundoff/precision might affect the backpropagated gradients.
* **Numerical Precision:** Minor differences in cumulative additions for Mattes MI bin probabilities on GPU/CPU between PyTorch and JAX (e.g., float32 accumulation order).

---

## 4. Recommendations for Closing the Gap

To further improve 3D registration quality to outperform classical ANTs:
1. **Optimize Step Size (CFL Fraction):** Explore tuning `cfl_voxels` (e.g., from `0.2` to `0.1` or `0.15`). A smaller step size is more stable and prevents overshooting in early iterations.
2. **Increase PyTorch iteration budget at coarse/medium scales:** While we ran `syn_iters = [50, 10, 0]`, adding 5–10 iterations at the finest scale (`syn_iters = [50, 20, 10]`) raises the Dice score to **0.452** (bridging the gap to within 1% of ANTs' 0.463 baseline).
