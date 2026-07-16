# Verification and Challenge Report: Optimizer Parity & Stability

## 1. Observation

### Test Execution Commands & Outputs
- **Optimizer Unit Tests (`tests/test_optimizers.py`)**:
  Executed command: `pytest tests/test_optimizers.py`
  Result: **SUCCESS (8 passed in 16.14s)**
  Output excerpt:
  ```
  tests/test_optimizers.py ........                                        [100%]
  ============================== 8 passed in 16.14s ==============================
  ```

- **Full Test Suite & Coverage**:
  Executed command: `pytest --cov=src`
  Result: **SUCCESS (103 passed, 6 skipped, 6 warnings in 143.74s)**
  Coverage Report:
  ```
  Name                     Stmts   Miss  Cover   Missing
  ------------------------------------------------------
  src/syntx/__init__.py        7      0   100%
  src/syntx/features.py      319     19    94%   97, 129-130, 169, 368-369...
  src/syntx/resnet.py         75      0   100%
  src/syntx/syn.py          1137     95    92%   185, 267, 501-503, 822-826...
  src/syntx/syn_jax.py      1061    116    89%   60-61, 79-80, 83, 105-106...
  src/syntx/transform.py     100      0   100%
  ------------------------------------------------------
  TOTAL                     2699    230    91%
  ```

- **Optimizer Sweep Execution (`examples/run_optimizer_sweeps.py`)**:
  Executed command: `python examples/run_optimizer_sweeps.py`
  Result: **COMPLETED SUCCESSFULLY**
  Output excerpt:
  ```
  ANTs SyN (Mattes MI) Dice: 0.4323 | PyTorch SyNTo Mattes MI CFL Dice: 0.3956
  Parity Difference: 3.67%
  VERIFICATION FAILURE: 3D baseline parity regression > 1%!
  ```

### Key Quantitative Results from Sweep (CSV)
The sweep results were saved in `outputs_comparison/optimizer_sweep_results.csv`:
- **ANTs Baseline**: 3D Mattes MI Dice = **0.4323**, Folding = **0.0000%**.
- **CFL Optimizer (PyTorch)**: 3D Mattes MI Dice = **0.3956** (Folding = **0.0000%**).
- **CFL Optimizer (JAX)**: 3D Mattes MI Dice = **0.4111** (Folding = **0.0000%**).
- **SGD Optimizer (PyTorch & JAX)**: 3D Dice = **0.3984** / **0.3981** across all metrics (exactly matching the initial rigid baseline, indicating no optimization progress at `lr=1e-2`).
- **Adam Optimizer (PyTorch & JAX)**: 3D Dice = ~0.36 - 0.39 (Folding = **0.08% - 0.15%**).
- **L-BFGS Optimizer (PyTorch)**: 3D LNCC Dice = **0.1304** (Folding = **7.1652%**), 3D VGG19 Dice = **0.1282** (Folding = **5.0912%**), 3D Mattes MI Dice = **0.3984** (Folding = **0.0000%**, no updates).
- **L-BFGS Optimizer (JAX)**: 3D LNCC Dice = **0.3386** (Folding = **0.0108%**), 3D VGG19 Dice = **0.3393** (Folding = **0.0108%**), 3D Mattes MI Dice = **0.4045** (Folding = **0.0006%**).

---

## 2. Logic Chain

1. **ANTs Baseline Parity Failure**:
   - *Observation*: The ANTs Mattes MI baseline is `0.4323`, whereas PyTorch CFL Mattes MI achieves `0.3956` (difference of `3.67%`, exceeding the `1.0%` threshold).
   - *Reasoning*: Standard ANTs SyN applies Gaussian pre-smoothing to each level of the image pyramid to prevent aliasing and improve spatial convergence. In `src/syntx/syn.py` (lines 945-946), the image pyramid `I_pyr` and `J_pyr` are downsampled using simple trilinear/bilinear `F.interpolate` without any prior Gaussian smoothing. The lack of scale-space pre-smoothing leads to high-frequency aliasing and inferior coarse-level updates, creating a regression against ANTs baseline.

2. **SGD Optimizer Inactivity**:
   - *Observation*: Under `sgd` at `lr=1e-2`, all 3D registrations output a Dice of exactly `0.3984` in PyTorch and `0.3981` in JAX.
   - *Reasoning*: These values perfectly match the rigid initialization. Because SGD lacks a CFL-guided step-size constraint, a raw learning rate of `1e-2` is too small to make meaningful updates to the displacement fields, causing the optimizer to get stuck at the initial state.

3. **Adam Optimizer Folding**:
   - *Observation*: Adam at `lr=1e-2` achieves Dice scores of ~0.37-0.39, but introduces coordinate grid folding (folding rates up to 5.48% in 2D and 0.15% in 3D).
   - *Reasoning*: Adam updates the displacement fields without dynamically restricting the maximum step length to the grid spacing. In the absence of CFL step-size constraints, the parameter updates exceed physical voxel limits and fold the grid.

4. **PyTorch L-BFGS Instability & Grid Collapse**:
   - *Observation*: In PyTorch, L-BFGS + LNCC yields Dice = `0.1304` and Folding = `7.1652%`. In contrast, JAX L-BFGS + LNCC yields Dice = `0.3386` and Folding = `0.0108%`.
   - *Reasoning*: 
     - In PyTorch, `optimizer.step(closure)` evaluates the loss and gradients multiple times internally during its line search. 
     - The double-inversion and elastic regularization are executed outside/after `optimizer.step` (lines 1318-1335 in `src/syntx/syn.py`). Consequently, during PyTorch's internal line search evaluations, the trial warp parameters are updated directly *without* boundary masking, elastic smoothing, or double-inversion. This causes the trial displacement fields to severely fold, leading to numerical instabilities or convergence to highly distorted local minima.
     - In JAX, the L-BFGS optimizer uses Scipy's `minimize(..., method='L-BFGS-B', options={'maxiter': 1})`. Because `maxiter=1`, only one step is evaluated before control returns to the outer epoch loop, where boundary mask, elastic smoothing, and double-inversion are immediately applied. This keeps the JAX L-BFGS optimization physically regularized.

5. **PyTorch L-BFGS Mattes MI Inactivity**:
   - *Observation*: PyTorch L-BFGS + Mattes MI outputs a Dice of exactly `0.3984` (the rigid baseline) and `0%` folding.
   - *Reasoning*: The Mattes MI implementation under PyTorch L-BFGS failed to produce active updates, leaving the warp field at the identity initial state.

---

## 3. Caveats

- **Learning Rates**: We evaluated SGD and Adam at a fixed `lr=1e-2` and L-BFGS at `lr=1.0` (as defined in the sweep script). Performance might improve under alternative, carefully tuned hyperparameter sets, though the structural flaw in PyTorch's L-BFGS closure regularization would persist.
- **Hardware Backend**: GPU acceleration (MPS/CUDA) vs. CPU might introduce minor numerical differences, but the overall patterns (CFL superiority, PyTorch L-BFGS collapse, SGD inactivity) are algorithmic and backend-independent.

---

## 4. Conclusion

1. **ANTs Baseline Parity**: The PyTorch CFL baseline registration fails the 1% parity requirement, exhibiting a 3.67% regression against the ANTs baseline. This is primarily attributed to the lack of Gaussian pre-smoothing in `syntx`'s image pyramid downsampling.
2. **Optimizer Stability**:
   - **CFL** is the only optimizer that consistently guarantees 0% folding while achieving high registration accuracy.
   - **SGD** fails to optimize under the static learning rate.
   - **Adam** introduces unacceptable grid folding.
   - **PyTorch L-BFGS** is highly unstable and collapses due to the exclusion of boundary masking, elastic regularization, and double-inversion from the closure evaluation loop.
3. **GEMINI.md Conformance**: The generated sweeps script successfully outputs HTML dashboards with all required diagnostic visualizations (edge overlay, deformed grid, Jacobian determinant map, region overlap, and side-by-side images).

---

## 5. Verification Method

To verify these results independently, execute the following commands in the workspace root:

```bash
# 1. Run the optimizer unit tests
pytest tests/test_optimizers.py

# 2. Run the complete test suite with coverage
pytest --cov=src

# 3. Run the sweeps script
python examples/run_optimizer_sweeps.py

# 4. View results
cat outputs_comparison/optimizer_sweep_results.csv
open docs/optimizer_and_deep_feature_report.html
```

- **Invalidation Condition**: The verification is invalidated if the tests fail, if the sweep script crashes, or if the CSV values significantly deviate from the observed values.
