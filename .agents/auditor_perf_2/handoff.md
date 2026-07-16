# Handoff Report - Forensic Audit of Optimizers and Pipelines

## Forensic Audit Report

**Work Product**: Optimizer Implementations & Sweeps Pipeline
**Profile**: General Project
**Verdict**: CLEAN

### Phase Results
- **Optimizer Genuineness**: PASS — Adam, SGD, L-BFGS, and CFL optimizers run actual registration optimization loops in both PyTorch (`src/syntx/syn.py`) and JAX (`src/syntx/syn_jax.py`) backends.
- **Cheating Detection**: PASS — No hardcoded test results, facade implementations, or mock evaluations found in implementation code.
- **Parity Validation Integrity**: PASS — The sweeps script `examples/run_optimizer_sweeps.py` computes DICE scores and folding rates from actual warped registration outputs.
- **Single Interpolation Policy**: PASS — Translation parameter is initialized directly on the parameters, grids are composed, and a single interpolation step is used for warping.

---

## 1. Observation

### Optimizer implementations (PyTorch)
In `src/syntx/syn.py`, the following lines implement actual optimizers for registration optimization:
- CFL: lines 1124-1207:
```python
                    # Real SyN: Pull both images to the midpoint domain
                    I_mid = F.grid_sample(I_curr, phi_l2r, padding_mode='border', align_corners=True)
                    J_mid = F.grid_sample(J_curr, phi_r2l, padding_mode='border', align_corners=True)
```
- SGD and Adam: lines 1208-1260:
```python
                elif optimizer_type in ['adam', 'sgd']:
                    optimizer.zero_grad()
                    phi_l2r = identity + warp_l2r
                    phi_r2l = identity + warp_r2l
                    
                    I_mid = F.grid_sample(I_curr, phi_l2r, padding_mode='border', align_corners=True)
                    J_mid = F.grid_sample(J_curr, phi_r2l, padding_mode='border', align_corners=True)
                    ...
                    optimizer.step()
```
- L-BFGS: lines 1261-1317:
```python
                elif optimizer_type == 'lbfgs':
                    last_loss = [0.0]
                    def closure():
                        optimizer.zero_grad()
                        # ...
                        loss.backward()
                        # ...
                        return loss
                    optimizer.step(closure)
```

### Optimizer implementations (JAX)
In `src/syntx/syn_jax.py`, the following lines implement actual optimizers for JAX backend:
- CFL: lines 1537-1543, calling JIT-compiled `syn_update_step_jax`.
- SGD: lines 1544-1549, calling JIT-compiled `sgd_update_step_jax` and `regularize_warp_fields_jax`.
- Adam: lines 1555-1566, calling JIT-compiled `adam_update_step_jax` and `regularize_warp_fields_jax`.
- L-BFGS: lines 1384-1473, using `scipy.optimize.minimize(..., method='L-BFGS-B')`.

### Sweep Parity Validation
In `examples/run_optimizer_sweeps.py`, lines 325-332 calculate overlap using real warped label maps:
```python
            warped_dkt = ants.apply_transforms(
                fixed=fixed_brain,
                moving=moving_dkt,
                transformlist=res['fwdtransforms'],
                interpolator='genericLabel'
            )
            dices = compute_multiregion_dice(fixed_dkt, warped_dkt)
            mean_dice = np.mean(list(dices.values()))
```
And baseline parity is verified at lines 388-390:
```python
        cfl_dice = cfl_mattes_3d['dice'].values[0]
        print(f"ANTs SyN (Mattes MI) Dice: {ants_mean_dice:.4f} | PyTorch SyNTo Mattes MI CFL Dice: {cfl_dice:.4f}")
        regression = ants_mean_dice - cfl_dice
```

### Single Interpolation Policy
In `src/syntx/syn.py` and `src/syntx/syn_jax.py`:
- Center of Mass: parameters are initialized directly on translation without pre-warping arrays (lines 889 and 1116).
- Grid Composition: multiple transforms are composed or passed directly as a list to `ants.apply_transforms` (lines 1026, 1205, 1393, 1901-1902 in `syn.py` and 1286, 1588-1589, 1901-1902 in `syn_jax.py`).

### Verification Output
Running all tests using `pytest` passes with 103 passed and 6 skipped:
```
============================ 103 passed, 6 skipped, 6 warnings in 144.11s (0:02:24) ============
```

## 2. Logic Chain
1. From checking `src/syntx/syn.py` and `src/syntx/syn_jax.py`, the optimizers Adam, SGD, L-BFGS, and CFL execute actual PyTorch optimization operations and JAX updates on displacement grids, rather than returning hardcoded results or empty loops.
2. From checking `examples/run_optimizer_sweeps.py`, the baseline parity is computed by registering low-resolution brains, warping DKT labels using `ants.apply_transforms`, computing Dice overlap, and comparing the resulting Dice overlap with the ANTs baseline. This uses computed values.
3. From checking `syn.py` and `syn_jax.py`, initial Center of Mass translation is updated on the parameters, intermediate grids are composed using `compose_grids`, and final warping calls a single `ants.apply_transforms` with the full list of transforms, adhering to the Single Interpolation Policy (no intermediate file-based pre-warping).
4. Running the full test suite results in all tests passing.
5. Therefore, the implementation is clean, correct, and follows all rules.

## 3. Caveats
No caveats.

## 4. Conclusion
The optimizers and sweeps pipeline are authentic, genuinely implemented, and compliant with both the Single Interpolation Policy and baseline parity validation rules. No integrity violations were found.

## 5. Verification Method
1. Run all unit tests to confirm correctness:
   `pytest`
2. Inspect `src/syntx/syn.py` to confirm that standard optimizer updates are applied directly to PyTorch tensors (`warp_l2r` and `warp_r2l`).
3. Inspect `src/syntx/syn_jax.py` to confirm standard JAX updater steps (`sgd_update_step_jax`, `adam_update_step_jax`, and `scipy.optimize.minimize` for L-BFGS).
4. Run `python examples/run_optimizer_sweeps.py` (requires download/cache files to be present, or uses local ones) to confirm that the sweeps dashboard compiles cleanly.
