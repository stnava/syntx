# Architectural Diagnosis & Performance Verification: Multi-Resolution CFL Step Scaling in `syntx.syn`

## Executive Summary

We investigated and benchmarked multi-resolution CFL step scaling in `syntx.syn` against ITK C++ `ants.registration` (SyN CC r=2). 

### Key Empirical Result
**YES! Enabling level shrink-ratio CFL step scaling (`level_cfl_voxels = cfl_voxels * shrink_ratio`) yields a MASSIVE +2.65% Cortical Gray Matter Dice Gain at default `grad_step = 0.15`, bringing `syntx.syn` into virtual 1-to-1 parity with ANTsPy out-of-the-box!**

---

## Head-to-Head Empirical Benchmark (`r16` $\rightarrow$ `r64`, `reg_iterations = [250, 250, 60]`)

| Engine & Configuration | Label 2 SymDice (Cortical GM) | Label 2+3 SymDice (Parenchyma) | Delta vs. ANTsPy Target | Performance Impact |
| :--- | :---: | :---: | :---: | :---: |
| **ITK `ants.registration` (SyN CC r=2)** | `0.786742` | `0.961619` | — | Target Reference |
| **`syntx.syn` UNFIXED** (Static `cfl_voxels = 0.15`) | `0.761189` | `0.955577` | `0.025554` | Stalled at Level 0 |
| **`syntx.syn` FIXED** (`cfl_voxels * shrink_ratio`) | **`0.787659`** | **`0.960766`** | **`0.000917`** ($<0.09\%$) | **+2.65% Dice Gain!** 🏆 |

---

## Why the Fix Yields a +2.65% Accuracy Improvement

1. **Resolving Coarse-Level Under-Stepping**:
   - Without shrink-ratio scaling, at Level 0 ($4\times$ downsampled), `grad_step = 0.15` in `syntx.syn` was taking $4\times$ smaller physical steps than ITK C++ ($0.15\text{ mm}$ vs. ITK's $0.60\text{ mm}$ physical step).
   - This caused coarse-level optimization to stall before global shape moves were resolved.

2. **Closing the Parity Gap**:
   - By scaling `level_cfl_voxels = cfl_voxels * (grid_shape / curr_spatial)`, Level 0 takes proper coarse physical steps ($0.60\text{ mm}$ per iteration), resolving global anatomical shifts at Level 0.
   - This provides Level 1 ($2\times$) and Level 2 ($1\times$) with a significantly better pre-aligned starting condition, increasing Cortical Gray Matter (Label 2) SymDice from **`0.761189` up to `0.787659`** and closing the ANTsPy parity gap to **`< 0.0009` Dice**!

---

## Implementation Diff (`src/syntx/syn.py` & `src/syntx/syn_jax.py`)

```python
# Scale level_cfl_voxels by current level shrink ratio relative to full resolution
shrink_ratio = float(self.grid_shape[0] / curr_spatial[0]) if hasattr(self, 'grid_shape') and self.grid_shape is not None and curr_spatial[0] > 0 else 1.0
level_cfl_voxels = cfl_voxels * shrink_ratio
```

---

## Provenance Artifacts

* **Benchmark Script**: [`scratch/compare_shrink_scaling_performance.py`](file:///Users/stnava/data/syntx/scratch/compare_shrink_scaling_performance.py)
* **Parity JSON Dataset**: [`docs/provenance/syn_ants_parity_map.json`](file:///Users/stnava/data/syntx/docs/provenance/syn_ants_parity_map.json)
* **HTML Report**: [`docs/syn_ants_parameter_parity_report.html`](file:///Users/stnava/data/syntx/docs/syn_ants_parameter_parity_report.html)
