# 2D Continuum Mechanics Autotuning Report (`r16_r64`)

**Date:** 2026-10-07  
**Dataset:** ANTs `r16` (fixed) -> `r64` (moving) 2D axial brain slices  
**Labels:** 3-class Otsu segmentation (Mean Sørensen-Dice across Class 2 cortical GM & Class 3 WM)  
**Preprocessing:** Foreground 2–98% percentile intensity normalization to [0, 1]  
**Device:** Apple Silicon GPU (MPS)  
**Raw JSON Record:** [`docs/provenance/tuning/continuum_2d_autotune_verified.json`](file:///Users/stnava/code/syntx/docs/provenance/tuning/continuum_2d_autotune_verified.json)

---

## 1. Reference Baselines

| Method | Dice | Runtime (s) | Status |
| :--- | :---: | :---: | :---: |
| **ANTsPy C++ SyN Reference** | **0.8143** | 2.28s | Baseline Reference |
| **Sobolev SyN (`step=0.25, alpha=1.5, fast_smooth=True`)** | **0.8225** | 5.16s | Standard syntx Baseline (+0.0082 vs ANTs) |

---

## 2. Top-Performing Continuum Regularizers

| Rank | Model | Parameters | Dice | Gain vs. ANTs | Gain vs. Sobolev |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **1** | **`div_curl`** | $\text{step}=0.35, \beta=3.0, \gamma=0.5, \text{env}=1.0$ | **0.8311** | **+0.0168** | **+0.0086** |
| 2 | `div_curl` | $\text{step}=0.25, \beta=3.0, \gamma=0.5, \text{env}=1.0$ | 0.8307 | +0.0164 | +0.0082 |
| 3 | `div_curl` | $\text{step}=0.25, \beta=5.0, \gamma=0.5, \text{env}=1.0$ | 0.8305 | +0.0162 | +0.0080 |
| **4** | **`hyperelastic`** | $\text{step}=0.35, K=5.0, \text{env}=1.0$ | **0.8287** | **+0.0144** | **+0.0062** |
| **5** | **`navier`** | $\text{step}=0.35, \nu=0.35, s=2.0$ | **0.8283** | **+0.0140** | **+0.0058** |
| 6 | `navier` | $\text{step}=0.25, \nu=0.35, s=1.5$ | 0.8275 | +0.0132 | +0.0050 |
| 7 | `hyperelastic` | $\text{step}=0.25, K=10.0, \text{env}=1.0$ | 0.8275 | +0.0132 | +0.0050 |
| 8 | `hyperelastic` | $\text{step}=0.25, K=3.0, \text{env}=1.0$ | 0.8275 | +0.0132 | +0.0050 |
| 9 | `sobolev` | $\text{step}=0.25, \alpha=1.5, \text{fast\_smooth}=\text{True}$ | 0.8225 | +0.0082 | 0.0000 |
| 10 | `solenoidal` | $\text{step}=0.35, \alpha=1.5, s=1.5$ | 0.7785 | -0.0358 | -0.0440 |

---

## 3. Key Findings

1. **Biharmonic Spectral Envelope ($\text{envelope\_power}=1.0$):**  
   Across all decoupled and continuum models, $H^2$ biharmonic decay ($\text{envelope\_power}=1.0$) consistently outperformed tri-harmonic $H^3$ damping ($\text{envelope\_power}=2.0$) by $+0.015$ to $+0.020$ Dice, preventing over-regularization at fine scales while maintaining smooth deformation fields.
2. **Shear Permissiveness vs. Dilatation Resistance:**  
   The decoupled `div_curl` regularizer achieved the highest Dice score (**0.8311**), outperforming both isotropic Sobolev and ANTs C++. Relaxing shear resistance ($\gamma = 0.5$) while stiffening volume dilatation resistance ($\beta = 3.0$) matches physiological tissue deformation during registration.
