# Continuum Top 5 90-Pair Benchmark: Provenance Manifest

**Benchmark Identifier:** `continuum_top5_90pair_20261007`  
**Execution Date:** 2026-10-07  
**Execution Mode:** Subprocess-isolated, randomized 90-pair population cohort  
**Primary Models:** `['sobolev', 'syn_divcurl', 'syn_hyperelastic', 'syn_navier', 'gaussian']` vs. `ants` (ANTs C++ SyN Reference)

---

## 1. Machine & Environment Provenance

* **Host Platform:** macOS (Darwin 24.x, Apple Silicon M-series)
* **Compute Backend:** PyTorch MPS (Metal Performance Shaders) / Accelerate C++
* **Process Model:** Standalone Python subprocess per (pair, model) run to guarantee heap and Metal buffer reclamation
* **Python Version:** 3.13.2
* **Git Commit:** `85edc8d484` (branch `main`)
* **Git Diff Hash:** `5c56b99e47bcd958320074ee9fd44e0e47a261705797c21c92865435228f939a`
* **Syntx Version:** `6.1.0`
* **Dataset Directory:** `/Users/stnava/data/mindboggle/volumes`
* **Cohort Definition:** `examples/pairs.csv` (90 distinct pairs across 5 sub-cohorts)

---

## 2. Invariants & Preprocessing Pipeline

* **Preprocessing:**
  * N4 Bias Field Correction: `use_n4=True` (ANTsTorch N4, cached under `/Users/stnava/data/mindboggle/volumes/.n4_cache`)
  * Intensity Normalization: Foreground 2–98% percentile normalization to $[0, 1]$
  * Denoising: `denoise=False`
* **Affine Seeding:**
  * Native PyTorch Mattes-MI multi-start optimizer (`syntx.robust_affine`, `mode='auto'`, backend key `pt7`)
  * Pre-computed, verified, and shared identically across all deformable arms
* **Single Interpolation:**
  * Forward transform list: `[deformable_warp, affine_mat]` applied in a single resampling pass
  * Symmetrical inverse transform list: `[affine_mat (inverted), inverse_warp]`
  * Labels interpolated with `nearestNeighbor` (Sørensen-Dice metric over 62 DKT31 cortical structures)
* **Random Ordering:**
  * Deterministic permutation order generated via `np.random.RandomState(42).permutation(90)`
  * Initial pairs in permutation order: `[40, 22, 55, 70, 0, 26, 39, 65, 10, 44, 5, 83, 75, 41, 19, 63, 18, 28, ...]`

---

## 3. Top 5 Parameter Configurations

All parameters locked from empirical 2D/3D autotuning on `mbhard` (Pair 44):

| Model | Regularizer | Alpha ($\alpha$) | Step Size | Modulus / Ratio | Decay Order | Filter Type | Iterations |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`syn_divcurl`** | `div_curl` | 1.5 | 0.30 | $\beta=4.0, \gamma=0.8$ | $H^2$ ($\text{env}=1.0$) | Decoupled Helmholtz | `[100, 100, 20]` |
| **`syn_hyperelastic`** | `hyperelastic` | 1.5 | 0.35 | $K=5.0$ | $H^2$ ($\text{env}=1.0$) | Simo-Pister Log-Jacobian | `[100, 100, 20]` |
| **`sobolev`** | `sobolev` | 1.5 | 0.30 | — | $H^2$ ($s=2.0$) | FFT Sobolev Spectral | `[100, 100, 20]` |
| **`syn_navier`** | `navier` | 1.5 | 0.35 | $\nu=0.35$ | $H^2$ ($s=2.0$) | Navier-Cauchy Elastic | `[100, 100, 20]` |
| **`gaussian`** | `gaussian` | — | 0.25 | $\sigma_{\text{fluid}}=3.0$ | Spatial | Separable Gaussian | `[100, 100, 20]` |

---

## 4. Live Benchmark Output Paths

* **Per-Pair Result JSONs:** [`results/benchmark_continuum_top5/pair_*.json`](file:///Users/stnava/code/syntx/results/benchmark_continuum_top5/) (each file contains full machine-captured code, git diff, environment, and call manifests)
* **Master Summary JSON:** [`results/benchmark_continuum_top5/summary.json`](file:///Users/stnava/code/syntx/results/benchmark_continuum_top5/summary.json)
* **Interactive Population HTML Report:** [`docs/reports/continuum_top5_90pair_report.html`](file:///Users/stnava/code/syntx/docs/reports/continuum_top5_90pair_report.html)
* **Active Daemon Task ID:** `task-1291`
