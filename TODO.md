# Project TODO List & Engineering Roadmap

This document tracks prioritized engineering items, optimizations, and algorithmic improvements for `syntx`.

---

## 1. High Priority: Accelerate `syntx.robust_affine` Using `affine_fast` Innovations

### Context & Evidence
In the 90-pair Mindboggle-101 cohort benchmark on Apple Silicon MPS:
- **`syntx.affine (pt5)`**: `0.3472 ± 0.0219` cortical DICE in **`25.6s`** (consistently beating ANTs C++ `0.3453` at 33–52s).
- **`syntx.affine_fast`**: `0.3431 ± 0.0237` cortical DICE in **`3.3s`** (7.8× faster than `pt5`, 10–15× faster than ANTs C++).

`affine_fast` captures **98.8% of the alignment accuracy** of the full 4-level schedule while executing in just **3.3 seconds**.

### Mechanisms Behind `affine_fast` Speedup
1. **Pyramid Level Pruning (`L4 -> L2 -> L1`)**:
   - Default `pt5` optimizes across 4 levels (`L4 -> L3 -> L2 -> L1`) for 310 total iterations (`50 + 100 + 60 + 100`).
   - `affine_fast` completely skips level `L3`, moving directly from `L4` rigid alignment to `L2` affine and `L1` refinement in only 130 iterations (`50 + 50 + 30`).
2. **Monte Carlo Point Sampling vs Full/Strided Grids**:
   - `pt5` evaluates Mattes Mutual Information over full voxel grids at L4 and L3, and strided grids at L2 (25%) and L1 (10%).
   - `affine_fast` evaluates on a fixed random set of 100k spatial sample points across all levels (`n_sample_points=100_000`), reducing memory bandwidth pressure and kernel launch overhead by an order of magnitude.
3. **Reduced L1 Iteration Count**:
   - `pt5` spends 100 iterations at full resolution (`L1`), where each gradient evaluation is relatively expensive.
   - `affine_fast` runs only 30 iterations at `L1`, which suffices for fine parameter settling once the affine basin is established.

### Proposed Action Items & Implementation Plan
1. **Hybridize Multi-Start Candidate Pruning**:
   - In [`src/syntx/robust_affine.py`](file:///Users/stnava/code/syntx/src/syntx/robust_affine.py), evaluate multi-start rotation candidates and L4 rigid alignment strictly with 100k point-sampled Mattes-MI (~1.0s total).
2. **Eliminate Redundant L3 Level in Default Schedule**:
   - Adopt the 3-level hierarchy (`L4 -> L2 -> L1`) as default, saving an entire multi-resolution pooling and optimization pass.
3. **Adaptive Early-Stopping at L1**:
   - Replace the fixed 100-iteration L1 loop with an adaptive convergence check: stop when parameter update norm $\|\Delta \theta\|_2 < \epsilon$ or objective delta $|\Delta \mathcal{L}| < 10^{-4}$ for 5 consecutive steps, capping maximum iterations at 40.
4. **Performance Target**:
   - Target runtime: **6.0 – 8.0s on Apple Silicon MPS** (down from 25.6s).
   - Target accuracy: Retain **$\ge 0.3470$ mean cortical DICE** across the 90-pair cohort.
   - Retain full determinism and bitwise reproducibility per [`GEMINI.md`](file:///Users/stnava/code/syntx/GEMINI.md).

*Reference documentation*: [`docs/AFFINE_GUIDE.md`](file:///Users/stnava/code/syntx/docs/AFFINE_GUIDE.md)

---

## 2. Deformable Registration: Post-Benchmark TVF Parameter Tuning

### Context
In the v5.4.3 benchmark, `syntx.tvf` achieved virtually zero folding (`0.0005%` folding, 17 pairs with exactly 0.0000% fold), but registered `0.6358` DICE compared to `0.6482` in v5.4.2.

### Action Items (Do Not Implement Until Current Cohort Completes)
1. **Decouple Elastic Penalty**:
   - Set `total_sigma = 0.0` (or `total_sigma = 0.5`) in `tvf` to eliminate elastic field over-stiffening while letting Eulerian velocity smoothing control diffeomorphism smoothness.
2. **Relax Boundary Clamping**:
   - Investigate moving from Dirichlet DST-I boundary constraints to Sobolev/Neumann boundary conditions to prevent cortical edge pinning at volume boundaries.
3. **Validation**:
   - Verify on the 10-pair hard subset (`mbhard`) before testing on full cohort.

*Reference documentation*: [`docs/tvf_folding_mechanisms_and_solutions.md`](file:///Users/stnava/code/syntx/docs/tvf_folding_mechanisms_and_solutions.md)

---

## 3. General Architecture & Efficiency Roadmap

1. **Subsampling Kernel Vectorization**:
   - Benchmark PyTorch-native coordinate gathering vs strided slice sampling for Mattes-MI point sampling.
2. **Unified Evaluator Reporting**:
   - Standardize automated generation of LaTeX tables and publication-ready figures directly from unified cohort summary CSVs.
