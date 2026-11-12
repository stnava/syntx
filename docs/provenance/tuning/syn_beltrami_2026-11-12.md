# Tuning report: `syntx.syn` + `beltrami` (2026-11-12)

- **pairs**: `[77, 44, 0]` (Mindboggle-101 canonical 3-pair calibration set; NLM + `syntx.robust_affine` cached per pair)
- **registration code**: `44d223eca30d76cf6a4886a8461ee241efd2d307` (`v6.35.1`)
- **evaluations**: `162` single-pair runs across `52` distinct configurations (`results/tune_syn_beltrami_20261112`)
- **noise** (`mean |rep0 - rep1| Dice`): `0.00000` (exact bitwise determinism on Apple MPS); acceptance margin: `0.00050`
- **fixed (not tuned)**: `{'reg_iterations': [100, 100, 20]}`
- **`syntx.syn` canonical defaults (`sobolev` baseline)**:
  `{'grad_step': 0.4, 'flow_sigma': 2.4, 'total_sigma': 0.0, 'sobolev_alpha': 2.25, 'fast_smooth': False, 'in_loop_inv_steps': 10, 'inv_refresh_every': (1, 2, 4), 'polish_inv_steps': 30, 'polish_inv_tol': 0.02, 'compose_polish_tol': 0.3, 'optimizer': 'cfl', 'regularizer': 'sobolev', 'reg_iterations': [100, 100, 20]}`
- **#1 `syn_beltrami` configuration (2-rep confirmed)**:
  `{'regularizer': 'beltrami', 'grad_step': 0.6, 'dilatation_weight': 0.1, 'sobolev_alpha': 2.25, 's': 2.0}`
  - **3-pair Mean Bidirectional Cortical Dice**: **`0.6283`** (`+0.0021` / `+0.21 pp` above canonical `sobolev` `0.6261`)
  - **2-rep spread**: **`0.000000`** across all 3 pairs (`77: 0.615149`, `44: 0.620747`, `0: 0.648889`)
  - **Topology**: **`0.0000%`** folding on all 3 pairs (`jac_min >= +0.0104`, interior inverse consistency max `<= 0.3000 mm`)
  - **Default policy (`GEMINI.md` §3 & §6)**: `sobolev` remains the universal out-of-the-box `syntx.syn` default because `beltrami` uses aspect-ratio-normalized spectral grids (`spacing_rel = spacing / min(spacing)`), trading `-0.44 pp` on un-resampled thick-slice Pair 77 (`1.0x1.0x1.2 mm`, `0.6151` vs `0.6195`) for **`+0.70 pp` on isotropic `mbhard` Pair 44 (`0.6207` vs `0.6137`)** and **`+0.37 pp` on isotropic Pair 0 (`0.6489` vs `0.6452`)**. Tuned `syn_beltrami` is codified as a first-class named preset (`syn_beltrami`) in `DEFAULT_BENCHMARK_CONFIG` and `docs/provenance/run_config.json`.

## Top Ranked Configurations

| rank | configuration | stage | reps | mean Dice | mean gain vs `sobolev` | strict per-pair feasible (`drop <= 0.001`) | pair 77 (`1.0x1.0x1.2 mm`): Dice / fold% / Jmin / inv int max (mm) / s | pair 44 (`mbhard` 1 mm): Dice / fold% / Jmin / inv int max (mm) / s | pair 0 (`OASIS` 1 mm): Dice / fold% / Jmin / inv int max (mm) / s |
|---:|---|---|---:|---:|---:|:---:|---|---|---|
| 1 | `dilatation_weight=0.1, grad_step=0.6, regularizer=beltrami` | confirm | 2 | **0.6283** | **+0.0021** | no (`77: -0.0044`) | `0.6151 / 0.0000 / 0.010 / 0.30 / 51` | **`0.6207 / 0.0000 / 0.029 / 0.30 / 56`** | **`0.6489 / 0.0000 / 0.037 / 0.30 / 47`** |
| 2 | `grad_step=0.75, regularizer=beltrami` | screen | 1 | `0.6278` | `+0.0017` | no (`77: -0.0033`) | `0.6162 / 0.0000 / 0.010 / 0.30 / 51` | `0.6201 / 0.0000 / 0.023 / 0.30 / 59` | `0.6471 / 0.0000 / 0.029 / 0.30 / 49` |
| 3 | `grad_step=0.6, regularizer=beltrami` (`eta=0.25, s=2.0`) | screen | 1 | `0.6275` | `+0.0014` | no (`77: -0.0051`) | `0.6144 / 0.0000 / 0.010 / 0.30 / 51` | `0.6210 / 0.0000 / 0.021 / 0.30 / 57` | `0.6472 / 0.0000 / 0.032 / 0.30 / 46` |
| 4 | `dilatation_weight=0.1, grad_step=0.8, regularizer=beltrami` | refine1 | 1 | `0.6275` | `+0.0014` | no (`77: -0.0021`) | `0.6174 / 0.0000 / 0.006 / 0.30 / 51` | `0.6179 / 0.0000 / 0.013 / 0.30 / 60` | `0.6474 / 0.0000 / 0.037 / 0.48 / 50` |
| 5 | `grad_step=0.8, regularizer=beltrami, sobolev_alpha=1.6875` | refine1 | 1 | `0.6275` | `+0.0014` | no (`77: -0.0049`) | `0.6146 / 0.0000 / 0.005 / 0.71 / 51` | `0.6205 / 0.0000 / 0.019 / 0.30 / 60` | `0.6475 / 0.0000 / 0.018 / 0.30 / 49` |
| 6 | `grad_step=0.8, regularizer=beltrami` | screen | 1 | `0.6272` | `+0.0010` | no (`77: -0.0034`) | `0.6161 / 0.0000 / 0.006 / 0.30 / 52` | `0.6186 / 0.0000 / 0.012 / 0.30 / 59` | `0.6469 / 0.0000 / 0.023 / 0.30 / 50` |
| 7 | `regularizer=dsti1` (Dirichlet Sobolev twin) | screen | 1 | `0.6265` | `+0.0004` | no (`77: -0.0019`) | `0.6176 / 0.0000 / 0.016 / 0.30 / 51` | `0.6161 / 0.0000 / 0.050 / 0.30 / 54` | `0.6459 / 0.0000 / 0.054 / 0.30 / 47` |
| 8 | **`defaults` (`regularizer=sobolev`, canonical)** | baseline | 2 | **`0.6261`** | **`+0.0000`** | **yes** | **`0.6195 / 0.0000 / 0.019 / 0.30 / 44`** | `0.6137 / 0.0000 / 0.021 / 0.30 / 48` | `0.6452 / 0.0000 / 0.054 / 0.30 / 41` |
| 9 | `dilatation_weight=0.1, grad_step=0.675, regularizer=beltrami, sobolev_alpha=3.0` | screen | 1 | `0.6255` | `-0.0006` | yes | `0.6193 / 0.0000 / 0.014 / 0.30 / 47` | `0.6128 / 0.0000 / 0.021 / 0.30 / 59` | `0.6444 / 0.0000 / 0.053 / 0.30 / 46` |
| 10 | `regularizer=beltrami` (`grad_step=0.4`, default step) | start | 1 | `0.6251` | `-0.0010` | no | `0.6123 / 0.0000 / 0.010 / 0.30 / 46` | `0.6182 / 0.0000 / 0.018 / 0.30 / 51` | `0.6448 / 0.0000 / 0.023 / 0.30 / 44` |
| 11 | `regularizer=gaussian` (`flow_sigma=2.4`) | screen | 1 | `0.6217` | `-0.0045` | no | `0.6018 / 0.0000 / 0.006 / 0.30 / 42` | `0.6170 / 0.0000 / 0.024 / 0.30 / 49` | `0.6462 / 0.0000 / 0.034 / 0.30 / 41` |

## Physical Interpretation

1. **Step-size coupling with conformal shear damping**: Because the Beltrami operator $G_{\text{Beltrami}}(k) = (1 + \alpha \|k\|^2)^{-2}[I - \eta (k k^\top / \|k\|^2)]$ removes longitudinal shear-sliver energy, `syntx.syn`'s CFL step size safely scales from `grad_step=0.4` (`0.6251`) to `grad_step=0.6` (`0.6283` with `dilatation_weight=0.1`) while preserving `0.0000%` folding and `jac_min >= +0.0104`.
2. **`dsti1` redundancy**: `regularizer='dsti1'` (`0.6265`) is the exact Discrete Sine Transform type-I zero-Dirichlet counterpart of `sobolev` (`0.6261`), differing by `+0.04 pp` due solely to boundary conditions outside the brain mask.
