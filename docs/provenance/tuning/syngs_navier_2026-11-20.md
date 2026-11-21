<!-- copied from results/tune_syngs_navier_20261120/report.md -->
# Tuning report: syngs

- pairs: [77, 44, 0]; code: `ddd4b70157e49731225ab3e0ab491e1921586712`
- noise (mean |rep0-rep1| Dice): 0.00001; acceptance margin: 0.00050
- criteria: {'folding_abs': 0.0005, 'folding_rel': 1.0, 'inv_interior_abs': 1.0, 'inv_interior_rel': 1.1, 'positive_jac_if_baseline': True, 'inverse_cap_only_if_baseline_fold_free': True, 'max_pair_drop': 0.001, 'jac_min_rel': None, 'min_gain': 0.0005, 'noise_k': 3.0}
- fixed (not tuned): {'reg_iterations': [100, 100, 20]}
- defaults: {'alpha': 0.448, 'max_step_norm': 0.6, 'optimizer': 'reg_adam', 'optimizer_lr': 2.246094, 'n_steps': 8, 'bootstrap_mode': 'none', 'regularizer': 'sobolev', 'reg_iterations': [100, 100, 20]}
- **winner**: {'regularizer': 'navier', 'poisson_ratio': 0.0, 's': 2.0, 'alpha': 0.27, 'max_step_norm': 0.8} (mean Dice gain +0.0108)

| rank | configuration | stage | reps | mean gain | feasible | pair 77 Dice / fold % / inv int max / inv max (mm) | pair 44 Dice / fold % / inv int max / inv max (mm) | pair 0 Dice / fold % / inv int max / inv max (mm) |
|---|---|---|---|---|---|---|---|---|
| 1 | alpha=0.27, max_step_norm=0.825, poisson_ratio=0.0, regularizer=navier, s=2.0 | refine2 | 1 | +0.0110 | yes | 0.6139 / 0.0000 / 0.97 / 1.47 | 0.6195 / 0.0000 / 0.57 / 1.69 | 0.6534 / 0.0000 / 0.49 / 1.48 |
| 2 | alpha=0.27, max_step_norm=0.8, poisson_ratio=0.0, regularizer=navier, s=2.0 | refine1 | 2 | +0.0108 | yes | 0.6136 / 0.0000 / 0.62 / 1.32 | 0.6187 / 0.0000 / 0.77 / 1.45 | 0.6539 / 0.0000 / 0.77 / 1.61 |
| 3 | alpha=0.31, max_step_norm=0.75, poisson_ratio=0.0, regularizer=navier, s=2.0 | refine1 | 1 | +0.0103 | yes | 0.6137 / 0.0000 / 0.56 / 1.45 | 0.6212 / 0.0000 / 0.49 / 1.61 | 0.6499 / 0.0000 / 0.52 / 1.39 |
| 4 | alpha=0.27, max_step_norm=0.75, poisson_ratio=0.0, regularizer=navier, s=2.0 | screen | 1 | +0.0101 | yes | 0.6131 / 0.0000 / 0.82 / 1.97 | 0.6180 / 0.0000 / 0.75 / 1.79 | 0.6531 / 0.0000 / 0.53 / 1.80 |
| 5 | alpha=0.235, max_step_norm=0.75, poisson_ratio=0.0, regularizer=navier, s=2.0 | refine1 | 1 | +0.0090 | yes | 0.6110 / 0.0000 / 0.71 / 1.43 | 0.6157 / 0.0000 / 0.75 / 1.75 | 0.6543 / 0.0000 / 0.62 / 1.75 |
| 6 | alpha=0.27, max_step_norm=0.75, poisson_ratio=0.075, regularizer=navier, s=2.0 | refine1 | 1 | +0.0085 | yes | 0.6112 / 0.0000 / 0.82 / 1.27 | 0.6169 / 0.0000 / 0.66 / 1.94 | 0.6513 / 0.0000 / 0.56 / 1.56 |
| 7 | alpha=0.27, max_step_norm=0.95, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | +0.0034 | yes | 0.6039 / 0.0000 / 0.99 / 1.96 | 0.6135 / 0.0000 / 0.77 / 2.01 | 0.6468 / 0.0000 / 0.71 / 1.39 |
| 8 | defaults | baseline | 2 | +0.0000 | yes | 0.5997 / 0.0000 / 0.30 / 1.12 | 0.6094 / 0.0000 / 0.73 / 1.12 | 0.6449 / 0.0000 / 0.32 / 1.26 |
| 9 | alpha=0.27, max_step_norm=0.7, poisson_ratio=0.0, regularizer=navier, s=2.0 | refine1 | 1 | +0.0088 | no | 0.6116 / 0.0000 / 1.19 / 1.19 | 0.6169 / 0.0000 / 0.71 / 1.67 | 0.6518 / 0.0000 / 0.57 / 1.80 |
| 10 | alpha=0.27, max_step_norm=0.75, poisson_ratio=0.15, regularizer=navier, s=2.0 | screen | 1 | +0.0063 | no | 0.6083 / 0.0000 / 1.01 / 1.52 | 0.6155 / 0.0000 / 0.55 / 1.66 | 0.6489 / 0.0000 / 0.54 / 1.51 |
| 11 | alpha=0.27, max_step_norm=0.85, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | +0.0019 | no | 0.6025 / 0.0000 / 0.92 / 1.22 | 0.6115 / 0.0000 / 1.09 / 1.41 | 0.6457 / 0.0000 / 0.77 / 1.63 |
| 12 | alpha=0.2, max_step_norm=0.75, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | +0.0002 | no | 0.5996 / 0.0000 / 1.18 / 1.69 | 0.6093 / 0.0000 / 1.03 / 1.76 | 0.6456 / 0.0000 / 0.67 / 1.31 |
| 13 | alpha=0.27, max_step_norm=0.75, n_steps=10, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | +0.0000 | no | 0.5998 / 0.0000 / 0.82 / 1.41 | 0.6108 / 0.0000 / 0.63 / 1.75 | 0.6433 / 0.0000 / 1.05 / 1.71 |
| 14 | alpha=0.27, max_step_norm=0.75, poisson_ratio=0.25, regularizer=navier, s=2.0 | start | 1 | -0.0000 | no | 0.6002 / 0.0000 / 0.68 / 1.34 | 0.6105 / 0.0000 / 0.51 / 1.40 | 0.6432 / 0.0000 / 0.61 / 1.68 |
| 15 | alpha=0.27, max_step_norm=0.75, optimizer_lr=1.8, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | -0.0003 | no | 0.5993 / 0.0000 / 0.83 / 1.22 | 0.6107 / 0.0000 / 0.54 / 1.37 | 0.6431 / 0.0000 / 1.01 / 1.61 |
| 16 | alpha=0.27, max_step_norm=0.75, optimizer_lr=2.8, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | -0.0008 | no | 0.5998 / 0.0000 / 0.52 / 1.24 | 0.6092 / 0.0000 / 0.52 / 1.66 | 0.6426 / 0.0000 / 0.45 / 1.14 |
| 17 | alpha=0.35, max_step_norm=0.75, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | -0.0019 | no | 0.5989 / 0.0000 / 0.54 / 1.13 | 0.6123 / 0.0000 / 0.42 / 1.73 | 0.6369 / 0.0000 / 0.37 / 1.16 |
| 18 | alpha=0.27, max_step_norm=0.75, poisson_ratio=0.25, regularizer=navier, s=1.0 | screen | 1 | -0.0039 | no | 0.5914 / 0.0000 / 1.22 / 2.24 | 0.6092 / 0.0000 / 0.91 / 1.74 | 0.6417 / 0.0000 / 1.09 / 1.88 |
| 19 | alpha=0.27, max_step_norm=0.65, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | -0.0039 | no | 0.5949 / 0.0000 / 0.63 / 1.56 | 0.6080 / 0.0000 / 0.56 / 1.71 | 0.6393 / 0.0000 / 0.48 / 1.16 |
| 20 | alpha=0.45, max_step_norm=0.75, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | -0.0079 | no | 0.5949 / 0.0000 / 0.34 / 0.76 | 0.6054 / 0.0000 / 0.42 / 0.72 | 0.6301 / 0.0000 / 0.31 / 0.94 |
| 21 | alpha=0.27, max_step_norm=0.75, poisson_ratio=0.25, regularizer=navier, s=3.0 | screen | 1 | -0.0083 | no | 0.5940 / 0.0000 / 0.30 / 0.95 | 0.6056 / 0.0000 / 0.34 / 0.77 | 0.6295 / 0.0000 / 0.35 / 1.09 |
| 22 | alpha=0.27, max_step_norm=0.5, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | -0.0131 | no | 0.5833 / 0.0000 / 0.32 / 1.18 | 0.5996 / 0.0000 / 0.39 / 1.24 | 0.6318 / 0.0000 / 0.56 / 1.05 |
| 23 | alpha=0.27, max_step_norm=0.75, poisson_ratio=0.35, regularizer=navier, s=2.0 | screen | 1 | -0.0183 | no | 0.5734 / 0.0000 / 0.57 / 1.39 | 0.5992 / 0.0000 / 0.62 / 1.18 | 0.6265 / 0.0000 / 0.56 / 1.44 |
| 24 | alpha=0.6, max_step_norm=0.75, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | -0.0184 | no | 0.5859 / 0.0000 / 0.29 / 0.57 | 0.5907 / 0.0000 / 0.16 / 0.65 | 0.6221 / 0.0000 / 0.25 / 0.62 |
| 25 | alpha=0.27, max_step_norm=0.75, poisson_ratio=0.42, regularizer=navier, s=2.0 | screen | 1 | -0.0416 | no | 0.5408 / 0.0000 / 0.46 / 1.26 | 0.5846 / 0.0000 / 0.44 / 0.95 | 0.6038 / 0.0000 / 0.50 / 0.98 |
| 26 | alpha=1.5, max_step_norm=0.75, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | -0.0712 | no | 0.5451 / 0.0000 / 0.02 / 0.02 | 0.5309 / 0.0000 / 0.02 / 0.02 | 0.5644 / 0.0000 / 0.02 / 0.02 |

Violations of infeasible configurations:

- alpha=0.27, max_step_norm=0.7, poisson_ratio=0.0, regularizer=navier, s=2.0: {77: ['interior inverse max 1.19 > 1.00 mm']}
- alpha=0.27, max_step_norm=0.75, poisson_ratio=0.15, regularizer=navier, s=2.0: {77: ['interior inverse max 1.01 > 1.00 mm']}
- alpha=0.27, max_step_norm=0.85, poisson_ratio=0.25, regularizer=navier, s=2.0: {44: ['interior inverse max 1.09 > 1.00 mm']}
- alpha=0.2, max_step_norm=0.75, poisson_ratio=0.25, regularizer=navier, s=2.0: {77: ['interior inverse max 1.18 > 1.00 mm'], 44: ['interior inverse max 1.03 > 1.00 mm']}
- alpha=0.27, max_step_norm=0.75, n_steps=10, poisson_ratio=0.25, regularizer=navier, s=2.0: {0: ['dice drop 0.0015 > 0.001', 'interior inverse max 1.05 > 1.00 mm']}
- alpha=0.27, max_step_norm=0.75, poisson_ratio=0.25, regularizer=navier, s=2.0: {0: ['dice drop 0.0017 > 0.001']}
- alpha=0.27, max_step_norm=0.75, optimizer_lr=1.8, poisson_ratio=0.25, regularizer=navier, s=2.0: {0: ['dice drop 0.0018 > 0.001', 'interior inverse max 1.01 > 1.00 mm']}
- alpha=0.27, max_step_norm=0.75, optimizer_lr=2.8, poisson_ratio=0.25, regularizer=navier, s=2.0: {0: ['dice drop 0.0023 > 0.001']}
- alpha=0.35, max_step_norm=0.75, poisson_ratio=0.25, regularizer=navier, s=2.0: {0: ['dice drop 0.0079 > 0.001']}
- alpha=0.27, max_step_norm=0.75, poisson_ratio=0.25, regularizer=navier, s=1.0: {77: ['dice drop 0.0082 > 0.001', 'interior inverse max 1.22 > 1.00 mm'], 0: ['dice drop 0.0032 > 0.001', 'interior inverse max 1.09 > 1.00 mm']}
- alpha=0.27, max_step_norm=0.65, poisson_ratio=0.25, regularizer=navier, s=2.0: {77: ['dice drop 0.0047 > 0.001'], 44: ['dice drop 0.0014 > 0.001'], 0: ['dice drop 0.0056 > 0.001']}
- alpha=0.45, max_step_norm=0.75, poisson_ratio=0.25, regularizer=navier, s=2.0: {77: ['dice drop 0.0048 > 0.001'], 44: ['dice drop 0.0040 > 0.001'], 0: ['dice drop 0.0147 > 0.001']}
- alpha=0.27, max_step_norm=0.75, poisson_ratio=0.25, regularizer=navier, s=3.0: {77: ['dice drop 0.0057 > 0.001'], 44: ['dice drop 0.0038 > 0.001'], 0: ['dice drop 0.0153 > 0.001']}
- alpha=0.27, max_step_norm=0.5, poisson_ratio=0.25, regularizer=navier, s=2.0: {77: ['dice drop 0.0164 > 0.001'], 44: ['dice drop 0.0098 > 0.001'], 0: ['dice drop 0.0131 > 0.001']}
- alpha=0.27, max_step_norm=0.75, poisson_ratio=0.35, regularizer=navier, s=2.0: {77: ['dice drop 0.0263 > 0.001'], 44: ['dice drop 0.0102 > 0.001'], 0: ['dice drop 0.0184 > 0.001']}
- alpha=0.6, max_step_norm=0.75, poisson_ratio=0.25, regularizer=navier, s=2.0: {77: ['dice drop 0.0138 > 0.001'], 44: ['dice drop 0.0187 > 0.001'], 0: ['dice drop 0.0227 > 0.001']}
- alpha=0.27, max_step_norm=0.75, poisson_ratio=0.42, regularizer=navier, s=2.0: {77: ['dice drop 0.0589 > 0.001'], 44: ['dice drop 0.0248 > 0.001'], 0: ['dice drop 0.0411 > 0.001']}
- alpha=1.5, max_step_norm=0.75, poisson_ratio=0.25, regularizer=navier, s=2.0: {77: ['dice drop 0.0546 > 0.001'], 44: ['dice drop 0.0785 > 0.001'], 0: ['dice drop 0.0805 > 0.001']}
