<!-- copied from results/tune_tvf_navier_20261120/report.md -->
# Tuning report: tvf

- pairs: [77, 44, 0]; code: `ddd4b70157e49731225ab3e0ab491e1921586712`
- noise (mean |rep0-rep1| Dice): 0.00008; acceptance margin: 0.00050
- criteria: {'folding_abs': 0.0005, 'folding_rel': 1.0, 'inv_interior_abs': 1.0, 'inv_interior_rel': 1.1, 'positive_jac_if_baseline': True, 'inverse_cap_only_if_baseline_fold_free': True, 'max_pair_drop': 0.001, 'jac_min_rel': None, 'min_gain': 0.0005, 'noise_k': 3.0}
- fixed (not tuned): {'reg_iterations': [100, 100, 20]}
- defaults: {'regularizer': 'beltrami', 'n_time_steps': 3, 'multipoint_loss': (0.0, 1.0), 'cfl_max': 0.0, 'syn_sampling': 2, 'energy_weight': 0.0008, 'optimizer': 'cfl', 'reg_iterations': [100, 100, 20], 'grad_step': 1.25, 'cfl_momentum': 0.9, 'total_alpha': 0.0, 'constant_speed': True, 'alpha': 2.5}
- **winner**: {'regularizer': 'navier', 'poisson_ratio': 0.25, 's': 2.0, 'alpha': 3.5, 'energy_weight': 0.0004, 'cfl_momentum': 0.8, 'grad_step': 1.8} (mean Dice gain +0.0097)

| rank | configuration | stage | reps | mean gain | feasible | pair 77 Dice / fold % / inv int max / inv max (mm) | pair 44 Dice / fold % / inv int max / inv max (mm) | pair 0 Dice / fold % / inv int max / inv max (mm) |
|---|---|---|---|---|---|---|---|---|
| 1 | alpha=3.5, cfl_momentum=0.8, energy_weight=0.0004, grad_step=2.0, poisson_ratio=0.25, regularizer=navier, s=2.0 | refine2 | 1 | +0.0099 | yes | 0.6305 / 0.0000 / 0.54 / 1.52 | 0.6362 / 0.0000 / 0.90 / 3.28 | 0.6659 / 0.0000 / 0.50 / 1.66 |
| 2 | alpha=3.5, cfl_momentum=0.8, energy_weight=0.0004, grad_step=1.8, poisson_ratio=0.25, regularizer=navier, s=2.0 | refine1 | 2 | +0.0097 | yes | 0.6302 / 0.0000 / 0.52 / 1.83 | 0.6363 / 0.0000 / 0.73 / 2.21 | 0.6654 / 0.0000 / 0.70 / 1.57 |
| 3 | alpha=3.6875, cfl_momentum=0.8, energy_weight=0.0004, grad_step=1.8, poisson_ratio=0.25, regularizer=navier, s=2.0 | refine2 | 1 | +0.0097 | yes | 0.6317 / 0.0000 / 1.04 / 1.77 | 0.6353 / 0.0000 / 0.71 / 2.17 | 0.6649 / 0.0000 / 0.44 / 1.50 |
| 4 | alpha=3.5, cfl_momentum=0.8, energy_weight=0.0004, grad_step=1.65, poisson_ratio=0.25, regularizer=navier, s=2.0 | refine2 | 1 | +0.0096 | yes | 0.6304 / 0.0000 / 0.56 / 1.45 | 0.6360 / 0.0000 / 0.85 / 2.99 | 0.6652 / 0.0000 / 0.65 / 1.99 |
| 5 | alpha=3.5, cfl_momentum=0.8, energy_weight=0.00045, grad_step=1.8, poisson_ratio=0.25, regularizer=navier, s=2.0 | refine2 | 1 | +0.0095 | yes | 0.6300 / 0.0000 / 0.57 / 2.51 | 0.6358 / 0.0000 / 0.78 / 2.44 | 0.6655 / 0.0000 / 0.62 / 1.47 |
| 6 | alpha=3.5, cfl_momentum=0.8, energy_weight=0.0004, grad_step=1.8, poisson_ratio=0.275, regularizer=navier, s=2.0 | refine2 | 1 | +0.0094 | yes | 0.6295 / 0.0000 / 0.56 / 1.99 | 0.6361 / 0.0000 / 0.78 / 2.50 | 0.6654 / 0.0000 / 0.71 / 1.93 |
| 7 | alpha=3.5, cfl_momentum=0.8, energy_weight=0.0004, grad_step=1.5, poisson_ratio=0.25, regularizer=navier, s=2.0 | refine1 | 1 | +0.0090 | yes | 0.6298 / 0.0000 / 0.65 / 1.58 | 0.6356 / 0.0000 / 0.61 / 2.38 | 0.6645 / 0.0000 / 0.89 / 1.81 |
| 8 | alpha=3.5, cfl_momentum=0.825, energy_weight=0.0004, grad_step=1.8, poisson_ratio=0.25, regularizer=navier, s=2.0 | refine2 | 1 | +0.0089 | yes | 0.6293 / 0.0000 / 0.48 / 1.40 | 0.6351 / 0.0000 / 0.65 / 2.87 | 0.6651 / 0.0000 / 0.52 / 1.56 |
| 9 | alpha=3.5, cfl_momentum=0.6, energy_weight=0.0004, poisson_ratio=0.25, regularizer=navier, s=2.0 | refine1 | 1 | +0.0089 | yes | 0.6303 / 0.0000 / 0.66 / 1.69 | 0.6353 / 0.0000 / 0.54 / 1.73 | 0.6638 / 0.0000 / 0.92 / 1.78 |
| 10 | alpha=3.5, cfl_momentum=0.8, energy_weight=0.00035, poisson_ratio=0.25, regularizer=navier, s=2.0 | refine1 | 1 | +0.0087 | yes | 0.6298 / 0.0000 / 0.89 / 1.71 | 0.6354 / 0.0000 / 0.58 / 1.91 | 0.6637 / 0.0000 / 0.57 / 1.55 |
| 11 | alpha=3.25, cfl_momentum=0.8, energy_weight=0.0004, poisson_ratio=0.25, regularizer=navier, s=2.0 | refine1 | 1 | +0.0087 | yes | 0.6293 / 0.0000 / 0.64 / 2.00 | 0.6350 / 0.0000 / 0.57 / 2.68 | 0.6645 / 0.0000 / 0.71 / 1.46 |
| 12 | alpha=3.5, cfl_momentum=0.8, energy_weight=0.0004, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | +0.0086 | yes | 0.6295 / 0.0000 / 0.89 / 1.96 | 0.6352 / 0.0000 / 0.75 / 1.98 | 0.6639 / 0.0000 / 0.88 / 1.68 |
| 13 | alpha=3.5, cfl_momentum=0.8, energy_weight=0.0004, poisson_ratio=0.2, regularizer=navier, s=2.0 | refine1 | 1 | +0.0085 | yes | 0.6299 / 0.0000 / 0.70 / 2.02 | 0.6344 / 0.0000 / 0.65 / 1.38 | 0.6641 / 0.0000 / 0.60 / 1.83 |
| 14 | alpha=3.5, cfl_momentum=0.8, energy_weight=0.0004, poisson_ratio=0.3, regularizer=navier, s=2.0 | refine1 | 1 | +0.0082 | yes | 0.6291 / 0.0000 / 0.81 / 1.62 | 0.6348 / 0.0000 / 0.66 / 1.72 | 0.6634 / 0.0000 / 0.80 / 1.67 |
| 15 | alpha=3.5, energy_weight=0.0004, grad_step=1.8, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | +0.0081 | yes | 0.6288 / 0.0000 / 1.19 / 2.34 | 0.6347 / 0.0000 / 0.87 / 2.41 | 0.6636 / 0.0000 / 0.64 / 1.49 |
| 16 | alpha=3.875, cfl_momentum=0.8, energy_weight=0.0004, poisson_ratio=0.25, regularizer=navier, s=2.0 | refine1 | 1 | +0.0079 | yes | 0.6298 / 0.0000 / 1.08 / 1.25 | 0.6341 / 0.0000 / 0.50 / 1.69 | 0.6626 / 0.0000 / 0.64 / 1.37 |
| 17 | alpha=3.5, cfl_momentum=0.85, energy_weight=0.0004, poisson_ratio=0.25, regularizer=navier, s=2.0 | refine1 | 1 | +0.0079 | yes | 0.6287 / 0.0000 / 0.84 / 1.86 | 0.6346 / 0.0000 / 0.78 / 1.81 | 0.6630 / 0.0000 / 0.47 / 1.74 |
| 18 | alpha=3.5, energy_weight=0.0004, grad_step=1.5, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | +0.0077 | yes | 0.6283 / 0.0000 / 0.75 / 1.33 | 0.6347 / 0.0000 / 0.56 / 2.10 | 0.6630 / 0.0000 / 0.85 / 1.49 |
| 19 | alpha=3.5, energy_weight=0.0003, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | +0.0076 | yes | 0.6291 / 0.0000 / 0.68 / 1.69 | 0.6340 / 0.0000 / 0.57 / 1.85 | 0.6624 / 0.0000 / 0.65 / 1.47 |
| 20 | alpha=3.5, energy_weight=0.0004, poisson_ratio=0.35, regularizer=navier, s=2.0 | screen | 1 | +0.0072 | yes | 0.6284 / 0.0000 / 0.63 / 1.28 | 0.6340 / 0.0000 / 0.64 / 1.50 | 0.6621 / 0.0000 / 0.78 / 1.59 |
| 21 | alpha=3.5, energy_weight=0.0004, poisson_ratio=0.25, regularizer=navier, s=2.0 | start | 1 | +0.0070 | yes | 0.6278 / 0.0000 / 0.94 / 1.90 | 0.6337 / 0.0000 / 0.80 / 1.59 | 0.6623 / 0.0000 / 0.76 / 1.53 |
| 22 | alpha=3.0, energy_weight=0.0004, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | +0.0069 | yes | 0.6267 / 0.0000 / 0.56 / 1.99 | 0.6338 / 0.0000 / 0.73 / 2.08 | 0.6632 / 0.0000 / 0.62 / 1.84 |
| 23 | alpha=3.5, energy_weight=0.0004, poisson_ratio=0.15, regularizer=navier, s=2.0 | screen | 1 | +0.0067 | yes | 0.6275 / 0.0000 / 0.52 / 1.71 | 0.6328 / 0.0000 / 0.55 / 1.98 | 0.6625 / 0.0000 / 0.88 / 1.81 |
| 24 | alpha=3.5, energy_weight=0.0004, poisson_ratio=0.0, regularizer=navier, s=2.0 | screen | 1 | +0.0066 | yes | 0.6273 / 0.0000 / 0.65 / 3.22 | 0.6326 / 0.0000 / 0.60 / 1.95 | 0.6627 / 0.0000 / 0.68 / 1.58 |
| 25 | alpha=4.25, energy_weight=0.0004, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | +0.0064 | yes | 0.6297 / 0.0000 / 0.70 / 1.16 | 0.6321 / 0.0000 / 0.39 / 1.18 | 0.6601 / 0.0000 / 0.46 / 1.31 |
| 26 | alpha=3.5, energy_weight=0.0006, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | +0.0053 | yes | 0.6256 / 0.0000 / 1.06 / 3.33 | 0.6320 / 0.0000 / 0.56 / 2.05 | 0.6611 / 0.0000 / 0.83 / 1.76 |
| 27 | alpha=3.5, energy_weight=0.0004, grad_step=1.0, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | +0.0052 | yes | 0.6264 / 0.0000 / 1.04 / 1.92 | 0.6314 / 0.0000 / 0.65 / 1.52 | 0.6608 / 0.0000 / 0.63 / 1.39 |
| 28 | alpha=2.0, energy_weight=0.0004, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | +0.0036 | yes | 0.6228 / 0.0000 / 1.06 / 2.82 | 0.6264 / 0.0000 / 0.86 / 4.78 | 0.6644 / 0.0000 / 0.97 / 1.89 |
| 29 | alpha=3.5, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | +0.0036 | yes | 0.6232 / 0.0000 / 0.61 / 1.68 | 0.6309 / 0.0000 / 0.62 / 2.08 | 0.6595 / 0.0000 / 0.57 / 1.75 |
| 30 | defaults | baseline | 2 | +0.0000 | yes | 0.6164 / 0.0000 / 1.34 / 2.63 | 0.6272 / 0.0000 / 0.80 / 2.34 | 0.6592 / 0.0000 / 0.51 / 1.62 |
| 31 | alpha=3.5, cfl_momentum=0.7, energy_weight=0.0004, grad_step=1.8, poisson_ratio=0.25, regularizer=navier, s=2.0 | refine2 | 2 | +0.0103 | no | 0.6309 / 0.0000 / 0.62 / 1.61 | 0.6368 / 0.0000 / 0.81 / 2.79 | 0.6661 / 0.0000 / 1.02 / 1.79 |
| 32 | alpha=3.5, cfl_momentum=0.8, energy_weight=0.000375, grad_step=1.8, poisson_ratio=0.25, regularizer=navier, s=2.0 | refine2 | 1 | +0.0098 | no | 0.6303 / 0.0000 / 0.51 / 1.39 | 0.6364 / 0.0000 / 0.85 / 2.38 | 0.6654 / 0.0000 / 1.05 / 1.41 |
| 33 | alpha=3.5, cfl_momentum=0.8, energy_weight=0.0004, grad_step=1.8, poisson_ratio=0.225, regularizer=navier, s=2.0 | refine2 | 1 | +0.0097 | no | 0.6304 / 0.0000 / 0.48 / 1.70 | 0.6358 / 0.0000 / 0.66 / 2.40 | 0.6657 / 0.0000 / 1.12 / 1.52 |
| 34 | alpha=3.375, cfl_momentum=0.8, energy_weight=0.0004, grad_step=1.8, poisson_ratio=0.25, regularizer=navier, s=2.0 | refine2 | 1 | +0.0094 | no | 0.6295 / 0.0000 / 0.62 / 1.66 | 0.6356 / 0.0000 / 1.23 / 2.59 | 0.6658 / 0.0000 / 0.80 / 1.57 |
| 35 | alpha=3.5, cfl_momentum=0.8, energy_weight=0.0005, poisson_ratio=0.25, regularizer=navier, s=2.0 | refine1 | 1 | +0.0075 | no | 0.6281 / 0.0000 / 0.46 / 1.84 | 0.6342 / 0.0000 / 0.47 / 1.64 | 0.6632 / 0.0000 / 1.06 / 1.91 |
| 36 | alpha=3.5, energy_weight=0.0004, poisson_ratio=0.42, regularizer=navier, s=2.0 | screen | 1 | +0.0068 | no | 0.6288 / 0.0000 / 0.62 / 1.69 | 0.6345 / 0.0000 / 0.59 / 2.23 | 0.6601 / 0.0000 / 1.73 / 1.79 |
| 37 | energy_weight=0.0004, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | +0.0066 | no | 0.6255 / 0.0000 / 0.69 / 1.99 | 0.6332 / 0.0000 / 0.66 / 4.39 | 0.6639 / 0.0000 / 1.07 / 1.61 |
| 38 | alpha=1.5, energy_weight=0.0004, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | +0.0007 | no | 0.6193 / 0.0000 / 1.19 / 2.53 | 0.6227 / 0.0000 / 1.06 / 3.39 | 0.6630 / 0.0000 / 0.75 / 2.22 |
| 39 | alpha=3.5, energy_weight=0.0004, poisson_ratio=0.25, regularizer=navier, s=1.0 | screen | 1 | -0.0086 | no | 0.6068 / 0.0000 / 1.46 / 2.76 | 0.6180 / 0.0000 / 1.23 / 2.26 | 0.6521 / 0.0000 / 0.96 / 1.70 |
| 40 | alpha=3.5, energy_weight=0.0004, poisson_ratio=0.25, regularizer=navier, s=3.0 | screen | 1 | -0.0095 | no | 0.6259 / 0.0000 / 0.25 / 0.65 | 0.6042 / 0.0000 / 0.02 / 0.39 | 0.6444 / 0.0000 / 0.04 / 0.71 |

Violations of infeasible configurations:

- alpha=3.5, cfl_momentum=0.7, energy_weight=0.0004, grad_step=1.8, poisson_ratio=0.25, regularizer=navier, s=2.0: {0: ['interior inverse max 1.02 > 1.00 mm']}
- alpha=3.5, cfl_momentum=0.8, energy_weight=0.000375, grad_step=1.8, poisson_ratio=0.25, regularizer=navier, s=2.0: {0: ['interior inverse max 1.05 > 1.00 mm']}
- alpha=3.5, cfl_momentum=0.8, energy_weight=0.0004, grad_step=1.8, poisson_ratio=0.225, regularizer=navier, s=2.0: {0: ['interior inverse max 1.12 > 1.00 mm']}
- alpha=3.375, cfl_momentum=0.8, energy_weight=0.0004, grad_step=1.8, poisson_ratio=0.25, regularizer=navier, s=2.0: {44: ['interior inverse max 1.23 > 1.00 mm']}
- alpha=3.5, cfl_momentum=0.8, energy_weight=0.0005, poisson_ratio=0.25, regularizer=navier, s=2.0: {0: ['interior inverse max 1.06 > 1.00 mm']}
- alpha=3.5, energy_weight=0.0004, poisson_ratio=0.42, regularizer=navier, s=2.0: {0: ['interior inverse max 1.73 > 1.00 mm']}
- energy_weight=0.0004, poisson_ratio=0.25, regularizer=navier, s=2.0: {0: ['interior inverse max 1.07 > 1.00 mm']}
- alpha=1.5, energy_weight=0.0004, poisson_ratio=0.25, regularizer=navier, s=2.0: {44: ['dice drop 0.0045 > 0.001', 'interior inverse max 1.06 > 1.00 mm']}
- alpha=3.5, energy_weight=0.0004, poisson_ratio=0.25, regularizer=navier, s=1.0: {77: ['dice drop 0.0096 > 0.001'], 44: ['dice drop 0.0092 > 0.001', 'interior inverse max 1.23 > 1.00 mm'], 0: ['dice drop 0.0071 > 0.001']}
- alpha=3.5, energy_weight=0.0004, poisson_ratio=0.25, regularizer=navier, s=3.0: {44: ['dice drop 0.0231 > 0.001'], 0: ['dice drop 0.0148 > 0.001']}
