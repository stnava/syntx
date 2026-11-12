# Tuning report: syngs

- pairs: [77, 44, 0]; code: `c3a0b3cc2b0622377fc540be3d3dd7daf836571c`
- noise (mean |rep0-rep1| Dice): 0.00000; acceptance margin: 0.00050
- criteria: {'folding_abs': 0.0005, 'folding_rel': 1.0, 'inv_interior_abs': 1.0, 'inv_interior_rel': 1.1, 'positive_jac_if_baseline': True, 'inverse_cap_only_if_baseline_fold_free': True, 'max_pair_drop': 0.001, 'jac_min_rel': None, 'min_gain': 0.0005, 'noise_k': 3.0}
- fixed (not tuned): {'reg_iterations': [100, 100, 20]}
- defaults: {'alpha': 0.448, 'max_step_norm': 0.6, 'optimizer': 'reg_adam', 'optimizer_lr': 2.246094, 'n_steps': 8, 'bootstrap_mode': 'none', 'regularizer': 'sobolev', 'reg_iterations': [100, 100, 20]}
- **winner**: {'regularizer': 'beltrami', 'alpha': 0.27, 'dilatation_weight': 0.075, 'max_step_norm': 0.825} (mean Dice gain +0.0123)

| rank | configuration | stage | reps | mean gain | feasible | pair 77 Dice / fold % / inv int max / inv max (mm) | pair 44 Dice / fold % / inv int max / inv max (mm) | pair 0 Dice / fold % / inv int max / inv max (mm) |
|---|---|---|---|---|---|---|---|---|
| 1 | alpha=0.27, dilatation_weight=0.075, max_step_norm=0.825, regularizer=beltrami | refine2 | 2 | +0.0123 | yes | 0.6165 / 0.0000 / 0.67 / 2.35 | 0.6192 / 0.0000 / 0.78 / 1.56 | 0.6558 / 0.0000 / 0.63 / 2.49 |
| 2 | alpha=0.27, dilatation_weight=0.075, max_step_norm=0.75, regularizer=beltrami | local1 | 1 | +0.0113 | yes | 0.6153 / 0.0000 / 0.53 / 2.08 | 0.6181 / 0.0000 / 0.69 / 2.28 | 0.6551 / 0.0000 / 0.65 / 1.68 |
| 3 | alpha=0.27, dilatation_weight=0.05625, max_step_norm=0.75, regularizer=beltrami | refine2 | 1 | +0.0112 | yes | 0.6158 / 0.0000 / 0.71 / 1.97 | 0.6179 / 0.0000 / 0.71 / 2.01 | 0.6545 / 0.0000 / 0.66 / 1.76 |
| 4 | alpha=0.243, dilatation_weight=0.075, max_step_norm=0.75, regularizer=beltrami | refine2 | 1 | +0.0111 | yes | 0.6144 / 0.0000 / 0.72 / 2.06 | 0.6178 / 0.0000 / 0.98 / 2.18 | 0.6556 / 0.0000 / 0.62 / 2.26 |
| 5 | alpha=0.2925, dilatation_weight=0.075, max_step_norm=0.75, regularizer=beltrami | refine2 | 1 | +0.0103 | yes | 0.6158 / 0.0000 / 0.46 / 1.76 | 0.6154 / 0.0000 / 0.61 / 1.81 | 0.6543 / 0.0000 / 0.72 / 1.90 |
| 6 | alpha=0.27, dilatation_weight=0.075, max_step_norm=0.675, regularizer=beltrami | refine2 | 1 | +0.0102 | yes | 0.6140 / 0.0000 / 0.40 / 1.77 | 0.6170 / 0.0000 / 0.53 / 1.77 | 0.6542 / 0.0000 / 0.49 / 1.38 |
| 7 | alpha=0.27, dilatation_weight=0.075, max_step_norm=0.66, regularizer=beltrami | local2 | 1 | +0.0098 | yes | 0.6133 / 0.0000 / 0.39 / 2.13 | 0.6166 / 0.0000 / 0.54 / 1.87 | 0.6540 / 0.0000 / 0.60 / 1.51 |
| 8 | alpha=0.216, dilatation_weight=0.075, regularizer=beltrami | local1 | 1 | +0.0083 | yes | 0.6099 / 0.0000 / 0.68 / 2.17 | 0.6152 / 0.0000 / 0.66 / 1.80 | 0.6545 / 0.0000 / 0.53 / 2.29 |
| 9 | alpha=0.27, dilatation_weight=0.075, optimizer_lr=1.796875, regularizer=beltrami | local1 | 1 | +0.0080 | yes | 0.6112 / 0.0000 / 0.51 / 1.72 | 0.6151 / 0.0000 / 0.87 / 1.71 | 0.6524 / 0.0000 / 0.36 / 2.28 |
| 10 | alpha=0.27, dilatation_weight=0.075, optimizer_lr=2.807617, regularizer=beltrami | local1 | 1 | +0.0079 | yes | 0.6111 / 0.0000 / 0.51 / 1.65 | 0.6150 / 0.0000 / 0.78 / 1.70 | 0.6524 / 0.0000 / 0.36 / 2.08 |
| 11 | alpha=0.27, dilatation_weight=0.075, regularizer=beltrami | refine1 | 2 | +0.0079 | yes | 0.6110 / 0.0000 / 0.51 / 1.67 | 0.6150 / 0.0000 / 0.80 / 1.72 | 0.6524 / 0.0000 / 0.36 / 2.23 |
| 12 | alpha=0.2025, dilatation_weight=0.1, regularizer=beltrami | refine1 | 1 | +0.0079 | yes | 0.6088 / 0.0000 / 0.77 / 2.06 | 0.6148 / 0.0000 / 0.92 / 1.98 | 0.6547 / 0.0000 / 0.56 / 1.83 |
| 13 | alpha=0.27, dilatation_weight=0.1, regularizer=beltrami | screen | 1 | +0.0079 | yes | 0.6111 / 0.0000 / 0.44 / 1.71 | 0.6149 / 0.0000 / 0.81 / 1.81 | 0.6522 / 0.0000 / 0.60 / 2.32 |
| 14 | alpha=0.27, dilatation_weight=0.175, regularizer=beltrami | refine1 | 1 | +0.0069 | yes | 0.6107 / 0.0000 / 0.44 / 1.53 | 0.6149 / 0.0000 / 0.82 / 1.37 | 0.6499 / 0.0000 / 0.66 / 1.35 |
| 15 | alpha=0.315, dilatation_weight=0.1, regularizer=beltrami | refine1 | 1 | +0.0061 | yes | 0.6117 / 0.0000 / 0.41 / 1.40 | 0.6119 / 0.0000 / 0.59 / 1.54 | 0.6493 / 0.0000 / 0.51 / 1.47 |
| 16 | alpha=0.3375, dilatation_weight=0.075, regularizer=beltrami | local1 | 1 | +0.0047 | yes | 0.6105 / 0.0000 / 0.32 / 1.39 | 0.6107 / 0.0000 / 0.47 / 1.61 | 0.6476 / 0.0000 / 0.27 / 1.37 |
| 17 | alpha=0.27, dilatation_weight=0.075, max_step_norm=0.48, regularizer=beltrami | local1 | 1 | +0.0043 | yes | 0.6069 / 0.0000 / 0.58 / 1.51 | 0.6121 / 0.0000 / 0.52 / 1.29 | 0.6485 / 0.0000 / 0.60 / 1.46 |
| 18 | defaults | baseline | 2 | +0.0000 | yes | 0.5998 / 0.0000 / 0.38 / 1.28 | 0.6096 / 0.0000 / 0.74 / 1.48 | 0.6453 / 0.0000 / 0.53 / 1.43 |
| 19 | alpha=0.18, dilatation_weight=0.1, max_step_norm=1.2, regularizer=beltrami | screen | 1 | +0.0127 | no | 0.6133 / 0.0000 / 1.51 / 2.39 | 0.6192 / 0.0000 / 1.05 / 2.58 | 0.6602 / 0.0000 / 1.06 / 2.92 |
| 20 | alpha=0.216, dilatation_weight=0.075, max_step_norm=0.75, regularizer=beltrami | refine2 | 1 | +0.0114 | no | 0.6128 / 0.0000 / 1.41 / 2.30 | 0.6184 / 0.0000 / 0.95 / 2.13 | 0.6575 / 0.0000 / 0.70 / 2.34 |
| 21 | alpha=0.18, dilatation_weight=0.1, max_step_norm=0.9, regularizer=beltrami | screen | 1 | +0.0112 | no | 0.6116 / 0.0000 / 1.36 / 3.19 | 0.6174 / 0.0000 / 1.18 / 2.90 | 0.6592 / 0.0000 / 1.32 / 2.54 |
| 22 | alpha=0.27, dilatation_weight=0.125, max_step_norm=0.75, regularizer=beltrami | refine2 | 1 | +0.0108 | no | 0.6144 / 0.0000 / 0.43 / 2.25 | 0.6179 / 0.0000 / 0.65 / 1.53 | 0.6547 / 0.0000 / 1.26 / 1.70 |
| 23 | alpha=0.18, bootstrap_mode=antithetic, dilatation_weight=0.1, regularizer=beltrami | screen | 1 | +0.0079 | no | 0.6082 / 0.0000 / 1.01 / 2.63 | 0.6155 / 0.0000 / 0.86 / 1.97 | 0.6546 / 0.0000 / 0.99 / 2.03 |
| 24 | alpha=0.18, dilatation_weight=0.1, n_steps=6, regularizer=beltrami | screen | 1 | +0.0076 | no | 0.6076 / 0.0000 / 0.72 / 1.95 | 0.6144 / 0.0000 / 1.03 / 1.82 | 0.6555 / 0.0000 / 0.75 / 2.09 |
| 25 | alpha=0.18, dilatation_weight=0.1, optimizer_lr=3.369141, regularizer=beltrami | screen | 1 | +0.0074 | no | 0.6074 / 0.0000 / 0.86 / 1.95 | 0.6142 / 0.0000 / 1.08 / 1.74 | 0.6554 / 0.0000 / 0.82 / 2.09 |
| 26 | alpha=0.18, dilatation_weight=0.1, regularizer=beltrami | start | 1 | +0.0074 | no | 0.6074 / 0.0000 / 0.86 / 1.95 | 0.6141 / 0.0000 / 1.08 / 1.80 | 0.6554 / 0.0000 / 0.82 / 2.12 |
| 27 | alpha=0.18, dilatation_weight=0.1, regularizer=beltrami, s=2.0 | screen | 1 | +0.0074 | no | 0.6074 / 0.0000 / 0.86 / 1.95 | 0.6141 / 0.0000 / 1.08 / 1.80 | 0.6554 / 0.0000 / 0.82 / 2.12 |
| 28 | alpha=0.18, dilatation_weight=0.1, optimizer_lr=3.5, regularizer=beltrami | screen | 1 | +0.0074 | no | 0.6073 / 0.0000 / 0.86 / 1.97 | 0.6141 / 0.0000 / 1.08 / 1.80 | 0.6554 / 0.0000 / 0.82 / 1.97 |
| 29 | alpha=0.18, dilatation_weight=0.1, optimizer_lr=1.68457, regularizer=beltrami | screen | 1 | +0.0074 | no | 0.6072 / 0.0000 / 0.78 / 1.95 | 0.6141 / 0.0000 / 1.08 / 1.79 | 0.6555 / 0.0000 / 0.81 / 1.97 |
| 30 | alpha=0.18, dilatation_weight=0.1, n_steps=10, regularizer=beltrami | screen | 1 | +0.0072 | no | 0.6070 / 0.0000 / 0.75 / 2.18 | 0.6142 / 0.0000 / 1.14 / 1.95 | 0.6549 / 0.0000 / 0.83 / 2.19 |
| 31 | alpha=0.18, dilatation_weight=0.1, n_steps=12, regularizer=beltrami | screen | 1 | +0.0071 | no | 0.6067 / 0.0000 / 0.76 / 2.16 | 0.6144 / 0.0000 / 1.16 / 2.07 | 0.6550 / 0.0000 / 0.83 / 2.05 |
| 32 | alpha=0.18, dilatation_weight=0.1, optimizer_lr=1.123047, regularizer=beltrami | screen | 1 | +0.0071 | no | 0.6072 / 0.0000 / 0.84 / 1.97 | 0.6140 / 0.0000 / 1.12 / 1.76 | 0.6549 / 0.0000 / 0.86 / 2.11 |
| 33 | alpha=0.135, dilatation_weight=0.1, regularizer=beltrami | screen | 1 | +0.0067 | no | 0.6042 / 0.0000 / 0.98 / 2.24 | 0.6162 / 0.0000 / 0.82 / 1.86 | 0.6544 / 0.0000 / 1.20 / 1.82 |
| 34 | alpha=0.18, dilatation_weight=0.25, regularizer=beltrami | screen | 1 | +0.0060 | no | 0.6058 / 0.0000 / 0.83 / 2.02 | 0.6153 / 0.0000 / 0.76 / 1.97 | 0.6516 / 0.0000 / 1.11 / 2.00 |
| 35 | alpha=0.09, dilatation_weight=0.1, regularizer=beltrami | screen | 1 | +0.0049 | no | 0.6018 / 0.0000 / 1.39 / 2.55 | 0.6147 / 0.0000 / 0.98 / 2.19 | 0.6529 / 0.0000 / 0.90 / 2.18 |
| 36 | alpha=0.36, dilatation_weight=0.1, regularizer=beltrami | screen | 1 | +0.0037 | no | 0.6119 / 0.0000 / 0.25 / 1.27 | 0.6083 / 0.0000 / 0.72 / 1.76 | 0.6456 / 0.0000 / 0.18 / 1.50 |
| 37 | alpha=0.18, dilatation_weight=0.1, max_step_norm=0.45, regularizer=beltrami | screen | 1 | +0.0034 | no | 0.6027 / 0.0000 / 0.64 / 1.86 | 0.6117 / 0.0000 / 1.14 / 1.69 | 0.6503 / 0.0000 / 0.60 / 1.65 |
| 38 | alpha=0.18, dilatation_weight=0.1, regularizer=beltrami, s=1.0 | screen | 1 | +0.0023 | no | 0.6001 / 0.0000 / 1.25 / 2.03 | 0.6124 / 0.0000 / 0.94 / 2.51 | 0.6493 / 0.0000 / 1.21 / 2.13 |
| 39 | alpha=0.18, dilatation_weight=0.5, regularizer=beltrami | screen | 1 | -0.0012 | no | 0.5976 / 0.0000 / 0.93 / 1.98 | 0.6133 / 0.0000 / 0.95 / 2.02 | 0.6400 / 0.0000 / 0.54 / 1.77 |
| 40 | alpha=0.18, dilatation_weight=0.1, max_step_norm=0.3, regularizer=beltrami | screen | 1 | -0.0061 | no | 0.5916 / 0.0000 / 0.39 / 1.45 | 0.6050 / 0.0000 / 0.56 / 1.50 | 0.6397 / 0.0000 / 0.34 / 1.87 |

Violations of infeasible configurations:

- alpha=0.18, dilatation_weight=0.1, max_step_norm=1.2, regularizer=beltrami: {77: ['interior inverse max 1.51 > 1.00 mm'], 44: ['interior inverse max 1.05 > 1.00 mm'], 0: ['interior inverse max 1.06 > 1.00 mm']}
- alpha=0.216, dilatation_weight=0.075, max_step_norm=0.75, regularizer=beltrami: {77: ['interior inverse max 1.41 > 1.00 mm']}
- alpha=0.18, dilatation_weight=0.1, max_step_norm=0.9, regularizer=beltrami: {77: ['interior inverse max 1.36 > 1.00 mm'], 44: ['interior inverse max 1.18 > 1.00 mm'], 0: ['interior inverse max 1.32 > 1.00 mm']}
- alpha=0.27, dilatation_weight=0.125, max_step_norm=0.75, regularizer=beltrami: {0: ['interior inverse max 1.26 > 1.00 mm']}
- alpha=0.18, bootstrap_mode=antithetic, dilatation_weight=0.1, regularizer=beltrami: {77: ['interior inverse max 1.01 > 1.00 mm']}
- alpha=0.18, dilatation_weight=0.1, n_steps=6, regularizer=beltrami: {44: ['interior inverse max 1.03 > 1.00 mm']}
- alpha=0.18, dilatation_weight=0.1, optimizer_lr=3.369141, regularizer=beltrami: {44: ['interior inverse max 1.08 > 1.00 mm']}
- alpha=0.18, dilatation_weight=0.1, regularizer=beltrami: {44: ['interior inverse max 1.08 > 1.00 mm']}
- alpha=0.18, dilatation_weight=0.1, regularizer=beltrami, s=2.0: {44: ['interior inverse max 1.08 > 1.00 mm']}
- alpha=0.18, dilatation_weight=0.1, optimizer_lr=3.5, regularizer=beltrami: {44: ['interior inverse max 1.08 > 1.00 mm']}
- alpha=0.18, dilatation_weight=0.1, optimizer_lr=1.68457, regularizer=beltrami: {44: ['interior inverse max 1.08 > 1.00 mm']}
- alpha=0.18, dilatation_weight=0.1, n_steps=10, regularizer=beltrami: {44: ['interior inverse max 1.14 > 1.00 mm']}
- alpha=0.18, dilatation_weight=0.1, n_steps=12, regularizer=beltrami: {44: ['interior inverse max 1.16 > 1.00 mm']}
- alpha=0.18, dilatation_weight=0.1, optimizer_lr=1.123047, regularizer=beltrami: {44: ['interior inverse max 1.12 > 1.00 mm']}
- alpha=0.135, dilatation_weight=0.1, regularizer=beltrami: {0: ['interior inverse max 1.20 > 1.00 mm']}
- alpha=0.18, dilatation_weight=0.25, regularizer=beltrami: {0: ['interior inverse max 1.11 > 1.00 mm']}
- alpha=0.09, dilatation_weight=0.1, regularizer=beltrami: {77: ['interior inverse max 1.39 > 1.00 mm']}
- alpha=0.36, dilatation_weight=0.1, regularizer=beltrami: {44: ['dice drop 0.0013 > 0.001']}
- alpha=0.18, dilatation_weight=0.1, max_step_norm=0.45, regularizer=beltrami: {44: ['interior inverse max 1.14 > 1.00 mm']}
- alpha=0.18, dilatation_weight=0.1, regularizer=beltrami, s=1.0: {77: ['interior inverse max 1.25 > 1.00 mm'], 0: ['interior inverse max 1.21 > 1.00 mm']}
- alpha=0.18, dilatation_weight=0.5, regularizer=beltrami: {77: ['dice drop 0.0022 > 0.001'], 0: ['dice drop 0.0052 > 0.001']}
- alpha=0.18, dilatation_weight=0.1, max_step_norm=0.3, regularizer=beltrami: {77: ['dice drop 0.0082 > 0.001'], 44: ['dice drop 0.0046 > 0.001'], 0: ['dice drop 0.0055 > 0.001']}
