<!-- copied from results/tune_greedy_navier_20261120/report.md -->
# Tuning report: greedy

- pairs: [77, 44, 0]; code: `ddd4b70157e49731225ab3e0ab491e1921586712`
- noise (mean |rep0-rep1| Dice): 0.00000; acceptance margin: 0.00050
- criteria: {'folding_abs': 0.0005, 'folding_rel': 1.0, 'inv_interior_abs': 1.0, 'inv_interior_rel': 1.1, 'positive_jac_if_baseline': True, 'inverse_cap_only_if_baseline_fold_free': True, 'max_pair_drop': 0.001, 'jac_min_rel': None, 'min_gain': 0.0005, 'noise_k': 3.0}
- fixed (not tuned): {'reg_iterations': [100, 100, 20]}
- defaults: {'learning_rate': 1.5, 'flow_sigma': 1.62, 'total_sigma': 0.28, 'optimizer': 'cfl', 'regadam_sigma': 0.8, 'cfl_momentum': 0.1875, 'regularizer': 'div_curl', 'sobolev_alpha': None, 'lncc_radius': 2, 'reg_iterations': [100, 100, 20]}
- **winner**: {'regularizer': 'navier', 'poisson_ratio': 0.25, 's': 2.0, 'sobolev_alpha': 2.625, 'total_sigma': 0.0} (mean Dice gain +0.0088)

| rank | configuration | stage | reps | mean gain | feasible | pair 77 Dice / fold % / inv int max / inv max (mm) | pair 44 Dice / fold % / inv int max / inv max (mm) | pair 0 Dice / fold % / inv int max / inv max (mm) |
|---|---|---|---|---|---|---|---|---|
| 1 | poisson_ratio=0.225, regularizer=navier, s=2.0, sobolev_alpha=2.625, total_sigma=0.0 | refine2 | 1 | +0.0088 | yes | 0.5994 / 0.0000 / NaN / NaN | 0.6102 / 0.0000 / NaN / NaN | 0.6447 / 0.0000 / NaN / NaN |
| 2 | poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.625, total_sigma=0.0 | refine1 | 2 | +0.0088 | yes | 0.5980 / 0.0000 / NaN / NaN | 0.6129 / 0.0000 / NaN / NaN | 0.6433 / 0.0000 / NaN / NaN |
| 3 | poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.625, total_sigma=0.035 | refine2 | 1 | +0.0088 | yes | 0.5980 / 0.0000 / NaN / NaN | 0.6129 / 0.0000 / NaN / NaN | 0.6433 / 0.0000 / NaN / NaN |
| 4 | poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.5, total_sigma=0.0 | screen | 1 | +0.0085 | yes | 0.5964 / 0.0000 / NaN / NaN | 0.6132 / 0.0000 / NaN / NaN | 0.6439 / 0.0000 / NaN / NaN |
| 5 | poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.5, total_sigma=0.07 | refine1 | 1 | +0.0085 | yes | 0.5964 / 0.0000 / NaN / NaN | 0.6132 / 0.0000 / NaN / NaN | 0.6439 / 0.0000 / NaN / NaN |
| 6 | poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.5, total_sigma=0.14 | screen | 1 | +0.0085 | yes | 0.5963 / 0.0000 / NaN / NaN | 0.6122 / 0.0000 / NaN / NaN | 0.6448 / 0.0000 / NaN / NaN |
| 7 | poisson_ratio=0.2, regularizer=navier, s=2.0, sobolev_alpha=2.5, total_sigma=0.0 | refine1 | 1 | +0.0084 | yes | 0.5961 / 0.0000 / NaN / NaN | 0.6104 / 0.0000 / NaN / NaN | 0.6465 / 0.0000 / NaN / NaN |
| 8 | poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.6875, total_sigma=0.0 | refine2 | 1 | +0.0082 | yes | 0.5985 / 0.0000 / NaN / NaN | 0.6106 / 0.0000 / NaN / NaN | 0.6435 / 0.0000 / NaN / NaN |
| 9 | poisson_ratio=0.275, regularizer=navier, s=2.0, sobolev_alpha=2.625, total_sigma=0.0 | refine2 | 1 | +0.0079 | yes | 0.5979 / 0.0000 / NaN / NaN | 0.6114 / 0.0000 / NaN / NaN | 0.6423 / 0.0000 / NaN / NaN |
| 10 | poisson_ratio=0.3, regularizer=navier, s=2.0, sobolev_alpha=2.5, total_sigma=0.0 | refine1 | 1 | +0.0074 | yes | 0.5949 / 0.0000 / NaN / NaN | 0.6121 / 0.0000 / NaN / NaN | 0.6432 / 0.0000 / NaN / NaN |
| 11 | poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.375, total_sigma=0.0 | refine1 | 1 | +0.0071 | yes | 0.5926 / 0.0000 / NaN / NaN | 0.6113 / 0.0000 / NaN / NaN | 0.6452 / 0.0000 / NaN / NaN |
| 12 | learning_rate=2.25, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.5 | screen | 1 | +0.0002 | yes | 0.5927 / 0.0000 / NaN / NaN | 0.5975 / 0.0000 / NaN / NaN | 0.6382 / 0.0000 / NaN / NaN |
| 13 | defaults | baseline | 2 | +0.0000 | yes | 0.5922 / 0.0000 / NaN / NaN | 0.5974 / 0.0000 / NaN / NaN | 0.6383 / 0.0000 / NaN / NaN |
| 14 | cfl_momentum=0.35, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.5 | screen | 1 | +0.0017 | no | 0.5955 / 0.0000 / NaN / NaN | 0.6019 / 0.0000 / NaN / NaN | 0.6357 / 0.0000 / NaN / NaN |
| 15 | learning_rate=2.0, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.5 | screen | 1 | +0.0002 | no | 0.5939 / 0.0000 / NaN / NaN | 0.5983 / 0.0000 / NaN / NaN | 0.6364 / 0.0000 / NaN / NaN |
| 16 | learning_rate=1.75, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.5 | screen | 1 | -0.0003 | no | 0.5951 / 0.0000 / NaN / NaN | 0.5974 / 0.0000 / NaN / NaN | 0.6345 / 0.0000 / NaN / NaN |
| 17 | poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.25 | screen | 1 | -0.0003 | no | 0.5948 / 0.0000 / NaN / NaN | 0.5961 / 0.0000 / NaN / NaN | 0.6360 / 0.0000 / NaN / NaN |
| 18 | poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.5 | start | 1 | -0.0005 | no | 0.5950 / 0.0000 / NaN / NaN | 0.5968 / 0.0000 / NaN / NaN | 0.6344 / 0.0000 / NaN / NaN |
| 19 | poisson_ratio=0.15, regularizer=navier, s=2.0, sobolev_alpha=2.5 | screen | 1 | -0.0008 | no | 0.5936 / 0.0000 / NaN / NaN | 0.5980 / 0.0000 / NaN / NaN | 0.6339 / 0.0000 / NaN / NaN |
| 20 | poisson_ratio=0.0, regularizer=navier, s=2.0, sobolev_alpha=2.5 | screen | 1 | -0.0016 | no | 0.5897 / 0.0000 / NaN / NaN | 0.5953 / 0.0000 / NaN / NaN | 0.6380 / 0.0000 / NaN / NaN |
| 21 | poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.0 | screen | 1 | -0.0021 | no | 0.5912 / 0.0000 / NaN / NaN | 0.5963 / 0.0000 / NaN / NaN | 0.6341 / 0.0000 / NaN / NaN |
| 22 | poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.75 | screen | 1 | -0.0023 | no | 0.5939 / 0.0000 / NaN / NaN | 0.5957 / 0.0000 / NaN / NaN | 0.6313 / 0.0000 / NaN / NaN |
| 23 | learning_rate=1.25, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.5 | screen | 1 | -0.0031 | no | 0.5919 / 0.0000 / NaN / NaN | 0.5952 / 0.0000 / NaN / NaN | 0.6314 / 0.0000 / NaN / NaN |
| 24 | poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=3.0 | screen | 1 | -0.0036 | no | 0.5914 / 0.0000 / NaN / NaN | 0.5944 / 0.0000 / NaN / NaN | 0.6312 / 0.0000 / NaN / NaN |
| 25 | poisson_ratio=0.35, regularizer=navier, s=2.0, sobolev_alpha=2.5 | screen | 1 | -0.0042 | no | 0.5911 / 0.0000 / NaN / NaN | 0.5946 / 0.0000 / NaN / NaN | 0.6297 / 0.0000 / NaN / NaN |
| 26 | cfl_momentum=0.0, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.5 | screen | 1 | -0.0089 | no | 0.5865 / 0.0000 / NaN / NaN | 0.5871 / 0.0000 / NaN / NaN | 0.6276 / 0.0000 / NaN / NaN |
| 27 | poisson_ratio=0.25, regularizer=navier, s=3.0, sobolev_alpha=2.5 | screen | 1 | -0.0126 | no | 0.5916 / 0.0000 / NaN / NaN | 0.5824 / 0.0000 / NaN / NaN | 0.6162 / 0.0000 / NaN / NaN |

Violations of infeasible configurations:

- cfl_momentum=0.35, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.5: {0: ['dice drop 0.0026 > 0.001']}
- learning_rate=2.0, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.5: {0: ['dice drop 0.0019 > 0.001']}
- learning_rate=1.75, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.5: {0: ['dice drop 0.0038 > 0.001']}
- poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.25: {44: ['dice drop 0.0013 > 0.001'], 0: ['dice drop 0.0022 > 0.001']}
- poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.5: {0: ['dice drop 0.0039 > 0.001']}
- poisson_ratio=0.15, regularizer=navier, s=2.0, sobolev_alpha=2.5: {0: ['dice drop 0.0044 > 0.001']}
- poisson_ratio=0.0, regularizer=navier, s=2.0, sobolev_alpha=2.5: {77: ['dice drop 0.0026 > 0.001'], 44: ['dice drop 0.0021 > 0.001']}
- poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.0: {77: ['dice drop 0.0010 > 0.001'], 44: ['dice drop 0.0011 > 0.001'], 0: ['dice drop 0.0042 > 0.001']}
- poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.75: {44: ['dice drop 0.0017 > 0.001'], 0: ['dice drop 0.0070 > 0.001']}
- learning_rate=1.25, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.5: {44: ['dice drop 0.0022 > 0.001'], 0: ['dice drop 0.0068 > 0.001']}
- poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=3.0: {44: ['dice drop 0.0030 > 0.001'], 0: ['dice drop 0.0071 > 0.001']}
- poisson_ratio=0.35, regularizer=navier, s=2.0, sobolev_alpha=2.5: {77: ['dice drop 0.0011 > 0.001'], 44: ['dice drop 0.0028 > 0.001'], 0: ['dice drop 0.0086 > 0.001']}
- cfl_momentum=0.0, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.5: {77: ['dice drop 0.0057 > 0.001'], 44: ['dice drop 0.0103 > 0.001'], 0: ['dice drop 0.0107 > 0.001']}
- poisson_ratio=0.25, regularizer=navier, s=3.0, sobolev_alpha=2.5: {44: ['dice drop 0.0150 > 0.001'], 0: ['dice drop 0.0221 > 0.001']}
