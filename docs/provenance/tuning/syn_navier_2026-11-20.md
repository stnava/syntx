<!-- copied from results/tune_syn_navier_20261120/report.md -->
# Tuning report: syn

- pairs: [77, 44, 0]; code: `ddd4b70157e49731225ab3e0ab491e1921586712`
- noise (mean |rep0-rep1| Dice): 0.00006; acceptance margin: 0.00050
- criteria: {'folding_abs': 0.0005, 'folding_rel': 1.0, 'inv_interior_abs': 1.0, 'inv_interior_rel': 1.1, 'positive_jac_if_baseline': True, 'inverse_cap_only_if_baseline_fold_free': True, 'max_pair_drop': 0.001, 'jac_min_rel': None, 'min_gain': 0.0005, 'noise_k': 3.0}
- fixed (not tuned): {'reg_iterations': [100, 100, 20]}
- defaults: {'grad_step': 0.4, 'flow_sigma': 2.4, 'total_sigma': 0.0, 'sobolev_alpha': 2.25, 'fast_smooth': False, 'in_loop_inv_steps': 10, 'inv_refresh_every': (1, 2, 4), 'polish_inv_steps': 30, 'polish_inv_tol': 0.02, 'compose_polish_tol': 0.3, 'optimizer': 'cfl', 'regularizer': 'sobolev', 'reg_iterations': [100, 100, 20]}
- **winner**: {'regularizer': 'navier', 'poisson_ratio': 0.25, 's': 2.0, 'sobolev_alpha': 1.5, 'grad_step': 0.6} (mean Dice gain +0.0302)

| rank | configuration | stage | reps | mean gain | feasible | pair 77 Dice / fold % / inv int max / inv max (mm) | pair 44 Dice / fold % / inv int max / inv max (mm) | pair 0 Dice / fold % / inv int max / inv max (mm) |
|---|---|---|---|---|---|---|---|---|
| 1 | grad_step=0.6, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=1.5 | refine1 | 2 | +0.0302 | yes | 0.6214 / 0.0000 / 0.30 / 1.66 | 0.6275 / 0.0000 / 0.30 / 1.19 | 0.6548 / 0.0000 / 0.30 / 0.60 |
| 2 | grad_step=0.625, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=1.5 | refine2 | 1 | +0.0301 | yes | 0.6221 / 0.0000 / 0.30 / 0.72 | 0.6271 / 0.0000 / 0.31 / 0.85 | 0.6544 / 0.0000 / 0.30 / 0.67 |
| 3 | grad_step=0.55, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=1.5 | screen | 1 | +0.0301 | yes | 0.6212 / 0.0000 / 0.30 / 1.08 | 0.6276 / 0.0000 / 0.42 / 1.00 | 0.6545 / 0.0000 / 0.31 / 0.54 |
| 4 | grad_step=0.55, poisson_ratio=0.3, regularizer=navier, s=2.0, sobolev_alpha=1.5 | refine1 | 1 | +0.0301 | yes | 0.6204 / 0.0000 / 0.75 / 1.16 | 0.6286 / 0.0000 / 0.30 / 1.31 | 0.6544 / 0.0000 / 0.31 / 0.67 |
| 5 | grad_step=0.55, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=1.35 | refine1 | 1 | +0.0299 | yes | 0.6192 / 0.0000 / 0.30 / 0.64 | 0.6288 / 0.0000 / 0.30 / 0.80 | 0.6549 / 0.0000 / 0.30 / 1.24 |
| 6 | grad_step=0.65, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=1.8 | screen | 1 | +0.0298 | yes | 0.6232 / 0.0000 / 0.30 / 0.84 | 0.6251 / 0.0000 / 0.30 / 0.90 | 0.6542 / 0.0000 / 0.30 / 0.70 |
| 7 | grad_step=0.55, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=1.8 | start | 1 | +0.0293 | yes | 0.6226 / 0.0000 / 0.30 / 0.81 | 0.6245 / 0.0000 / 0.30 / 1.02 | 0.6537 / 0.0000 / 0.30 / 0.57 |
| 8 | grad_step=0.55, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=1.875 | refine1 | 1 | +0.0292 | yes | 0.6229 / 0.0000 / 0.30 / 0.66 | 0.6242 / 0.0000 / 0.30 / 1.29 | 0.6537 / 0.0000 / 0.30 / 0.60 |
| 9 | grad_step=0.55, poisson_ratio=0.15, regularizer=navier, s=2.0, sobolev_alpha=1.8 | screen | 1 | +0.0289 | yes | 0.6217 / 0.0000 / 0.30 / 0.63 | 0.6236 / 0.0000 / 0.30 / 0.79 | 0.6544 / 0.0000 / 0.30 / 0.55 |
| 10 | grad_step=0.55, poisson_ratio=0.2, regularizer=navier, s=2.0, sobolev_alpha=1.5 | refine1 | 1 | +0.0287 | yes | 0.6203 / 0.0000 / 0.30 / 1.08 | 0.6247 / 0.0000 / 0.30 / 1.20 | 0.6544 / 0.0000 / 0.30 / 0.71 |
| 11 | grad_step=0.5, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=1.5 | refine1 | 1 | +0.0287 | yes | 0.6191 / 0.0000 / 0.30 / 0.82 | 0.6267 / 0.0000 / 0.31 / 0.72 | 0.6533 / 0.0000 / 0.30 / 0.74 |
| 12 | grad_step=0.55, poisson_ratio=0.35, regularizer=navier, s=2.0, sobolev_alpha=1.8 | screen | 1 | +0.0284 | yes | 0.6194 / 0.0000 / 0.30 / 0.71 | 0.6272 / 0.0000 / 0.30 / 0.60 | 0.6517 / 0.0000 / 0.30 / 0.56 |
| 13 | grad_step=0.55, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=1.2 | screen | 1 | +0.0282 | yes | 0.6167 / 0.0000 / 0.30 / 0.98 | 0.6263 / 0.0000 / 0.30 / 0.80 | 0.6547 / 0.0000 / 0.30 / 0.62 |
| 14 | grad_step=0.55, poisson_ratio=0.0, regularizer=navier, s=2.0, sobolev_alpha=1.8 | screen | 1 | +0.0281 | yes | 0.6208 / 0.0000 / 0.39 / 0.89 | 0.6242 / 0.0000 / 0.30 / 0.60 | 0.6525 / 0.0000 / 0.30 / 0.54 |
| 15 | grad_step=0.8, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=1.8 | screen | 1 | +0.0280 | yes | 0.6240 / 0.0000 / 0.30 / 0.81 | 0.6222 / 0.0000 / 0.30 / 0.67 | 0.6510 / 0.0000 / 0.30 / 0.53 |
| 16 | grad_step=0.45, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=1.8 | screen | 1 | +0.0280 | yes | 0.6201 / 0.0000 / 0.30 / 0.90 | 0.6240 / 0.0000 / 0.30 / 0.62 | 0.6531 / 0.0000 / 0.32 / 0.58 |
| 17 | grad_step=0.35, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=1.8 | screen | 1 | +0.0276 | yes | 0.6197 / 0.0000 / 0.30 / 0.83 | 0.6241 / 0.0000 / 0.30 / 0.58 | 0.6523 / 0.0000 / 0.30 / 0.46 |
| 18 | grad_step=0.55, poisson_ratio=0.25, regularizer=navier, s=2.0 | screen | 1 | +0.0275 | yes | 0.6226 / 0.0000 / 0.30 / 1.08 | 0.6220 / 0.0000 / 0.30 / 0.73 | 0.6509 / 0.0000 / 0.30 / 0.61 |
| 19 | grad_step=0.55, poisson_ratio=0.42, regularizer=navier, s=2.0, sobolev_alpha=1.8 | screen | 1 | +0.0261 | yes | 0.6164 / 0.0000 / 0.30 / 0.77 | 0.6261 / 0.0000 / 0.30 / 0.70 | 0.6490 / 0.0000 / 0.30 / 1.00 |
| 20 | grad_step=0.55, poisson_ratio=0.25, regularizer=navier, s=2.0, sobolev_alpha=2.75 | screen | 1 | +0.0250 | yes | 0.6222 / 0.0000 / 0.32 / 0.72 | 0.6187 / 0.0000 / 0.30 / 0.58 | 0.6471 / 0.0000 / 0.30 / 0.50 |
| 21 | grad_step=0.55, poisson_ratio=0.25, regularizer=navier, s=3.0, sobolev_alpha=1.8 | screen | 1 | +0.0185 | yes | 0.6186 / 0.0000 / 0.29 / 0.37 | 0.6110 / 0.0000 / 0.30 / 0.30 | 0.6389 / 0.0000 / 0.30 / 0.30 |
| 22 | defaults | baseline | 2 | +0.0000 | yes | 0.5850 / 0.0000 / 0.39 / 0.76 | 0.5974 / 0.0000 / 0.51 / 0.89 | 0.6308 / 0.0000 / 0.30 / 0.55 |

Violations of infeasible configurations:

