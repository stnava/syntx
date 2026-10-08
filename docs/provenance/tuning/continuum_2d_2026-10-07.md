# Tuning report: continuum_syn_2d

- date: 2026-10-07
- dataset: 2d (`TWO_D_PAIRS`, pair 0)
- code: `85edc8d48493f2831fad2189e959bcd0a3ddcf01`
- noise (mean |rep0-rep1| Dice): 0.00000; acceptance margin: 0.00050
- criteria: {'folding_abs': 0.0005, 'folding_rel': 1.0, 'inv_interior_abs': 1.0, 'inv_interior_rel': 1.1, 'positive_jac_if_baseline': True, 'inverse_cap_only_if_baseline_fold_free': True, 'max_pair_drop': 0.001, 'min_gain': 0.0005, 'noise_k': 3.0}
- fixed (not tuned): {'reg_iterations': [100, 100, 20]}
- defaults: {'grad_step': 0.4, 'flow_sigma': 2.4, 'total_sigma': 0.0, 'sobolev_alpha': 2.25, 'fast_smooth': False, 'in_loop_inv_steps': 10, 'optimizer': 'cfl', 'regularizer': 'sobolev', 'reg_iterations': [100, 100, 20]}
- **winner**: {'regularizer': 'navier', 's': 2.0, 'poisson_ratio': 0.35, 'grad_step': 0.3, 'sobolev_alpha': 2.25} (mean Dice gain +0.0021, zero folding, Jmin 0.0528, 4x speedup)

## Accumulated tuning results: `results/tune_2d_continuum` (31 evaluations, 27 configurations)

| rank | configuration | stage | runs | mean Dice | gain | feasible | pair 0: Dice / fold% / Jmin / inv int max / inv max / s |
|---|---|---|---|---|---|---|---|
| 1 | grad_step=0.3, poisson_ratio=0.35, regularizer=navier, s=2.0 | start | 2 | 0.7880 | +0.0021 | yes | 0.7880 / 0.0000 / 0.053 / 2.97 / 2.97 / 7 |
| 2 | flow_sigma=1.2, grad_step=0.3, poisson_ratio=0.35, regularizer=navier, s=2.0 | screen | 1 | 0.7880 | +0.0021 | yes | 0.7880 / 0.0000 / 0.053 / 2.97 / 2.97 / 7 |
| 3 | flow_sigma=1.8, grad_step=0.3, poisson_ratio=0.35, regularizer=navier, s=2.0 | screen | 1 | 0.7880 | +0.0021 | yes | 0.7880 / 0.0000 / 0.053 / 2.97 / 2.97 / 8 |
| 4 | flow_sigma=3.6, grad_step=0.3, poisson_ratio=0.35, regularizer=navier, s=2.0 | screen | 1 | 0.7880 | +0.0021 | yes | 0.7880 / 0.0000 / 0.053 / 2.97 / 2.97 / 8 |
| 5 | flow_sigma=4.8, grad_step=0.3, poisson_ratio=0.35, regularizer=navier, s=2.0 | screen | 1 | 0.7880 | +0.0021 | yes | 0.7880 / 0.0000 / 0.053 / 2.97 / 2.97 / 7 |
| 6 | grad_step=0.3, poisson_ratio=0.35, regularizer=navier, s=2.0, sobolev_alpha=2.5 | start | 1 | 0.7870 | +0.0011 | yes | 0.7870 / 0.0000 / 0.071 / 2.89 / 2.89 / 8 |
| 7 | grad_step=0.3, poisson_ratio=0.3, regularizer=navier, s=2.0 | start | 1 | 0.7861 | +0.0002 | yes | 0.7861 / 0.0000 / 0.045 / 2.97 / 2.97 / 7 |
| 8 | defaults | baseline | 4 | 0.7859 | +0.0000 | yes | 0.7859 / 0.0000 / 0.035 / 2.90 / 3.01 / 29 |
| 9 | beta=5.0, gamma=1.0, grad_step=0.2, regularizer=div_curl | screen | 1 | 0.7857 | -0.0002 | yes | 0.7857 / 0.0000 / 0.042 / 2.42 / 2.42 / 5 |
| 10 | beta=2.5, gamma=1.0, grad_step=0.25, regularizer=div_curl | start | 1 | 0.7857 | -0.0003 | yes | 0.7857 / 0.0000 / 0.038 / 2.65 / 2.65 / 5 |
| 11 | grad_step=0.3, poisson_ratio=0.35, regularizer=navier, s=2.0, sobolev_alpha=1.125 | screen | 1 | 0.7906 | +0.0047 | no | 0.7906 / 0.0000 / 0.073 / 5.89 / 5.89 / 6 |
| 12 | grad_step=0.3, poisson_ratio=0.35, regularizer=navier, s=2.0, sobolev_alpha=1.75 | start | 1 | 0.7836 | -0.0023 | no | 0.7836 / 0.0000 / 0.041 / 3.22 / 3.22 / 6 |
| 13 | beta=10.0, gamma=1.0, grad_step=0.2, regularizer=div_curl | start | 1 | 0.7825 | -0.0035 | no | 0.7825 / 0.0000 / 0.049 / 1.91 / 1.91 / 5 |
| 14 | beta=5.0, gamma=1.0, regularizer=div_curl | start | 1 | 0.7818 | -0.0041 | no | 0.7818 / 0.0000 / 0.038 / 2.77 / 2.77 / 9 |
| 15 | beta=5.0, gamma=1.0, grad_step=0.3, regularizer=div_curl | screen | 1 | 0.7815 | -0.0044 | no | 0.7815 / 0.0000 / 0.039 / 3.19 / 3.19 / 6 |
| 16 | beta=5.0, gamma=0.5, grad_step=0.25, regularizer=div_curl | start | 1 | 0.7814 | -0.0045 | no | 0.7814 / 0.0000 / 0.041 / 2.48 / 2.48 / 5 |
| 17 | grad_step=0.3, poisson_ratio=0.38, regularizer=navier, s=2.0 | start | 1 | 0.7811 | -0.0048 | no | 0.7811 / 0.0000 / 0.077 / 3.03 / 3.03 / 8 |
| 18 | grad_step=0.45, poisson_ratio=0.35, regularizer=navier, s=2.0 | screen | 1 | 0.7805 | -0.0055 | no | 0.7805 / 0.0000 / 0.061 / 2.98 / 2.98 / 12 |
| 19 | dilatation_weight=0.25, grad_step=0.2, regularizer=beltrami | start | 1 | 0.7801 | -0.0059 | no | 0.7801 / 0.0000 / 0.026 / 3.10 / 3.10 / 5 |
| 20 | bulk_modulus=5.0, grad_step=0.3, regularizer=hyperelastic | start | 1 | 0.7800 | -0.0059 | no | 0.7800 / 0.0000 / 0.045 / 1.15 / 1.15 / 7 |
| 21 | grad_step=0.15, poisson_ratio=0.35, regularizer=navier, s=2.0 | screen | 1 | 0.7800 | -0.0060 | no | 0.7800 / 0.0000 / 0.066 / 3.06 / 3.06 / 5 |
| 22 | grad_step=0.225, poisson_ratio=0.35, regularizer=navier, s=2.0 | screen | 1 | 0.7797 | -0.0063 | no | 0.7797 / 0.0000 / 0.096 / 3.12 / 3.12 / 7 |
| 23 | bulk_modulus=10.0, grad_step=0.2, regularizer=hyperelastic | start | 1 | 0.7748 | -0.0111 | no | 0.7748 / 0.0000 / 0.065 / 0.82 / 0.82 / 6 |
| 24 | grad_step=0.6, poisson_ratio=0.35, regularizer=navier, s=2.0 | screen | 1 | 0.7741 | -0.0118 | no | 0.7741 / 0.0000 / 0.100 / 2.62 / 2.62 / 13 |
| 25 | grad_step=0.2, poisson_ratio=0.48, regularizer=navier, s=2.0 | start | 1 | 0.7649 | -0.0210 | no | 0.7649 / 0.0000 / 0.501 / 1.14 / 1.85 / 7 |
| 26 | darcy_permeability=0.1, grad_step=0.2, regularizer=poroelastic | start | 1 | 0.7481 | -0.0379 | no | 0.7481 / 0.0000 / 0.172 / 0.74 / 1.55 / 7 |
| 27 | regularizer=solenoidal | start | 1 | 0.7151 | -0.0708 | no | 0.7151 / 0.0000 / 0.575 / 1.24 / 1.24 / 13 |
