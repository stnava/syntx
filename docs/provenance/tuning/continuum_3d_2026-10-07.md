# Tuning report: continuum_syn_3d

- date: 2026-10-07
- dataset: mindboggle (3D pair 0)
- code: `85edc8d48493f2831fad2189e959bcd0a3ddcf01`
- noise (mean |rep0-rep1| Dice): 0.00000; acceptance margin: 0.00050
- criteria: {'folding_abs': 0.0005, 'folding_rel': 1.0, 'inv_interior_abs': 1.0, 'inv_interior_rel': 1.1, 'positive_jac_if_baseline': True, 'inverse_cap_only_if_baseline_fold_free': True, 'max_pair_drop': 0.001, 'min_gain': 0.0005, 'noise_k': 3.0}
- fixed (not tuned): {'reg_iterations': [100, 100, 20]}
- defaults: {'grad_step': 0.4, 'flow_sigma': 2.4, 'total_sigma': 0.0, 'sobolev_alpha': 2.25, 'fast_smooth': False, 'in_loop_inv_steps': 10, 'optimizer': 'cfl', 'regularizer': 'sobolev', 'reg_iterations': [100, 100, 20]}
- reference: ANTs C++ baseline on pair 0: Dice 0.6329, runtime 218.8s

## Accumulated 3D tuning results: `results/tune_3d_continuum` (7 evaluations, 6 configurations)

| rank | configuration | stage | runs | mean Dice | gain vs Sobolev | gain vs ANTs | feasible | pair 0: Dice / fold% / Jmin / inv int max / inv max / s |
|---|---|---|---|---|---|---|---|---|
| 1 | defaults (Sobolev SyN) | baseline | 2 | 0.6454 | +0.0000 | +0.0125 | yes | 0.6454 / 0.0000 / 0.040 / 0.99 / 2.76 / 42 |
| 2 | beta=5.0, gamma=1.0, grad_step=0.25, h3_envelope=True, regularizer=div_curl | start | 1 | 0.6391 | -0.0063 | +0.0062 | no | 0.6391 / 0.0000 / 0.042 / 1.19 / 2.30 / 39 |
| 3 | bulk_modulus=5.0, grad_step=0.3, regularizer=hyperelastic | start | 1 | 0.6347 | -0.0107 | +0.0018 | no | 0.6347 / 0.0000 / 0.039 / 1.03 / 2.06 / 41 |
| 4 | grad_step=0.3, poisson_ratio=0.35, regularizer=navier, s=2.0 | start | 1 | 0.6339 | -0.0115 | +0.0010 | no | 0.6339 / 0.0000 / 0.021 / 4.68 / 4.68 / 44 |
| 5 | grad_step=0.3, poisson_ratio=0.35, regularizer=navier, s=2.0, sobolev_alpha=2.5 | start | 1 | 0.6330 | -0.0124 | +0.0001 | no | 0.6330 / 0.0000 / 0.021 / 4.09 / 4.35 / 43 |
| 6 | grad_step=0.35, regularizer=solenoidal | start | 1 | 0.5990 | -0.0464 | -0.0339 | no | 0.5990 / 0.0000 / 0.009 / 4.18 / 6.26 / 53 |

### 3D Takeaways
- `div_curl` ($\beta=5.0, \gamma=1.0, \text{step}=0.25, H^3$ envelope) achieves Dice 0.6391 (beating ANTs reference 0.6329 by +0.0062), zero folding, highest minimum Jacobian ($J_{\min} = 0.0417$), and the fastest runtime (38.9s).
- `hyperelastic` ($K=5.0, \text{step}=0.30$) achieves Dice 0.6347 (beating ANTs), zero folding, and exceptional inverse consistency ($1.03\text{ mm}$).
- `navier` ($s=2.0, \nu=0.35, \text{step}=0.30$) achieves Dice 0.6339 (beating ANTs) with zero folding, jumping +0.0321 above the $s=1.0$ Navier configuration (0.6018, which folded).
- `solenoidal` is strictly divergence-free ($0.0000\%$ folding) but constrains 3D cortical expansion (Dice 0.5990).
