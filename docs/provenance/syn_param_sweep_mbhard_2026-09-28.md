# syntx.syn parameter sweep on mbhard (Mindboggle pair 44) -- 2026-09-28

**Provenance (machine-captured, `syntx.provenance`):** every run from commit `1dba81e`
(v5.4.47), clean checkout, runner `scripts/sweep_syn_mbhard_parameters.py` (text embedded in
each manifest), MPS, load average 1.9-4.0. Per-variant JSON with full manifests and standard
reports: `results/syn_param_sweep_mbhard/` (local, git-ignored). Resolved parameters were
verified against each variant's intended overrides.

Path: `evaluate_mindboggle_pair(44, "sobolev", **overrides)` -- N4, 2-98 % normalisation,
cached canonical affine (`pt7`), canonical `syntx.syn` defaults plus the overrides below.
Inverse error = |inv(x) + fwd(x + inv(x))| in mm; interior = 5-voxel-eroded brain mask.

| variant                |   dice_sym |   dice_fixed |   dice_moving |   folding_pct |   jac_min |   jac_max |   inv_mean_mm |   inv_p95_mm |   inv_interior_mean_mm |   inv_interior_max_mm |   inv_max_mm |   time_s |   load_avg_1m_at_end |
|:-----------------------|-----------:|-------------:|--------------:|--------------:|----------:|----------:|--------------:|-------------:|-----------------------:|----------------------:|-------------:|---------:|---------------------:|
| canonical              |     0.6083 |       0.6122 |        0.6045 |        0.0000 |    0.0188 |   26.2474 |        0.0114 |       0.0685 |                 0.0372 |                0.8201 |       2.9808 |  69.2247 |               2.1504 |
| fast_smooth_on         |     0.6123 |       0.6160 |        0.6085 |        0.0031 |    0.0000 |   29.4485 |        0.0139 |       0.0803 |                 0.0504 |                5.6493 |       5.6493 |  90.7129 |               2.0283 |
| drift_0920_committed   |     0.6129 |       0.6171 |        0.6087 |        0.0090 |    0.0000 |   31.0629 |        0.0144 |       0.0832 |                 0.0533 |                6.7037 |       6.7037 |  89.4670 |               2.1089 |
| in_loop_6              |     0.6084 |       0.6123 |        0.6045 |        0.0000 |    0.0186 |   26.4062 |        0.0114 |       0.0685 |                 0.0372 |                0.7891 |       2.9814 |  68.1666 |               1.8501 |
| alpha_0866             |     0.6111 |       0.6131 |        0.6091 |        0.0067 |    0.0000 |   28.8542 |        0.0137 |       0.0801 |                 0.0493 |                5.9892 |       5.9892 |  93.5697 |               3.0171 |
| step_035               |     0.6102 |       0.6144 |        0.6060 |        0.0003 |    0.0000 |   26.6327 |        0.0121 |       0.0717 |                 0.0397 |                0.8187 |       3.3132 |  76.6932 |               3.9619 |
| no_stationary_boundary |     0.6085 |       0.6124 |        0.6046 |        0.0000 |    0.0207 |   26.3675 |        0.0114 |       0.0683 |                 0.0371 |                0.7905 |       2.9853 | 121.1431 |               2.6123 |
| canonical_repeat       |     0.6085 |       0.6123 |        0.6046 |        0.0000 |    0.0184 |   26.5015 |        0.0115 |       0.0685 |                 0.0372 |                0.7909 |       2.9971 |  68.2930 |               2.8525 |

| variant                | description                                                                             |
|:-----------------------|:----------------------------------------------------------------------------------------|
| canonical              | canonical defaults: step 0.25, flow 3.0, fast_smooth off, alpha 1.5, in-loop inverse 10 |
| fast_smooth_on         | canonical + fast_smooth on (= the 2026-09-20 record's stated SyN settings)              |
| drift_0920_committed   | config.py at 3e862c9: step 0.35, flow 2.5, fast_smooth on, in-loop inverse 6            |
| in_loop_6              | canonical + in-loop inverse 6 (pre-5.4.46 default)                                      |
| alpha_0866             | canonical + sobolev_alpha 0.866 (pre-5.4.46 implicit default)                           |
| step_035               | canonical + grad_step 0.35                                                              |
| no_stationary_boundary | canonical + stationary_boundary off (pre-5.4.45)                                        |
| canonical_repeat       | canonical again: run-to-run (MPS) noise                                                 |

## Findings (single pair; confirm on more pairs before cohort-level claims)
- **Run-to-run noise** (canonical vs canonical_repeat): Dice 0.6083 vs 0.6085, interior
  max inverse 0.82 vs 0.79 mm, 69 vs 68 s.
- **Less regularisation -> small Dice gain, broken topology and inverse.** `fast_smooth` on
  (drops the Gaussian post-filter) and alpha 0.866 (weaker Sobolev) behave alike: +0.003 to
  +0.004 Dice, folding 0.003-0.007 %, Jacobian min 0, interior max inverse error ~0.8 ->
  ~6 mm, and ~30 % slower (91-94 s vs 69 s).
- **The committed 2026-09-20 config** (step 0.35, flow 2.5, fast_smooth on, in-loop
  inverse 6) is the least regularised: +0.0046 Dice, folding 0.009 %, inverse 6.7 mm --
  reproducing the earlier "standard" mbhard runs. Consistent with the 2026-09-20 cohort's
  syn folding (0.0066 %).
- **Step 0.35 alone**: +0.0019 Dice, folding 0.0003 % (Jacobian min 0), inverse unchanged
  (0.82 mm), 77 s. A milder trade than reducing smoothing.
- **in_loop_inv_steps 6 vs 10**: indistinguishable (Dice, folding, inverse, time).
- **stationary_boundary off**: accuracy identical (content is far from the FOV edge), but
  121 s vs 68-69 s -- unexplained, single run.
- **Conclusion**: canonical (`fast_smooth` off, alpha 1.5) is the only configuration with
  zero folding and a sub-mm interior inverse, and it is also the fastest.
