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


## Regularisation sweep on pair 44 (2026-09-28/29) -- affine held constant

Every run: `evaluate_mindboggle_pair(44, "sobolev", **overrides)`, cached canonical affine
`results/canonical_affines/pair_044_pt7_affine.mat` (sha256 `34f14ed8...`, verified unchanged
before/after every run), clean committed checkout, resolved parameters verified against each
variant's overrides, `changed_during_run` false (stages 2-3). "Better" = Dice gain > 0.0005
over the same-session canonical anchor, folding <= 0.0005 %, interior max inverse <= 1.0 mm.
Canonical anchors across sessions: 0.6083 / 0.6085 / 0.6084 / 0.6084 / 0.6085.

### Stage 1 (commit `6a52a9d`, clean; `results/syn_reg_sweep_pair44/`)

| variant              |   dice_sym |   dice_gain_vs_canonical |   folding_pct |   jac_min |   jac_max |   inv_interior_mean_mm |   inv_interior_max_mm |   inv_max_mm |   time_s | better_than_canonical   |
|:---------------------|-----------:|-------------------------:|--------------:|----------:|----------:|-----------------------:|----------------------:|-------------:|---------:|:------------------------|
| sobolev_a1.0         |     0.6120 |                   0.0035 |        0.0021 |    0.0000 |   30.0348 |                 0.0465 |                4.4150 |       4.7419 |  92.5691 | False                   |
| sobolev_flow2.0      |     0.6114 |                   0.0029 |        0.0013 |    0.0000 |   31.3533 |                 0.0408 |                2.2632 |       4.0271 |  83.8002 | False                   |
| dsti_a1.5            |     0.6112 |                   0.0028 |        0.0020 |    0.0000 |   30.7335 |                 0.0412 |                2.9805 |       4.5193 | 108.9788 | False                   |
| dsti1_a0.866         |     0.6112 |                   0.0028 |        0.0094 |    0.0000 |   35.2710 |                 0.0513 |                5.2733 |       5.4060 | 114.9467 | False                   |
| dsti1_a1.5           |     0.6112 |                   0.0027 |        0.0019 |    0.0000 |   30.8864 |                 0.0411 |                2.9534 |       4.4973 | 154.1808 | False                   |
| gaussian_flow2.0     |     0.6109 |                   0.0025 |        0.0261 |    0.0000 |   57.0971 |                 0.0596 |                3.3575 |       5.4043 | 101.2639 | False                   |
| gaussian_flow3.0     |     0.6106 |                   0.0022 |        0.0010 |    0.0000 |   30.8775 |                 0.0459 |                1.4817 |       4.7662 |  79.5695 | False                   |
| dsti1_a2.5           |     0.6094 |                   0.0010 |        0.0000 |    0.0232 |   28.4504 |                 0.0311 |                0.8341 |       2.4418 | 138.9437 | True                    |
| sobolev_flow4.5      |     0.6090 |                   0.0006 |        0.0000 |    0.0235 |   19.1499 |                 0.0338 |                0.6571 |       2.4505 |  68.6571 | True                    |
| reg_canonical_anchor |     0.6084 |                   0.0000 |        0.0000 |    0.0191 |   26.0134 |                 0.0372 |                0.7931 |       2.9787 |  68.4832 | False                   |
| sobolev_a2.0         |     0.6082 |                  -0.0002 |        0.0000 |    0.0272 |   31.4887 |                 0.0327 |                0.8244 |       2.2715 |  84.3636 | False                   |
| gaussian_flow4.5     |     0.6050 |                  -0.0034 |        0.0000 |    0.0307 |   27.9356 |                 0.0342 |                1.2200 |       2.9847 |  63.8244 | False                   |
| sobolev_a3.0         |     0.6020 |                  -0.0064 |        0.0000 |    0.0398 |   22.1959 |                 0.0270 |                0.8723 |       1.8955 |  66.3752 | False                   |
| compact_flow3.0      |     0.5988 |                  -0.0096 |        0.0005 |    0.0000 |   41.6671 |                 0.0458 |                1.9515 |       3.2500 |  68.4159 | False                   |
| sobolev_total0.5     |     0.5014 |                  -0.1071 |        0.0000 |    0.2866 |    3.4065 |                 0.0086 |                0.0898 |       0.1963 |  57.7346 | False                   |
| bspline              |     0.4669 |                  -0.1415 |        0.0000 |    0.0775 |    7.7648 |                 0.0143 |                0.1705 |       0.3936 | 374.8991 | False                   |
| sobolev_total1.0     |     0.4584 |                  -0.1500 |        0.0000 |    0.4034 |    2.4797 |                 0.0056 |                0.0746 |       0.0901 |  55.9607 | False                   |

### Stage 2 (commit `084bd09`, clean; `results/syn_reg_sweep_pair44_stage2/`)

| variant                     |   dice_sym |   dice_gain_vs_canonical |   folding_pct |   jac_min |   jac_max |   inv_interior_mean_mm |   inv_interior_max_mm |   inv_max_mm |   time_s | better_than_canonical   |
|:----------------------------|-----------:|-------------------------:|--------------:|----------:|----------:|-----------------------:|----------------------:|-------------:|---------:|:------------------------|
| s2_sobolev_flow4.5_step0.35 |     0.6117 |                   0.0034 |        0.0000 |    0.0153 |   20.0971 |                 0.0367 |                0.7913 |       2.6969 |  72.8905 | True                    |
| s2_dsti1_a2.0               |     0.6117 |                   0.0033 |        0.0004 |    0.0000 |   20.6553 |                 0.0346 |                0.7818 |       3.4643 |  90.7320 | True                    |
| s2_dsti1_a2.5_step0.35      |     0.6104 |                   0.0020 |        0.0000 |    0.0296 |   27.9535 |                 0.0328 |                0.8805 |       2.6001 |  91.7703 | True                    |
| s2_sobolev_flow4.5_a1.0     |     0.6098 |                   0.0015 |        0.0004 |    0.0000 |   27.7191 |                 0.0401 |                0.8916 |       3.4659 |  77.3510 | True                    |
| s2_dsti1_a2.5_repeat        |     0.6096 |                   0.0012 |        0.0000 |    0.0232 |   28.7371 |                 0.0311 |                0.8501 |       2.4317 |  90.9824 | True                    |
| s2_sobolev_flow4.5_repeat   |     0.6090 |                   0.0006 |        0.0000 |    0.0241 |   19.1351 |                 0.0338 |                0.6570 |       2.4606 |  64.3689 | True                    |
| s2_canonical_anchor         |     0.6084 |                   0.0000 |        0.0000 |    0.0183 |   26.4185 |                 0.0372 |                0.7893 |       2.9691 |  81.9560 | False                   |
| s2_sobolev_flow4.5_a1.25    |     0.6078 |                  -0.0005 |        0.0001 |    0.0000 |   23.7374 |                 0.0366 |                0.7040 |       3.4403 |  72.3394 | False                   |
| s2_sobolev_flow6.0          |     0.6067 |                  -0.0016 |        0.0000 |    0.0328 |   16.0852 |                 0.0311 |                0.6863 |       2.2020 |  65.2908 | False                   |
| s2_dsti1_a3.0               |     0.6061 |                  -0.0022 |        0.0000 |    0.0308 |   26.9695 |                 0.0286 |                0.8641 |       1.9542 |  85.7319 | False                   |
| s2_dsti1_a2.5_flow4.5       |     0.6042 |                  -0.0041 |        0.0000 |    0.0362 |   26.5870 |                 0.0285 |                0.8722 |       1.8460 |  82.6892 | False                   |
| s2_sobolev_total0.1         |     0.5823 |                  -0.0260 |        0.0000 |    0.0819 |   11.5339 |                 0.0188 |                0.2603 |       0.8348 |  60.3157 | False                   |

### Stage 3 (commit `d96efb4`, clean; `results/syn_reg_sweep_pair44_stage3/`)

| variant                    |   dice_sym |   dice_gain_vs_canonical |   folding_pct |   jac_min |   jac_max |   inv_interior_mean_mm |   inv_interior_max_mm |   inv_max_mm |   time_s | better_than_canonical   |
|:---------------------------|-----------:|-------------------------:|--------------:|----------:|----------:|-----------------------:|----------------------:|-------------:|---------:|:------------------------|
| s3_flow4.5_step0.45        |     0.6128 |                   0.0044 |        0.0000 |    0.0210 |   18.5392 |                 0.0376 |                0.8077 |       2.7945 |  88.9360 | True                    |
| s3_flow4.5_step0.5         |     0.6127 |                   0.0042 |        0.0002 |    0.0000 |   20.1527 |                 0.0387 |                1.5485 |       2.8215 | 103.8092 | False                   |
| s3_flow4.0_step0.35        |     0.6121 |                   0.0036 |        0.0000 |    0.0107 |   20.7322 |                 0.0370 |                0.7831 |       2.9689 |  74.3896 | True                    |
| s3_dsti1_a2.0_repeat       |     0.6117 |                   0.0032 |        0.0004 |    0.0000 |   20.6709 |                 0.0345 |                0.7702 |       3.4451 | 105.6668 | True                    |
| s3_flow4.5_step0.35_repeat |     0.6116 |                   0.0031 |        0.0000 |    0.0174 |   20.0688 |                 0.0366 |                0.7927 |       2.6927 |  71.9639 | True                    |
| s3_flow5.0_step0.45        |     0.6109 |                   0.0024 |        0.0001 |    0.0000 |   17.9696 |                 0.0361 |                0.7309 |       2.6755 |  92.8140 | True                    |
| s3_flow5.0_step0.35        |     0.6106 |                   0.0021 |        0.0000 |    0.0238 |   20.4214 |                 0.0356 |                0.7698 |       2.5656 |  94.2404 | True                    |
| s3_canonical_anchor        |     0.6085 |                   0.0000 |        0.0000 |    0.0185 |   26.3629 |                 0.0372 |                0.7895 |       2.9800 |  71.2723 | False                   |

### Findings
- **Every regulariser shows the same trade-off**: weakening it (sobolev alpha 1.0,
  flow_sigma 2.0, dsti/dsti1 alpha <= 1.5, gaussian flow_sigma <= 3.0) gives +0.002 to
  +0.0035 Dice with folding, Jacobian min 0 and 1.5-5 mm interior inverse errors.
  Over-regularising (total_sigma >= 0.1, bspline defaults, alpha 3.0, flow_sigma 6.0) loses Dice.
- **Wider Sobolev post-filter + larger step is a clean gain.** flow_sigma 4.5 lets the step
  grow without folding: step 0.35 -> 0.6117 / 0.6116 (repeat), step 0.45 -> **0.6128
  (+0.0044), 0 % folding, Jacobian min 0.021, max 18.5 (canonical 26), interior inverse
  0.81 mm**, 89 s. Step 0.5 reaches the folding onset (0.0002 %, inverse 1.55 mm).
  flow_sigma 4.0 / step 0.35: 0.6121, clean. flow_sigma 5.0 costs Dice.
- dsti1 alpha 2.0: 0.6117 twice but folding 0.0004 % and Jacobian min 0 both times (borderline);
  ~1.4x slower than sobolev.
- **Single pair.** Best clean pair-44 setting (sobolev, flow_sigma 4.5, grad_step 0.45) must
  be confirmed on other pairs before any change to the canonical defaults.
