# Mindboggle-101 Registration Benchmark (`syntx` TVF / SyNGS / SyN vs `ants.registration` — 90/90 Pairs)

- **Status:** COMPLETE — `90/90` pairs (`100.0%`) in `1506.5 min`
- **Generated (UTC):** 2026-11-15T05:44:47Z
- **syntx Version / Commit:** `v6.37.0` (`eae7b794f9`, dirty=True)
- **Host / Hardware:** `macOS-26.5.2-arm64-arm-64bit-Mach-O` (arm64, 24 cores, MPS=True)
- **Load Average (start → current, 1m):** `2.12 → 8.67` (isolated sequential subprocesses, MPS before CPU)
- **Pairs Evaluated:** `90/90` completed (`40` intra-site, `50` inter-site; shuffled seed=`20261112`)
- **Schedule:** `reg_iterations = [100, 100, 20]`

## 1. Track 1 — Affine Registration (`ants.registration` vs `syntx.robust_affine`)

| arm                  |   n_pairs |   mean_dice_sym |   std_dice_sym |   median_dice_sym |   mean_delta_vs_ants_pp | wins_vs_ants   |   mean_time_s |   mean_speedup_vs_ants | qc_pass_count           |
|:---------------------|----------:|----------------:|---------------:|------------------:|------------------------:|:---------------|--------------:|-----------------------:|:------------------------|
| syntx_affine_default |        90 |          0.3456 |         0.0224 |            0.3457 |                 -0.1438 | 37/90          |        7.3186 |                 0.4873 | 0/90 fail (0/90 diffeo) |
| ants_affine          |        90 |          0.3470 |         0.0219 |            0.3464 |                  0.0000 | 90/90          |        3.5479 |                 1.0000 | 0/90 fail (0/90 diffeo) |

## 2. Track 2 — Deformable Registration & Regularizers (Shared Canonical Affine `pt7`)

| arm               |   n_pairs |   mean_dice_sym |   std_dice_sym |   median_dice_sym |   mean_delta_vs_ants_pp | wins_vs_ants   | wilcoxon_p   |   mean_folding_pct |   worst_min_jac |   mean_fd_folding_pct |   worst_fd_min_jac |   mean_inv_mm |   mean_inv_p95_mm |   mean_time_s |   mean_speedup_vs_ants | qc_pass_count            |
|:------------------|----------:|----------------:|---------------:|------------------:|------------------------:|:---------------|:-------------|-------------------:|----------------:|----------------------:|-------------------:|--------------:|------------------:|--------------:|-----------------------:|:-------------------------|
| tvf_best          |        90 |          0.6440 |         0.0217 |            0.6429 |                  3.4480 | 90/90          | 1.74e-16     |             0.0000 |          0.0043 |                0.0000 |            -0.0178 |        0.0056 |            0.0304 |      184.6438 |                 0.6496 | 0/90 fail (90/90 diffeo) |
| syngs_beltrami    |        90 |          0.6417 |         0.0215 |            0.6404 |                  3.2204 | 90/90          | 1.74e-16     |             0.0000 |          0.0016 |                0.0000 |            -0.0481 |        0.0070 |            0.0137 |       70.0858 |                 1.7079 | 0/90 fail (90/90 diffeo) |
| syn_beltrami      |        90 |          0.6373 |         0.0207 |            0.6348 |                  2.7788 | 90/90          | 1.74e-16     |             0.0000 |          0.0342 |                0.0000 |             0.0328 |        0.0094 |            0.0626 |       52.0523 |                 2.3002 | 0/90 fail (90/90 diffeo) |
| syn_beltrami_proj |        90 |          0.6359 |         0.0210 |            0.6347 |                  2.6363 | 90/90          | 1.74e-16     |             0.0000 |          0.0507 |                0.0000 |             0.0522 |        0.0090 |            0.0601 |       93.2283 |                 1.2852 | 0/90 fail (90/90 diffeo) |
| syn_sobolev       |        90 |          0.6341 |         0.0213 |            0.6347 |                  2.4643 | 90/90          | 1.74e-16     |             0.0000 |          0.0179 |                0.0000 |             0.0117 |        0.0085 |            0.0550 |       44.4323 |                 2.6954 | 0/90 fail (90/90 diffeo) |
| syngs_best        |        90 |          0.6335 |         0.0218 |            0.6326 |                  2.3989 | 90/90          | 1.74e-16     |             0.0000 |          0.0086 |                0.0000 |             0.0101 |        0.0037 |            0.0108 |       62.4829 |                 1.9161 | 0/90 fail (90/90 diffeo) |
| syn_gaussian      |        90 |          0.6328 |         0.0219 |            0.6311 |                  2.3267 | 90/90          | 1.74e-16     |             0.0000 |          0.0350 |                0.0000 |             0.0190 |        0.0098 |            0.0653 |       42.0978 |                 2.8496 | 0/90 fail (90/90 diffeo) |
| syn_hyperelastic  |        90 |          0.6319 |         0.0216 |            0.6299 |                  2.2376 | 89/90          | 1.93e-16     |             0.0000 |          0.0140 |                0.0000 |             0.0199 |        0.0101 |            0.0692 |       53.7741 |                 2.2286 | 0/90 fail (90/90 diffeo) |
| ants_syn_matched  |        90 |          0.6095 |         0.0222 |            0.6096 |                  0.0000 | 90/90          | —            |             0.0000 |          0.0668 |                0.0000 |             0.0668 |        0.0053 |            0.0353 |      119.4522 |                 1.0000 | 0/90 fail (90/90 diffeo) |
| syn_matched       |        90 |          0.6072 |         0.0214 |            0.6070 |                 -0.2259 | 33/90          | 0.0012       |             0.0000 |          0.0653 |                0.0000 |             0.0568 |        0.0052 |            0.0344 |       55.5410 |                 2.1883 | 0/90 fail (90/90 diffeo) |

### 2b. Intra-Site vs. Inter-Site Deformable Subgroup Breakdown

| arm               |   n_intra |   mean_dice_intra |   std_dice_intra |   n_inter |   mean_dice_inter |   std_dice_inter |   iqr_dice_sym |
|:------------------|----------:|------------------:|-----------------:|----------:|------------------:|-----------------:|---------------:|
| tvf_best          |        40 |            0.6484 |           0.0187 |        50 |            0.6404 |           0.0234 |         0.0273 |
| syngs_beltrami    |        40 |            0.6469 |           0.0191 |        50 |            0.6375 |           0.0226 |         0.0266 |
| syn_beltrami      |        40 |            0.6409 |           0.0203 |        50 |            0.6344 |           0.0208 |         0.0260 |
| syn_beltrami_proj |        40 |            0.6395 |           0.0209 |        50 |            0.6330 |           0.0209 |         0.0250 |
| syn_sobolev       |        40 |            0.6366 |           0.0219 |        50 |            0.6321 |           0.0208 |         0.0275 |
| syngs_best        |        40 |            0.6384 |           0.0196 |        50 |            0.6296 |           0.0229 |         0.0295 |
| syn_gaussian      |        40 |            0.6367 |           0.0217 |        50 |            0.6296 |           0.0217 |         0.0263 |
| syn_hyperelastic  |        40 |            0.6355 |           0.0206 |        50 |            0.6290 |           0.0221 |         0.0264 |
| ants_syn_matched  |        40 |            0.6112 |           0.0211 |        50 |            0.6081 |           0.0232 |         0.0298 |
| syn_matched       |        40 |            0.6084 |           0.0224 |        50 |            0.6063 |           0.0208 |         0.0273 |
