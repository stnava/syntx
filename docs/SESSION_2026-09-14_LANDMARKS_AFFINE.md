# Session record — 2026-09-14/15: landmarks physical-space fix, landmark guidance, reproducible affine

Releases: **v5.2.0** (landmarks), **v5.3.0** (affine). All commits local, nothing pushed.

## What was delivered

| area | outcome | where |
|---|---|---|
| `syntx.landmarks` physical space | XYZ tensor layout, LPS-correct display, frame-independent SIFT3D/MIND, PCA-seeded rotation search (60° recovered), MPS `F.pad` workaround | `docs/LANDMARKS_GUIDE.md`, `docs/reports/mbhard_landmarks_spatial_report.html`, tests `tests/test_landmarks_*.py` (28) |
| landmark guidance benchmark | landmark affine ≈ standard affine at ¼ cost; K=8 landmark-cluster soft-Dice channels give +0.0096 Dice to Sobolev SyN on mbhard | `docs/reports/mbhard_landmark_guidance_report.html`, `results/mbhard_landmark_guidance.json` |
| `robust_affine(mode='pytorch')` | equals/beats ANTs C++ Affine on 10/10 Mindboggle pairs (default +0.0013, accurate +0.0031 mean Dice), 3× faster, bitwise reproducible | `docs/AFFINE_GUIDE.md`, `docs/reports/affine_cohort10_report.html`, `results/affine_baseline/`, tests `tests/test_affine_reproducibility.py`, `tests/test_mattes_mi_determinism.py` |

## Root causes found (worth remembering)

1. `F.pad` on MPS corrupts 5-D tensors with 256×256 slices → slicing/cat only (`blob._shift_pad`).
2. MPS matmul over ~10⁶ Parzen weights is non-deterministic and wrong → blocked `bmm` (`_parzen_joint_histogram`).
3. MI masked to fixed foreground never penalises the moving brain spilling into background → whole-domain sampling (`mask_mode='none'`), matching ANTs.
4. Pyramid pooled voxels mapped to `level·i` instead of `level·i + (level−1)/2` and normalised with the full-res shape → fixed; made dense coarse stages viable.
5. `antstorch.denoise_image` did not exist until the ANTsTorch update this session; earlier "N4+NLM" runs were N4 only. N4 is now off by default in landmark preprocessing.

## Open items / suggestions

* Re-baseline SyN + Sobolev cohort numbers: denoising now really runs, and the GEMINI.md union-mask MI rule was reversed on evidence for the affine objective (other MI users should be re-checked the same way: score the reference transform under your own objective).
* Landmark-cluster guidance (+1 % Dice) was shown on one pair; validate on the cohort before changing SyN defaults.
* `preset='accurate'` timing (≈20 s) was measured uncontended on 6 pairs; the 10-pair run was contended.
* Frame voting from structure-tensor frames is available (`rotation_invariant=True`) but unreliable on real cortex; the PCA-seeded iterative search is the supported path for large rotations.

## Correctness audit (2026-09-15, v5.3.2)

* Full test suite (`pytest tests`, excluding the two long real-data suites run separately): **1073 passed, 15 failed, 19 skipped**.
  All 15 failures were checked against the last importable pre-session commit (`7f52436`, v5.1.5): the 7 that were
  rerun there fail identically (SyN Sobolev `warp.grad is None` TypeError, TVF↔JAX loss parity, `diagnose` brain
  heuristics, 0.95 ≥ 0.95 hit-rate boundaries, `test_native_com_initialization`), and the rest are in the same
  untouched families; one affine coverage test failed only under full-suite load on a 15 s wall-time assertion and
  passes standalone. None involve code changed in this session.
* New tests: `tests/test_affine_known_transform.py` — recovery of a known 3-D affine (TRE < 0.75 mm, presets
  `default` and `fast`), a 40° case rescued by `initial_transform`, and a 2-D recovery; a regression test that
  solver-only kwargs never reach `ants.registration` and that `multi_start` is forwarded.
* Repairs (behaviour-preserving for the validated defaults): `multi_start` is now forwarded to the PyTorch solver;
  solver-only kwargs (`preset`, `n_sample_points`, …) are filtered out of the ANTs path in `mode='auto'`;
  `n_starts` default made consistent (3, as used in the cohort validation).
* Not changed on purpose: the pre-existing failures above, and `robust_affine`'s process-wide
  `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1` (mandated for determinism; note it slows any later ANTs call in the same process).

## Standard benchmark re-run: mbhard (pair 44) SyN + Sobolev (2026-09-15)

Standard evaluation = `syntx.benchmark.evaluate_mindboggle_pair(44, model='sobolev')` (N4 on, cc2, Sobolev α=1.5,
grad step 0.25, `fast_smooth=True`, `syn_sampling=2`, iterations [100,100,20], `robust_affine(mode='auto')`), with its
standard 5-figure report. The Sobolev branch configuration is byte-identical to the one at commit `2ac6505`
(the previous standard-suite run, 2026-09-14 01:17 UTC, `results/complete_validation/pair_044_sobolev.json`;
its "BoxLNCC" label in `validation_summary.json` was metadata only — the model ran cc2 then as now).

| run | denoise actually applied | affine Dice | Sobolev SyN Dice | folding | time* |
|---|---|---|---|---|---|
| 2026-09-14 standard suite (`2ac6505`) | no (`antstorch.denoise_image` did not exist; silent fallback) | 0.3324 (cached affine) | **0.6082** | 0.0033 % | 46 s |
| 2026-09-15 standard suite, `denoise=False` (same conditions) | no | 0.3288 | **0.6123** | 0.0063 % | 117 s |
| 2026-09-15 standard suite, default (`denoise=True`) — `docs/reports/report_pair_044_sobolev.html` | yes | 0.3257 | **0.6019** | 0.0075 % | 133 s |
| ANTs C++ SyN baseline (recorded 2026-09-14) | — | — | 0.5876 | 0.0000 % | 150 s |

\* `syntx_time` = SyN fit + affine; the 14 Sept run reused a cached canonical affine (≈0 s) on an idle machine, today's
runs computed the affine fresh (~40 s); today's SyN fit alone was 93 s. Cause of the remaining fit-time difference not
established.

Reading: **no regression** — under identical conditions the standard suite is +0.0041 Dice better than the last record
and +0.025 above ANTs SyN. The 0.6019 of the new default is lower only because denoising now *really* runs; on this pair,
NLM + N4 costs −0.010 Dice relative to N4 alone (single pair; the cohort-level denoising gain in the findings record was
measured before denoising was actually active and needs re-baselining).

Side experiments on the same pair (all Sørensen–Dice via `compute_bidirectional_dice`, Sep-14 affine A, denoised, no N4,
`results/mbhard_syn_sobolev_drift_check_2026-09-15.json`): CFL Sobolev 0.6025 (reproduces the 14 Sept value exactly);
RegAdam + Gaussian 0.6171 (0.054 % folds), RegAdam + Sobolev 0.6116, RegAdam + Gaussian on raw inputs 0.6036. RegAdam is
the better optimiser for SyN here too (+0.009 to +0.015), but the 20 Aug "0.6340" RegAdam figure predates strict
Sørensen–Dice enforcement and is not a valid reference. The `run_standard_report_demo` path (no denoising, CFL) gave
0.5840 and is not the standard suite.
