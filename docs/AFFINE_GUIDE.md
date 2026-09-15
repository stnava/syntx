# `syntx.robust_affine(mode='pytorch')` — fast, reproducible affine registration

Native PyTorch (MPS / CUDA / CPU) multi-resolution affine registration with a Mattes mutual
information objective, designed to be **computationally reproducible** (bitwise on a given
device) and to **match or beat the ANTs C++ affine** on Mindboggle brain pairs at a fraction of
the time.

**Default backend since 2026-09-15**: `robust_affine(mode='auto')` — the default mode, and what every
benchmark path uses — now runs this PyTorch solver, with the ANTs C++ pipeline kept as `mode='ants_fast'`
and as an automatic fallback if the solver raises. End-to-end check on 5 Mindboggle pairs (identical
standard SyN + Sobolev downstream): mean downstream Dice +0.0013, worst −0.0005, affine 3–4× faster
(10–18 s vs 37–61 s), bitwise reproducible instead of varying run to run
(`results/complete_validation/affine_backend_endtoend_cohort_2026-09-15.json`).

## Quick start

```python
from syntx.robust_affine import robust_affine
from syntx.landmarks import preprocess_for_landmarks      # NLM denoise + 2–98 % normalisation

fi_p, mi_p = preprocess_for_landmarks(fixed), preprocess_for_landmarks(moving)
r = robust_affine(fi_p, mi_p, mode="pytorch", device="mps", seed=42)          # preset='default'
r = robust_affine(fi_p, mi_p, mode="pytorch", preset="accurate")               # ~2x slower, higher Dice
r = robust_affine(fi_p, mi_p, mode="pytorch", initial_transform="landmarks.mat")  # extra start candidate
r["fwdtransforms"]          # ITK .mat (fixed mm -> moving mm), use with ants.apply_transforms
```

Inputs should be foreground-normalised to [0, 1] (the solver normalises internally as well); the
MI histogram uses fixed bounds (0, 1).

## How it works

1. **Candidates** at the coarsest level: identity at the centre of mass, single-axis rotations of
   ±4/8/12°, and any `initial_transform` (e.g. a landmark affine from
   `syntx.landmarks.match_sift3d_with_rotation_search`), scored by exact MI; the best `n_starts`
   (default 3) are optimised in parallel.
2. **Schedule** (`_default_affine_schedule`, override with `schedule=` or `preset=`):

   | preset | stages | 6-pair Δ Dice vs ANTs | time (MPS) |
   |---|---|---|---|
   | `default` | L4 rigid exact → L3 affine exact (100 it, select) → L2 → L1 point-sampled | +0.0002, 6/6 ties or wins | ~11 s |
   | `accurate` | L4 rigid exact → L2 affine on regular 50 % grid (100 it, select) → L1 | +0.0021, 6/6 wins | ~20 s |
   | `fast` | all point-sampled (100k) | −0.0015 | ~5 s |

   Each stage is Adam with cosine annealing (or `optimizer='lbfgs'`); parameters are translation,
   rotation vector, log-scales and shears about the fixed centre of mass.
3. **Objective**: negative Mattes MI, 32 cubic-B-spline Parzen bins, boundary-padded, fixed
   bounds, evaluated over the **whole fixed domain** (`mask_mode='none'`, as ANTs does). Dense
   stages use the exact voxel grid; point-sampled stages a fixed, seeded 100k-voxel sample.
4. **Determinism**: seeded candidates and samples; the joint histogram is accumulated with a
   blocked `bmm` (`core.losses._parzen_joint_histogram`) because a single large matmul is
   non-deterministic and inaccurate on Apple MPS; strided subsamples are made contiguous.

## Validation (Mindboggle, `scripts/benchmark_affine_cohort.py`)

Ten pairs drawn with seed 7 from `examples/pairs.csv`; ANTs C++ `Affine` (default threading) as
reference; symmetric DKT31 Dice via `compute_bidirectional_dice`.


| arm | mean Dice | Δ vs ANTs | wins/ties | worst Δ | mean time | reproducible |
|---|---|---|---|---|---|---|
| ANTs C++ `Affine` (default threads) | 0.3503 | — | — | — | 34 s | no (run-to-run Δparam up to 0.33) |
| syntx `preset='default'` | 0.3516 | +0.0013 | 10/10 | -0.0019 | 11.2 s | yes (bitwise) |
| syntx `preset='accurate'` | 0.3527 | +0.0031 | 10/10 | +0.0002 | ~20 s uncontended* | yes (bitwise) |
| syntx default + landmark seed | 0.3523 | +0.0020 | 9/10 | -0.0024 | default + ~16 s landmarks | yes |
| landmark least-squares affine alone | 0.3377 | -0.0125 | 2/10 | -0.0322 | ~16 s | yes |

\* the `accurate` cohort run was heavily contended (ANTs took up to 1000 s on the same pairs); the 6-pair sweep
measured 17–22 s per pair uncontended (`results/affine_baseline/sweep_cohort_subset8.json`).
ANTs Dice differs slightly between the two cohort runs (e.g. pair 04: 0.3384 vs 0.3325) because ANTs is not
run-to-run reproducible; each syntx arm is compared with the ANTs run of its own cohort.

Per pair:

| pair | type | ANTs Dice | default Dice (Δ) | accurate Dice (Δ vs its ANTs run) | default + landmark seed (Δ) |
|---|---|---|---|---|---|
| 04 | intra | 0.3384 | 0.3416 (+0.0032) | 0.3391 (+0.0066) | 0.3410 (+0.0026) |
| 19 | intra | 0.3636 | 0.3617 (-0.0019) | 0.3641 (+0.0006) | 0.3617 (-0.0019) |
| 27 | intra | 0.3499 | 0.3504 (+0.0005) | 0.3518 (+0.0018) | 0.3505 (+0.0006) |
| 49 | inter | 0.3487 | 0.3499 (+0.0013) | 0.3507 (+0.0021) | 0.3492 (+0.0005) |
| 51 | inter | 0.3070 | 0.3073 (+0.0003) | 0.3079 (+0.0009) | 0.3128 (+0.0058) |
| 56 | inter | 0.3251 | 0.3289 (+0.0037) | 0.3325 (+0.0073) | 0.3306 (+0.0055) |
| 66 | inter | 0.3488 | 0.3520 (+0.0032) | 0.3524 (+0.0036) | 0.3520 (+0.0032) |
| 72 | inter | 0.4183 | 0.4182 (-0.0001) | 0.4185 (+0.0002) | 0.4215 (+0.0032) |
| 75 | inter | 0.3509 | 0.3541 (+0.0033) | 0.3527 (+0.0018) | 0.3541 (+0.0033) |
| 76 | inter | 0.3521 | 0.3515 (-0.0006) | 0.3577 (+0.0055) | 0.3498 (-0.0024) |

Verdict: the default preset **equals or beats** the ANTs C++ affine on every pair (no loss beyond 0.002) at a third of
the wall time; the accurate preset is strictly better on all ten pairs. The landmark seed is optional: it helps on some
inter-study pairs (51: +0.0055, 72: +0.0033) and slightly hurts on others (76: −0.0017), so it is not part of the default.


Reproducibility: every syntx run repeated twice per pair gave bitwise-identical parameters
(`param_spread == 0`); ANTs C++ varies run-to-run (max |Δparam| up to 0.33 on mbhard).

## What was wrong before (2026-09-14)

* MPS matmul over ~10⁶ Parzen weights was non-deterministic and wrong → first-call-different
  results and Dice 0.22 on GPU vs 0.30 on CPU.
* MI restricted to the fixed foreground let the moving brain spill into background unpenalised;
  longer optimisation lowered Dice while lowering the loss.
* Pyramid pooled voxels were mapped to `level·i` instead of their centre `level·i + (level−1)/2`,
  and pooled moving tensors were normalised with the full-resolution shape (1.5-voxel bias at
  level 4).

See `GEMINI.md` §"Affine" invariants and `tests/test_affine_reproducibility.py`,
`tests/test_mattes_mi_determinism.py`.
