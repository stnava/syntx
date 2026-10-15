# GPU sync and copy reduction plan for `syntx.syn` (CUDA and MPS)

Status: WP1 to WP4 implemented in v6.11.0 (see `docs/RELEASE_NOTES_6.11.0.md`); WP5 and WP6 open.
Post-change counts on the same setup: 1.8 syncs and 1 clone per epoch. Timing after the change was
not recorded (other GPU load on the host); run the protocol in section 4 on an idle device.
Drafted 2026-10-15 from measurements on Apple Silicon MPS,
torch 2.13.0, syntx 6.10.0. CUDA numbers are not measured; the structural argument transfers, the
magnitudes do not.

## 1. Question

A colleague reported that syntx "loops over `.item()` calls along with `.clone()`" and that this may
explain GPU timing differences. This document records what was measured and what to change.

## 2. Measurement

Script: `count_syncs.py` and `sync_cost.py` (session scratchpad; reproduce from the description
below). Setup: MNI template resampled to 3 mm (61x73x61) and 2 mm (91x109x91); moving image is the
fixed image warped by a B-spline random displacement field (sd 6 mm, 30 control points, stationary
boundary). `syntx.registration(type_of_transform='SyNOnly', reg_iterations=(20, 10), device='mps')`
with all other defaults (optimizer `cfl`, Anderson inverse, `in_loop_inv_steps=10`,
`project_inverse=True`, `check_interval=1`). A warm-up call precedes timing. Every call of
`Tensor.item`, `__bool__`, `__float__`, `cpu`, `numpy`, `clone` on a non-CPU tensor is attributed
to the innermost syntx frame.

### 2.1 Sync points per SyN epoch (3 mm, 30 epochs)

| Site | Kind | Per epoch | Role |
| --- | --- | ---: | --- |
| `core/inverse.py:450` | `.item()` | 12.5 | Anderson convergence test, every step, 4 inverse calls per epoch |
| `syn.py:1366` | `.item()` | 1.0 | total loss to Python float (best tracking, convergence window) |
| `syn.py:1276` | `.item()` | 1.0 | per-metric loss into `metric_losses_dict` (only read when verbose) |
| `syn.py:1539` | `bool(tensor)` | 1.0 | `if max_norm_l > 1e-12` on a device tensor |
| `syn.py:1544` | `bool(tensor)` | 1.0 | `if max_norm_r > 1e-12` on a device tensor |
| total | | 16.5 | |

### 2.2 Clones per SyN epoch

| Site | Per epoch | Role |
| --- | ---: | --- |
| `core/inverse.py:491` | 8.5 | `g_newest.clone()` as the Anderson candidate accumulator |
| `core/inverse.py:437` | 2.0 | `W_inv_disp.clone()` at inverse entry |
| `syn.py:1384-1387` | 4 x 0.7 | snapshot of four warps whenever the loss improves |

### 2.3 Time blocked inside sync calls

| Resolution | Wall per epoch | Blocked in sync calls | Blocked share |
| --- | ---: | ---: | ---: |
| 3 mm (61x73x61) | 167 ms | 51 ms | 30.6 % |
| 2 mm (91x109x91) | 249 ms | 74 ms | 29.7 % |

Blocked time is the host waiting for queued kernels. The GPU is busy during that wait, so this is
not wasted GPU time. The loss is the serialization: with a sync every few kernels, host launch work
and device execution run one after the other instead of overlapping. The torch profiler for 5 epochs
at 2 mm shows host self-time of about 0.7 s outside sync waits against 1.25 s inside them, so the
host launch path and the device path are of the same order. Removing syncs lets them overlap and
bounds the gain at roughly the blocked share, about 30 %, before any kernel-level work.

### 2.4 Other host-side costs seen in the profile (2 mm, 5 epochs)

| Op | Calls | Host self time | Note |
| --- | ---: | ---: | --- |
| `aten::_local_scalar_dense` | 321 | 1.245 s | the syncs above |
| `aten::copy_` | 2938 | 165 ms | scalar writes into the Anderson Gram matrix, snapshot copies |
| `aten::linalg_solve_ex` + `linalg_inv_ex` | 105 | 264 ms total | 5x5 Anderson solve on MPS, about 2.5 ms each |
| `aten::mm` | 264 | 103 ms | coordinate transforms |
| `aten::grid_sampler_3d` | 169 | 85 ms | sampling |

The Anderson step builds the Gram matrix with a double Python loop of `torch.dot` and scalar
assignments and then calls `torch.linalg.solve` on device. On MPS a 5x5 solve costs about 2.5 ms
of host time, more than a full-volume `grid_sample`. This is a second, independent inefficiency.

### 2.5 Verdict on the colleague's claim

- `.item()` in loops: correct and material. About 16 syncs per epoch, three quarters of them from
  the per-step Anderson convergence check. Removing them is worth up to about 30 % on MPS at
  these sizes. The share will fall at 1 mm, where kernels are longer and the fixed per-sync cost
  is amortized, and it will differ on CUDA.
- `.clone()` in loops: minor. About 11 full-field clones per epoch. At 2 mm one field is 10.8 MB,
  at 1 mm 151 MB, so at 1 mm the clones move about 1.7 GB per epoch, around 10 to 20 ms on
  Apple Silicon bandwidth. They cause allocator churn rather than compute, and the caching
  allocator hides most of it. Replace with `copy_` into preallocated buffers, but do not expect a
  large win.
- A prior audit (`docs/compute_and_memory_efficiency_audit.md`, September 2026) already removed
  many sync and copy sites. What remains is listed above; the earlier audit's "38 to 10 syncs per
  Anderson call" is consistent with the 12.5 per epoch measured here (about 3 per call).

## 3. Plan

Each item is one tracking row, one owner, one commit. Gate for every item: 2D and 3D parity on
the existing test suite (`tests/test_syn*.py`, `tests/test_inverse*.py`), loss trajectory within
float32 noise of the baseline, and the per-epoch sync count from `count_syncs.py` recorded in the
commit body. Baseline must be green first.

### WP1. Anderson convergence check without a per-step sync (largest win)

Files: `src/syntx/core/inverse.py`, call sites in `src/syntx/syn.py` (lines 1613-1665, 1806-1842).

1. Batch the two half-warp inverses (`warp_l2r` and `warp_r2l`, and the two projection calls when
   `project_inverse`) into one call with batch dimension 2 so one convergence test covers both.
   The fixed-point step, smoothing and `sample_field_cf` are already batch-safe; verify with
   `inspect` and a unit test on a batch of two distinct fields against two single calls.
2. Replace per-step checking with predicted-step checking: keep the number of steps the previous
   epoch needed (warm start converges in about 3 steps), run that many steps unchecked, then test
   once; if not converged, continue with per-step checks. Expected: 1 to 2 syncs per epoch for
   the inverse instead of 12.5, same iterate as today because the check is only deferred.
3. Expose `check_interval` from `registration()` for the user who wants the old behaviour.

Risk: a deferred check can run one or two extra Anderson steps per call. Measure cost against
saved syncs; report both.

### WP2. Remove the two tensor-bool branches in the CFL step

File: `src/syntx/syn.py:1539-1548`. `max_norm_l_safe` is already clamped at 1e-8, so the
`if max_norm_l > 1e-12` branch only guards a zero gradient. Replace the branch with
`torch.where(max_norm_l > 1e-12, effective_cfl / max_norm_l_safe, 0.0) * grad_l`. Zero syncs,
bitwise identical result for every non-degenerate gradient; the degenerate case gives zeros as
before.

### WP3. Loss bookkeeping on device, sync every k epochs

File: `src/syntx/syn.py:1276, 1366, 1384-1387, 1855`.

1. Keep `loss` as a device tensor; track `best_level_loss` on device with `torch.minimum`.
2. Snapshot the best warps with a device-side mask instead of four clones:
   `best = torch.where(improved, warp, best)` on preallocated buffers, or `copy_` under a device
   condition. This removes the four conditional clones and the host decision.
3. Sync the loss to Python every 5 epochs (as `tvf.py:1523` already does) for the convergence
   window and divergence warning, and at the last epoch. The convergence test then lags by at
   most 4 epochs; state this in the docstring.
4. Fill `metric_losses_dict` only when `verbose` or on the final epoch.

Expected: 2 syncs per epoch become 0.2.

### WP4. Anderson Gram matrix and solve

File: `src/syntx/core/inverse.py:470-490`.

1. Stack the residual differences once, `dR = torch.stack(dR_cols)` with shape (n, N); compute
   `gram = dR @ dR.T` and `rhs = dR @ r_newest` as two matmuls instead of n(n+1)/2 dots and
   n(n+1) scalar copies.
2. Solve the n x n system (n <= 5) with a closed-form or Cholesky path that is cheap on device, or
   evaluate whether `torch.linalg.solve` on a (n, n) CPU tensor after one small transfer is
   cheaper than the 2.5 ms MPS solve. The transfer is a sync, so this choice interacts with WP1;
   prefer the on-device route and measure.
3. Replace `g_newest.clone()` plus the loop of subtractions with one matmul:
   `v_new = g_newest - gamma @ (dG + dR)`.
4. Drop the `try/except LinAlgError`: on device the error is raised lazily if at all; add a
   deterministic regularization and a unit test with a singular Gram matrix.

Expected: fewer than 10 kernels per Anderson step instead of about 60; removes about 250 ms of
host time per 5 epochs at 2 mm.

### WP5. Preallocated buffers for remaining per-epoch clones

Files: `src/syntx/core/inverse.py:437`, `src/syntx/syn.py:1098-1101`. `copy_` into buffers
allocated once per level. Low priority; do after WP1 to WP4 and only if the measured allocator
churn at 1 mm justifies it.

### WP6. Launch-overhead reduction (after the syncs are gone)

Once the host no longer waits, the host launch path (about 600 kernels per epoch) becomes the
floor. Candidates, in order: `torch.compile` of the fixed-point step and the smoothing path on
CUDA (CUDA Graphs through `mode='reduce-overhead'`); on MPS compile support is partial in
torch 2.13 and must be measured, not assumed. Separate tracking row; not part of this plan's gate.

## 4. Verification protocol

1. Baseline: `pytest tests/test_syn.py tests/test_inverse*.py` green; record counts.
2. `count_syncs.py` before and after each WP; sync and clone counts per epoch in the commit body.
3. Timing at 3 mm, 2 mm and 1 mm on MPS, 5 repeats, median, `torch.mps.synchronize()` bracketed.
   Same script on a CUDA host when one is available; until then the CUDA claim is unverified.
4. Parity: final loss and warp L-infinity against the pre-change run at float32 tolerance 1e-5
   relative; Mindboggle 10-pair Dice unchanged within 1e-3 (`docs/mindboggle_benchmark_guide.md`).
5. HTML report under `docs/` with the before/after table.
