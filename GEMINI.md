# syntx — working rules for agents and contributors

These are the durable rules distilled from the project's history. Each rule states *what* to do and *why*
in one or two lines; the evidence, numbers and dated experiments live in
`docs/PROJECT_FINDINGS_DETAILED.md` (section numbers in brackets), `docs/AFFINE_GUIDE.md`,
`docs/LANDMARKS_GUIDE.md` and `docs/SESSION_*.md`. When a rule and the code disagree, verify with an
experiment before changing either.

## 1. How to work

* **Verify before explaining.** Never present a hypothesis as a cause. Form it, write the experiment or
  inspection that would falsify it, and report the outcome. Say "I do not know" when that is true. [§9]
* **Attribute failures to verified causes.** Rerun on CPU and GPU, in a fresh process, and at the real data
  size before calling anything "algorithmic divergence" or "transient hardware instability". [§20]
* **Compare against the reference implementation, not against intuition.** When a result trails ANTs/ITK,
  score the *reference's* transform under your own objective first: if it scores better, your optimiser is
  under-converging; if worse, your objective differs from the reference's. [`AFFINE_GUIDE`]
* **Debug order for GPU regressions.** (1) CPU vs GPU bitwise at the real tensor size, (2) reference
  transform under your objective, (3) coordinate conventions on a known-transform phantom, and only then
  (4) hyper-parameter sweeps. Three separate bugs this project hit looked like "needs tuning" until (1)–(3)
  were done. [`SESSION_2026-09-14`]
* **No ad-hoc benchmark scripts.** Use the validated evaluators (`compute_bidirectional_dice`,
  `image_compare`) and the benchmark runners; scratch loops over `ants.label_overlap_measures` have
  produced wrong baselines before. [§4]
* **Fine-level optimizer convergence.** When an optimizer under-converges on smooth images, inspect
  fine-pyramid sampling before tuning learning rates. Random point-sampling (<5% domain coverage) at
  L2/L1 frequently starves the gradient; strided full-grid subsampling (10–25%) provides stable convergence
  while maintaining substantial speedups over full dense grids.
* **Commits and pushes only on explicit user instruction.** Never commit, tag or push as a side effect.

## 2. Registration pipeline invariants

* **Single interpolation.** Never pre-warp inputs to disk; compose all transforms and apply once
  (`transformlist=[deformable, affine, initial]`). Initialisations act on the transform, not the arrays. [§1]
* **Preprocessing.** Foreground 2–98 % percentile normalisation to [0, 1] for every optimisation input; NLM
  denoising (`antstorch.denoise_image`) before normalisation for 3-D MRI; N4 is optional and off by default in
  landmark preprocessing. Verify that a preprocessing step actually ran (log line), do not assume. [§2]
* **Affine first.** Every deformable pipeline is seeded by `syntx.robust_affine` (`mode='auto'` = the native
  PyTorch Mattes-MI solver since 2026-09-15, ANTs C++ as fallback); cache and share the affine across the
  methods being compared so only the deformable stage differs, and key the cache by affine backend so a
  backend change cannot silently reuse stale transforms. [§16, `AFFINE_GUIDE`]
* **Backend parity.** JAX, PyTorch and C++ are compute engines, not algorithm variants; any clamp, step bound
  or smoothing added in one must be added in all. [§9]
* **Smoothing units.** `flow_sigma`/`total_sigma` are standard deviations (not variances); smoothing is
  isotropic in voxel units at every pyramid level; ITK `SetVariance(v)` ≡ syntx `sigma = sqrt(v)` — settled,
  do not re-derive. [§6, §10]
* **Pyramid geometry.** A pooled voxel `i` at factor `L` sits at full-resolution index `L·i + (L−1)/2`, and a
  pooled tensor is addressed in its own index space normalised by its own shape. Getting this wrong biases
  every coarse level by `(L−1)/2` voxels and is invisible on small phantoms. [§2 Pyramid]
* **Coordinate domains.** Normalise lookup coordinates with the metadata of the tensor being sampled; map
  physical origin/spacing/direction explicitly, never assume normalised grids align. [§6]
* **Determinism.** Seed everything; make sample sets explicit and fixed; use deterministic accumulations
  (see §4 below). Same device, same inputs → bitwise identical parameters is the standard, and it is tested. [§16]
* **Landmark role in multi-start pools.** Feature landmarks (e.g. SIFT3D + RANSAC) provide coarse initial
  basins for hard pairs, but cortical self-similarity can cause spurious inlier counts that deceive RANSAC.
  Never unconditionally force landmark transforms to displace intensity-based multi-start candidates;
  always let global intensity metrics (e.g. coarse-level Mattes-MI) score and select candidates objectively.

## 3. Metrics and objectives

* **Defaults.** `similarity_metric='cc2'` and `regularizer='sobolev'` (α = 1.5) for SyN; Mattes MI for
  multi-modal and for affine initialisation. Deep features only in verified 3-D LNCC modes. [§2]
* **LNCC variance floor** 1e-6 in every implementation; autograd through LNCC beats the centre-of-window
  approximation. [§2]
* **Mattes MI.** Boundary-padded partition of unity (pad = 2 bins), float32 accumulation, fixed histogram
  bounds when comparing or optimising transforms, and the *whole fixed domain* as the sample set for affine
  optimisation. Foreground-only or union masks are for *reporting* similarity of aligned images: as an
  optimisation objective they let the moving brain spill into background unpenalised. [§2 Mattes]
* **Objective ≠ evaluation metric.** Lower loss does not imply better Dice; when they disagree, suspect the
  objective definition (masking, bounds, sampling) before the optimiser. [`AFFINE_GUIDE`]
* **Guidance channels.** Multi-channel guidance (sulcal curvature, landmark clusters) uses soft membership
  maps scored with soft Dice, one metric per channel, modest weight (≈0.2); heavy weights over-constrain. [§2]

## 4. Hardware rules (Apple MPS in particular)

* **Never `F.pad` a 5-D tensor on MPS** with `H·W ≥ 65536`; use slicing + `cat` (`blob._shift_pad`). [§23]
* **Never a single matmul with a huge inner dimension** (≳1e5, e.g. Parzen weights `wᵀw`): non-deterministic
  and wrong for concentrated data. Use blocked `bmm` with fixed-order summation (`_parzen_joint_histogram`). [§2]
* **Make strided views contiguous** before feeding MPS kernels.
* **One MPS process at a time**; run MPS jobs before CPU-heavy jobs; clear caches between pairs; isolate
  long benchmarks in subprocesses. Timings measured under contention are not timings. [§15, §17]
* **`robust_affine` pins ITK to one thread** for determinism; remember this when timing ANTs afterwards in
  the same process.

## 5. Physical space, transforms and display

* **ITK LPS** physical coordinates; arrays and tensors keep ANTs XYZ layout (`[1,1,nx,ny,nz]`);
  `grid_sample` grids are ordered `(iz, iy, ix)` for that layout. All voxel↔mm math goes through
  `syntx.landmarks.spatial`; no inline `x = w * spacing`. [§23]
* **Never reorient or resample arrays for display**; derive orientation from the direction matrix and pass the
  same view spec to slice extraction and point projection. Radiological convention by default. [§22, §23]
* **ANTs affine centre.** A `.mat` encodes `y = A(x − c) + c + t`; convert translations with the centre, and
  `whichtoinvert` must have one flag per transform. [§6, §22]
* **Scales are millimetres** (detector sigmas, MIND offsets); descriptors are computed in physical axes so
  storage frame (LAS vs RPS, anisotropy) does not matter. [§23]
* **Large relative rotations** are handled by a PCA-seeded iterative global rotation search, not by
  per-keypoint frame voting, which is unreliable on real cortex. [§23]

## 6. Evaluation and reporting

* **Labels:** nearest-neighbour interpolation, bidirectional fixed/moving Dice, background label excluded,
  correct inversion flags; report Sørensen–Dice only. [§4, §21]
* **Never report FRE as TRE**; hold-out landmarks only.
* **Every report** uses the `syntx.viz` standard figures, the full deformation-metric suite (folding %, min
  Jacobian, harmonic/bending energy, inverse consistency), and states device and contention conditions. [§3]
* **Accuracy thresholds.** A ≥ 0.01 mean Dice drop on cortical labels is a regression; equal-or-better than
  the ANTs C++ reference on a held-out pair set is the bar for changing a default. [§2]
* **Reproducibility is a reported metric**: repeated-run parameter spread, same process and fresh process.

## 7. Where the evidence lives

`docs/PROJECT_FINDINGS_DETAILED.md` (all historical measurements and parameter provenance),
`docs/AFFINE_GUIDE.md`, `docs/LANDMARKS_GUIDE.md`, `docs/SESSION_2026-09-14_LANDMARKS_AFFINE.md`,
`docs/provenance/best_parameters.json`, `results/affine_baseline/`, `docs/reports/`.
