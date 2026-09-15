# `syntx.landmarks` — physical-space landmark detection, description and matching

Modality- and anatomy-independent 3D landmarks in **physical LPS millimetres**, with
display and matching tools that never reorient or resample the input arrays.
Validated on siq procedural-brain phantoms stored in different native frames and on the
Mindboggle `mbhard` pair (see `docs/reports/mbhard_landmarks_spatial_report.html`).

## Quick start

```python
import syntx
from syntx.landmarks import (preprocess_for_landmarks, detect_sift3d,
                             match_landmarks, ransac_filter,
                             match_sift3d_with_rotation_search)
from syntx.landmarks import spatial as S

ds = syntx.benchmark_data("mbhard")
fi = preprocess_for_landmarks(ds["fixed"])     # NLM (ANTsTorch, MPS) + 2–98 % normalisation; N4 off by default
mi = preprocess_for_landmarks(ds["moving"])

# heads roughly aligned (< ~30° relative rotation): plain axis-aligned matching
cf, df = detect_sift3d(fi, preprocess=False, max_keypoints=2000, n_scales=10, sigma_min=1.0, sigma_max=8.0)
cm, dm = detect_sift3d(mi, preprocess=False, max_keypoints=2000, n_scales=10, sigma_min=1.0, sigma_max=8.0)
m = match_landmarks(cf, cm, df, dm, ratio_thresh=0.95, mutual=True)
inliers, M = ransac_filter(cf, cm, m, model="affine", inlier_thresh_mm=8.0)   # M: fixed mm -> moving mm

# unknown / large relative rotation: PCA-seeded iterative global rotation search
res = match_sift3d_with_rotation_search(fi, mi, max_keypoints=2000, n_scales=10, sigma_min=1.0, sigma_max=8.0)
res["inliers"], res["affine"], res["rotation_deg"], res["candidates"]
```

Display (radiological by default, `convention="neurological"` for Left-on-Left):

```python
sl = S.extract_ortho_slices(fi)                           # 'ax', 'cor', 'sag' arrays + 'views' specs
u, v, in_slab = S.project_to_slice(fi, cf[:, :3], view=sl["views"]["ax"], slab_half_mm=4.0)
plt.imshow(sl["ax"], cmap="gray", origin="lower"); plt.scatter(u[in_slab], v[in_slab])
```

## Conventions (see docs/PROJECT_FINDINGS_DETAILED.md §23 and GEMINI.md §5 for the invariants)

* Physical space is ITK LPS (+x Left, +y Posterior, +z Superior). Coordinates are `[N, 4]`
  `(x, y, z, sigma_mm)`.
* Arrays and tensors keep the ANTs XYZ layout: `image_to_tensor` gives `[1, 1, nx, ny, nz]`;
  `argwhere` indices go straight into `vox_to_physical`. `vox_zyx_to_physical` is only for
  tensors explicitly transposed to ZYX.
* Scales (`sigma_min`, `sigma_max`, MIND `offset_distance`) are mm; detectors derive per-axis
  voxel sigmas / shifts from spacing and direction, so anisotropic images are handled.
* Descriptors are computed in physical axes: the same anatomy stored as LAS vs RPS arrays gives
  bit-identical keypoints and descriptors.
* Border handling uses slicing (`blob._shift_pad`), never `F.pad`, because MPS corrupts `F.pad`
  on 5-D tensors with 256×256 slices.

## Detectors and descriptors

| function | output | notes |
|---|---|---|
| `detect_blobs_dog` / `detect_blobs_log` | `[N, 4]` | scale-space extrema in mm, foreground-masked, response-ranked NMS |
| `detect_sift3d` | `[N, 4]`, `[N, 512]` | DoG extrema + 4³ cells × 8 directions gradient descriptor sampled on a mm lattice |
| `detect_sift3d(..., rotation_invariant=True)` | same | descriptor in a local structure-tensor frame; exact on phantoms, weaker on real brains |
| `sift3d_keypoints` / `sift3d_descriptors` | state dict / descriptors | two-stage API to rebuild descriptors in other frames cheaply |
| `compute_mind` / `extract_mind_at_points` | `[1, 12, nx, ny, nz]` / `[N, 12]` | MIND-SSC with offsets defined in mm along LPS axes |

## Matching

* `match_landmarks(..., ratio_thresh, mutual=True)`: batched GPU nearest neighbour + Lowe ratio,
  optional mutual check. `knn_matches` returns high-recall k-NN candidates.
* `ransac_filter(model="affine" | "rigid")`: returns inlier pairs and the fixed→moving 4×4.
* `match_sift3d_with_rotation_search`: the axis-aligned descriptor tolerates ~30° of relative
  rotation. Beyond that, seed a global rotation from the principal axes of the two landmark
  clouds (identity + 4 sign combinations), refine iteratively (descriptors rebuilt in the current
  frame → match → rigid RANSAC → rotation; coarse-to-fine over keypoint strength), keep the
  converged hypothesis with most inliers. On `mbhard` rotated ±30° this recovers the 60° relative
  yaw to within 1° and matches as well as the unrotated pair.

## Validation summary (2026-09-14)

| test | result |
|---|---|
| phantom, LPS / RAS / permuted frames | keypoints and descriptors identical; MIND identical |
| phantom, anisotropic resampling | descriptor cosine 0.995 at same points |
| phantom, known rigid 8°+5° | TRE 0.38 mm |
| phantom, 60° about a tilted axis, rotation search | rotation error < 1°, TRE 0.07 mm |
| mbhard native pair | 182 affine inliers, 75 % whole-brain label agreement (chance 7 %) |
| mbhard fixed +30° / moving −30° | 182 inliers, 70 % agreement; recovered relative yaw 60.8° |
| Ex2 affine vs prediction from Ex1 and known rotations | 1.4 mm mean discrepancy |

Scripts: `scripts/landmarks_mbhard_report.py`, tests: `tests/test_landmarks_*.py`.

## Benchmark: do landmarks help affine initialisation and Sobolev SyN? (mbhard, 2026-09-14)

Script `scripts/benchmark_mbhard_landmark_guidance.py`; report `docs/reports/mbhard_landmark_guidance_report.html`;
raw numbers `results/mbhard_landmark_guidance.json`. 182 RANSAC-inlier landmark pairs (75 % share a whole-brain label).

**Part 1 — affine** (symmetric DKT31 Dice; similarity of affinely resampled moving T1 vs fixed T1)

| affine | Dice sym | ncc (foreground) | lncc | Mattes MI (loss) | time |
|---|---|---|---|---|---|
| A standard `robust_affine(mode='auto')` | 0.3244 | 0.535 | −0.0139 | −0.247 | 59 s |
| B least-squares on landmark inliers | **0.3259** | **0.551** | **−0.0145** | −0.236 | 16.5 s (incl. detection) |
| C = B refined by intensity (`robust_affine(initial_transform=B)`) | 0.3237 | 0.526 | −0.0145 | −0.246 | 86 s |

The landmark affine is as good as or better than the intensity affine on Dice and correlation metrics at ~¼ of the
cost; MI alone slightly prefers A. Refining B with the ANTs affine stage made it *worse* (C), which points at the
affine optimiser rather than the initialisation — see the affine review/plan. A and B differ by 2.0° / 9.6 mm at the COM.

**Part 2 — Sobolev SyN** (from affine A unless stated; levels [4,2,1], iters [100,100,20], α = 1.5, grad step 0.25)

| arm | Dice sym | Δ vs baseline | folding % |
|---|---|---|---|
| baseline, single-channel cc2 | 0.6025 | — | 0.008 |
| guided K=8 clusters, w=0.2 | **0.6121** | **+0.0096** | 0.020 |
| guided K=16, w=0.2 | 0.6107 | +0.0082 | 0.014 |
| guided K=8, w=0.5 | 0.6057 | +0.0032 | 0.003 |
| oracle: 136 label-agreeing inliers only, K=8, w=0.2 | 0.6083 | +0.0058 | 0.010 |
| landmark affine C + guided K=8, w=0.2 | 0.6069 | +0.0044 | 0.0002 |

Guidance channels: k-means clusters of the inlier landmarks; per cluster a soft membership map (3 mm Gaussian blobs at the
fixed / matched moving positions, softmax-like normalisation across clusters); `similarity_metric=['cc2'] + ['dice']*K`,
weights `[1-w] + [w/K]*K` (same mechanism as sulcal guidance).  Findings: a modest landmark weight (0.2) with 8 clusters
gives ≈ +1 % Dice, close to the "massive" threshold in GEMINI.md; a heavy weight (0.5) over-constrains; the oracle subset
is *worse* than all inliers, so quantity of roughly-correct correspondences matters more than purity at this weight.
Folding stays below 0.02 % in every arm.  Single pair — a cohort run is needed before changing defaults.
