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
