# syntx 6.3.0

New: `robust_affine(..., setting='intermodality_slab')` (opt-in), a thin-slab capture strategy in `syntx.slab_capture`; contents of the former `RELEASE_NOTES_pending_fixes.md` (landmarks and tests, 6.2.2 base) below. The default behaviour of `robust_affine` is unchanged.
Design: `docs/AFFINE_GUIDE.md`, last section. Harness: `benchmarks/slab_setting_eval.py` (`real`, `synthetic`, `stress`). Device mps throughout, seed 42, error = maximum probe-point displacement against the ANTs rigid reference (mm); synthetic = mean TRE (mm).

## Ten real ds001242 pairs (rebased code, re-measured after the rebase onto 6.2.2 + landmark fixes; identical to the pre-rebase numbers)
| sub | default | 32-rotation slab | z-shift (negative result) |
|---|---|---|---|
| 01 | 0.384 | 0.384 | 0.384 |
| 02 | 0.055 | 0.055 | 0.055 |
| 03 | 0.142 | 0.143 | 0.142 |
| 04 | 0.307 | 0.333 | 0.307 |
| 05 | 0.188 | 0.188 | 0.188 |
| 06 | 0.339 | 0.187 | 0.339 |
| 07 | 1.238 | 1.344 | 1.238 |
| 08 | 0.110 | 0.107 | 0.110 |
| 09 | 0.290 | 0.290 | 0.290 |
| 10 | 0.136 | 0.143 | 0.136 |
All ten under 2 mm with the setting (max 1.344, sub-07, 0.1 mm above the default). The slab pose is the default solver's own in 4 of 10 pairs, a rotation start in 6. The pure translation-first strategy (without the default solver in the pool) failed sub-01 at 7.58 mm (translation stage in a wrong basin at z = 14.9 mm; native MI of the 7.6 mm pose 0.642, of the default pose 0.685): this is why the default result joins the pool.

## Synthetic thin slab, pose A (TRE mm)
| ratio | default | 32-rotation | z-shift |
|---|---|---|---|
| 1 | 0.09 | 0.05 | |
| 2 | 0.03 | 0.03 | |
| 4 | 0.70 | 0.05 | 0.70 |
| 6 | 4.42 | 0.13 | 2.75 |
| 8 | 0.92 | 0.31 | 0.92 |
| 10 | 1.07 | 0.23 | 1.07 |
The 6:1 case is solved with the rotation setting (`test_thin_slab_6to1_pose_a_is_solved_by_the_intermodality_slab_setting`); the default's strict xfail is unchanged. z-shift only improves it to 2.75 mm.

## Time (sub-07, uncontended, mps)
default 14.1 s; z-shift 54.7 s (3.9x); 32-rotation 260 s (18x). Synthetic: default 4 to 7 s, rotation 55 to 89 s (measured with 3 to 8 processes on the GPU, indicative). Per-start cost is about 5.3 s.

## Capture stress (INCOMPLETE: the sweep was stopped by the owner)
The centre of mass of the T1 (the start of every default candidate) is displaced by cropping the T1 at one end, so the truth is unchanged (an origin shift would move truth and centre of mass together and change nothing). 33 default / rotation pairs of records finished.
- In-plane (x, y; 5, 10, 15 mm; sub-01 and sub-02 and part of sub-03..05): 22 of 22 under 2 mm for both settings, medians 0.28 / 0.28 (x) and 0.26 / 0.25 (y); sub-01 at 15 mm x: default 0.29, slab 1.05.
- z (11 cases): 2 of 11 under 2 mm for both settings (default median 19.9 mm, rotation 16.4 mm).
| z case | default | 32-rotation | z-shift |
|---|---|---|---|
| sub-01 5 / 10 / 15 mm | 18.9 / 22.8 / 29.2 | 16.4 / 22.6 / 28.3 | 11.5 / 22.8 / 29.2 |
| sub-02 5 / 10 / 15 mm | 0.44 / 12.5 / 18.1 | 0.46 / 8.1 / - | 0.62 / 8.8 / 21.9 |
| sub-03 10 mm | 27.1 | 11.5 | 13.1 |
| sub-04 10 mm | 19.9 | 19.3 | 19.9 |
| sub-04 5 / 15 mm, sub-05 5 / 10 mm | 6.5 / 25.2, 21.6 / 28.0 | 9.1 / 25.9, 21.6 / - | |
Neither setting captures a z displacement of the T1 centre of mass of 5 to 15 mm in general: this remains an open limitation of the metric (capture about one coarse voxel); the setting helps partially on sub-02 and sub-03 at 10 mm and is worse on sub-04 at 5 mm.

## Negative results
- Pure translation-first strategy: fails real sub-01 (7.58 mm).
- z-shift strategy (default solver plus 8 shifted starts, +-3.75 ... 15 mm along the thick axis, ranked by native MI): on all ten real pairs the default result wins (identical to default); 6:1 only 4.42 -> 2.75 mm; stress cases essentially unchanged. Kept as `strategy='z_shift'`, not used.
- Native-MI ranking cannot rescue a pool whose members all share a wrong basin.

## antsxfunctional
Keep the default. The setting costs 18x, gives no gain on most real pairs and does not cure a poor z start; offer it as an opt-in only.

# Folded in from RELEASE_NOTES_pending_fixes.md

## Rotation-invariant SIFT3D frame (src/syntx/landmarks/sift3d.py)
The local frame of `detect_sift3d(rotation_invariant=True)` was estimated over a cube aligned with
the scanner axes and a window twice the descriptor width. A cube is not rotation invariant, and the
wide window mixes in anatomy that does not rotate with the keypoint neighbourhood. On a 60 degree
rotation of a siq phantom only ~2% of true correspondences had a frame within 10 degrees.
The frame window is now a Gaussian inside the inscribed ball and `frame_scale` defaults to 0.75
(was 2.0). Rigid-RANSAC inliers (400 keypoints): 10 -> 75; (800 keypoints): 29 -> ~140.
`frame_scale` sweep (400 kp): 0.5 -> 89, 0.75 -> 75, 1.0 -> 65, 1.5 -> 43, 2.0 -> 15.
Open: axes 2 and 3 remain ambiguous at near-isotropic (blob) keypoints (about half are aligned).

## Tests that were measuring the wrong thing (no threshold loosened)
- `test_restrict_transformation.py`: `syn(type_of_transform="SyNOnly")` without `initial_transform`
  runs the foreground centre-of-mass translation init; it absorbed the pure-shift pair so the
  deformation field was ~0.016. Tests now pass `initial_transform="identity"` (deformable only);
  Y field ~0.5 against the unchanged 0.05 threshold. The earlier "default step oscillates"
  explanation was wrong and the pinned step/sigma were removed.
- `test_dewarp.py::test_single_pe_dewarp_to_anatomical`: a PE translation is absorbed by the rigid
  anatomy->reference stage and the output lives in that frame, so comparing with the unshifted
  phantom could not exceed 0.82. The test now uses a non-rigid PE displacement and compares with
  `anatomy_rigid`; raw 0.90 -> dewarped 0.95-0.96.

## Open
- On already-aligned, noise-free blocky pairs SyNOnly with cc2 still deforms (corr 0.9987 -> 0.944,
  max field 3.7 mm) because local CC rewards edge texture in flat regions; real-data check pending.
- Docs say SyNOnly has no affine; it does run a CoM translation init unless `initial_transform='identity'`.
