# Pending fixes for the next release notes

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
