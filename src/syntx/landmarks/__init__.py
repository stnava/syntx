"""
syntx.landmarks — landmark detection, description and matching for registration initialisation.

The detectors respond to local image structure (blobs, gradient-orientation patterns,
self-similarity) rather than to absolute intensities, after rescaling to [0, 1]. Keypoints
are returned in physical mm (ANTs / LPS world coordinates), so images stored in different
native frames give comparable coordinates. Cross-modality robustness varies by descriptor
(MIND-style self-similarity is the most contrast-independent; SIFT is insensitive to gain /
offset only).

Preprocessing
-------------
By default (``preprocess=True``) the blob / SIFT detectors run ``preprocess_for_landmarks``
on ANTsImage input before feature extraction:

    MRI  →  (N4, off by default)  →  NLM denoising (Rician, shrink=2, p=1, r=1)
         →  ``normalize_image(method='auto')`` → [0, 1]
    CT   →  optional fixed HU window, else ``normalize_image(method='auto')`` → [0, 1]
            (no N4, no denoising)

CT is auto-detected from negative voxel values (``is_ct_image``). Pass ``preprocess=False``
if you have already preprocessed the image. ``compute_mind`` does no preprocessing.

Public API
----------
Preprocessing
    preprocess_for_landmarks : (N4) + NLM + normalisation (MRI) or normalisation only (CT)
    is_ct_image              : heuristic CT / MRI classifier from voxel values

Spatial framework (``spatial``)
    voxel <-> physical conversion, tensor layout, orthogonal display slices, overlay
    projection, ``safe_whichtoinvert``

Blob detection (LoG / DoG)
    detect_blobs_log   : 3D Laplacian-of-Gaussian scale space  → [N, 4]
    detect_blobs_dog   : 3D Difference-of-Gaussians            → [N, 4]

2D SIFT (multi-slice + 3D back-projection)
    detect_sift2d      : OpenCV SIFT on slices along each array axis → ([N,4], [N,128][, planes])

3D SIFT (full volumetric)
    detect_sift3d      : DoG keypoints + 3D gradient histogram  → ([N,4], [N,512])
    sift3d_keypoints / sift3d_descriptors : the two stages separately

MIND-style self-similarity descriptors
    compute_mind              : dense descriptor volume          → [1, C, nx, ny, nz]
    extract_mind_at_points    : descriptors at sparse mm coords  → [N, C]

Matching & RANSAC
    match_landmarks   : L2 / cosine NN + Lowe ratio test        → [K, 2] index pairs
    knn_matches       : k nearest neighbours, no ratio test      → [N*k, 2]
    ransac_filter     : RANSAC affine / rigid verification       → (filtered_matches, M44)
    compute_tre       : mean distance between corresponding points (mm)

Rotation search (``orient``)
    match_sift3d_with_rotation_search, refine_rotation_iteratively,
    estimate_rotation_from_frames, rotation_grid, pca_rotation_candidates

Optimal transport (``optimal_transport``)
    sinkhorn_matching, weighted_procrustes, sampled_optimal_transport_affine,
    score_rotation_candidates_sampled

Warp fitting
    Displacement field fitting from matched landmarks is handled by
    ``syntx.scattered``, which provides:
        - ``fit_bspline_landmark_warp``  : B-spline displacement field (ANTsTorch backend)
        - ``syn_scattered``              : diffeomorphic SyN on point clouds
    Example (``fit_bspline_landmark_warp`` expects the grid size in tensor order and
    ``domain_bounds`` in the landmarks' units; its default (-1, 1) does not suit mm)::

        from syntx.landmarks import detect_sift2d, match_landmarks, ransac_filter
        from syntx.scattered import fit_bspline_landmark_warp

        c_f, d_f = detect_sift2d(fixed)    # preprocess=True by default
        c_m, d_m = detect_sift2d(moving)
        matches = match_landmarks(c_f, c_m, d_f, d_m)
        matches, _ = ransac_filter(c_f, c_m, matches)
        u = fit_bspline_landmark_warp(
            fixed_landmarks=c_f[matches[:, 0], :3],
            moving_landmarks=c_m[matches[:, 1], :3],
            grid_shape=fixed.shape[::-1],
            domain_bounds='auto',
        )
"""

from .preprocess import preprocess_for_landmarks, is_ct_image
from .blob import detect_blobs_log, detect_blobs_dog
from .sift2d import detect_sift2d
from .sift3d import detect_sift3d, sift3d_keypoints, sift3d_descriptors
from .mind import compute_mind, extract_mind_at_points
from .matcher import match_landmarks, ransac_filter, compute_tre, knn_matches
from .orient import (estimate_rotation_from_frames, match_sift3d_with_rotation_search, rotation_grid,
                     pca_rotation_candidates, refine_rotation_iteratively)
from .optimal_transport import (
    sinkhorn_matching,
    weighted_procrustes,
    sampled_optimal_transport_affine,
    score_rotation_candidates_sampled,
)
from .spatial import (
    get_image_affine,
    vox_to_physical,
    vox_zyx_to_physical,
    physical_to_vox,
    physical_offset_to_voxel,
    voxel_gradient_to_physical,
    image_to_tensor,
    sample_tensor_at_physical,
    axis_orientation_code,
    anatomical_axis_labels,
    format_axis_xlabel,
    ortho_view_spec,
    slice_from_spec,
    extract_ortho_slices,
    project_to_slice,
    safe_whichtoinvert,
)

__all__ = [
    # preprocessing
    "preprocess_for_landmarks",
    "is_ct_image",
    # spatial framework
    "get_image_affine",
    "vox_to_physical",
    "vox_zyx_to_physical",
    "physical_to_vox",
    "physical_offset_to_voxel",
    "voxel_gradient_to_physical",
    "image_to_tensor",
    "sample_tensor_at_physical",
    "axis_orientation_code",
    "anatomical_axis_labels",
    "format_axis_xlabel",
    "ortho_view_spec",
    "slice_from_spec",
    "extract_ortho_slices",
    "project_to_slice",
    "safe_whichtoinvert",
    # blob
    "detect_blobs_log",
    "detect_blobs_dog",
    # sift
    "detect_sift2d",
    "detect_sift3d",
    "sift3d_keypoints",
    "sift3d_descriptors",
    # mind
    "compute_mind",
    "extract_mind_at_points",
    # matcher
    "match_landmarks",
    "ransac_filter",
    "compute_tre",
    "knn_matches",
    # orientation search
    "estimate_rotation_from_frames",
    "match_sift3d_with_rotation_search",
    "rotation_grid",
    "pca_rotation_candidates",
    "refine_rotation_iteratively",
    # optimal transport & sampled correlation
    "sinkhorn_matching",
    "weighted_procrustes",
    "sampled_optimal_transport_affine",
    "score_rotation_candidates_sampled",
]

