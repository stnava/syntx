"""
syntx.landmarks — Modality / anatomy-independent landmark detection & matching.
===============================================================================

All methods respond only to **local geometric structure** (gradient extrema, scale-space
blobs, self-similarity patterns).  After foreground 2nd–98th percentile normalization they
work identically on brain MRI, abdominal CT, cardiac MRI, lung CT, prostate MRI, etc.

Preprocessing
-------------
By default (``preprocess=True``) every detector applies the same pipeline used in the
syntx registration benchmark *before* feature extraction:

    MRI  →  (N4 bias correction)  →  NLM denoising (Rician, shrink=2, r=1)
         →  foreground 2nd–98th pct normalization → [0, 1]
    CT   →  foreground 2nd–98th pct normalization → [0, 1]   (no N4, no denoising)

CT is auto-detected from the presence of negative Hounsfield-unit voxels.
Pass ``preprocess=False`` if you have already preprocessed the image.

Public API
----------
Preprocessing
    preprocess_for_landmarks : apply benchmark preprocessing (N4+NLM+norm or norm-only)
    is_ct_image              : heuristic CT/MRI classifier from voxel values

Blob detection (LoG / DoG)
    detect_blobs_log   : 3D Laplacian-of-Gaussian scale-space  → [N, 4]
    detect_blobs_dog   : 3D Difference-of-Gaussians (faster)   → [N, 4]

2D SIFT (multi-slice + 3D back-projection)
    detect_sift2d      : SIFT on axial/coronal/sagittal slices  → ([N,4], [N,128])

3D SIFT (full volumetric)
    detect_sift3d      : DoG extrema + 3D gradient histogram    → ([N,4], [N,512])

MIND-SSC descriptors (modality-independent by construction)
    compute_mind              : dense descriptor volume          → [1, C, D, H, W]
    extract_mind_at_points    : descriptors at sparse mm coords  → [N, C]

Matching & RANSAC
    match_landmarks   : L2 NN + Lowe ratio test                 → [K, 2] index pairs
    ransac_filter     : RANSAC affine/rigid verification         → (filtered_matches, M44)
    compute_tre       : Target Registration Error in mm (hold-out points only)

Warp fitting
    Displacement field fitting from matched landmarks is handled by
    ``syntx.scattered``, which provides:
        - ``fit_bspline_landmark_warp``  : C² B-spline warp (ANTsTorch backend)
        - ``syn_scattered``              : diffeomorphic SyN on point clouds
    Example::

        from syntx.landmarks import detect_sift2d, match_landmarks, ransac_filter
        from syntx.scattered import fit_bspline_landmark_warp

        c_f, d_f = detect_sift2d(fixed)    # preprocess=True by default
        c_m, d_m = detect_sift2d(moving)
        matches = match_landmarks(c_f, c_m, d_f, d_m)
        matches, _ = ransac_filter(c_f, c_m, matches)
        phi_init = fit_bspline_landmark_warp(
            fixed_landmarks=c_f[matches[:, 0], :3],
            moving_landmarks=c_m[matches[:, 1], :3],
            grid_shape=fixed.shape,
        )
"""

from .preprocess import preprocess_for_landmarks, is_ct_image
from .blob import detect_blobs_log, detect_blobs_dog
from .sift2d import detect_sift2d
from .sift3d import detect_sift3d
from .mind import compute_mind, extract_mind_at_points
from .matcher import match_landmarks, ransac_filter, compute_tre

__all__ = [
    # preprocessing
    "preprocess_for_landmarks",
    "is_ct_image",
    # blob
    "detect_blobs_log",
    "detect_blobs_dog",
    # sift
    "detect_sift2d",
    "detect_sift3d",
    # mind
    "compute_mind",
    "extract_mind_at_points",
    # matcher
    "match_landmarks",
    "ransac_filter",
    "compute_tre",
]
