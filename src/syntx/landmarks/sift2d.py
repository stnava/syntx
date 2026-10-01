"""
2-D SIFT (OpenCV) on evenly spaced slices of a 3-D volume, back-projected to physical space.

Slices are taken at fixed array indices along each array axis (iz, iy, ix; called axial,
coronal and sagittal, which is only anatomically right for a near-RAS/LPS-aligned array).
Each slice is rescaled to uint8 by its foreground 2nd-98th percentiles before SIFT. Keypoint
pixel coordinates plus the slice index give an array index, mapped to physical mm with the full
image affine. SIFT descriptors are gradient-orientation histograms, so they are insensitive to
intensity gain / offset but not to arbitrary (e.g. contrast-inverting) intensity mappings, and
they describe the in-slice pattern only: descriptors from different slice orientations are not
comparable.

Functions
---------
detect_sift2d : keypoint coordinates + 128-D SIFT descriptors
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _normalize_slice_uint8(sl: np.ndarray) -> np.ndarray:
    """2-D float slice -> uint8: clip to the 2nd-98th percentiles of values > 1e-4, scale to
    0..255. If there is no usable range the slice is cast as ``sl * 255`` (assumes [0, 1])."""
    fg = sl[sl > 1e-4]
    if fg.size > 0:
        lo, hi = np.percentile(fg, (2.0, 98.0))
        if hi > lo + 1e-4:
            sl = np.clip((sl - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return (sl * 255).astype(np.uint8)


def detect_sift2d(
    image,
    n_slices_per_axis: int = 8,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0,
    sigma: float = 1.6,
    min_distance_mm: float = 3.0,
    max_keypoints: int = 512,
    preprocess: bool = True,
    use_n4: bool = False,
    use_denoise: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 2D SIFT features from slices along each array axis and back-project to 3D
    physical space.

    Slices are at ``linspace(0, n - 1, n_slices_per_axis)`` (integer) along iz, iy and ix,
    including the first and last slice. All keypoints of all slices are pooled, then thinned
    by a greedy 3-D NMS that visits them in decreasing keypoint size (not response) and drops
    any within ``min_distance_mm`` of a kept one.

    Parameters
    ----------
    image : ants.ANTsImage | np.ndarray
        3D volume (anything else raises ValueError). ndarray input has identity geometry
        (spacing 1) and is not preprocessed.
    n_slices_per_axis : int, default 8
        Number of slices per array axis.
    contrast_threshold : float, default 0.04
        Passed to ``cv2.SIFT_create`` (``nfeatures=0``, ``nOctaveLayers=3``).
    edge_threshold : float, default 10.0
        Passed to ``cv2.SIFT_create``.
    sigma : float, default 1.6
        Passed to ``cv2.SIFT_create`` (pixels).
    min_distance_mm : float, default 3.0
        Greedy 3-D NMS radius (mm).
    max_keypoints : int, default 512
        Upper bound on returned keypoints.
    preprocess : bool, default True
        Run ``preprocess_for_landmarks`` first (ANTsImage input only).
    use_n4 : bool, default False
        N4 bias correction (MRI only, used only when preprocessing).
    use_denoise : bool, default True
        NLM denoising (MRI only, used only when preprocessing).

    Returns
    -------
    coords : np.ndarray, shape [N, 4], float32
        Physical coordinates (x_mm, y_mm, z_mm, scale_mm); scale_mm is the OpenCV keypoint
        diameter ``kp.size`` times the geometric mean of the two in-slice spacings.
    descriptors : np.ndarray, shape [N, 128], float32
        L2-normalised SIFT descriptors, row-aligned with ``coords``. The slice orientation
        of each keypoint is not returned.

    Raises
    ------
    ImportError
        If OpenCV (``cv2``) is not installed.
    """
    if preprocess and hasattr(image, "numpy") and hasattr(image, "new_image_like"):
        from .preprocess import preprocess_for_landmarks
        image = preprocess_for_landmarks(image, use_n4=use_n4, use_denoise=use_denoise)

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python-headless is required: pip install opencv-python-headless") from exc

    from .spatial import get_image_affine, vox_to_physical

    # --- extract array and full affine from image ---------------------------------
    org, sp, D = get_image_affine(image)
    if hasattr(image, "numpy"):
        arr = image.numpy().astype(np.float32)          # [X, Y, Z] = [ix, iy, iz] ANTs layout
    else:
        arr = np.asarray(image, dtype=np.float32)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {arr.shape}")

    DX, DY, DZ = arr.shape   # (n_ix, n_iy, n_iz)

    def _phys(ix: float, iy: float, iz: float) -> tuple[float, float, float]:
        """Physical (x,y,z) mm for continuous voxel index (ix, iy, iz)."""
        p = vox_to_physical((org, sp, D), np.array([[ix, iy, iz]]))[0]
        return float(p[0]), float(p[1]), float(p[2])

    sift = cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=3,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma,
    )

    all_coords: list[np.ndarray] = []
    all_descs:  list[np.ndarray] = []

    def _run_slice(sl2d: np.ndarray,
                   fixed_ix: Optional[float],
                   fixed_iy: Optional[float],
                   fixed_iz: Optional[float],
                   u_maps_ix: bool,   # True = OpenCV col (u) → ix axis
                   v_maps_iy: bool,   # True = OpenCV row (v) → iy axis
                   # if not u_maps_ix then u→iy; if not v_maps_iy then v→iz
                   ):
        """Run SIFT on one 2-D slice and append physical keypoints / descriptors to the
        enclosing lists. The orientation is chosen by which ``fixed_*`` index is set;
        ``u_maps_ix`` / ``v_maps_iy`` are not used."""
        img8 = _normalize_slice_uint8(sl2d)
        kps, descs = sift.detectAndCompute(img8, None)
        if kps is None or len(kps) == 0 or descs is None:
            return

        for kp, desc in zip(kps, descs):
            u_px, v_px = kp.pt   # OpenCV: (col=u, row=v)
            scale_px   = kp.size

            # Reconstruct (ix, iy, iz) from the 2D pixel and the fixed axis index
            if fixed_iz is not None:
                # Axial slice (constant iz): arr[:,:,iz] → col=u=ix, row=v=iy
                ix, iy, iz = u_px, v_px, fixed_iz
                scale_mm = scale_px * float(np.sqrt(sp[0] * sp[1]))
            elif fixed_iy is not None:
                # Coronal slice (constant iy): arr[:,iy,:] → col=u=ix, row=v=iz
                ix, iy, iz = u_px, fixed_iy, v_px
                scale_mm = scale_px * float(np.sqrt(sp[0] * sp[2]))
            else:
                # Sagittal slice (constant ix): arr[ix,:,:] → col=u=iy, row=v=iz
                ix, iy, iz = fixed_ix, u_px, v_px
                scale_mm = scale_px * float(np.sqrt(sp[1] * sp[2]))

            xm, ym, zm = _phys(ix, iy, iz)
            all_coords.append([xm, ym, zm, scale_mm])
            all_descs.append(desc.astype(np.float32))

    # ANTs array layout: arr[ix, iy, iz]
    z_indices = np.linspace(0, DZ - 1, n_slices_per_axis, dtype=int)
    y_indices  = np.linspace(0, DY - 1, n_slices_per_axis, dtype=int)
    x_indices  = np.linspace(0, DX - 1, n_slices_per_axis, dtype=int)

    # Axial slices: fix iz, sample in (ix→col, iy→row)
    for iz in z_indices:
        sl = arr[:, :, iz].T    # shape [DY, DX] → row=iy, col=ix for OpenCV
        _run_slice(sl, None, None, float(iz), u_maps_ix=True, v_maps_iy=True)

    # Coronal slices: fix iy, sample in (ix→col, iz→row)
    for iy in y_indices:
        sl = arr[:, iy, :].T    # shape [DZ, DX] → row=iz, col=ix
        _run_slice(sl, None, float(iy), None, u_maps_ix=True, v_maps_iy=False)

    # Sagittal slices: fix ix, sample in (iy→col, iz→row)
    for ix in x_indices:
        sl = arr[ix, :, :].T    # shape [DZ, DY] → row=iz, col=iy
        _run_slice(sl, float(ix), None, None, u_maps_ix=False, v_maps_iy=False)

    if not all_coords:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0, 128), dtype=np.float32)

    coords = np.array(all_coords, dtype=np.float32)   # [N, 4]
    descs  = np.array(all_descs,  dtype=np.float32)   # [N, 128]

    # L2-normalise descriptors
    norms = np.linalg.norm(descs, axis=1, keepdims=True)
    descs = descs / (norms + 1e-8)

    # 3-D greedy NMS in physical space
    order = np.argsort(-coords[:, 3])   # largest scale first
    coords, descs = coords[order], descs[order]
    kept = []
    kept_c: list[np.ndarray] = []
    for i in range(len(coords)):
        if len(kept) >= max_keypoints:
            break
        if kept_c:
            dists = np.linalg.norm(np.stack(kept_c) - coords[i, :3], axis=1)
            if dists.min() < min_distance_mm:
                continue
        kept.append(i)
        kept_c.append(coords[i, :3])

    idx = np.array(kept, dtype=int)
    return coords[idx], descs[idx]

