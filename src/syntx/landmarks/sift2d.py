"""
syntx.landmarks.sift2d — 2D SIFT multi-slice landmark detection
================================================================

Extracts SIFT keypoints from equally-spaced axial, coronal, and sagittal slices
of a 3D volume, then back-projects each (u, v) pixel coordinate to a physical
(x_mm, y_mm, z_mm) coordinate using the image spacing and slice index.

Modality / anatomy independence
--------------------------------
After foreground 2nd–98th percentile normalization (applied internally), SIFT
gradient-histogram descriptors encode local geometric structure.  They are
explicitly invariant to monotone intensity transformations, so the same detector
works on brain MRI, abdominal CT, cardiac MRI, lung CT, etc.

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
    """Foreground-percentile normalize a 2-D float slice to uint8."""
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
    use_n4: bool = True,
    use_denoise: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 2D SIFT features from axial/coronal/sagittal slices and back-project
    to 3D physical space.

    Parameters
    ----------
    image : ants.ANTsImage | np.ndarray
        3D volume.  Preprocessed and normalized internally.
    n_slices_per_axis : int
        Number of equally-spaced slices sampled from each of the three planes.
    contrast_threshold : float
        Passed to ``cv2.SIFT_create``.
    edge_threshold : float
        Passed to ``cv2.SIFT_create``.
    sigma : float
        Passed to ``cv2.SIFT_create``.
    min_distance_mm : float
        Greedy 3-D NMS radius (mm).
    max_keypoints : int
        Upper bound on returned keypoints.
    preprocess : bool
        Apply standard benchmark preprocessing (N4+NLM+norm for MRI;
        norm only for CT).  Default True.
    use_n4 : bool
        N4 bias correction (MRI only, requires preprocess=True).
    use_denoise : bool
        NLM denoising (MRI only, requires preprocess=True).

    Returns
    -------
    coords : np.ndarray, shape [N, 4]
        Physical coordinates: (x_mm, y_mm, z_mm, scale_mm).
    descriptors : np.ndarray, shape [N, 128]
        L2-normalised SIFT descriptors.
    """
    if preprocess and hasattr(image, "numpy") and hasattr(image, "new_image_like"):
        from .preprocess import preprocess_for_landmarks
        image = preprocess_for_landmarks(image, use_n4=use_n4, use_denoise=use_denoise)

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python-headless is required: pip install opencv-python-headless") from exc

    # --- extract numpy array and spacing ---------------------------------
    if hasattr(image, "numpy"):
        arr = image.numpy().astype(np.float32)          # [X, Y, Z] ANTs layout
        sp = tuple(float(s) for s in image.spacing)     # (sx, sy, sz)
    else:
        arr = np.asarray(image, dtype=np.float32)
        sp = (1.0, 1.0, 1.0)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got shape {arr.shape}")

    DX, DY, DZ = arr.shape
    sx, sy, sz = sp[0], sp[1], sp[2] if len(sp) > 2 else 1.0

    sift = cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=3,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma,
    )

    all_coords: list[np.ndarray] = []
    all_descs:  list[np.ndarray] = []

    def _run_slice(sl2d: np.ndarray, x_mm_fixed: Optional[float],
                   y_mm_fixed: Optional[float], z_mm_fixed: Optional[float],
                   u_axis: str, v_axis: str):
        """Run SIFT on one 2-D uint8 slice and back-project to 3-D mm."""
        img8 = _normalize_slice_uint8(sl2d)
        kps, descs = sift.detectAndCompute(img8, None)
        if kps is None or len(kps) == 0 or descs is None:
            return

        for kp, desc in zip(kps, descs):
            u_px, v_px = kp.pt          # OpenCV: (col, row) = (x_img, y_img)
            scale = kp.size             # SIFT scale (pixels)

            # Map (u_px, v_px) back to (x_mm, y_mm, z_mm)
            if u_axis == 'x' and v_axis == 'y':
                xm, ym, zm = u_px * sx, v_px * sy, z_mm_fixed
            elif u_axis == 'x' and v_axis == 'z':
                xm, ym, zm = u_px * sx, y_mm_fixed, v_px * sz
            elif u_axis == 'y' and v_axis == 'z':
                xm, ym, zm = x_mm_fixed, u_px * sy, v_px * sz
            else:
                return

            scale_mm = scale * float(np.mean([sx, sy, sz]))
            all_coords.append([xm, ym, zm, scale_mm])
            all_descs.append(desc.astype(np.float32))

    # ANTs array layout: arr[ix, iy, iz] where ix ↔ x, iy ↔ y, iz ↔ z
    z_indices = np.linspace(0, DZ - 1, n_slices_per_axis, dtype=int)
    y_indices  = np.linspace(0, DY - 1, n_slices_per_axis, dtype=int)
    x_indices  = np.linspace(0, DX - 1, n_slices_per_axis, dtype=int)

    # Axial slices: fix z, sample in (x, y)
    for iz in z_indices:
        sl = arr[:, :, iz].T    # shape [DY, DX] → row=y, col=x for OpenCV
        _run_slice(sl, None, None, float(iz) * sz, 'x', 'y')

    # Coronal slices: fix y, sample in (x, z)
    for iy in y_indices:
        sl = arr[:, iy, :].T    # shape [DZ, DX] → row=z, col=x
        _run_slice(sl, None, float(iy) * sy, None, 'x', 'z')

    # Sagittal slices: fix x, sample in (y, z)
    for ix in x_indices:
        sl = arr[ix, :, :].T    # shape [DZ, DY] → row=z, col=y
        _run_slice(sl, float(ix) * sx, None, None, 'y', 'z')

    if not all_coords:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0, 128), dtype=np.float32)

    coords = np.array(all_coords, dtype=np.float32)   # [N, 4]
    descs  = np.array(all_descs,  dtype=np.float32)   # [N, 128]

    # L2-normalise descriptors
    norms = np.linalg.norm(descs, axis=1, keepdims=True)
    descs = descs / (norms + 1e-8)

    # 3-D greedy NMS
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
