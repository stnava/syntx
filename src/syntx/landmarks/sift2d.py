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
        """Physical (x,y,z) mm for voxel index (ix, iy, iz)."""
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
        """Run SIFT on one 2-D uint8 slice and back-project to physical mm via affine."""
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

