"""
syntx.data.surrogates -- threshold / connected-component masks used as overlap targets.

Some MSD tasks label only focal pathology (Task06_Lung: nodules; Task01_BrainTumour: tumour)
or thin structures (Task08_HepaticVessel: vessels). Those labels are at different places in
different subjects, so inter-subject label overlap is near zero whatever the registration.
These functions build a shared whole-organ-like mask from the image itself instead:

- CT lung: HU in [-950, -350] inside the body trunk, the two largest components.
- CT abdomen: HU in [25, 160] inside the body trunk, morphologically closed, large components.
- Brain MRI: positive (or above a low percentile) voxels, largest component, holes filled.

All masks are simple heuristics with fixed thresholds; they are not anatomical segmentations.
"""

from typing import Optional
import numpy as np
import scipy.ndimage as ndi
import ants


def extract_ct_body_trunk(
    image: ants.ANTsImage,
    hu_threshold: float = -500.0
) -> np.ndarray:
    """Body mask of a 3-D CT: per slice, the largest connected region above a HU threshold.

    Slices are taken along the third array axis of ``image.numpy()`` (ANTs index z; this is
    axial only if the image is stored that way). In each slice the voxels ``> hu_threshold``
    are labelled (2-D 4-connectivity), the largest component is kept and its holes are filled
    (so the lungs and bowel gas inside the body are included).

    Parameters
    ----------
    image : ants.ANTsImage
        3-D CT in Hounsfield units.
    hu_threshold : float, default -500.0
        Voxels above this value count as body.

    Returns
    -------
    np.ndarray of bool, same shape as ``image.numpy()`` (ANTs (x, y, z) order). Slices with no
    voxel above the threshold are all False. Note: a NumPy array, not an ANTsImage.
    """
    arr = image.numpy()
    trunk_filled = np.zeros_like(arr, dtype=bool)

    for z in range(arr.shape[2]):
        sl = arr[:, :, z] > hu_threshold
        lbl, n_components = ndi.label(sl)
        if n_components > 0:
            sizes = ndi.sum(sl, lbl, range(1, n_components + 1))
            trunk = (lbl == (np.argmax(sizes) + 1))
            trunk_filled[:, :, z] = ndi.binary_fill_holes(trunk)

    return trunk_filled


def extract_ct_lung_parenchyma(
    image: ants.ANTsImage,
    min_hu: float = -950.0,
    max_hu: float = -350.0,
    min_volume_voxels: int = 20000,
) -> ants.ANTsImage:
    """Lung mask of a thoracic CT: low-HU voxels inside the body, two largest 3-D components.

    Voxels with ``min_hu <= HU <= max_hu`` inside ``extract_ct_body_trunk(image)`` (default
    trunk threshold -500 HU) are labelled in 3-D (6-connectivity). Of the two largest
    components, each one with at least ``min_volume_voxels`` voxels is kept. Airways connected
    to the lungs within the HU window are included.

    Parameters
    ----------
    image : ants.ANTsImage
        3-D CT in Hounsfield units.
    min_hu, max_hu : float, default -950.0, -350.0
        Inclusive HU window.
    min_volume_voxels : int, default 20000
        Minimum component size in voxels (not mm^3, so it depends on resolution).

    Returns
    -------
    ants.ANTsImage
        float32 mask (1 = lung, 0 = outside) with the geometry of ``image``; all zeros when
        no voxel is in the window or no component is large enough.
    """
    arr = image.numpy()
    body_trunk = extract_ct_body_trunk(image)

    lung_voxels = (arr >= min_hu) & (arr <= max_hu) & body_trunk
    lbl, n_feat = ndi.label(lung_voxels)

    if n_feat == 0:
        return image.new_image_like(np.zeros_like(arr, dtype=np.float32))

    sizes = ndi.sum(lung_voxels, lbl, range(1, n_feat + 1))
    top_indices = np.argsort(sizes)[::-1]

    lung_mask = np.zeros_like(arr, dtype=np.float32)
    for idx in top_indices[:2]:  # Keep left and right lung
        if sizes[idx] >= min_volume_voxels:
            lung_mask[lbl == (idx + 1)] = 1.0

    return image.new_image_like(lung_mask)


def extract_ct_abdominal_viscera(
    image: ants.ANTsImage,
    min_hu: float = 25.0,
    max_hu: float = 160.0,
    min_volume_voxels: int = 5000,
) -> ants.ANTsImage:
    """Soft-tissue mask of an abdominal CT: HU window inside the body, closed, large components.

    Voxels with ``min_hu <= HU <= max_hu`` inside ``extract_ct_body_trunk(image)`` are closed
    with a 6-connected structuring element (2 iterations), labelled in 3-D, and every component
    with at least ``min_volume_voxels`` voxels is kept. The default window excludes fat and
    dense bone, but it also contains muscle, blood vessels and contrast-free bowel wall, so
    the mask is "soft tissue inside the body", not specifically liver / spleen / kidneys.

    Parameters
    ----------
    image : ants.ANTsImage
        3-D CT in Hounsfield units.
    min_hu, max_hu : float, default 25.0, 160.0
        Inclusive HU window.
    min_volume_voxels : int, default 5000
        Minimum component size in voxels.

    Returns
    -------
    ants.ANTsImage
        float32 mask (1 = inside, 0 = outside) with the geometry of ``image``; all zeros
        when nothing is in the window.
    """
    arr = image.numpy()
    body_trunk = extract_ct_body_trunk(image)

    viscera = (arr >= min_hu) & (arr <= max_hu) & body_trunk
    struct = ndi.generate_binary_structure(3, 1)
    viscera_closed = ndi.binary_closing(viscera, structure=struct, iterations=2)

    lbl, n_feat = ndi.label(viscera_closed)
    if n_feat == 0:
        return image.new_image_like(np.zeros_like(arr, dtype=np.float32))

    sizes = ndi.sum(viscera_closed, lbl, range(1, n_feat + 1))
    valid_components = np.where(sizes >= min_volume_voxels)[0] + 1

    mask = np.isin(lbl, valid_components).astype(np.float32)
    return image.new_image_like(mask)


def extract_brain_parenchyma(
    image: ants.ANTsImage,
    min_volume_voxels: int = 50000,
) -> ants.ANTsImage:
    """Whole-head-foreground mask of a (skull-stripped) brain MRI, independent of the tumour.

    - 4-D input: channel index 1 (T1w in the Task01 channel order) is thresholded at > 0.
    - 3-D input: threshold at the 5th percentile of the positive voxels (0 if none).

    The largest 3-D connected component (6-connectivity) is kept and its holes are filled. If
    that component is smaller than ``min_volume_voxels`` (or there is none), the raw
    threshold mask is returned unchanged. This is a foreground mask: on images that are not
    skull-stripped it will include skull and scalp.

    Parameters
    ----------
    image : ants.ANTsImage
        3-D image, or 4-D image with channels on the last axis (MSD Task01 layout).
    min_volume_voxels : int, default 50000
        Size in voxels the largest component must reach to be used.

    Returns
    -------
    ants.ANTsImage
        float32 mask (1 = inside), 3-D. For 4-D input it takes the geometry of channel 0.
    """
    if image.dimension == 4:
        # For 4D MRI (e.g. BraTS FLAIR/T1/T1gd/T2), extract T1 channel for robust brain boundary
        sl = ants.slice_image(image, axis=3, idx=1)  # T1w
        arr = sl.numpy()
        mask = (arr > 0)
        ref = ants.slice_image(image, axis=3, idx=0)
    else:
        arr = image.numpy()
        pos = arr[arr > 0]
        thresh = float(np.percentile(pos, 5.0)) if len(pos) > 0 else 0.0
        mask = (arr > thresh)
        ref = image

    # Retain largest connected component (brain) and fill internal ventricular holes
    lbl, n = ndi.label(mask)
    if n > 0:
        sizes = ndi.sum(mask, lbl, range(1, n + 1))
        top_idx = int(np.argmax(sizes)) + 1
        if sizes[top_idx - 1] >= min_volume_voxels:
            brain = (lbl == top_idx)
            brain_filled = ndi.binary_fill_holes(brain)
        else:
            brain_filled = mask
    else:
        brain_filled = mask

    return ref.new_image_like(brain_filled.astype(np.float32))


def extract_surrogate_target(
    image: ants.ANTsImage,
    task_name: str,
    **kwargs
) -> Optional[ants.ANTsImage]:
    """Pick a surrogate mask by substring match on the task name.

    Matching is on ``task_name.lower()``, checked in this order:

    - contains 'lung' or 'task06' -> ``extract_ct_lung_parenchyma``
    - contains 'vessel', 'hepatic' or 'task08' -> ``extract_ct_abdominal_viscera``
    - contains 'brain', 'task01' or 'brats' -> ``extract_brain_parenchyma``

    Parameters
    ----------
    image : ants.ANTsImage
        Input image passed to the chosen extractor.
    task_name : str
        Task key or name, e.g. 'Task06_Lung'.
    **kwargs
        Forwarded to the chosen extractor (so only its own keyword names are accepted).

    Returns
    -------
    ants.ANTsImage or None
        The mask, or None when no rule matches (e.g. 'Task04_Hippocampus', 'Task03_Liver').
    """
    task_clean = task_name.lower()
    if "lung" in task_clean or "task06" in task_clean:
        return extract_ct_lung_parenchyma(image, **kwargs)
    elif "vessel" in task_clean or "hepatic" in task_clean or "task08" in task_clean:
        return extract_ct_abdominal_viscera(image, **kwargs)
    elif "brain" in task_clean or "braintumour" in task_clean or "task01" in task_clean or "brats" in task_clean:
        return extract_brain_parenchyma(image, **kwargs)
    return None


