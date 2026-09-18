"""
syntx.data.surrogates — Tissue and Shape Surrogate Extractors for Registration Evaluation
========================================================================================

In multi-organ cohorts (e.g. Medical Segmentation Decathlon), certain tasks label only
focal pathology (e.g. Task06_Lung labels solitary cancer nodules) or sub-millimeter
branching structures (e.g. Task08_HepaticVessel labels 1-voxel vessels). In inter-subject
registration, focal lesions reside at anatomically disparate sites, and fine vessels cannot
overlap under rigid/affine mappings, making raw label overlap biologically meaningless (Dice ≈ 0).

This module extracts standardized, physical tissue/shape surrogate targets directly
from CT Hounsfield Units (HU) and 3D topology:
- Thoracic CT: Bilateral lung parenchyma cavity (HU ∈ [-950, -350] inside trunk).
- Abdominal CT: Visceral soft-tissue compartment (HU ∈ [25, 160] inside trunk).
"""

from typing import Optional
import numpy as np
import scipy.ndimage as ndi
import ants


def extract_ct_body_trunk(
    image: ants.ANTsImage,
    hu_threshold: float = -500.0
) -> np.ndarray:
    """
    Extracts the binary patient body trunk mask from a 3D CT image via slice-by-slice
    axial connected component analysis and topological hole filling.
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
    """
    Extracts the whole bilateral lung parenchyma shape from a thoracic CT scan.

    Returns a binary ANTsImage (1.0 = lung parenchyma, 0.0 = outside).
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
    """
    Extracts the abdominal visceral soft-tissue envelope (liver, spleen, kidneys)
    from an abdominal CT scan, excluding subcutaneous fat and bone.

    Returns a binary ANTsImage (1.0 = visceral parenchyma, 0.0 = outside).
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


def extract_surrogate_target(
    image: ants.ANTsImage,
    task_name: str,
    **kwargs
) -> Optional[ants.ANTsImage]:
    """
    Convenience dispatcher to extract the appropriate tissue/shape surrogate
    evaluation target for CT tasks with focal pathologies or non-shared structures.
    """
    task_clean = task_name.lower()
    if "lung" in task_clean or "task06" in task_clean:
        return extract_ct_lung_parenchyma(image, **kwargs)
    elif "vessel" in task_clean or "hepatic" in task_clean or "task08" in task_clean:
        return extract_ct_abdominal_viscera(image, **kwargs)
    return None

