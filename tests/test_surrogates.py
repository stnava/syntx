"""
Tests for syntx.data.surrogates module.
"""

import numpy as np
import ants
from syntx.data.surrogates import (
    extract_ct_body_trunk,
    extract_ct_lung_parenchyma,
    extract_ct_abdominal_viscera,
    extract_brain_parenchyma,
    extract_surrogate_target,
)


def _create_synthetic_ct_thorax():
    """
    Creates a synthetic 3D CT thorax volume (32x32x16):
    - Background air: -1000 HU
    - Body wall / trunk: +40 HU
    - Bilateral lung cavities: -700 HU
    """
    arr = np.full((32, 32, 16), -1000.0, dtype=np.float32)
    # Body trunk (ellipse)
    y, x = np.ogrid[:32, :32]
    body_mask = ((x - 16)**2 / 14**2 + (y - 16)**2 / 12**2) <= 1.0
    for z in range(16):
        arr[body_mask, z] = 40.0

    # Lungs (two smaller ellipses inside body)
    left_lung = ((x - 10)**2 / 4**2 + (y - 16)**2 / 6**2) <= 1.0
    right_lung = ((x - 22)**2 / 4**2 + (y - 16)**2 / 6**2) <= 1.0
    for z in range(2, 14):
        arr[left_lung, z] = -700.0
        arr[right_lung, z] = -700.0

    img = ants.from_numpy(arr, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    return img


def _create_synthetic_ct_abdomen():
    """
    Creates a synthetic 3D CT abdomen volume (32x32x16):
    - Background air: -1000 HU
    - Subcutaneous fat: -80 HU
    - Visceral organ envelope (liver/spleen): +60 HU
    """
    arr = np.full((32, 32, 16), -1000.0, dtype=np.float32)
    y, x = np.ogrid[:32, :32]
    # Body trunk with fat
    body_mask = ((x - 16)**2 / 14**2 + (y - 16)**2 / 12**2) <= 1.0
    for z in range(16):
        arr[body_mask, z] = -80.0

    # Viscera inside trunk
    viscera_mask = ((x - 16)**2 / 9**2 + (y - 16)**2 / 8**2) <= 1.0
    for z in range(2, 14):
        arr[viscera_mask, z] = 60.0

    img = ants.from_numpy(arr, origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0))
    return img


def test_extract_ct_lung_parenchyma():
    img = _create_synthetic_ct_thorax()
    lung_mask = extract_ct_lung_parenchyma(img, min_volume_voxels=50)

    assert isinstance(lung_mask, ants.ANTsImage)
    assert lung_mask.shape == img.shape
    arr_mask = lung_mask.numpy()
    assert (arr_mask == 1.0).sum() > 0
    # Boundary air should NOT be part of lung mask
    assert arr_mask[0, 0, 0] == 0.0
    # Center of left lung should be in lung mask
    assert arr_mask[16, 10, 8] == 1.0
    assert arr_mask[16, 22, 8] == 1.0


def test_extract_ct_abdominal_viscera():
    img = _create_synthetic_ct_abdomen()
    viscera_mask = extract_ct_abdominal_viscera(img, min_volume_voxels=50)

    assert isinstance(viscera_mask, ants.ANTsImage)
    assert viscera_mask.shape == img.shape
    arr_mask = viscera_mask.numpy()
    assert (arr_mask == 1.0).sum() > 0
    # Boundary air and fat should NOT be part of viscera mask
    assert arr_mask[0, 0, 0] == 0.0
    assert arr_mask[16, 3, 8] == 0.0  # Fat layer
    # Core viscera should be in mask
    assert arr_mask[16, 16, 8] == 1.0


def test_extract_brain_parenchyma():
    # 4D synthetic brain MRI (32x32x16x4)
    arr4d = np.zeros((32, 32, 16, 4), dtype=np.float32)
    y, x = np.ogrid[:32, :32]
    brain_mask = ((x - 16)**2 / 12**2 + (y - 16)**2 / 10**2) <= 1.0
    for c in range(4):
        for z in range(2, 14):
            arr4d[brain_mask, z, c] = 100.0 * (c + 1)

    img4d = ants.from_numpy(arr4d, origin=(0.0, 0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0, 1.0))
    tgt_brain = extract_brain_parenchyma(img4d, min_volume_voxels=50)

    assert isinstance(tgt_brain, ants.ANTsImage)
    assert tgt_brain.dimension == 3
    assert (tgt_brain.numpy() == 1.0).sum() > 0
    assert tgt_brain.numpy()[0, 0, 0] == 0.0
    assert tgt_brain.numpy()[16, 16, 8] == 1.0


def test_extract_surrogate_target_dispatcher():
    img_lung = _create_synthetic_ct_thorax()
    tgt_lung = extract_surrogate_target(img_lung, "Task06_Lung", min_volume_voxels=50)
    assert tgt_lung is not None
    assert (tgt_lung.numpy() == 1.0).sum() > 0

    img_hep = _create_synthetic_ct_abdomen()
    tgt_hep = extract_surrogate_target(img_hep, "Task08_HepaticVessel", min_volume_voxels=50)
    assert tgt_hep is not None
    assert (tgt_hep.numpy() == 1.0).sum() > 0

    # 4D synthetic brain MRI
    arr4d = np.zeros((32, 32, 16, 4), dtype=np.float32)
    y, x = np.ogrid[:32, :32]
    brain_mask = ((x - 16)**2 / 12**2 + (y - 16)**2 / 10**2) <= 1.0
    arr4d[brain_mask, 8, 1] = 100.0
    img4d = ants.from_numpy(arr4d)
    tgt_brain = extract_surrogate_target(img4d, "Task01_BrainTumour", min_volume_voxels=10)
    assert tgt_brain is not None
    assert (tgt_brain.numpy() == 1.0).sum() > 0

    tgt_none = extract_surrogate_target(img_lung, "UnknownTask")
    assert tgt_none is None


def test_preprocess_for_landmarks_channel_idx():
    from syntx.landmarks.preprocess import preprocess_for_landmarks

    arr4d = np.zeros((24, 24, 12, 4), dtype=np.float32)
    arr4d[4:20, 4:20, 2:10, 0] = 50.0   # channel 0
    arr4d[4:20, 4:20, 2:10, 1] = 150.0  # channel 1
    arr4d[4:20, 4:20, 2:10, 3] = 300.0  # channel 3
    img4d = ants.from_numpy(arr4d)

    pre_ch0 = preprocess_for_landmarks(img4d, channel_idx=0, use_n4=False, use_denoise=False)
    pre_ch1 = preprocess_for_landmarks(img4d, channel_idx=1, use_n4=False, use_denoise=False)
    pre_ch3 = preprocess_for_landmarks(img4d, channel_idx=3, use_n4=False, use_denoise=False)

    assert pre_ch0.dimension == 3
    assert pre_ch1.dimension == 3
    assert pre_ch3.dimension == 3
    assert pre_ch0.numpy().max() > 0.0
    assert pre_ch1.numpy().max() > 0.0
    assert pre_ch3.numpy().max() > 0.0

