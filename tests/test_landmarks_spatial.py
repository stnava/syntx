import pytest
import numpy as np
import ants
import syntx
from syntx.landmarks.spatial import (
    get_image_affine,
    vox_to_physical,
    vox_zyx_to_physical,
    physical_to_vox,
    extract_ortho_slices,
    project_to_slice,
    safe_whichtoinvert,
)
from syntx.landmarks import (
    detect_sift3d,
    detect_blobs_log,
    detect_blobs_dog,
    extract_mind_at_points,
    match_landmarks,
    ransac_filter,
)

def test_safe_whichtoinvert():
    assert safe_whichtoinvert(['w', 'a'], [False]) == [False, False]
    assert safe_whichtoinvert(['w', 'a'], [True, False]) == [True, False]
    assert safe_whichtoinvert(['w'], [True, False]) == [True]
    assert safe_whichtoinvert(['w'], [False]) == [False]

def test_spatial_affine_roundtrip():
    ds = syntx.benchmark_data('mbhard')
    fi = ds['fixed']
    org, sp, D = get_image_affine(fi)
    assert len(org) == 3 and len(sp) == 3 and D.shape == (3, 3)

    sample_vox = np.array([[0, 0, 0], [12, 24, 36], [60, 110, 75]], dtype=float)
    phys = vox_to_physical(fi, sample_vox)
    recovered = physical_to_vox(fi, phys)
    assert np.allclose(sample_vox, recovered, atol=1e-5)

    for v in sample_vox:
        ants_pt = ants.transform_index_to_physical_point(fi, tuple(int(x) for x in v))
        our_pt = vox_to_physical(fi, v)[0]
        assert np.allclose(ants_pt, our_pt, atol=1e-4)

def test_extract_ortho_slices():
    ds = syntx.benchmark_data('mbhard')
    fi = ds['fixed']
    slices = extract_ortho_slices(fi)
    for k in ('ax', 'cor', 'sag', 'center_vox', 'center_mm', 'direction'):
        assert k in slices
    assert slices['ax'].ndim == 2
    assert slices['cor'].ndim == 2
    assert slices['sag'].ndim == 2

def test_project_to_slice():
    ds = syntx.benchmark_data('mbhard')
    fi = ds['fixed']
    slices = extract_ortho_slices(fi)
    sample_vox = np.array([[slices['center_vox'][0], slices['center_vox'][1], slices['center_vox'][2]]], dtype=float)
    phys = vox_to_physical(fi, sample_vox)
    u, v, mask = project_to_slice(fi, phys, slice_axis=2, slice_pos_mm=slices['center_mm'][2], slab_half_mm=5.0)
    assert mask[0]
    assert np.isclose(u[0], slices['center_vox'][0])
    assert np.isclose(v[0], slices['center_vox'][1])

def test_detect_sift3d_spatial():
    ds = syntx.benchmark_data('mbhard')
    fi = ds['fixed']
    c, d = detect_sift3d(fi, max_keypoints=32, preprocess=False)
    assert len(c) > 0 and len(d) > 0
    assert c.shape[1] == 4  # (x, y, z, sigma)

def test_detect_blobs_dog_spatial():
    ds = syntx.benchmark_data('mbhard')
    fi = ds['fixed']
    c = detect_blobs_dog(fi, max_keypoints=32, min_distance_mm=6.0, preprocess=False)
    assert len(c) > 0
    assert c.shape[1] == 4
