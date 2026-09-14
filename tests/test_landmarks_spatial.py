"""
Physical-space framework tests for syntx.landmarks.spatial and the detectors.

All tests run on simulated data (siq procedural brain phantom) stored in
several native frames that describe the *same physical object*:
identity (LPS array), x/y flipped (RAS-like array), axis-permuted + flipped,
and an anisotropically resampled copy.  Real-data smoke tests on the
Mindboggle `mbhard` pair live in ``test_landmarks_mbhard.py``.
"""
from __future__ import annotations

import numpy as np
import pytest
import ants
import torch

from syntx.landmarks import spatial as S
from syntx.landmarks.spatial import (
    get_image_affine,
    vox_to_physical,
    vox_zyx_to_physical,
    physical_to_vox,
    physical_offset_to_voxel,
    sample_tensor_at_physical,
    axis_orientation_code,
    extract_ortho_slices,
    project_to_slice,
    safe_whichtoinvert,
)

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Fixtures: same physical object, different native storage frames
# ---------------------------------------------------------------------------

def reframe(img, perm, flips):
    """Transpose + flip the array and fix origin/direction so physical content is unchanged."""
    a = img.numpy()
    org, sp, D = get_image_affine(img)
    perm = list(perm)
    a2 = np.transpose(a, perm)
    D2 = D[:, perm].copy()
    sp2 = sp[perm]
    corner = np.zeros(3)
    for new_ax, f in enumerate(flips):
        if f:
            a2 = np.flip(a2, axis=new_ax)
            D2[:, new_ax] *= -1
            corner[perm[new_ax]] = a.shape[perm[new_ax]] - 1
    org2 = vox_to_physical(img, corner)[0].astype(np.float64)
    return ants.from_numpy(np.ascontiguousarray(a2), origin=tuple(org2), spacing=tuple(sp2), direction=D2)


@pytest.fixture(scope="module")
def phantom():
    import siq
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(0)
    np.random.seed(1234)                      # siq draws from the global numpy RNG
    ph = siq.simulate_brain_procedural((80, 88, 72))
    arr = gaussian_filter(ph.numpy().astype(np.float32), 1.0)
    arr = arr + rng.normal(0, 0.02, arr.shape).astype(np.float32) * (arr > 0)
    return ants.from_numpy(arr, origin=(-40.0, -55.0, -30.0), spacing=(1.0, 1.0, 1.0), direction=np.eye(3))


@pytest.fixture(scope="module")
def frames(phantom):
    return {
        "identity": phantom,
        "flip_xy": reframe(phantom, (0, 1, 2), (True, True, False)),
        "perm_zxy_flipz": reframe(phantom, (2, 0, 1), (True, False, False)),
        "aniso": ants.resample_image(phantom, (1.2, 1.0, 0.9), use_voxels=False, interp_type=0),
    }


# ---------------------------------------------------------------------------
# Coordinate mapping
# ---------------------------------------------------------------------------

def test_safe_whichtoinvert():
    assert safe_whichtoinvert(['w', 'a'], [False]) == [False, False]
    assert safe_whichtoinvert(['w', 'a'], [True, False]) == [True, False]
    assert safe_whichtoinvert(['w'], [True, False]) == [True]


def test_orientation_codes(frames):
    assert axis_orientation_code(frames["identity"]) == "LPS"
    assert axis_orientation_code(frames["flip_xy"]) == "RAS"
    assert axis_orientation_code(frames["perm_zxy_flipz"]) == "ILP"


def test_affine_parity_with_ants(frames):
    for name, im in frames.items():
        idx = np.column_stack([RNG.integers(0, n, 25) for n in im.shape]).astype(float)
        ours = vox_to_physical(im, idx)
        ref = np.array([ants.transform_index_to_physical_point(im, [int(v) for v in r]) for r in idx])
        assert np.allclose(ours, ref, atol=1e-4), name
        back = physical_to_vox(im, ours)
        assert np.allclose(back, idx, atol=1e-4), name
        ref_back = np.array([ants.transform_physical_point_to_index(im, list(map(float, p))) for p in ours])
        assert np.allclose(back, ref_back, atol=1e-4), name


def test_reframed_images_share_physical_content(frames):
    A = frames["identity"]
    pts = vox_to_physical(A, np.column_stack([RNG.integers(3, n - 3, 100) for n in A.shape]).astype(float))
    ref = A.numpy()[tuple(np.rint(physical_to_vox(A, pts)).astype(int).T)]
    for name in ("flip_xy", "perm_zxy_flipz"):
        im = frames[name]
        vals = im.numpy()[tuple(np.rint(physical_to_vox(im, pts)).astype(int).T)]
        assert np.allclose(vals, ref), name


def test_vox_zyx_only_reverses_index_order():
    aff = (np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.5, 2.0]), np.eye(3))
    assert np.allclose(vox_zyx_to_physical(aff, [[5, 6, 7]]), vox_to_physical(aff, [[7, 6, 5]]))


def test_physical_offset_to_voxel_flipped_frame(frames):
    im = frames["flip_xy"]                     # ix increases toward -x (Right)
    v = physical_offset_to_voxel(im, [[1.0, 0.0, 0.0]])[0]
    assert np.allclose(v, [-1.0, 0.0, 0.0])
    im = frames["perm_zxy_flipz"]              # array axis 0 is I (-z), axis 1 is L (+x), axis 2 is P (+y)
    v = physical_offset_to_voxel(im, [[0.0, 0.0, 2.0]])[0]
    assert np.allclose(v, [-2.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Torch sampling layout
# ---------------------------------------------------------------------------

def test_sample_tensor_at_physical_recovers_coordinates(frames):
    for name, im in frames.items():
        ii = np.indices(im.shape).reshape(3, -1).T.astype(float)
        coords = vox_to_physical(im, ii).reshape(*im.shape, 3)
        vol = torch.from_numpy(np.moveaxis(coords, -1, 0).copy()).unsqueeze(0)   # [1,3,nx,ny,nz]
        q = vox_to_physical(im, np.column_stack([RNG.uniform(1, n - 2, 40) for n in im.shape]))
        got = sample_tensor_at_physical(vol, im, q).numpy()
        assert np.abs(got - q).max() < 1e-3, name


def test_image_to_tensor_keeps_xyz_layout(frames):
    im = frames["perm_zxy_flipz"]
    t = S.image_to_tensor(im)
    assert tuple(t.shape) == (1, 1) + tuple(im.shape)
    arr = im.numpy()
    i = (5, 7, 9)
    assert float(t[0, 0][i]) == pytest.approx(float(arr[i]))


# ---------------------------------------------------------------------------
# Display standards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("convention", ["radiological", "neurological"])
def test_ortho_slices_and_projection_follow_display_standard(frames, convention):
    A = frames["identity"]
    ctr = vox_to_physical(A, np.array(A.shape) / 2.0)[0]
    dot_mm = ctr + np.array([12.0, -9.0, 7.0])       # patient Left (+x), Anterior (-y), Superior (+z)
    for name, im in frames.items():
        z = np.zeros(im.shape, np.float32)
        z[tuple(np.rint(physical_to_vox(im, dot_mm)[0]).astype(int))] = 1.0
        dot = ants.from_numpy(z, origin=im.origin, spacing=im.spacing, direction=im.direction)
        sl = extract_ortho_slices(dot, center_mm=dot_mm, convention=convention)
        for view in ("ax", "cor", "sag"):
            spec = sl["views"][view]
            r, c = np.unravel_index(np.argmax(sl[view]), sl[view].shape)
            u, v, m = project_to_slice(dot, dot_mm[None], view=spec)
            uc, vc, _ = project_to_slice(dot, ctr[None], view=spec)
            # projected coordinate hits the bright pixel of the extracted slice
            assert m[0] and abs(u[0] - c) < 0.51 and abs(v[0] - r) < 0.51, (name, view)
            # vertical: anterior (axial) / superior (coronal, sagittal) is UP
            assert v[0] > vc[0], (name, view)
            if view == "sag":
                assert u[0] > uc[0], (name, view)                     # anterior on the right
                assert spec["labels"] == dict(left="P", right="A", bottom="I", top="S")
            else:
                if convention == "radiological":
                    assert u[0] > uc[0], (name, view)                 # patient Left on viewer's right
                    assert spec["labels"]["right"] == "L"
                else:
                    assert u[0] < uc[0], (name, view)
                    assert spec["labels"]["left"] == "L"
            assert sl[view].shape == (spec["n_v"], spec["n_u"])


def test_project_to_slice_legacy_signature(frames):
    im = frames["flip_xy"]
    ctr = np.array(ants.get_center_of_mass(im))
    sl = extract_ortho_slices(im, center_mm=ctr)
    spec = sl["views"]["ax"]
    u1, v1, m1 = project_to_slice(im, ctr[None], view=spec)
    u2, v2, m2 = project_to_slice(im, ctr[None], slice_axis=2, slice_pos_mm=float(ctr[2]))
    assert np.allclose(u1, u2) and np.allclose(v1, v2) and m1[0] and m2[0]


def test_slab_mask_uses_plane_distance(frames):
    im = frames["perm_zxy_flipz"]
    ctr = np.array(ants.get_center_of_mass(im))
    spec = S.ortho_view_spec(im, "axial", ctr)
    pts = np.stack([ctr, ctr + [0, 0, 3.0], ctr + [0, 0, 20.0]])
    _, _, m = project_to_slice(im, pts, view=spec, slab_half_mm=5.0)
    assert m.tolist() == [True, True, False]
