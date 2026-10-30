"""tests/test_frames.py — Comprehensive tests for syntx.frames and F1–F6 conventions."""

import pytest
import numpy as np
import ants

from syntx.frames import (
    VECTOR_FRAME,
    fsl_handedness,
    bvecs_to_image_frame,
    image_frame_to_bvecs,
    image_to_lps,
    lps_to_image,
    reorient_gradient_table,
    dec_rgb,
    reference_directions,
)
from syntx.transforms import (
    TransformChain,
    reorient_vectors,
    reorient_tensors,
)
from syntx.spatial import (
    ortho_view_spec,
    slice_from_spec,
    extract_ortho_slices,
)


def test_vector_frame_constant():
    assert VECTOR_FRAME == "image_axes"


def test_reference_directions_orthonormal_and_determinants():
    refs = reference_directions()
    assert set(refs.keys()) == {"LAS", "RPI", "RAS", "LPI", "PIR", "oblique_pos", "oblique_neg"}

    for name, D in refs.items():
        assert D.shape == (3, 3)
        assert D.dtype == np.float64
        # Orthogonality: D.T @ D == I
        np.testing.assert_allclose(D.T @ D, np.eye(3), atol=1e-12)
        det = np.linalg.det(D)
        if name in ("LAS", "RPI", "oblique_neg"):
            assert np.isclose(det, -1.0)
        else:
            assert np.isclose(det, 1.0)


def test_fsl_handedness_involution_and_signs():
    refs = reference_directions()
    for name, D in refs.items():
        F = fsl_handedness(D)
        # F is an involution: F @ F == I
        np.testing.assert_allclose(F @ F, np.eye(3), atol=1e-12)
        det = np.linalg.det(D)
        if det > 0:
            np.testing.assert_allclose(F, np.diag([-1.0, 1.0, 1.0]))
        else:
            np.testing.assert_allclose(F, np.eye(3))

    # Singular matrix check
    with pytest.raises(ValueError, match="Singular"):
        fsl_handedness(np.zeros((3, 3)))


@pytest.mark.parametrize("name", list(reference_directions().keys()))
def test_bvec_image_frame_roundtrips(name):
    D = reference_directions()[name]
    rng = np.random.default_rng(42)

    # Unit bvecs plus zero rows (b0)
    raw = rng.standard_normal((12, 3))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    raw[0] = [0.0, 0.0, 0.0]
    raw[5] = [0.0, 0.0, 0.0]

    g_img = bvecs_to_image_frame(raw, D)
    assert np.all(g_img[0] == 0.0)
    assert np.all(g_img[5] == 0.0)

    # Non-zero rows are unit vectors
    nz_norms = np.linalg.norm(g_img[[1, 2, 3, 4, 6, 7]], axis=1)
    np.testing.assert_allclose(nz_norms, 1.0, atol=1e-12)

    b_roundtrip = image_frame_to_bvecs(g_img, D)
    np.testing.assert_allclose(b_roundtrip, raw, atol=1e-12)

    # Single 1-D vector roundtrip
    v1 = raw[1]
    g1 = bvecs_to_image_frame(v1, D)
    assert g1.shape == (3,)
    v1_roundtrip = image_frame_to_bvecs(g1, D)
    np.testing.assert_allclose(v1_roundtrip, v1, atol=1e-12)


@pytest.mark.parametrize("name", list(reference_directions().keys()))
def test_image_to_lps_vectors_and_tensors(name):
    D = reference_directions()[name]
    rng = np.random.default_rng(123)

    # 1. Vectors (N, 3) and (X, Y, Z, 3)
    v_img = rng.standard_normal((10, 3))
    v_lps = image_to_lps(v_img, D)
    v_back = lps_to_image(v_lps, D)
    np.testing.assert_allclose(v_back, v_img, atol=1e-12)
    # Physical relation: v_lps = D @ v_img
    np.testing.assert_allclose(v_lps, v_img @ D.T, atol=1e-12)

    # 2. 3x3 Tensors (..., 3, 3)
    # Construct symmetric positive definite tensor
    A = rng.standard_normal((3, 3))
    T_img = A @ A.T + np.eye(3)
    T_lps = image_to_lps(T_img, D, kind="tensor")
    T_back = lps_to_image(T_lps, D, kind="tensor")
    np.testing.assert_allclose(T_back, T_img, atol=1e-12)
    np.testing.assert_allclose(T_lps, D @ T_img @ D.T, atol=1e-12)

    # Eigenvector parity: T_img e = λ e => T_lps (D e) = λ (D e)
    eigvals_img, eigvecs_img = np.linalg.eigh(T_img)
    eigvals_lps, eigvecs_lps = np.linalg.eigh(T_lps)
    np.testing.assert_allclose(eigvals_img, eigvals_lps, atol=1e-12)
    # Principal eigenvector
    v1_img = eigvecs_img[:, -1]
    v1_lps = eigvecs_lps[:, -1]
    # D @ v1_img must align with v1_lps (up to sign)
    dot = np.abs(np.dot(D @ v1_img, v1_lps))
    np.testing.assert_allclose(dot, 1.0, atol=1e-12)

    # 3. Sym6 tensors [xx, xy, xz, yy, yz, zz]
    sym6_img = np.array([T_img[0, 0], T_img[0, 1], T_img[0, 2], T_img[1, 1], T_img[1, 2], T_img[2, 2]])
    sym6_lps = image_to_lps(sym6_img, D, kind="tensor")
    sym6_back = lps_to_image(sym6_lps, D, kind="tensor")
    np.testing.assert_allclose(sym6_back, sym6_img, atol=1e-12)
    expected_sym6_lps = np.array([T_lps[0, 0], T_lps[0, 1], T_lps[0, 2], T_lps[1, 1], T_lps[1, 2], T_lps[2, 2]])
    np.testing.assert_allclose(sym6_lps, expected_sym6_lps, atol=1e-12)

    # 4. ANTsImage roundtrip
    grid = np.zeros((4, 4, 4, 3), dtype=np.float32)
    grid[..., 0] = 1.0
    img = ants.from_numpy(grid, direction=D, has_components=True)
    img_lps = image_to_lps(img)
    img_back = lps_to_image(img_lps)
    np.testing.assert_allclose(img_back.numpy(), grid, atol=1e-6)


def test_reorient_gradient_table_10deg_yaw():
    """Verify 10 deg yaw on RPI header has error < 0.1 deg vs analytical ground truth."""
    D_rpi = np.diag([1.0, -1.0, 1.0])  # LAS / RPI

    # 10 deg yaw rotation around Z in physical LPS
    theta = np.deg2rad(10.0)
    c, s = np.cos(theta), np.sin(theta)
    # ITK affine matrix maps: x_mov = M @ x_fix + b
    # For a pure rotation M = R_z(theta)
    M = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Mock a TransformChain with linear_part returning M
    class MockChain:
        def linear_part(self):
            return M

    chain = MockChain()

    # Direction pointing along physical X in moving space: v_phys_mov = [1, 0, 0]
    # In moving image frame: v_img_mov = D_mov.T @ v_phys_mov = [1, 0, 0]
    # Direction along physical Y in moving space: v_phys_mov = [0, 1, 0]
    # In moving image frame: v_img_mov = D_mov.T @ [0, 1, 0] = [0, -1, 0]
    g_mov = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],  # b0
    ], dtype=np.float64)

    g_reoriented = reorient_gradient_table(
        g_mov,
        chains=chain,
        moving_direction=D_rpi,
        fixed_direction=D_rpi,
        normalize=True,
    )

    # In physical space, pushing forward directions from moving to fixed space
    # applies M^-1: v_phys_fix = M^-1 @ v_phys_mov = R_z(-10 deg) @ v_phys_mov
    M_inv = np.linalg.inv(M)
    for k in range(3):
        v_phys_mov = D_rpi @ g_mov[k]
        expected_phys_fix = M_inv @ v_phys_mov
        expected_img_fix = D_rpi.T @ expected_phys_fix

        actual_img_fix = g_reoriented[k]
        cos_err = np.clip(np.dot(actual_img_fix, expected_img_fix), -1.0, 1.0)
        ang_err_deg = np.rad2deg(np.arccos(cos_err))
        assert ang_err_deg < 0.1, f"Gradient {k} angular error {ang_err_deg} >= 0.1 deg"

    # b0 stays zero
    np.testing.assert_allclose(g_reoriented[3], [0.0, 0.0, 0.0])

    # Cross-check agreement with reorient_vectors (which takes LPS rotation matrix M_inv)
    v_vol = np.zeros((4, 4, 4, 3), dtype=np.float32)
    v_vol[:] = g_mov[0]
    v_img_obj = ants.from_numpy(v_vol, direction=D_rpi, has_components=True)
    v_rot_img = reorient_vectors(v_img_obj, transformlist=M_inv, moving_direction=D_rpi, fixed_direction=D_rpi)
    rot_vec = v_rot_img.numpy()[0, 0, 0]
    cos_comp = np.clip(np.dot(g_reoriented[0], rot_vec / np.linalg.norm(rot_vec)), -1.0, 1.0)
    assert np.rad2deg(np.arccos(cos_comp)) < 0.1


def test_dec_rgb_anatomical_pir():
    """Verify anatomical DEC on PIR scan correctly maps physical Left-Right to Red."""
    refs = reference_directions()
    D_pir = refs["PIR"]
    # PIR direction matrix:
    # [[ 0,  0, -1],
    #  [ 1,  0,  0],
    #  [ 0, -1,  0]]
    # Physical LPS: axis 0 is L-R, axis 1 is A-P, axis 2 is I-S
    # v_lps = D @ v_img
    # If v_img = [0, 0, 1], then v_lps = [-1, 0, 0], which is purely along physical L-R!
    # In anatomical DEC, |v_lps| = [1, 0, 0], which is 100% Red (Channel 0).
    # In image-frame DEC, |v_img| = [0, 0, 1], which is 100% Blue (Channel 2).

    v_arr = np.zeros((8, 8, 8, 3), dtype=np.float32)
    v_arr[:] = [0.0, 0.0, 1.0]
    v_img = ants.from_numpy(v_arr, direction=D_pir, has_components=True)

    # 1. Saved standard image DEC
    rgb_image_frame = dec_rgb(v_img, frame="image")
    sample_img = rgb_image_frame.numpy()[0, 0, 0]
    np.testing.assert_allclose(sample_img, [0.0, 0.0, 1.0], atol=1e-6)

    # 2. Anatomical report DEC
    rgb_anat = dec_rgb(v_img, frame="anatomical")
    sample_anat = rgb_anat.numpy()[0, 0, 0]
    # Must be [1.0, 0.0, 0.0] -> Red channel!
    np.testing.assert_allclose(sample_anat, [1.0, 0.0, 0.0], atol=1e-6)


def test_opposite_handed_sandwich_det_negative():
    """Verify reorientation between RPI (det -1) and LPI (det +1) produces det -1 sandwiched operator."""
    refs = reference_directions()
    D_mov = refs["RPI"]  # det = -1
    D_fix = refs["LPI"]  # det = +1

    # Identity transform in LPS
    M_identity = np.eye(3, dtype=np.float64)

    # Image-frame gradient
    g_mov = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    g_reoriented = reorient_gradient_table(
        g_mov,
        chains=None,
        moving_direction=D_mov,
        fixed_direction=D_fix,
        normalize=True,
    )

    # Sandwiched operator is R_img = D_fix.T @ R_lps @ D_mov = D_fix.T @ I @ D_mov
    R_img = D_fix.T @ M_identity @ D_mov
    det_sandwiched = np.linalg.det(R_img)
    assert np.isclose(det_sandwiched, -1.0), f"Expected det -1 for opposite-handed sandwich, got {det_sandwiched}"

    # Physical vector must be preserved: v_phys_fix == v_phys_mov
    v_phys_mov = D_mov @ g_mov[0]
    v_phys_fix = D_fix @ g_reoriented[0]
    np.testing.assert_allclose(v_phys_fix, v_phys_mov, atol=1e-12)

    # Test tensor reorientation under opposite-handed frames with finite_strain
    T_img = np.diag([2.0, 1.0, 0.5])
    sym6_in = np.array([[T_img[0, 0], T_img[0, 1], T_img[0, 2], T_img[1, 1], T_img[1, 2], T_img[2, 2]]], dtype=np.float32)
    t_rot = reorient_tensors(
        sym6_in,
        transformlist=M_identity,
        moving_direction=D_mov,
        fixed_direction=D_fix,
        method="finite_strain",
    )
    # Check LPS physical tensor invariance
    idx = np.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])
    T_rot_33 = t_rot[0, idx]
    T_lps_fix = D_fix @ T_rot_33 @ D_fix.T
    T_lps_mov = D_mov @ T_img @ D_mov.T
    np.testing.assert_allclose(T_lps_fix, T_lps_mov, atol=1e-6)


def test_reorient_vectors_warp_bare_array_and_oblique():
    """Verify reorient_vectors(warp=...) handles bare arrays and oblique warp headers."""
    refs = reference_directions()
    D_oblique = refs["oblique_pos"]

    # Construct synthetic displacement field with pure 10 deg rotation in LPS
    theta = np.deg2rad(10.0)
    c, s = np.cos(theta), np.sin(theta)
    R_z = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    shape = (12, 12, 12)
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)

    # Coordinate grid
    gx, gy, gz = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    vox = np.stack([gx, gy, gz], axis=-1).astype(np.float64)
    x_phys = origin + (vox * spacing) @ D_oblique.T
    # Forward displacement: u(x) = (R_z - I) @ x
    u_phys = (x_phys @ (R_z - np.eye(3)).T).astype(np.float32)
    warp_img = ants.from_numpy(u_phys, origin=origin, spacing=spacing, direction=D_oblique, has_components=True)

    # Bare numpy vector array
    v_bare = np.zeros((*shape, 3), dtype=np.float32)
    v_bare[:] = [1.0, 0.0, 0.0]

    v_reoriented = reorient_vectors(v_bare, warp=warp_img, method="finite_strain")
    assert isinstance(v_reoriented, np.ndarray)
    assert v_reoriented.shape == (*shape, 3)

    # The interior deformation gradient F = I + \nabla u = R_z
    # Sandwiched rotation in image frame: R_img = D_oblique.T @ R_z @ D_oblique
    R_img_expected = D_oblique.T @ R_z @ D_oblique
    v_img_expected = v_bare[6, 6, 6] @ R_img_expected.T

    np.testing.assert_allclose(v_reoriented[6, 6, 6], v_img_expected, atol=1e-4)


def test_s7_multichannel_display_slicing():
    """Validate S7 multichannel RGB display slicing across headers."""
    refs = reference_directions()
    for name in ("PIR", "oblique_pos", "RPI"):
        D = refs[name]
        X, Y, Z = 16, 18, 20
        arr = np.random.default_rng(101).random((X, Y, Z, 3)).astype(np.float32)

        # 1. 3-D multicomponent ANTsImage
        img_mc = ants.from_numpy(arr, direction=D, has_components=True)
        res_mc = extract_ortho_slices(img_mc)

        # 2. 4-D ANTsImage with matching spatial direction
        D_4d = np.eye(4, dtype=np.float64)
        D_4d[:3, :3] = D
        img_4d = ants.from_numpy(arr, direction=D_4d)
        res_4d = extract_ortho_slices(img_4d)

        # 3. Plain 4-D ANTsImage without explicit direction (defaults to 4-D identity)
        img_4d_plain = ants.from_numpy(arr)
        res_4d_plain = extract_ortho_slices(img_4d_plain)

        for res, img in [(res_mc, img_mc), (res_4d, img_4d), (res_4d_plain, img_4d_plain)]:
            for view in ("ax", "cor", "sag"):
                sl = res[view]
                spec = res["views"][view]
                assert sl.shape == (spec["n_v"], spec["n_u"], 3)
                # Verify each channel matches scalar slice
                for ch in range(3):
                    scalar_sl = slice_from_spec(arr[..., ch], spec)
                    np.testing.assert_array_equal(sl[..., ch], scalar_sl)

        # 3. Test slice_index override and geometric center optimization
        spec_idx = ortho_view_spec(img_mc, "ax", slice_index=5)
        assert spec_idx["slice_index"] == 5

    # 4. Test all-zero image guard (no silent fallback)
    zero_mc = ants.from_numpy(np.zeros((10, 10, 10, 3), dtype=np.float32), has_components=True)
    with pytest.raises(ValueError, match="all zero"):
        extract_ortho_slices(zero_mc)

    zero_scalar = ants.from_numpy(np.zeros((10, 10, 10), dtype=np.float32))
    with pytest.raises(ValueError, match="all zero"):
        extract_ortho_slices(zero_scalar)


def test_s7_display_orientation_lower():
    """Confirm slice_from_spec with imshow(..., origin='lower') places labels['top'] at the top."""
    arr = np.zeros((41, 41, 41), dtype=np.float32)
    # Origin at center
    im = ants.from_numpy(arr, origin=(-20.0, -20.0, -20.0), spacing=(1.0, 1.0, 1.0), direction=np.eye(3))

    # In LPS, Anterior is negative Y:
    # Set an anterior dot at voxel y = 10 (phys y = -10, center is 20)
    arr_ant = arr.copy()
    arr_ant[20, 10, 20] = 1.0
    spec = ortho_view_spec(im, "ax", center_mm=[0.0, 0.0, 0.0])
    sl_ant = slice_from_spec(arr_ant, spec)

    r_ant, c_ant = np.unravel_index(np.argmax(sl_ant), sl_ant.shape)
    # For axial view in radiological convention:
    # labels["top"] == "A", labels["bottom"] == "P"
    # Row index of Anterior dot must be higher than center row (20)
    assert r_ant > 20
    assert spec["labels"]["top"] == "A"
    assert spec["labels"]["bottom"] == "P"


def test_image_to_lps_fixel_three_peaks_pir():
    """Verify 3-peak fixel array (X, Y, Z, 3, 3) on a PIR header with kind='vector'."""
    refs = reference_directions()
    D_pir = refs["PIR"]
    # PIR direction matrix:
    # [[ 0,  0, -1],
    #  [ 1,  0,  0],
    #  [ 0, -1,  0]]
    # For a vector v = [0, 1, 0]:
    # v_lps = D_pir @ v = [0, 0, -1]
    # For a vector v = [1, 0, 0]:
    # v_lps = D_pir @ v = [0, 1, 0]

    X, Y, Z, K = 4, 5, 6, 3
    fixel_img = np.zeros((X, Y, Z, K, 3), dtype=np.float64)
    # Peak 0: [1, 0, 0] -> maps to [0, 1, 0]
    fixel_img[..., 0, :] = [1.0, 0.0, 0.0]
    # Peak 1: [0, 1, 0] -> maps to [0, 0, -1]
    fixel_img[..., 1, :] = [0.0, 1.0, 0.0]
    # Peak 2: [0, 0, 1] -> maps to [-1, 0, 0]
    fixel_img[..., 2, :] = [0.0, 0.0, 1.0]

    fixel_lps = image_to_lps(fixel_img, D_pir, kind="vector")
    assert fixel_lps.shape == (X, Y, Z, K, 3)

    # Check peak 0 maps [1, 0, 0] to [0, 1, 0]
    expected_p0 = np.broadcast_to(np.array([0.0, 1.0, 0.0]), (X, Y, Z, 3))
    np.testing.assert_allclose(fixel_lps[..., 0, :], expected_p0, atol=1e-12)

    # Check peak 1 maps [0, 1, 0] to [0, 0, -1]
    expected_p1 = np.broadcast_to(np.array([0.0, 0.0, -1.0]), (X, Y, Z, 3))
    np.testing.assert_allclose(fixel_lps[..., 1, :], expected_p1, atol=1e-12)

    # Check peak 2 maps [0, 0, 1] to [-1, 0, 0]
    expected_p2 = np.broadcast_to(np.array([-1.0, 0.0, 0.0]), (X, Y, Z, 3))
    np.testing.assert_allclose(fixel_lps[..., 2, :], expected_p2, atol=1e-12)

    # None of the peaks should be corrupted to [0, 0, 0]
    assert not np.allclose(fixel_lps[0, 0, 0, 0], [0.0, 0.0, 0.0])
    assert not np.allclose(fixel_lps[0, 0, 0, 1], [0.0, 0.0, 0.0])
    assert not np.allclose(fixel_lps[0, 0, 0, 2], [0.0, 0.0, 0.0])

    # Test roundtrip
    fixel_back = lps_to_image(fixel_lps, D_pir, kind="vector")
    np.testing.assert_allclose(fixel_back, fixel_img, atol=1e-12)


def test_image_to_lps_z3_vector_field():
    """Verify vector field with Z=3 slices (X, Y, 3, 3) is treated as vectors, not tensor."""
    refs = reference_directions()
    D_pir = refs["PIR"]
    rng = np.random.default_rng(999)
    # Shape where slice dimension Z == 3 and vector dimension is 3
    X, Y, Z = 5, 7, 3
    v_field = rng.standard_normal((X, Y, Z, 3))

    v_lps = image_to_lps(v_field, D_pir, kind="vector")
    assert v_lps.shape == (X, Y, Z, 3)

    # Verify each vector is rotated by D_pir: v_lps = D @ v_img
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                expected = D_pir @ v_field[x, y, z]
                np.testing.assert_allclose(v_lps[x, y, z], expected, atol=1e-12)

    v_back = lps_to_image(v_lps, D_pir, kind="vector")
    np.testing.assert_allclose(v_back, v_field, atol=1e-12)


def test_image_to_lps_invalid_kind():
    """Verify ValueError is raised when kind is invalid."""
    D = np.eye(3)
    v = np.zeros((3,))
    with pytest.raises(ValueError, match="kind must be 'vector' or 'tensor'"):
        image_to_lps(v, D, kind="scalar")
    with pytest.raises(ValueError, match="kind must be 'vector' or 'tensor'"):
        lps_to_image(v, D, kind="scalar")


def test_dec_rgb_anatomical_numpy_requires_direction():
    """Verify dec_rgb(..., frame='anatomical') raises ValueError when no direction is provided for NumPy input."""
    v_arr = np.zeros((4, 4, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="requires an image direction matrix; none was provided"):
        dec_rgb(v_arr, frame="anatomical")


def test_dec_rgb_anatomical_numpy_with_direction():
    """Verify dec_rgb(..., frame='anatomical', direction=D) succeeds for NumPy input and preserves geometry."""
    refs = reference_directions()
    D = refs["RAS"]
    v_arr = np.zeros((4, 4, 4, 3), dtype=np.float32)
    v_arr[:] = [1.0, 0.0, 0.0]

    origin = (10.0, 20.0, 30.0)
    spacing = (1.5, 1.5, 2.0)
    out = dec_rgb(
        v_arr,
        frame="anatomical",
        direction=D,
        origin=origin,
        spacing=spacing,
    )
    assert ants.is_image(out)
    assert out.has_components is True
    assert out.components == 3
    np.testing.assert_allclose(out.direction, D)
    np.testing.assert_allclose(out.origin, origin)
    np.testing.assert_allclose(out.spacing, spacing)

    # In RAS: D = diag(-1, -1, 1). For v = [1, 0, 0], v_lps = [-1, 0, 0], |v_lps| = [1, 0, 0] (Red)
    np.testing.assert_allclose(out.numpy()[0, 0, 0], [1.0, 0.0, 0.0], atol=1e-6)
