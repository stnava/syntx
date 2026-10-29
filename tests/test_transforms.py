"""Deep, near-instantaneous correctness test suite for ``syntx.transforms``.

Tests ``apply_transforms``, ``TransformChain.apply``, ``map_points_to_moving``,
``resample_series``, ``ContinuousField``, and Jacobian helpers on tiny (6x7x8 to
12x13x14) grids against ``ants.apply_transforms`` across:
  - Diverse physical-space headers: canonical Mindboggle ``mbhard`` LAS vs RPS
    frames, LPI/RAI/ASR permutations, oblique 3D rotation matrices, anisotropic
    spacings, non-cubic shapes, and shifted origins.
  - All ITK linear transform classes (``AffineTransform``, ``Euler3DTransform``,
    ``Similarity3DTransform``, ``CenteredAffineTransform``) with non-zero center
    of rotation (fixed parameters) and ``whichtoinvert=True/False``.
  - Multi-stage transform concatenations: ``[warp, affine]``,
    ``[affine_inv, inv_warp]``, 4-element ``[warp_anat, aff_anat, warp_sdc, rigid_mot]``
    with displacement fields on distinct grids, partial-FOV warps, and ``"identity"``
    elements.
  - Interpolators and exact ITK ``ResampleImageFilter`` boundary semantics:
    ``[-0.5, N - 0.5)`` half-open buffer interval, ``RoundHalfIntegerUp`` for
    ``nearestNeighbor``, and 8-corner trilinear voting for ``genericLabel``.
"""

from typing import Sequence

import numpy as np
import pandas as pd
import pytest
import torch

from syntx.transforms import (
    ContinuousField,
    TransformChain,
    apply_ants_transform_to_point,
    apply_ants_transform_to_vector,
    apply_transforms,
    apply_transforms_to_points,
    classify_transform,
    compose_displacement,
    compose_displacement_fields,
    create_jacobian_determinant_image,
    jacobian_modulation,
    map_points_to_moving,
    reorient_tensors,
    reorient_vectors,
    resample_image_to_target,
    resample_series,
    series_rebase,
    signed_jacobian_determinant,
    transform_index_to_physical_point,
    transform_physical_point_to_index,
    transform_streamlines,
)

ants = pytest.importorskip("ants")


# ---------------------------------------------------------------------------
# Tiny geometry & transform helpers
# ---------------------------------------------------------------------------

def _rot_matrix(ax_deg: float, ay_deg: float, az_deg: float) -> np.ndarray:
    rx, ry, rz = np.deg2rad([ax_deg, ay_deg, az_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return (Rz @ Ry @ Rx).astype(np.float64)


# Canonical direction matrices covering LPS, RAS (LPI), LAS (mbhard fixed),
# RPS (mbhard moving), ASR (sagittal), and two oblique clinical orientations.
SPACE_CONFIGS: dict[str, dict] = {
    "lps_iso": {
        "shape": (8, 9, 10),
        "spacing": (1.0, 1.0, 1.0),
        "origin": (-4.0, -4.5, -5.0),
        "direction": np.eye(3),
    },
    "mbhard_fixed_las": {
        # Mirrors NKI-TRT-20-2 (LAS array, 1.0 mm isotropic, offset origin)
        "shape": (9, 11, 10),
        "spacing": (1.0, 1.0, 1.0),
        "origin": (-90.0, 126.0, -72.0),
        "direction": np.diag([1.0, -1.0, 1.0]),
    },
    "mbhard_moving_rps": {
        # Mirrors MMRR-21-2 (RPS array, 1.2 x 1.0 x 1.0 mm, distinct origin)
        "shape": (8, 10, 11),
        "spacing": (1.2, 1.0, 1.0),
        "origin": (95.0, -130.0, -68.0),
        "direction": np.diag([-1.0, 1.0, 1.0]),
    },
    "ras_aniso": {
        "shape": (10, 8, 9),
        "spacing": (0.8, 1.3, 2.1),
        "origin": (12.3, -24.5, 8.7),
        "direction": np.diag([-1.0, -1.0, 1.0]),
    },
    "asr_sagittal": {
        "shape": (8, 10, 9),
        "spacing": (1.1, 0.9, 1.5),
        "origin": (-15.0, 20.0, -10.0),
        "direction": np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    },
    "oblique_a": {
        "shape": (9, 10, 8),
        "spacing": (1.1, 1.4, 1.8),
        "origin": (-8.5, -11.2, 4.3),
        "direction": _rot_matrix(12.0, -8.0, 19.0),
    },
    "oblique_b": {
        "shape": (10, 9, 11),
        "spacing": (1.3, 0.95, 1.6),
        "origin": (6.0, -9.0, -12.0),
        "direction": _rot_matrix(-15.0, 22.0, -11.0),
    },
}


def _make_smooth_image(cfg: dict, seed: int = 0) -> ants.ANTsImage:
    rng = np.random.default_rng(seed)
    arr = rng.uniform(5.0, 100.0, size=cfg["shape"]).astype(np.float32)
    return ants.from_numpy(
        arr,
        origin=cfg["origin"],
        spacing=cfg["spacing"],
        direction=cfg["direction"],
    )


def _write_affine(
    path,
    matrix: np.ndarray,
    offset: Sequence[float] = (0.0, 0.0, 0.0),
    center: Sequence[float] = (0.0, 0.0, 0.0),
) -> str:
    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
    ants.set_ants_transform_fixed_parameters(tx, np.asarray(center, dtype=float))
    params = np.concatenate([np.asarray(matrix, float).ravel(), np.asarray(offset, float)])
    ants.set_ants_transform_parameters(tx, params)
    ants.write_transform(tx, str(path))
    return str(path)


def _write_warp(path, cfg: dict, scale_mm: float = 1.2, seed: int = 1) -> str:
    rng = np.random.default_rng(seed)
    w_arr = rng.normal(0.0, scale_mm, size=cfg["shape"] + (3,)).astype(np.float32)
    w_img = ants.from_numpy(
        w_arr,
        origin=cfg["origin"],
        spacing=cfg["spacing"],
        direction=cfg["direction"],
        has_components=True,
    )
    ants.image_write(w_img, str(path))
    return str(path)


# ---------------------------------------------------------------------------
# 1. Exact ITK boundary & interpolator parity tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("interpolator", ["linear", "nearestNeighbor"])
def test_exact_itk_half_open_boundary_and_half_integer_rounding(tmp_path, interpolator):
    """Verify ITK's half-open [-0.5, N - 0.5) buffer interval and RoundHalfIntegerUp."""
    mov = ants.from_numpy(
        np.arange(216, dtype=np.float32).reshape(6, 6, 6) + 1.0,
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
    )
    aff_id = _write_affine(tmp_path / "id.mat", np.eye(3))
    for x in [-0.6, -0.5, -0.49, -0.1, 0.0, 1.5, 2.5, 4.9, 5.0, 5.1, 5.49, 5.5, 5.6]:
        fix = ants.from_numpy(
            np.zeros((2, 2, 2), dtype=np.float32),
            origin=(x, 2.0, 2.0),
            spacing=(1.0, 1.0, 1.0),
        )
        expected = ants.apply_transforms(
            fixed=fix, moving=mov, transformlist=[aff_id], interpolator=interpolator
        ).numpy()
        actual = apply_transforms(
            fixed=fix, moving=mov, transformlist=[aff_id], interpolator=interpolator
        ).numpy()
        np.testing.assert_allclose(
            actual, expected, atol=1e-4, err_msg=f"Mismatch at x={x} for {interpolator}"
        )


def test_generic_label_matches_ants_across_noncontiguous_labels(tmp_path):
    """Verify 8-corner trilinear genericLabel voting matches ants.apply_transforms."""
    lbl_arr = np.zeros((10, 11, 12), dtype=np.float32)
    lbl_arr[1:5, 1:5, 1:6] = 10
    lbl_arr[5:9, 1:5, 1:6] = 25
    lbl_arr[1:5, 5:9, 1:6] = 1002
    lbl_arr[3:8, 3:8, 6:11] = 2035
    lbl = ants.from_numpy(
        lbl_arr,
        origin=(-5.0, -6.0, -7.0),
        spacing=(1.2, 1.1, 1.3),
        direction=_rot_matrix(8.0, -5.0, 12.0),
    )
    fix = ants.from_numpy(
        np.zeros((9, 10, 11), dtype=np.float32),
        origin=(-4.5, -5.5, -6.5),
        spacing=(1.1, 1.2, 1.25),
        direction=_rot_matrix(-4.0, 7.0, -6.0),
    )
    aff = _write_affine(
        tmp_path / "rot.mat",
        _rot_matrix(10.0, -6.0, 14.0),
        offset=(0.4, -0.3, 0.2),
        center=(1.0, -1.0, 0.5),
    )
    expected = ants.apply_transforms(
        fixed=fix, moving=lbl, transformlist=[aff], interpolator="genericLabel"
    ).numpy()
    actual = apply_transforms(
        fixed=fix, moving=lbl, transformlist=[aff], interpolator="genericLabel"
    ).numpy()
    assert np.array_equal(actual, expected)


# ---------------------------------------------------------------------------
# 2. All ITK linear transform classes with non-zero center & inversion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "ttype,params",
    [
        (
            "AffineTransform",
            np.concatenate([(_rot_matrix(14, -9, 22) @ np.diag([1.05, 0.95, 1.02])).ravel(), [2.3, -1.8, 3.1]]),
        ),
        ("Euler3DTransform", np.array([0.18, -0.22, 0.14, 1.9, -2.4, 0.8])),
        ("Similarity3DTransform", np.array([0.12, -0.15, 0.20, -1.5, 2.1, -0.9, 1.08])),
        (
            "CenteredAffineTransform",
            np.concatenate([_rot_matrix(-11, 16, -8).ravel(), [4.0, -3.0, 2.0], [1.5, -0.7, 2.2]]),
        ),
    ],
)
@pytest.mark.parametrize("invert", [False, True])
def test_itk_linear_transform_types_with_nonzero_center(tmp_path, ttype, params, invert):
    fix = _make_smooth_image(SPACE_CONFIGS["oblique_a"], seed=10)
    mov = _make_smooth_image(SPACE_CONFIGS["oblique_b"], seed=11)

    tx = ants.create_ants_transform(transform_type=ttype, dimension=3)
    fp = ants.get_ants_transform_fixed_parameters(tx)
    if len(fp) >= 3:
        ants.set_ants_transform_fixed_parameters(tx, [7.5, -4.2, 3.8] + list(fp[3:]))
    ants.set_ants_transform_parameters(tx, params)
    tx_path = str(tmp_path / f"{ttype}.mat")
    ants.write_transform(tx, tx_path)

    expected = ants.apply_transforms(
        fixed=fix, moving=mov, transformlist=[tx_path], whichtoinvert=[invert], interpolator="linear"
    ).numpy()
    actual = apply_transforms(
        fixed=fix, moving=mov, transformlist=[tx_path], whichtoinvert=[invert], interpolator="linear"
    ).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-4)

    # Also verify TransformChain.linear_part() matches the 3x3 linear matrix
    chain = TransformChain.from_paths([tx_path], whichtoinvert=[invert])
    lin = chain.linear_part()
    assert lin.shape == (3, 3) and np.all(np.isfinite(lin))


# ---------------------------------------------------------------------------
# 3. Cross-space pairs (including mbhard LAS <-> RPS & oblique spaces)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fix_key,mov_key",
    [
        ("mbhard_fixed_las", "mbhard_moving_rps"),
        ("mbhard_moving_rps", "mbhard_fixed_las"),
        ("lps_iso", "ras_aniso"),
        ("ras_aniso", "asr_sagittal"),
        ("oblique_a", "oblique_b"),
        ("asr_sagittal", "oblique_a"),
    ],
)
def test_cross_space_warp_and_affine_parity(tmp_path, fix_key, mov_key):
    """Test [warp, affine] and [affine_inv, inv_warp] between radically different spaces."""
    fix_cfg = SPACE_CONFIGS[fix_key]
    mov_cfg = SPACE_CONFIGS[mov_key]
    fix = _make_smooth_image(fix_cfg, seed=21)
    mov = _make_smooth_image(mov_cfg, seed=22)

    # Build an affine that maps fixed center of mass to moving center of mass + rotation
    c_fix = np.asarray(ants.get_center_of_mass(fix), dtype=float)
    c_mov = np.asarray(ants.get_center_of_mass(mov), dtype=float)
    R = _rot_matrix(9.0, -7.0, 13.0)
    # y = R(x - c_fix) + c_mov = R(x - c_fix) + c_fix + (c_mov - c_fix)
    aff_path = _write_affine(tmp_path / "0GenericAffine.mat", R, offset=(c_mov - c_fix), center=c_fix)
    warp_path = _write_warp(tmp_path / "1Warp.nii.gz", fix_cfg, scale_mm=1.0, seed=23)
    inv_warp_path = _write_warp(tmp_path / "1InverseWarp.nii.gz", mov_cfg, scale_mm=1.0, seed=24)

    # Forward chain: [1Warp, 0GenericAffine] on fixed grid
    fwd_tl = [warp_path, aff_path]
    for interp in ("linear", "nearestNeighbor"):
        exp_fwd = ants.apply_transforms(
            fixed=fix, moving=mov, transformlist=fwd_tl, interpolator=interp
        ).numpy()
        act_fwd = apply_transforms(
            fixed=fix, moving=mov, transformlist=fwd_tl, interpolator=interp
        ).numpy()
        np.testing.assert_allclose(act_fwd, exp_fwd, rtol=1e-4, atol=2.5e-3)

    # Inverse chain: [0GenericAffine (inverted), 1InverseWarp] on moving grid
    inv_tl = [aff_path, inv_warp_path]
    inv_flags = [True, False]
    exp_inv = ants.apply_transforms(
        fixed=mov, moving=fix, transformlist=inv_tl, whichtoinvert=inv_flags, interpolator="linear"
    ).numpy()
    act_inv = apply_transforms(
        fixed=mov, moving=fix, transformlist=inv_tl, whichtoinvert=inv_flags, interpolator="linear"
    ).numpy()
    np.testing.assert_allclose(act_inv, exp_inv, rtol=1e-4, atol=2.5e-3)


# ---------------------------------------------------------------------------
# 4. Complex multi-stage transform concatenations & partial-FOV warps
# ---------------------------------------------------------------------------

def test_four_stage_multimodal_chain_across_three_distinct_grids(tmp_path):
    """Simulate full multimodal ledger chain: [warp_template, aff_template, warp_sdc, rigid_motion]
    where template, anatomical, and EPI grids all have different orientations, spacings, and origins.
    """
    tmpl_cfg = SPACE_CONFIGS["lps_iso"]
    anat_cfg = SPACE_CONFIGS["oblique_a"]
    epi_cfg = SPACE_CONFIGS["oblique_b"]

    tmpl_img = _make_smooth_image(tmpl_cfg, seed=30)
    epi_img = _make_smooth_image(epi_cfg, seed=31)

    c_tmpl = np.asarray(ants.get_center_of_mass(tmpl_img), dtype=float)
    c_anat = np.asarray(ants.get_center_of_mass(_make_smooth_image(anat_cfg, seed=32)), dtype=float)
    c_epi = np.asarray(ants.get_center_of_mass(epi_img), dtype=float)

    warp_tmpl = _write_warp(tmp_path / "tmpl_1Warp.nii.gz", tmpl_cfg, scale_mm=0.8, seed=33)
    aff_tmpl = _write_affine(
        tmp_path / "tmpl_0GenericAffine.mat",
        _rot_matrix(6.0, -5.0, 8.0),
        offset=(c_anat - c_tmpl),
        center=c_tmpl,
    )
    warp_sdc = _write_warp(tmp_path / "sdc_1Warp.nii.gz", anat_cfg, scale_mm=0.9, seed=34)
    rigid_mot = _write_affine(
        tmp_path / "mot_0GenericAffine.mat",
        _rot_matrix(-4.0, 7.0, -6.0),
        offset=(c_epi - c_anat),
        center=c_anat,
    )

    chain_paths = [warp_tmpl, aff_tmpl, "identity", warp_sdc, rigid_mot]
    chain_flags = [False, False, False, False, False]

    # Note: ants.apply_transforms accepts "identity" as well
    expected = ants.apply_transforms(
        fixed=tmpl_img,
        moving=epi_img,
        transformlist=[warp_tmpl, aff_tmpl, warp_sdc, rigid_mot],
        whichtoinvert=[False, False, False, False],
        interpolator="linear",
    ).numpy()
    actual = apply_transforms(
        fixed=tmpl_img,
        moving=epi_img,
        transformlist=chain_paths,
        whichtoinvert=chain_flags,
        interpolator="linear",
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=5e-4)


def test_partial_fov_displacement_field_zero_outside_buffer(tmp_path):
    """When fixed grid extends outside a displacement field's domain, points outside
    [-0.5, N-0.5) of the displacement field receive zero displacement (matching ITK)."""
    rng = np.random.default_rng(123)
    mov = ants.from_numpy(
        rng.uniform(1.0, 10.0, (10, 11, 12)).astype(np.float32),
        origin=(-10.0, -12.0, -14.0),
        spacing=(2.0, 2.0, 2.0),
    )
    fix = ants.from_numpy(
        np.zeros((12, 13, 14), dtype=np.float32),
        origin=(-15.0, -15.0, -15.0),
        spacing=(2.0, 2.0, 2.0),
    )
    w_arr = rng.normal(0.0, 1.5, (6, 6, 6, 3)).astype(np.float32)
    w_img = ants.from_numpy(w_arr, origin=(-5.0, -5.0, -5.0), spacing=(2.0, 2.0, 2.0), has_components=True)
    w_path = str(tmp_path / "partial_warp.nii.gz")
    ants.image_write(w_img, w_path)

    expected = ants.apply_transforms(fixed=fix, moving=mov, transformlist=[w_path], interpolator="linear").numpy()
    actual = apply_transforms(fixed=fix, moving=mov, transformlist=[w_path], interpolator="linear").numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# 5. Point mapping (map_points_to_moving) vs coordinate ramp resampling
# ---------------------------------------------------------------------------

def test_map_points_to_moving_matches_coordinate_ramps(tmp_path):
    """Verify map_points_to_moving(fixed_points, chain) equals the physical coordinates
    sampled by ants.apply_transforms on linear coordinate ramp images X, Y, Z."""
    cfg = SPACE_CONFIGS["oblique_a"]
    # Build a moving grid large enough that all mapped points fall strictly inside its interior [0, N-1]
    mov_cfg = {
        "shape": (20, 20, 20),
        "spacing": (2.5, 2.5, 2.5),
        "origin": (-20.0, -20.0, -20.0),
        "direction": _rot_matrix(-10.0, 8.0, -5.0),
    }
    ii, jj, kk = np.meshgrid(
        np.arange(20, dtype=np.float64),
        np.arange(20, dtype=np.float64),
        np.arange(20, dtype=np.float64),
        indexing="ij",
    )
    idx = np.stack([ii, jj, kk], axis=-1)
    m_i2p = mov_cfg["direction"] @ np.diag(mov_cfg["spacing"])
    phys_grid = idx @ m_i2p.T + np.asarray(mov_cfg["origin"])

    fix_cfg = {
        "shape": (6, 6, 6),
        "spacing": (1.2, 1.2, 1.2),
        "origin": (-3.0, -3.0, -3.0),
        "direction": cfg["direction"],
    }
    fix = ants.from_numpy(
        np.zeros(fix_cfg["shape"], dtype=np.float32),
        origin=fix_cfg["origin"],
        spacing=fix_cfg["spacing"],
        direction=fix_cfg["direction"],
    )

    aff1 = _write_affine(tmp_path / "a1.mat", _rot_matrix(5.0, -4.0, 6.0), offset=(0.5, -0.4, 0.3), center=(1.0, 1.0, 1.0))
    warp = _write_warp(tmp_path / "w.nii.gz", fix_cfg, scale_mm=0.6, seed=55)
    aff2 = _write_affine(tmp_path / "a2.mat", _rot_matrix(-3.0, 5.0, -2.0), offset=(-0.2, 0.3, -0.1), center=(-1.0, 0.5, 0.0))
    chain = TransformChain.from_paths([warp, aff1, aff2], whichtoinvert=[False, False, True])

    # Fixed grid points in LPS mm
    f_ii, f_jj, f_kk = np.meshgrid(
        np.arange(6, dtype=np.float64),
        np.arange(6, dtype=np.float64),
        np.arange(6, dtype=np.float64),
        indexing="ij",
    )
    f_idx = np.stack([f_ii, f_jj, f_kk], axis=-1).reshape(-1, 3)
    f_pts = f_idx @ (fix_cfg["direction"] @ np.diag(fix_cfg["spacing"])).T + np.asarray(fix_cfg["origin"])

    mapped_pts = map_points_to_moving(f_pts, chain).reshape(6, 6, 6, 3)

    for axis in range(3):
        ramp_img = ants.from_numpy(
            phys_grid[..., axis].astype(np.float32),
            origin=mov_cfg["origin"],
            spacing=mov_cfg["spacing"],
            direction=mov_cfg["direction"],
        )
        warped_ramp = ants.apply_transforms(
            fixed=fix,
            moving=ramp_img,
            transformlist=chain.paths(),
            whichtoinvert=chain.whichtoinvert(),
            interpolator="linear",
        ).numpy()
        np.testing.assert_allclose(mapped_pts[..., axis], warped_ramp, atol=1e-4)


# ---------------------------------------------------------------------------
# 6. 4D series resampling & Jacobian modulation parity
# ---------------------------------------------------------------------------

def test_resample_series_matches_per_volume_ants_across_oblique_spaces(tmp_path):
    """Verify resample_series matches per-volume ants.apply_transforms + jacobian_modulation."""
    rng = np.random.default_rng(77)
    epi_dir = _rot_matrix(11.0, -7.0, 15.0)
    ref_dir = _rot_matrix(-6.0, 9.0, -12.0)

    arr4d = rng.uniform(10.0, 100.0, size=(8, 9, 10, 4)).astype(np.float32)
    dir4d = np.eye(4)
    dir4d[:3, :3] = epi_dir
    img4d = ants.from_numpy(
        arr4d,
        origin=(-8.0, -9.0, -10.0, 0.0),
        spacing=(2.0, 1.8, 2.2, 1.5),
        direction=dir4d,
    )
    ref = ants.from_numpy(
        np.zeros((9, 8, 10), dtype=np.float32),
        origin=(-7.5, -8.5, -9.5),
        spacing=(1.9, 2.0, 2.0),
        direction=ref_dir,
    )

    # Compose a non-trivial susceptibility + anatomical displacement field on ref
    aff_anat = _write_affine(tmp_path / "anat.mat", _rot_matrix(4.0, -3.0, 5.0), offset=(0.3, -0.2, 0.4))
    warp_sdc = _write_warp(
        tmp_path / "sdc.nii.gz",
        {"shape": (9, 8, 10), "spacing": (1.9, 2.0, 2.0), "origin": (-7.5, -8.5, -9.5), "direction": ref_dir},
        scale_mm=0.5,
        seed=88,
    )
    field_path = compose_displacement([warp_sdc, aff_anat], [False, False], ref, str(tmp_path / "composed_field.nii.gz"))

    mot1 = _write_affine(tmp_path / "m1.mat", _rot_matrix(3.0, 0.0, -2.0), offset=(0.4, -0.2, 0.1), center=(1.0, -1.0, 0.0))
    mot2 = _write_affine(tmp_path / "m2.mat", _rot_matrix(-2.0, 4.0, 3.0), offset=(-0.3, 0.5, -0.2), center=(0.0, 2.0, -1.0))
    mot3 = _write_affine(tmp_path / "m3.mat", _rot_matrix(5.0, -3.0, 4.0), offset=(0.2, 0.1, -0.4))
    chains = [
        None,
        TransformChain.from_paths([mot1]),
        TransformChain.from_paths([mot2], whichtoinvert=[True]),
        TransformChain.from_paths([mot1, mot3], whichtoinvert=[False, True]),
    ]

    for use_jac in (False, True):
        actual_4d = resample_series(img4d, chains, field_path, ref, jacobian=use_jac, interpolator="linear")
        assert actual_4d.shape == (9, 8, 10, 4)
        np.testing.assert_allclose(actual_4d.spacing, (1.9, 2.0, 2.0, 1.5), atol=1e-5)
        np.testing.assert_allclose(actual_4d.origin, (-7.5, -8.5, -9.5, 0.0), atol=1e-5)
        np.testing.assert_allclose(np.asarray(actual_4d.direction)[:3, :3], ref_dir, atol=1e-5)

        jac = jacobian_modulation(field_path, ref).numpy() if use_jac else 1.0
        for k, ch in enumerate(chains):
            tl = [field_path] + (ch.paths() if ch is not None else [])
            inv = [False] + (ch.whichtoinvert() if ch is not None else [])
            vol = ants.slice_image(img4d, axis=3, idx=k)
            exp_vol = ants.apply_transforms(
                fixed=ref, moving=vol, transformlist=tl, whichtoinvert=inv, interpolator="linear"
            ).numpy() * jac
            np.testing.assert_allclose(actual_4d.numpy()[..., k], exp_vol, atol=1e-4)


# ---------------------------------------------------------------------------
# 7. CPU vs MPS determinism & 2D fallback
# ---------------------------------------------------------------------------

def test_cpu_vs_mps_consistency_and_2d_fallback(tmp_path):
    fix = _make_smooth_image(SPACE_CONFIGS["oblique_a"], seed=91)
    mov = _make_smooth_image(SPACE_CONFIGS["oblique_b"], seed=92)
    aff = _write_affine(tmp_path / "aff.mat", _rot_matrix(7.0, -5.0, 9.0), offset=(0.6, -0.4, 0.2))
    warp = _write_warp(tmp_path / "warp.nii.gz", SPACE_CONFIGS["oblique_a"], scale_mm=0.7, seed=93)

    out_cpu = apply_transforms(fixed=fix, moving=mov, transformlist=[warp, aff], device="cpu").numpy()
    if torch.backends.mps.is_available():
        out_mps = apply_transforms(fixed=fix, moving=mov, transformlist=[warp, aff], device="mps").numpy()
        np.testing.assert_allclose(out_mps, out_cpu, atol=1e-4)

    # 2D image fallback to ants.apply_transforms
    img2d = ants.from_numpy(np.ones((8, 8), dtype=np.float32))
    tx2d = ants.create_ants_transform(transform_type="AffineTransform", dimension=2)
    tx2d_path = str(tmp_path / "aff2d.mat")
    ants.write_transform(tx2d, tx2d_path)
    out2d = apply_transforms(fixed=img2d, moving=img2d, transformlist=[tx2d_path])
    assert out2d.shape == (8, 8)

    # ContinuousField & series_rebase sanity
    cf = ContinuousField(fix)
    assert cf.spatial_shape == fix.shape
    np.testing.assert_allclose(series_rebase(fix.direction, fix.direction), np.eye(3), atol=1e-7)
    det = signed_jacobian_determinant(warp, fix)
    assert det.shape == fix.shape


# ---------------------------------------------------------------------------
# 8. Vector image (imagetype=1) & Tensor image (imagetype=2) parity vs ANTs
# ---------------------------------------------------------------------------

def test_vector_imagetype1_parity_across_oblique_affine_and_warp(tmp_path):
    """Verify apply_transforms(..., imagetype=1) matches ants.apply_transforms(..., imagetype=1)
    with polar rotation reorientation across oblique headers, affines, and warps."""
    rng = np.random.default_rng(201)
    shape = (8, 8, 8)
    fix_cfg = {
        "shape": shape,
        "spacing": (1.2, 1.5, 1.8),
        "origin": (-4.0, -5.0, -6.0),
        "direction": _rot_matrix(12.0, -8.0, 15.0),
    }
    mov_cfg = {
        "shape": shape,
        "spacing": (1.1, 1.3, 1.6),
        "origin": (-3.5, -4.5, -5.5),
        "direction": _rot_matrix(-9.0, 14.0, -11.0),
    }
    fix = ants.from_numpy(
        np.zeros(shape, dtype=np.float32),
        origin=fix_cfg["origin"],
        spacing=fix_cfg["spacing"],
        direction=fix_cfg["direction"],
    )
    vec_mov = ants.from_numpy(
        rng.normal(0.0, 1.0, shape + (3,)).astype(np.float32),
        origin=mov_cfg["origin"],
        spacing=mov_cfg["spacing"],
        direction=mov_cfg["direction"],
        has_components=True,
    )
    M = _rot_matrix(18.0, -12.0, 25.0) @ np.diag([1.08, 0.94, 1.03])
    aff = _write_affine(tmp_path / "vec_aff.mat", M, offset=(0.3, -0.2, 0.4), center=(1.0, -1.0, 0.5))
    warp = _write_warp(tmp_path / "vec_warp.nii.gz", fix_cfg, scale_mm=0.25, seed=202)

    for tl in ([aff], [warp], [warp, aff]):
        expected = ants.apply_transforms(
            fixed=fix, moving=vec_mov, transformlist=tl, imagetype=1, interpolator="linear"
        ).numpy()
        actual = apply_transforms(
            fixed=fix, moving=vec_mov, transformlist=tl, imagetype=1, interpolator="linear"
        ).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=5e-4)

    # Also verify non-matching grid shapes work cleanly (where ANTsPy C++ wrapper fails)
    fix_diff = ants.from_numpy(
        np.zeros((7, 9, 10), dtype=np.float32),
        origin=fix_cfg["origin"],
        spacing=fix_cfg["spacing"],
        direction=fix_cfg["direction"],
    )
    out_diff = apply_transforms(fixed=fix_diff, moving=vec_mov, transformlist=[warp, aff], imagetype=1)
    assert out_diff.shape == (7, 9, 10) and out_diff.components == 3


def test_tensor_imagetype2_log_euclidean_ppd_parity_vs_ants(tmp_path):
    """Verify apply_transforms(..., imagetype=2) matches ants.apply_transforms(..., imagetype=2)
    with Log-Euclidean interpolation and Preservation of Principal Directions (PPD)."""
    rng = np.random.default_rng(301)
    shape = (8, 8, 8)
    fix_cfg = {
        "shape": shape,
        "spacing": (1.2, 1.5, 1.8),
        "origin": (-4.0, -5.0, -6.0),
        "direction": _rot_matrix(12.0, -8.0, 15.0),
    }
    mov_cfg = {
        "shape": shape,
        "spacing": (1.1, 1.3, 1.6),
        "origin": (-3.5, -4.5, -5.5),
        "direction": _rot_matrix(-9.0, 14.0, -11.0),
    }
    fix = ants.from_numpy(
        np.zeros(shape, dtype=np.float32),
        origin=fix_cfg["origin"],
        spacing=fix_cfg["spacing"],
        direction=fix_cfg["direction"],
    )
    # Build random SPD 6-component tensors [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]
    # Set D[0, 0, :] = I so log(D[0, 0, :]) == 0 (avoiding ANTs C++ LogTensorImageFilter's
    # uninitialized (0, 0, k) iterator bug while comparing the entire 8x8x8 volume)
    A = rng.normal(0.0, 0.3, shape + (3, 3))
    D = np.einsum("...ij,...kj->...ik", A, A) + 0.5 * np.eye(3)
    D[0, 0, :] = np.eye(3)
    dt6 = np.stack(
        [D[..., 0, 0], D[..., 0, 1], D[..., 0, 2], D[..., 1, 1], D[..., 1, 2], D[..., 2, 2]],
        axis=-1,
    ).astype(np.float32)
    dt_mov = ants.from_numpy(
        dt6,
        origin=mov_cfg["origin"],
        spacing=mov_cfg["spacing"],
        direction=mov_cfg["direction"],
        has_components=True,
    )
    M = _rot_matrix(18.0, -12.0, 25.0) @ np.diag([1.08, 0.94, 1.03])
    aff = _write_affine(tmp_path / "dt_aff.mat", M, offset=(0.3, -0.2, 0.4), center=(1.0, -1.0, 0.5))
    warp = _write_warp(tmp_path / "dt_warp.nii.gz", fix_cfg, scale_mm=0.25, seed=302)

    for tl in ([aff], [warp], [warp, aff]):
        expected = ants.apply_transforms(
            fixed=fix, moving=dt_mov, transformlist=tl, imagetype=2, interpolator="linear"
        ).numpy()
        actual = apply_transforms(
            fixed=fix, moving=dt_mov, transformlist=tl, imagetype=2, interpolator="linear"
        ).numpy()
        # Exclude (0, 0, k) where ANTs C++ ExpTensorImageFilter leaves output uninitialized
        np.testing.assert_allclose(actual[1:, 1:, 1:], expected[1:, 1:, 1:], rtol=1e-4, atol=5e-4)


def test_timeseries_imagetype3_and_compose_parity_vs_ants(tmp_path):
    """Verify imagetype=3 (4D time series) and compose=... against ants.apply_transforms."""
    rng = np.random.default_rng(401)
    fix_cfg = SPACE_CONFIGS["oblique_a"]
    mov_cfg = SPACE_CONFIGS["oblique_b"]
    fix = _make_smooth_image(fix_cfg, seed=402)

    arr4d = rng.uniform(5.0, 50.0, size=mov_cfg["shape"] + (3,)).astype(np.float32)
    dir4d = np.eye(4)
    dir4d[:3, :3] = mov_cfg["direction"]
    mov4d = ants.from_numpy(
        arr4d,
        origin=tuple(mov_cfg["origin"]) + (0.0,),
        spacing=tuple(mov_cfg["spacing"]) + (2.0,),
        direction=dir4d,
    )
    aff = _write_affine(tmp_path / "ts_aff.mat", _rot_matrix(8.0, -6.0, 10.0), offset=(0.5, -0.3, 0.2))
    warp = _write_warp(tmp_path / "ts_warp.nii.gz", fix_cfg, scale_mm=0.6, seed=403)

    exp_ts = ants.apply_transforms(
        fixed=fix, moving=mov4d, transformlist=[warp, aff], imagetype=3, interpolator="linear"
    ).numpy()
    act_ts = apply_transforms(
        fixed=fix, moving=mov4d, transformlist=[warp, aff], imagetype=3, interpolator="linear"
    ).numpy()
    np.testing.assert_allclose(act_ts, exp_ts, rtol=1e-4, atol=5e-4)

    # Test compose=prefix parity vs ants.apply_transforms(..., compose=...)
    exp_comp_path = ants.apply_transforms(
        fixed=fix, moving=fix, transformlist=[warp, aff], compose=str(tmp_path / "ants_comp_")
    )
    act_comp_path = apply_transforms(
        fixed=fix, moving=fix, transformlist=[warp, aff], compose=str(tmp_path / "syntx_comp_")
    )
    exp_comp = ants.image_read(exp_comp_path).numpy()
    act_comp = ants.image_read(act_comp_path).numpy()
    np.testing.assert_allclose(act_comp, exp_comp, rtol=1e-4, atol=5e-4)

    # Test compose_displacement_fields parity vs ants.compose_displacement_fields
    w1 = ants.image_read(warp)
    w2 = ants.image_read(_write_warp(tmp_path / "ts_warp2.nii.gz", fix_cfg, scale_mm=0.5, seed=404))
    exp_cdf = ants.compose_displacement_fields(w1, w2).numpy()
    act_cdf = compose_displacement_fields(w1, w2).numpy()
    np.testing.assert_allclose(act_cdf, exp_cdf, rtol=1e-4, atol=5e-4)


# ---------------------------------------------------------------------------
# 9. apply_transforms_to_points & streamline / reorientation helpers
# ---------------------------------------------------------------------------

def test_apply_transforms_to_points_matches_ants(tmp_path):
    """Verify apply_transforms_to_points matches ants.apply_transforms_to_points for
    DataFrame, ndarray, and torch.Tensor, including default whichtoinvert rules."""
    cfg = SPACE_CONFIGS["oblique_a"]
    aff = _write_affine(
        tmp_path / "pts_aff.mat",
        _rot_matrix(11.0, -7.0, 14.0),
        offset=(1.2, -0.8, 0.5),
        center=(0.5, -0.5, 1.0),
    )
    warp = _write_warp(tmp_path / "pts_warp.nii.gz", cfg, scale_mm=0.5, seed=501)

    pts_np = np.array(
        [[-3.0, -5.0, 8.0], [-2.0, -4.0, 9.0], [-1.0, -3.0, 10.0], [0.0, -2.0, 11.0]],
        dtype=np.float64,
    )
    df = pd.DataFrame(pts_np, columns=["x", "y", "z"])

    # Default whichtoinvert on [aff, warp] should invert aff (True) and not warp (False)
    exp_df = ants.apply_transforms_to_points(3, df, [aff, warp])
    act_df = apply_transforms_to_points(3, df, [aff, warp])
    assert isinstance(act_df, pd.DataFrame)
    np.testing.assert_allclose(act_df[["x", "y", "z"]].to_numpy(), exp_df[["x", "y", "z"]].to_numpy(), atol=1e-4)

    # Explicit whichtoinvert with extra metadata column preserved
    df_meta = df.copy()
    df_meta["label"] = [1, 2, 3, 4]
    exp_df2 = ants.apply_transforms_to_points(3, df_meta, [warp, aff], whichtoinvert=[False, False])
    act_df2 = apply_transforms_to_points(3, df_meta, [warp, aff], whichtoinvert=[False, False])
    np.testing.assert_allclose(act_df2.to_numpy(), exp_df2.to_numpy(), atol=1e-4)

    # ndarray and torch.Tensor inputs preserve their types
    act_np = apply_transforms_to_points(3, pts_np, [warp, aff], whichtoinvert=[False, False])
    assert isinstance(act_np, np.ndarray)
    np.testing.assert_allclose(act_np, exp_df2[["x", "y", "z"]].to_numpy(), atol=1e-4)

    pts_t = torch.as_tensor(pts_np, dtype=torch.float32)
    act_t = apply_transforms_to_points(3, pts_t, [warp, aff], whichtoinvert=[False, False])
    assert isinstance(act_t, torch.Tensor)
    np.testing.assert_allclose(act_t.cpu().numpy(), exp_df2[["x", "y", "z"]].to_numpy(), atol=1e-4)


def test_transform_streamlines_and_standalone_reorient_helpers(tmp_path):
    """Verify transform_streamlines (with and without tangents), reorient_vectors, and reorient_tensors."""
    cfg = SPACE_CONFIGS["lps_iso"]
    R = _rot_matrix(20.0, -15.0, 30.0)
    aff = _write_affine(tmp_path / "sl_aff.mat", R, offset=(2.0, -1.0, 3.0))

    s1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    s2 = np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 4.0, 0.0]], dtype=np.float32)
    warped_sl, tangents = transform_streamlines([s1, s2], [aff], compute_tangents=True)
    assert len(warped_sl) == 2 and len(tangents) == 2
    assert warped_sl[0].shape == (3, 3) and warped_sl[1].shape == (4, 3)
    # Under forward mapping x' = R x + t, unit tangent along +x becomes R[:, 0]
    expected_t1 = (R @ np.array([1.0, 0.0, 0.0])).astype(np.float32)
    for row in tangents[0]:
        np.testing.assert_allclose(row, expected_t1, atol=1e-5)

    # Standalone reorient_vectors & reorient_tensors under pure rotation
    v = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    v_rot = reorient_vectors(v, R)
    np.testing.assert_allclose(v_rot[0], R[:, 0], atol=1e-5)
    np.testing.assert_allclose(v_rot[1], R[:, 1], atol=1e-5)

    # Prolate tensor with principal axis along +x (lambda1=3, lambda2=1, lambda3=1)
    # D = 2 * e1 e1^T + I -> in rotated frame D' = 2 * (R e1)(R e1)^T + I
    dt6 = np.array([[3.0, 0.0, 0.0, 1.0, 0.0, 1.0]], dtype=np.float32)
    dt6_rot = reorient_tensors(dt6, R)
    r0 = R[:, 0]
    D_exp = 2.0 * np.outer(r0, r0) + np.eye(3)
    exp6 = np.array(
        [[D_exp[0, 0], D_exp[0, 1], D_exp[0, 2], D_exp[1, 1], D_exp[1, 2], D_exp[2, 2]]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(dt6_rot, exp6, atol=1e-5)


def test_ants_geometric_and_transform_helper_parity(tmp_path):
    """Verify resample_image_to_target, create_jacobian_determinant_image,
    apply_ants_transform_to_point/vector, and transform_index_to_physical_point / physical_point_to_index."""
    fix = _make_smooth_image(SPACE_CONFIGS["oblique_a"], seed=601)
    mov = _make_smooth_image(SPACE_CONFIGS["oblique_b"], seed=602)

    # resample_image_to_target
    exp_res = ants.resample_image_to_target(mov, fix, interp_type="linear").numpy()
    act_res = resample_image_to_target(mov, fix, interp_type="linear").numpy()
    np.testing.assert_allclose(act_res, exp_res, rtol=1e-4, atol=5e-4)

    # transform_index_to_physical_point & transform_physical_point_to_index
    for idx in [(0, 0, 0), (3, 4, 2)]:
        pt_exp = ants.transform_index_to_physical_point(fix, idx)
        pt_act = transform_index_to_physical_point(fix, idx)
        np.testing.assert_allclose(pt_act, pt_exp, atol=1e-6)
        idx_exp = ants.transform_physical_point_to_index(fix, pt_exp)
        idx_act = transform_physical_point_to_index(fix, pt_act)
        np.testing.assert_allclose(idx_act, idx_exp, atol=1e-6)
    # Continuous float and batched (N, 3) index <-> physical round-trip
    c_idx = np.array([[1.5, 2.25, 3.75], [0.25, 4.5, 1.125]], dtype=np.float64)
    c_pts = transform_index_to_physical_point(fix, c_idx)
    np.testing.assert_allclose(transform_physical_point_to_index(fix, c_pts), c_idx, atol=1e-6)

    # apply_ants_transform_to_point & apply_ants_transform_to_vector
    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
    ants.set_ants_transform_fixed_parameters(tx, np.array([1.0, -2.0, 0.5]))
    params = np.concatenate([_rot_matrix(10.0, -5.0, 12.0).ravel(), [0.7, -0.4, 1.1]])
    ants.set_ants_transform_parameters(tx, params)
    tx_path = str(tmp_path / "helper_aff.mat")
    ants.write_transform(tx, tx_path)

    pt = (2.5, -1.5, 4.0)
    vec = (1.0, 2.0, -0.5)
    np.testing.assert_allclose(
        apply_ants_transform_to_point(tx, pt),
        ants.apply_ants_transform_to_point(tx, pt),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        apply_ants_transform_to_point(tx_path, pt),
        ants.apply_ants_transform_to_point(tx, pt),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        apply_ants_transform_to_vector(tx, vec),
        ants.apply_ants_transform_to_vector(tx, vec),
        atol=1e-6,
    )

    # create_jacobian_determinant_image
    warp = _write_warp(tmp_path / "jac_warp.nii.gz", SPACE_CONFIGS["lps_iso"], scale_mm=0.15, seed=603)
    dom = _make_smooth_image(SPACE_CONFIGS["lps_iso"], seed=604)
    jac_lin = create_jacobian_determinant_image(dom, warp, do_log=False)
    jac_log = create_jacobian_determinant_image(dom, warp, do_log=True)
    assert jac_lin.shape == dom.shape and jac_log.shape == dom.shape
    np.testing.assert_allclose(np.log(jac_lin.numpy()), jac_log.numpy(), atol=1e-5)


def test_auto_select_4d_batch_size_and_chunked_streaming(tmp_path, monkeypatch):
    """Verify _auto_select_4d_batch_size adapts to 3D volume size and device memory budget,
    and multi-chunk streaming in _apply_timeseries_4d and resample_series matches exact parity."""
    import syntx.transforms as tx_mod
    from syntx.transforms import _auto_select_4d_batch_size, resample_series

    dev = torch.device("cpu")
    b_small = _auto_select_4d_batch_size((16, 16, 16), (16, 16, 16), 64, dev, per_volume_grid=False)
    b_large = _auto_select_4d_batch_size((256, 256, 256), (256, 256, 256), 64, dev, per_volume_grid=False)
    b_pvg_small = _auto_select_4d_batch_size((16, 16, 16), (16, 16, 16), 64, dev, per_volume_grid=True)
    b_pvg_large = _auto_select_4d_batch_size((256, 256, 256), (256, 256, 256), 64, dev, per_volume_grid=True)
    assert b_small >= b_large >= 1
    assert b_pvg_small >= b_pvg_large >= 1
    assert b_pvg_large == 1  # 256^3 per-volume grid is >1 GB/vol, so must auto-clamp to 1

    # Force batch_sz = 2 across 5 volumes to exercise multi-chunk boundary streaming
    monkeypatch.setattr(tx_mod, "_auto_select_4d_batch_size", lambda *a, **kw: 2)
    rng = np.random.default_rng(701)
    fix_cfg = SPACE_CONFIGS["oblique_a"]
    fix = _make_smooth_image(fix_cfg, seed=702)
    arr4d = rng.uniform(1.0, 20.0, size=fix_cfg["shape"] + (5,)).astype(np.float32)
    dir4d = np.eye(4)
    dir4d[:3, :3] = fix_cfg["direction"]
    mov4d = ants.from_numpy(
        arr4d,
        origin=tuple(fix_cfg["origin"]) + (0.0,),
        spacing=tuple(fix_cfg["spacing"]) + (1.5,),
        direction=dir4d,
    )
    aff = _write_affine(tmp_path / "chunk_aff.mat", _rot_matrix(5.0, -4.0, 6.0), offset=(0.2, -0.1, 0.3))
    warp = _write_warp(tmp_path / "chunk_warp.nii.gz", fix_cfg, scale_mm=0.3, seed=703)

    exp_4d = ants.apply_transforms(fixed=fix, moving=mov4d, transformlist=[warp, aff], imagetype=3).numpy()
    act_4d = apply_transforms(fixed=fix, moving=mov4d, transformlist=[warp, aff], imagetype=3).numpy()
    np.testing.assert_allclose(act_4d, exp_4d, rtol=1e-4, atol=5e-4)

    chains = [TransformChain.from_paths([aff]) for _ in range(5)]
    res_b2 = resample_series(mov4d, chains, warp, fix, interpolator="linear").numpy()
    monkeypatch.setattr(tx_mod, "_auto_select_4d_batch_size", lambda *a, **kw: 1)
    res_b1 = resample_series(mov4d, chains, warp, fix, interpolator="linear").numpy()
    monkeypatch.setattr(tx_mod, "_auto_select_4d_batch_size", lambda *a, **kw: 5)
    res_b5 = resample_series(mov4d, chains, warp, fix, interpolator="linear").numpy()
    np.testing.assert_allclose(res_b2, res_b1, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_b2, res_b5, rtol=1e-5, atol=1e-5)


def test_bspline_and_fieldless_resample_series_parity_vs_ants(tmp_path):
    """Verify pure-PyTorch order-3 B-spline interpolation (3D, 4D imagetype=3, and resample_series)
    and field_path=None batched motion resampling match ants.apply_transforms."""
    rng = np.random.default_rng(801)
    fix_cfg = SPACE_CONFIGS["oblique_a"]
    mov_cfg = SPACE_CONFIGS["oblique_b"]
    fix = _make_smooth_image(fix_cfg, seed=802)
    mov = _make_smooth_image(mov_cfg, seed=803)
    aff1 = _write_affine(tmp_path / "bs_aff1.mat", _rot_matrix(6.0, -4.0, 7.0), offset=(0.35, -0.25, 0.15))
    aff2 = _write_affine(tmp_path / "bs_aff2.mat", _rot_matrix(-3.0, 5.0, -2.0), offset=(-0.2, 0.4, -0.3))
    warp = _write_warp(tmp_path / "bs_warp.nii.gz", fix_cfg, scale_mm=0.4, seed=804)

    # 3D scalar bSpline parity (affine + warp)
    exp_3d = ants.apply_transforms(fixed=fix, moving=mov, transformlist=[warp, aff1], interpolator="bSpline").numpy()
    act_3d = apply_transforms(fixed=fix, moving=mov, transformlist=[warp, aff1], interpolator="bSpline").numpy()
    np.testing.assert_allclose(act_3d, exp_3d, rtol=1e-4, atol=5e-4)

    # 4D shared-grid imagetype=3 bSpline parity
    arr4d = rng.uniform(2.0, 30.0, size=fix_cfg["shape"] + (4,)).astype(np.float32)
    dir4d = np.eye(4)
    dir4d[:3, :3] = fix_cfg["direction"]
    img4d = ants.from_numpy(
        arr4d,
        origin=tuple(fix_cfg["origin"]) + (0.0,),
        spacing=tuple(fix_cfg["spacing"]) + (1.2,),
        direction=dir4d,
    )
    exp_4d = ants.apply_transforms(
        fixed=fix, moving=img4d, transformlist=[warp, aff1], imagetype=3, interpolator="bSpline"
    ).numpy()
    act_4d = apply_transforms(
        fixed=fix, moving=img4d, transformlist=[warp, aff1], imagetype=3, interpolator="bSpline"
    ).numpy()
    np.testing.assert_allclose(act_4d, exp_4d, rtol=1e-4, atol=5e-4)

    # resample_series with field_path=None, volume_scales, identity frame bypass, linear & bSpline
    chains = [
        None,
        TransformChain.from_paths([aff1]),
        TransformChain.from_paths([aff2]),
        TransformChain.from_paths([aff1, aff2]),
    ]
    scales = np.array([1.0, 1.03, 0.97, 1.01], dtype=np.float64)
    for interp in ("linear", "bSpline"):
        res_4d = resample_series(
            img4d,
            chains,
            None,
            fix,
            jacobian=False,
            interpolator=interp,
            volume_scales=scales,
        ).numpy()
        # Volume 0 has chain=None on identical 3D grid -> exact bit-identity
        np.testing.assert_array_equal(res_4d[..., 0], arr4d[..., 0])
        for k in (1, 2, 3):
            vol_k = ants.slice_image(img4d, axis=3, idx=k)
            exp_k = (
                ants.apply_transforms(
                    fixed=fix, moving=vol_k, transformlist=chains[k].paths(), interpolator=interp
                ).numpy()
                * scales[k]
            )
            np.testing.assert_allclose(res_4d[..., k], exp_k, rtol=1e-4, atol=5e-4)
