"""Tests for exact 1-D physical directional projection ``u <- (u . v_PE) v_PE`` on oblique EPI headers.

Verifies across ``syntx.syn`` (``cfl`` and ``reg_adam``), ``syntx.greedy``, ``syntx.batch_greedy``,
``syntx.batch_syn``, and ``syntx.dewarp`` on known 1-D oblique distortion phantoms (32.2, 41.3, and
45.0 deg pitch about physical X):
  (a) ``||u - (u . v_PE) v_PE||_inf < 1e-5`` mm on exported ``1Warp.nii.gz`` and ``1InverseWarp.nii.gz``,
  (b) both world Y (AP) and world Z (SI) components of the oblique distortion are recovered
      (whereas legacy ``(0, 1, 0)`` leaves ``dz == 0`` and fails to recover the Z component),
  (c) ``0.0%`` Liouville Jacobian folding.
"""
from __future__ import annotations

import numpy as np
import pytest

ants = pytest.importorskip("ants")
torch = pytest.importorskip("torch")

import syntx
from syntx import dewarp
from syntx.phase_encode import (
    PhaseEncodeObliqueError,
    pe_to_dict,
    resolve_phase_encode,
    restrict_for_image,
    restrict_transformation,
)


def _pitch_direction_lps(pitch_deg: float) -> np.ndarray:
    """Rotation by ``pitch_deg`` about physical LPS +X (tilting voxel j between +Y and +Z)."""
    th = np.deg2rad(float(pitch_deg))
    c, s = float(np.cos(th)), float(np.sin(th))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def _make_oblique_pe_phantom(
    tmp_path,
    pitch_deg: float,
    *,
    pe_bids: str = "j-",
    shape: tuple[int, int, int] = (28, 32, 28),
    spacing: tuple[float, float, float] = (2.0, 2.0, 2.0),
    amp_mm: float = 4.2,
):
    """Create ``(fixed, moving, v_pe)`` with a smooth diffeomorphic 1-D distortion strictly along ``v_pe``."""
    direction = _pitch_direction_lps(pitch_deg)
    origin = tuple(-0.5 * s * (n - 1) for n, s in zip(shape, spacing))

    gx = np.linspace(-1.0, 1.0, shape[0], dtype=np.float32)[:, None, None]
    gy = np.linspace(-1.0, 1.0, shape[1], dtype=np.float32)[None, :, None]
    gz = np.linspace(-1.0, 1.0, shape[2], dtype=np.float32)[None, None, :]

    # Smooth asymmetric multi-structure phantom with rich 3-D intensity gradients
    r2_outer = (gx / 0.72) ** 2 + (gy / 0.78) ** 2 + (gz / 0.70) ** 2
    shell = np.exp(-2.2 * r2_outer) * (r2_outer < 1.15).astype(np.float32)
    lobe_a = 0.65 * np.exp(-14.0 * ((gx - 0.18) ** 2 + (gy + 0.15) ** 2 + (gz - 0.12) ** 2))
    lobe_b = 0.55 * np.exp(-16.0 * ((gx + 0.22) ** 2 + (gy - 0.20) ** 2 + (gz + 0.16) ** 2))
    frontal = 0.75 * np.exp(-18.0 * ((gx - 0.05) ** 2 + (gy + 0.34) ** 2 + (gz - 0.04) ** 2))
    vol = np.clip(shell + lobe_a + lobe_b + frontal, 0.0, None).astype(np.float32)
    vol /= float(vol.max() + 1e-8)

    fixed = ants.from_numpy(vol, origin=origin, spacing=spacing, direction=direction)
    pe = resolve_phase_encode(fixed, sidecar={"PhaseEncodingDirection": pe_bids})
    v_pe = np.asarray(restrict_transformation(pe), dtype=np.float64)

    # Smooth interior 1-D B0-like distortion strictly along v_pe (zero on all boundary faces)
    win_x = np.sin(np.pi * np.clip(0.5 * (gx + 1.0), 0.0, 1.0)) ** 2
    win_y = np.sin(np.pi * np.clip(0.5 * (gy + 1.0), 0.0, 1.0)) ** 2
    win_z = np.sin(np.pi * np.clip(0.5 * (gz + 1.0), 0.0, 1.0)) ** 2
    envelope = win_x * win_y * win_z
    bump = np.exp(-4.5 * ((gx - 0.04) ** 2 + (gy + 0.12) ** 2 + (gz - 0.05) ** 2))
    scalar_mm = (float(amp_mm) * envelope * bump).astype(np.float64)

    disp_arr = (scalar_mm[..., None] * v_pe[None, None, None, :]).astype(np.float32)
    gt_warp = ants.from_numpy(
        disp_arr,
        origin=origin,
        spacing=spacing,
        direction=direction,
        has_components=True,
    )
    gt_warp_path = str(tmp_path / f"gt_oblique_{pitch_deg:.1f}_warp.nii.gz")
    ants.image_write(gt_warp, gt_warp_path)
    moving = syntx.apply_transforms(fixed=fixed, moving=fixed, transformlist=[gt_warp_path])
    return fixed, moving, pe, tuple(float(x) for x in v_pe)


def _read_warp_xyz(path: str) -> np.ndarray:
    arr = ants.image_read(str(path)).numpy()
    return arr.reshape(*arr.shape[:3], -1)[..., :3].astype(np.float64)


def _orthogonal_leakage_linf_mm(disp_xyz: np.ndarray, v_pe: tuple[float, ...]) -> float:
    v = np.asarray(v_pe, dtype=np.float64)
    v = v / np.linalg.norm(v)
    proj = (disp_xyz @ v)[..., None] * v
    resid = disp_xyz - proj
    return float(np.max(np.abs(resid)))


def _folding_pct(reg: dict, fixed) -> float:
    jac = syntx.liouville_determinant(reg, fixed).numpy()
    return float(100.0 * np.mean(jac <= 0.0))


@pytest.mark.parametrize("pitch_deg", [32.2, 41.3, 45.0])
def test_oblique_pe_syn_cfl_projects_along_exact_unit_vector_and_beats_axis_010(tmp_path, pitch_deg):
    fixed, moving, pe, v_pe = _make_oblique_pe_phantom(tmp_path, pitch_deg)

    # Confirm oblique geometry matches ds005237 / 45-deg test regimes
    assert pe.alignment == pytest.approx(np.cos(np.deg2rad(pitch_deg)), rel=1e-5)
    assert abs(v_pe[0]) == 0.0
    assert v_pe[1] > 0.5 and abs(v_pe[2]) > 0.5

    reg_vec = syntx.syn(
        fixed,
        moving,
        type_of_transform="SyNOnly",
        initial_transform="identity",
        restrict_transformation=v_pe,
        levels=[2, 1],
        reg_iterations=[16, 10],
        device="cpu",
        seed=42,
        outprefix=str(tmp_path / f"syn_vec_{pitch_deg:.1f}_"),
        verbose=False,
    )
    fwd_path = next(p for p in reg_vec["fwdtransforms"] if str(p).endswith("1Warp.nii.gz"))
    inv_path = next(p for p in reg_vec["invtransforms"] if str(p).endswith("1InverseWarp.nii.gz"))
    u_fwd = _read_warp_xyz(fwd_path)
    u_inv = _read_warp_xyz(inv_path)

    # (a) Strict 1-D subspace confinement on exported warps: ||u - (u . v_PE) v_PE||_inf < 1e-5 mm
    assert _orthogonal_leakage_linf_mm(u_fwd, v_pe) < 1e-5
    assert _orthogonal_leakage_linf_mm(u_inv, v_pe) < 1e-5

    # (b) Both world Y (AP) and world Z (SI) components of the oblique distortion are recovered
    max_abs_vec = np.max(np.abs(u_fwd), axis=(0, 1, 2))
    assert max_abs_vec[0] == 0.0
    assert max_abs_vec[1] > 0.6, f"expected world Y recovery at {pitch_deg} deg, got {max_abs_vec}"
    assert max_abs_vec[2] > 0.5, f"expected world Z recovery at {pitch_deg} deg, got {max_abs_vec}"

    # (c) 0.0% Liouville Jacobian folding
    assert _folding_pct(reg_vec, fixed) == 0.0

    # Compare against legacy world-Y mask (0, 1, 0), which zeroes dz and under-corrects oblique distortion
    reg_010 = syntx.syn(
        fixed,
        moving,
        type_of_transform="SyNOnly",
        initial_transform="identity",
        restrict_transformation=(0.0, 1.0, 0.0),
        levels=[2, 1],
        reg_iterations=[16, 10],
        device="cpu",
        seed=42,
        outprefix=str(tmp_path / f"syn_010_{pitch_deg:.1f}_"),
        verbose=False,
    )
    u_010 = _read_warp_xyz(next(p for p in reg_010["fwdtransforms"] if str(p).endswith("1Warp.nii.gz")))
    assert float(np.max(np.abs(u_010[..., 2]))) == 0.0

    fix_np = fixed.numpy()
    mse_init = float(np.mean((moving.numpy() - fix_np) ** 2))
    mse_010 = float(np.mean((reg_010["warpedmovout"].numpy() - fix_np) ** 2))
    mse_vec = float(np.mean((reg_vec["warpedmovout"].numpy() - fix_np) ** 2))
    assert mse_vec < mse_init * 0.45
    assert mse_vec < mse_010, f"exact v_PE ({mse_vec:.6f}) must beat legacy (0,1,0) ({mse_010:.6f})"


def test_oblique_pe_syn_reg_adam_projects_along_unit_vector(tmp_path):
    fixed, moving, pe, v_pe = _make_oblique_pe_phantom(tmp_path, 41.3)
    reg = syntx.syn(
        fixed,
        moving,
        type_of_transform="SyNOnly",
        initial_transform="identity",
        optimizer="reg_adam",
        regularizer="sobolev",
        restrict_transformation=pe,  # direct PhaseEncode object accepted
        levels=[2, 1],
        reg_iterations=[14, 8],
        device="cpu",
        seed=42,
        outprefix=str(tmp_path / "syn_regadam_41.3_"),
        verbose=False,
    )
    fwd_path = next(p for p in reg["fwdtransforms"] if str(p).endswith("1Warp.nii.gz"))
    inv_path = next(p for p in reg["invtransforms"] if str(p).endswith("1InverseWarp.nii.gz"))
    u_fwd = _read_warp_xyz(fwd_path)
    u_inv = _read_warp_xyz(inv_path)

    assert _orthogonal_leakage_linf_mm(u_fwd, v_pe) < 1e-5
    assert _orthogonal_leakage_linf_mm(u_inv, v_pe) < 1e-5
    assert float(np.max(np.abs(u_fwd[..., 1]))) > 0.4
    assert float(np.max(np.abs(u_fwd[..., 2]))) > 0.35
    assert _folding_pct(reg, fixed) == 0.0


@pytest.mark.parametrize("pitch_deg", [32.2, 41.3, 45.0])
def test_oblique_pe_greedy_and_batch_greedy_project_along_unit_vector(tmp_path, pitch_deg):
    fixed, moving, pe, v_pe = _make_oblique_pe_phantom(tmp_path, pitch_deg)
    reg = syntx.greedy(
        fixed,
        moving,
        initial_transform=False,
        return_inverse=True,
        restrict_transformation=v_pe,
        scales=[2, 1],
        reg_iterations=[16, 10],
        device="cpu",
        outprefix=str(tmp_path / f"greedy_vec_{pitch_deg:.1f}_"),
    )
    fwd_path = next(p for p in reg["fwdtransforms"] if str(p).endswith("1Warp.nii.gz"))
    inv_path = next(p for p in reg["invtransforms"] if str(p).endswith("1InverseWarp.nii.gz"))
    u_fwd = _read_warp_xyz(fwd_path)
    u_inv = _read_warp_xyz(inv_path)

    assert _orthogonal_leakage_linf_mm(u_fwd, v_pe) < 1e-5
    assert _orthogonal_leakage_linf_mm(u_inv, v_pe) < 1e-5
    assert float(np.max(np.abs(u_fwd[..., 1]))) > 0.5
    assert float(np.max(np.abs(u_fwd[..., 2]))) > 0.45
    assert _folding_pct(reg, fixed) == 0.0

    # Verify batch_greedy parity on the 41.3-deg oblique phantom
    if abs(pitch_deg - 41.3) < 1e-3:
        b_res = syntx.batch_greedy(
            fixed,
            [moving],
            initial_transforms=["identity"],
            return_inverse=True,
            write_transforms=True,
            restrict_transformation=v_pe,
            scales=[2, 1],
            reg_iterations=[12, 8],
            device="cpu",
            outprefixes=[str(tmp_path / "bgreedy_41.3_")],
        )[0]
        bu_fwd = _read_warp_xyz(next(p for p in b_res["fwdtransforms"] if str(p).endswith("1Warp.nii.gz")))
        bu_inv = _read_warp_xyz(next(p for p in b_res["invtransforms"] if str(p).endswith("1InverseWarp.nii.gz")))
        assert _orthogonal_leakage_linf_mm(bu_fwd, v_pe) < 1e-5
        assert _orthogonal_leakage_linf_mm(bu_inv, v_pe) < 1e-5
        assert _folding_pct(b_res, fixed) == 0.0


def test_oblique_pe_batch_syn_projects_along_unit_vector(tmp_path):
    fixed, moving_a, _, v_pe = _make_oblique_pe_phantom(tmp_path, 41.3, amp_mm=4.0)
    _, moving_b, _, _ = _make_oblique_pe_phantom(tmp_path, 41.3, amp_mm=-3.6)

    results = syntx.batch_syn(
        fixed,
        [moving_a, moving_b],
        type_of_transform="SyNOnly",
        initial_transforms=["identity", "identity"],
        restrict_transformation=v_pe,
        levels=[2, 1],
        reg_iterations=[14, 8],
        write_transforms=True,
        outprefixes=[str(tmp_path / "bsyn_0_"), str(tmp_path / "bsyn_1_")],
        device="cpu",
        seed=42,
    )
    for idx, res in enumerate(results):
        fwd_path = next(p for p in res["fwdtransforms"] if str(p).endswith("1Warp.nii.gz"))
        inv_path = next(p for p in res["invtransforms"] if str(p).endswith("1InverseWarp.nii.gz"))
        u_fwd = _read_warp_xyz(fwd_path)
        u_inv = _read_warp_xyz(inv_path)
        assert _orthogonal_leakage_linf_mm(u_fwd, v_pe) < 1e-5, f"subject {idx} fwd leakage"
        assert _orthogonal_leakage_linf_mm(u_inv, v_pe) < 1e-5, f"subject {idx} inv leakage"
        assert float(np.max(np.abs(u_fwd[..., 1]))) > 0.45
        assert float(np.max(np.abs(u_fwd[..., 2]))) > 0.40
        assert _folding_pct(res, fixed) == 0.0


def test_dewarp_resolver_and_syn_only_support_oblique_pe(tmp_path):
    fixed, moving, pe, v_pe = _make_oblique_pe_phantom(tmp_path, 41.3)
    sidecar_path = tmp_path / "oblique_bold.json"
    sidecar_path.write_text('{"PhaseEncodingDirection": "j-"}', encoding="utf-8")

    # Every oblique BIDS / PhaseEncode / unit-vector form resolves cleanly to exact v_PE
    for form in (
        "j-",
        "j",
        str(sidecar_path),
        pe,
        pe_to_dict(pe),
        v_pe,
        tuple(-x for x in v_pe),  # signed unit vector also accepted
    ):
        resolved_vec = dewarp._resolve_restrict_transformation(form, moving)
        assert resolved_vec is not None
        assert np.linalg.norm(resolved_vec) == pytest.approx(1.0, abs=1e-6)
        assert abs(float(np.dot(resolved_vec, v_pe))) == pytest.approx(1.0, abs=1e-6)

    # Ambiguous bare anatomical label without BIDS voxel-axis source still raises on 41.3-deg oblique header
    with pytest.raises(PhaseEncodeObliqueError):
        dewarp._resolve_restrict_transformation("AP", moving)

    # End-to-end dewarp.syn_only_reference_to_anatomy_rigid with oblique sidecar
    dw = dewarp.syn_only_reference_to_anatomy_rigid(
        reference=moving,
        anatomy_rigid=fixed,
        output_directory=tmp_path,
        restrict_transformation=str(sidecar_path),
        reg_iterations=[12, 6],
        device="cpu",
    )
    u_fwd = _read_warp_xyz(next(p for p in dw.forward_transformlist if str(p).endswith("1Warp.nii.gz")))
    u_inv = _read_warp_xyz(next(p for p in dw.inverse_transformlist if str(p).endswith("1InverseWarp.nii.gz")))
    assert _orthogonal_leakage_linf_mm(u_fwd, v_pe) < 1e-5
    assert _orthogonal_leakage_linf_mm(u_inv, v_pe) < 1e-5
    assert float(np.max(np.abs(u_fwd[..., 1]))) > 0.4
    assert float(np.max(np.abs(u_fwd[..., 2]))) > 0.35
