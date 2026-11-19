"""Unit and cross-solver parity tests for hybrid VoxelMorph warm-starts and universal ``initial_warp`` support."""

from __future__ import annotations

import os
import tempfile
from typing import Tuple

import ants
import numpy as np
import pytest
import torch

import syntx
from syntx.benchmark.external import (
    _voxel_disp_nat_to_norm_zyx,
    run_voxelmorph_hybrid_greedy,
    run_voxelmorph_hybrid_syn,
)
from syntx.deformation_metrics import compute_jacobian_metrics, flow_jacobian_metrics
from syntx.core.initial_warp import compute_unclamped_initial_grid_zyx
from syntx.core.inverse import update_inverse_field_nd_anderson
from syntx.greedy import GreedyRegistrationModel, greedy_registration
from syntx.spatial import (
    disp_tensor_to_itk,
    export_ants_affine_transform,
    normalized_to_physical_disp,
    physical_to_normalized_disp,
)
from syntx.transforms import apply_transforms as syntx_apply_transforms


def _make_oblique_blob_pair(
    shape: Tuple[int, int, int] = (24, 28, 32),
    shift_mm: Tuple[float, float, float] = (1.5, -1.0, 0.8),
) -> Tuple[ants.ANTsImage, ants.ANTsImage, str, torch.Tensor, torch.Tensor, ants.ANTsImage, ants.ANTsImage, str, str]:
    """Create a non-cubic anisotropic oblique-direction 3-D phantom pair with a known
    affine ``.mat`` and smooth 0%-folding initial warp ``(u0_norm, u0_inv_norm)``."""
    nx, ny, nz = shape
    spacing = (1.2, 1.0, 1.5)
    origin = (-12.0, 5.0, -8.0)
    theta_rot = np.deg2rad(15.0)
    direction = np.array(
        [
            [-np.cos(theta_rot), np.sin(theta_rot), 0.0],
            [-np.sin(theta_rot), -np.cos(theta_rot), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    gx, gy, gz = np.meshgrid(
        np.linspace(-1.0, 1.0, nx, dtype=np.float32),
        np.linspace(-1.0, 1.0, ny, dtype=np.float32),
        np.linspace(-1.0, 1.0, nz, dtype=np.float32),
        indexing="ij",
    )
    r2 = (gx / 0.55) ** 2 + (gy / 0.60) ** 2 + (gz / 0.55) ** 2
    fixed_arr = np.exp(-1.5 * r2) + 0.35 * np.exp(-8.0 * ((gx - 0.15) ** 2 + (gy + 0.1) ** 2 + gz**2))
    fixed_arr = fixed_arr.astype(np.float32)
    fixed_img = ants.from_numpy(fixed_arr, origin=origin, spacing=spacing, direction=direction)

    # Smooth window-tapered physical displacement (zero on boundary -> guaranteed 0% folding)
    env = (
        np.sin(np.pi * np.linspace(0.0, 1.0, nz, dtype=np.float32))[:, None, None]
        * np.sin(np.pi * np.linspace(0.0, 1.0, ny, dtype=np.float32))[None, :, None]
        * np.sin(np.pi * np.linspace(0.0, 1.0, nx, dtype=np.float32))[None, None, :]
    )
    zz, yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, nz, dtype=np.float32),
        np.linspace(-1.0, 1.0, ny, dtype=np.float32),
        np.linspace(-1.0, 1.0, nx, dtype=np.float32),
        indexing="ij",
    )
    u_phys_np = np.zeros((1, nz, ny, nx, 3), dtype=np.float32)
    u_phys_np[0, ..., 0] = 1.8 * env * np.cos(0.5 * np.pi * xx)
    u_phys_np[0, ..., 1] = -1.4 * env * np.cos(0.5 * np.pi * yy)
    u_phys_np[0, ..., 2] = 1.2 * env * np.cos(0.5 * np.pi * zz)
    u_phys = torch.from_numpy(u_phys_np)
    u_inv_phys = update_inverse_field_nd_anderson(
        u_phys,
        spacing=spacing,
        origin=origin,
        direction=direction,
        steps=30,
        max_error_threshold=1e-4,
    )

    u0_norm = physical_to_normalized_disp(u_phys, (nz, ny, nx), spacing, direction, dtype=torch.float32)
    u0_inv_norm = physical_to_normalized_disp(u_inv_phys, (nz, ny, nx), spacing, direction, dtype=torch.float32)

    warp_img = disp_tensor_to_itk(u_phys, fixed_img)
    inv_warp_img = disp_tensor_to_itk(u_inv_phys, fixed_img)

    tmpdir = tempfile.mkdtemp(prefix="syntx_hybrid_test_")
    warp_path = os.path.join(tmpdir, "init_fwd_Warp.nii.gz")
    inv_warp_path = os.path.join(tmpdir, "init_inv_Warp.nii.gz")
    ants.image_write(warp_img, warp_path)
    ants.image_write(inv_warp_img, inv_warp_path)

    M_phys = np.eye(3, dtype=np.float64)
    t_phys = np.array(shift_mm, dtype=np.float64)
    aff_path = os.path.join(tmpdir, "init_Affine.mat")
    export_ants_affine_transform(M_phys, t_phys, dim=3, filename=aff_path)

    # Create moving image as inverse-transformed version of fixed so T_aff o (Id + u0) aligns moving -> fixed
    moving_img = syntx_apply_transforms(
        fixed=fixed_img,
        moving=fixed_img,
        transformlist=[aff_path, inv_warp_path],
        whichtoinvert=[True, False],
        interpolator="linear",
    )
    return (
        fixed_img,
        moving_img,
        aff_path,
        u0_norm,
        u0_inv_norm,
        warp_img,
        inv_warp_img,
        warp_path,
        inv_warp_path,
    )


def test_coordinate_roundtrip_oblique_anisotropic():
    """Verify voxel -> normalized -> physical -> normalized -> initial_grid round-trip
    matches within 1e-5 mm on a non-cubic anisotropic oblique-direction phantom."""
    shape_xyz = (24, 28, 32)
    shape_zyx = (32, 28, 24)
    spacing = (1.25, 0.95, 1.6)
    origin = (-15.0, 8.0, 4.0)
    ang = np.deg2rad(22.0)
    direction = np.array(
        [
            [-np.cos(ang), np.sin(ang), 0.0],
            [np.sin(ang), np.cos(ang), 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    fixed = ants.from_numpy(np.zeros(shape_xyz, dtype=np.float32), origin=origin, spacing=spacing, direction=direction)
    moving = ants.from_numpy(np.zeros(shape_xyz, dtype=np.float32), origin=origin, spacing=spacing, direction=direction)

    rng = np.random.default_rng(7)
    disp_vox_nat = torch.from_numpy(rng.uniform(-2.0, 2.0, size=(1, 3, *shape_xyz)).astype(np.float32))
    u_norm = _voxel_disp_nat_to_norm_zyx(disp_vox_nat)
    assert tuple(u_norm.shape) == (1, *shape_zyx, 3)

    u_phys = normalized_to_physical_disp(u_norm, shape_zyx, spacing, direction, dtype=torch.float32)
    u_norm_rt = physical_to_normalized_disp(u_phys, shape_zyx, spacing, direction, dtype=torch.float32)
    u_phys_rt = normalized_to_physical_disp(u_norm_rt, shape_zyx, spacing, direction, dtype=torch.float32)
    assert float((u_phys - u_phys_rt).abs().max()) < 1e-5

    # Expected physical displacement in tensor ZYX component order (u_z, u_y, u_x)
    D = torch.from_numpy(direction @ np.diag(spacing)).float()
    disp_vox_zyx = disp_vox_nat.permute(0, 4, 3, 2, 1)
    expected_phys_xyz = torch.einsum("ij,...j->...i", D, disp_vox_zyx)
    expected_phys = expected_phys_xyz[..., [2, 1, 0]]
    assert float((u_phys - expected_phys).abs().max()) < 1e-5

    grid_zyx = compute_unclamped_initial_grid_zyx(
        u_norm, fixed, moving, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), torch.device("cpu")
    )
    id_zyx = torch.nn.functional.affine_grid(
        torch.eye(3, 4, dtype=torch.float32).unsqueeze(0), [1, 1, *shape_zyx], align_corners=True
    )
    assert float(((grid_zyx - id_zyx) - u_norm).abs().max()) < 1e-5


def test_greedy_initial_warp_formats_and_export_contract():
    """Verify GreedyRegistrationModel.fit and syntx.greedy accept Tensor, ANTsImage,
    .nii.gz path, and [warp, affine], export single-interpolation chains, and record provenance."""
    fixed, moving, aff_path, u0_norm, u0_inv_norm, warp_img, inv_warp_img, warp_path, inv_warp_path = (
        _make_oblique_blob_pair()
    )

    # 1. [affine, warp] reversed order must fail loudly
    with pytest.raises(NotImplementedError):
        greedy_registration(
            fixed,
            moving,
            initial_transform=[aff_path, warp_path],
            reg_iterations=[0],
            scales=[1],
            device="cpu",
        )

    # 2. Direct GreedyRegistrationModel.fit(..., initial_warp=...) check
    fi_t = torch.from_numpy(fixed.numpy().T).float().unsqueeze(0).unsqueeze(0)
    mi_t = torch.from_numpy(moving.numpy().T).float().unsqueeze(0).unsqueeze(0)
    model = GreedyRegistrationModel(dim=3, device=torch.device("cpu"))
    theta = torch.eye(3, 4, dtype=torch.float32).unsqueeze(0)
    disp_fit = model.fit(
        fi_t,
        mi_t,
        theta=theta,
        scales=[1],
        iterations=[2],
        spacing=fixed.spacing,
        initial_warp=u0_norm,
    )
    assert tuple(disp_fit.shape) == (1, *fi_t.shape[2:], 3)

    # 3. Verify reg_iterations=[0] across all input forms reproduces the direct composed warp
    expected_warped = syntx_apply_transforms(fixed=fixed, moving=moving, transformlist=[warp_path, aff_path], interpolator="linear")
    expected_np = expected_warped.numpy()

    cases = [
        ("tensor", dict(initial_transform=aff_path, initial_warp=u0_norm, initial_inverse_warp=u0_inv_norm)),
        ("ants_image", dict(initial_transform=aff_path, initial_warp=warp_img, initial_inverse_warp=inv_warp_img)),
        ("file", dict(initial_transform=aff_path, initial_warp=warp_path, initial_inverse_warp=inv_warp_path)),
        ("transform_list", dict(initial_transform=[warp_path, aff_path])),
    ]
    for expected_kind, extra in cases:
        res0 = greedy_registration(
            fixed,
            moving,
            scales=[1],
            reg_iterations=[0],
            return_inverse=True,
            digital_diffeomorphism=True,
            device="cpu",
            seed=42,
            **extra,
        )
        assert len(res0["fwdtransforms"]) == 2
        assert res0["fwdtransforms"][0].endswith(".nii.gz")
        assert res0["fwdtransforms"][1].endswith(".mat")
        assert len(res0["invtransforms"]) == 2
        assert res0["invtransforms"][0].endswith(".mat")
        assert res0["invtransforms"][1].endswith(".nii.gz")
        assert res0["whichtoinvert_inv"] == [True, False]
        assert res0["model"].has_initial_warp is True
        assert res0["model"].initial_warp_kind == expected_kind
        assert res0["provenance"]["has_initial_warp"] is True
        assert res0["provenance"]["initial_warp_kind"] == expected_kind
        mae = float(np.mean(np.abs(res0["warpedmovout"].numpy() - expected_np)))
        assert mae < 5e-3, f"kind={expected_kind} MAE={mae}"

    # 4. Refinement run with 0% folding
    res_ref = greedy_registration(
        fixed,
        moving,
        scales=[2, 1],
        reg_iterations=[3, 2],
        optimizer="regadam",
        learning_rate=0.4,
        initial_transform=aff_path,
        initial_warp=u0_norm,
        return_inverse=True,
        digital_diffeomorphism=True,
        device="cpu",
        seed=42,
    )
    assert len(res_ref["fwdtransforms"]) == 2
    assert res_ref["whichtoinvert_inv"] == [True, False]
    metrics = compute_jacobian_metrics(fixed, res_ref["fwdtransforms"][0])
    assert metrics["folding_pct"] == 0.0
    lmet = flow_jacobian_metrics(fixed, res_ref)
    if lmet is not None:
        assert lmet["folding_pct"] == 0.0


@pytest.mark.parametrize(
    "solver,backend",
    [
        ("greedy", "pytorch"),
        ("syn", "pytorch"),
        ("syn", "jax"),
        ("tvf", "pytorch"),
        ("tvf", "jax"),
        ("syngs", "pytorch"),
        ("syngs", "jax"),
        ("auto_reg", "pytorch"),
    ],
)
def test_cross_solver_initial_warp_parity(solver: str, backend: str):
    """Verify zero-epoch reproduction, refinement convergence with 0% folding, single-interpolation
    transform chains, and provenance across all 4 solver families and PyTorch/JAX backends."""
    fixed, moving, aff_path, u0_norm, u0_inv_norm, _, _, warp_path, inv_warp_path = _make_oblique_blob_pair(
        shape=(16, 18, 20), shift_mm=(0.9, -0.6, 0.5)
    )
    expected_warped = syntx_apply_transforms(
        fixed=fixed, moving=moving, transformlist=[warp_path, aff_path], interpolator="linear"
    ).numpy()

    def _call(reg_iters):
        if solver == "greedy":
            return greedy_registration(
                fixed,
                moving,
                scales=[1],
                reg_iterations=reg_iters,
                initial_transform=aff_path,
                initial_warp=u0_norm,
                initial_inverse_warp=u0_inv_norm,
                return_inverse=True,
                digital_diffeomorphism=True,
                device="cpu",
                seed=42,
            )
        if solver == "syn":
            return syntx.syn(
                fixed,
                moving,
                levels=[1],
                reg_iterations=reg_iters,
                initial_transform=aff_path,
                initial_warp=u0_norm,
                initial_inverse_warp=u0_inv_norm,
                backend=backend,
                device="cpu" if backend == "pytorch" else None,
                digital_diffeomorphism=True,
                seed=42,
            )
        if solver == "tvf":
            extra = {"regularizer": "gaussian", "optimizer": "cfl"} if backend == "jax" else {"device": "cpu"}
            return syntx.tvf(
                fixed,
                moving,
                levels=[1],
                reg_iterations=reg_iters,
                initial_transform=aff_path,
                initial_warp=u0_norm,
                initial_inverse_warp=u0_inv_norm,
                backend=backend,
                digital_diffeomorphism=True,
                **extra,
            )
        if solver == "syngs":
            extra = {"device": "cpu"} if backend == "pytorch" else {"optimizer": "cfl"}
            return syntx.syngs(
                fixed,
                moving,
                levels=[1],
                reg_iterations=reg_iters,
                initial_transform=aff_path,
                initial_warp=u0_norm,
                initial_inverse_warp=u0_inv_norm,
                backend=backend,
                digital_diffeomorphism=True,
                seed=42,
                **extra,
            )
        if solver == "auto_reg":
            return syntx.auto_reg(
                fixed,
                moving,
                type_of_transform="SyN",
                levels=[1],
                reg_iterations=reg_iters,
                initial_transform=[warp_path, aff_path],
                diagnose=False,
                device="cpu",
                digital_diffeomorphism=True,
                seed=42,
            )
        raise ValueError(solver)

    res0 = _call([0])
    assert len(res0["fwdtransforms"]) == 2
    assert res0["fwdtransforms"][0].endswith(".nii.gz")
    assert res0["fwdtransforms"][1].endswith(".mat")
    assert len(res0["invtransforms"]) == 2
    assert res0["invtransforms"][0].endswith(".mat")
    assert res0["invtransforms"][1].endswith(".nii.gz")
    assert res0["whichtoinvert_inv"] == [True, False]
    assert res0["model"].has_initial_warp is True
    expected_kind = "transform_list" if solver == "auto_reg" else "tensor"
    assert res0["model"].initial_warp_kind == expected_kind
    assert res0["provenance"]["has_initial_warp"] is True
    assert res0["provenance"]["initial_warp_kind"] == expected_kind
    mae0 = float(np.mean(np.abs(res0["warpedmovout"].numpy() - expected_warped)))
    assert mae0 < 5e-3, f"{solver}/{backend} zero-iter MAE={mae0}"

    res2 = _call([2])
    assert len(res2["fwdtransforms"]) == 2
    assert len(res2["invtransforms"]) == 2
    assert res2["whichtoinvert_inv"] == [True, False]
    jmet = compute_jacobian_metrics(fixed, res2["fwdtransforms"][0])
    assert jmet["folding_pct"] == 0.0, f"{solver}/{backend} folding={jmet['folding_pct']}"


def test_voxelmorph_hybrid_adapters_contract():
    """Verify run_voxelmorph_hybrid_greedy and run_voxelmorph_hybrid_syn run end-to-end
    with pretrained weights, preserve single-interpolation chains, and achieve 0% folding."""
    pytest.importorskip("voxelmorph")
    pytest.importorskip("antsxdata")
    fixed, moving, aff_path, *_ = _make_oblique_blob_pair(shape=(28, 32, 36), shift_mm=(1.2, -0.8, 0.4))

    res_hg = run_voxelmorph_hybrid_greedy(
        fixed,
        moving,
        initial_transform=aff_path,
        scales=(1,),
        reg_iterations=(3,),
        device="cpu",
        seed=42,
    )
    assert len(res_hg["fwdtransforms"]) == 2
    assert res_hg["fwdtransforms"][0].endswith(".nii.gz")
    assert res_hg["fwdtransforms"][1].endswith(".mat")
    assert res_hg["provenance"]["method"] == "voxelmorph_hybrid_greedy"
    assert res_hg["provenance"]["has_initial_warp"] is True
    j_hg = compute_jacobian_metrics(fixed, res_hg["fwdtransforms"][0])
    assert j_hg["folding_pct"] == 0.0

    res_hs = run_voxelmorph_hybrid_syn(
        fixed,
        moving,
        initial_transform=aff_path,
        levels=(1,),
        reg_iterations=(2,),
        device="cpu",
        seed=42,
    )
    assert len(res_hs["fwdtransforms"]) == 2
    assert res_hs["fwdtransforms"][0].endswith(".nii.gz")
    assert res_hs["fwdtransforms"][1].endswith(".mat")
    assert len(res_hs["invtransforms"]) == 2
    assert res_hs["invtransforms"][0].endswith(".mat")
    assert res_hs["invtransforms"][1].endswith(".nii.gz")
    assert res_hs["whichtoinvert_inv"] == [True, False]
    assert res_hs["provenance"]["method"] == "voxelmorph_hybrid_syn"
    assert res_hs["provenance"]["has_initial_warp"] is True
    j_hs = compute_jacobian_metrics(fixed, res_hs["fwdtransforms"][0])
    assert j_hs["folding_pct"] == 0.0


def test_syn_initial_warp_cross_orientation_nonzero_loss():
    """Regression test for cross-orientation (RPI -> LAI) initial_grid in SyN.fit():
    M_phys must stay Identity when initial_grid is set so y_norm_fixed stays in [-1, 1]
    and produces non-zero CC2 similarity gradient and non-zero residual deformation."""
    fixed, _, _, u0_norm, u0_inv_norm, _, _, _, inv_warp_path = _make_oblique_blob_pair(
        shape=(16, 18, 20), shift_mm=(0.0, 0.0, 0.0)
    )
    arr_m = fixed.numpy()
    # Place moving on a distinct LAI-like frame with shifted origin & flipped X direction
    dir_m = np.diag([1.0, -1.0, 1.0])
    origin_m = (-35.0, 12.0, -5.0)
    moving_alt = ants.from_numpy(arr_m, origin=origin_m, spacing=fixed.spacing, direction=dir_m)
    # Build affine mapping fixed center -> moving center
    c_f = np.asarray(fixed.direction) @ (np.asarray(fixed.spacing) * (np.asarray(fixed.shape) - 1) / 2.0) + np.asarray(fixed.origin)
    c_m = dir_m @ (np.asarray(moving_alt.spacing) * (np.asarray(moving_alt.shape) - 1) / 2.0) + np.asarray(origin_m)
    M_phys = dir_m @ np.linalg.inv(np.asarray(fixed.direction))
    t_phys = c_m - M_phys @ c_f
    tmpdir = tempfile.mkdtemp(prefix="syntx_cross_ori_")
    aff_cross = os.path.join(tmpdir, "cross_ori.mat")
    export_ants_affine_transform(M_phys, t_phys, dim=3, filename=aff_cross)

    res = syntx.syn(
        fixed,
        moving_alt,
        levels=[1],
        reg_iterations=[3],
        initial_transform=aff_cross,
        initial_warp=u0_norm,
        initial_inverse_warp=u0_inv_norm,
        backend="pytorch",
        device="cpu",
        digital_diffeomorphism=True,
        seed=42,
    )
    assert len(res["syn_losses"]) == 3
    assert float(res["syn_losses"][0]) < -0.1
    assert float(res["model"].warp_l2r.detach().abs().max()) > 0.0
    jmet = compute_jacobian_metrics(fixed, res["fwdtransforms"][0])
    assert jmet["folding_pct"] == 0.0

