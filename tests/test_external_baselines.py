"""Unit and contract tests for ``syntx.benchmark.external`` and ``antsxdata`` VoxelMorph asset."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import ants
import antsxdata
import numpy as np
import pytest
import torch

import syntx
import syntx.benchmark.external as ext
from syntx.benchmark.external import (
    ants_mat_to_4x4,
    apply_axis_permutation,
    canonical_orientation_transform,
    invert_displacement_permutation,
    voxel_disp_to_itk_lps,
)
from syntx.core.inverse import calculate_inverse_identity_error
from syntx.provenance import (
    assert_manifest_complete,
    build_manifest,
    capture_registration_calls,
    resolved_parameters,
)
from syntx.spatial import disp_itk_to_tensor


def _make_rps_anisotropic_phantom(shape=(32, 32, 32)):
    origin = (12.0, -18.0, 6.0)
    spacing = (1.2, 1.0, 0.9)
    direction = np.diag([1.0, -1.0, 1.0])  # ANTs RPI / axis-increasing LAS
    ix, iy, iz = np.meshgrid(
        np.arange(shape[0], dtype=np.float64),
        np.arange(shape[1], dtype=np.float64),
        np.arange(shape[2], dtype=np.float64),
        indexing="ij",
    )
    cx, cy, cz = shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0
    r2 = ((ix - cx) / 6.0) ** 2 + ((iy - cy) / 5.0) ** 2 + ((iz - cz) / 7.0) ** 2
    env = np.exp(-0.5 * r2)
    tex = 1.0 + 0.35 * np.sin(0.55 * (ix - cx)) * np.cos(0.60 * (iy - cy)) + 0.30 * np.sin(0.50 * (iz - cz))
    fi = ants.from_numpy((env * tex).astype(np.float32), origin=origin, spacing=spacing, direction=direction)
    fl = ants.from_numpy((env > 0.45).astype(np.uint32), origin=origin, spacing=spacing, direction=direction)
    return fi, fl, env, (ix, iy, iz)


def _write_test_affine_mat(path: str) -> str:
    theta = 0.07
    mat_a = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    trans_t = np.array([1.2, -0.8, 0.6], dtype=np.float64)
    center_c = np.array([12.0 + 16.0 * 1.2, -18.0 - 16.0 * 1.0, 6.0 + 16.0 * 0.9], dtype=np.float64)
    tx = ants.create_ants_transform(
        transform_type="AffineTransform",
        precision="float",
        dimension=3,
        parameters=np.concatenate([mat_a.ravel(), trans_t]),
        fixed_parameters=center_c,
    )
    ants.write_transform(tx, path)
    return path


def test_antsxdata_voxelmorph_registry_and_fetch():
    prov = antsxdata.provenance("syntx_features/voxelmorph_pairwise_3d_brain")
    assert prov["collection"] == "syntx_features"
    assert prov["asset"] == "voxelmorph_pairwise_3d_brain"
    assert prov["size"] == 435477
    assert prov["sha256"] == "94c7094b79b54bb7d80c7e83103a11cc03d33cfcc779b13ba09ca60913d14b85"
    assert "MIT" in prov["license"]
    weights_path = Path(antsxdata.fetch("syntx_features/voxelmorph_pairwise_3d_brain"))
    assert weights_path.exists() and weights_path.stat().st_size == 435477


@pytest.mark.parametrize("target_code", ["PIR", "LPS", "RAS", "LIA", "AIL"])
def test_canonical_orientation_permutation_lossless_round_trip(target_code):
    # Non-trivial oblique-permuted / sign-flipped direction cosine matrix
    direction = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
    perm, flips = canonical_orientation_transform(direction, target_code)

    rng = np.random.default_rng(7)
    scalar = torch.from_numpy(rng.normal(size=(1, 1, 10, 12, 14)).astype(np.float32))
    disp = torch.from_numpy(rng.normal(size=(1, 3, 10, 12, 14)).astype(np.float32))

    # Scalar round-trip via inverse perm/flip
    s_can = apply_axis_permutation(scalar, perm, flips)
    inv_perm = [0, 0, 0]
    for i, p in enumerate(perm):
        inv_perm[p] = i
    flip_dims = [2 + i for i, s in enumerate(flips) if s < 0]
    s_rec = torch.flip(s_can, dims=flip_dims) if flip_dims else s_can
    s_rec = s_rec.permute(0, 1, *(2 + p for p in inv_perm)).contiguous()
    assert torch.equal(scalar, s_rec)

    # Vector field round-trip: apply_axis_permutation on scalar components with component transform
    # Construct canonical vector field matching physical shift at each point
    d_can = apply_axis_permutation(disp[:, list(perm), ...], perm, flips)
    signs = torch.tensor([float(s) for s in flips]).view(1, 3, 1, 1, 1)
    d_can = d_can * signs
    d_rec = invert_displacement_permutation(d_can, perm, flips)
    assert torch.allclose(disp, d_rec, atol=1e-6)


def test_rps_anisotropic_displacement_phantom_recovery():
    vxm = pytest.importorskip("voxelmorph")
    fi, fl, env, (ix, iy, iz) = _make_rps_anisotropic_phantom((32, 32, 32))
    shift_vox = (2.0, -1.5, 1.0)

    with tempfile.TemporaryDirectory() as td:
        aff_mat = _write_test_affine_mat(os.path.join(td, "aff.mat"))
        mat_4x4 = ants_mat_to_4x4(aff_mat)
        assert mat_4x4.shape == (4, 4)

        integrator = vxm.nn.modules.IntegrateVelocityField(steps=7)
        st = vxm.nn.modules.SpatialTransformer()
        vel = torch.zeros((1, 3, 32, 32, 32), dtype=torch.float32)
        vel[:, 0] = shift_vox[0]
        vel[:, 1] = shift_vox[1]
        vel[:, 2] = shift_vox[2]
        with torch.no_grad():
            disp_fwd = integrator(vel)
            disp_inv = integrator(-vel)

        fwd_path = os.path.join(td, "fwd.nii.gz")
        inv_path = os.path.join(td, "inv.nii.gz")
        ants.image_write(voxel_disp_to_itk_lps(disp_fwd, fi), fwd_path)
        ants.image_write(voxel_disp_to_itk_lps(disp_inv, fi), inv_path)

        mi = ants.apply_transforms(
            fixed=fi,
            moving=fi,
            transformlist=[aff_mat, inv_path],
            whichtoinvert=[True, False],
            interpolator="bSpline",
        )
        mi_aff = ants.apply_transforms(fixed=fi, moving=mi, transformlist=[aff_mat], interpolator="linear")
        with torch.no_grad():
            warped_st = st(torch.from_numpy(mi_aff.numpy()).float()[None, None], disp_fwd).numpy()[0, 0]
        warped_ants = ants.apply_transforms(
            fixed=fi, moving=mi, transformlist=[fwd_path, aff_mat], interpolator="linear"
        ).numpy()
        assert float(np.mean(np.abs(warped_st - warped_ants))) < 0.01

        ice = calculate_inverse_identity_error(
            disp_itk_to_tensor(fwd_path, device="cpu").float(),
            disp_itk_to_tensor(inv_path, device="cpu").float(),
            spacing=fi.spacing,
            origin=fi.origin,
            direction=fi.direction,
        )
        assert float(ice["mean_error"]) < 0.05

        # Run VxmPairwise SVF optimization on the phantom to verify < 0.05 vox recovery
        import neurite as ne

        torch.manual_seed(42)
        src_t = torch.from_numpy(mi_aff.numpy()).float()[None, None]
        tgt_t = torch.from_numpy(fi.numpy()).float()[None, None]
        vxm_opt = vxm.nn.models.VxmPairwise(
            ndim=3,
            source_channels=1,
            target_channels=1,
            nb_features=(16, 32, 32, 32),
            integration_steps=7,
            device="cpu",
        )
        grad_fn = ne.nn.modules.SpatialGradient("l2")
        opt = torch.optim.Adam(vxm_opt.parameters(), lr=3e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=3e-4)
        for _ in range(200):
            opt.zero_grad()
            vel_p = vxm_opt(src_t, tgt_t, return_field_type="velocity")
            disp_p = vxm_opt.velocity_field_integrator(vel_p)
            w = vxm_opt.spatial_transformer(src_t, disp_p)
            loss = torch.nn.functional.mse_loss(w, tgt_t) + 0.01 * grad_fn(vel_p)
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            rec_disp = vxm_opt(src_t, tgt_t, return_field_type="displacement").numpy()[0]
        fg = env > 0.5
        mae_vox = float(
            np.mean([np.abs(rec_disp[k][fg] - shift_vox[k]).mean() for k in range(3)])
        )
        assert mae_vox < 0.05


def test_external_adapters_unidirectional_and_provenance(tmp_path):
    pytest.importorskip("voxelmorph")
    pytest.importorskip("fireants")
    fi, _, _, _ = _make_rps_anisotropic_phantom((32, 32, 32))
    aff_mat = _write_test_affine_mat(str(tmp_path / "aff.mat"))
    mi = ants.apply_transforms(
        fixed=fi, moving=fi, transformlist=[aff_mat], whichtoinvert=[True], interpolator="linear"
    )

    # 1. syntx.greedy unidirectional contract
    with capture_registration_calls() as cap_g:
        res_g = syntx.greedy(
            fi, mi, initial_transform=aff_mat, return_inverse=False, reg_iterations=[2, 2], device="cpu", verbose=False
        )
    assert res_g["invtransforms"] == []
    assert res_g["whichtoinvert_inv"] == []
    assert len(res_g["fwdtransforms"]) == 2
    man_g = build_manifest(calls=cap_g.calls, run={"arm": "syntx_greedy"})
    assert_manifest_complete(man_g, require_calls=True)

    # 2. run_voxelmorph_pretrained (zero-shot)
    with capture_registration_calls() as cap_vp:
        res_vp = ext.run_voxelmorph_pretrained(
            fi, mi, initial_transform=aff_mat, polish_steps=0, return_inverse=False, device="cpu", out_dir=str(tmp_path / "vp")
        )
    assert res_vp["invtransforms"] == []
    assert res_vp["whichtoinvert_inv"] == []
    assert len(res_vp["fwdtransforms"]) == 2 and res_vp["fwdtransforms"][1] == aff_mat
    assert res_vp["warpedmovout"].shape == fi.shape
    man_vp = build_manifest(calls=cap_vp.calls, run={"arm": "voxelmorph_pretrained"})
    assert_manifest_complete(man_vp, require_calls=True)
    rp_vp = resolved_parameters(man_vp, function_prefix="syntx.benchmark.external.")
    assert rp_vp["model_attributes"]["asset_sha256"] == "94c7094b79b54bb7d80c7e83103a11cc03d33cfcc779b13ba09ca60913d14b85"

    # 3. run_voxelmorph_multiscale
    with capture_registration_calls() as cap_ms:
        res_ms = ext.run_voxelmorph_multiscale(
            fi, mi, initial_transform=aff_mat, scales=(2, 1), reg_iterations=(2, 2), return_inverse=False, device="cpu", out_dir=str(tmp_path / "ms")
        )
    assert res_ms["invtransforms"] == []
    assert res_ms["whichtoinvert_inv"] == []
    assert len(res_ms["fwdtransforms"]) == 2
    man_ms = build_manifest(calls=cap_ms.calls, run={"arm": "voxelmorph_svf"})
    assert_manifest_complete(man_ms, require_calls=True)

    # 4. run_fireants_greedy
    with capture_registration_calls() as cap_fa:
        res_fa = ext.run_fireants_greedy(
            fi, mi, initial_transform=aff_mat, scales=(2, 1), reg_iterations=(2, 2), return_inverse=False, device="cpu", out_dir=str(tmp_path / "fa")
        )
    assert res_fa["invtransforms"] == []
    assert res_fa["whichtoinvert_inv"] == []
    assert len(res_fa["fwdtransforms"]) == 2
    man_fa = build_manifest(calls=cap_fa.calls, run={"arm": "fireants_greedy"})
    assert_manifest_complete(man_fa, require_calls=True)
