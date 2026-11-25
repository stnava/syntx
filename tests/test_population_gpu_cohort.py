"""Unit and contract tests for R53 population-scale GPU cohort architecture (WP1-WP4)."""

from __future__ import annotations

import os
import tempfile

import ants
import numpy as np
import pytest
import torch

import syntx
from syntx.cohort import FixedTarget, batch_greedy, batch_robust_affine
from syntx.core.losses.mutual_information import (
    batched_mattes_mi_loss_shared_fixed,
    mattes_mi_loss_nd,
    parzen_weights,
)
from syntx.core.optimizers import FieldStepOptimizer
from syntx.deformation_metrics import compute_bidirectional_dice
from syntx.liouville import liouville_determinant
from syntx.transforms import (
    TransformChain,
    TransformElement,
    apply_transforms,
    signed_jacobian_determinant,
)

from tests.test_large_image_and_transfer_efficiency import (
    DispatchTransferSyncCounter,
    _accel_device,
    _oblique_direction,
)


def _make_oblique_phantom(
    shape=(24, 28, 24),
    shift=(0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
    spacing=(1.2, 1.0, 1.3),
):
    x, y, z = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[1], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[2], dtype=np.float32),
        indexing="ij",
    )
    xs = scale[0] * (x - shift[0])
    ys = scale[1] * (y - shift[1])
    zs = scale[2] * (z - shift[2])
    r2 = xs**2 + (1.15 * ys) ** 2 + (0.9 * zs) ** 2
    arr = np.clip(
        np.exp(-2.6 * r2) * (0.65 + 0.35 * np.cos(4.5 * xs) * np.cos(4.5 * ys) * np.cos(3.5 * zs)),
        0.0,
        1.0,
    ).astype(np.float32)
    arr[r2 > 0.82] = 0.0
    lbl = np.zeros(shape, dtype=np.float32)
    lbl[r2 <= 0.65] = 1.0
    lbl[r2 <= 0.25] = 2.0
    d_mat = _oblique_direction()
    origin = (-0.5 * shape[0] * spacing[0], -0.5 * shape[1] * spacing[1], -0.5 * shape[2] * spacing[2])
    img = ants.from_numpy(arr, origin=origin, spacing=spacing, direction=d_mat)
    seg = ants.from_numpy(lbl, origin=origin, spacing=spacing, direction=d_mat)
    return img, seg


def test_transform_element_and_chain_in_memory_roundtrip():
    dev = _accel_device()
    fi, _ = _make_oblique_phantom(shape=(20, 24, 22), shift=(0.0, 0.0, 0.0))
    mi, _ = _make_oblique_phantom(shape=(20, 24, 22), shift=(0.06, -0.05, 0.04))

    theta = 0.05
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    b_phys = np.array([1.2, -0.8, 0.5], dtype=np.float64)
    aff_el = TransformElement.from_affine(rot, b_phys)

    nx, ny, nz = fi.shape
    disp_zyx = 0.35 * torch.sin(
        torch.linspace(-1.0, 1.0, nz * ny * nx * 3, device=dev, dtype=torch.float32)
    ).reshape(1, nz, ny, nx, 3)
    disp_el = TransformElement.from_disp_tensor(disp_zyx, fi)
    chain_mem = TransformChain([disp_el, aff_el])
    assert chain_mem.has_in_memory()

    warped_mem = apply_transforms(fi, mi, chain_mem)
    warped_ten = apply_transforms(fi, mi, chain_mem, return_tensor=True)
    assert isinstance(warped_ten, torch.Tensor)
    assert warped_ten.shape == (1, 1, nz, ny, nx)
    np.testing.assert_allclose(
        warped_mem.numpy(),
        warped_ten[0, 0].permute(2, 1, 0).contiguous().detach().cpu().numpy(),
        atol=1e-6,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "rt_")
        saved_paths = chain_mem.save(prefix)
        assert len(saved_paths) == 2
        assert all(os.path.exists(p) for p in saved_paths)
        chain_disk = TransformChain.from_paths(saved_paths)
        warped_disk = apply_transforms(fi, mi, chain_disk)
        np.testing.assert_allclose(warped_mem.numpy(), warped_disk.numpy(), atol=2e-5)

        jac_mem = signed_jacobian_determinant(chain_mem, fi).numpy()
        jac_disk = signed_jacobian_determinant(chain_disk, fi).numpy()
        np.testing.assert_allclose(jac_mem, jac_disk, atol=2e-5)


def test_greedy_write_transforms_false_zero_d2h_and_downstream_compat():
    dev = _accel_device()
    fi, seg_f = _make_oblique_phantom(shape=(24, 28, 24), shift=(0.0, 0.0, 0.0))
    mi, seg_m = _make_oblique_phantom(shape=(24, 28, 24), shift=(0.08, -0.06, 0.05))

    reg_disk = syntx.greedy(
        fi,
        mi,
        reg_iterations=(6, 4),
        scales=(2, 1),
        return_inverse=True,
        write_transforms=True,
        return_warped=True,
    )
    reg_mem = syntx.greedy(
        fi,
        mi,
        reg_iterations=(6, 4),
        scales=(2, 1),
        return_inverse=True,
        write_transforms=False,
        return_warped=False,
    )
    assert reg_mem["fwdtransforms"] == []
    assert reg_mem["invtransforms"] == []
    assert reg_mem["warpedmovout"] is None
    assert isinstance(reg_mem["fwd_chain"], TransformChain)
    assert isinstance(reg_mem["inv_chain"], TransformChain)
    assert reg_mem["fwd_chain"].has_in_memory()

    warped_mem = apply_transforms(fi, mi, reg_mem["fwd_chain"])
    np.testing.assert_allclose(warped_mem.numpy(), reg_disk["warpedmovout"].numpy(), atol=1e-5)

    jac_mem = liouville_determinant(reg_mem, fi)
    jac_disk = liouville_determinant(reg_disk, fi)
    # ITK .mat float32 serialization quantizes affine parameters at ~1e-6 vs float64 in-memory TransformElement,
    # yielding ~4e-5 finite-difference derivative spread after 10 deformable steps; pure deformable is < 1e-6.
    np.testing.assert_allclose(jac_mem.numpy(), jac_disk.numpy(), atol=1e-4)

    g_id_disk = syntx.greedy(fi, mi, initial_transform="identity", reg_iterations=(4, 2), scales=(2, 1), write_transforms=True)
    g_id_mem = syntx.greedy(fi, mi, initial_transform="identity", reg_iterations=(4, 2), scales=(2, 1), write_transforms=False, return_warped=False)
    np.testing.assert_allclose(
        liouville_determinant(g_id_mem, fi).numpy(),
        liouville_determinant(g_id_disk, fi).numpy(),
        atol=1e-5,
    )

    _, _, dice_mem = compute_bidirectional_dice(
        seg_f, seg_m, fi, mi, reg_mem["fwd_chain"], reg_mem["inv_chain"]
    )
    _, _, dice_disk = compute_bidirectional_dice(
        seg_f,
        seg_m,
        fi,
        mi,
        reg_disk["fwdtransforms"],
        reg_disk["invtransforms"],
    )
    assert abs(dice_mem - dice_disk) < 1e-6

    if dev.type != "cpu":
        # Warm caches once, then verify zero D2H bytes and zero syncs in deformable-only sync_free mode.
        target = FixedTarget.from_image(
            fi,
            scales=(2, 1),
            window_size=5,
            device=dev,
        )
        _ = syntx.greedy(
            target,
            mi,
            initial_transform="identity",
            reg_iterations=(4, 2),
            scales=(2, 1),
            lncc_radius=2,
            return_inverse=False,
            write_transforms=False,
            return_warped=False,
            sync_free=True,
        )
        with DispatchTransferSyncCounter(dev.type) as ctr:
            _ = syntx.greedy(
                target,
                mi,
                initial_transform="identity",
                reg_iterations=(4, 2),
                scales=(2, 1),
                lncc_radius=2,
                return_inverse=False,
                write_transforms=False,
                return_warped=False,
                sync_free=True,
            )
        assert ctr.d2h_bytes == 0
        # 0 in-loop solver syncs + 1 post-solve digital_diffeomorphism fast-exit boolean check (down from 155 syncs).
        assert ctr.sync_count <= 2
        assert ctr.h2d_bytes <= int(1.15 * mi.numpy().nbytes)

        with DispatchTransferSyncCounter(dev.type) as ctr_zero:
            _ = syntx.greedy(
                target,
                mi,
                initial_transform="identity",
                reg_iterations=(4, 2),
                scales=(2, 1),
                lncc_radius=2,
                return_inverse=False,
                write_transforms=False,
                return_warped=False,
                digital_diffeomorphism=False,
                sync_free=True,
            )
        assert ctr_zero.d2h_bytes == 0
        assert ctr_zero.sync_count == 0


def test_robust_affine_and_syn_in_memory_chaining():
    fi, _ = _make_oblique_phantom(shape=(24, 26, 24), shift=(0.0, 0.0, 0.0))
    mi, _ = _make_oblique_phantom(shape=(24, 26, 24), shift=(0.07, -0.05, 0.04))

    aff_mem = syntx.robust_affine(
        fi,
        mi,
        dof="affine",
        iterations=(15, 10),
        levels=(2, 1),
        multi_start=False,
        write_transforms=False,
    )
    aff_disk = syntx.robust_affine(
        fi,
        mi,
        dof="affine",
        iterations=(15, 10),
        levels=(2, 1),
        multi_start=False,
        write_transforms=True,
    )
    assert aff_mem["fwdtransforms"] == []
    assert aff_mem["fwd_chain"].has_in_memory()
    w_aff_mem = apply_transforms(fi, mi, aff_mem["fwd_chain"])
    w_aff_disk = apply_transforms(fi, mi, aff_disk["fwdtransforms"])
    np.testing.assert_allclose(w_aff_mem.numpy(), w_aff_disk.numpy(), atol=1e-4)

    syn_mem = syntx.syn(
        fi,
        mi,
        initial_transform=aff_mem["fwd_chain"],
        reg_iterations=(3, 2),
        levels=(2, 1),
        write_transforms=False,
        sync_free=True,
    )
    assert syn_mem["fwdtransforms"] == []
    assert syn_mem["fwd_chain"].has_in_memory()
    jac_syn = liouville_determinant(syn_mem, fi)
    assert float((jac_syn.numpy() <= 0).mean()) == 0.0


def test_fixed_target_parity_in_greedy_and_robust_affine():
    fi, _ = _make_oblique_phantom(shape=(24, 28, 24), shift=(0.0, 0.0, 0.0))
    mi, _ = _make_oblique_phantom(shape=(24, 28, 24), shift=(0.06, -0.04, 0.05))

    target = FixedTarget.from_image(
        fi,
        scales=(2, 1),
        window_size=5,
    )
    g_ref = syntx.greedy(
        fi,
        mi,
        reg_iterations=(5, 3),
        scales=(2, 1),
        lncc_radius=2,
        return_inverse=False,
    )
    g_tgt = syntx.greedy(
        target,
        mi,
        reg_iterations=(5, 3),
        scales=(2, 1),
        lncc_radius=2,
        return_inverse=False,
    )
    np.testing.assert_allclose(g_tgt["warpedmovout"].numpy(), g_ref["warpedmovout"].numpy(), atol=1e-4)

    a_ref = syntx.robust_affine(
        fi,
        mi,
        dof="affine",
        iterations=(15, 10),
        levels=(2, 1),
        multi_start=False,
    )
    a_tgt = syntx.robust_affine(
        target,
        mi,
        dof="affine",
        iterations=(15, 10),
        levels=(2, 1),
        multi_start=False,
    )
    np.testing.assert_allclose(a_tgt["warpedmovout"].numpy(), a_ref["warpedmovout"].numpy(), atol=1e-4)


def test_batched_mattes_mi_forward_and_grad_parity():
    torch.manual_seed(42)
    dev = _accel_device()
    p_cnt = 512
    m_cnt = 5
    num_bins = 32
    i_fix = torch.rand(p_cnt, device=dev, dtype=torch.float32)
    i_mov_raw = torch.rand(m_cnt, p_cnt, device=dev, dtype=torch.float32)

    i_batch = i_mov_raw.clone().requires_grad_(True)
    fw = parzen_weights(i_fix * 2.0 - 1.0, num_bins=num_bins, min_val=-1.0, max_val=1.0, pad=2.0)
    losses_batched = batched_mattes_mi_loss_shared_fixed(
        i_batch,
        fw,
        num_bins=num_bins,
        fixed_range=(0.0, 1.0),
    )
    losses_batched.sum().backward()

    ref_losses = []
    ref_grads = []
    for m in range(m_cnt):
        im = i_mov_raw[m].clone().requires_grad_(True)
        lm = mattes_mi_loss_nd(
            im,
            i_fix,
            num_bins=num_bins,
            auto_mask=False,
            fixed_range=(0.0, 1.0),
        )
        lm.backward()
        ref_losses.append(lm.detach())
        ref_grads.append(im.grad.detach())
    ref_losses_t = torch.stack(ref_losses)
    ref_grads_t = torch.stack(ref_grads)

    np.testing.assert_allclose(
        losses_batched.detach().cpu().numpy(),
        ref_losses_t.cpu().numpy(),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        i_batch.grad.detach().cpu().numpy(),
        ref_grads_t.cpu().numpy(),
        atol=1e-6,
    )


@pytest.mark.parametrize("mode", ["cfl", "reg_adam"])
def test_field_step_optimizer_batch_invariance(mode: str):
    torch.manual_seed(99)
    dev = _accel_device()
    b_sz = 3
    shape1 = (8, 10, 8)
    shape2 = (16, 20, 16)
    spacing = (1.1, 1.0, 1.2)

    g1 = torch.randn(b_sz, *shape1, 3, device=dev, dtype=torch.float32)
    # Give each batch item a wildly different gradient magnitude to verify zero cross-item coupling.
    g1[0] *= 10.0
    g1[2] *= 0.1
    g2 = torch.randn(b_sz, *shape2, 3, device=dev, dtype=torch.float32)
    g2[0] *= 5.0
    g2[1] *= 0.25

    w_batched = torch.zeros(b_sz, *shape1, 3, device=dev, dtype=torch.float32)
    opt_b = FieldStepOptimizer(mode=mode, cfl_momentum=0.5, sobolev_alpha=1.0)
    d1_b = opt_b.step(g1.clone(), 0.25, grid_shape=shape1, spacing=spacing)
    w_batched = w_batched + d1_b
    w_batched_up = torch.nn.functional.interpolate(
        w_batched.permute(0, 4, 1, 2, 3),
        size=shape2,
        mode="trilinear",
        align_corners=True,
    ).permute(0, 2, 3, 4, 1).contiguous()
    opt_b.on_level_change(shape2)
    d2_b = opt_b.step(g2.clone(), 0.25, grid_shape=shape2, spacing=spacing)
    w_batched_up = w_batched_up + d2_b

    for b in range(b_sz):
        wb = torch.zeros(1, *shape1, 3, device=dev, dtype=torch.float32)
        opt_1 = FieldStepOptimizer(mode=mode, cfl_momentum=0.5, sobolev_alpha=1.0)
        d1_1 = opt_1.step(g1[b : b + 1].clone(), 0.25, grid_shape=shape1, spacing=spacing)
        wb = wb + d1_1
        wb_up = torch.nn.functional.interpolate(
            wb.permute(0, 4, 1, 2, 3),
            size=shape2,
            mode="trilinear",
            align_corners=True,
        ).permute(0, 2, 3, 4, 1).contiguous()
        opt_1.on_level_change(shape2)
        d2_1 = opt_1.step(g2[b : b + 1].clone(), 0.25, grid_shape=shape2, spacing=spacing)
        wb_up = wb_up + d2_1
        np.testing.assert_allclose(
            w_batched_up[b].detach().cpu().numpy(),
            wb_up[0].detach().cpu().numpy(),
            atol=1e-6,
        )


def test_batch_robust_affine_and_batch_greedy_cohort_parity():
    fi, _ = _make_oblique_phantom(shape=(24, 28, 24), shift=(0.0, 0.0, 0.0))
    m0, _ = _make_oblique_phantom(shape=(24, 28, 24), shift=(0.08, -0.05, 0.04))
    m1, _ = _make_oblique_phantom(shape=(24, 28, 24), shift=(-0.06, 0.07, -0.03), scale=(1.04, 0.97, 1.02))
    # Heterogeneous shape & spacing on subject 2
    m2, _ = _make_oblique_phantom(
        shape=(22, 26, 20),
        shift=(0.05, 0.04, -0.06),
        spacing=(1.3, 1.1, 1.5),
    )
    mov_list = [m0, m1, m2]

    target = FixedTarget.from_image(
        fi,
        scales=(2, 1),
        window_size=5,
    )
    aff_batch = batch_robust_affine(
        target,
        mov_list,
        dof="affine",
        iterations=(20, 10),
        levels=(2, 1),
        multi_start=False,
        write_transforms=False,
        return_warped=True,
    )
    assert len(aff_batch) == 3
    for k, mov_k in enumerate(mov_list):
        c0 = float(np.corrcoef(fi.numpy().ravel(), ants.resample_image_to_target(mov_k, fi).numpy().ravel())[0, 1])
        c1 = float(np.corrcoef(fi.numpy().ravel(), aff_batch[k]["warpedmovout"].numpy().ravel())[0, 1])
        assert c1 > c0

    init_chains = [a["fwd_chain"] for a in aff_batch]
    g_batch = batch_greedy(
        target,
        mov_list,
        initial_transforms=init_chains,
        reg_iterations=(6, 4),
        scales=(2, 1),
        lncc_radius=2,
        return_inverse=False,
        write_transforms=False,
        return_warped=True,
        sync_free=True,
    )
    assert len(g_batch) == 3
    for k, mov_k in enumerate(mov_list):
        g_seq = syntx.greedy(
            target,
            mov_k,
            initial_transform=init_chains[k],
            reg_iterations=(6, 4),
            scales=(2, 1),
            lncc_radius=2,
            return_inverse=False,
            write_transforms=False,
            return_warped=True,
            sync_free=True,
        )
        np.testing.assert_allclose(
            g_batch[k]["warpedmovout"].numpy(),
            g_seq["warpedmovout"].numpy(),
            atol=5e-5,
        )
        jac = liouville_determinant(g_batch[k], fi)
        assert float((jac.numpy() <= 0).mean()) == 0.0


def test_greedy_amp_and_sync_free():
    fi, _ = _make_oblique_phantom(shape=(24, 28, 24), shift=(0.0, 0.0, 0.0))
    mi, _ = _make_oblique_phantom(shape=(24, 28, 24), shift=(0.08, -0.06, 0.05))

    g_fp32 = syntx.greedy(
        fi,
        mi,
        reg_iterations=(6, 4),
        scales=(2, 1),
        lncc_radius=2,
        amp=False,
        sync_free=True,
        return_inverse=False,
        write_transforms=False,
    )
    g_amp = syntx.greedy(
        fi,
        mi,
        reg_iterations=(6, 4),
        scales=(2, 1),
        lncc_radius=2,
        amp=True,
        sync_free=True,
        return_inverse=False,
        write_transforms=False,
    )
    r = float(np.corrcoef(g_fp32["warpedmovout"].numpy().ravel(), g_amp["warpedmovout"].numpy().ravel())[0, 1])
    assert r > 0.999
    jac_amp = liouville_determinant(g_amp, fi)
    assert float((jac_amp.numpy() <= 0).mean()) == 0.0


def test_build_template_gpu_accumulation_in_memory():
    m0, _ = _make_oblique_phantom(shape=(22, 24, 22), shift=(0.05, -0.04, 0.03))
    m1, _ = _make_oblique_phantom(shape=(22, 24, 22), shift=(-0.05, 0.04, -0.03))
    m2, _ = _make_oblique_phantom(shape=(22, 24, 22), shift=(0.0, 0.02, -0.02))

    res_mem = syntx.build_template(
        image_list=[m0, m1, m2],
        iterations=2,
        type_of_transform="Greedy",
        reg_iterations=[5, 3],
        scales=(2, 1),
        output_dir=None,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        res_disk = syntx.build_template(
            image_list=[m0, m1, m2],
            iterations=2,
            type_of_transform="Greedy",
            reg_iterations=[5, 3],
            scales=(2, 1),
            output_dir=tmpdir,
        )
        r_tpl = float(
            np.corrcoef(
                res_mem["template"].numpy().ravel(),
                res_disk["template"].numpy().ravel(),
            )[0, 1]
        )
        assert r_tpl > 0.999
        assert len(res_mem["warped_images"]) == 3
        assert len(res_mem["fwdtransforms"]) == 3
        assert all(os.path.exists(p) for chain in res_disk["fwdtransforms"] for p in chain)


def test_batch_apply_transforms_multimap_and_multisubject_parity():
    fi, _ = _make_oblique_phantom(shape=(22, 26, 20), shift=(0.0, 0.0, 0.0))
    m0, seg0 = _make_oblique_phantom(shape=(22, 26, 20), shift=(0.06, -0.05, 0.04))
    m1, _ = _make_oblique_phantom(shape=(22, 26, 20), shift=(-0.05, 0.04, -0.03))
    m2, _ = _make_oblique_phantom(shape=(20, 22, 18), shift=(0.03, 0.02, -0.04), spacing=(1.3, 1.1, 1.4))

    reg0 = syntx.greedy(fi, m0, reg_iterations=(4, 2), scales=(2, 1), write_transforms=False, return_warped=False)
    chain0 = reg0["fwd_chain"]

    # Mode A: 1 TransformChain applied to 4 downstream scalar maps of subject 0 (including heterogeneous geometry m2)
    maps_in = [m0, m0 * 0.5 + 0.1, m1, m2]
    warped_batch = syntx.batch_apply_transforms(fi, maps_in, chain0, interpolator="linear", max_channels_per_pass=2)
    assert len(warped_batch) == 4
    for k, mk in enumerate(maps_in):
        ref_k = apply_transforms(fi, mk, chain0, interpolator="linear")
        np.testing.assert_allclose(warped_batch[k].numpy(), ref_k.numpy(), atol=1e-6)

    # Nearest-neighbor multi-map parity (e.g. parcellation + mask)
    lbl_batch = syntx.batch_apply_transforms(fi, [seg0, (seg0 > 0).astype("float32")], chain0, interpolator="nearestNeighbor")
    for k, lk in enumerate([seg0, (seg0 > 0).astype("float32")]):
        ref_lk = apply_transforms(fi, lk, chain0, interpolator="nearestNeighbor")
        np.testing.assert_allclose(lbl_batch[k].numpy(), ref_lk.numpy(), atol=1e-6)

    # Mode B: B TransformChains applied to B subjects onto shared fixed template
    reg1 = syntx.greedy(fi, m1, reg_iterations=(4, 2), scales=(2, 1), write_transforms=False, return_warped=False)
    reg2 = syntx.greedy(fi, m2, reg_iterations=(4, 2), scales=(2, 1), write_transforms=False, return_warped=False)
    cohort_warped = syntx.batch_apply_transforms(
        fi,
        [m0, m1, m2],
        [reg0["fwd_chain"], reg1["fwd_chain"], reg2["fwd_chain"]],
    )
    for m_in, r_in, w_out in zip([m0, m1, m2], [reg0, reg1, reg2], cohort_warped):
        w_ref = apply_transforms(fi, m_in, r_in["fwd_chain"])
        np.testing.assert_allclose(w_out.numpy(), w_ref.numpy(), atol=1e-6)


def test_tvf_and_syngs_gpu_resident_fixed_target_and_sync_free():
    fi, seg_f = _make_oblique_phantom(shape=(22, 24, 20), shift=(0.0, 0.0, 0.0))
    mi, seg_m = _make_oblique_phantom(shape=(22, 24, 20), shift=(0.06, -0.04, 0.04))
    target = FixedTarget.from_image(fi, scales=(2, 1), window_size=5)

    tvf_mem = syntx.tvf(
        target,
        mi,
        reg_iterations=[3, 2],
        levels=[2, 1],
        affine_mode="com_only",
        write_transforms=False,
        return_warped=False,
        sync_free=True,
    )
    assert tvf_mem["fwdtransforms"] == []
    assert tvf_mem["warpedmovout"] is None
    assert tvf_mem["fwd_chain"].has_in_memory()
    jac_tvf = liouville_determinant(tvf_mem, fi)
    assert float((jac_tvf.numpy() <= 0).mean()) == 0.0
    _, _, dice_tvf = compute_bidirectional_dice(
        seg_f, seg_m, fi, mi, tvf_mem["fwd_chain"], tvf_mem["inv_chain"]
    )
    assert 0.0 < dice_tvf <= 1.0
    assert tvf_mem["inverse_identity_errors"]["phi_1"]["error_map"].shape is not None

    syngs_mem = syntx.syngs(
        target,
        mi,
        reg_iterations=[3, 2],
        levels=[2, 1],
        affine_mode="com_only",
        write_transforms=False,
        return_warped=False,
        sync_free=True,
    )
    assert syngs_mem["fwdtransforms"] == []
    assert syngs_mem["warpedmovout"] is None
    assert syngs_mem["fwd_chain"].has_in_memory()
    jac_syngs = liouville_determinant(syngs_mem, fi)
    assert float((jac_syngs.numpy() <= 0).mean()) == 0.0
    _, _, dice_syngs = compute_bidirectional_dice(
        seg_f, seg_m, fi, mi, syngs_mem["fwd_chain"], syngs_mem["inv_chain"]
    )
    assert 0.0 < dice_syngs <= 1.0
    assert syngs_mem["inverse_identity_errors"]["phi_1"]["error_map"].shape is not None


def test_build_template_batched_zero_disk_mode():
    m0, _ = _make_oblique_phantom(shape=(22, 24, 22), shift=(0.05, -0.04, 0.03))
    m1, _ = _make_oblique_phantom(shape=(22, 24, 22), shift=(-0.05, 0.04, -0.03))
    m2, _ = _make_oblique_phantom(shape=(22, 24, 22), shift=(0.0, 0.02, -0.02))

    res_seq = syntx.build_template(
        image_list=[m0, m1, m2],
        iterations=2,
        type_of_transform="Greedy",
        reg_iterations=[5, 3],
        scales=(2, 1),
        batch_size=1,
        write_transforms=False,
        return_warped=False,
        sync_free=True,
        output_dir=None,
    )
    res_bat = syntx.build_template(
        image_list=[m0, m1, m2],
        iterations=2,
        type_of_transform="Greedy",
        reg_iterations=[5, 3],
        scales=(2, 1),
        batch_size=3,
        write_transforms=False,
        return_warped=False,
        amp=True,
        sync_free=True,
        output_dir=None,
    )
    assert all(len(tx) == 0 for tx in res_bat["fwdtransforms"])
    assert all(ch is not None and ch.has_in_memory() for ch in res_bat["fwd_chains"])
    r = float(np.corrcoef(res_seq["template"].numpy().ravel(), res_bat["template"].numpy().ravel())[0, 1])
    assert r > 0.995
    syntx.clear_transform_gpu_caches(0)
