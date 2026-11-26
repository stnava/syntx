"""R55 verification tests: fused Metal/ATen kernels, register-Parzen MI, Dual-Grid ``control_stride``,
``batch_syn``, warm-started ``build_template``, and ``hybrid_greedy`` / ``hybrid_syn``."""

from __future__ import annotations

import os

import ants
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import syntx
from syntx.core.inverse import update_inverse_field_nd_anderson
from syntx.core.jacobian import compute_physical_jacobian_determinant
from syntx.core.kernels import (
    RegisterParzenMattesMIFunction,
    exact_lncc_1sided_affine_warp_loss_and_grad_3d,
    exact_lncc_2sided_loss_and_grad_3d,
    exact_mattes_mi_1sided_affine_warp_loss_and_grad_3d,
    exact_mattes_mi_2sided_loss_and_grad_3d,
    fused_box_mean_3d,
    fused_box_mean_adjoint_3d,
    fused_genericlabel_8corner_3d,
    fused_jacobian_det_3d,
    has_mps_shader_backend,
    prepack_fixed_parzen_sparse4,
)
from syntx.core.losses import batched_mattes_mi_from_weights, get_similarity_loss, mattes_mi_loss_nd, parzen_weights
from syntx.core.losses.cross_correlation import _get_window_count_views
from syntx.core.losses.mutual_information import BatchedMattesMIFunction, get_4tap_splines


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_oblique_phantom_pair(shape=(36, 40, 36), seed: int = 7):
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[1], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[2], dtype=np.float32),
        indexing="ij",
    )
    r2 = xx**2 + 1.25 * yy**2 + 0.85 * zz**2
    base = np.exp(-3.2 * r2) * (0.65 + 0.35 * np.cos(4.0 * xx) * np.cos(3.5 * yy))
    base = np.clip(base, 0.0, 1.0).astype(np.float32)

    ang = 0.14
    c, s = float(np.cos(ang)), float(np.sin(ang))
    direction = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    spacing = (2.0, 2.0, 2.2)
    origin = (-34.0, -38.0, -36.0)

    fi = ants.from_numpy(base, origin=origin, spacing=spacing, direction=direction)

    r2_m = (xx - 0.08) ** 2 + 1.25 * (yy + 0.07) ** 2 + 0.85 * (zz - 0.05) ** 2
    mov_arr = np.exp(-3.2 * r2_m) * (
        0.65 + 0.35 * np.cos(4.0 * (xx - 0.08)) * np.cos(3.5 * (yy + 0.07))
    )
    mov_arr = np.clip(mov_arr + 0.005 * rng.standard_normal(shape, dtype=np.float32), 0.0, 1.0)
    mi = ants.from_numpy(mov_arr.astype(np.float32), origin=origin, spacing=spacing, direction=direction)
    return fi, mi


def test_fused_jacobian_det_3d_oblique_and_boundary_parity():
    dev = _best_device()
    torch.manual_seed(11)
    disp = 0.25 * torch.randn((2, 18, 20, 22, 3), device=dev, dtype=torch.float32)
    disp.is_physical = True
    ang = 0.23
    c, s = float(np.cos(ang)), float(np.sin(ang))
    direction = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    spacing = (1.5, 1.7, 2.2)

    det_fast = compute_physical_jacobian_determinant(disp, direction=direction, spacing=spacing)
    os.environ["SYNTX_DISABLE_CUSTOM_KERNELS"] = "1"
    try:
        det_ref = compute_physical_jacobian_determinant(disp, direction=direction, spacing=spacing)
    finally:
        os.environ.pop("SYNTX_DISABLE_CUSTOM_KERNELS", None)

    max_diff = float((det_fast - det_ref).abs().max().item())
    assert max_diff < 5e-6, f"compute_physical_jacobian_determinant max_diff={max_diff:.3e} >= 5e-6"

    # Direct fused_jacobian_det_3d + autograd fallback check
    disp_g = (0.1 * torch.randn((1, 6, 6, 6, 3), dtype=torch.float64)).requires_grad_(True)
    assert fused_jacobian_det_3d(disp_g, (1.5, 1.5, 1.5)) is None
    assert torch.autograd.gradcheck(
        lambda u: compute_physical_jacobian_determinant(u, direction=direction, spacing=spacing).sum(),
        (disp_g,),
        eps=1e-6,
        atol=1e-4,
    )


def test_fused_box_mean_and_adjoint_identity():
    dev = _best_device()
    torch.manual_seed(19)
    x = torch.randn((2, 2, 16, 18, 20), device=dev, dtype=torch.float32)
    y = torch.randn_like(x)
    ws = 5
    _, _, pad_ratios = _get_window_count_views((16, 18, 20), ws, dev, torch.float32)

    bx_f = fused_box_mean_3d(x, ws, count_include_pad=False)
    bx_cpu = fused_box_mean_3d(x.cpu(), ws, count_include_pad=False).to(dev)
    assert float((bx_f - bx_cpu).abs().max().item()) < 5e-6

    adj_y = fused_box_mean_adjoint_3d(y, ws, pad_ratios)
    lhs = float((bx_f * y).sum().item())
    rhs = float((x * adj_y).sum().item())
    rel_err = abs(lhs - rhs) / max(abs(lhs), 1.0)
    assert rel_err < 1e-5, f"Adjoint inner-product mismatch: lhs={lhs}, rhs={rhs}, rel_err={rel_err}"


def test_exact_lncc_1sided_and_2sided_autograd_parity():
    dev = _best_device()
    torch.manual_seed(23)
    shape = (18, 20, 22)
    ws = 5
    _, _, pad_ratios = _get_window_count_views(shape, ws, dev, torch.float32)
    loss_mod = get_similarity_loss("cc2", window_size=ws).to(dev)

    fi = torch.rand((1, 1, *shape), device=dev, dtype=torch.float32)
    mi = torch.rand((2, 1, *shape), device=dev, dtype=torch.float32)
    id_grid = F.affine_grid(torch.eye(3, 4, device=dev).unsqueeze(0), [1, 1, *shape], align_corners=True)
    warp = (0.04 * torch.randn((2, *shape, 3), device=dev, dtype=torch.float32)).requires_grad_(True)
    theta = torch.eye(3, 4, device=dev, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
    theta[:, :, :3] += 0.02 * torch.randn((2, 3, 3), device=dev)
    theta[:, :, 3] += 0.03 * torch.randn((2, 3), device=dev)

    with torch.no_grad():
        j_mean = fused_box_mean_3d(fi, ws, count_include_pad=False)
        m_c = fi - j_mean
        j_var = torch.clamp(fused_box_mean_3d(m_c.square(), ws, count_include_pad=False), min=1e-6)

    l_fast, g_fast = exact_lncc_1sided_affine_warp_loss_and_grad_3d(
        mi, id_grid, warp.detach(), theta[:, :, :3].contiguous(), theta[:, :, 3].contiguous(),
        False, m_c, j_var, ws, pad_ratios, squared=True, padding_mode="border",
    )

    c_clamped = torch.clamp(id_grid + warp, -1.0, 1.0)
    sg = torch.bmm(c_clamped.view(2, -1, 3), theta[:, :, :3].transpose(1, 2)).view(2, *shape, 3) + theta[:, :, 3].view(2, 1, 1, 1, 3)
    moved_ref = F.grid_sample(mi, sg, mode="bilinear", padding_mode="border", align_corners=True)
    l_ref = loss_mod(moved_ref[0:1], fi) + loss_mod(moved_ref[1:2], fi)
    l_ref.backward()

    assert float((l_fast - l_ref.detach()).abs().item()) < 5e-6
    assert float((g_fast - warp.grad).abs().max().item()) < 5e-6

    gi = (id_grid.expand(2, -1, -1, -1, -1) + 0.04 * torch.randn((2, *shape, 3), device=dev)).detach().requires_grad_(True)
    gj = (id_grid.expand(2, -1, -1, -1, -1) + 0.04 * torch.randn((2, *shape, 3), device=dev)).detach().requires_grad_(True)
    in_b = (((gi >= -1.0) & (gi <= 1.0)).all(dim=-1) & ((gj >= -1.0) & (gj <= 1.0)).all(dim=-1)).unsqueeze(1).float()

    l_vec_fast, g_gi_fast, g_gj_fast = exact_lncc_2sided_loss_and_grad_3d(
        fi, gi.detach(), mi, gj.detach(), ws, pad_ratios,
        squared=True, in_bounds_mask=in_b, padding_mode="border", return_per_batch_loss=True,
    )
    im_ref = F.grid_sample(fi.expand(2, -1, -1, -1, -1), gi, mode="bilinear", padding_mode="border", align_corners=True)
    jm_ref = F.grid_sample(mi, gj, mode="bilinear", padding_mode="border", align_corners=True)
    l0 = loss_mod(im_ref[0:1], jm_ref[0:1], mask=in_b[0:1])
    l1 = loss_mod(im_ref[1:2], jm_ref[1:2], mask=in_b[1:2])
    (l0 + l1).backward()

    assert float((l_vec_fast[0] - l0.detach()).abs().item()) < 5e-6
    assert float((l_vec_fast[1] - l1.detach()).abs().item()) < 5e-6
    assert float((g_gi_fast - gi.grad).abs().max().item()) < 5e-6
    assert float((g_gj_fast - gj.grad).abs().max().item()) < 5e-6


def test_fused_genericlabel_8corner_3d_exact_voting():
    dev = _best_device()
    rng = np.random.default_rng(31)
    fi, _ = _make_oblique_phantom_pair(shape=(20, 22, 20), seed=31)
    lbl_arr = rng.integers(0, 15, size=fi.shape, dtype=np.int32).astype(np.float32)
    lbl_img = fi.new_image_like(lbl_arr)
    tx_el = syntx.transforms.TransformElement.from_affine(
        np.eye(3, dtype=np.float64), np.array([1.3, -0.9, 0.7], dtype=np.float64), invert=False
    )
    chain = syntx.transforms.TransformChain([tx_el])

    out_fast = syntx.transforms.apply_transforms(
        fixed=fi, moving=lbl_img, transformlist=chain, interpolator="genericLabel", device=str(dev)
    )
    os.environ["SYNTX_DISABLE_CUSTOM_KERNELS"] = "1"
    try:
        out_ref = syntx.transforms.apply_transforms(
            fixed=fi, moving=lbl_img, transformlist=chain, interpolator="genericLabel", device=str(dev)
        )
    finally:
        os.environ.pop("SYNTX_DISABLE_CUSTOM_KERNELS", None)

    n_diff = int(np.sum(out_fast.numpy() != out_ref.numpy()))
    assert n_diff == 0, f"genericLabel mismatch vs reference: {n_diff} voxels"
    assert fused_genericlabel_8corner_3d(torch.zeros((4, 4, 4)), torch.zeros((5, 3))) is None


def test_fused_anderson_and_register_parzen_mi():
    dev = _best_device()
    torch.manual_seed(37)
    w_disp = 0.08 * torch.randn((2, 16, 18, 16, 3), device=dev, dtype=torch.float32)
    inv_fast = update_inverse_field_nd_anderson(
        W_disp=w_disp, W_inv_disp=None, steps=8, m=5, spacing=(1.5, 1.5, 1.5),
        max_error_threshold=-1.0, mean_error_threshold=-1.0,
    )
    inv_fast2 = update_inverse_field_nd_anderson(
        W_disp=w_disp, W_inv_disp=None, steps=8, m=5, spacing=(1.5, 1.5, 1.5),
        max_error_threshold=-1.0, mean_error_threshold=-1.0,
    )
    assert torch.equal(inv_fast, inv_fast2)

    os.environ["SYNTX_DISABLE_CUSTOM_KERNELS"] = "1"
    try:
        inv_ref = update_inverse_field_nd_anderson(
            W_disp=w_disp, W_inv_disp=None, steps=8, m=5, spacing=(1.5, 1.5, 1.5),
            max_error_threshold=-1.0, mean_error_threshold=-1.0,
        )
    finally:
        os.environ.pop("SYNTX_DISABLE_CUSTOM_KERNELS", None)
    assert float((inv_fast - inv_ref).abs().max().item()) < 5e-5

    # RegisterParzenMattesMIFunction parity & bitwise determinism
    s_pts = 8192
    f_vals = torch.rand(s_pts, device=dev, dtype=torch.float32) * 2.0 - 1.0
    f_idx4, f_w4, _ = get_4tap_splines(f_vals, 32, -1.0, 1.0)
    w_dense = torch.zeros((s_pts, 32), device=dev, dtype=torch.float32)
    w_dense.scatter_add_(1, f_idx4, f_w4)
    f_b0, f_w4_packed, _ = prepack_fixed_parzen_sparse4(f_vals, num_bins=32, min_val=-1.0, max_val=1.0)

    mov_a = (torch.rand((3, s_pts), device=dev, dtype=torch.float32) * 2.0 - 1.0).detach().requires_grad_(True)
    mov_b = mov_a.detach().clone().requires_grad_(True)

    l_reg = RegisterParzenMattesMIFunction.apply(mov_a, f_b0, f_w4_packed, 32, -1.0, 1.0, 2.0, None)
    l_reg.sum().backward()

    l_ref_mi = BatchedMattesMIFunction.apply(mov_b, w_dense, 32, -1.0, 1.0, 2.0, 4096)
    l_ref_mi.sum().backward()

    assert float((l_reg.detach() - l_ref_mi.detach()).abs().max().item()) < 5e-6
    assert float((mov_a.grad - mov_b.grad).abs().max().item()) < 5e-6

    mov_c = mov_a.detach().clone().requires_grad_(True)
    l_reg2 = RegisterParzenMattesMIFunction.apply(mov_c, f_b0, f_w4_packed, 32, -1.0, 1.0, 2.0, None)
    l_reg2.sum().backward()
    assert torch.equal(l_reg.detach(), l_reg2.detach())
    assert torch.equal(mov_a.grad, mov_c.grad)

    # Per-frame fixed + point_weights parity (4-D motion_batched path)
    f_pf = torch.rand((3, 2048), device=dev, dtype=torch.float32) * 2.0 - 1.0
    m_pf = (torch.rand((3, 2048), device=dev, dtype=torch.float32) * 2.0 - 1.0).detach().requires_grad_(True)
    m_pf_ref = m_pf.detach().clone().requires_grad_(True)
    w_pts = torch.rand((3, 2048), device=dev, dtype=torch.float32) + 0.1
    fi_pf_i4, fi_pf_w4, _ = prepack_fixed_parzen_sparse4(f_pf, num_bins=32, min_val=-1.0, max_val=1.0)
    l_pf = RegisterParzenMattesMIFunction.apply(m_pf, fi_pf_i4, fi_pf_w4, 32, -1.0, 1.0, 2.0, w_pts)
    l_pf.sum().backward()

    sw = torch.sqrt(w_pts).unsqueeze(-1)
    wx_ref = parzen_weights(m_pf_ref.reshape(-1), num_bins=32).reshape(3, 2048, 32) * sw
    wy_ref = parzen_weights(f_pf.reshape(-1), num_bins=32).reshape(3, 2048, 32) * sw
    l_pf_ref = batched_mattes_mi_from_weights(wx_ref, wy_ref)
    l_pf_ref.sum().backward()
    assert float((l_pf.detach() - l_pf_ref.detach()).abs().max().item()) < 5e-5
    assert float((m_pf.grad - m_pf_ref.grad).abs().max().item()) < 5e-5


def test_batch_mattes_mi_and_deformable_batch_mi():
    dev = _best_device()
    fi, mi1 = _make_oblique_phantom_pair(shape=(28, 32, 28), seed=61)
    _, mi2 = _make_oblique_phantom_pair(shape=(28, 32, 28), seed=62)

    # 1. Public syntx.batch_mattes_mi parity vs mattes_mi_loss_nd + differentiability
    scores_np = syntx.batch_mattes_mi(fi, [mi1, mi2], num_bins=32, fixed_range=(0.0, 1.0), device=str(dev))
    assert scores_np.shape == (2,)
    t_f = torch.from_numpy(fi.numpy().transpose(2, 1, 0)).to(dev).unsqueeze(0).unsqueeze(0)
    t_m1 = torch.from_numpy(mi1.numpy().transpose(2, 1, 0)).to(dev).unsqueeze(0).unsqueeze(0)
    t_m2 = torch.from_numpy(mi2.numpy().transpose(2, 1, 0)).to(dev).unsqueeze(0).unsqueeze(0)
    ref1 = float(mattes_mi_loss_nd(t_m1, t_f, num_bins=32, auto_mask=False, fixed_range=(0.0, 1.0)).item())
    ref2 = float(mattes_mi_loss_nd(t_m2, t_f, num_bins=32, auto_mask=False, fixed_range=(0.0, 1.0)).item())
    assert abs(float(scores_np[0]) - ref1) < 5e-6
    assert abs(float(scores_np[1]) - ref2) < 5e-6

    mov_stack = torch.cat([t_m1, t_m2], dim=0).detach().requires_grad_(True)
    scores_t = syntx.batch_mattes_mi(t_f, mov_stack, num_bins=32, fixed_range=(0.0, 1.0), return_tensor=True)
    scores_t.sum().backward()
    assert mov_stack.grad is not None and torch.isfinite(mov_stack.grad).all()

    # 2. exact_mattes_mi_2sided_loss_and_grad_3d parity vs autograd
    mi_mod = get_similarity_loss("mattes", num_bins=32, fixed_range=(0.0, 1.0)).to(dev)
    id_g = F.affine_grid(torch.eye(3, 4, device=dev).unsqueeze(0), [1, 1, *t_f.shape[2:]], align_corners=True)
    gi = (id_g.expand(2, -1, -1, -1, -1) + 0.03 * torch.randn((2, *t_f.shape[2:], 3), device=dev)).detach().requires_grad_(True)
    gj = (id_g.expand(2, -1, -1, -1, -1) + 0.03 * torch.randn((2, *t_f.shape[2:], 3), device=dev)).detach().requires_grad_(True)
    in_b = (((gi >= -1.0) & (gi <= 1.0)).all(dim=-1) & ((gj >= -1.0) & (gj <= 1.0)).all(dim=-1)).unsqueeze(1).float()

    lv_f, ggi_f, ggj_f = exact_mattes_mi_2sided_loss_and_grad_3d(
        t_f, gi.detach(), torch.cat([t_m1, t_m2], dim=0), gj.detach(),
        num_bins=32, fixed_range=(0.0, 1.0), in_bounds_mask=in_b, padding_mode="border", return_per_batch_loss=True,
    )
    im_r = F.grid_sample(t_f.expand(2, -1, -1, -1, -1), gi, mode="bilinear", padding_mode="border", align_corners=True)
    jm_r = F.grid_sample(torch.cat([t_m1, t_m2], dim=0), gj, mode="bilinear", padding_mode="border", align_corners=True)
    lr0 = mi_mod(im_r[0:1], jm_r[0:1], mask=in_b[0:1])
    lr1 = mi_mod(im_r[1:2], jm_r[1:2], mask=in_b[1:2])
    (lr0 + lr1).backward()
    assert float((lv_f[0] - lr0.detach()).abs().item()) < 5e-5
    assert float((lv_f[1] - lr1.detach()).abs().item()) < 5e-5
    assert float((ggi_f - gi.grad).abs().max().item()) < 5e-5
    assert float((ggj_f - gj.grad).abs().max().item()) < 5e-5

    # 3. Multi-modal batch_greedy & batch_syn with Mattes MI (0.0000% folding)
    ft = syntx.FixedTarget.from_image(fi)
    bg_mi = syntx.batch_greedy(
        ft, [mi1, mi2], initial_transforms=["identity", "identity"],
        similarity_metric="mattes", scales=[2, 1], reg_iterations=[6, 4],
        batch_size=2, write_transforms=False, return_warped=False, sync_free=True,
    )
    bs_mi = syntx.batch_syn(
        ft, [mi1, mi2], type_of_transform="SyNOnly",
        syn_metric="mattes", levels=[2, 1], reg_iterations=[5, 3],
        batch_size=2, write_transforms=False, return_warped=False, sync_free=True,
    )
    for res_entry in (*bg_mi, *bs_mi):
        d_f = res_entry["fwd_chain"][0].disp_tensor_zyx()
        det_j = compute_physical_jacobian_determinant(d_f, direction=fi.direction, spacing=fi.spacing)
        assert float((det_j <= 0.0).float().mean().item()) == 0.0
        _ = exact_mattes_mi_1sided_affine_warp_loss_and_grad_3d  # verify import symbol


def test_control_stride_batch_syn_and_warm_template():
    fi, mi1 = _make_oblique_phantom_pair(shape=(32, 36, 32), seed=41)
    _, mi2 = _make_oblique_phantom_pair(shape=(32, 36, 32), seed=42)
    _, mi3 = _make_oblique_phantom_pair(shape=(32, 36, 32), seed=43)

    res_g_cs2 = syntx.greedy(
        fi, mi1, initial_transform="identity", scales=[2, 1], reg_iterations=[8, 6],
        control_stride=2, write_transforms=False, return_warped=False, sync_free=True,
    )
    disp_g = res_g_cs2["fwd_chain"][0].disp_tensor_zyx()
    det_g = compute_physical_jacobian_determinant(disp_g, direction=fi.direction, spacing=fi.spacing)
    assert float((det_g <= 0.0).float().mean().item()) == 0.0

    res_s_cs2 = syntx.syn(
        fi, mi1, type_of_transform="SyNOnly", levels=[2, 1], reg_iterations=[6, 4],
        control_stride=2, write_transforms=False, return_warped=False, export_midpoint=False, sync_free=True,
    )
    disp_s = res_s_cs2["fwd_chain"][0].disp_tensor_zyx()
    det_s = compute_physical_jacobian_determinant(disp_s, direction=fi.direction, spacing=fi.spacing)
    assert float((det_s <= 0.0).float().mean().item()) == 0.0

    ft = syntx.FixedTarget.from_image(fi)
    bs_12 = syntx.batch_syn(
        ft, [mi1, mi2], type_of_transform="SyNOnly", levels=[2, 1], reg_iterations=[8, 5],
        batch_size=2, write_transforms=False, return_warped=True,
    )
    bs_13 = syntx.batch_syn(
        ft, [mi1, mi3], type_of_transform="SyNOnly", levels=[2, 1], reg_iterations=[8, 5],
        batch_size=2, write_transforms=False, return_warped=False,
    )
    assert len(bs_12) == 2
    for entry in bs_12:
        d_f = entry["fwd_chain"][0].disp_tensor_zyx()
        det_b = compute_physical_jacobian_determinant(d_f, direction=fi.direction, spacing=fi.spacing)
        assert float((det_b <= 0.0).float().mean().item()) == 0.0
        liouv = syntx.liouville_determinant(entry, fi)
        assert liouv.shape == fi.shape
    d0_a = bs_12[0]["fwd_chain"][0].disp_tensor_zyx()
    d0_b = bs_13[0]["fwd_chain"][0].disp_tensor_zyx()
    assert float((d0_a - d0_b).abs().max().item()) < 1e-4

    tmpl_g = syntx.build_template(
        image_list=[fi, mi1, mi2],
        iterations=2,
        type_of_transform="Greedy",
        initial_transforms=["identity", "identity", "identity"],
        scales=[2, 1],
        reg_iterations=[10, 6],
        warm_start_iterations=True,
        continuation_iterations=[5, 4],
        write_transforms=False,
        return_warped=False,
        sync_free=True,
    )
    assert tmpl_g["n_iterations"] == 2
    assert np.isfinite(tmpl_g["template"].numpy()).all()

    tmpl_syn = syntx.build_template(
        image_list=[fi, mi1],
        iterations=2,
        type_of_transform="SyN",
        initial_transforms=["identity", "identity"],
        levels=[2, 1],
        reg_iterations=[6, 4],
        warm_start_iterations=True,
        continuation_iterations=[4, 3],
        write_transforms=False,
        return_warped=False,
        sync_free=True,
    )
    assert tmpl_syn["n_iterations"] == 2
    assert np.isfinite(tmpl_syn["template"].numpy()).all()


def test_hybrid_greedy_and_hybrid_syn():
    pytest.importorskip("voxelmorph")
    fi, mi = _make_oblique_phantom_pair(shape=(32, 32, 32), seed=55)
    dev_s = str(_best_device())

    m1, prov1 = syntx.get_cached_voxelmorph_3d(device=dev_s)
    m2, _ = syntx.get_cached_voxelmorph_3d(device=dev_s)
    assert m1 is m2
    assert "sha256" in prov1

    hg = syntx.hybrid_greedy(
        fi, mi, initial_transform="identity", scales=(2, 1), reg_iterations=(6, 4),
        write_transforms=False, return_warped=True, sync_free=True, device=dev_s,
    )
    assert hg["fwdtransforms"] == []
    assert len(hg["fwd_chain"]) >= 1
    det_hg = compute_physical_jacobian_determinant(
        hg["fwd_chain"][0].disp_tensor_zyx(), direction=fi.direction, spacing=fi.spacing
    )
    assert float((det_hg <= 0.0).float().mean().item()) == 0.0

    hs = syntx.hybrid_syn(
        fi, mi, initial_transform="identity", type_of_transform="SyNOnly",
        levels=(2, 1), reg_iterations=(5, 3),
        write_transforms=False, return_warped=True, sync_free=True, device=dev_s,
    )
    assert hs["fwdtransforms"] == []
    det_hs = compute_physical_jacobian_determinant(
        hs["fwd_chain"][0].disp_tensor_zyx(), direction=fi.direction, spacing=fi.spacing
    )
    assert float((det_hs <= 0.0).float().mean().item()) == 0.0
    assert has_mps_shader_backend("cpu") is False


def test_batch_cc_batch_cc2_and_batch_correlation():
    from syntx.core.losses import local_ncc_loss_nd

    dev = _best_device()
    fi, mi1 = _make_oblique_phantom_pair(shape=(28, 32, 28), seed=71)
    _, mi2 = _make_oblique_phantom_pair(shape=(28, 32, 28), seed=72)
    ft = syntx.FixedTarget.from_image(fi, device=dev)

    # 1. batch_cc & batch_cc2 exact parity vs image_compare / local_ncc_loss_nd
    cc_vals = syntx.batch_cc(ft, [mi1, mi2], window_size=5, device=str(dev))
    cc2_vals = syntx.batch_cc2(ft, [mi1, mi2], window_size=5, device=str(dev))
    corr_vals = syntx.batch_correlation(ft, [mi1, mi2], device=str(dev))
    assert cc_vals.shape == (2,) and cc2_vals.shape == (2,) and corr_vals.shape == (2,)

    assert abs(float(cc_vals[0]) - syntx.image_compare(fi, mi1, "cc", device=str(dev))) < 5e-6
    assert abs(float(cc_vals[1]) - syntx.image_compare(fi, mi2, "cc", device=str(dev))) < 5e-6
    assert abs(float(corr_vals[0]) - syntx.correlation(fi, mi1, device=str(dev))) < 5e-6
    assert abs(float(corr_vals[1]) - syntx.correlation(fi, mi2, device=str(dev))) < 5e-6

    t_f = torch.from_numpy(fi.numpy().transpose(2, 1, 0)).to(dev).unsqueeze(0).unsqueeze(0)
    t_m1 = torch.from_numpy(mi1.numpy().transpose(2, 1, 0)).to(dev).unsqueeze(0).unsqueeze(0)
    t_m2 = torch.from_numpy(mi2.numpy().transpose(2, 1, 0)).to(dev).unsqueeze(0).unsqueeze(0)
    ref_cc2_0 = float(local_ncc_loss_nd(t_m1, t_f, window_size=5, squared=True).item())
    ref_cc2_1 = float(local_ncc_loss_nd(t_m2, t_f, window_size=5, squared=True).item())
    assert abs(float(cc2_vals[0]) - ref_cc2_0) < 5e-6
    assert abs(float(cc2_vals[1]) - ref_cc2_1) < 5e-6

    # 2. Exact autograd gradient parity for BatchedLNCC3DFunction (both cc and cc2)
    for sq in (False, True):
        mov_a = torch.cat([t_m1, t_m2], dim=0).detach().clone().requires_grad_(True)
        mov_b = mov_a.detach().clone().requires_grad_(True)
        fn_b = syntx.batch_cc2 if sq else syntx.batch_cc
        lb = fn_b(t_f, mov_a, window_size=5, return_tensor=True)
        lb.sum().backward()

        lr0 = local_ncc_loss_nd(mov_b[0:1], t_f, window_size=5, squared=sq)
        lr1 = local_ncc_loss_nd(mov_b[1:2], t_f, window_size=5, squared=sq)
        (lr0 + lr1).backward()

        assert float((lb[0].detach() - lr0.detach()).abs().item()) < 5e-6
        assert float((lb[1].detach() - lr1.detach()).abs().item()) < 5e-6
        assert float((mov_a.grad - mov_b.grad).abs().max().item()) < 5e-6


def test_batch_tbm_and_batch_jacobian():
    dev = _best_device()
    fi, mi1 = _make_oblique_phantom_pair(shape=(28, 32, 28), seed=81)
    _, mi2 = _make_oblique_phantom_pair(shape=(28, 32, 28), seed=82)
    ft = syntx.FixedTarget.from_image(fi, scales=(2, 1), device=dev)

    tbm = syntx.batch_tbm(
        ft,
        [mi1, mi2],
        method="greedy",
        similarity_metric="cc2",
        do_log=True,
        include_affine=False,
        initial_transforms=["identity", "identity"],
        scales=(2, 1),
        reg_iterations=(6, 4),
        device=str(dev),
    )
    assert len(tbm["jacobians"]) == 2
    assert tbm["jacobians"][0].shape == fi.shape
    assert np.allclose(tbm["jacobians"][0].spacing, fi.spacing)
    assert np.allclose(tbm["jacobians"][0].origin, fi.origin)
    assert np.allclose(tbm["jacobians"][0].direction, fi.direction)
    assert tbm["mean_jacobian"].shape == fi.shape
    assert tbm["std_jacobian"].shape == fi.shape
    assert tbm["similarity"].shape == (2,)
    assert tbm["correlation"].shape == (2,)
    assert np.all(tbm["folding_pct"] == 0.0)
    assert np.all(tbm["min_jacobian"] > 0.0)

    # Verify batch_jacobian tensor mode matches image mode
    jac_t = syntx.batch_jacobian(ft, tbm["fwd_chains"], do_log=True, return_tensors=True, device=str(dev))
    assert tuple(jac_t.shape) == (2, 1, fi.shape[2], fi.shape[1], fi.shape[0])
    jac_0_xyz = jac_t[0, 0].permute(2, 1, 0).detach().cpu().numpy()
    assert np.allclose(jac_0_xyz, tbm["jacobians"][0].numpy(), atol=1e-6)
