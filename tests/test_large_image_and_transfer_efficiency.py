"""Regression tests for R50: large-image memory scaling and CPU<=>GPU transfer/sync elimination."""

from __future__ import annotations

import os
import tempfile

import ants
import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import syntx
from syntx.core.digital_diffeomorphism import (
    _eval_dual_fd_det_min_fast,
    _eval_dual_fd_det_min_scalar_slabbed,
    enforce_digital_diffeomorphism_nd,
)
from syntx.core.losses.cross_correlation import local_ncc_loss_nd
from syntx.core.losses.mutual_information import mattes_sample_indices
from syntx.core.optimizers import FieldStepOptimizer
from syntx.core.regularizers import (
    _NAVIER_OPS_CACHE,
    _get_navier_ops_cached,
    apply_navier_green_operator,
    clear_k_mesh_cache,
)
from syntx.liouville import _fd_det, _fd_det_3d_zyx, liouville_determinant
from syntx.robust_affine import _rodrigues_rotation_matrix_3d
from syntx.spatial import disp_tensor_to_itk, get_physical_grid_torch
from syntx.syn import SyNTo
from syntx.transforms import (
    ContinuousField,
    TransformChain,
    _DISP_FIELD_CACHE,
    _DISP_GRAD_CACHE,
    _DISP_IMG_CACHE,
    _apply_tensor_3d,
    _apply_vector_3d,
    _cache_disp_image_and_field,
    _get_disp_and_grad_fields,
    _polar_svd_3x3,
    _polar_svd_cpu_exact,
    _write_ants_images_fast,
)
from syntx.tvf import AdjointODEIntegrate, TVFModel


class DispatchTransferSyncCounter(TorchDispatchMode):
    """Counts host<->device copies, scalar syncs, and peak live bytes on the active accelerator."""

    def __init__(self, dev_type: str):
        super().__init__()
        self.dev_type = dev_type
        self.h2d_count = 0
        self.h2d_bytes = 0
        self.d2h_count = 0
        self.d2h_bytes = 0
        self.sync_count = 0
        self.peak_bytes = 0

    def _cur_mem(self) -> int:
        if self.dev_type == "mps" and torch.backends.mps.is_available():
            return int(torch.mps.current_allocated_memory())
        if self.dev_type == "cuda" and torch.cuda.is_available():
            return int(torch.cuda.memory_allocated())
        return 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        op_name = str(func)
        has_dev_in = False

        def _scan(x):
            nonlocal has_dev_in
            if isinstance(x, torch.Tensor):
                if x.device.type == self.dev_type:
                    has_dev_in = True
            elif isinstance(x, (list, tuple)):
                for y in x:
                    _scan(y)

        _scan(args)
        _scan(list(kwargs.values()))
        out = func(*args, **kwargs)
        cur = self._cur_mem()
        if cur > self.peak_bytes:
            self.peak_bytes = cur

        if "_local_scalar_dense" in op_name and has_dev_in:
            self.sync_count += 1
        elif "_to_copy" in op_name:
            src = args[0] if args and isinstance(args[0], torch.Tensor) else None
            dst = out if isinstance(out, torch.Tensor) else None
            if src is not None and dst is not None and src.device.type != dst.device.type:
                nb = dst.numel() * dst.element_size()
                if src.device.type == "cpu" and dst.device.type == self.dev_type:
                    self.h2d_count += 1
                    self.h2d_bytes += nb
                elif src.device.type == self.dev_type and dst.device.type == "cpu":
                    self.d2h_count += 1
                    self.d2h_bytes += nb
                    if nb <= 64:
                        self.sync_count += 1
        return out


def _accel_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _oblique_direction() -> np.ndarray:
    m = np.array([[0.97, -0.18, 0.14], [0.19, 0.98, -0.05], [-0.13, 0.08, 0.99]], dtype=np.float64)
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        r[:, 0] *= -1.0
    return r


def _synthetic_3d_pair(shape=(24, 28, 24)):
    x, y, z = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[1], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[2], dtype=np.float32),
        indexing="ij",
    )
    rf = x**2 + (1.1 * y) ** 2 + (0.9 * z) ** 2
    rm = (x - 0.08) ** 2 + (1.05 * (y + 0.05)) ** 2 + (0.95 * (z - 0.04)) ** 2
    af = np.clip(np.exp(-2.5 * rf) * (0.65 + 0.35 * np.cos(5 * x) * np.cos(5 * y)), 0.0, 1.0).astype(np.float32)
    am = np.clip(np.exp(-2.5 * rm) * (0.65 + 0.35 * np.cos(5 * (x - 0.07)) * np.cos(5 * (y + 0.04))), 0.0, 1.0).astype(
        np.float32
    )
    af[rf > 0.85] = 0.0
    am[rm > 0.85] = 0.0
    d_mat = _oblique_direction()
    fi = ants.from_numpy(af, origin=(-12.0, -14.0, -12.0), spacing=(1.1, 0.9, 1.2), direction=d_mat)
    mi = ants.from_numpy(am, origin=(-12.0, -14.0, -12.0), spacing=(1.1, 0.9, 1.2), direction=d_mat)
    return fi, mi


@pytest.mark.parametrize(
    ("opt_type", "momentum"),
    [("cfl", 0.0), ("cfl", 0.8), ("adam", 0.0), ("reg_adam", 0.0)],
)
def test_field_step_optimizer_inplace_bitwise_parity(opt_type: str, momentum: float):
    torch.manual_seed(17)
    dev = _accel_device()
    grid_shape = (16, 18, 20)
    shape = (1, *grid_shape, 3)
    spacing = (1.2, 0.9, 1.5)

    opt = FieldStepOptimizer(
        mode=opt_type,
        param_domain="physical_mm",
        cfl_momentum=momentum,
    )

    m_ref = None
    for step_idx in range(3):
        g = torch.randn(shape, device=dev, dtype=torch.float32)
        u_reg = (0.5 * g).clone()
        u_reg_cpu = u_reg.cpu()

        # Reference calculation on CPU float32 matching legacy out-of-place physical CFL
        sp_rev = torch.tensor(list(reversed(spacing)), dtype=torch.float32).view(1, 1, 1, 1, 3)
        if opt_type == "cfl":
            v_vox_ref = u_reg_cpu / sp_rev
            if momentum > 0.0:
                if step_idx == 0:
                    m_ref = (1.0 - momentum) * v_vox_ref
                else:
                    m_ref = momentum * m_ref + (1.0 - momentum) * v_vox_ref
                v_vox_ref = m_ref
            mx = v_vox_ref.norm(dim=-1).amax().clamp(min=1e-8)
            expected = -(0.25 * (v_vox_ref / mx) * sp_rev)

        delta = opt.step(
            g,
            0.25,
            grid_shape=grid_shape,
            spacing=spacing,
            reg_fn=(lambda x, _u=u_reg: _u),
            reg_smooth_fn=(lambda x: 0.8 * x),
        )
        assert delta.shape == shape
        assert torch.isfinite(delta).all()
        if opt_type == "cfl":
            assert torch.allclose(delta.cpu(), expected, atol=1e-6, rtol=1e-5)


def test_navier_cached_1d_broadcast_and_direction_cache():
    clear_k_mesh_cache()
    dev = _accel_device()
    shape = (20, 22, 24)
    spacing = (1.1, 0.9, 1.3)
    direction = _oblique_direction()
    g = torch.randn(1, *shape, 3, device=dev, dtype=torch.float32)

    out1 = apply_navier_green_operator(
        g, fluid_sigma=2.0, poisson_ratio=0.2, s=2.0, spacing=spacing, direction=direction
    )
    assert len(_NAVIER_OPS_CACHE) >= 1
    k_views, k_base, k_proj = next(iter(_NAVIER_OPS_CACHE.values()))
    n_rfft = shape[0] * shape[1] * (shape[2] // 2 + 1)
    cached_elems = sum(v.numel() for v in k_views) + k_base.numel() + k_proj.numel()
    assert cached_elems < int(2.1 * n_rfft)

    if dev.type != "cpu":
        prof = DispatchTransferSyncCounter(dev.type)
        with prof:
            out2 = apply_navier_green_operator(
                g, fluid_sigma=2.0, poisson_ratio=0.2, s=2.0, spacing=spacing, direction=direction
            )
        assert prof.h2d_count == 0
        assert prof.sync_count == 0
        assert torch.equal(out1, out2)


def test_mattes_sample_indices_zero_h2d_on_device():
    dev = _accel_device()
    idx1 = mattes_sample_indices(50_000, 0.25, device=dev)
    assert idx1.device.type == dev.type
    if dev.type != "cpu":
        prof = DispatchTransferSyncCounter(dev.type)
        with prof:
            idx2 = mattes_sample_indices(50_000, 0.25, device=dev)
        assert prof.h2d_count == 0
        assert prof.sync_count == 0
        assert torch.equal(idx1, idx2)


def test_rodrigues_sync_free_and_zero_angle_autograd():
    dev = _accel_device()
    # 1. Exact zero vector: autograd gradient must be finite and exact skew-symmetric generator
    w0 = torch.zeros(3, device=dev, dtype=torch.float32, requires_grad=True)
    if dev.type != "cpu":
        prof = DispatchTransferSyncCounter(dev.type)
        with prof:
            r0 = _rodrigues_rotation_matrix_3d(w0)
        assert prof.sync_count == 0
        assert prof.h2d_count == 0
    else:
        r0 = _rodrigues_rotation_matrix_3d(w0)
    assert torch.allclose(r0, torch.eye(3, device=dev), atol=1e-7)
    loss = (r0 * torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=dev)).sum()
    loss.backward()
    assert w0.grad is not None and torch.isfinite(w0.grad).all()
    assert torch.allclose(w0.grad.cpu(), torch.tensor([0.0, 0.0, 2.0]), atol=1e-6)

    # 2. Non-zero rotation matches CPU torch.linalg.matrix_exp(skew(w))
    w1 = torch.tensor([0.35, -0.22, 0.48], device=dev, dtype=torch.float32, requires_grad=True)
    r1 = _rodrigues_rotation_matrix_3d(w1)
    skew_cpu = torch.tensor(
        [[0.0, -0.48, -0.22], [0.48, 0.0, -0.35], [0.22, 0.35, 0.0]],
        dtype=torch.float32,
    )
    r_exp = torch.linalg.matrix_exp(skew_cpu)
    assert torch.allclose(r1.detach().cpu(), r_exp, atol=1e-6)


def test_robust_affine_return_warped_false_and_low_sync_budget():
    fi, mi = _synthetic_3d_pair((24, 28, 24))
    dev = _accel_device()
    r_true = syntx.robust_affine(
        fi,
        mi,
        dof="affine",
        mode="pytorch",
        device=str(dev),
        multi_start=False,
        levels=[4, 2],
        iterations=[3, 3],
        return_warped=True,
    )
    if dev.type != "cpu":
        prof = DispatchTransferSyncCounter(dev.type)
        with prof:
            r_false = syntx.robust_affine(
                fi,
                mi,
                dof="affine",
                mode="pytorch",
                device=str(dev),
                multi_start=False,
                levels=[4, 2],
                iterations=[3, 3],
                return_warped=False,
            )
        # Inner Adam steps + Rodrigues + objective must be sync-free (< 40 stage-level syncs total)
        assert prof.sync_count < 40
    else:
        r_false = syntx.robust_affine(
            fi,
            mi,
            dof="affine",
            mode="pytorch",
            device=str(dev),
            multi_start=False,
            levels=[4, 2],
            iterations=[3, 3],
            return_warped=False,
        )
    assert r_true["warpedmovout"] is not None
    assert r_false["warpedmovout"] is None
    tx_t = ants.read_transform(r_true["fwdtransforms"][0])
    tx_f = ants.read_transform(r_false["fwdtransforms"][0])
    np.testing.assert_allclose(tx_t.parameters, tx_f.parameters, atol=1e-6)


def test_tvf_coarse_init_vmax_cache_and_adjoint_parity():
    dev = _accel_device()
    model = TVFModel(
        dim=3,
        image_shape=(24, 24, 24),
        velocity_shape=(24, 24, 24),
        init_velocity_shape=(6, 6, 6),
        n_time_steps=3,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
    ).to(dev)
    assert tuple(model.velocity.shape) == (3, 1, 6, 6, 6, 3)

    # First call populates _v_max_cache; second call at same _version incurs 0 device syncs
    vmax0 = model._get_v_max_voxel(model.velocity, (24, 24, 24))
    assert vmax0 == 0.0
    assert (id(model.velocity), model.velocity._version, (24, 24, 24)) in model._v_max_cache
    if dev.type != "cpu":
        prof = DispatchTransferSyncCounter(dev.type)
        with prof:
            vmax0_cached = model._get_v_max_voxel(model.velocity, (24, 24, 24))
        assert vmax0_cached == 0.0
        assert prof.sync_count == 0

    # Non-zero velocity per-keyframe norm matches dense norm
    with torch.no_grad():
        model.velocity.normal_(0.0, 0.4)
    vmax = model._get_v_max_voxel(model.velocity, (12, 12, 12))
    sp_step = 1.0 * (23.0 / 11.0)
    expected = float((model.velocity.detach() / sp_step).norm(dim=-1).max().item())
    assert abs(vmax - expected) < 1e-6

    # AdjointODEIntegrate path in TVFModel.integrate matches standard Euler forward integration
    model.adjoint = False
    disp_euler = model.integrate(0.0, 1.0, image_shape=(12, 12, 12))
    model.adjoint = True
    disp_adj = model.integrate(0.0, 1.0, image_shape=(12, 12, 12))
    assert torch.allclose(disp_adj, disp_euler, atol=1e-5)


def test_lncc_chunked_and_greedy_metric_chunks_parity():
    torch.manual_seed(9)
    dev = _accel_device()
    x = torch.rand(1, 1, 18, 20, 22, device=dev, dtype=torch.float32, requires_grad=True)
    y = torch.rand(1, 1, 18, 20, 22, device=dev, dtype=torch.float32)

    loss_dense = local_ncc_loss_nd(x, y, window_size=5, metric_chunks=1)
    loss_dense.backward()
    grad_dense = x.grad.detach().clone()

    x2 = x.detach().clone().requires_grad_(True)
    loss_chunk = local_ncc_loss_nd(x2, y, window_size=5, metric_chunks=3)
    loss_chunk.backward()
    grad_chunk = x2.grad.detach().clone()

    assert torch.allclose(loss_dense, loss_chunk, atol=1e-6)
    assert torch.allclose(grad_dense, grad_chunk, atol=1e-5)

    fi, mi = _synthetic_3d_pair((20, 22, 20))
    r1 = syntx.greedy(
        fi, mi, initial_transform=False, scales=[2, 1], reg_iterations=[2, 2], metric_chunks=1, return_inverse=True, device=str(dev)
    )
    r2 = syntx.greedy(
        fi, mi, initial_transform=False, scales=[2, 1], reg_iterations=[2, 2], metric_chunks=2, device=str(dev)
    )
    assert r2["provenance"]["metric_chunks"] == 2
    assert "phi_1" in r1["inverse_identity_errors"]
    assert "error_map" in r1["inverse_identity_errors"]["phi_1"]
    np.testing.assert_allclose(r1["warpedmovout"].numpy(), r2["warpedmovout"].numpy(), atol=1e-4)


def test_synto_lazy_field_allocation():
    dev = _accel_device()
    m_lazy = SyNTo(dim=3, grid_shape=(32, 32, 32), allocate_fields=False).to(dev)
    assert m_lazy.warp_l2r is None
    assert m_lazy.warp_r2l is None
    assert m_lazy.warp_l2r_inv is None
    assert m_lazy.warp_r2l_inv is None


def test_digital_diffeomorphism_z_slabbed_exact_boundary_parity():
    torch.manual_seed(23)
    dev = _accel_device()
    shape = (18, 16, 14)
    spacing = (1.2, 0.9, 1.4)
    origin = (-5.0, 3.0, -2.0)
    direction = _oblique_direction()
    u = 0.25 * torch.randn(1, *shape, 3, device=dev, dtype=torch.float32)
    v = -0.22 * u + 0.05 * torch.randn_like(u)

    steps_rev = list(reversed(spacing))
    dir_rev = direction[::-1, ::-1].copy()
    d_rev_t = torch.tensor(dir_rev.T.copy(), device=dev, dtype=torch.float32)
    s_d = float(np.linalg.det(np.linalg.inv(dir_rev)))

    min_dense = float(_eval_dual_fd_det_min_fast(u, steps_rev, d_rev_t, s_d, True, True).amin().item())
    min_slab = float(
        _eval_dual_fd_det_min_scalar_slabbed(u, steps_rev, d_rev_t, s_d, True, True, max_slab_voxels=900).item()
    )
    assert abs(min_dense - min_slab) < 1e-6

    stats: dict = {}
    u_dd, v_dd = enforce_digital_diffeomorphism_nd(
        u,
        v,
        spacing=spacing,
        origin=origin,
        direction=direction,
        jac_floor=0.05,
        steps=5,
        stats=stats,
    )
    assert u_dd.shape == u.shape and v_dd.shape == v.shape
    assert "initial_min_det_fwd" in stats


def test_cache_disp_field_from_tensor_zero_h2d_and_streaming_gzip():
    dev = _accel_device()
    direction = _oblique_direction()
    ref = ants.from_numpy(
        np.zeros((24, 20, 18), dtype=np.float32),
        origin=(2.0, -4.0, 1.0),
        spacing=(1.1, 1.2, 0.9),
        direction=direction,
    )
    disp_t = 0.5 * torch.randn(1, 18, 20, 24, 3, device=dev, dtype=torch.float32)
    disp_img = disp_tensor_to_itk(disp_t, ref)

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "warp.nii.gz")
        _write_ants_images_fast([(disp_img, p)], compresslevel=1)
        _DISP_FIELD_CACHE.clear()
        _DISP_IMG_CACHE.clear()
        if dev.type != "cpu":
            prof = DispatchTransferSyncCounter(dev.type)
            with prof:
                _cache_disp_image_and_field(p, disp_img, dev, disp_tensor=disp_t)
            # Only the tiny 3x3 + 3 M_norm/b_norm metadata (< 256 B) may upload; 0 field upload!
            assert prof.h2d_bytes < 512
            assert prof.d2h_count == 0
        else:
            _cache_disp_image_and_field(p, disp_img, dev, disp_tensor=disp_t)

        cached_field = next(iter(_DISP_FIELD_CACHE.values()))
        ref_field = ContinuousField(disp_img, device=dev, dtype=torch.float32)
        assert torch.equal(cached_field.tensor, ref_field.tensor)

        # Test multi-chunk (> 9 MiB uncompressed) bounded streaming gzip round-trip
        big_arr = np.random.RandomState(5).randn(96, 96, 96, 3).astype(np.float32)
        big_img = ants.from_numpy(big_arr, origin=ref.origin, spacing=ref.spacing, direction=direction, has_components=True)
        p_big = os.path.join(td, "big_warp.nii.gz")
        _write_ants_images_fast([(big_img, p_big)], compresslevel=1)
        reloaded = ants.image_read(p_big)
        np.testing.assert_array_equal(reloaded.numpy(), big_arr)


def test_gpu_polar_svd_and_slabbed_vector_and_tensor_transforms():
    torch.manual_seed(31)
    dev = _accel_device()
    # Include well-conditioned, anisotropic, and reflected (det < 0) matrices
    m = torch.eye(3, device=dev).unsqueeze(0) + 0.3 * torch.randn(2048, 3, 3, device=dev)
    m[:32] *= -1.0
    r_gpu = _polar_svd_3x3(m)
    r_cpu = torch.as_tensor(_polar_svd_cpu_exact(m.detach().cpu().numpy().astype(np.float64)).astype(np.float32), device=dev)
    assert torch.allclose(r_gpu, r_cpu, atol=2e-6)

    # Verify slabbed vs unslabbed _apply_vector_3d and _apply_tensor_3d + ANTs parity
    direction = _oblique_direction()
    ref = ants.from_numpy(
        np.zeros((16, 18, 14), dtype=np.float32),
        origin=(-8.0, -9.0, -7.0),
        spacing=(1.1, 1.0, 1.3),
        direction=direction,
    )
    rng = np.random.RandomState(11)
    u_arr = (0.3 * rng.randn(16, 18, 14, 3)).astype(np.float32)
    warp_img = ants.from_numpy(u_arr, origin=ref.origin, spacing=ref.spacing, direction=direction, has_components=True)
    vec_img = ants.from_numpy(
        rng.randn(16, 18, 14, 3).astype(np.float32),
        origin=ref.origin,
        spacing=ref.spacing,
        direction=direction,
        has_components=True,
    )
    # Build SPD 6-component tensor image [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]
    a = rng.randn(16, 18, 14, 3, 3)
    spd = np.einsum("...ij,...kj->...ik", a, a) + 0.5 * np.eye(3)
    sym6 = np.stack([spd[..., 0, 0], spd[..., 0, 1], spd[..., 0, 2], spd[..., 1, 1], spd[..., 1, 2], spd[..., 2, 2]], axis=-1).astype(
        np.float32
    )
    dti_img = ants.from_numpy(sym6, origin=ref.origin, spacing=ref.spacing, direction=direction, has_components=True)

    with tempfile.TemporaryDirectory() as td:
        wp = os.path.join(td, "w.nii.gz")
        ants.image_write(warp_img, wp)
        _DISP_FIELD_CACHE.clear()
        _DISP_GRAD_CACHE.clear()
        chain = TransformChain.from_paths([wp])

        v_full = _apply_vector_3d(image=vec_img, reference=ref, chain=chain, mode="bilinear", defaultvalue=0.0, dev=dev)
        v_slab = _apply_vector_3d(
            image=vec_img, reference=ref, chain=chain, mode="bilinear", defaultvalue=0.0, dev=dev, max_slab_voxels=900
        )
        v_ants = ants.apply_transforms(fixed=ref, moving=vec_img, transformlist=[wp], imagetype=1)
        np.testing.assert_allclose(v_full.numpy(), v_slab.numpy(), atol=1e-6)
        np.testing.assert_allclose(v_full.numpy(), v_ants.numpy(), atol=2e-4)

        t_full = _apply_tensor_3d(image=dti_img, reference=ref, chain=chain, mode="bilinear", dev=dev)
        t_slab = _apply_tensor_3d(
            image=dti_img, reference=ref, chain=chain, mode="bilinear", dev=dev, max_slab_voxels=900
        )
        np.testing.assert_allclose(t_full.numpy(), t_slab.numpy(), atol=1e-5)


def test_liouville_syn_gpu_fd3_parity():
    fi, mi = _synthetic_3d_pair((20, 22, 18))
    dev = _accel_device()
    reg = syntx.syn(fi, mi, affine_dof="none", levels=[2, 1], reg_iterations=[2, 2], device=str(dev))

    # Direct _fd_det_3d_zyx matches legacy CPU _fd_det
    det_gpu = _fd_det_3d_zyx(reg["model"].midpoint_warp_l2r, fi, dev)
    det_cpu = _fd_det(fi, reg["model"].midpoint_warp_l2r.detach().cpu()).numpy()
    np.testing.assert_allclose(np.transpose(det_gpu.cpu().numpy(), (2, 1, 0)), det_cpu, atol=1e-5)

    det_img, details = liouville_determinant(reg, fi, return_details=True)
    det_img_fast = liouville_determinant(reg, fi, return_details=False)
    np.testing.assert_allclose(det_img.numpy(), det_img_fast.numpy(), atol=1e-6)
    assert "det_l2r" in details and "det_r2l" in details


def test_syn_navier_with_total_sigma_positive():
    """Regression guard: total_sigma > 0 under default regularizer='navier' (and other continuum
    regularizers) must map elastic smoothing kernel_type to 'bessel' rather than raising ValueError."""
    fi, mi = _synthetic_3d_pair((18, 20, 18))
    dev = _accel_device()
    for reg_name in ("navier", "div_curl", "dsti1"):
        res = syntx.syn(
            fi,
            mi,
            affine_dof="none",
            regularizer=reg_name,
            total_sigma=0.5,
            levels=[2, 1],
            reg_iterations=[2, 2],
            device=str(dev),
        )
        assert res["warpedmovout"] is not None
        assert np.isfinite(res["warpedmovout"].numpy()).all()



def test_syn_midpoint_preserves_disp_field_cache_and_reuses_moving_field():
    """WP1: 4-slot GPU _DISP_FIELD_CACHE retains primary fwd/inv warps even when midpoint warps are cached."""
    fi, mi = _synthetic_3d_pair((20, 22, 18))
    dev = _accel_device()
    _DISP_FIELD_CACHE.clear()
    reg = syntx.syn(
        fi,
        mi,
        affine_dof="none",
        export_midpoint=True,
        levels=[2, 1],
        reg_iterations=[2, 2],
        device=str(dev),
    )
    fwd_p = reg["fwdtransforms"][0]
    inv_p = reg["invtransforms"][-1]
    cached_paths = {k[0] for k in _DISP_FIELD_CACHE.keys()}
    assert fwd_p in cached_paths
    assert inv_p in cached_paths

    # Downstream apply_transforms with _moving_field incurs 0 3D-volume H2D uploads
    cf_mi = ContinuousField(mi, device=dev, dtype=torch.float32)
    if dev.type != "cpu":
        prof = DispatchTransferSyncCounter(dev.type)
        with prof:
            w_fast = syntx.apply_transforms(fixed=fi, moving=mi, transformlist=reg["fwdtransforms"], device=str(dev), _moving_field=cf_mi)
        assert prof.h2d_bytes < 1024
    else:
        w_fast = syntx.apply_transforms(fixed=fi, moving=mi, transformlist=reg["fwdtransforms"], device=str(dev), _moving_field=cf_mi)
    np.testing.assert_allclose(w_fast.numpy(), reg["warpedmovout"].numpy(), atol=1e-6)


def test_inverse_fixed_step_zero_syncs_and_free_boundary():
    """WP2: Negative thresholds (fixed-step mode) execute 0 host syncs; stationary_boundary=False preserves rim."""
    from syntx.core.inverse import update_inverse_field_nd

    dev = _accel_device()
    direction = _oblique_direction()
    # Constant translation field (non-zero on boundary faces)
    u = torch.zeros(1, 14, 16, 14, 3, device=dev, dtype=torch.float32)
    u[..., 0] = 0.8
    u[..., 1] = -0.5

    for method in ("anderson", "fixed_point"):
        if dev.type != "cpu":
            # Warm cache once so we isolate per-step loop syncs and H2D
            _ = update_inverse_field_nd(
                u, None, steps=2, method=method, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), direction=direction,
                max_error_threshold=-1.0, mean_error_threshold=-1.0, stationary_boundary=False,
            )
            prof = DispatchTransferSyncCounter(dev.type)
            with prof:
                inv_free = update_inverse_field_nd(
                    u, None, steps=6, method=method, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), direction=direction,
                    max_error_threshold=-1.0, mean_error_threshold=-1.0, stationary_boundary=False,
                )
            assert prof.sync_count == 0
            assert prof.h2d_count == 0
        else:
            inv_free = update_inverse_field_nd(
                u, None, steps=6, method=method, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), direction=direction,
                max_error_threshold=-1.0, mean_error_threshold=-1.0, stationary_boundary=False,
            )
        # Boundary rim must hold ~(-0.8, +0.5, 0) rather than being zeroed
        assert float(inv_free[0, 0, 8, 8, 0].item()) < -0.75
        assert float(inv_free[0, 0, 8, 8, 1].item()) > 0.45


def test_greedy_return_inverse_preserves_forward_warp_bitwise():
    """WP2: return_inverse=True in syntx.greedy must leave forward warp and warpedmovout bitwise identical."""
    fi, mi = _synthetic_3d_pair((22, 24, 20))
    dev = _accel_device()
    r_fwd = syntx.greedy(
        fi, mi, initial_transform=False, scales=[2, 1], reg_iterations=[4, 3], return_inverse=False, device=str(dev)
    )
    r_inv = syntx.greedy(
        fi, mi, initial_transform=False, scales=[2, 1], reg_iterations=[4, 3], return_inverse=True, device=str(dev)
    )
    w_fwd_only = ants.image_read(r_fwd["fwdtransforms"][0]).numpy()
    w_with_inv = ants.image_read(r_inv["fwdtransforms"][0]).numpy()
    np.testing.assert_array_equal(w_fwd_only, w_with_inv)
    np.testing.assert_array_equal(r_fwd["warpedmovout"].numpy(), r_inv["warpedmovout"].numpy())
    assert r_inv["inverse_identity_errors"]["phi_1"]["mean_error"] < 0.05


def test_diagnostic_classifier_cache_eliminates_repeat_model_uploads():
    """WP3: Cached DiagnosticClassifier3D avoids reloading 55 MiB ResNet weights on repeat calls."""
    from syntx.classifier import get_cached_diagnostic_classifier, predict_diagnosis_deep

    weights_path = os.path.join(os.path.dirname(syntx.__file__), "models", "diagnostic_resnet10_3d.pth")
    if not os.path.exists(weights_path):
        pytest.skip("diagnostic_resnet10_3d.pth not present")
    fi, mi = _synthetic_3d_pair((44, 44, 44))
    dev = _accel_device()
    m1 = get_cached_diagnostic_classifier(weights_path, device=str(dev))
    m2 = get_cached_diagnostic_classifier(weights_path, device=str(dev))
    assert m1 is m2
    if dev.type != "cpu":
        prof = DispatchTransferSyncCounter(dev.type)
        with prof:
            out = predict_diagnosis_deep(m2, mi, device=str(dev))
        # Only the 64^3 float32 input tensor (1.0 MiB) uploads, never 55 MiB of model weights
        assert prof.h2d_bytes <= 2 * 1024 * 1024
        assert prof.sync_count <= 2
    else:
        out = predict_diagnosis_deep(m2, mi, device=str(dev))
    assert "modality" in out and "body_part" in out


def test_point_sampler_at_pool_cpu_gather_bitwise_parity():
    """WP4: PointSampler._at_pool CPU nearest-voxel gather matches F.grid_sample(nearest) bitwise with <1/10th H2D."""
    from syntx.sampling import PointSampler

    rng = np.random.RandomState(17)
    arr = np.clip(rng.uniform(0.0, 5.0, size=(36, 40, 32)).astype(np.float32), 0.0, 5.0)
    arr[:4] = 0.0
    img = ants.from_numpy(arr, spacing=(1.1, 1.1, 1.3))
    dev = _accel_device()
    ps = PointSampler(img, seed=42, device=dev)
    _ = ps.pool()
    _ = ps._at_pool(arr)  # prime _pool_flat_idx_cpu once

    field2 = rng.uniform(-1.0, 3.0, size=arr.shape).astype(np.float32)
    if dev.type != "cpu":
        prof = DispatchTransferSyncCounter(dev.type)
        with prof:
            w_fast = ps._at_pool(field2)
        # Full 3D volume is 36*40*32*4 = 184,320 B; pool upload must be strictly smaller
        assert prof.h2d_bytes <= ps.n_pool * 4 + 256
    else:
        w_fast = ps._at_pool(field2)

    t_ref = torch.from_numpy(np.ascontiguousarray(np.transpose(field2, (2, 1, 0)))).to(dev)
    sh_xyz = torch.tensor(list(img.shape), dtype=torch.float32, device=dev)
    norm = 2.0 * ps.vox_xyz / (sh_xyz - 1.0).clamp_min(1.0) - 1.0
    w_ref = torch.nn.functional.grid_sample(t_ref[None, None], norm.view(1, 1, 1, -1, 3), mode="nearest", align_corners=True).reshape(-1).clamp_min(0.0)
    assert torch.equal(w_fast, w_ref)


def test_motion_correction_batched_seed_and_restore_sync_budget():
    """WP4: Batched seed search + sync-free _restore keeps motion_correction accurate with low sync count."""
    fi, _ = _synthetic_3d_pair((36, 40, 36))
    arr0 = fi.numpy()
    vols = [arr0, np.roll(arr0, shift=1, axis=0), np.roll(arr0, shift=-1, axis=1)]
    arr4d = np.stack(vols, axis=-1).astype(np.float32)
    dir4 = np.eye(4, dtype=np.float64)
    dir4[:3, :3] = fi.direction
    img4d = ants.from_numpy(arr4d, origin=(*fi.origin, 0.0), spacing=(*fi.spacing, 2.0), direction=dir4)
    dev = _accel_device()
    if dev.type != "cpu":
        prof = DispatchTransferSyncCounter(dev.type)
        with prof:
            mc = syntx.motion_correction(img4d, reference=0, type_of_transform="Rigid", device=str(dev))
        assert prof.sync_count < 80
    else:
        mc = syntx.motion_correction(img4d, reference=0, type_of_transform="Rigid", device=str(dev))
    assert mc.motion_corrected.shape == img4d.shape
