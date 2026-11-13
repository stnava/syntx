"""
Inverting displacement fields, measuring inverse consistency, and integrating velocities.

Conventions shared by every function here:

- Fields are displacements in tensor order, shape (B, *spatial, dim) with spatial (z, y, x)
  and components (x, y, z). Channels-first input (B, dim, *spatial) is detected for the
  inverse solvers and returned in the same layout.
- ``spacing`` / ``origin`` / ``direction`` are ANTs (x-first) metadata. With all three
  given, displacements are in physical units (mm); without them, in normalised [-1, 1]
  grid units.
- Convergence thresholds and reported solver errors are in voxels.

Inversion: ``update_inverse_field_nd`` dispatches to the ITK-style fixed-point iteration,
``update_inverse_field_nd_anderson`` (default) or ``update_inverse_field_nd_hybrid_lm``.
Inverse error: ``calculate_inverse_identity_error`` (summary dict; exported as
``syntx.calculate_inverse_identity_error``) and ``compute_inverse_identity_error_nd``
(error map).
"""

import numpy as np
import torch
import torch.nn.functional as F

from .smoothing import separable_gaussian_filter, get_boundary_mask
from .grid import (
    get_physical_grid_torch,
    get_physical_to_normalized_affine,
    physical_to_normalized_fast,
    physical_to_normalized_torch,
    physical_to_normalized_torch_cached,
    grid_sample_nd,
    compose_grids,
    sample_field_cf,
)


def _batch_reduce_amax(x: torch.Tensor) -> torch.Tensor:
    """Per-batch-item max over all spatial and channel axes.

    Uses a 2-stage 1-axis reduction on 5-D+ tensors so PyTorch MPS never dispatches a
    4-axis reduction kernel on large 3-D volumes (which corrupts outputs above ~16M elements).
    """
    b = x.shape[0]
    if x.dim() >= 5:
        return x.reshape(b, x.shape[1], -1).amax(dim=2).amax(dim=1)
    return x.reshape(b, -1).amax(dim=1)


def _batch_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    """Per-batch-item mean over all spatial and channel axes (2-stage safe on 5-D+ MPS)."""
    b = x.shape[0]
    if x.dim() >= 5:
        return x.reshape(b, x.shape[1], -1).mean(dim=2).mean(dim=1)
    return x.reshape(b, -1).mean(dim=1)


def _batch_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """Per-batch-item sum over all spatial and channel axes (2-stage safe on 5-D+ MPS)."""
    b = x.shape[0]
    if x.dim() >= 5:
        return x.reshape(b, x.shape[1], -1).sum(dim=2).sum(dim=1)
    return x.reshape(b, -1).sum(dim=1)


def _scaled_sq_sum(error: torch.Tensor, scale: torch.Tensor, divide: bool) -> torch.Tensor:
    """Compute ``sum_c ((error[..., c] / scale[c]) ** 2)`` (or ``* scale[c]``) as ``(B, *spatial, 1)``
    without allocating a full multi-channel ``(B, *spatial, dim)`` squared tensor."""
    dim = error.shape[-1]
    if divide:
        sq = (error[..., 0:1] / scale[0]).square_()
        for c in range(1, dim):
            sq.add_((error[..., c:c + 1] / scale[c]).square_())
    else:
        sq = (error[..., 0:1] * scale[0]).square_()
        for c in range(1, dim):
            sq.add_((error[..., c:c + 1] * scale[c]).square_())
    return sq


def _dot_1d(a: torch.Tensor, b: torch.Tensor, block_size: int = 4_194_304) -> torch.Tensor:
    """1-D inner product preserving exact ``torch.dot`` for ``n <= block_size`` while blocking
    very large vectors (>4M elements) to avoid MPS large-inner-dim kernel bugs and float32
    mantissa swamping."""
    n = a.numel()
    if n <= block_size:
        return torch.dot(a, b)
    acc = torch.zeros((), device=a.device, dtype=a.dtype)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        acc = acc + torch.dot(a[start:end], b[start:end])
    return acc


def _multilinear_val_and_jac_torch(
    u_grid: torch.Tensor,
    vox_coords: torch.Tensor,
    right_mat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact multilinear value and Jacobian of ``u_grid`` at continuous voxel coordinates.

    Matches ``F.grid_sample(..., mode='bilinear', padding_mode='border', align_corners=True)``
    for both 2-D ``(H, W, 2)`` and 3-D ``(Z, Y, X, 3)`` fields. Cell derivatives are zeroed
    along any axis clamped at the domain border and mapped into target coordinate space via
    right-multiplication by ``right_mat`` (``du / dy = (du / dvox) @ right_mat``).
    """
    dim = u_grid.shape[-1]
    dtype = u_grid.dtype
    if dim == 2:
        H, W, _ = u_grid.shape
        vy, vx = vox_coords[:, 0], vox_coords[:, 1]
        in_y = ((vy >= 0.0) & (vy <= float(H - 1))).to(dtype)
        in_x = ((vx >= 0.0) & (vx <= float(W - 1))).to(dtype)
        vy_c = vy.clamp(0.0, float(max(H - 1, 0)))
        vx_c = vx.clamp(0.0, float(max(W - 1, 0)))
        iy0 = torch.floor(vy_c).long().clamp(0, max(H - 2, 0))
        ix0 = torch.floor(vx_c).long().clamp(0, max(W - 2, 0))
        iy1 = (iy0 + 1).clamp(max=max(H - 1, 0))
        ix1 = (ix0 + 1).clamp(max=max(W - 1, 0))
        ty = (vy_c - iy0.to(dtype)).unsqueeze(-1)
        tx = (vx_c - ix0.to(dtype)).unsqueeze(-1)
        omy, omx = 1.0 - ty, 1.0 - tx

        flat = u_grid.reshape(-1, 2)
        c00 = flat[iy0 * W + ix0]
        c01 = flat[iy0 * W + ix1]
        c10 = flat[iy1 * W + ix0]
        c11 = flat[iy1 * W + ix1]

        c0 = c00 * omx + c01 * tx
        c1 = c10 * omx + c11 * tx
        u_val = c0 * omy + c1 * ty

        du_dvy = (c1 - c0) * in_y.unsqueeze(-1)
        du_dvx = ((c01 - c00) * omy + (c11 - c10) * ty) * in_x.unsqueeze(-1)
        du_dvox = torch.stack([du_dvy, du_dvx], dim=-1)
        return u_val, du_dvox @ right_mat

    Z, Y, X, _ = u_grid.shape
    vz, vy, vx = vox_coords[:, 0], vox_coords[:, 1], vox_coords[:, 2]
    in_z = ((vz >= 0.0) & (vz <= float(Z - 1))).to(dtype)
    in_y = ((vy >= 0.0) & (vy <= float(Y - 1))).to(dtype)
    in_x = ((vx >= 0.0) & (vx <= float(X - 1))).to(dtype)
    vz_c = vz.clamp(0.0, float(max(Z - 1, 0)))
    vy_c = vy.clamp(0.0, float(max(Y - 1, 0)))
    vx_c = vx.clamp(0.0, float(max(X - 1, 0)))
    iz0 = torch.floor(vz_c).long().clamp(0, max(Z - 2, 0))
    iy0 = torch.floor(vy_c).long().clamp(0, max(Y - 2, 0))
    ix0 = torch.floor(vx_c).long().clamp(0, max(X - 2, 0))
    iz1 = (iz0 + 1).clamp(max=max(Z - 1, 0))
    iy1 = (iy0 + 1).clamp(max=max(Y - 1, 0))
    ix1 = (ix0 + 1).clamp(max=max(X - 1, 0))
    tz = (vz_c - iz0.to(dtype)).unsqueeze(-1)
    ty = (vy_c - iy0.to(dtype)).unsqueeze(-1)
    tx = (vx_c - ix0.to(dtype)).unsqueeze(-1)
    omz, omy, omx = 1.0 - tz, 1.0 - ty, 1.0 - tx

    flat = u_grid.reshape(-1, 3)
    sz, sy = Y * X, X
    c000 = flat[iz0 * sz + iy0 * sy + ix0]
    c001 = flat[iz0 * sz + iy0 * sy + ix1]
    c010 = flat[iz0 * sz + iy1 * sy + ix0]
    c011 = flat[iz0 * sz + iy1 * sy + ix1]
    c100 = flat[iz1 * sz + iy0 * sy + ix0]
    c101 = flat[iz1 * sz + iy0 * sy + ix1]
    c110 = flat[iz1 * sz + iy1 * sy + ix0]
    c111 = flat[iz1 * sz + iy1 * sy + ix1]

    c00 = c000 * omx + c001 * tx
    c01 = c010 * omx + c011 * tx
    c10 = c100 * omx + c101 * tx
    c11 = c110 * omx + c111 * tx
    c0 = c00 * omy + c01 * ty
    c1 = c10 * omy + c11 * ty
    u_val = c0 * omz + c1 * tz

    du_dvz = (c1 - c0) * in_z.unsqueeze(-1)
    du_dvy = ((c01 - c00) * omz + (c11 - c10) * tz) * in_y.unsqueeze(-1)
    du_dvx = (
        ((c001 - c000) * omy + (c011 - c010) * ty) * omz
        + ((c101 - c100) * omy + (c111 - c110) * ty) * tz
    ) * in_x.unsqueeze(-1)
    du_dvox = torch.stack([du_dvz, du_dvy, du_dvx], dim=-1)
    return u_val, du_dvox @ right_mat


def _solve_spd_lm_torch(
    JTJ: torch.Tensor,
    rhs: torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    """Closed-form Jacobi-equilibrated Cholesky solve of ``(JTJ + mu * I) dv = rhs`` for 2-D/3-D."""
    dim = rhs.shape[-1]
    diag = JTJ.diagonal(dim1=-2, dim2=-1)
    diag_scale = diag.amax(dim=-1).clamp(min=1.0)
    mu = (lam * diag_scale).clamp(min=1e-6)
    h_diag = (diag + mu.unsqueeze(-1)).clamp(min=1e-12)
    s = h_diag.sqrt()

    if dim == 2:
        s0, s1 = s[:, 0], s[:, 1]
        h10 = JTJ[:, 1, 0] / (s1 * s0)
        b0 = rhs[:, 0] / s0
        b1 = rhs[:, 1] / s1
        l11 = (1.0 - h10 * h10).clamp(min=1e-7).sqrt()
        y0 = b0
        y1 = (b1 - h10 * y0) / l11
        z1 = y1 / l11
        z0 = y0 - h10 * z1
        return torch.stack([z0 / s0, z1 / s1], dim=-1)

    s0, s1, s2 = s[:, 0], s[:, 1], s[:, 2]
    h10 = JTJ[:, 1, 0] / (s1 * s0)
    h20 = JTJ[:, 2, 0] / (s2 * s0)
    h21 = JTJ[:, 2, 1] / (s2 * s1)
    b0 = rhs[:, 0] / s0
    b1 = rhs[:, 1] / s1
    b2 = rhs[:, 2] / s2
    l11 = (1.0 - h10 * h10).clamp(min=1e-7).sqrt()
    l21 = (h21 - h20 * h10) / l11
    l22 = (1.0 - h20 * h20 - l21 * l21).clamp(min=1e-7).sqrt()
    y0 = b0
    y1 = (b1 - h10 * y0) / l11
    y2 = (b2 - h20 * y0 - l21 * y1) / l22
    z2 = y2 / l22
    z1 = (y1 - l21 * z2) / l11
    z0 = y0 - h10 * z1 - h20 * z2
    return torch.stack([z0 / s0, z1 / s1, z2 / s2], dim=-1)


def update_inverse_field_nd_hybrid_lm(
    W_disp: torch.Tensor,
    W_inv_disp: torch.Tensor = None,
    steps: int = 30,
    relaxation: float = 1.0,
    smoothing_sigma: float = 0.0,
    max_error_threshold: float = 0.1,
    mean_error_threshold: float = 0.001,
    damping_factor: float = 1.0,
    spacing=None,
    origin=None,
    direction=None,
    X_phys=None,
    stationary_boundary: bool = True,
    stats: dict = None,
) -> torch.Tensor:
    """
    Inverse of a displacement field by safeguarded per-voxel Levenberg-Marquardt steps.

    Minimises ``0.5 * ||v(x) + u(x + v(x))||^2`` per voxel using the exact analytical
    multilinear Jacobian ``J(y) = I + grad_y u(y)`` evaluated at ``y = x + v(x)`` (respecting
    full ``spacing``, ``origin`` and ``direction`` cosines in physical mode, or normalised
    ``[-1, 1]`` coordinates when ``spacing`` is omitted). Each step solves the SPD normal
    equations ``(J^T J + mu I) dv = -J^T r``, caps ``||dv||`` to a trust-region radius, and
    accepts ``v + dv`` per voxel strictly when ``||r(v + dv)|| < ||r(v)||`` while adapting the
    damping parameter ``lambda``.

    When ``smoothing_sigma == 0``, only unconverged interior voxels are gathered into an
    active set so warm-started polishing runs in sparse ``O(N_active)`` time.
    """
    channels_first = False
    if W_disp.dim() >= 3 and W_disp.shape[1] in [2, 3] and W_disp.shape[-1] not in [2, 3]:
        channels_first = True
        perm = (0,) + tuple(range(2, W_disp.dim())) + (1,)
        W_disp = W_disp.permute(perm)
        if W_inv_disp is not None and W_inv_disp.dim() >= 3 and W_inv_disp.shape[1] in [2, 3]:
            W_inv_disp = W_inv_disp.permute(perm)

    B = W_disp.shape[0]
    dim = W_disp.shape[-1]
    spatial = tuple(W_disp.shape[1:-1])
    device = W_disp.device
    dtype = W_disp.dtype

    if spacing is not None and origin is None:
        origin = (0.0,) * dim
    if spacing is not None and direction is None:
        direction = np.eye(dim).flatten()
    if X_phys is not None and (spacing is None or origin is None or direction is None):
        raise ValueError("X_phys needs the spacing / origin / direction it was built from")

    use_physical = (
        X_phys is not None
        or (spacing is not None and origin is not None and direction is not None)
    )
    boundary_mask = get_boundary_mask(spatial, device, dtype)
    W_disp_cf = torch.movedim(W_disp, -1, 1)

    if use_physical:
        if X_phys is None:
            X_phys = get_physical_grid_torch(
                spatial, spacing, origin, direction, device=device, dtype=dtype
            )
        X_base = X_phys
        spacing_rev = tuple(float(s) for s in reversed(spacing))
        origin_rev = tuple(float(o) for o in reversed(origin))
        dir_arr = np.asarray(direction, dtype=np.float64)
        if dir_arr.ndim == 1:
            dir_arr = dir_arr.reshape(dim, dim)
        direction_rev = dir_arr[::-1, ::-1].copy()

        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        shape_t = torch.tensor(list(spatial), device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        M_norm, b_norm = get_physical_to_normalized_affine(
            shape_t, spacing_t, origin_t, direction_t
        )

        D_inv_np = np.linalg.inv(direction_rev)
        sp_rev_np = np.asarray(spacing_rev, dtype=np.float64)
        A_to_vox_np = D_inv_np.T / sp_rev_np[None, :]
        b_to_vox_np = -np.asarray(origin_rev, dtype=np.float64) @ A_to_vox_np
        right_mat_np = np.diag(1.0 / sp_rev_np) @ D_inv_np

        A_to_vox = torch.tensor(A_to_vox_np, device=device, dtype=dtype)
        b_to_vox = torch.tensor(b_to_vox_np, device=device, dtype=dtype)
        right_mat = torch.tensor(right_mat_np, device=device, dtype=dtype)
        err_to_vox_scale = 1.0 / spacing_t
        max_step_unit = float(min(spacing_rev)) * 2.0 * max(float(relaxation), 1e-3)
    else:
        grids = [torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in spatial]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        X_base = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0)

        k_vec = np.asarray([max(s - 1, 1) / 2.0 for s in spatial], dtype=np.float64)
        p_flip = np.eye(dim, dtype=np.float64)[::-1].copy()
        A_to_vox_np = p_flip @ np.diag(k_vec)
        b_to_vox_np = k_vec.copy()
        right_mat_np = A_to_vox_np.T

        A_to_vox = torch.tensor(A_to_vox_np, device=device, dtype=dtype)
        b_to_vox = torch.tensor(b_to_vox_np, device=device, dtype=dtype)
        right_mat = torch.tensor(right_mat_np, device=device, dtype=dtype)
        err_to_vox_scale = torch.tensor(
            [float(max(s - 1, 1) / 2.0) for s in reversed(spatial)],
            device=device,
            dtype=dtype,
        )
        max_step_unit = (2.0 / float(max(max(spatial) - 1, 1))) * 2.0 * max(float(relaxation), 1e-3)

    def _eval_full_error(v_field: torch.Tensor) -> torch.Tensor:
        if use_physical:
            c_norm = physical_to_normalized_fast(X_base + v_field, M_norm, b_norm)
        else:
            c_norm = X_base + v_field
        return sample_field_cf(W_disp_cf, c_norm).add_(v_field)

    if W_inv_disp is None:
        v = torch.zeros_like(W_disp)
        best_r_sq = ((W_disp * err_to_vox_scale) ** 2).sum(dim=-1, keepdim=True)
        for alpha in (0.25, 0.5, 1.0):
            v_cand = W_disp * (-alpha)
            if stationary_boundary:
                v_cand.mul_(boundary_mask)
            err_cand = _eval_full_error(v_cand)
            r_cand_sq = ((err_cand * err_to_vox_scale) ** 2).sum(dim=-1, keepdim=True)
            better = r_cand_sq < best_r_sq
            v = torch.where(better, v_cand, v)
            best_r_sq = torch.where(better, r_cand_sq, best_r_sq)
        del v_cand, err_cand, r_cand_sq, best_r_sq, better
    else:
        v = W_inv_disp.clone()
        if stationary_boundary:
            v.mul_(boundary_mask)

    if steps <= 0:
        if stats is not None:
            stats['steps'] = 0
            stats['converged'] = False
        return torch.movedim(v, -1, 1) if channels_first else v

    domain_mask = (
        (boundary_mask[0, ..., 0] > 0.5)
        if stationary_boundary
        else torch.ones(spatial, dtype=torch.bool, device=device)
    )
    n_domain = max(int(domain_mask.sum().item()), 1)
    lam_init = 1e-3 * max(float(damping_factor), 1e-6)
    step_scale = float(relaxation) if float(relaxation) < 1.0 else 1.0
    I_d = torch.eye(dim, device=device, dtype=dtype).unsqueeze(0)

    if smoothing_sigma > 0.0:
        active_idx_all = torch.nonzero(domain_mask, as_tuple=False)
        steps_taken = steps
        converged = False
        lam_dense = torch.full(
            (B, active_idx_all.shape[0]), lam_init, device=device, dtype=dtype
        )
        for iteration in range(steps):
            err_full = _eval_full_error(v)
            vox_norm = ((err_full * err_to_vox_scale) ** 2).sum(dim=-1).sqrt_()
            vox_norm_dom = vox_norm[:, domain_mask]
            max_err = float(vox_norm_dom.amax().item()) if vox_norm_dom.numel() > 0 else 0.0
            mean_err = float(vox_norm_dom.mean().item()) if vox_norm_dom.numel() > 0 else 0.0
            if max_err <= max_error_threshold and mean_err <= mean_error_threshold:
                converged = True
                steps_taken = iteration
                break
            for b in range(B):
                if active_idx_all.shape[0] == 0:
                    continue
                u_grid = W_disp[b]
                x_grid_b = X_base[0 if X_base.shape[0] == 1 else b]
                idx_tuple = tuple(active_idx_all[:, d] for d in range(dim))
                x_a = x_grid_b[idx_tuple]
                v_a = v[b][idx_tuple]
                y_vox = (x_a + v_a) @ A_to_vox + b_to_vox
                u_a, grad_u_a = _multilinear_val_and_jac_torch(u_grid, y_vox, right_mat)
                r_a = v_a + u_a
                r_vox_sq_a = ((r_a * err_to_vox_scale) ** 2).sum(dim=-1)
                J = I_d + grad_u_a
                JT = J.transpose(-1, -2)
                JTJ = JT @ J
                rhs = -(JT @ r_a.unsqueeze(-1)).squeeze(-1)
                dv = _solve_spd_lm_torch(JTJ, rhs, lam_dense[b]) * step_scale
                dv_norm = torch.linalg.vector_norm(dv, dim=-1, keepdim=True)
                dv = torch.where(
                    dv_norm > max_step_unit,
                    dv * (max_step_unit / dv_norm.clamp(min=1e-12)),
                    dv,
                )
                v_trial = v_a + dv
                y_vox_t = (x_a + v_trial) @ A_to_vox + b_to_vox
                u_t, _ = _multilinear_val_and_jac_torch(u_grid, y_vox_t, right_mat)
                r_t = v_trial + u_t
                r_vox_sq_t = ((r_t * err_to_vox_scale) ** 2).sum(dim=-1)
                imp = r_vox_sq_t < r_vox_sq_a
                v[b][idx_tuple] = torch.where(imp.unsqueeze(-1), v_trial, v_a)
                lam_dense[b] = torch.where(
                    imp,
                    (lam_dense[b] * 0.25).clamp(min=1e-6),
                    (lam_dense[b] * 8.0).clamp(max=1e8),
                )
            v = separable_gaussian_filter(v, smoothing_sigma)
            if stationary_boundary:
                v.mul_(boundary_mask)
        if stats is not None:
            stats['steps'] = steps_taken
            stats['converged'] = converged
        return torch.movedim(v, -1, 1) if channels_first else v

    # Sparse active-set safeguarded Levenberg-Marquardt (smoothing_sigma == 0)
    err_init = _eval_full_error(v)
    vox_err_sq = ((err_init * err_to_vox_scale) ** 2).sum(dim=-1)
    del err_init

    max_steps_used = 0
    all_converged = True
    max_chunk = 2_000_000

    for b in range(B):
        vox_sq_dom = vox_err_sq[b][domain_mask]
        if vox_sq_dom.numel() == 0:
            continue
        vox_err_dom = vox_sq_dom.sqrt()
        max_err_b = float(vox_err_dom.amax().item())
        if vox_err_dom.device.type == "mps":
            acc = 0.0
            for _s in range(0, vox_err_dom.numel(), 4_194_304):
                acc += float(vox_err_dom[_s : _s + 4_194_304].sum().item())
            mean_err_b = acc / n_domain
        else:
            mean_err_b = float(vox_err_dom.sum(dtype=torch.float64).item() / n_domain)
        del vox_err_dom
        if max_err_b <= max_error_threshold and mean_err_b <= mean_error_threshold:
            continue

        if mean_err_b <= mean_error_threshold:
            active_tol_vox = float(max_error_threshold)
        else:
            active_tol_vox = float(
                min(max_error_threshold, max(mean_error_threshold * 0.8, 1e-6))
            )
        target_vox_sq = max((0.25 * active_tol_vox) ** 2, 1e-12)

        active_idx_full = torch.nonzero(
            (vox_err_sq[b] > (active_tol_vox ** 2)) & domain_mask,
            as_tuple=False,
        )
        n_act_total = active_idx_full.shape[0]
        if n_act_total == 0:
            continue

        u_grid = W_disp[b]
        x_grid_b = X_base[0 if X_base.shape[0] == 1 else b]
        b_steps_used = 0
        b_any_unconverged = False

        for c_start in range(0, n_act_total, max_chunk):
            active_idx = active_idx_full[c_start : c_start + max_chunk]
            n_act = active_idx.shape[0]
            idx_tuple = tuple(active_idx[:, d] for d in range(dim))
            x_act = x_grid_b[idx_tuple]
            v_act = v[b][idx_tuple]

            y_vox = (x_act + v_act) @ A_to_vox + b_to_vox
            u_val, _ = _multilinear_val_and_jac_torch(u_grid, y_vox, right_mat)
            r = v_act + u_val
            r_vox_sq = ((r * err_to_vox_scale) ** 2).sum(dim=-1)
            lam = torch.full((n_act,), lam_init, device=device, dtype=dtype)
            alive = torch.arange(n_act, device=device)

            chunk_steps = 0
            for iteration in range(steps):
                alive = alive[(r_vox_sq[alive] > target_vox_sq) & (lam[alive] < 1e7)]
                if alive.numel() == 0:
                    break
                chunk_steps = iteration + 1
                x_a = x_act[alive]
                v_a = v_act[alive]
                r_a = r[alive]
                r_vox_sq_a = r_vox_sq[alive]
                lam_a = lam[alive]

                y_vox_a = (x_a + v_a) @ A_to_vox + b_to_vox
                _, grad_u_a = _multilinear_val_and_jac_torch(u_grid, y_vox_a, right_mat)
                J = I_d + grad_u_a
                JT = J.transpose(-1, -2)
                JTJ = JT @ J
                rhs = -(JT @ r_a.unsqueeze(-1)).squeeze(-1)
                dv = _solve_spd_lm_torch(JTJ, rhs, lam_a) * step_scale
                dv_norm = torch.linalg.vector_norm(dv, dim=-1, keepdim=True)
                dv = torch.where(
                    dv_norm > max_step_unit,
                    dv * (max_step_unit / dv_norm.clamp(min=1e-12)),
                    dv,
                )

                v_trial = v_a + dv
                y_vox_t = (x_a + v_trial) @ A_to_vox + b_to_vox
                u_t, _ = _multilinear_val_and_jac_torch(u_grid, y_vox_t, right_mat)
                r_t = v_trial + u_t
                r_vox_sq_t = ((r_t * err_to_vox_scale) ** 2).sum(dim=-1)

                imp = r_vox_sq_t < r_vox_sq_a
                v_act[alive] = torch.where(imp.unsqueeze(-1), v_trial, v_a)
                r[alive] = torch.where(imp.unsqueeze(-1), r_t, r_a)
                r_vox_sq[alive] = torch.where(imp, r_vox_sq_t, r_vox_sq_a)
                lam[alive] = torch.where(
                    imp,
                    (lam_a * 0.25).clamp(min=1e-6),
                    (lam_a * 8.0).clamp(max=1e8),
                )

            v[b][idx_tuple] = v_act
            b_steps_used = max(b_steps_used, chunk_steps)
            if bool((r_vox_sq > (max_error_threshold ** 2)).any().item()):
                b_any_unconverged = True

        max_steps_used = max(max_steps_used, b_steps_used)
        if b_any_unconverged:
            all_converged = False

    if stationary_boundary:
        v.mul_(boundary_mask)

    if stats is not None:
        stats['steps'] = max_steps_used
        stats['converged'] = all_converged

    return torch.movedim(v, -1, 1) if channels_first else v


def integrate_time_varying_velocity_field(
    velocity_fields,
    dt: float = None,
    mode: str = 'forward',
    solver: str = 'rk4',
    spacing=None,
    origin=None,
    direction=None
):
    """
    Displacement of the flow of a time-varying velocity field sampled at T time points.

    Starting from phi = 0, one step is taken per time point k with the velocity v_k held
    fixed during the step: phi <- phi + dt * v_k(x + phi) (or its RK4 / midpoint version).

    Parameters
    ----------
    velocity_fields : list of Tensor, or Tensor (T, B, *spatial, dim)
        v_0 ... v_{T-1}, displacements per unit time.
    dt : float, optional
        Step size; default 1 / T (integrates over [0, 1]).
    mode : {'forward', 'backward'}
        'forward': v_0 first, steps of +dt; 'backward': v_{T-1} first, steps of -dt (others raise).
    solver : {'rk4', 'midpoint', 'euler'}
        Other values raise ValueError.
    spacing, origin, direction : optional
        All three: physical mode (mm). Otherwise normalised [-1, 1] units.

    Returns
    -------
    Tensor, same shape as one v_k: the displacement phi(x) - x.
    """
    if solver not in ('rk4', 'midpoint', 'euler'):
        raise ValueError(f"solver must be 'rk4', 'midpoint' or 'euler', got {solver!r}")
    if mode not in ('forward', 'backward'):
        raise ValueError(f"mode must be 'forward' or 'backward', got {mode!r}")
    if isinstance(velocity_fields, torch.Tensor) and velocity_fields.ndim == 5:
        T = velocity_fields.shape[0]
        vel_list = [velocity_fields[i] for i in range(T)]
    else:
        vel_list = list(velocity_fields)
        T = len(vel_list)
    if dt is None:
        dt = 1.0 / T
        
    B = vel_list[0].shape[0]
    dim = vel_list[0].shape[-1]
    spatial = vel_list[0].shape[1:-1]
    device = vel_list[0].device
    dtype = vel_list[0].dtype
    
    # Initialize composite deformation field at identity
    if spacing is not None and origin is not None and direction is not None:
        X_phys = get_physical_grid_torch(spatial, spacing, origin, direction, device=device, dtype=dtype)
        spacing_rev = tuple(reversed(spacing))
        origin_rev = tuple(reversed(origin))
        direction_rev = np.asarray(direction)[::-1, ::-1].copy()
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        shape_t = torch.tensor(list(spatial), device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        M_norm, b_norm = get_physical_to_normalized_affine(shape_t, spacing_t, origin_t, direction_t)
        
        # Continuous displacement field mapping
        phi = torch.zeros_like(vel_list[0])
        
        step_range = range(T) if mode == 'forward' else range(T - 1, -1, -1)
        sign = 1.0 if mode == 'forward' else -1.0
        
        for k in step_range:
            v_k = vel_list[k]
            v_k_cf = torch.movedim(v_k, -1, 1)
            
            def eval_v(curr_phi):
                coords_phys = X_phys + curr_phi
                coords_norm = physical_to_normalized_fast(coords_phys, M_norm, b_norm)
                return sample_field_cf(v_k_cf, coords_norm)
            
            if solver == 'rk4':
                k1 = eval_v(phi)
                k2 = eval_v(phi + (sign * dt / 2.0) * k1)
                k3 = eval_v(phi + (sign * dt / 2.0) * k2)
                k4 = eval_v(phi + (sign * dt) * k3)
                phi = phi + (sign * dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            elif solver == 'midpoint':
                k1 = eval_v(phi)
                k2 = eval_v(phi + (sign * dt / 2.0) * k1)
                phi = phi + (sign * dt) * k2
            else:  # euler
                k1 = eval_v(phi)
                phi = phi + (sign * dt) * k1
                
        return phi
    else:
        # Standard normalized space branch
        grids = [torch.linspace(-1, 1, s, device=device, dtype=dtype) for s in spatial]
        identity = torch.stack(                    # (x, y, z) components on the tensor grid
            torch.meshgrid(*grids, indexing='ij')[::-1], dim=-1
        ).unsqueeze(0).expand(B, *spatial, dim)
        
        phi = torch.zeros_like(vel_list[0])
        step_range = range(T) if mode == 'forward' else range(T - 1, -1, -1)
        sign = 1.0 if mode == 'forward' else -1.0
        
        for k in step_range:
            v_k = vel_list[k]
            v_k_cf = torch.movedim(v_k, -1, 1)
            
            def eval_v(curr_phi):
                sample_coords = identity + curr_phi
                return sample_field_cf(v_k_cf, sample_coords)
            
            if solver == 'rk4':
                k1 = eval_v(phi)
                k2 = eval_v(phi + (sign * dt / 2.0) * k1)
                k3 = eval_v(phi + (sign * dt / 2.0) * k2)
                k4 = eval_v(phi + (sign * dt) * k3)
                phi = phi + (sign * dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            elif solver == 'midpoint':
                k1 = eval_v(phi)
                k2 = eval_v(phi + (sign * dt / 2.0) * k1)
                phi = phi + (sign * dt) * k2
            else:
                k1 = eval_v(phi)
                phi = phi + (sign * dt) * k1
                
        return phi


def _solve_small_spd(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solve ``A x = b`` for a batch of small symmetric positive-definite systems, A (B, n, n),
    b (B, n, 1), returning x as (B, 1, n). Jacobi-equilibrated Gauss-Jordan elimination
    without pivoting using only elementwise kernels: valid for SPD matrices (all pivots
    positive) and several times cheaper than ``torch.linalg.solve`` for n <= 8 on MPS and
    CUDA, where the library solve pays a fixed per-call cost far above the arithmetic.
    """
    n = A.shape[-1]
    # Jacobi equilibration D A D y = D b, x = D y with D = diag(A)^-1/2: unit diagonal keeps
    # the unpivoted elimination accurate on the ill-conditioned Gram matrices Anderson
    # produces (measured 3.8e-7 against torch.linalg.solve on a 32x32 ripple field, 3.3e-6
    # without equilibration).
    d_inv = torch.rsqrt(torch.diagonal(A, dim1=1, dim2=2))          # (B, n)
    A = A * d_inv.unsqueeze(1) * d_inv.unsqueeze(2)
    b = b * d_inv.unsqueeze(2)
    M = torch.cat([A, b], dim=-1)                       # (B, n, n+1)
    for i in range(n):
        row = M[:, i:i + 1, :] / M[:, i:i + 1, i:i + 1]  # normalised pivot row (B, 1, n+1)
        fac = M[:, :, i:i + 1]                           # column i (B, n, 1)
        M = M - fac * row
        M = torch.cat([M[:, :i], row, M[:, i + 1:]], dim=1)
    return (M[:, :, -1] * d_inv).unsqueeze(1)


def update_inverse_field_nd_anderson(
    W_disp: torch.Tensor,
    W_inv_disp: torch.Tensor = None,
    steps: int = 30,
    m: int = 5,
    smoothing_sigma: float = 0.0,
    max_error_threshold: float = 0.1,
    mean_error_threshold: float = 0.001,
    spacing=None,
    origin=None,
    direction=None,
    X_phys=None,
    check_interval: int = 1,
    check_after: int | None = None,
    stats: dict | None = None,
) -> torch.Tensor:
    """
    Inverse of a displacement field: the fixed-point iteration of ``update_inverse_field_nd``
    with Anderson acceleration over the last ``m`` iterates.

    Each accelerated iterate is a least-squares combination of the recent fixed-point
    iterates. It is kept only if its residual is below that of the plain fixed-point step;
    otherwise the plain step is used. The run stops when both the max and the mean error are
    at or below their thresholds.

    Every batch element is solved independently: the clip threshold, the Anderson
    least-squares system and the stopping test are per element, and an element that has met
    the thresholds is frozen (its value no longer changes) while the others continue. A call on
    a batch of fields therefore returns exactly what the same calls on each field alone return.

    Host-device synchronisation happens only at the stopping tests. ``check_interval`` is how
    often (in iterations) the test runs; ``check_after`` moves the first test to that
    iteration (clamped to ``[1, steps]``), so a caller that knows from the previous call how
    many iterations were needed can run them unchecked and pay one synchronisation. Freezing
    makes the result independent of when the test runs. When ``stats`` is a dict, it receives
    ``'steps'`` (the largest number of iterations any element needed to converge, or the
    number run when not converged) and ``'converged'``.
    Other parameters and the return value: see ``update_inverse_field_nd``.
    """
    channels_first = False
    if W_disp.dim() >= 3 and W_disp.shape[1] in [2, 3] and W_disp.shape[-1] not in [2, 3]:
        channels_first = True
        perm = (0,) + tuple(range(2, W_disp.dim())) + (1,)
        W_disp = W_disp.permute(perm)
        if W_inv_disp is not None and W_inv_disp.dim() >= 3 and W_inv_disp.shape[1] in [2, 3]:
            W_inv_disp = W_inv_disp.permute(perm)

    B = W_disp.shape[0]
    dim = W_disp.shape[-1]
    spatial = W_disp.shape[1:-1]
    device = W_disp.device
    dtype = W_disp.dtype

    if spacing is not None and origin is None:
        origin = (0.0,) * dim
    if spacing is not None and direction is None:
        direction = np.eye(dim).flatten()

    use_physical = (X_phys is not None or
                    (spacing is not None and origin is not None and direction is not None))

    if use_physical:
        if X_phys is None:
            X_phys = get_physical_grid_torch(spatial, spacing, origin, direction, device=device, dtype=dtype)
        boundary_mask = get_boundary_mask(spatial, device, dtype)
        spacing_rev = tuple(reversed(spacing))
        origin_rev = tuple(reversed(origin))
        dir_arr = np.asarray(direction)
        if dir_arr.ndim == 1:
            dir_arr = dir_arr.reshape(dim, dim)
        direction_rev = dir_arr[::-1, ::-1].copy()
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        shape_t = torch.tensor(list(spatial), device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        M_norm, b_norm = get_physical_to_normalized_affine(shape_t, spacing_t, origin_t, direction_t)
        W_disp_cf = torch.movedim(W_disp, -1, 1)
    else:
        grids = [torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in spatial]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0).expand(B, *([-1] * (dim + 1)))
        boundary_mask = get_boundary_mask(spatial, device, dtype)
        voxel_scale = torch.tensor(
            [float((s - 1) / 2.0) for s in reversed(spatial)],
            device=device, dtype=dtype
        )
        W_disp_cf = torch.movedim(W_disp, -1, 1)

    if W_inv_disp is None:
        W_inv_disp = -W_disp.clone()

    bshape = (B,) + (1,) * (W_disp.dim() - 1)
    ones_b = torch.ones(bshape, device=device, dtype=dtype)

    def itk_fixed_point_step(v_curr, iteration, compute_mean: bool = True):
        if use_physical:
            coords_phys = X_phys + v_curr
            coords_norm = physical_to_normalized_fast(coords_phys, M_norm, b_norm)
            del coords_phys
            forward_at_inv = sample_field_cf(W_disp_cf, coords_norm)
            del coords_norm
            error = forward_at_inv.add_(v_curr)
            sq_sum = _scaled_sq_sum(error, spacing_t, divide=True)
        else:
            coords = identity + v_curr
            forward_at_inv = sample_field_cf(W_disp_cf, coords)
            del coords
            error = forward_at_inv.add_(v_curr)
            sq_sum = _scaled_sq_sum(error, voxel_scale, divide=False)

        # Compute squared residual sum (B,) before turning sq_sum into scaled_norm in-place,
        # so the 3-channel error field does not need to stay alive until the Anderson check.
        residual_fp = _batch_reduce_sum(sq_sum)
        scaled_norm = sq_sum.sqrt_()

        # Per-element reductions (B,): each field in the batch is independent.
        max_err = _batch_reduce_amax(scaled_norm)
        mean_err = _batch_reduce_mean(scaled_norm) if compute_mean else None

        epsilon = 0.75 if iteration == 0 else 0.5
        clip_threshold = (epsilon * max_err).reshape(bshape)
        clip_scale = torch.where(
            scaled_norm > clip_threshold,
            clip_threshold / scaled_norm.clamp(min=1e-10),
            ones_b,
        )
        del scaled_norm
        error.neg_()
        error.mul_(clip_scale)
        del clip_scale
        error.mul_(epsilon)
        v_new = error.add_(v_curr)
        del error

        if smoothing_sigma > 0.0:
            v_new = separable_gaussian_filter(v_new, smoothing_sigma)     # sigma in voxels

        v_new.mul_(boundary_mask)
        return v_new, max_err, mean_err, residual_fp

    v_k = W_inv_disp.clone()
    R_history: list[torch.Tensor] = []
    G_history: list[torch.Tensor] = []
    done = torch.zeros(B, dtype=torch.bool, device=device)         # element has met the thresholds
    conv_step = torch.full((B,), steps, dtype=torch.int64, device=device)   # 0-based iteration it converged at
    eye_reg = 1e-10 * torch.eye(max(m, 1), device=device, dtype=dtype)
    first_check = int(min(max(check_after, 1), steps)) if check_after is not None else None
    steps_taken = steps
    converged = False
    mps_large_field = (device.type == 'mps' and v_k.numel() >= 16_777_216)

    for iteration in range(steps):
        checks_on = (check_interval is not None and check_interval > 0)
        if first_check is None:
            should_check = checks_on and ((iteration + 1) % check_interval == 0 or iteration == steps - 1)
        else:
            should_check = (iteration + 1 == first_check
                            or (checks_on and iteration + 1 > first_check
                                and (iteration + 1 - first_check) % check_interval == 0)
                            or iteration == steps - 1)
        g_k, max_err, mean_err, residual_fp = itk_fixed_point_step(v_k, iteration, compute_mean=True)
        conv_now = (max_err <= max_error_threshold) & (mean_err <= mean_error_threshold)   # (B,)
        conv_step = torch.where(conv_now & ~done, torch.full_like(conv_step, iteration), conv_step)
        done_next = done | conv_now

        # Check convergence before allocating history/Anderson buffers: when every element in
        # the batch has already met the thresholds at ``v_k`` (idempotent on feasible fields:
        # ``steps_taken == 0`` when the initial guess already satisfies the thresholds), leave
        # ``v_k`` untouched rather than applying an extra unverified fixed-point resample.
        if should_check:
            flag = torch.stack([done_next.all().to(torch.int64), conv_step.max()]).tolist()   # one sync
            if flag[0]:
                del g_k
                converged = True
                steps_taken = int(flag[1])
                break

        g_k_flat = g_k.reshape(B, -1)
        if len(R_history) == m + 1:
            r_k = R_history.pop(0)
            G_history.pop(0)
            torch.sub(g_k_flat, v_k.reshape(B, -1), out=r_k)
        else:
            r_k = g_k_flat - v_k.reshape(B, -1)
        R_history.append(r_k)
        G_history.append(g_k_flat)

        m_k = len(R_history)

        if m_k < 2:
            v_next = g_k
        else:
            n_cols = m_k - 1
            r_newest = R_history[-1]
            g_newest = G_history[-1]
            # Compute residual differences dR_j = r_newest - R_j on the fly into reusable
            # scratch buffers instead of materialising all m difference fields at once.
            diff_i = torch.empty_like(r_newest)
            diff_j = torch.empty_like(r_newest) if n_cols > 1 else None
            gram = torch.zeros(B, n_cols, n_cols, device=device, dtype=dtype)
            rhs = torch.zeros(B, n_cols, 1, device=device, dtype=dtype)
            for i_col in range(n_cols):
                torch.sub(r_newest, R_history[i_col], out=diff_i)
                for b_idx in range(B):
                    rhs[b_idx, i_col, 0] = _dot_1d(diff_i[b_idx], r_newest[b_idx])
                    gram[b_idx, i_col, i_col] = _dot_1d(diff_i[b_idx], diff_i[b_idx])
                for j_col in range(i_col + 1, n_cols):
                    torch.sub(r_newest, R_history[j_col], out=diff_j)
                    for b_idx in range(B):
                        val = _dot_1d(diff_i[b_idx], diff_j[b_idx])
                        gram[b_idx, i_col, j_col] = val
                        gram[b_idx, j_col, i_col] = val
            del diff_j
            gram = gram + eye_reg[:n_cols, :n_cols]
            gamma = _solve_small_spd(gram, rhs).squeeze(1)                    # (B, n)
            v_new_flat = g_newest.clone()
            tmp_flat = torch.empty_like(r_newest)
            for j in range(n_cols):
                torch.sub(r_newest, R_history[j], out=diff_i)
                torch.sub(g_newest, G_history[j], out=tmp_flat)
                tmp_flat.add_(diff_i)
                tmp_flat.mul_(gamma[:, j:j + 1])
                v_new_flat.sub_(tmp_flat)
            del diff_i, tmp_flat

            v_candidate = v_new_flat.reshape(W_inv_disp.shape)
            v_candidate.mul_(boundary_mask)

            if use_physical:
                coords_phys_c = X_phys + v_candidate
                coords_norm_c = physical_to_normalized_fast(coords_phys_c, M_norm, b_norm)
                del coords_phys_c
                fwd_at_c = sample_field_cf(W_disp_cf, coords_norm_c)
                del coords_norm_c
                error_c = fwd_at_c.add_(v_candidate)
                sq_c = _scaled_sq_sum(error_c, spacing_t, divide=True)
                del error_c
                residual_aa = _batch_reduce_sum(sq_c)
                del sq_c
            else:
                coords_c = identity + v_candidate
                fwd_at_c = sample_field_cf(W_disp_cf, coords_c)
                del coords_c
                error_c = fwd_at_c.add_(v_candidate)
                sq_c = _scaled_sq_sum(error_c, voxel_scale, divide=False)
                del error_c
                residual_aa = _batch_reduce_sum(sq_c)
                del sq_c

            accept = (residual_aa <= residual_fp * 1.21).reshape(bshape)      # squared norms: 1.1**2
            v_next = torch.where(accept, v_candidate, g_k)
            del v_candidate

        # Freeze: any element that has already met the thresholds at ``v_k`` (``done_next``)
        # keeps ``v_k`` intact (idempotent); only unconverged elements advance to ``v_next``.
        v_k = torch.where(done_next.reshape(bshape), v_k, v_next)
        del v_next, g_k
        done = done_next

        if not should_check and mps_large_field and (iteration + 1) % 4 == 0:
            torch.mps.synchronize()

    del R_history, G_history
    if stats is not None:
        stats['steps'] = steps_taken
        stats['converged'] = converged

    return torch.movedim(v_k, -1, 1) if channels_first else v_k


def update_inverse_field_nd(
    W_disp: torch.Tensor, 
    W_inv_disp: torch.Tensor = None, 
    steps: int = 30,
    relaxation: float = 1.0,
    smoothing_sigma: float = 0.0,
    method: str = 'anderson',
    max_error_threshold: float = 0.1,
    mean_error_threshold: float = 0.001,
    spacing = None,
    origin = None,
    direction = None,
    X_phys = None,
    check_interval: int = 1,
    max_iters: int | None = None,
    check_after: int | None = None,
    stats: dict | None = None,
    **kwargs
) -> torch.Tensor:
    """
    Inverse of a displacement field u: v with v(x) + u(x + v(x)) = 0, so that x -> x + v(x)
    undoes x -> x + u(x).

    The fixed-point iteration (``method='fixed_point'``; other names raise ValueError) follows
    ITK's InvertDisplacementFieldImageFilter:
    v <- v - eps * clip(v + u(x + v)), with eps = 0.75 on the first iteration and 0.5 after.
    Each voxel's update is clipped to eps * (the current max error). After each step, v is
    optionally Gaussian-smoothed and set to zero on the one-voxel boundary rim.

    Parameters
    ----------
    W_disp : Tensor (B, *spatial, dim) or (B, dim, *spatial)
        The field u to invert.
    W_inv_disp : Tensor, optional
        Starting guess (warm start). Default -u.
    steps : int, default 30
        Maximum iterations (``max_iters`` overrides it).
    relaxation : float, default 1.0
        Extra step scale ('fixed_point' and 'hybrid_lm'; a value other than 1 with 'anderson'
        raises ValueError).
    smoothing_sigma : float, default 0
        Gaussian sigma (voxels) applied to v after each step (0 = none).
    method : {'anderson', 'hybrid_lm', 'fixed_point'}, default 'anderson'
        See ``update_inverse_field_nd_anderson`` / ``update_inverse_field_nd_hybrid_lm``.
    max_error_threshold, mean_error_threshold : float, default 0.1, 0.001
        Stopping thresholds on the max / mean residual |v + u(x + v)|, in voxels; the
        solvers stop when both are met.
    spacing, origin, direction : optional
        ANTs metadata. With all three, u and v are in mm. Otherwise they are in normalised
        [-1, 1] units.
    X_phys : Tensor, optional
        Precomputed physical grid (1, *spatial, dim) of that metadata, to save rebuilding it
        (the metadata is still required: ValueError without it).
    check_interval : int, default 1
        'anderson' and 'fixed_point': how often the stopping test runs.
    check_after : int, optional
        'anderson' and 'fixed_point': iteration of the first stopping test (see
        ``update_inverse_field_nd_anderson``).
    stats : dict, optional
        'anderson' and 'fixed_point': receives ``'steps'`` and ``'converged'``.

    Returns
    -------
    Tensor: v, in the layout of ``W_disp``.
    """
    channels_first = False
    if W_disp.dim() >= 3 and W_disp.shape[1] in [2, 3] and W_disp.shape[-1] not in [2, 3]:
        channels_first = True
        perm = (0,) + tuple(range(2, W_disp.dim())) + (1,)
        W_disp = W_disp.permute(perm)
        if W_inv_disp is not None and W_inv_disp.dim() >= 3 and W_inv_disp.shape[1] in [2, 3]:
            W_inv_disp = W_inv_disp.permute(perm)

    if max_iters is not None:
        steps = max_iters
    if method not in ('anderson', 'hybrid_lm', 'fixed_point'):
        raise ValueError(f"unknown inverse method {method!r}: use 'anderson', 'hybrid_lm' or 'fixed_point'")
    if method == 'anderson' and relaxation != 1.0:
        raise ValueError("relaxation is not used by method='anderson' (use 'fixed_point' or 'hybrid_lm')")
    if method == 'hybrid_lm':
        res = update_inverse_field_nd_hybrid_lm(
            W_disp, W_inv_disp, steps=steps, relaxation=relaxation,
            smoothing_sigma=smoothing_sigma, max_error_threshold=max_error_threshold,
            mean_error_threshold=mean_error_threshold,
            damping_factor=kwargs.get('damping_factor', 1.0),
            spacing=spacing, origin=origin, direction=direction, X_phys=X_phys,
            stationary_boundary=kwargs.get('stationary_boundary', True),
            stats=stats,
        )
        return torch.movedim(res, -1, 1) if channels_first else res

    effective_check_interval = kwargs.get('check_interval', check_interval)
    if method == 'anderson':
        res = update_inverse_field_nd_anderson(
            W_disp, W_inv_disp, steps=steps,
            smoothing_sigma=smoothing_sigma, max_error_threshold=max_error_threshold,
            mean_error_threshold=mean_error_threshold, spacing=spacing,
            origin=origin, direction=direction, X_phys=X_phys,
            check_interval=effective_check_interval,
            check_after=check_after, stats=stats,
        )
        return torch.movedim(res, -1, 1) if channels_first else res

    B = W_disp.shape[0]
    dim = W_disp.shape[-1]
    spatial = W_disp.shape[1:-1]
    device = W_disp.device
    dtype = W_disp.dtype

    if X_phys is not None and (spacing is None or origin is None or direction is None):
        raise ValueError("X_phys needs the spacing / origin / direction it was built from")

    use_physical = (X_phys is not None or
                    (spacing is not None and origin is not None and direction is not None))

    if use_physical:
        if X_phys is None:
            X_phys = get_physical_grid_torch(spatial, spacing, origin, direction, device=device, dtype=dtype)
        boundary_mask = get_boundary_mask(spatial, device, dtype)
        spacing_rev = tuple(reversed(spacing))
        origin_rev = tuple(reversed(origin))
        dir_arr = np.asarray(direction)
        if dir_arr.ndim == 1:
            dir_arr = dir_arr.reshape(dim, dim)
        direction_rev = dir_arr[::-1, ::-1].copy()
        spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
        shape_t = torch.tensor(list(spatial), device=device, dtype=dtype)
        origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
        direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
        M_norm, b_norm = get_physical_to_normalized_affine(shape_t, spacing_t, origin_t, direction_t)
    else:
        grids = [torch.linspace(-1, 1, size, device=device, dtype=dtype) for size in spatial]
        meshgrid = torch.meshgrid(*grids, indexing='ij')
        identity = torch.stack(list(reversed(meshgrid)), dim=-1).unsqueeze(0).expand(B, *([-1] * (dim + 1)))
        boundary_mask = get_boundary_mask(spatial, device, dtype)
        voxel_scale = torch.tensor(
            [float((s - 1) / 2.0) for s in reversed(spatial)],
            device=device, dtype=dtype
        )

    W_disp_cf = torch.movedim(W_disp, -1, 1)
    v_k = -W_disp.clone() if W_inv_disp is None else W_inv_disp.clone()
    bshape = (B,) + (1,) * (W_disp.dim() - 1)
    ones_b = torch.ones(bshape, device=device, dtype=dtype)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    conv_step = torch.full((B,), steps, dtype=torch.int64, device=device)
    first_check = int(min(max(check_after, 1), steps)) if check_after is not None else None
    steps_taken = steps
    converged = False
    mps_large_field = (device.type == 'mps' and v_k.numel() >= 16_777_216)

    for iteration in range(steps):
        checks_on = (effective_check_interval is not None and effective_check_interval > 0)
        if first_check is None:
            should_check = checks_on and (
                (iteration + 1) % effective_check_interval == 0 or iteration == steps - 1
            )
        else:
            should_check = (
                iteration + 1 == first_check
                or (checks_on and iteration + 1 > first_check
                    and (iteration + 1 - first_check) % effective_check_interval == 0)
                or iteration == steps - 1
            )

        if use_physical:
            coords_phys = X_phys + v_k
            coords_norm = physical_to_normalized_fast(coords_phys, M_norm, b_norm)
            del coords_phys
            forward_at_inv = sample_field_cf(W_disp_cf, coords_norm)
            del coords_norm
            error = forward_at_inv.add_(v_k)
            scaled_norm = _scaled_sq_sum(error, spacing_t, divide=True).sqrt_()
        else:
            coords = identity + v_k
            forward_at_inv = sample_field_cf(W_disp_cf, coords)
            del coords
            error = forward_at_inv.add_(v_k)
            scaled_norm = _scaled_sq_sum(error, voxel_scale, divide=False).sqrt_()

        max_err = _batch_reduce_amax(scaled_norm)
        mean_err = _batch_reduce_mean(scaled_norm)
        conv_now = (max_err <= max_error_threshold) & (mean_err <= mean_error_threshold)
        conv_step = torch.where(conv_now & ~done, torch.full_like(conv_step, iteration), conv_step)
        done_next = done | conv_now

        if should_check:
            flag = torch.stack([done_next.all().to(torch.int64), conv_step.max()]).tolist()
            if flag[0]:
                del scaled_norm, error
                converged = True
                steps_taken = int(flag[1])
                break

        epsilon = 0.75 if iteration == 0 else 0.5
        clip_threshold = (epsilon * max_err).reshape(bshape)
        clip_scale = torch.where(
            scaled_norm > clip_threshold,
            clip_threshold / scaled_norm.clamp(min=1e-10),
            ones_b,
        )
        del scaled_norm
        error.neg_()
        error.mul_(clip_scale)
        del clip_scale
        error.mul_(relaxation * epsilon)
        g_k = error.add_(v_k)
        del error

        if smoothing_sigma > 0.0:
            g_k = separable_gaussian_filter(g_k, smoothing_sigma)
        g_k.mul_(boundary_mask)

        v_k = torch.where(done_next.reshape(bshape), v_k, g_k)
        del g_k
        done = done_next

        if not should_check and mps_large_field and (iteration + 1) % 4 == 0:
            torch.mps.synchronize()

    if stats is not None:
        stats['steps'] = steps_taken
        stats['converged'] = converged

    return torch.movedim(v_k, -1, 1) if channels_first else v_k


def compute_inverse_identity_error_nd(
    warp_fwd: torch.Tensor,
    warp_inv: torch.Tensor,
    spacing=None,
    origin=None,
    direction=None,
    is_displacement: bool = True,
    fwd_is_disp: bool = None,
    inv_is_disp: bool = None
) -> torch.Tensor:
    """
    Map of the inverse-consistency error |u_fwd(x) + u_inv(x + u_fwd(x))| in mm: how far
    x -> forward -> inverse lands from x.

    Parameters
    ----------
    warp_fwd, warp_inv : Tensor (B, *spatial, dim) or (*spatial, dim)
        Physical displacements (or absolute positions, see below), tensor order.
    spacing, origin, direction : optional
        ANTs metadata. Default unit spacing, zero origin, identity direction.
    is_displacement : bool, default True
        False: both inputs are absolute positions x + u(x).
    fwd_is_disp, inv_is_disp : bool, optional
        Per-input override of ``is_displacement``.

    Returns
    -------
    Tensor (B, *spatial). No boundary or out-of-domain masking: points that leave the grid
    read u_inv from the border.
    """
    dim = warp_fwd.shape[-1]
    if warp_fwd.dim() == dim + 1:
        warp_fwd = warp_fwd.unsqueeze(0)
    if warp_inv.dim() == dim + 1:
        warp_inv = warp_inv.unsqueeze(0)

    spatial = warp_fwd.shape[1:-1]
    device = warp_fwd.device
    dtype = warp_fwd.dtype

    if spacing is None:
        spacing = [1.0] * dim
    if origin is None:
        origin = [0.0] * dim
    if direction is None:
        direction = np.eye(dim)
    else:
        direction = np.asarray(direction)[:dim, :dim]

    spatial_tuple = tuple(spatial)
    spacing_tuple = tuple(float(s) for s in spacing)
    origin_tuple = tuple(float(o) for o in origin)

    X_phys = get_physical_grid_torch(spatial_tuple, spacing_tuple, origin_tuple, direction, device=device, dtype=dtype)

    use_fwd_disp = is_displacement if fwd_is_disp is None else fwd_is_disp
    use_inv_disp = is_displacement if inv_is_disp is None else inv_is_disp

    disp_fwd = warp_fwd if use_fwd_disp else (warp_fwd - X_phys)
    disp_inv = warp_inv if use_inv_disp else (warp_inv - X_phys)

    y_norm = physical_to_normalized_torch(X_phys + disp_fwd, spatial_tuple, spacing_tuple, origin_tuple, direction)
    disp_inv_sampled = compose_grids(disp_inv, y_norm)

    return torch.norm(disp_fwd + disp_inv_sampled, dim=-1)


def calculate_inverse_identity_error(W_disp: torch.Tensor, W_inv_disp: torch.Tensor, spacing, origin, direction) -> dict:
    """
    Inverse-consistency error of a displacement field and its inverse, in mm:
    |u_inv(x) + u(x + u_inv(x))|, i.e. how far x -> inverse -> forward lands from x. The
    order is the reverse of ``compute_inverse_identity_error_nd``.

    Parameters
    ----------
    W_disp, W_inv_disp : Tensor (1, *spatial, dim) or (*spatial, dim)
        Physical displacements u and u_inv, tensor order (spatial z, y, x).
    spacing, origin, direction : ANTs metadata of the grid (all required).

    Returns
    -------
    dict
        - ``'max_error'``: max over interior voxels.
        - ``'mean_error'``: sum over interior voxels / number of interior voxels.
        - ``'error_map'``: Tensor (*spatial).

        Interior voxels exclude the one-voxel boundary rim and points whose x + u_inv(x)
        leaves the grid; they are set to 0 in the map and excluded from the mean.
    """
    dim = len(spacing)
    if W_disp.ndim == dim + 1:
        W_disp = W_disp.unsqueeze(0)
    if W_inv_disp.ndim == dim + 1:
        W_inv_disp = W_inv_disp.unsqueeze(0)
    spatial = W_disp.shape[1:-1]
    device = W_disp.device
    dtype = W_disp.dtype
    
    X_phys = get_physical_grid_torch(spatial, spacing, origin, direction, device=device, dtype=dtype)
    coords_phys = X_phys + W_inv_disp
    del X_phys
    
    spacing_rev = tuple(reversed(spacing))
    origin_rev = tuple(reversed(origin))
    direction_rev = np.asarray(direction)[::-1, ::-1].copy()
    
    shape_t = torch.tensor(spatial, device=device, dtype=dtype)
    spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
    origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
    direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
    
    coords_norm = physical_to_normalized_torch_cached(coords_phys, shape_t, spacing_t, origin_t, direction_t)
    del coords_phys
    
    forward_at_inv = compose_grids(W_disp, coords_norm)
    valid_mask = ((coords_norm >= -1) & (coords_norm <= 1)).all(dim=-1)
    del coords_norm
    for ax in range(len(spatial)):
        sl_lo = [slice(None)] * valid_mask.ndim
        sl_lo[ax + 1] = slice(0, 1)
        valid_mask[tuple(sl_lo)] = False
        sl_hi = [slice(None)] * valid_mask.ndim
        sl_hi[ax + 1] = slice(-1, None)
        valid_mask[tuple(sl_hi)] = False

    forward_at_inv.add_(W_inv_disp)
    error_norm = torch.linalg.vector_norm(forward_at_inv, dim=-1)
    del forward_at_inv
    error_norm.masked_fill_(~valid_mask, 0.0)
    if valid_mask.device.type == 'mps':
        n_valid = max(float(_batch_reduce_sum(valid_mask.to(dtype)).item()), 1.0)
        err_sum = float(_batch_reduce_sum(error_norm).item())
        err_max = float(_batch_reduce_amax(error_norm).item())
    else:
        n_valid = max(float(valid_mask.sum(dtype=torch.int64).item()), 1.0)
        err_sum = float(error_norm.sum(dtype=torch.float64).item())
        err_max = float(error_norm.max().item())
    del valid_mask
    return {
        'max_error': err_max,
        'mean_error': err_sum / n_valid,
        'error_map': error_norm.squeeze(0),
    }
