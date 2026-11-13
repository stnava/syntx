"""
Digital diffeomorphism constraint & post-composition projection (default on at finish, opt-out via
``digital_diffeomorphism=False``).

Continuous diffeomorphism guarantees (positive Euler step Jacobians along a velocity flow in
``tvf`` / ``syngs``, or positive half-field Jacobians at the midpoint in ``syn`` /
``syn_scattered``) do not automatically guarantee a **digital diffeomorphism** on the discrete
voxel lattice: multilinear re-tabulation of composed maps ``Phi_12 = phi_2 o phi_1^-1`` or
discrete trajectory integration can create isolated 2-voxel central-difference secant folds
(``det(I + D_{2h} u) <= 0``) or sub-voxel creases where asymmetric inverse polishing
(holding ``W_fwd`` fixed) cannot invert a locally 2-to-1 discrete grid map.

This module provides model-agnostic **digital diffeomorphism** primitives across all ``syntx``
registration families (``syn``, ``tvf``, ``syngs``, ``greedy``, ``syn_scattered``):

1. ``enforce_digital_diffeomorphism_nd``:
   - **Symmetric paired mode** (``W_inv is not None``): jointly updates ``(W_fwd, W_inv)`` on
     the sparse 2-ring dilated active defect neighborhood where bidirectional chord ICE exceeds
     ``ice_tol`` (voxels) or physical central-difference Jacobian determinant
     ``det(I + D_{2h} u)`` falls below ``jac_floor`` (respecting full ``spacing`` and
     ``direction`` cosines, or normalized ``[-1, 1]`` coordinates when ``spacing`` is omitted).
   - **Unidirectional mode** (``W_inv is None``, e.g. ``syntx.greedy`` with
     ``return_inverse=False``): unfolds any physical central-difference Jacobian violations
     ``det(I + D_{2h} W_fwd) < jac_floor`` under a proximity anchor ``anchor_weight * ||u - u_0||^2``
     without requiring an inverse field.
   - **Strict idempotency & locality**: if no interior voxel violates ``ice_tol`` or
     ``jac_floor``, returns inputs bitwise untouched with ``stats['steps'] == 0``. Otherwise,
     all voxels outside the dilated defect neighborhood remain bitwise identical.

2. ``pullback_digital_diffeomorphism_to_halves``:
   - Propagates a small endpoint correction ``delta_fwd = W_fwd_polished - W_fwd_init``
     symmetrically back onto midpoint half-fields ``(w_l2r, w_r2l, w_l2r_inv, w_r2l_inv)`` in
     both physical-mm layout (``syntx.syn``) and normalized ``[-1, 1]`` layout
     (``syntx.syn_scattered``), preserving half-field diffeomorphic integrity and keeping
     ``syntx.liouville_determinant`` consistent with exported endpoints.

3. ``enforce_digital_diffeomorphism_jax``:
   - Backend-parity wrapper for JAX arrays (``GEMINI.md`` §2).
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F

from .grid import (
    get_physical_grid_torch,
    get_physical_to_normalized_affine,
    physical_to_normalized_fast,
    sample_field_cf,
)
from .inverse import (
    _multilinear_val_and_jac_torch,
    _solve_spd_lm_torch,
)
from .smoothing import get_boundary_mask


def _cofactor_transpose_nd(F_mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(det(F), cof(F))`` for ``(..., dim, dim)`` matrices ``F_mat`` in 2-D or 3-D,
    where ``cof(F)[..., i, j] = d det(F) / d F[..., i, j]``."""
    dim = F_mat.shape[-1]
    if dim == 2:
        a = F_mat[..., 0, 0]
        b = F_mat[..., 0, 1]
        c = F_mat[..., 1, 0]
        d = F_mat[..., 1, 1]
        det = a * d - b * c
        r0 = torch.stack([d, -c], dim=-1)
        r1 = torch.stack([-b, a], dim=-1)
        return det, torch.stack([r0, r1], dim=-2)

    a, b, c = F_mat[..., 0, 0], F_mat[..., 0, 1], F_mat[..., 0, 2]
    d, e, f = F_mat[..., 1, 0], F_mat[..., 1, 1], F_mat[..., 1, 2]
    g, h, k = F_mat[..., 2, 0], F_mat[..., 2, 1], F_mat[..., 2, 2]

    c00 = e * k - f * h
    c01 = -(d * k - f * g)
    c02 = d * h - e * g

    c10 = -(b * k - c * h)
    c11 = a * k - c * g
    c12 = -(a * h - b * g)

    c20 = b * f - c * e
    c21 = -(a * f - c * d)
    c22 = a * e - b * d

    det = a * c00 + b * c01 + c * c02
    cof = torch.stack(
        [
            torch.stack([c00, c01, c02], dim=-1),
            torch.stack([c10, c11, c12], dim=-1),
            torch.stack([c20, c21, c22], dim=-1),
        ],
        dim=-2,
    )
    return det, cof


def _physical_fd_det_only(
    u: torch.Tensor,
    steps_rev: list[float],
    D_rev_inv: torch.Tensor,
) -> torch.Tensor:
    """Fast 1-pass physical central-difference ``det(I + (du/d_axis) @ D_rev_inv)`` without
    allocating cofactor tensors or running the 3-axis adjoint gradient pass."""
    dim = u.shape[-1]
    grads = torch.gradient(u, spacing=steps_rev, dim=tuple(range(1, dim + 1)))
    G = torch.stack(grads, dim=-1)  # (B, *spatial, comp_i, axis_j)
    del grads
    eye = torch.eye(dim, device=u.device, dtype=u.dtype)
    F_phys = G @ D_rev_inv + eye
    del G
    if dim == 2:
        return F_phys[..., 0, 0] * F_phys[..., 1, 1] - F_phys[..., 0, 1] * F_phys[..., 1, 0]
    a, b, c = F_phys[..., 0, 0], F_phys[..., 0, 1], F_phys[..., 0, 2]
    d, e, f = F_phys[..., 1, 0], F_phys[..., 1, 1], F_phys[..., 1, 2]
    g, h, k = F_phys[..., 2, 0], F_phys[..., 2, 1], F_phys[..., 2, 2]
    return a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g)


def _physical_fd_det_and_grad(
    u: torch.Tensor,
    steps_rev: list[float],
    D_rev_inv: torch.Tensor,
    jac_floor: float,
    voxel_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute physical central-difference ``det(I + (du/d_axis) @ D_rev_inv)``, quadratic hinge
    energy ``0.5 * sum(w * relu(jac_floor - det)^2)``, and exact discrete adjoint gradient w.r.t.
    ``u`` (shape ``(B, *spatial, dim)``, tensor component order)."""
    dim = u.shape[-1]
    grads = torch.gradient(u, spacing=steps_rev, dim=tuple(range(1, dim + 1)))
    G = torch.stack(grads, dim=-1)  # (B, *spatial, comp_i, axis_j)
    eye = torch.eye(dim, device=u.device, dtype=u.dtype)
    F_phys = G @ D_rev_inv + eye
    det, cof = _cofactor_transpose_nd(F_phys)
    viol = F.relu(float(jac_floor) - det)
    if voxel_weights is None:
        energy = 0.5 * (viol * viol).sum(dim=tuple(range(1, dim + 1)))  # (B,)
        w_viol = viol
    else:
        energy = 0.5 * (voxel_weights * viol * viol).sum(dim=tuple(range(1, dim + 1)))
        w_viol = voxel_weights * viol

    if not bool((w_viol > 0.0).any().item()):
        return det, energy, torch.zeros_like(u)

    dE_dF = -w_viol.unsqueeze(-1).unsqueeze(-1) * cof
    dE_dG = dE_dF @ D_rev_inv.transpose(-1, -2)

    grad_u = torch.zeros_like(u)
    for ax in range(dim):
        h = float(steps_rev[ax])
        n_ax = u.shape[ax + 1]
        p = dE_dG[..., ax]
        if n_ax < 2:
            continue
        adj = torch.zeros_like(p)
        sl_all = [slice(None)] * p.ndim

        def _sl(start, stop):
            s = list(sl_all)
            s[ax + 1] = slice(start, stop)
            return tuple(s)

        if n_ax == 2:
            adj[_sl(0, 1)] = (-p[_sl(0, 1)] - p[_sl(1, 2)]) / h
            adj[_sl(1, 2)] = (p[_sl(0, 1)] + p[_sl(1, 2)]) / h
        else:
            adj[_sl(0, 1)] = adj[_sl(0, 1)] - p[_sl(0, 1)] / h
            adj[_sl(1, 2)] = adj[_sl(1, 2)] + p[_sl(0, 1)] / h
            adj[_sl(2, n_ax)] = adj[_sl(2, n_ax)] + p[_sl(1, n_ax - 1)] / (2.0 * h)
            adj[_sl(0, n_ax - 2)] = adj[_sl(0, n_ax - 2)] - p[_sl(1, n_ax - 1)] / (2.0 * h)
            adj[_sl(n_ax - 1, n_ax)] = adj[_sl(n_ax - 1, n_ax)] + p[_sl(n_ax - 1, n_ax)] / h
            adj[_sl(n_ax - 2, n_ax - 1)] = adj[_sl(n_ax - 2, n_ax - 1)] - p[_sl(n_ax - 1, n_ax)] / h
        grad_u.add_(adj)

    return det, energy, grad_u


def _dilate_mask_nd(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Binary dilation of ``(B, *spatial)`` boolean mask with Chebyshev radius ``radius``
    without using 5-D ``F.pad`` (safe on MPS and CPU)."""
    if radius <= 0:
        return mask
    dim = mask.ndim - 1
    out = mask.clone()
    for _ in range(radius):
        cur = out
        nxt = cur.clone()
        for ax in range(1, dim + 1):
            n = cur.shape[ax]
            if n <= 1:
                continue
            sl_lo = [slice(None)] * cur.ndim
            sl_hi = [slice(None)] * cur.ndim
            sl_lo[ax] = slice(0, n - 1)
            sl_hi[ax] = slice(1, n)
            nxt[tuple(sl_lo)] = nxt[tuple(sl_lo)] | cur[tuple(sl_hi)]
            nxt[tuple(sl_hi)] = nxt[tuple(sl_hi)] | cur[tuple(sl_lo)]
        out = nxt
    return out


def _coords_to_flat_idx(coords: torch.Tensor, spatial: tuple[int, ...]) -> torch.Tensor:
    """Convert ``(N, 1 + dim)`` integer ``(b, i_0, ..., i_{dim-1})`` coordinates into 1-D flat
    indices into ``(B * prod(spatial),)``."""
    n_full = max(int(np.prod(spatial)), 1)
    fl = coords[:, 0] * n_full
    st = 1
    for d_ax in range(len(spatial) - 1, -1, -1):
        fl = fl + coords[:, d_ax + 1] * st
        st *= spatial[d_ax]
    return fl


def _build_sparse_fd_stencil(
    active_mask: torch.Tensor,
    steps_fd: list[float],
    dtype: torch.dtype,
):
    """Build exact sparse 2-voxel central-difference stencil tables around ``active_mask``
    (shape ``(B, *spatial)``).

    Any voxel outside ``sten_mask = _dilate_mask_nd(active_mask, 1)`` has zero neighbors inside
    ``active_mask``, so its central-difference Jacobian ``det(I + D_{2h} u)`` is strictly
    invariant to perturbations supported on ``active_mask``. Evaluating central differences on
    ``sten_mask`` requires field values only on ``pool_mask = _dilate_mask_nd(sten_mask, 1)``,
    reducing full-resolution 3-D ``(192x224x192 = 8.2M)`` per-iteration work to ``O(|active|)``
    even when defects are scattered across opposite hemispheres!
    """
    B = active_mask.shape[0]
    spatial = tuple(active_mask.shape[1:])
    dim = len(spatial)
    device = active_mask.device
    n_full = max(int(np.prod(spatial)), 1)

    sten_mask = _dilate_mask_nd(active_mask, 1)
    pool_mask = _dilate_mask_nd(sten_mask, 1)

    act_coords = torch.nonzero(active_mask, as_tuple=False)
    sten_coords = torch.nonzero(sten_mask, as_tuple=False)
    pool_coords = torch.nonzero(pool_mask, as_tuple=False)

    act_flat = _coords_to_flat_idx(act_coords, spatial)
    sten_flat = _coords_to_flat_idx(sten_coords, spatial)
    pool_flat = _coords_to_flat_idx(pool_coords, spatial)

    lut_pool = torch.full((B * n_full,), -1, dtype=torch.long, device=device)
    lut_pool[pool_flat] = torch.arange(pool_flat.shape[0], device=device)
    act_in_pool = lut_pool[act_flat]

    lo_in_pool_list = []
    hi_in_pool_list = []
    inv_span_list = []
    for ax in range(dim):
        n_ax = spatial[ax]
        h = float(steps_fd[ax])
        c_ax = sten_coords[:, ax + 1]
        if n_ax <= 1:
            lo_c = sten_coords
            hi_c = sten_coords
            inv_s = torch.zeros((sten_coords.shape[0],), device=device, dtype=dtype)
        else:
            lo_idx = (c_ax - 1).clamp(min=0)
            hi_idx = (c_ax + 1).clamp(max=n_ax - 1)
            lo_c = sten_coords.clone()
            hi_c = sten_coords.clone()
            lo_c[:, ax + 1] = lo_idx
            hi_c[:, ax + 1] = hi_idx
            span = (hi_idx - lo_idx).to(dtype) * h
            inv_s = torch.where(span > 0.0, 1.0 / span.clamp(min=1e-12), torch.zeros_like(span))
        lo_in_pool_list.append(lut_pool[ _coords_to_flat_idx(lo_c, spatial) ])
        hi_in_pool_list.append(lut_pool[ _coords_to_flat_idx(hi_c, spatial) ])
        inv_span_list.append(inv_s)

    del lut_pool
    lo_in_pool = torch.stack(lo_in_pool_list, dim=-1)    # (K_sten, dim)
    hi_in_pool = torch.stack(hi_in_pool_list, dim=-1)    # (K_sten, dim)
    inv_span = torch.stack(inv_span_list, dim=-1)        # (K_sten, dim)
    sten_batch = sten_coords[:, 0]
    return {
        "sten_mask": sten_mask,
        "pool_mask": pool_mask,
        "act_coords": act_coords,
        "act_flat": act_flat,
        "sten_flat": sten_flat,
        "pool_flat": pool_flat,
        "act_in_pool": act_in_pool,
        "lo_in_pool": lo_in_pool,
        "hi_in_pool": hi_in_pool,
        "inv_span": inv_span,
        "sten_batch": sten_batch,
    }


def _eval_sparse_fd_det(
    u_pool: torch.Tensor,
    stencil: dict,
    D_rev_inv: torch.Tensor,
    eye: torch.Tensor,
    use_physical: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate exact central-difference physical Jacobian determinant ``det(I + D_{2h} u)`` and
    cofactor matrix on the ``K_sten`` stencil coordinates from ``u_pool`` ``(K_pool, dim)``."""
    u_fd = u_pool if use_physical else torch.flip(u_pool, dims=[-1])
    u_hi = u_fd[stencil["hi_in_pool"]]  # (K_sten, dim_ax, dim_comp)
    u_lo = u_fd[stencil["lo_in_pool"]]  # (K_sten, dim_ax, dim_comp)
    # G[k, comp_i, axis_j] = (u_hi[k, axis_j, comp_i] - u_lo[k, axis_j, comp_i]) * inv_span[k, axis_j]
    diff = (u_hi - u_lo) * stencil["inv_span"].unsqueeze(-1)
    G = diff.transpose(-1, -2)
    F_phys = G @ D_rev_inv + eye
    return _cofactor_transpose_nd(F_phys)


def _gather_multilinear_sparse(
    u0_flat: torch.Tensor,
    lut_act: torch.Tensor,
    delta_act: torch.Tensor,
    batch_idx: torch.Tensor,
    vox_coords: torch.Tensor,
    spatial: tuple[int, ...],
) -> torch.Tensor:
    """Differentiable sparse multilinear interpolation (``padding_mode='border'``,
    ``align_corners=True``) of ``u = u0 + scatter(delta_act)`` at continuous spatial voxel
    coordinates ``vox_coords`` ``(M, dim)`` for batch indices ``batch_idx`` ``(M,)``.

    Gathers only the ``2**dim`` corners for the ``M`` queried points directly from flat tensors
    with zero 5-D ``grid_sample`` or bounding-box allocations (fast and deterministic on MPS/CPU).
    """
    dim = len(spatial)
    dtype = u0_flat.dtype
    if vox_coords.shape[0] == 0:
        return vox_coords.new_zeros((0, u0_flat.shape[-1]))
    n_full = max(int(np.prod(spatial)), 1)
    base_off = batch_idx * n_full

    if dim == 2:
        H, W = spatial
        vy = vox_coords[:, 0].clamp(0.0, float(max(H - 1, 0)))
        vx = vox_coords[:, 1].clamp(0.0, float(max(W - 1, 0)))
        iy0 = torch.floor(vy).long().clamp(0, max(H - 2, 0))
        ix0 = torch.floor(vx).long().clamp(0, max(W - 2, 0))
        iy1 = (iy0 + 1).clamp(max=max(H - 1, 0))
        ix1 = (ix0 + 1).clamp(max=max(W - 1, 0))
        ty = (vy - iy0.to(dtype)).unsqueeze(-1)
        tx = (vx - ix0.to(dtype)).unsqueeze(-1)
        omy, omx = 1.0 - ty, 1.0 - tx
        corners = (
            (base_off + iy0 * W + ix0, omy * omx),
            (base_off + iy0 * W + ix1, omy * tx),
            (base_off + iy1 * W + ix0, ty * omx),
            (base_off + iy1 * W + ix1, ty * tx),
        )
    else:
        Z, Y, X = spatial
        vz = vox_coords[:, 0].clamp(0.0, float(max(Z - 1, 0)))
        vy = vox_coords[:, 1].clamp(0.0, float(max(Y - 1, 0)))
        vx = vox_coords[:, 2].clamp(0.0, float(max(X - 1, 0)))
        iz0 = torch.floor(vz).long().clamp(0, max(Z - 2, 0))
        iy0 = torch.floor(vy).long().clamp(0, max(Y - 2, 0))
        ix0 = torch.floor(vx).long().clamp(0, max(X - 2, 0))
        iz1 = (iz0 + 1).clamp(max=max(Z - 1, 0))
        iy1 = (iy0 + 1).clamp(max=max(Y - 1, 0))
        ix1 = (ix0 + 1).clamp(max=max(X - 1, 0))
        tz = (vz - iz0.to(dtype)).unsqueeze(-1)
        ty = (vy - iy0.to(dtype)).unsqueeze(-1)
        tx = (vx - ix0.to(dtype)).unsqueeze(-1)
        omz, omy, omx = 1.0 - tz, 1.0 - ty, 1.0 - tx
        sz, sy = Y * X, X
        corners = (
            (base_off + iz0 * sz + iy0 * sy + ix0, omz * omy * omx),
            (base_off + iz0 * sz + iy0 * sy + ix1, omz * omy * tx),
            (base_off + iz0 * sz + iy1 * sy + ix0, omz * ty * omx),
            (base_off + iz0 * sz + iy1 * sy + ix1, omz * ty * tx),
            (base_off + iz1 * sz + iy0 * sy + ix0, tz * omy * omx),
            (base_off + iz1 * sz + iy0 * sy + ix1, tz * omy * tx),
            (base_off + iz1 * sz + iy1 * sy + ix0, tz * ty * omx),
            (base_off + iz1 * sz + iy1 * sy + ix1, tz * ty * tx),
        )

    out = vox_coords.new_zeros((vox_coords.shape[0], u0_flat.shape[-1]))
    has_delta = delta_act.shape[0] > 0
    for c_flat, w_c in corners:
        val = u0_flat[c_flat]
        if has_delta:
            a_idx = lut_act[c_flat]
            is_act = (a_idx >= 0).to(dtype).unsqueeze(-1)
            val = val + delta_act[a_idx.clamp_min(0)] * is_act
        out = out + w_c * val
    return out


def _pointwise_lm_refine_subset(
    target_field: torch.Tensor,
    init_inv_field: torch.Tensor,
    subset_mask: torch.Tensor,
    X_base: torch.Tensor,
    A_to_vox: torch.Tensor,
    b_to_vox: torch.Tensor,
    right_mat: torch.Tensor,
    err_to_vox_mat: torch.Tensor,
    steps: int,
    tol_vox: float,
    max_step_unit: float,
) -> torch.Tensor:
    """Run damped Levenberg-Marquardt point-wise inversion ``v(x) + target_field(x + v(x)) = 0``
    strictly on ``subset_mask`` ``(B, *spatial)`` in ``O(|subset| * steps)`` time, leaving all
    voxels outside ``subset_mask`` bitwise untouched."""
    if steps <= 0 or not bool(subset_mask.any().item()):
        return init_inv_field
    B = target_field.shape[0]
    dim = target_field.shape[-1]
    device = target_field.device
    dtype = target_field.dtype
    I_d = torch.eye(dim, device=device, dtype=dtype).unsqueeze(0)
    tol_sq = float(tol_vox) * float(tol_vox)
    out = init_inv_field.clone()

    for b in range(B):
        idx = torch.nonzero(subset_mask[b], as_tuple=False)
        if idx.shape[0] == 0:
            continue
        idx_t = tuple(idx[:, d] for d in range(dim))
        u_grid = target_field[b]
        x_a = X_base[0 if X_base.shape[0] == 1 else b][idx_t]
        v_a = out[b][idx_t].clone()
        lam = torch.full((idx.shape[0],), 1e-3, device=device, dtype=dtype)

        for _ in range(int(steps)):
            y_vox = (x_a + v_a) @ A_to_vox + b_to_vox
            u_a, grad_u_a = _multilinear_val_and_jac_torch(u_grid, y_vox, right_mat)
            r_a = v_a + u_a
            r_vox = r_a @ err_to_vox_mat
            r_sq = (r_vox * r_vox).sum(dim=-1)
            if float(r_sq.amax().item()) <= tol_sq:
                break
            J = I_d + grad_u_a
            JT = J.transpose(-1, -2)
            JTJ = JT @ J
            rhs = -(JT @ r_a.unsqueeze(-1)).squeeze(-1)
            dv = _solve_spd_lm_torch(JTJ, rhs, lam)
            dv_norm = torch.linalg.vector_norm(dv, dim=-1, keepdim=True)
            dv = torch.where(
                dv_norm > max_step_unit,
                dv * (max_step_unit / dv_norm.clamp(min=1e-12)),
                dv,
            )
            v_trial = v_a + dv
            y_vox_t = (x_a + v_trial) @ A_to_vox + b_to_vox
            u_t, _ = _multilinear_val_and_jac_torch(u_grid, y_vox_t, right_mat)
            r_t_vox = (v_trial + u_t) @ err_to_vox_mat
            r_sq_t = (r_t_vox * r_t_vox).sum(dim=-1)
            imp = r_sq_t < r_sq
            v_a = torch.where(imp.unsqueeze(-1), v_trial, v_a)
            lam = torch.where(
                imp,
                (lam * 0.25).clamp(min=1e-6),
                (lam * 8.0).clamp(max=1e8),
            )
        out[b][idx_t] = v_a
    return out


def _splat_multilinear_grad_inplace(
    grad_grid: torch.Tensor,
    vox_coords: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    """Adjoint of multilinear interpolation (``padding_mode='border', align_corners=True``):
    scatter-adds ``weights`` ``(N, dim)`` at continuous ``vox_coords`` ``(N, dim)`` into
    ``grad_grid`` ``(*spatial, dim)``."""
    dim = grad_grid.shape[-1]
    dtype = grad_grid.dtype
    if vox_coords.shape[0] == 0:
        return

    if dim == 2:
        H, W, _ = grad_grid.shape
        vy = vox_coords[:, 0].clamp(0.0, float(max(H - 1, 0)))
        vx = vox_coords[:, 1].clamp(0.0, float(max(W - 1, 0)))
        iy0 = torch.floor(vy).long().clamp(0, max(H - 2, 0))
        ix0 = torch.floor(vx).long().clamp(0, max(W - 2, 0))
        iy1 = (iy0 + 1).clamp(max=max(H - 1, 0))
        ix1 = (ix0 + 1).clamp(max=max(W - 1, 0))
        ty = (vy - iy0.to(dtype)).unsqueeze(-1)
        tx = (vx - ix0.to(dtype)).unsqueeze(-1)
        omy, omx = 1.0 - ty, 1.0 - tx

        flat = grad_grid.view(-1, 2)
        flat.index_add_(0, iy0 * W + ix0, weights * (omy * omx))
        flat.index_add_(0, iy0 * W + ix1, weights * (omy * tx))
        flat.index_add_(0, iy1 * W + ix0, weights * (ty * omx))
        flat.index_add_(0, iy1 * W + ix1, weights * (ty * tx))
        return

    Z, Y, X, _ = grad_grid.shape
    vz = vox_coords[:, 0].clamp(0.0, float(max(Z - 1, 0)))
    vy = vox_coords[:, 1].clamp(0.0, float(max(Y - 1, 0)))
    vx = vox_coords[:, 2].clamp(0.0, float(max(X - 1, 0)))
    iz0 = torch.floor(vz).long().clamp(0, max(Z - 2, 0))
    iy0 = torch.floor(vy).long().clamp(0, max(Y - 2, 0))
    ix0 = torch.floor(vx).long().clamp(0, max(X - 2, 0))
    iz1 = (iz0 + 1).clamp(max=max(Z - 1, 0))
    iy1 = (iy0 + 1).clamp(max=max(Y - 1, 0))
    ix1 = (ix0 + 1).clamp(max=max(X - 1, 0))
    tz = (vz - iz0.to(dtype)).unsqueeze(-1)
    ty = (vy - iy0.to(dtype)).unsqueeze(-1)
    tx = (vx - ix0.to(dtype)).unsqueeze(-1)
    omz, omy, omx = 1.0 - tz, 1.0 - ty, 1.0 - tx

    flat = grad_grid.view(-1, 3)
    sz, sy = Y * X, X
    flat.index_add_(0, iz0 * sz + iy0 * sy + ix0, weights * (omz * omy * omx))
    flat.index_add_(0, iz0 * sz + iy0 * sy + ix1, weights * (omz * omy * tx))
    flat.index_add_(0, iz0 * sz + iy1 * sy + ix0, weights * (omz * ty * omx))
    flat.index_add_(0, iz0 * sz + iy1 * sy + ix1, weights * (omz * ty * tx))
    flat.index_add_(0, iz1 * sz + iy0 * sy + ix0, weights * (tz * omy * omx))
    flat.index_add_(0, iz1 * sz + iy0 * sy + ix1, weights * (tz * omy * tx))
    flat.index_add_(0, iz1 * sz + iy1 * sy + ix0, weights * (tz * ty * omx))
    flat.index_add_(0, iz1 * sz + iy1 * sy + ix1, weights * (tz * ty * tx))


def _smooth_max_nd(
    values: torch.Tensor,
    spatial_dims: tuple[int, ...],
    beta: float = 20.0,
) -> torch.Tensor:
    """Differentiable log-sum-exp smooth ``L_infinity`` approximation
    ``(1 / beta) * logmeanexp(beta * values)`` over ``spatial_dims``.

    Unlike ``torch.mean(values**2)`` (which dilutes a single folded or inconsistent voxel by
    ``1 / N_voxels ~ 1e-6`` on a 3-D volume), ``logmeanexp`` concentrates ``softmax(beta * v)``
    gradient weight onto the worst-case ``L_infinity`` violations while converging smoothly to
    ``amax(values)`` as ``beta -> inf`` and remaining identically ``0`` when ``values == 0``
    everywhere.
    """
    if beta <= 0.0:
        return values.amax(dim=spatial_dims)
    flat = values.reshape(values.shape[0], -1)
    vmax = flat.amax(dim=1, keepdim=True)
    # Subtract log(N)/beta from logsumexp and add log(N)/beta back -> (1/beta)*log(1 + sum_{v>0}(exp(beta*v)-1)/N),
    # or equivalently (1/beta)*log(sum_i exp(beta*v_i) / N) + log(N)/beta = (1/beta)*logsumexp(beta*v) - log(N)/beta + log(N)/beta:
    # Using (1/beta) * log(1 + (sum_i (exp(beta * v_i) - 1)) / N) — wait: standard shifted logsumexp:
    #   lse_0(v) = (1/beta) * (log(sum_i exp(beta * v_i)) - log(N))
    # Notice lse_0(0) == 0, lse_0(v) >= 0 for ALL v >= 0 (since exp(beta*v_i) >= 1 => sum >= N => log(sum) >= log(N)),
    # AND grad_v lse_0(v) == softmax(beta * v) > 0 for ANY single violating voxel regardless of how small vmax is!
    n_elem = float(max(flat.shape[1], 1))
    shifted = torch.exp(float(beta) * (flat - vmax))
    lse0 = vmax.squeeze(1) + (torch.log(shifted.sum(dim=1)) - float(np.log(n_elem))) / float(beta)
    return torch.where(
        vmax.squeeze(1) > 0.0,
        F.relu(lse0),
        torch.zeros_like(vmax.squeeze(1)),
    )


def digital_diffeomorphism_loss(
    warp_field: torch.Tensor,
    warp_inv: Optional[torch.Tensor] = None,
    *,
    spacing=None,
    origin=None,
    direction=None,
    X_phys: Optional[torch.Tensor] = None,
    jac_floor: float = 0.05,
    ice_tol: float = 0.0,
    ice_weight: float = 1.0,
    linf_beta: Optional[float] = 25.0,
    mean_weight: float = 0.1,
    stationary_boundary: bool = True,
    is_physical: Optional[bool] = None,
) -> torch.Tensor:
    """Differentiable ``L_infinity`` **digital diffeomorphism** loss following the Anderson
    worst-case (supremum-norm) inversion philosophy.

    Measures worst-case discrete 2-mm central-difference Jacobian folding violations
    ``v_jac(x) = relu(jac_floor - det(I + D_{2h} u(x)))`` (and, when ``warp_inv`` is supplied,
    both ``v_jac(u_inv)`` and bidirectional voxel-chord inverse-consistency violations
    ``v_ice(x) = relu(||r(x)||_vox - ice_tol)``) via a smooth ``L_infinity`` norm:

    - ``linf_beta > 0`` (default ``25.0``): smooth ``L_infinity`` ``logmeanexp`` norm
      ``(1/beta) log((1/|Omega|) sum_x exp(beta * v(x))) + mean_weight * mean(v(x)^2)``, which
      concentrates ``softmax(beta * v)`` gradient directly onto the worst offending voxels
      instead of diluting single-voxel folds by ``1 / N_voxels``.
    - ``linf_beta is None`` or ``<= 0``: exact ``L_infinity`` ``amax_x v(x)`` subgradient
      plus ``mean_weight * mean(v(x)^2)``.

    Strictly ``0.0`` when every interior voxel satisfies ``det >= jac_floor`` and
    ``ICE <= ice_tol``. Respects full ``spacing``, ``origin``, and oblique ``direction``
    cosines (as well as normalized ``[-1, 1]`` grids when ``is_physical=False``).
    """
    def _to_cl(w: torch.Tensor) -> torch.Tensor:
        cf = w.dim() >= 3 and w.shape[1] in (2, 3) and w.shape[-1] not in (2, 3)
        out = w.permute((0,) + tuple(range(2, w.dim())) + (1,)) if cf else w
        if out.dim() == out.shape[-1] + 1:
            out = out.unsqueeze(0)
        return out

    u_f = _to_cl(warp_field)
    u_i = _to_cl(warp_inv) if warp_inv is not None else None
    dim = u_f.shape[-1]
    spatial = tuple(u_f.shape[1:-1])
    device = u_f.device
    dtype = u_f.dtype
    spatial_dims = tuple(range(1, dim + 1))

    if is_physical is None:
        is_physical = getattr(
            warp_field, "is_physical", spacing is not None or X_phys is not None
        )

    bmask = get_boundary_mask(spatial, device, dtype)
    dom = (
        (bmask[0, ..., 0] > 0.5).to(dtype).unsqueeze(0)
        if stationary_boundary
        else torch.ones((1, *spatial), device=device, dtype=dtype)
    )

    if is_physical:
        sp = [1.0] * dim if spacing is None else [float(x) for x in spacing]
        org = [0.0] * dim if origin is None else [float(x) for x in origin]
        steps_rev = list(reversed(sp))
        origin_rev = list(reversed(org))
        dir_arr = (
            np.eye(dim)
            if direction is None
            else np.asarray(
                direction.detach().cpu().numpy()
                if isinstance(direction, torch.Tensor)
                else direction,
                dtype=np.float64,
            ).reshape(dim, dim)
        )
        dir_rev = dir_arr[::-1, ::-1].copy()
        D_inv_np = np.linalg.inv(dir_rev)
        D_rev_inv = torch.as_tensor(D_inv_np, device=device, dtype=dtype)
        comps_f = u_f
        comps_i = u_i
    else:
        steps_rev = [2.0 / max(n - 1, 1) for n in spatial]
        D_rev_inv = torch.eye(dim, device=device, dtype=dtype)
        comps_f = torch.flip(u_f, dims=[-1])
        comps_i = torch.flip(u_i, dims=[-1]) if u_i is not None else None

    eye = torch.eye(dim, device=device, dtype=dtype)

    def _jac_viol(comps: torch.Tensor) -> torch.Tensor:
        grads = torch.gradient(comps, spacing=steps_rev, dim=spatial_dims)
        G = torch.stack(grads, dim=-1)
        det, _ = _cofactor_transpose_nd(G @ D_rev_inv + eye)
        return F.relu(float(jac_floor) - det) * dom

    def _reduce_linf(viol: torch.Tensor) -> torch.Tensor:
        if linf_beta is None or float(linf_beta) <= 0.0:
            linf_term = viol.amax(dim=spatial_dims)
        else:
            linf_term = _smooth_max_nd(viol, spatial_dims, beta=float(linf_beta))
        if float(mean_weight) > 0.0:
            linf_term = linf_term + float(mean_weight) * (viol * viol).mean(dim=spatial_dims)
        return linf_term.mean()

    loss = _reduce_linf(_jac_viol(comps_f))

    if u_i is not None:
        loss = loss + _reduce_linf(_jac_viol(comps_i))
        if float(ice_weight) > 0.0:
            if is_physical:
                X_b = (
                    X_phys
                    if X_phys is not None
                    else get_physical_grid_torch(
                        spatial, sp, org, dir_arr, device=device, dtype=dtype
                    )
                )
                spacing_t = torch.tensor(steps_rev, device=device, dtype=dtype)
                shape_t = torch.tensor(list(spatial), device=device, dtype=dtype)
                origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
                direction_t = torch.tensor(dir_rev, device=device, dtype=dtype)
                M_norm, b_norm = get_physical_to_normalized_affine(
                    shape_t, spacing_t, origin_t, direction_t
                )
                sp_rev_np = np.asarray(steps_rev, dtype=np.float64)
                err_to_vox_mat = torch.as_tensor(
                    (np.diag(1.0 / sp_rev_np) @ D_inv_np).T, device=device, dtype=dtype
                )
                y_norm = physical_to_normalized_fast(X_b + u_f, M_norm, b_norm)
                x_norm = physical_to_normalized_fast(X_b + u_i, M_norm, b_norm)
            else:
                grids = [torch.linspace(-1.0, 1.0, s, device=device, dtype=dtype) for s in spatial]
                mesh = torch.meshgrid(*grids, indexing="ij")
                X_b = torch.stack(list(reversed(mesh)), dim=-1).unsqueeze(0)
                err_to_vox_mat = torch.diag(
                    torch.tensor(
                        [float(max(s - 1, 1) / 2.0) for s in reversed(spatial)],
                        device=device,
                        dtype=dtype,
                    )
                )
                y_norm = X_b + u_f
                x_norm = X_b + u_i

            in_y = (((y_norm >= -1.0) & (y_norm <= 1.0)).all(dim=-1)).to(dtype) * dom
            in_x = (((x_norm >= -1.0) & (x_norm <= 1.0)).all(dim=-1)).to(dtype) * dom
            r_fi = (u_f + sample_field_cf(torch.movedim(u_i, -1, 1), y_norm)) @ err_to_vox_mat
            r_if = (u_i + sample_field_cf(torch.movedim(u_f, -1, 1), x_norm)) @ err_to_vox_mat
            # Smooth norm sqrt(||r||^2 + eps) for clean gradient at 0
            n_fi = torch.sqrt((r_fi * r_fi).sum(dim=-1) + 1e-12) - 1e-6
            n_if = torch.sqrt((r_if * r_if).sum(dim=-1) + 1e-12) - 1e-6
            v_fi = F.relu(n_fi - float(ice_tol)) * in_y
            v_if = F.relu(n_if - float(ice_tol)) * in_x
            loss = loss + float(ice_weight) * 0.5 * (_reduce_linf(v_fi) + _reduce_linf(v_if))

    return loss


def digital_diffeomorphism_penalty(
    warp_field: torch.Tensor,
    warp_inv: Optional[torch.Tensor] = None,
    *,
    spacing=None,
    origin=None,
    direction=None,
    jac_floor: float = 0.05,
    ice_tol: float = 0.0,
    linf_beta: Optional[float] = 25.0,
    mean_weight: float = 0.1,
    stationary_boundary: bool = True,
    is_physical: Optional[bool] = None,
) -> torch.Tensor:
    """Alias for ``digital_diffeomorphism_loss`` (smooth ``L_infinity`` digital diffeomorphism
    barrier following the Anderson worst-case inversion philosophy)."""
    return digital_diffeomorphism_loss(
        warp_field,
        warp_inv,
        spacing=spacing,
        origin=origin,
        direction=direction,
        jac_floor=jac_floor,
        ice_tol=ice_tol,
        linf_beta=linf_beta,
        mean_weight=mean_weight,
        stationary_boundary=stationary_boundary,
        is_physical=is_physical,
    )


def enforce_digital_diffeomorphism_nd(
    W_fwd: torch.Tensor,
    W_inv: Optional[torch.Tensor] = None,
    *,
    spacing=None,
    origin=None,
    direction=None,
    X_phys: Optional[torch.Tensor] = None,
    ice_tol: float = 0.30,
    jac_floor: float = 0.05,
    jac_weight: float = 200.0,
    anchor_weight: float = 0.05,
    steps: int = 25,
    step_size: float = 0.30,
    dilation_radius: int = 2,
    lm_sync_steps: int = 25,
    stationary_boundary: bool = True,
    stats: Optional[dict] = None,
):
    """Enforce the **digital diffeomorphism** constraint on discrete displacement field(s).

    - When ``W_inv`` is given (symmetric paired mode: ``syn``, ``tvf``, ``syngs``,
      ``syn_scattered``, or ``greedy(..., return_inverse=True)``): jointly polishes
      ``(W_fwd, W_inv)`` on the sparse 2-ring dilated active defect set so both fields satisfy
      ``det(I + D_{2h} u) >= jac_floor`` and bidirectional chord ICE ``<= ice_tol`` voxels,
      returning ``(W_fwd_polished, W_inv_polished)``.
    - When ``W_inv is None`` (unidirectional mode: ``greedy(..., return_inverse=False)``):
      unfolds ``W_fwd`` on the sparse active fold neighborhood under proximity anchor
      ``anchor_weight``, returning ``W_fwd_polished``.

    Idempotent: returns inputs bitwise untouched (``stats['steps'] == 0``) when all interior
    voxels already satisfy the digital diffeomorphism bounds.
    """
    unidirectional = W_inv is None
    channels_first = False
    if W_fwd.dim() >= 3 and W_fwd.shape[1] in (2, 3) and W_fwd.shape[-1] not in (2, 3):
        channels_first = True
        perm = (0,) + tuple(range(2, W_fwd.dim())) + (1,)
        W_fwd_in = W_fwd.permute(perm)
        W_inv_in = W_inv.permute(perm) if W_inv is not None else None
    else:
        W_fwd_in = W_fwd
        W_inv_in = W_inv

    B = W_fwd_in.shape[0]
    dim = W_fwd_in.shape[-1]
    spatial = tuple(W_fwd_in.shape[1:-1])
    device = W_fwd_in.device
    dtype = W_fwd_in.dtype

    if spacing is not None and origin is None:
        origin = (0.0,) * dim
    if spacing is not None and direction is None:
        direction = np.eye(dim)
    if X_phys is not None and (spacing is None or origin is None or direction is None):
        raise ValueError("X_phys requires spacing, origin, and direction")

    use_physical = X_phys is not None or (
        spacing is not None and origin is not None and direction is not None
    )
    boundary_mask = get_boundary_mask(spatial, device, dtype)
    domain_mask = (
        (boundary_mask[0, ..., 0] > 0.5)
        if stationary_boundary
        else torch.ones(spatial, dtype=torch.bool, device=device)
    )

    if use_physical:
        if X_phys is None:
            X_phys = get_physical_grid_torch(
                spatial, spacing, origin, direction, device=device, dtype=dtype
            )
        X_base = X_phys
        spacing_rev = [float(s) for s in reversed(tuple(spacing))]
        origin_rev = [float(o) for o in reversed(tuple(origin))]
        dir_arr = np.asarray(
            direction.detach().cpu().numpy() if isinstance(direction, torch.Tensor) else direction,
            dtype=np.float64,
        )
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
        A_to_vox = torch.tensor(D_inv_np.T / sp_rev_np[None, :], device=device, dtype=dtype)
        b_to_vox = torch.tensor(
            -np.asarray(origin_rev, dtype=np.float64) @ (D_inv_np.T / sp_rev_np[None, :]),
            device=device,
            dtype=dtype,
        )
        right_mat = torch.tensor(
            np.diag(1.0 / sp_rev_np) @ D_inv_np, device=device, dtype=dtype
        )
        D_rev_inv = torch.tensor(D_inv_np, device=device, dtype=dtype)
        err_to_vox_mat = right_mat.T
        vox_to_unit_mat = torch.tensor(
            np.linalg.inv((np.diag(1.0 / sp_rev_np) @ D_inv_np).T),
            device=device,
            dtype=dtype,
        )
        steps_fd = spacing_rev
    else:
        grids = [torch.linspace(-1.0, 1.0, s, device=device, dtype=dtype) for s in spatial]
        mesh = torch.meshgrid(*grids, indexing="ij")
        X_base = torch.stack(list(reversed(mesh)), dim=-1).unsqueeze(0)
        k_vec = np.asarray([max(s - 1, 1) / 2.0 for s in spatial], dtype=np.float64)
        p_flip = np.eye(dim, dtype=np.float64)[::-1].copy()
        A_to_vox = torch.tensor(p_flip @ np.diag(k_vec), device=device, dtype=dtype)
        b_to_vox = torch.tensor(k_vec.copy(), device=device, dtype=dtype)
        right_mat = torch.tensor((p_flip @ np.diag(k_vec)).T, device=device, dtype=dtype)
        D_rev_inv = torch.eye(dim, device=device, dtype=dtype)
        err_to_vox_scale = torch.tensor(
            [float(max(s - 1, 1) / 2.0) for s in reversed(spatial)], device=device, dtype=dtype
        )
        err_to_vox_mat = torch.diag(err_to_vox_scale)
        vox_to_unit_mat = torch.diag(1.0 / err_to_vox_scale)
        steps_fd = [2.0 / max(s - 1, 1) for s in spatial]
    metric_mat = err_to_vox_mat @ err_to_vox_mat.T

    def _fd_field_input(u_t: torch.Tensor) -> torch.Tensor:
        return u_t if use_physical else torch.flip(u_t, dims=[-1])

    def _eval_chord_errors_vox(u_f: torch.Tensor, u_i: torch.Tensor):
        u_f_cf = torch.movedim(u_f, -1, 1)
        u_i_cf = torch.movedim(u_i, -1, 1)
        if use_physical:
            y_norm = physical_to_normalized_fast(X_base + u_f, M_norm, b_norm)
            x_norm = physical_to_normalized_fast(X_base + u_i, M_norm, b_norm)
        else:
            y_norm = X_base + u_f
            x_norm = X_base + u_i
        in_y = ((y_norm >= -1.0) & (y_norm <= 1.0)).all(dim=-1) & domain_mask.unsqueeze(0)
        in_x = ((x_norm >= -1.0) & (x_norm <= 1.0)).all(dim=-1) & domain_mask.unsqueeze(0)
        r_fi = (u_f + sample_field_cf(u_i_cf, y_norm)) @ err_to_vox_mat
        r_if = (u_i + sample_field_cf(u_f_cf, x_norm)) @ err_to_vox_mat
        n_fi = torch.linalg.vector_norm(r_fi, dim=-1) * in_y.to(dtype)
        n_if = torch.linalg.vector_norm(r_if, dim=-1) * in_x.to(dtype)
        return r_fi, r_if, n_fi, n_if

    with torch.no_grad():
        dom_b = domain_mask.unsqueeze(0).expand(B, *spatial)
        full_b = torch.ones_like(dom_b)
        jac_margin = float(jac_floor) + 0.04
        det_f0 = _physical_fd_det_only(_fd_field_input(W_fwd_in), steps_fd, D_rev_inv)
        init_min_det_f = float(det_f0.amin().item())

        if unidirectional:
            defect_mask = det_f0 < float(jac_floor)
            n_defect = int(defect_mask.sum().item())
            if steps <= 0 or n_defect == 0:
                if stats is not None:
                    stats["steps"] = 0
                    stats["active_voxels"] = 0
                    stats["initial_min_det_fwd"] = init_min_det_f
                    stats["final_min_det_fwd"] = init_min_det_f
                    stats["initial_max_ice_vox"] = 0.0
                    stats["final_max_ice_vox"] = 0.0
                return W_fwd

            active_mask = _dilate_mask_nd(defect_mask, int(dilation_radius)) & dom_b
            n_active = int(active_mask.sum().item())
            if n_active == 0:
                if stats is not None:
                    stats["steps"] = 0
                    stats["active_voxels"] = 0
                    stats["initial_min_det_fwd"] = init_min_det_f
                    stats["final_min_det_fwd"] = init_min_det_f
                    stats["initial_max_ice_vox"] = 0.0
                    stats["final_max_ice_vox"] = 0.0
                return W_fwd

            u_f0 = W_fwd_in.clone() if stationary_boundary else W_fwd_in
            if stationary_boundary:
                u_f0.mul_(boundary_mask)
            eye_u = torch.eye(dim, device=device, dtype=dtype)
            mean_sp_u = (
                float(np.mean(spacing_rev))
                if use_physical
                else float(np.mean([2.0 / max(s - 1, 1) for s in spatial]))
            )
            lr_u = 0.08 * mean_sp_u * max(min(float(step_size) / 0.30, 2.0), 0.25)
            max_iters_u = max(int(steps) * 4, 95)
            steps_taken = 0
            u_f_cur = u_f0.clone()
            best_min_u = init_min_det_f
            max_outer_u = 2 if int(steps) >= 8 else 1

            for outer_u in range(max_outer_u):
                stencil_u = _build_sparse_fd_stencil(active_mask, steps_fd, dtype)
                ext_u_mask = ~stencil_u["sten_mask"]
                ext_min_u = (
                    float(det_f0[ext_u_mask].amin().item())
                    if bool(ext_u_mask.any().item())
                    else float("inf")
                )
                u_f_base_pool = u_f_cur.reshape(-1, dim)[stencil_u["pool_flat"]]

                with torch.enable_grad():
                    df_u = torch.zeros(
                        (stencil_u["act_coords"].shape[0], dim),
                        device=device,
                        dtype=dtype,
                        requires_grad=True,
                    )
                    opt_u = torch.optim.Adam([df_u], lr=max(lr_u, 1e-4))
                    sched_u = torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt_u, T_max=max_iters_u, eta_min=max(lr_u * 0.1, 1e-5)
                    )
                    best_df_u = df_u.detach().clone()
                    for it in range(max_iters_u):
                        opt_u.zero_grad()
                        uf_pool = u_f_base_pool.index_add(0, stencil_u["act_in_pool"], df_u)
                        det_s, _ = _eval_sparse_fd_det(
                            uf_pool, stencil_u, D_rev_inv, eye_u, use_physical
                        )
                        cur_min_u = min(
                            ext_min_u,
                            float(det_s.amin().item()) if det_s.numel() > 0 else ext_min_u,
                        )
                        if cur_min_u > best_min_u:
                            best_min_u = cur_min_u
                            best_df_u = df_u.detach().clone()
                        if cur_min_u >= float(jac_floor):
                            best_min_u = cur_min_u
                            best_df_u = df_u.detach().clone()
                            break
                        viol_u = F.relu(float(jac_margin) + 0.02 - det_s)
                        vmax_u = viol_u.amax() if viol_u.numel() > 0 else viol_u.new_zeros(())
                        lme_u = (
                            vmax_u
                            + torch.log(
                                torch.exp(30.0 * (viol_u - vmax_u)).mean().clamp_min(1e-20)
                            )
                            / 30.0
                            if viol_u.numel() > 0
                            else vmax_u
                        )
                        df_u_vox = df_u @ err_to_vox_mat
                        loss_u = (
                            25.0 * lme_u
                            + 0.5 * float(jac_weight) * (viol_u * viol_u).sum()
                            + 0.05 * float(anchor_weight) * (df_u_vox * df_u_vox).sum()
                        )
                        loss_u.backward()
                        opt_u.step()
                        sched_u.step()
                        steps_taken += 1

                u_f_cur.view(-1, dim).index_add_(0, stencil_u["act_flat"], best_df_u)
                if best_min_u >= float(jac_floor) or outer_u + 1 >= max_outer_u:
                    break
                det_rem_u, _ = _eval_sparse_fd_det(
                    u_f_cur.reshape(-1, dim)[stencil_u["pool_flat"]],
                    stencil_u,
                    D_rev_inv,
                    eye_u,
                    use_physical,
                )
                rem_u = det_rem_u < float(jac_floor)
                if not bool(rem_u.any().item()):
                    break
                rem_u_mask = torch.zeros_like(active_mask)
                rem_u_mask.view(-1)[stencil_u["sten_flat"][rem_u]] = True
                active_mask = (active_mask | _dilate_mask_nd(rem_u_mask, int(dilation_radius))) & dom_b
                n_active = int(active_mask.sum().item())

            u_f = torch.where(active_mask.unsqueeze(-1), u_f_cur, W_fwd_in)
            if stats is not None:
                stats["steps"] = steps_taken
                stats["active_voxels"] = n_active
                stats["initial_min_det_fwd"] = init_min_det_f
                stats["final_min_det_fwd"] = best_min_u
                stats["initial_max_ice_vox"] = 0.0
                stats["final_max_ice_vox"] = 0.0
            return torch.movedim(u_f, -1, 1) if channels_first else u_f

        # Symmetric paired mode (W_inv is not None)
        _, _, n_fi0, n_if0 = _eval_chord_errors_vox(W_fwd_in, W_inv_in)
        det_i0 = _physical_fd_det_only(_fd_field_input(W_inv_in), steps_fd, D_rev_inv)
        defect_mask = (
            (dom_b & ((n_fi0 > float(ice_tol)) | (n_if0 > float(ice_tol))))
            | (det_f0 < float(jac_floor))
            | (det_i0 < float(jac_floor))
        )
        init_max_ice = float(torch.maximum(n_fi0, n_if0).amax().item())
        init_min_det_i = float(det_i0.amin().item())
        n_defect = int(defect_mask.sum().item())

        if steps <= 0 or n_defect == 0:
            if stats is not None:
                stats["steps"] = 0
                stats["active_voxels"] = 0
                stats["initial_min_det_fwd"] = init_min_det_f
                stats["final_min_det_fwd"] = init_min_det_f
                stats["initial_max_ice_vox"] = init_max_ice
                stats["final_max_ice_vox"] = init_max_ice
            return W_fwd, W_inv

        def _expand_paired_active_mask(
            seed_defect: torch.Tensor, ref_fwd: torch.Tensor, ref_inv: torch.Tensor
        ) -> torch.Tensor:
            act_s = _dilate_mask_nd(seed_defect, int(dilation_radius)) & dom_b
            tgt_m = act_s.clone()
            for b_i in range(B):
                s_idx = torch.nonzero(seed_defect[b_i] & dom_b[b_i], as_tuple=False)
                if s_idx.shape[0] == 0:
                    continue
                idx_tup = tuple(s_idx[:, d] for d in range(dim))
                x_s = X_base[0 if X_base.shape[0] == 1 else b_i][idx_tup]
                for u_src in (ref_fwd[b_i][idx_tup], ref_inv[b_i][idx_tup]):
                    t_vox = torch.round((x_s + u_src) @ A_to_vox + b_to_vox).long()
                    for d_ax in range(dim):
                        t_vox[:, d_ax].clamp_(0, spatial[d_ax] - 1)
                    tgt_m[b_i][tuple(t_vox[:, d] for d in range(dim))] = True
            act_m = _dilate_mask_nd(tgt_m, int(dilation_radius)) & dom_b
            if stationary_boundary:
                act_m = act_m & (boundary_mask[..., 0] > 0.5)
            return act_m

        active_mask = _expand_paired_active_mask(defect_mask, W_fwd_in, W_inv_in)
        n_active = int(active_mask.sum().item())

        if n_active == 0:
            if stats is not None:
                stats["steps"] = 0
                stats["active_voxels"] = 0
                stats["initial_min_det_fwd"] = init_min_det_f
                stats["final_min_det_fwd"] = init_min_det_f
                stats["initial_max_ice_vox"] = init_max_ice
                stats["final_max_ice_vox"] = init_max_ice
            return W_fwd, W_inv

        u_f_cur = W_fwd_in.clone()
        u_i_cur = W_inv_in.clone()
        if stationary_boundary:
            u_f_cur.mul_(boundary_mask)
            u_i_cur.mul_(boundary_mask)

        eye = torch.eye(dim, device=device, dtype=dtype)
        linf_beta = 30.0
        mean_sp = (
            float(np.mean(spacing_rev))
            if use_physical
            else float(np.mean([2.0 / max(s - 1, 1) for s in spatial]))
        )
        adam_lr = 0.055 * mean_sp * max(min(float(step_size) / 0.30, 2.0), 0.25)
        max_outer_passes = 2 if int(steps) >= 8 else 1
        max_iters_p0 = max(int(steps) * 5, 130) if max_outer_passes > 1 else max(int(steps) * 4, 20)
        max_iters_p1 = max(int(steps) * 3, 75)
        steps_taken = 0
        n_full = max(int(np.prod(spatial)), 1)
        X_flat = X_base.expand(B, *spatial, dim).reshape(-1, dim)
        vox_hi_t = torch.tensor(
            [float(max(s - 1, 0)) for s in spatial], device=device, dtype=dtype
        )
        final_min_df = init_min_det_f
        final_min_di = init_min_det_i
        final_max_ice = init_max_ice

        def _lme_padded_1d(
            v_1d: torch.Tensor, b_masks: list[torch.Tensor], n_tot: int
        ) -> torch.Tensor:
            """Smooth ``L_infinity`` ``(1/beta)*logmeanexp(beta*v)`` padded with ``n_tot - K``
            zeros per batch element so gradients match the full-grid formulation bitwise."""
            if B == 1:
                if v_1d.numel() == 0:
                    return v_1d.new_zeros((1,))
                vmax = v_1d.amax()
                n_zero = max(n_tot - v_1d.numel(), 0)
                sum_exp = torch.exp(float(linf_beta) * (v_1d - vmax)).sum()
                if n_zero > 0:
                    sum_exp = sum_exp + float(n_zero) * torch.exp(-float(linf_beta) * vmax)
                lme = vmax + torch.log(sum_exp / float(max(n_tot, 1))) / float(linf_beta)
                return torch.where(vmax > 0.0, lme, torch.zeros_like(lme)).unsqueeze(0)

            outs = []
            for bm in b_masks:
                vb = v_1d[bm]
                if vb.numel() == 0:
                    outs.append(v_1d.new_zeros(()))
                    continue
                vmax = vb.amax()
                n_zero = max(n_tot - vb.numel(), 0)
                sum_exp = torch.exp(float(linf_beta) * (vb - vmax)).sum()
                if n_zero > 0:
                    sum_exp = sum_exp + float(n_zero) * torch.exp(-float(linf_beta) * vmax)
                lme = vmax + torch.log(sum_exp / float(max(n_tot, 1))) / float(linf_beta)
                outs.append(torch.where(vmax > 0.0, lme, torch.zeros_like(lme)))
            return torch.stack(outs, dim=0)

        for outer_pass in range(max_outer_passes):
            max_iters_per_pass = max_iters_p0 if outer_pass == 0 else max_iters_p1
            # Pure O(|active|) sparse stencil & sparse multilinear gather (zero bounding-box or
            # 5-D grid_sample ops inside the optimization loop):
            stencil = _build_sparse_fd_stencil(active_mask, steps_fd, dtype)
            act_coords = stencil["act_coords"]
            act_flat_full = stencil["act_flat"]
            sten_mask = stencil["sten_mask"]

            # Outside sten_mask, u_f_cur == W_fwd_in and u_i_cur == W_inv_in bitwise, so det_f0
            # and det_i0 remain exact exterior bounds across passes!
            ext_sten_mask = ~sten_mask
            if bool(ext_sten_mask.any().item()):
                ext_min_det_f = float(det_f0[ext_sten_mask].amin().item())
                ext_min_det_i = float(det_i0[ext_sten_mask].amin().item())
            else:
                ext_min_det_f = float("inf")
                ext_min_det_i = float("inf")

            # Estimate peak active displacement magnitude (in voxels) so aff_fi_mask / aff_if_mask
            # include every source voxel whose forward/inverse chord lands in active_mask even on
            # large 8–20 mm cortical deformations!
            u_act_max_vox = max(
                float((u_f_cur.reshape(-1, dim)[act_flat_full] @ err_to_vox_mat).abs().amax().item()),
                float((u_i_cur.reshape(-1, dim)[act_flat_full] @ err_to_vox_mat).abs().amax().item()),
            )
            guard_r = min(max(3, int(np.ceil(u_act_max_vox * 0.25)) + 2), 6)
            act_dil_guard = _dilate_mask_nd(active_mask, guard_r)
            y0_vox = (X_base + u_f_cur) @ A_to_vox + b_to_vox
            x0_vox = (X_base + u_i_cur) @ A_to_vox + b_to_vox
            y0_idx = torch.round(y0_vox).long()
            x0_idx = torch.round(x0_vox).long()
            for d_ax in range(dim):
                y0_idx[..., d_ax].clamp_(0, spatial[d_ax] - 1)
                x0_idx[..., d_ax].clamp_(0, spatial[d_ax] - 1)

            aff_fi_mask = active_mask.clone()
            aff_if_mask = active_mask.clone()
            for b in range(B):
                land_f = act_dil_guard[b][tuple(y0_idx[b, ..., d] for d in range(dim))]
                land_i = act_dil_guard[b][tuple(x0_idx[b, ..., d] for d in range(dim))]
                aff_fi_mask[b] = (aff_fi_mask[b] | land_f) & dom_b[b]
                aff_if_mask[b] = (aff_if_mask[b] | land_i) & dom_b[b]
            del y0_vox, x0_vox, y0_idx, x0_idx, act_dil_guard

            ext_fi_mask = dom_b & (~aff_fi_mask)
            ext_if_mask = dom_b & (~aff_if_mask)
            ext_max_ice_fi = (
                float(n_fi0[ext_fi_mask].amax().item()) if bool(ext_fi_mask.any().item()) else 0.0
            )
            ext_max_ice_if = (
                float(n_if0[ext_if_mask].amax().item()) if bool(ext_if_mask.any().item()) else 0.0
            )
            ext_max_ice = max(ext_max_ice_fi, ext_max_ice_if)

            aff_fi_coords = torch.nonzero(aff_fi_mask, as_tuple=False)
            aff_if_coords = torch.nonzero(aff_if_mask, as_tuple=False)
            fi_flat = _coords_to_flat_idx(aff_fi_coords, spatial)
            if_flat = _coords_to_flat_idx(aff_if_coords, spatial)
            fi_batch = aff_fi_coords[:, 0]
            if_batch = aff_if_coords[:, 0]

            X_fi = X_flat[fi_flat]
            X_if = X_flat[if_flat]
            u_f0_flat = u_f_cur.reshape(-1, dim)
            u_i0_flat = u_i_cur.reshape(-1, dim)
            u_f0_pool = u_f0_flat[stencil["pool_flat"]]
            u_i0_pool = u_i0_flat[stencil["pool_flat"]]
            u_f0_fi = u_f0_flat[fi_flat]
            u_i0_if = u_i0_flat[if_flat]

            lut_act = torch.full((B * n_full,), -1, dtype=torch.long, device=device)
            lut_act[act_flat_full] = torch.arange(act_coords.shape[0], device=device)

            lut_fi = torch.full((B * n_full,), -1, dtype=torch.long, device=device)
            lut_fi[fi_flat] = torch.arange(fi_flat.shape[0], device=device)
            act_in_fi = lut_fi[act_flat_full]
            valid_act_fi = act_in_fi >= 0

            lut_if = torch.full((B * n_full,), -1, dtype=torch.long, device=device)
            lut_if[if_flat] = torch.arange(if_flat.shape[0], device=device)
            act_in_if = lut_if[act_flat_full]
            valid_act_if = act_in_if >= 0
            del lut_fi, lut_if

            sten_batch = stencil["sten_batch"]
            b_sten_masks = [sten_batch == b for b in range(B)]
            b_fi_masks = [fi_batch == b for b in range(B)]
            b_if_masks = [if_batch == b for b in range(B)]

            # Anchor penalty always measures total shift from original (W_fwd_in, W_inv_in)
            orig_f_act = W_fwd_in.reshape(-1, dim)[act_flat_full]
            orig_i_act = W_inv_in.reshape(-1, dim)[act_flat_full]
            base_shift_f = u_f0_flat[act_flat_full] - orig_f_act
            base_shift_i = u_i0_flat[act_flat_full] - orig_i_act

            def _eval_sparse_pair_state(df_cur: torch.Tensor, di_cur: torch.Tensor):
                uf_p = u_f0_pool.index_add(0, stencil["act_in_pool"], df_cur)
                ui_p = u_i0_pool.index_add(0, stencil["act_in_pool"], di_cur)
                det_f_s, _ = _eval_sparse_fd_det(uf_p, stencil, D_rev_inv, eye, use_physical)
                det_i_s, _ = _eval_sparse_fd_det(ui_p, stencil, D_rev_inv, eye, use_physical)

                uf_fi_c = u_f0_fi.index_add(0, act_in_fi[valid_act_fi], df_cur[valid_act_fi])
                ui_if_c = u_i0_if.index_add(0, act_in_if[valid_act_if], di_cur[valid_act_if])
                y_vox_c = (X_fi + uf_fi_c) @ A_to_vox + b_to_vox
                x_vox_c = (X_if + ui_if_c) @ A_to_vox + b_to_vox
                iny_c = (((y_vox_c >= 0.0) & (y_vox_c <= vox_hi_t)).all(dim=-1)).to(dtype)
                inx_c = (((x_vox_c >= 0.0) & (x_vox_c <= vox_hi_t)).all(dim=-1)).to(dtype)

                ui_at_y_c = _gather_multilinear_sparse(
                    u_i0_flat, lut_act, di_cur, fi_batch, y_vox_c, spatial
                )
                uf_at_x_c = _gather_multilinear_sparse(
                    u_f0_flat, lut_act, df_cur, if_batch, x_vox_c, spatial
                )
                r_fi_c = (uf_fi_c + ui_at_y_c) @ err_to_vox_mat
                r_if_c = (ui_if_c + uf_at_x_c) @ err_to_vox_mat
                n_fi_c = (torch.sqrt((r_fi_c * r_fi_c).sum(dim=-1) + 1e-12) - 1e-6) * iny_c
                n_if_c = (torch.sqrt((r_if_c * r_if_c).sum(dim=-1) + 1e-12) - 1e-6) * inx_c
                return det_f_s, det_i_s, n_fi_c, n_if_c

            best_worst_viol = float("inf")
            with torch.enable_grad():
                df_act = torch.zeros(
                    (act_coords.shape[0], dim), device=device, dtype=dtype, requires_grad=True
                )
                di_act = torch.zeros(
                    (act_coords.shape[0], dim), device=device, dtype=dtype, requires_grad=True
                )
                opt = torch.optim.Adam([df_act, di_act], lr=max(adam_lr, 1e-4))
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max_iters_per_pass, eta_min=max(adam_lr * 0.1, 1e-5)
                )
                best_df_act = df_act.detach().clone()
                best_di_act = di_act.detach().clone()
                best_min_df = final_min_df
                best_min_di = final_min_di
                best_max_ice = final_max_ice

                for it in range(max_iters_per_pass):
                    opt.zero_grad()
                    det_f_s, det_i_s, n_fi_sub, n_if_sub = _eval_sparse_pair_state(df_act, di_act)

                    with torch.no_grad():
                        min_df = min(
                            ext_min_det_f,
                            float(det_f_s.amin().item()) if det_f_s.numel() > 0 else ext_min_det_f,
                        )
                        min_di = min(
                            ext_min_det_i,
                            float(det_i_s.amin().item()) if det_i_s.numel() > 0 else ext_min_det_i,
                        )
                        max_ice_sub = max(
                            float(n_fi_sub.amax().item()) if n_fi_sub.numel() > 0 else 0.0,
                            float(n_if_sub.amax().item()) if n_if_sub.numel() > 0 else 0.0,
                        )
                        max_ice = max(ext_max_ice, max_ice_sub)
                        # Strictly prioritize positive Jacobian determinants (>= jac_floor) over ICE!
                        worst_v = (
                            200.0 * max(0.0, float(jac_floor) - min_df)
                            + 200.0 * max(0.0, float(jac_floor) - min_di)
                            + max(0.0, max_ice - float(ice_tol))
                            + 0.02 * max_ice
                        )
                        if worst_v < best_worst_viol:
                            best_worst_viol = worst_v
                            best_df_act = df_act.detach().clone()
                            best_di_act = di_act.detach().clone()
                            best_min_df = min_df
                            best_min_di = min_di
                            best_max_ice = max_ice
                        if (
                            min_df >= float(jac_floor)
                            and min_di >= float(jac_floor)
                            and max_ice <= float(ice_tol)
                        ):
                            best_df_act = df_act.detach().clone()
                            best_di_act = di_act.detach().clone()
                            best_min_df = min_df
                            best_min_di = min_di
                            best_max_ice = max_ice
                            break

                    vf = F.relu(float(jac_margin) + 0.02 - det_f_s)
                    vi = F.relu(float(jac_margin) + 0.02 - det_i_s)
                    ice_margin = 0.05 * float(ice_tol)
                    v_fi = F.relu(n_fi_sub - ice_margin)
                    v_if = F.relu(n_if_sub - ice_margin)

                    n_st_norm = float(max(stencil["sten_flat"].numel(), 64))
                    n_fi_norm = float(max(fi_flat.numel(), 64))
                    n_if_norm = float(max(if_flat.numel(), 64))
                    e_jac = 25.0 * (
                        _lme_padded_1d(vf, b_sten_masks, n_full).sum()
                        + _lme_padded_1d(vi, b_sten_masks, n_full).sum()
                    ) + float(jac_weight) * ((vf * vf).sum() + (vi * vi).sum()) / n_st_norm
                    e_ice = 7.5 * (
                        _lme_padded_1d(v_fi, b_fi_masks, n_full).sum()
                        + _lme_padded_1d(v_if, b_if_masks, n_full).sum()
                    ) + 15.0 * ((v_fi * v_fi).sum() / n_fi_norm + (v_if * v_if).sum() / n_if_norm)

                    df_tot_vox = (base_shift_f + df_act) @ err_to_vox_mat
                    di_tot_vox = (base_shift_i + di_act) @ err_to_vox_mat
                    e_anc = float(anchor_weight) * 0.005 * (
                        (df_tot_vox * df_tot_vox).sum() + (di_tot_vox * di_tot_vox).sum()
                    )
                    loss = e_jac + e_ice + e_anc
                    loss.backward()
                    opt.step()
                    sched.step()
                    steps_taken += 1

            u_f_cur.view(-1, dim).index_add_(0, act_flat_full, best_df_act)
            u_i_cur.view(-1, dim).index_add_(0, act_flat_full, best_di_act)
            final_min_df = best_min_df
            final_min_di = best_min_di
            final_max_ice = best_max_ice

            if (
                final_min_df >= float(jac_floor)
                and final_min_di >= float(jac_floor)
                and final_max_ice <= float(ice_tol)
            ) or (outer_pass + 1 >= max_outer_passes):
                break

            # Border shell refresh: if unfolding a deep crease compressed the immediate exterior
            # ring of sten_mask below jac_floor (or left residual ICE > ice_tol), expand active_mask
            # strictly around those remaining sparse violations for pass 2!
            det_f_end, det_i_end, n_fi_end, n_if_end = _eval_sparse_pair_state(
                best_df_act, best_di_act
            )
            rem_sten = (det_f_end < float(jac_floor)) | (det_i_end < float(jac_floor))
            rem_fi = n_fi_end > float(ice_tol)
            rem_if = n_if_end > float(ice_tol)
            if not bool(rem_sten.any().item() or rem_fi.any().item() or rem_if.any().item()):
                break
            rem_seed = torch.zeros_like(active_mask)
            if bool(rem_sten.any().item()):
                rem_seed.view(-1)[stencil["sten_flat"][rem_sten]] = True
                active_mask = active_mask | (
                    _dilate_mask_nd(rem_seed, max(int(dilation_radius), 2)) & dom_b
                )
            elif bool(rem_fi.any().item() or rem_if.any().item()):
                if bool(rem_fi.any().item()):
                    rem_seed.view(-1)[fi_flat[rem_fi]] = True
                if bool(rem_if.any().item()):
                    rem_seed.view(-1)[if_flat[rem_if]] = True
                active_mask = active_mask | _expand_paired_active_mask(rem_seed, u_f_cur, u_i_cur)
            if stationary_boundary:
                active_mask = active_mask & (boundary_mask[..., 0] > 0.5)
            n_active = int(active_mask.sum().item())

        del det_f0, det_i0, n_fi0, n_if0
        u_f = u_f_cur
        u_i = u_i_cur

        # Optional sparse safeguarded bidirectional LM touch-up ONLY when residual ICE > ice_tol
        if int(lm_sync_steps) > 0 and final_max_ice > float(ice_tol):
            zero_act = u_f.new_zeros((act_coords.shape[0], dim))
            _, _, n_fi_cur, n_if_cur = _eval_sparse_pair_state(best_df_act, best_di_act)
            max_step_unit = (
                float(min(spacing_rev)) * 2.0
                if use_physical
                else (2.0 / float(max(max(spatial) - 1, 1))) * 2.0
            )
            # Rebuild zero-centered reference on final active_mask for clean LM line search
            u_f0_flat = u_f.reshape(-1, dim).clone()
            u_i0_flat = u_i.reshape(-1, dim).clone()
            u_f0_pool = u_f0_flat[stencil["pool_flat"]]
            u_i0_pool = u_i0_flat[stencil["pool_flat"]]
            u_f0_fi = u_f0_flat[fi_flat]
            u_i0_if = u_i0_flat[if_flat]
            cur_df = zero_act.clone()
            cur_di = zero_act.clone()

            # 1. Sparse LM touch-up on active inverse voxels with residual ICE > ice_tol
            high_if_sub = n_if_cur > float(ice_tol)
            if bool(high_if_sub.any().item()):
                high_i_mask = torch.zeros_like(active_mask)
                high_i_mask.view(-1)[if_flat[high_if_sub]] = True
                high_i_mask = high_i_mask & active_mask
                if bool(high_i_mask.any().item()):
                    u_i_lm = _pointwise_lm_refine_subset(
                        u_f,
                        u_i,
                        high_i_mask,
                        X_base,
                        A_to_vox,
                        b_to_vox,
                        right_mat,
                        err_to_vox_mat,
                        steps=int(lm_sync_steps),
                        tol_vox=float(ice_tol) * 0.5,
                        max_step_unit=max_step_unit,
                    )
                    delta_di_lm = u_i_lm.view(-1, dim)[act_flat_full] - u_i0_flat[act_flat_full] - cur_di
                    for blend in (1.0, 0.75, 0.5, 0.35, 0.2):
                        di_try = cur_di + blend * delta_di_lm
                        _, det_i_try, n_fi_t, n_if_t = _eval_sparse_pair_state(cur_df, di_try)
                        min_di_t = min(
                            ext_min_det_i,
                            float(det_i_try.amin().item()) if det_i_try.numel() > 0 else ext_min_det_i,
                        )
                        ice_t = max(
                            ext_max_ice,
                            float(n_fi_t.amax().item()) if n_fi_t.numel() > 0 else 0.0,
                            float(n_if_t.amax().item()) if n_if_t.numel() > 0 else 0.0,
                        )
                        if min_di_t >= float(jac_floor) and ice_t < final_max_ice:
                            cur_di = di_try
                            final_max_ice = ice_t
                            n_fi_cur, n_if_cur = n_fi_t, n_if_t
                            u_i = u_i0_flat.reshape(u_i.shape).clone()
                            u_i.view(-1, dim).index_add_(0, act_flat_full, cur_di)
                            break

            # 2. Sparse LM touch-up on active forward voxels if forward ICE still > ice_tol
            if final_max_ice > float(ice_tol):
                high_fi_sub = n_fi_cur > float(ice_tol)
                if bool(high_fi_sub.any().item()):
                    high_f_mask = torch.zeros_like(active_mask)
                    high_f_mask.view(-1)[fi_flat[high_fi_sub]] = True
                    high_f_mask = high_f_mask & active_mask
                    if bool(high_f_mask.any().item()):
                        u_f_lm = _pointwise_lm_refine_subset(
                            u_i,
                            u_f,
                            high_f_mask,
                            X_base,
                            A_to_vox,
                            b_to_vox,
                            right_mat,
                            err_to_vox_mat,
                            steps=int(lm_sync_steps),
                            tol_vox=float(ice_tol) * 0.5,
                            max_step_unit=max_step_unit,
                        )
                        delta_df_lm = u_f_lm.view(-1, dim)[act_flat_full] - u_f0_flat[act_flat_full] - cur_df
                        for blend in (1.0, 0.75, 0.5, 0.35, 0.2):
                            df_try = cur_df + blend * delta_df_lm
                            det_f_try, _, n_fi_t, n_if_t = _eval_sparse_pair_state(df_try, cur_di)
                            min_df_t = min(
                                ext_min_det_f,
                                float(det_f_try.amin().item()) if det_f_try.numel() > 0 else ext_min_det_f,
                            )
                            ice_t = max(
                                ext_max_ice,
                                float(n_fi_t.amax().item()) if n_fi_t.numel() > 0 else 0.0,
                                float(n_if_t.amax().item()) if n_if_t.numel() > 0 else 0.0,
                            )
                            if min_df_t >= float(jac_floor) and ice_t < final_max_ice:
                                cur_df = df_try
                                final_min_df = min_df_t
                                final_max_ice = ice_t
                                u_f = u_f0_flat.reshape(u_f.shape).clone()
                                u_f.view(-1, dim).index_add_(0, act_flat_full, cur_df)
                                break

        # Strict bitwise preservation of exterior (~active_mask) voxels against -0.0 float addition
        u_f = torch.where(active_mask.unsqueeze(-1), u_f, W_fwd_in)
        u_i = torch.where(active_mask.unsqueeze(-1), u_i, W_inv_in)

        if stats is not None:
            stats["steps"] = steps_taken
            stats["active_voxels"] = n_active
            stats["initial_min_det_fwd"] = init_min_det_f
            stats["final_min_det_fwd"] = final_min_df
            stats["initial_max_ice_vox"] = init_max_ice
            stats["final_max_ice_vox"] = final_max_ice

    if channels_first:
        return torch.movedim(u_f, -1, 1), torch.movedim(u_i, -1, 1)
    return u_f, u_i


def pullback_digital_diffeomorphism_to_halves(
    delta_fwd: torch.Tensor,
    w_l2r: torch.Tensor,
    w_r2l: torch.Tensor,
    w_l2r_inv: torch.Tensor,
    w_r2l_inv: Optional[torch.Tensor] = None,
    *,
    spacing=None,
    origin=None,
    direction=None,
    X_phys: Optional[torch.Tensor] = None,
    jac_floor: float = 0.05,
    inv_tol: float = 0.05,
    inv_steps: int = 20,
    stationary_boundary: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Propagate a small endpoint correction ``delta_fwd = W_fwd_polished - W_fwd_0``
    symmetrically back onto midpoint half-fields ``(w_l2r, w_r2l, w_l2r_inv, w_r2l_inv)``.

    Operates strictly in ``O(|support(delta_fwd)|)`` sparse point-wise time: only midpoint
    voxels whose forward/inverse trajectories touch ``support(delta_fwd)`` are resampled,
    inverted via damped Levenberg-Marquardt, and checked on their 1-ring central-difference
    Jacobian stencil, leaving all exterior half-warp voxels bitwise untouched.
    """
    if not bool((delta_fwd != 0.0).any().item()):
        return w_l2r, w_r2l, w_l2r_inv, w_r2l_inv

    with torch.no_grad():
        B = w_l2r.shape[0]
        dim = w_l2r.shape[-1]
        spatial = tuple(w_l2r.shape[1:-1])
        device = w_l2r.device
        dtype = w_l2r.dtype
        n_full = max(int(np.prod(spatial)), 1)

        if spacing is not None and origin is None:
            origin = (0.0,) * dim
        if spacing is not None and direction is None:
            direction = np.eye(dim)
        use_physical = X_phys is not None or (
            spacing is not None and origin is not None and direction is not None
        )

        boundary_mask = get_boundary_mask(spatial, device, dtype)
        domain_mask = (
            (boundary_mask[0, ..., 0] > 0.5)
            if stationary_boundary
            else torch.ones(spatial, dtype=torch.bool, device=device)
        )
        dom_b = domain_mask.unsqueeze(0).expand(B, *spatial)

        nz_mask = (delta_fwd.abs().amax(dim=-1) > 0.0) & dom_b
        if not bool(nz_mask.any().item()):
            return w_l2r, w_r2l, w_l2r_inv, w_r2l_inv

        if use_physical:
            if X_phys is None:
                X_phys = get_physical_grid_torch(
                    spatial, spacing, origin, direction, device=device, dtype=dtype
                )
            X_base = X_phys
            spacing_rev = [float(s) for s in reversed(tuple(spacing))]
            origin_rev = [float(o) for o in reversed(tuple(origin))]
            dir_arr = np.asarray(
                direction.detach().cpu().numpy() if isinstance(direction, torch.Tensor) else direction,
                dtype=np.float64,
            )
            if dir_arr.ndim == 1:
                dir_arr = dir_arr.reshape(dim, dim)
            direction_rev = dir_arr[::-1, ::-1].copy()
            D_inv_np = np.linalg.inv(direction_rev)
            sp_rev_np = np.asarray(spacing_rev, dtype=np.float64)

            A_to_vox = torch.tensor(D_inv_np.T / sp_rev_np[None, :], device=device, dtype=dtype)
            b_to_vox = torch.tensor(
                -np.asarray(origin_rev, dtype=np.float64) @ (D_inv_np.T / sp_rev_np[None, :]),
                device=device,
                dtype=dtype,
            )
            right_mat = torch.tensor(
                np.diag(1.0 / sp_rev_np) @ D_inv_np, device=device, dtype=dtype
            )
            err_to_vox_mat = right_mat.T
            D_rev_inv = torch.tensor(D_inv_np, device=device, dtype=dtype)
            steps_fd = spacing_rev
            max_step_unit = float(min(spacing_rev)) * 2.0
        else:
            grids = [torch.linspace(-1.0, 1.0, s, device=device, dtype=dtype) for s in spatial]
            mesh = torch.meshgrid(*grids, indexing="ij")
            X_base = torch.stack(list(reversed(mesh)), dim=-1).unsqueeze(0)
            k_vec = np.asarray([max(s - 1, 1) / 2.0 for s in spatial], dtype=np.float64)
            p_flip = np.eye(dim, dtype=np.float64)[::-1].copy()
            A_to_vox = torch.tensor(p_flip @ np.diag(k_vec), device=device, dtype=dtype)
            b_to_vox = torch.tensor(k_vec.copy(), device=device, dtype=dtype)
            right_mat = torch.tensor((p_flip @ np.diag(k_vec)).T, device=device, dtype=dtype)
            err_to_vox_mat = torch.diag(
                torch.tensor(
                    [float(max(s - 1, 1) / 2.0) for s in reversed(spatial)],
                    device=device,
                    dtype=dtype,
                )
            )
            D_rev_inv = torch.eye(dim, device=device, dtype=dtype)
            steps_fd = [2.0 / max(s - 1, 1) for s in spatial]
            max_step_unit = (2.0 / float(max(max(spatial) - 1, 1))) * 2.0

        I_d = torch.eye(dim, device=device, dtype=dtype).unsqueeze(0)
        eye = torch.eye(dim, device=device, dtype=dtype)

        # 1. Sparse half-pullback update on w_l2r_inv strictly on nz_mask
        dw_l2r_inv_act_list = []
        nz_coords = torch.nonzero(nz_mask, as_tuple=False)
        nz_flat = _coords_to_flat_idx(nz_coords, spatial)
        for b in range(B):
            act_idx = torch.nonzero(nz_mask[b], as_tuple=False)
            if act_idx.shape[0] == 0:
                continue
            idx_t = tuple(act_idx[:, d] for d in range(dim))
            x_a = X_base[0 if X_base.shape[0] == 1 else b][idx_t]
            m_vox = (x_a + w_l2r_inv[b][idx_t]) @ A_to_vox + b_to_vox
            _, grad_r2l = _multilinear_val_and_jac_torch(w_r2l[b], m_vox, right_mat)
            J = I_d + grad_r2l
            JT = J.transpose(-1, -2)
            JTJ = JT @ J
            rhs = 0.5 * (JT @ delta_fwd[b][idx_t].unsqueeze(-1)).squeeze(-1)
            lam = torch.full((act_idx.shape[0],), 0.1, device=device, dtype=dtype)
            dw_inv = _solve_spd_lm_torch(JTJ, rhs, lam)
            dw_l2r_inv_act_list.append(dw_inv)
        dw_l2r_inv_act = (
            torch.cat(dw_l2r_inv_act_list, dim=0)
            if dw_l2r_inv_act_list
            else w_l2r_inv.new_zeros((0, dim))
        )

        # 2. Identify midpoint voxels m where x(m) = X_base(m) + w_l2r(m) lands within 2 voxels
        # of nz_mask: outside this sparse set, BOTH delta_fwd(x(m)) == 0 AND cand_l2r_inv(x(m))
        # == w_l2r_inv(x(m)) are unchanged!
        nz_guard = _dilate_mask_nd(nz_mask, 2)
        x_from_m_vox = (X_base + w_l2r) @ A_to_vox + b_to_vox
        x_from_m_idx = torch.round(x_from_m_vox).long()
        for d_ax in range(dim):
            x_from_m_idx[..., d_ax].clamp_(0, spatial[d_ax] - 1)

        m_mask = torch.zeros_like(nz_mask)
        for b in range(B):
            land_m = nz_guard[b][tuple(x_from_m_idx[b, ..., d] for d in range(dim))]
            m_mask[b] = land_m & dom_b[b]
        del x_from_m_idx, nz_guard

        # Dilate m_mask by 1 voxel so cand_l2r LM refinement covers any trajectory shifts
        l2r_refine_mask = _dilate_mask_nd(m_mask, 1) & dom_b
        r2l_mod_mask = m_mask

        # Sparse gather of delta_fwd(X(m) + w_l2r(m)) only at m_mask!
        m_coords = torch.nonzero(m_mask, as_tuple=False)
        m_flat = _coords_to_flat_idx(m_coords, spatial)
        m_batch = m_coords[:, 0]
        x_m_vox_sub = x_from_m_vox.reshape(-1, dim)[m_flat]
        del x_from_m_vox

        empty_lut = torch.empty((0,), dtype=torch.long, device=device)
        empty_delta = delta_fwd.new_zeros((0, dim))
        delta_at_m_sub = _gather_multilinear_sparse(
            delta_fwd.reshape(-1, dim), empty_lut, empty_delta, m_batch, x_m_vox_sub, spatial
        )

        # Build sparse FD stencils ONLY around the modified midpoint neighborhoods!
        has_l2r_mod = bool(l2r_refine_mask.any().item())
        has_r2l_mod = bool(r2l_mod_mask.any().item())
        w_l2r_flat = w_l2r.reshape(-1, dim)
        w_r2l_flat = w_r2l.reshape(-1, dim)

        if has_l2r_mod:
            sten_l2r = _build_sparse_fd_stencil(l2r_refine_mask, steps_fd, dtype)
            det_l2r_0s, _ = _eval_sparse_fd_det(
                w_l2r_flat[sten_l2r["pool_flat"]], sten_l2r, D_rev_inv, eye, use_physical
            )
            sten_dom_l2r = dom_b.reshape(-1)[sten_l2r["sten_flat"]]
            floor_l2r = min(
                float(jac_floor),
                float(det_l2r_0s[sten_dom_l2r].amin().item())
                if bool(sten_dom_l2r.any().item())
                else float(jac_floor),
            )
        else:
            sten_l2r = None
            sten_dom_l2r = None
            floor_l2r = float(jac_floor)

        if has_r2l_mod:
            sten_r2l = _build_sparse_fd_stencil(r2l_mod_mask, steps_fd, dtype)
            det_r2l_0s, _ = _eval_sparse_fd_det(
                w_r2l_flat[sten_r2l["pool_flat"]], sten_r2l, D_rev_inv, eye, use_physical
            )
            sten_dom_r2l = dom_b.reshape(-1)[sten_r2l["sten_flat"]]
            floor_r2l = min(
                float(jac_floor),
                float(det_r2l_0s[sten_dom_r2l].amin().item())
                if bool(sten_dom_r2l.any().item())
                else float(jac_floor),
            )
        else:
            sten_r2l = None
            sten_dom_r2l = None
            floor_r2l = float(jac_floor)

        best_l2r = w_l2r
        best_r2l = w_r2l
        best_l2r_inv = w_l2r_inv
        best_r2l_inv = w_r2l_inv

        for scale in (1.0, 0.5, 0.25):
            cand_l2r_inv = w_l2r_inv.clone()
            if nz_flat.numel() > 0:
                cand_l2r_inv.view(-1, dim).index_add_(0, nz_flat, scale * dw_l2r_inv_act)

            cand_r2l = w_r2l.clone()
            if m_flat.numel() > 0:
                cand_r2l.view(-1, dim).index_add_(0, m_flat, (0.5 * scale) * delta_at_m_sub)

            cand_l2r = _pointwise_lm_refine_subset(
                cand_l2r_inv,
                w_l2r,
                l2r_refine_mask,
                X_base,
                A_to_vox,
                b_to_vox,
                right_mat,
                err_to_vox_mat,
                steps=int(inv_steps),
                tol_vox=float(inv_tol),
                max_step_unit=max_step_unit,
            )

            ok_l2r = True
            if has_l2r_mod and bool(sten_dom_l2r.any().item()):
                det_l2r_c, _ = _eval_sparse_fd_det(
                    cand_l2r.reshape(-1, dim)[sten_l2r["pool_flat"]],
                    sten_l2r,
                    D_rev_inv,
                    eye,
                    use_physical,
                )
                ok_l2r = float(det_l2r_c[sten_dom_l2r].amin().item()) >= floor_l2r - 1e-4

            ok_r2l = True
            if has_r2l_mod and bool(sten_dom_r2l.any().item()):
                det_r2l_c, _ = _eval_sparse_fd_det(
                    cand_r2l.reshape(-1, dim)[sten_r2l["pool_flat"]],
                    sten_r2l,
                    D_rev_inv,
                    eye,
                    use_physical,
                )
                ok_r2l = float(det_r2l_c[sten_dom_r2l].amin().item()) >= floor_r2l - 1e-4

            if ok_l2r and ok_r2l:
                best_l2r = cand_l2r
                best_r2l = cand_r2l
                best_l2r_inv = cand_l2r_inv
                if w_r2l_inv is not None and has_r2l_mod:
                    r2l_guard = _dilate_mask_nd(r2l_mod_mask, 2)
                    m_from_y_vox = (X_base + w_r2l_inv) @ A_to_vox + b_to_vox
                    m_from_y_idx = torch.round(m_from_y_vox).long()
                    for d_ax in range(dim):
                        m_from_y_idx[..., d_ax].clamp_(0, spatial[d_ax] - 1)
                    r2l_inv_mask = torch.zeros_like(nz_mask)
                    for b in range(B):
                        land_y = r2l_guard[b][tuple(m_from_y_idx[b, ..., d] for d in range(dim))]
                        r2l_inv_mask[b] = land_y & dom_b[b]
                    best_r2l_inv = _pointwise_lm_refine_subset(
                        cand_r2l,
                        w_r2l_inv,
                        r2l_inv_mask,
                        X_base,
                        A_to_vox,
                        b_to_vox,
                        right_mat,
                        err_to_vox_mat,
                        steps=int(inv_steps),
                        tol_vox=float(inv_tol),
                        max_step_unit=max_step_unit,
                    )
                break

        return best_l2r, best_r2l, best_l2r_inv, best_r2l_inv


def enforce_digital_diffeomorphism_jax(
    W_fwd_jax,
    W_inv_jax=None,
    *,
    spacing=None,
    origin=None,
    direction=None,
    ice_tol: float = 0.30,
    jac_floor: float = 0.05,
    jac_weight: float = 200.0,
    anchor_weight: float = 0.05,
    steps: int = 25,
    step_size: float = 0.30,
    dilation_radius: int = 2,
    lm_sync_steps: int = 25,
    stationary_boundary: bool = True,
    stats: Optional[dict] = None,
):
    """Backend-parity JAX wrapper for ``enforce_digital_diffeomorphism_nd``."""
    import jax.numpy as jnp

    fwd_t = torch.from_numpy(np.asarray(W_fwd_jax).copy())
    inv_t = torch.from_numpy(np.asarray(W_inv_jax).copy()) if W_inv_jax is not None else None
    local_stats: dict = {}
    res = enforce_digital_diffeomorphism_nd(
        fwd_t,
        inv_t,
        spacing=spacing,
        origin=origin,
        direction=direction,
        ice_tol=ice_tol,
        jac_floor=jac_floor,
        jac_weight=jac_weight,
        anchor_weight=anchor_weight,
        steps=steps,
        step_size=step_size,
        dilation_radius=dilation_radius,
        lm_sync_steps=lm_sync_steps,
        stationary_boundary=stationary_boundary,
        stats=local_stats,
    )
    if stats is not None:
        stats.update(local_stats)
    if local_stats.get("steps", 0) == 0:
        return W_fwd_jax if W_inv_jax is None else (W_fwd_jax, W_inv_jax)
    if W_inv_jax is None:
        return jnp.asarray(res.numpy())
    fwd_out, inv_out = res
    return jnp.asarray(fwd_out.numpy()), jnp.asarray(inv_out.numpy())


# Semantic aliases for post-composition polish callers
polish_composed_pair_nd = enforce_digital_diffeomorphism_nd
pullback_composed_delta_to_halves = pullback_digital_diffeomorphism_to_halves


__all__ = [
    "digital_diffeomorphism_loss",
    "digital_diffeomorphism_penalty",
    "enforce_digital_diffeomorphism_nd",
    "enforce_digital_diffeomorphism_jax",
    "pullback_digital_diffeomorphism_to_halves",
    "polish_composed_pair_nd",
    "pullback_composed_delta_to_halves",
]
