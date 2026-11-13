"""Fast CPU-only unit tests for the default-on (opt-out) ``digital_diffeomorphism`` finish projection."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from syntx.core.digital_diffeomorphism import (
    _physical_fd_det_and_grad,
    digital_diffeomorphism_loss,
    digital_diffeomorphism_penalty,
    enforce_digital_diffeomorphism_jax,
    enforce_digital_diffeomorphism_nd,
    pullback_digital_diffeomorphism_to_halves,
)
from syntx.core.grid import get_physical_grid_torch, physical_to_normalized_torch, sample_field_cf
from syntx.core.inverse import (
    calculate_inverse_identity_error,
    update_inverse_field_nd_hybrid_lm,
)
from syntx.core.jacobian import (
    compute_jacobian_determinant_nd,
    compute_physical_jacobian_determinant,
)
from syntx.core.smoothing import get_boundary_mask


def _oblique_3d_direction() -> np.ndarray:
    """Right-handed oblique 3-D direction cosine matrix (RPI + 12-deg rotation)."""
    theta = np.deg2rad(12.0)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    rpi = np.diag([-1.0, 1.0, -1.0])
    return rpi @ rot


def test_physical_fd_det_and_grad_matches_autograd_oblique_3d():
    """Exact discrete cofactor adjoint matches torch.autograd on anisotropic oblique 3-D grids."""
    torch.manual_seed(0)
    spatial = (8, 9, 10)
    spacing = (1.2, 0.9, 1.5)
    direction = _oblique_3d_direction()
    steps_rev = [float(s) for s in reversed(spacing)]
    D_rev_inv = torch.tensor(
        np.linalg.inv(direction[::-1, ::-1].copy()), dtype=torch.float64
    )

    u = (torch.randn((1, *spatial, 3), dtype=torch.float64) * 0.65).requires_grad_(True)
    jac_floor = 0.25

    det_analytic, energy_analytic, grad_analytic = _physical_fd_det_and_grad(
        u.detach(), steps_rev, D_rev_inv, jac_floor
    )

    u.is_physical = True
    det_ref = compute_physical_jacobian_determinant(u, direction=direction, spacing=spacing)
    assert torch.allclose(det_analytic, det_ref.detach(), atol=1e-12, rtol=1e-12)

    viol_ref = torch.relu(jac_floor - det_ref)
    energy_ref = 0.5 * (viol_ref * viol_ref).sum()
    energy_ref.backward()

    assert torch.allclose(energy_analytic.sum(), energy_ref.detach(), atol=1e-12)
    assert torch.allclose(grad_analytic, u.grad, atol=1e-11, rtol=1e-10)

    # L_infinity digital_diffeomorphism_loss: exact amax when linf_beta=None, smooth logmeanexp
    # by default, and strictly 0.0 on clean fields!
    bmask = get_boundary_mask(spatial, torch.device("cpu"), torch.float64)
    dom = bmask[0, ..., 0] > 0.5
    loss_exact_linf = digital_diffeomorphism_loss(
        u.detach(),
        spacing=spacing,
        direction=direction,
        jac_floor=jac_floor,
        linf_beta=None,
        mean_weight=0.0,
    )
    assert torch.allclose(
        loss_exact_linf, (viol_ref.detach()[:, dom]).amax(), atol=1e-12
    )
    loss_smooth_linf = digital_diffeomorphism_loss(
        u.detach(), spacing=spacing, direction=direction, jac_floor=jac_floor
    )
    assert float(loss_smooth_linf.item()) > 0.0
    assert torch.allclose(
        loss_smooth_linf,
        digital_diffeomorphism_penalty(
            u.detach(), spacing=spacing, direction=direction, jac_floor=jac_floor
        ),
        atol=1e-12,
    )
    u_zero = torch.zeros_like(u.detach())
    assert (
        float(
            digital_diffeomorphism_loss(
                u_zero,
                u_zero,
                spacing=spacing,
                origin=(0.0, 0.0, 0.0),
                direction=direction,
                jac_floor=0.05,
                ice_tol=0.10,
            ).item()
        )
        == 0.0
    )


def test_digital_diffeomorphism_idempotent_when_clean():
    """Clean diffeomorphic low-ICE composed fields return bitwise identical with 0 steps."""
    torch.manual_seed(1)
    spatial = (14, 16, 18)
    spacing = (1.1, 1.0, 1.3)
    origin = (10.0, -20.0, 5.0)
    direction = _oblique_3d_direction()

    bmask = get_boundary_mask(spatial, torch.device("cpu"), torch.float32)
    z = torch.linspace(-1.0, 1.0, spatial[0]).view(1, -1, 1, 1, 1)
    y = torch.linspace(-1.0, 1.0, spatial[1]).view(1, 1, -1, 1, 1)
    x = torch.linspace(-1.0, 1.0, spatial[2]).view(1, 1, 1, -1, 1)
    env = torch.exp(-2.5 * (z * z + y * y + x * x))
    w_fwd = torch.cat([0.25 * env, -0.20 * env, 0.15 * env], dim=-1) * bmask
    w_inv = update_inverse_field_nd_hybrid_lm(
        w_fwd,
        steps=25,
        max_error_threshold=0.01,
        mean_error_threshold=0.001,
        spacing=spacing,
        origin=origin,
        direction=direction,
    )

    fwd_orig = w_fwd.clone()
    inv_orig = w_inv.clone()
    stats: dict = {}
    fwd_out, inv_out = enforce_digital_diffeomorphism_nd(
        w_fwd,
        w_inv,
        spacing=spacing,
        origin=origin,
        direction=direction,
        ice_tol=0.10,
        jac_floor=0.05,
        stats=stats,
    )

    assert stats["steps"] == 0
    assert stats["active_voxels"] == 0
    assert torch.equal(fwd_out, fwd_orig)
    assert torch.equal(inv_out, inv_orig)

    # Also verify unidirectional idempotency
    stats_uni: dict = {}
    fwd_uni = enforce_digital_diffeomorphism_nd(
        w_fwd,
        None,
        spacing=spacing,
        origin=origin,
        direction=direction,
        jac_floor=0.05,
        stats=stats_uni,
    )
    assert stats_uni["steps"] == 0
    assert torch.equal(fwd_uni, fwd_orig)


def test_repairs_composed_2mm_secant_fold_and_preserves_exterior_and_halves():
    """Repairs a localized 2-mm FD secant fold created by discrete composition of two
    strictly diffeomorphic half-fields (localized opposing physical swirl where each half has
    2-mm FD det > 0.25 everywhere, yet their composed ~115-deg cortical-turn swirl has negative
    2-mm central-difference secant determinant), preserves exterior voxels bitwise, prevents
    collapse, and pulls the correction back onto the midpoint half-fields."""
    spatial = (22, 22, 22)
    spacing = (1.2, 1.0, 1.1)
    origin = (-5.0, 8.0, 12.0)
    direction = _oblique_3d_direction()
    device = torch.device("cpu")
    dtype = torch.float32

    bmask = get_boundary_mask(spatial, device, dtype)
    X_phys = get_physical_grid_torch(
        spatial, spacing, origin, direction, device=device, dtype=dtype
    )

    iz = torch.arange(spatial[0], dtype=dtype).view(1, -1, 1, 1) - 11.0
    iy = torch.arange(spatial[1], dtype=dtype).view(1, 1, -1, 1) - 11.0
    ix = torch.arange(spatial[2], dtype=dtype).view(1, 1, 1, -1) - 11.0
    r2 = iz * iz + iy * iy + ix * ix

    # Global smooth background displacement (must NOT collapse toward 0!)
    bg_env = torch.exp(-r2 / 75.0).unsqueeze(-1)
    bg_disp = torch.tensor([0.5, -0.4, 0.6], dtype=dtype).view(1, 1, 1, 1, 3) * bg_env

    # Physical coordinates relative to grid center in the oblique physical frame
    center_phys = X_phys[:, 11:12, 11:12, 11:12, :]
    rel_phys = X_phys - center_phys
    dir_rev = direction[::-1, ::-1].copy()
    e1 = torch.tensor(dir_rev[:, 1], dtype=dtype).view(1, 1, 1, 1, 3)
    e2 = torch.tensor(dir_rev[:, 2], dtype=dtype).view(1, 1, 1, 1, 3)
    u1 = (rel_phys * e1).sum(dim=-1)
    u2 = (rel_phys * e2).sum(dim=-1)

    # Two pure shears in orthogonal physical directions (e1, e2):
    #   Half 1: shear along e2 as a function of iy -> continuous & 2-mm FD det ~ 1.0 everywhere!
    #   Half 2: shear along e1 as a function of ix -> continuous & 2-mm FD det ~ 1.0 everywhere!
    # Note sign: phi_1^-1 reverses Half 1's shear (+a -> -a), so to make the off-diagonal
    # product in det(D_{2h} Phi_12) = 1 + ... - (dPhi_1/dy)(dPhi_2/dx) NEGATIVE, Half 1 and
    # Half 2 must have the SAME sign (+a and +b) so (-a) * (+b) ... wait:
    # Let phi_1^-1(x, y) = (x - a f(y), y) and phi_2(m1, m2) = (m1, m2 + b f(m1)):
    #   Phi_12(x, y) = (x - a f(y), y + b f(x - a f(y)))!
    #   At (0, 0): dPhi_1/dx = 1, dPhi_1/dy = -a f(1),
    #              dPhi_2/dx = +b f(1), dPhi_2/dy = 1 - b f(a f(1))!
    #    Wait: det = 1 * (1 - b f(a f(1))) - (-a f(1))*(+b f(1)) = 1 - b f(a f(1)) + a b f(1)^2!
    #   Aha! Flip the sign of b -> -b so phi_2(m1, m2) = (m1, m2 - b f(m1)) AND flip a -> -a
    #   so phi_1(m1, m2) = (m1 - a f(m2), m2) => phi_1^-1(x, y) = (x + a f(y), y)!
    #   Let's check phi_1^-1(x, y) = (x + a f(y), y) with phi_2(m1, m2) = (m1, m2 + b f(m1)):
    #   Phi_12(x, y) = (x + a f(y), y + b f(x + a f(y)))!
    #   At (0, 0):
    #     dPhi_1/dx = 1,            dPhi_1/dy = +a f(1)
    #     dPhi_2/dx = +b f(1),      dPhi_2/dy = 1 + b f(a f(1))
    #     det = 1 + b f(a f(1)) - a b f(1)^2!
    #   With f(s) = s * exp(-s^2 / 1.1): f(1) = 0.403, and for a = 4.5, a f(1) = 1.81 =>
    #   f(1.81) = 0.091, so for b = 4.5:
    #   det = 1 + 4.5 * 0.091 - 20.25 * 0.162 = 1.41 - 3.29 = -1.88 < 0!
    env_z = torch.exp(-(iz * iz) / 4.0)
    f_iy = iy * torch.exp(-(iy * iy) / 0.8) * env_z * torch.exp(-(ix * ix) / 14.0)
    f_ix = ix * torch.exp(-(ix * ix) / 0.8) * env_z * torch.exp(-(iy * iy) / 14.0)
    w_l2r_loc = (-8.5 * float(spacing[0]) * f_iy).unsqueeze(-1) * e2
    w_r2l_loc = (+8.5 * float(spacing[1]) * f_ix).unsqueeze(-1) * e1

    w_l2r = (-0.5 * bg_disp + w_l2r_loc) * bmask
    w_r2l = (0.5 * bg_disp + w_r2l_loc) * bmask

    w_l2r_chk = w_l2r.clone()
    w_l2r_chk.is_physical = True
    w_r2l_chk = w_r2l.clone()
    w_r2l_chk.is_physical = True
    det_half_1 = compute_physical_jacobian_determinant(
        w_l2r_chk, direction=direction, spacing=spacing
    )
    det_half_2 = compute_physical_jacobian_determinant(
        w_r2l_chk, direction=direction, spacing=spacing
    )
    assert float(det_half_1.min().item()) > 0.25, f"half 1 folded: {det_half_1.min().item()}"
    assert float(det_half_2.min().item()) > 0.25, f"half 2 folded: {det_half_2.min().item()}"

    w_l2r_inv = update_inverse_field_nd_hybrid_lm(
        w_l2r, steps=30, max_error_threshold=0.01, spacing=spacing, origin=origin, direction=direction
    )
    w_r2l_inv = update_inverse_field_nd_hybrid_lm(
        w_r2l, steps=30, max_error_threshold=0.01, spacing=spacing, origin=origin, direction=direction
    )

    c_fwd = physical_to_normalized_torch(
        X_phys + w_l2r_inv, spatial, spacing, origin, direction
    )
    fwd_composed = (w_l2r_inv + sample_field_cf(torch.movedim(w_r2l, -1, 1), c_fwd)) * bmask

    c_inv = physical_to_normalized_torch(
        X_phys + w_r2l_inv, spatial, spacing, origin, direction
    )
    inv_composed = (w_r2l_inv + sample_field_cf(torch.movedim(w_l2r, -1, 1), c_inv)) * bmask

    # Asymmetric hybrid_lm alone leaves fwd_composed bitwise untouched with negative 2-mm FD det
    inv_asym = update_inverse_field_nd_hybrid_lm(
        fwd_composed,
        inv_composed,
        steps=30,
        max_error_threshold=0.15,
        spacing=spacing,
        origin=origin,
        direction=direction,
    )
    fwd_chk0 = fwd_composed.clone()
    fwd_chk0.is_physical = True
    det_before = compute_physical_jacobian_determinant(
        fwd_chk0, direction=direction, spacing=spacing
    )
    assert float(det_before.min().item()) < 0.0, (
        f"Expected negative 2-mm FD Jacobian before polish, got {det_before.min().item()}"
    )

    stats: dict = {}
    fwd_pol, inv_pol = enforce_digital_diffeomorphism_nd(
        fwd_composed,
        inv_asym,
        spacing=spacing,
        origin=origin,
        direction=direction,
        X_phys=X_phys,
        ice_tol=0.15,
        jac_floor=0.05,
        steps=35,
        stats=stats,
    )

    assert stats["steps"] > 0
    assert stats["active_voxels"] < int(np.prod(spatial)) // 2

    fwd_chk1 = fwd_pol.clone()
    fwd_chk1.is_physical = True
    inv_chk1 = inv_pol.clone()
    inv_chk1.is_physical = True
    det_after_fwd = compute_physical_jacobian_determinant(
        fwd_chk1, direction=direction, spacing=spacing
    )
    det_after_inv = compute_physical_jacobian_determinant(
        inv_chk1, direction=direction, spacing=spacing
    )

    # 1. Zero 2-mm FD folds after digital_diffeomorphism enforcement
    assert float(det_after_fwd.min().item()) >= 0.045
    assert float(det_after_inv.min().item()) >= 0.045

    # 2. Bidirectional ICE meets target
    ice_after = calculate_inverse_identity_error(fwd_pol, inv_pol, spacing, origin, direction)
    ice_rev = calculate_inverse_identity_error(inv_pol, fwd_pol, spacing, origin, direction)
    assert ice_after["max_error"] <= 0.25 * max(spacing)
    assert ice_rev["max_error"] <= 0.25 * max(spacing)

    # 3. Exterior voxels (far from center defect) are BITWISE untouched & non-collapsed
    far_mask = (r2[0] > 105.0) & (bmask[0, ..., 0] > 0.5)
    assert int(far_mask.sum().item()) > 400
    assert torch.equal(fwd_pol[0][far_mask], fwd_composed[0][far_mask])
    assert torch.equal(inv_pol[0][far_mask], inv_asym[0][far_mask])
    assert float(fwd_pol[0][far_mask].abs().mean().item()) > 0.05

    # 4. Pullback onto midpoint half-fields keeps both half-fields diffeomorphic and reduces
    # re-composed fold violation
    delta_fwd = fwd_pol - fwd_composed
    new_l2r, new_r2l, new_l2r_inv, new_r2l_inv = pullback_digital_diffeomorphism_to_halves(
        delta_fwd,
        w_l2r,
        w_r2l,
        w_l2r_inv,
        w_r2l_inv,
        spacing=spacing,
        origin=origin,
        direction=direction,
        X_phys=X_phys,
        jac_floor=0.05,
    )
    nl2r_chk = new_l2r.clone()
    nl2r_chk.is_physical = True
    nr2l_chk = new_r2l.clone()
    nr2l_chk.is_physical = True
    assert (
        float(
            compute_physical_jacobian_determinant(
                nl2r_chk, direction=direction, spacing=spacing
            ).min().item()
        )
        > 0.04
    )
    assert (
        float(
            compute_physical_jacobian_determinant(
                nr2l_chk, direction=direction, spacing=spacing
            ).min().item()
        )
        > 0.04
    )

    c_re = physical_to_normalized_torch(
        X_phys + new_l2r_inv, spatial, spacing, origin, direction
    )
    recomposed = (new_l2r_inv + sample_field_cf(torch.movedim(new_r2l, -1, 1), c_re)) * bmask
    re_chk = recomposed.clone()
    re_chk.is_physical = True
    det_re = compute_physical_jacobian_determinant(re_chk, direction=direction, spacing=spacing)
    assert float(det_re.min().item()) > float(det_before.min().item()) + 0.05


def test_unidirectional_mode_unfolds_without_inverse():
    """Unidirectional mode (W_inv=None, used by syntx.greedy) unfolds a 2-mm FD crease
    while preserving exterior voxels bitwise."""
    spatial = (16, 16, 16)
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    direction = np.eye(3)
    dtype = torch.float32

    bmask = get_boundary_mask(spatial, torch.device("cpu"), dtype)
    iz = torch.arange(spatial[0], dtype=dtype).view(1, -1, 1, 1) - 8.0
    iy = torch.arange(spatial[1], dtype=dtype).view(1, 1, -1, 1) - 8.0
    ix = torch.arange(spatial[2], dtype=dtype).view(1, 1, 1, -1) - 8.0
    r2 = iz * iz + iy * iy + ix * ix

    env = torch.exp(-r2 / 3.0)
    w_fwd = torch.zeros((1, *spatial, 3), dtype=dtype)
    w_fwd[..., 2] = -2.2 * ix.squeeze(0) * env.squeeze(0)
    w_fwd = w_fwd * bmask

    w_chk0 = w_fwd.clone()
    w_chk0.is_physical = True
    assert (
        float(
            compute_physical_jacobian_determinant(
                w_chk0, direction=direction, spacing=spacing
            ).min().item()
        )
        < -0.2
    )

    stats: dict = {}
    w_pol = enforce_digital_diffeomorphism_nd(
        w_fwd,
        None,
        spacing=spacing,
        origin=origin,
        direction=direction,
        jac_floor=0.05,
        steps=25,
        stats=stats,
    )
    assert stats["steps"] > 0
    w_chk1 = w_pol.clone()
    w_chk1.is_physical = True
    assert (
        float(
            compute_physical_jacobian_determinant(
                w_chk1, direction=direction, spacing=spacing
            ).min().item()
        )
        >= 0.045
    )

    far = (r2[0] > 36.0) & (bmask[0, ..., 0] > 0.5)
    assert torch.equal(w_pol[0][far], w_fwd[0][far])


def test_sparse_scattered_two_hemisphere_folds_preserve_interior_gap_bitwise():
    """Verify pure O(|active|) sparse stencil polish when two folds sit at opposite corners
    of a 3-D volume: both folds are repaired, ICE meets target, AND voxels in the clean slab
    strictly between the two defects (inside their joint bounding box!) stay bitwise untouched."""
    spatial = (24, 24, 24)
    spacing = (1.1, 1.0, 1.2)
    origin = (0.0, 0.0, 0.0)
    direction = _oblique_3d_direction()
    dtype = torch.float32

    bmask = get_boundary_mask(spatial, torch.device("cpu"), dtype)
    iz = torch.arange(spatial[0], dtype=dtype).view(1, -1, 1, 1)
    iy = torch.arange(spatial[1], dtype=dtype).view(1, 1, -1, 1)
    ix = torch.arange(spatial[2], dtype=dtype).view(1, 1, 1, -1)

    r2_a = (iz - 5.0) ** 2 + (iy - 5.0) ** 2 + (ix - 5.0) ** 2
    r2_b = (iz - 18.0) ** 2 + (iy - 18.0) ** 2 + (ix - 18.0) ** 2

    w_fwd = torch.zeros((1, *spatial, 3), dtype=dtype)
    # Smooth background displacement in the middle (11..12) inside the joint bounding box [3..20]^3
    r2_mid = (iz - 11.5) ** 2 + (iy - 11.5) ** 2 + (ix - 11.5) ** 2
    w_fwd[..., 0] = 0.20 * torch.exp(-r2_mid / 8.0).squeeze(0)
    # Two isolated creases near (5,5,5) and (18,18,18)
    dir_rev = direction[::-1, ::-1].copy()
    e2 = torch.tensor(dir_rev[:, 2], dtype=dtype).view(1, 1, 1, 1, 3)
    crease_a = (-2.6 * (ix - 5.0) * torch.exp(-r2_a / 1.5)).unsqueeze(-1) * e2
    crease_b = (-2.6 * (ix - 18.0) * torch.exp(-r2_b / 1.5)).unsqueeze(-1) * e2
    w_fwd = (w_fwd + crease_a + crease_b) * bmask
    w_inv = -w_fwd.clone()

    w_chk0 = w_fwd.clone()
    w_chk0.is_physical = True
    assert float(compute_physical_jacobian_determinant(w_chk0, direction=direction, spacing=spacing).min().item()) < -0.2

    stats: dict = {}
    fwd_p, inv_p = enforce_digital_diffeomorphism_nd(
        w_fwd,
        w_inv,
        spacing=spacing,
        origin=origin,
        direction=direction,
        ice_tol=0.20,
        jac_floor=0.05,
        steps=30,
        stats=stats,
    )
    assert stats["steps"] > 0
    # Active set is tiny (< 20% of volume) despite defects spanning corners (5,5,5) .. (18,18,18)
    assert stats["active_voxels"] < int(np.prod(spatial)) // 5

    fwd_chk = fwd_p.clone()
    fwd_chk.is_physical = True
    inv_chk = inv_p.clone()
    inv_chk.is_physical = True
    assert float(compute_physical_jacobian_determinant(fwd_chk, direction=direction, spacing=spacing).min().item()) >= 0.045
    assert float(compute_physical_jacobian_determinant(inv_chk, direction=direction, spacing=spacing).min().item()) >= 0.045

    ice_f = calculate_inverse_identity_error(fwd_p, inv_p, spacing, origin, direction)
    assert ice_f["max_error"] <= 0.25 * max(spacing)

    # Clean slab right between the two corners (inside their bounding box!) is bitwise untouched
    mid_slab = (iz[0] >= 10) & (iz[0] <= 13) & (iy[0] >= 10) & (iy[0] <= 13) & (ix[0] >= 10) & (ix[0] <= 13)
    assert torch.equal(fwd_p[0][mid_slab], w_fwd[0][mid_slab])
    assert torch.equal(inv_p[0][mid_slab], w_inv[0][mid_slab])


def test_normalized_grid_mode_and_jax_parity():
    """Normalized [-1, 1] mode (spacing=None, used by syntx.syn_scattered) and JAX parity."""
    spatial = (20, 20)
    dtype = torch.float32
    bmask = get_boundary_mask(spatial, torch.device("cpu"), dtype)
    gy = torch.linspace(-1.0, 1.0, spatial[0]).view(1, -1, 1)
    gx = torch.linspace(-1.0, 1.0, spatial[1]).view(1, 1, -1)
    r2 = gx * gx + gy * gy
    env = torch.exp(-r2 / 0.03)
    u_fwd = torch.stack([-2.4 * gx * env, torch.zeros_like(env)], dim=-1) * bmask
    u_inv = -u_fwd.clone()

    det0 = compute_jacobian_determinant_nd(u_fwd, is_physical=False)
    assert float(det0.min().item()) < 0.0

    u_fwd_p, u_inv_p = enforce_digital_diffeomorphism_nd(
        u_fwd, u_inv, ice_tol=0.25, jac_floor=0.05, steps=25
    )
    det1 = compute_jacobian_determinant_nd(u_fwd_p, is_physical=False)
    assert float(det1.min().item()) >= 0.045

    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    j_fwd, j_inv = enforce_digital_diffeomorphism_jax(
        jnp.asarray(u_fwd.numpy()),
        jnp.asarray(u_inv.numpy()),
        ice_tol=0.25,
        jac_floor=0.05,
        steps=25,
    )
    assert np.allclose(np.asarray(j_fwd), u_fwd_p.numpy(), atol=1e-6)
    assert np.allclose(np.asarray(j_inv), u_inv_p.numpy(), atol=1e-6)


def _tiny_oblique_2d_pair():
    import ants

    theta = np.deg2rad(9.0)
    c, s = np.cos(theta), np.sin(theta)
    direction = np.array([[c, -s], [s, c]], dtype=np.float64)
    spacing = (1.1, 0.9)
    origin = (3.0, -4.0)

    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, 16), np.linspace(-1.0, 1.0, 16), indexing="ij")
    fi_arr = np.exp(-3.0 * ((xx - 0.1) ** 2 + (yy + 0.05) ** 2)).astype(np.float32)
    mi_arr = np.exp(-3.0 * ((xx + 0.1) ** 2 + (yy - 0.05) ** 2)).astype(np.float32)
    fi = ants.from_numpy(fi_arr, origin=origin, spacing=spacing, direction=direction)
    mi = ants.from_numpy(mi_arr, origin=origin, spacing=spacing, direction=direction)
    return fi, mi


def test_top_level_exports_and_default_on_with_opt_out_across_all_models():
    """Verify top-level syntx exports, default-ON bitwise parity with explicit True, and opt-out
    ``digital_diffeomorphism=False`` across syntx.syn (PyTorch & JAX), syntx.tvf, syntx.syngs,
    syntx.greedy, and syntx.syn_scattered."""
    import syntx
    from syntx import ScatteredRegistrationConfig

    for sym in (
        "digital_diffeomorphism_loss",
        "digital_diffeomorphism_penalty",
        "enforce_digital_diffeomorphism_nd",
        "enforce_digital_diffeomorphism_jax",
        "pullback_digital_diffeomorphism_to_halves",
    ):
        assert hasattr(syntx, sym), f"Missing top-level syntx.{sym}"
        assert hasattr(syntx.core, sym), f"Missing syntx.core.{sym}"

    fi, mi = _tiny_oblique_2d_pair()

    # 1. syntx.syn: default (omitted) vs digital_diffeomorphism=True are bitwise identical,
    # and explicit opt-out digital_diffeomorphism=False runs cleanly.
    reg_default = syntx.syn(
        fi,
        mi,
        initial_transform="identity",
        reg_iterations=[1, 1],
        levels=[2, 1],
        device="cpu",
        seed=7,
    )
    reg_on = syntx.syn(
        fi,
        mi,
        initial_transform="identity",
        reg_iterations=[1, 1],
        levels=[2, 1],
        device="cpu",
        seed=7,
        digital_diffeomorphism=True,
    )
    assert np.array_equal(reg_default["warpedmovout"].numpy(), reg_on["warpedmovout"].numpy())
    assert torch.equal(reg_default["model"].warp_l2r.data, reg_on["model"].warp_l2r.data)
    assert reg_default["inverse_identity_errors"]["phi_1"]["max_error"] < 0.5

    reg_off = syntx.syn(
        fi,
        mi,
        initial_transform="identity",
        reg_iterations=[1, 1],
        levels=[2, 1],
        device="cpu",
        seed=7,
        digital_diffeomorphism=False,
    )
    assert reg_off["warpedmovout"] is not None

    pytest.importorskip("jax")
    reg_jax = syntx.syn(
        fi,
        mi,
        initial_transform="identity",
        reg_iterations=[1, 1],
        levels=[2, 1],
        backend="jax",
        digital_diffeomorphism_steps=10,
    )
    assert reg_jax["warpedmovout"] is not None

    # 2. syntx.tvf (PyTorch default ON + opt-out False)
    reg_tvf = syntx.tvf(
        fi,
        mi,
        initial_transform="identity",
        reg_iterations=[1, 1],
        levels=[2, 1],
        n_time_steps=2,
        device="cpu",
        digital_diffeomorphism_steps=10,
    )
    assert reg_tvf["warpedmovout"] is not None
    assert reg_tvf["inverse_identity_errors"]["phi_1"]["max_error"] < 0.5

    reg_tvf_off = syntx.tvf(
        fi,
        mi,
        initial_transform="identity",
        reg_iterations=[1, 1],
        levels=[2, 1],
        n_time_steps=2,
        device="cpu",
        digital_diffeomorphism=False,
    )
    assert reg_tvf_off["warpedmovout"] is not None

    # 3. syntx.syngs (PyTorch default ON + opt-out False)
    reg_syngs = syntx.syngs(
        fi,
        mi,
        initial_transform="identity",
        reg_iterations=[1, 1],
        levels=[2, 1],
        n_steps=2,
        device="cpu",
        digital_diffeomorphism_steps=10,
    )
    assert reg_syngs["warpedmovout"] is not None
    assert reg_syngs["inverse_identity_errors"]["max"] < 0.5

    reg_syngs_off = syntx.syngs(
        fi,
        mi,
        initial_transform="identity",
        reg_iterations=[1, 1],
        levels=[2, 1],
        n_steps=2,
        device="cpu",
        digital_diffeomorphism=False,
    )
    assert reg_syngs_off["warpedmovout"] is not None

    # 4. syntx.greedy (default ON unidirectional + return_inverse=True paired + opt-out False)
    reg_greedy_uni = syntx.greedy(
        fi,
        mi,
        initial_transform="identity",
        reg_iterations=[1, 1],
        scales=[2, 1],
        return_inverse=False,
        device="cpu",
        digital_diffeomorphism_steps=10,
    )
    assert reg_greedy_uni["provenance"]["digital_diffeomorphism"] is True
    assert reg_greedy_uni["warpedmovout"] is not None

    reg_greedy_off = syntx.greedy(
        fi,
        mi,
        initial_transform="identity",
        reg_iterations=[1, 1],
        scales=[2, 1],
        return_inverse=False,
        device="cpu",
        digital_diffeomorphism=False,
    )
    assert reg_greedy_off["provenance"]["digital_diffeomorphism"] is False

    reg_greedy_pair = syntx.greedy(
        fi,
        mi,
        initial_transform="identity",
        reg_iterations=[1, 1],
        scales=[2, 1],
        return_inverse=True,
        device="cpu",
        digital_diffeomorphism_steps=10,
    )
    assert reg_greedy_pair["warpedfixout"] is not None

    # 5. syntx.syn_scattered (normalized [-1, 1] grid default ON + opt-out False)
    assert ScatteredRegistrationConfig().digital_diffeomorphism is True
    cfg_sc = ScatteredRegistrationConfig(
        dim=2,
        grid_res=16,
        iterations=[1, 1],
        levels=[2, 1],
        initial_transform="identity",
        digital_diffeomorphism_steps=10,
        device="cpu",
    )
    res_sc = syntx.syn_scattered(
        fixed_grid=torch.from_numpy(fi.numpy()[None, None]),
        moving_grid=torch.from_numpy(mi.numpy()[None, None]),
        config=cfg_sc,
    )
    assert res_sc.jacobian_min > 0.04

