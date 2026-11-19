"""Physical-space geometry enforcement and O(3) direction-matrix equivariance tests."""

import math
import numpy as np
import pytest
import torch

from syntx.core.regularizers import (
    apply_beltrami_regularizer,
    apply_div_curl_green_operator,
    apply_hyperelastic_regularizer,
    apply_masked_incompressible_filter,
    apply_navier_green_operator,
    apply_poroelastic_filter,
    apply_solenoidal_sobolev_operator,
    compute_curl_nd,
    compute_divergence_nd,
    get_regularizer,
    project_solenoidal,
)
from syntx.pyramid import build_pyramid, plan_levels


CONTINUUM_REG_NAMES = [
    "sobolev",
    "dsti",
    "dsti1",
    "dirichlet",
    "solenoidal",
    "leray",
    "leray_pure",
    "incompressible",
    "solenoidal_sobolev",
    "div_curl",
    "helmholtz",
    "navier",
    "stokes",
    "elastic",
    "masked_incompressible",
    "beltrami",
    "quasiconformal",
    "conformal",
    "poroelastic",
    "darcy_stokes",
    "biot",
    "hyperelastic",
    "simo_pister",
    "log_jacobian",
]

VOXEL_REG_NAMES = [
    "gaussian",
    "gauss",
    "compact_gaussian",
    "compact",
    "fast_gaussian",
    "erf",
    "bspline",
    "none",
    "identity",
]


@pytest.mark.parametrize("reg_name", CONTINUUM_REG_NAMES)
def test_get_regularizer_requires_valid_spacing_for_continuum_and_spectral(reg_name):
    with pytest.raises(ValueError, match="spacing"):
        get_regularizer(reg_name, alpha=1.0, fluid_sigma=2.0)
    with pytest.raises(ValueError, match="spacing"):
        get_regularizer(reg_name, alpha=1.0, fluid_sigma=2.0, spacing=(1.0, -1.0, 1.0))
    with pytest.raises(ValueError, match="spacing"):
        get_regularizer(reg_name, alpha=1.0, fluid_sigma=2.0, spacing=(1.0, float("nan"), 1.0))

    # Wrong dimensionality vs field raises ValueError
    reg_fn = get_regularizer(reg_name, alpha=1.0, fluid_sigma=2.0, spacing=(1.0, 1.0))
    field_3d = torch.zeros(1, 8, 8, 8, 3)
    with pytest.raises(ValueError, match="spacing"):
        reg_fn(field_3d)


@pytest.mark.parametrize("reg_name", VOXEL_REG_NAMES)
def test_voxel_unit_regularizers_do_not_require_spacing(reg_name):
    reg_fn = get_regularizer(reg_name, fluid_sigma=1.0)
    field = torch.randn(1, 10, 10, 2)
    out = reg_fn(field)
    assert out.shape == field.shape


def test_differential_and_direct_operators_require_spacing():
    v3 = torch.randn(1, 8, 8, 8, 3)
    for fn in (
        lambda x: compute_divergence_nd(x),
        lambda x: compute_divergence_nd(x, method="spectral"),
        lambda x: compute_curl_nd(x),
        lambda x: project_solenoidal(x),
        lambda x: apply_solenoidal_sobolev_operator(x, alpha=1.0),
        lambda x: apply_div_curl_green_operator(x, alpha=1.0),
        lambda x: apply_navier_green_operator(x, alpha=1.0),
        lambda x: apply_masked_incompressible_filter(x, mask=torch.ones(1, 8, 8, 8)),
        lambda x: apply_beltrami_regularizer(x, alpha=1.0),
        lambda x: apply_poroelastic_filter(x, alpha=1.0),
        lambda x: apply_hyperelastic_regularizer(x, alpha=1.0),
    ):
        with pytest.raises(ValueError, match="spacing"):
            fn(v3)
        with pytest.raises(ValueError, match="spacing"):
            compute_divergence_nd(v3, spacing=(1.0, 0.0, 1.0))
        with pytest.raises(ValueError, match="spacing"):
            compute_curl_nd(v3, spacing=(1.0, 1.0))


def test_pyramid_requires_spacing():
    x = torch.zeros(1, 1, 16, 16, 16)
    with pytest.raises(ValueError, match="spacing"):
        plan_levels((16, 16, 16), None, [2, 1], "deformable")
    with pytest.raises(ValueError, match="spacing"):
        build_pyramid(x, None, [2, 1], "deformable")


def test_raw_tensor_fit_requires_geometry_all_pytorch_models():
    from syntx.greedy import GreedyRegistrationModel
    from syntx.syn import SyNTo
    from syntx.tvf import TVFModel
    from syntx.syngs import GeodesicShootingModel

    f_t = torch.zeros(1, 1, 16, 16, 16)
    m_t = torch.zeros(1, 1, 16, 16, 16)
    sp = (1.0, 1.0, 1.0)
    orig = (0.0, 0.0, 0.0)
    eye = np.eye(3)

    # 1. GreedyRegistrationModel
    g = GreedyRegistrationModel(dim=3)
    theta = torch.eye(3, 4)[None]
    with pytest.raises(ValueError, match="spacing"):
        g.fit(f_t, m_t, theta, scales=[1], iterations=[1])
    with pytest.raises(ValueError, match="origin"):
        g.fit(f_t, m_t, theta, scales=[1], iterations=[1], spacing=sp, direction=eye)
    with pytest.raises(ValueError, match="direction"):
        g.fit(f_t, m_t, theta, scales=[1], iterations=[1], spacing=sp, origin=orig)

    # 2. SyNTo
    s = SyNTo(dim=3, grid_shape=(16, 16, 16))
    with pytest.raises(ValueError, match="fixed_spacing"):
        s.fit(f_t, m_t, levels=[1], epochs_per_level=[1])
    with pytest.raises(ValueError, match="fixed_direction"):
        s.fit(f_t, m_t, levels=[1], epochs_per_level=[1], fixed_spacing=sp, fixed_origin=orig)
    with pytest.raises(ValueError, match="moving_spacing"):
        s.fit(f_t, m_t, levels=[1], epochs_per_level=[1], fixed_spacing=sp, fixed_origin=orig, fixed_direction=eye)

    # 3. TVFModel
    tvf_m = TVFModel(dim=3, image_shape=(16, 16, 16), velocity_shape=(16, 16, 16))
    with pytest.raises(ValueError, match="fixed_spacing"):
        tvf_m.fit(f_t, m_t, levels=[1], epochs_per_level=[1])
    with pytest.raises(ValueError, match="moving_direction"):
        tvf_m.fit(
            f_t, m_t, levels=[1], epochs_per_level=[1],
            fixed_spacing=sp, fixed_origin=orig, fixed_direction=eye,
            moving_spacing=sp, moving_origin=orig,
        )

    # 4. GeodesicShootingModel
    gs_m = GeodesicShootingModel(dim=3, image_shape=(16, 16, 16))
    with pytest.raises(ValueError, match="fixed_spacing"):
        gs_m.fit(f_t, m_t, levels=[1], epochs_per_level=[1])
    with pytest.raises(ValueError, match="moving_origin"):
        gs_m.fit(
            f_t, m_t, levels=[1], epochs_per_level=[1],
            fixed_spacing=sp, fixed_origin=orig, fixed_direction=eye,
            moving_spacing=sp, moving_direction=eye,
        )


def test_raw_tensor_fit_requires_geometry_all_jax_models():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from syntx.syn_jax import SyNJAX
    from syntx.tvf_jax import TVFModelJAX
    from syntx.syngs_jax import GeodesicShootingModelJAX

    f_j = jnp.zeros((1, 1, 16, 16, 16), dtype=jnp.float32)
    m_j = jnp.zeros((1, 1, 16, 16, 16), dtype=jnp.float32)
    sp = (1.0, 1.0, 1.0)
    orig = (0.0, 0.0, 0.0)
    eye = np.eye(3)

    # 1. SyNJAX
    sj = SyNJAX(dim=3, grid_shape=(16, 16, 16))
    with pytest.raises(ValueError, match="fixed_spacing"):
        sj.fit(f_j, m_j, levels=[1], epochs_per_level=[1], affine_epochs=0)
    with pytest.raises(ValueError, match="moving_direction"):
        sj.fit(
            f_j, m_j, levels=[1], epochs_per_level=[1], affine_epochs=0,
            fixed_spacing=sp, fixed_origin=orig, fixed_direction=eye,
            moving_spacing=sp, moving_origin=orig,
        )

    # 2. TVFModelJAX
    tj = TVFModelJAX(dim=3, image_shape=(16, 16, 16), velocity_shape=(16, 16, 16))
    with pytest.raises(ValueError, match="fixed_spacing"):
        tj.fit(f_j, m_j, levels=[1], epochs_per_level=[1], affine_epochs=0)
    with pytest.raises(ValueError, match="moving_spacing"):
        tj.fit(
            f_j, m_j, levels=[1], epochs_per_level=[1], affine_epochs=0,
            fixed_spacing=sp, fixed_origin=orig, fixed_direction=eye,
        )

    # 3. GeodesicShootingModelJAX
    gj = GeodesicShootingModelJAX(dim=3, image_shape=(16, 16, 16), velocity_shape=(16, 16, 16))
    with pytest.raises(ValueError, match="fixed_spacing"):
        gj.fit(f_j, m_j, levels=[1], epochs_per_level=[1], affine_epochs=0)
    with pytest.raises(ValueError, match="moving_origin"):
        gj.fit(
            f_j, m_j, levels=[1], epochs_per_level=[1], affine_epochs=0,
            fixed_spacing=sp, fixed_origin=orig, fixed_direction=eye,
            moving_spacing=sp, moving_direction=eye,
        )


def _rodrigues_rotation(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    u = np.asarray(axis, dtype=np.float64)
    u = u / np.linalg.norm(u)
    th = math.radians(angle_deg)
    K = np.array([
        [0.0, -u[2], u[1]],
        [u[2], 0.0, -u[0]],
        [-u[1], u[0], 0.0],
    ], dtype=np.float64)
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


@pytest.mark.parametrize("reg_name", ["solenoidal", "div_curl", "navier", "hyperelastic"])
def test_continuum_o3_direction_matrix_equivariance(reg_name):
    """Verify exact O(3) equivariance (< 1e-5 error) across LPS, RPI, PIR, and 25-deg oblique SO(3)."""
    torch.manual_seed(7)
    N = 16
    spacing_lps = (1.0, 1.2, 1.8)

    # Smooth mixed divergent + rotational 3-D field on (1, Z, Y, X, 3)
    gz, gy, gx = torch.meshgrid(
        torch.linspace(0, 2 * math.pi, N + 1)[:-1],
        torch.linspace(0, 2 * math.pi, N + 1)[:-1],
        torch.linspace(0, 2 * math.pi, N + 1)[:-1],
        indexing="ij",
    )
    v_lps = torch.stack([
        torch.sin(gz) * torch.cos(gy) + 0.4 * torch.cos(gx),
        torch.cos(gy) * torch.sin(gx) - 0.3 * torch.sin(gz),
        torch.sin(gx + gz) + 0.5 * torch.cos(gy),
    ], dim=-1).unsqueeze(0).to(torch.float32)

    reg_lps = get_regularizer(
        reg_name, alpha=1.5, fluid_sigma=2.0, spacing=spacing_lps, direction=np.eye(3)
    )
    out_lps = reg_lps(v_lps)

    # 1. Rotation / reflection on fixed array axes: RPI and 25-deg oblique SO(3)
    R_rpi = np.diag([-1.0, 1.0, -1.0])
    R_oblique = _rodrigues_rotation(np.array([1.0, 2.0, -1.5]), 25.0)
    for label, D_lps in (("RPI", R_rpi), ("oblique_25deg", R_oblique)):
        D_zyx = torch.as_tensor(D_lps[::-1, ::-1].copy(), dtype=torch.float32)
        v_rot = v_lps @ D_zyx.t()
        reg_rot = get_regularizer(
            reg_name, alpha=1.5, fluid_sigma=2.0, spacing=spacing_lps, direction=D_lps
        )
        out_rot = reg_rot(v_rot)
        expected = out_lps @ D_zyx.t()
        err = float((out_rot - expected).abs().max().item())
        assert err < 1e-5, f"{reg_name} failed O(3) equivariance for {label}: max err {err:.2e}"

    # 2. PIR discrete array permutation + sign flip:
    # ITK axes: x_pir = +y_lps, y_pir = -z_lps, z_pir = -x_lps
    D_pir = np.array([
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ], dtype=np.float64)
    spacing_pir = (spacing_lps[1], spacing_lps[2], spacing_lps[0])
    # Tensor order is (z, y, x):
    # v_lps has (iz, iy, ix); v_pir has (iz_pir, iy_pir, ix_pir) = (Nx-1-ix, Nz-1-iz, iy).
    # Vector components remain physical LPS-reversed (vz, vy, vx).
    v_pir = torch.flip(v_lps.permute(0, 3, 1, 2, 4), dims=(1, 2)).contiguous()
    reg_pir = get_regularizer(
        reg_name, alpha=1.5, fluid_sigma=2.0, spacing=spacing_pir, direction=D_pir
    )
    out_pir = reg_pir(v_pir)
    # Un-permute back to LPS (iz, iy, ix)
    out_pir_unperm = torch.flip(out_pir, dims=(1, 2)).permute(0, 2, 3, 1, 4).contiguous()
    err_pir = float((out_pir_unperm - out_lps).abs().max().item())
    assert err_pir < 1e-5, f"{reg_name} failed PIR permutation equivariance: max err {err_pir:.2e}"


def test_divergence_and_curl_o3_invariance():
    """Verify spectral divergence invariance and curl pseudovector equivariance under O(3)."""
    N = 16
    spacing = (1.0, 1.3, 1.7)
    gz, gy, gx = torch.meshgrid(
        torch.linspace(0, 2 * math.pi, N + 1)[:-1],
        torch.linspace(0, 2 * math.pi, N + 1)[:-1],
        torch.linspace(0, 2 * math.pi, N + 1)[:-1],
        indexing="ij",
    )
    v = torch.stack([
        torch.sin(gz) * torch.cos(gy),
        torch.cos(gy) * torch.sin(gx),
        torch.sin(gx + gz),
    ], dim=-1).unsqueeze(0).to(torch.float32)

    div0 = compute_divergence_nd(v, spacing=spacing, method="spectral", direction=np.eye(3))
    curl0 = compute_curl_nd(v, spacing=spacing, direction=np.eye(3))

    R_oblique = _rodrigues_rotation(np.array([1.0, -1.0, 2.0]), 25.0)
    D_zyx = torch.as_tensor(R_oblique[::-1, ::-1].copy(), dtype=torch.float32)
    v_rot = v @ D_zyx.t()

    div_r = compute_divergence_nd(v_rot, spacing=spacing, method="spectral", direction=R_oblique)
    curl_r = compute_curl_nd(v_rot, spacing=spacing, direction=R_oblique)

    assert float((div_r - div0).abs().max().item()) < 1e-5
    assert float((curl_r - (curl0 @ D_zyx.t())).abs().max().item()) < 1e-5
