import os
import sys
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import syntx
from syntx.syn import SyNTo, compute_jacobian_determinant_nd
from syntx.tvf import TVFModel
from syntx.syngs import GeodesicShootingModel
from syntx.syn_jax import SyNJAX
from syntx.tvf_jax import TVFModelJAX
from syntx.syngs_jax import GeodesicShootingModelJAX


def get_test_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def test_sobolev_3d_synto_stability_and_folding():
    device = get_test_device()
    model = SyNTo(dim=3, grid_shape=(16, 16, 16)).to(device)
    torch.manual_seed(42)
    m_3d = torch.randn(1, 16, 16, 16, 3, dtype=torch.float32, device=device) * 0.05

    v_out = model._apply_sobolev_green_operator(m_3d, fluid_sigma=2.0, alpha=1.0, border_width=0)
    
    assert v_out.shape == m_3d.shape
    assert v_out.dtype == torch.float32
    assert torch.isfinite(v_out).all(), "3D Sobolev output contains non-finite values"

    jac_det = compute_jacobian_determinant_nd(v_out.contiguous())
    jac_np = jac_det.squeeze().cpu().numpy()

    min_det_j = float(np.min(jac_np))
    folding_pct = float(np.mean(jac_np <= 0.0) * 100.0)

    assert folding_pct == 0.0, f"Grid folding detected in 3D Sobolev: {folding_pct}%"
    assert min_det_j > 0.0, f"Min det(J) must be positive, got {min_det_j}"


def test_sobolev_3d_tvf_stability_and_folding():
    device = get_test_device()
    model = TVFModel(dim=3, image_shape=(16, 16, 16), velocity_shape=(16, 16, 16)).to(device)
    torch.manual_seed(42)
    m_3d = torch.randn(1, 16, 16, 16, 3, dtype=torch.float32, device=device) * 0.05

    v_out = model._apply_sobolev_green_operator(m_3d, fluid_sigma=2.0, alpha=1.0, border_width=0)
    
    assert v_out.shape == m_3d.shape
    assert torch.isfinite(v_out).all()

    jac_det = compute_jacobian_determinant_nd(v_out.contiguous())
    jac_np = jac_det.squeeze().cpu().numpy()

    min_det_j = float(np.min(jac_np))
    folding_pct = float(np.mean(jac_np <= 0.0) * 100.0)

    assert folding_pct == 0.0, f"Grid folding detected in 3D TVF Sobolev: {folding_pct}%"
    assert min_det_j > 0.0, f"Min det(J) must be positive, got {min_det_j}"


def test_sobolev_3d_syngs_stability_and_folding():
    device = get_test_device()
    model = GeodesicShootingModel(dim=3, image_shape=(16, 16, 16)).to(device)
    torch.manual_seed(42)
    m_3d = torch.randn(1, 16, 16, 16, 3, dtype=torch.float32, device=device) * 0.05

    v_out = model._apply_sobolev_green_operator(m_3d, fluid_sigma=2.0, alpha=1.0, border_width=0)
    
    assert v_out.shape == m_3d.shape
    assert torch.isfinite(v_out).all()

    jac_det = compute_jacobian_determinant_nd(v_out.contiguous())
    jac_np = jac_det.squeeze().cpu().numpy()

    min_det_j = float(np.min(jac_np))
    folding_pct = float(np.mean(jac_np <= 0.0) * 100.0)

    assert folding_pct == 0.0, f"Grid folding detected in 3D SyNGS Sobolev: {folding_pct}%"
    assert min_det_j > 0.0, f"Min det(J) must be positive, got {min_det_j}"


def test_sobolev_3d_jax_stability_and_dsti_separable():
    import jax.numpy as jnp
    m_3d = jnp.array(np.random.randn(1, 16, 16, 16, 3).astype(np.float32)) * 0.05
    
    syn_jax = SyNJAX(dim=3, grid_shape=(16, 16, 16))
    v_out_syn = syn_jax._apply_sobolev_green_operator(m_3d, fluid_sigma=2.0, alpha=1.0, border_width=0)
    assert jnp.isfinite(v_out_syn).all()

    v_dst_syn = syn_jax._apply_dsti_green_operator(m_3d, fluid_sigma=2.0, alpha=1.0)
    assert jnp.isfinite(v_dst_syn).all()
    assert v_dst_syn.shape == m_3d.shape

    tvf_jax = TVFModelJAX(dim=3, image_shape=(16, 16, 16), velocity_shape=(16, 16, 16))
    v_out_tvf = tvf_jax._apply_sobolev_green_operator(m_3d, fluid_sigma=2.0, alpha=1.0, border_width=0)
    assert jnp.isfinite(v_out_tvf).all()

    v_dst_tvf = tvf_jax._apply_dsti_green_operator(m_3d, fluid_sigma=2.0, alpha=1.0)
    assert jnp.isfinite(v_dst_tvf).all()
    assert v_dst_tvf.shape == m_3d.shape

    gs_jax = GeodesicShootingModelJAX(dim=3, image_shape=(16, 16, 16), velocity_shape=(16, 16, 16))
    v_out_gs = gs_jax._apply_sobolev_green_operator(m_3d, fluid_sigma=2.0, alpha=1.0, border_width=0)
    assert jnp.isfinite(v_out_gs).all()
