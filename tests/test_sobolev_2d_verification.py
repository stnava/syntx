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


@pytest.mark.parametrize('alpha', [0.1, 1.0, 10.0, 50.0])
def test_sobolev_2d_synto_stability_and_folding(alpha):
    device = get_test_device()
    model = SyNTo(dim=2, grid_shape=(32, 32)).to(device)
    torch.manual_seed(42)
    m_2d = torch.randn(1, 32, 32, 2, dtype=torch.float32, device=device) * 0.005

    v_out = model._apply_sobolev_green_operator(m_2d, fluid_sigma=2.0, alpha=alpha)
    
    assert v_out.shape == m_2d.shape
    assert v_out.dtype == torch.float32
    assert torch.isfinite(v_out).all(), "2D Sobolev output contains non-finite values"

    jac_det = compute_jacobian_determinant_nd(v_out.contiguous())
    jac_np = jac_det.squeeze().cpu().numpy()

    min_det_j = float(np.min(jac_np))
    folding_pct = float(np.mean(jac_np <= 0.0) * 100.0)

    assert folding_pct == 0.0, f"Grid folding detected in 2D Sobolev (alpha={alpha}): {folding_pct}%"
    assert min_det_j > 0.0, f"Min det(J) must be positive (alpha={alpha}), got {min_det_j}"


@pytest.mark.parametrize('alpha', [0.1, 1.0, 10.0, 50.0])
def test_sobolev_2d_tvf_stability_and_folding(alpha):
    device = get_test_device()
    model = TVFModel(dim=2, image_shape=(32, 32), velocity_shape=(32, 32)).to(device)
    torch.manual_seed(42)
    m_2d = torch.randn(1, 32, 32, 2, dtype=torch.float32, device=device) * 0.005

    v_out = model._apply_sobolev_green_operator(m_2d, fluid_sigma=2.0, alpha=alpha, border_width=0)
    
    assert v_out.shape == m_2d.shape
    assert torch.isfinite(v_out).all()

    jac_det = compute_jacobian_determinant_nd(v_out.contiguous())
    jac_np = jac_det.squeeze().cpu().numpy()

    min_det_j = float(np.min(jac_np))
    folding_pct = float(np.mean(jac_np <= 0.0) * 100.0)

    assert folding_pct == 0.0, f"Grid folding detected in 2D TVF Sobolev (alpha={alpha}): {folding_pct}%"
    assert min_det_j > 0.0, f"Min det(J) must be positive (alpha={alpha}), got {min_det_j}"


@pytest.mark.parametrize('alpha', [0.1, 1.0, 10.0, 50.0])
def test_sobolev_2d_syngs_stability_and_folding(alpha):
    device = get_test_device()
    model = GeodesicShootingModel(dim=2, image_shape=(32, 32)).to(device)
    torch.manual_seed(42)
    m_2d = torch.randn(1, 32, 32, 2, dtype=torch.float32, device=device) * 0.005

    v_out = model._apply_sobolev_green_operator(m_2d, fluid_sigma=2.0, alpha=alpha, border_width=0)
    
    assert v_out.shape == m_2d.shape
    assert torch.isfinite(v_out).all()

    jac_det = compute_jacobian_determinant_nd(v_out.contiguous())
    jac_np = jac_det.squeeze().cpu().numpy()

    min_det_j = float(np.min(jac_np))
    folding_pct = float(np.mean(jac_np <= 0.0) * 100.0)

    assert folding_pct == 0.0, f"Grid folding detected in 2D SyNGS Sobolev (alpha={alpha}): {folding_pct}%"
    assert min_det_j > 0.0, f"Min det(J) must be positive (alpha={alpha}), got {min_det_j}"


@pytest.mark.parametrize('alpha', [0.1, 1.0, 10.0, 50.0])
def test_sobolev_2d_jax_stability(alpha):
    import jax.numpy as jnp
    m_2d = jnp.array(np.random.randn(1, 32, 32, 2).astype(np.float32)) * 0.05
    
    syn_jax = SyNJAX(dim=2, grid_shape=(32, 32))
    v_out_syn = syn_jax._apply_sobolev_green_operator(m_2d, fluid_sigma=2.0, alpha=alpha, border_width=0)
    assert jnp.isfinite(v_out_syn).all()

    tvf_jax = TVFModelJAX(dim=2, image_shape=(32, 32), velocity_shape=(32, 32))
    v_out_tvf = tvf_jax._apply_sobolev_green_operator(m_2d, fluid_sigma=2.0, alpha=alpha, border_width=0)
    assert jnp.isfinite(v_out_tvf).all()

    gs_jax = GeodesicShootingModelJAX(dim=2, image_shape=(32, 32), velocity_shape=(32, 32))
    v_out_gs = gs_jax._apply_sobolev_green_operator(m_2d, fluid_sigma=2.0, alpha=alpha, border_width=0)
    assert jnp.isfinite(v_out_gs).all()


@pytest.mark.parametrize('alpha', [0.1, 1.0, 10.0, 50.0])
@pytest.mark.parametrize('mname', ['syn', 'tvf'])
def test_sobolev_2d_registration_fit_extreme_alphas(mname, alpha):
    import ants
    from syntx.viz.reports import _compute_jacobian_stats
    bdata = syntx.benchmark_data('2d')
    fixed = bdata['fixed']
    moving = bdata['moving']
    
    reg_fn = getattr(syntx, mname)
    if mname == 'syn':
        res = reg_fn(fixed, moving, type_of_transform='SyNTo', regularizer='sobolev', sobolev_alpha=alpha, reg_iterations=[20, 20, 10], verbose=False)
    else:
        res = reg_fn(fixed, moving, type_of_transform='SyNTVF', regularizer='sobolev', sobolev_alpha=alpha, optimizer='lars', lr=0.60, flow_sigma=0.5, total_sigma=0.05, reg_iterations=[50, 50, 20], verbose=False)

    warped = res['warpedmovout']
    warped_np = warped.numpy() if hasattr(warped, 'numpy') else np.asarray(warped)
    assert np.isfinite(warped_np).all(), f"Warped image for {mname} (alpha={alpha}) contains non-finite values"

    transforms = res['fwdtransforms']
    warp_file = [t for t in transforms if 'warp' in t or 'Warp' in t or t.endswith('.nii.gz') or t.endswith('.nii')][0]
    warp_img = ants.image_read(warp_file)

    _, jstats = _compute_jacobian_stats(warp_img, fixed)
    assert jstats['folding_pct'] <= 1.0, f"Grid folding detected in {mname} fit (alpha={alpha}): {jstats['folding_pct']}%"
    assert jstats['min'] > -5.0, f"Min det(J) must be reasonable in {mname} fit (alpha={alpha}), got {jstats['min']}"

