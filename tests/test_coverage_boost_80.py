"""
Comprehensive test file specifically targeting lines in tvf.py, syn.py, syn_jax.py, syngs.py, syngs_jax.py,
transform.py, and viz/reports.py to push total code coverage above 80%.
"""

import os
import numpy as np
import torch
import pytest
import ants
import matplotlib.pyplot as plt

from syntx import syn
from syntx import tvf
from syntx import syngs
from syntx.robust_affine import robust_affine, create_translation_transform
from syntx.transform import SyNToTransform, export_ants_displacement_field, export_ants_affine_transform
from syntx.viz.reports import create_registration_report, build_engine_provenance

try:
    from syntx.syn_jax import registration_jax
    from syntx.tvf_jax import tvf_registration_jax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def test_tvf_registration_full_coverage(tmp_path):
    fi_arr = np.pad(np.ones((16, 16)), 8).astype(np.float32)
    mi_arr = np.pad(np.ones((16, 16)), 8).astype(np.float32)
    fi = ants.from_numpy(fi_arr, origin=(0.0, 0.0), spacing=(1.0, 1.0))
    mi = ants.from_numpy(mi_arr, origin=(0.0, 0.0), spacing=(1.0, 1.0))

    # Initial translation transform
    tx_path, _ = create_translation_transform(fi, mi, np.array([1.0, 1.0]))

    # Test PyTorch backend with initial transform, winsorizing, and LARS
    reg_pt = tvf(
        fixed=fi,
        moving=mi,
        type_of_transform='TVF',
        initial_transform=tx_path,
        syn_metric='lncc',
        reg_iterations=[2],
        affine_iterations=[1],
        grad_step=0.15,
        flow_sigma=1.0,
        total_sigma=0.05,
        n_time_steps=3,
        n_steps=3,
        cfl_momentum=0.95,
        multipoint_loss=[0.0, 0.5, 1.0],
        fast_smooth=True,
        winsorize_quantiles=(0.05, 0.95),
        backend='pytorch',
        verbose=False
    )
    assert 'warpedmovout' in reg_pt
    assert 'fwdtransforms' in reg_pt
    assert 'invtransforms' in reg_pt

    # Test create_registration_report from TVF output
    rep = create_registration_report(
        fixed=fi,
        moving=mi,
        reg=reg_pt,
        output_html=str(tmp_path / "tvf_report" / "report.html")
    )
    assert os.path.exists(rep['html_path'])


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_tvf_registration_jax_coverage():
    fi_arr = np.pad(np.ones((16, 16)), 8).astype(np.float32)
    mi_arr = np.pad(np.ones((16, 16)), 8).astype(np.float32)
    fi = ants.from_numpy(fi_arr, origin=(0.0, 0.0), spacing=(1.0, 1.0))
    mi = ants.from_numpy(mi_arr, origin=(0.0, 0.0), spacing=(1.0, 1.0))

    reg_jax = tvf(
        fixed=fi,
        moving=mi,
        type_of_transform='TVF',
        reg_iterations=[2],
        affine_iterations=[0],
        backend='jax',
        verbose=False
    )
    assert 'warpedmovout' in reg_jax
    assert 'fwdtransforms' in reg_jax


def test_syn_registration_advanced_options(tmp_path):
    fi_arr = np.pad(np.ones((16, 16)), 8).astype(np.float32)
    mi_arr = np.pad(np.ones((16, 16)), 8).astype(np.float32)
    fi = ants.from_numpy(fi_arr, origin=(0.0, 0.0), spacing=(1.0, 1.0))
    mi = ants.from_numpy(mi_arr, origin=(0.0, 0.0), spacing=(1.0, 1.0))

    # Test initial_transform with robust_affine result
    aff_res = robust_affine(fi, mi, mode='pytorch', device='cpu')
    init_tx = aff_res['fwdtransforms']

    # Test PyTorch backend with winsorize_quantiles, Mattes MI, write_composite_transform
    reg_syn = syn(
        fixed=fi,
        moving=mi,
        type_of_transform='SyN',
        initial_transform=init_tx,
        syn_metric='mattes_mi',
        reg_iterations=[2],
        affine_iterations=[1],
        winsorize_quantiles=(0.01, 0.99),
        write_composite_transform=True,
        outprefix=str(tmp_path / "syn_out"),
        backend='pytorch',
        verbose=False
    )
    assert 'warpedmovout' in reg_syn
    assert 'fwdtransforms' in reg_syn


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_syn_registration_jax_advanced():
    fi_arr = np.pad(np.ones((16, 16)), 8).astype(np.float32)
    mi_arr = np.pad(np.ones((16, 16)), 8).astype(np.float32)
    fi = ants.from_numpy(fi_arr, origin=(0.0, 0.0), spacing=(1.0, 1.0))
    mi = ants.from_numpy(mi_arr, origin=(0.0, 0.0), spacing=(1.0, 1.0))

    reg_jax = syn(
        fixed=fi,
        moving=mi,
        type_of_transform='SyN',
        reg_iterations=[2],
        affine_iterations=[1],
        syn_metric='lncc',
        backend='jax',
        verbose=False
    )
    assert 'warpedmovout' in reg_jax
    assert 'fwdtransforms' in reg_jax


def test_syngs_registration_advanced(tmp_path):
    fi_arr = np.pad(np.ones((16, 16)), 8).astype(np.float32)
    mi_arr = np.pad(np.ones((16, 16)), 8).astype(np.float32)
    fi = ants.from_numpy(fi_arr, origin=(0.0, 0.0), spacing=(1.0, 1.0))
    mi = ants.from_numpy(mi_arr, origin=(0.0, 0.0), spacing=(1.0, 1.0))

    reg_syngs = syngs(
        fixed=fi,
        moving=mi,
        type_of_transform='SyNGS',
        reg_iterations=[2],
        affine_iterations=[1],
        syn_metric='lncc',
        backend='pytorch',
        verbose=False
    )
    assert 'warpedmovout' in reg_syngs
    assert 'fwdtransforms' in reg_syngs

    rep = create_registration_report(
        fixed=fi,
        moving=mi,
        reg=reg_syngs,
        output_html=str(tmp_path / "syngs_report" / "report.html")
    )
    assert os.path.exists(rep['html_path'])


def test_synto_transform_export_classic(tmp_path):
    aff_grid = torch.eye(2, 3).unsqueeze(0)
    aff_grid_res = torch.nn.functional.affine_grid(aff_grid, size=[1, 1, 10, 10], align_corners=True)
    warp = torch.zeros(1, 10, 10, 2)
    meta = {
        'origin': [0.0, 0.0],
        'spacing': [1.0, 1.0],
        'direction': np.eye(2),
        'shape': (10, 10)
    }

    st = SyNToTransform(aff_grid_res, warp, meta, device='cpu')
    paths = st.export_classic(prefix=str(tmp_path / "classic_tx"))
    assert len(paths) > 0
    assert os.path.exists(paths[0])


def test_viz_stats_more():
    from syntx.viz.stats import plot_label_overlap_stats, plot_loss_convergence

    dice_dict = {
        'fixed_dice': [0.8, 0.82, 0.85],
        'moving_dice': [0.78, 0.81, 0.84],
        'sym_dice': [0.79, 0.815, 0.845]
    }
    fig1 = plot_label_overlap_stats(dice_dict, show_figure=False)
    assert fig1 is not None
    plt.close(fig1)

    loss_history = [0.5, 0.4, 0.3, 0.2, 0.15]
    fig2 = plot_loss_convergence(loss_history, show_figure=False)
    assert fig2 is not None
    plt.close(fig2)


def test_dsti_operator_tvf():
    from syntx.tvf import TVFModel

    m2d = torch.randn(1, 16, 16, 2)
    model2d = TVFModel(dim=2, image_shape=(16, 16), velocity_shape=(16, 16), fluid_sigma=1.0)
    dsti_out_2d = model2d._apply_dsti_green_operator(m2d)
    assert dsti_out_2d.shape == m2d.shape

    m3d = torch.randn(1, 12, 12, 12, 3)
    model3d = TVFModel(dim=3, image_shape=(12, 12, 12), velocity_shape=(12, 12, 12), fluid_sigma=1.0)
    dsti_out_3d = model3d._apply_dsti_green_operator(m3d)
    assert dsti_out_3d.shape == m3d.shape


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_tvf_jax_initial_transform():
    fi_arr = np.pad(np.ones((16, 16)), 8).astype(np.float32)
    mi_arr = np.pad(np.ones((16, 16)), 8).astype(np.float32)
    fi = ants.from_numpy(fi_arr, origin=(0.0, 0.0), spacing=(1.0, 1.0))
    mi = ants.from_numpy(mi_arr, origin=(0.0, 0.0), spacing=(1.0, 1.0))

    tx_path, _ = create_translation_transform(fi, mi, np.array([1.0, 1.0]))

    reg_jax = tvf(
        fixed=fi,
        moving=mi,
        type_of_transform='TVF',
        initial_transform=tx_path,
        reg_iterations=[2],
        affine_iterations=[1],
        backend='jax',
        verbose=False
    )
    assert 'warpedmovout' in reg_jax


def test_syngs_affine_prealignment():
    from syntx.syngs import GeodesicShootingModel

    model = GeodesicShootingModel(
        dim=2,
        image_shape=(16, 16),
        velocity_shape=(16, 16),
        spacing=[1.0, 1.0],
        origin=[0.0, 0.0]
    )
    fixed = torch.randn(1, 1, 16, 16)
    moving = torch.randn(1, 1, 16, 16)

    model.fit(
        fixed, moving,
        levels=[1],
        epochs_per_level=[1],
        affine_epochs=2,
        verbose=False
    )
    assert model.get_forward_warp() is not None


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_syn_jax_lbfgs():
    import jax.numpy as jnp
    from syntx.syn_jax import SyNTo as SyNToJAX

    model_syn = SyNToJAX(
        dim=2,
        grid_shape=(16, 16),
        spacing=[1.0, 1.0],
        origin=[0.0, 0.0]
    )
    I_jax = jnp.ones((1, 1, 16, 16), dtype=jnp.float32)
    J_jax = jnp.ones((1, 1, 16, 16), dtype=jnp.float32)

    model_syn.fit(
        I_jax, J_jax,
        levels=[1],
        epochs_per_level=[2],
        affine_epochs=0,
        optimizer_type='lbfgs',
        verbose=False
    )
    assert model_syn.warp_l2r is not None
