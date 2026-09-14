import pytest
import ants
import numpy as np
import syntx

def test_auto_reg_zero_effort_2d():
    fi = ants.image_read(ants.get_data('r16'))
    mi = ants.image_read(ants.get_data('r64'))

    # Zero-effort invocation
    res = syntx.auto_reg(fixed=fi, moving=mi, reg_iterations=[20, 10], affine_iterations=[20, 10], verbose=True)

    print("fwdtransforms:", res['fwdtransforms'])
    print("metrics:", res['metrics'])

    assert 'warpedmovout' in res
    assert 'warpedfixout' in res
    assert 'fwdtransforms' in res
    assert 'invtransforms' in res
    assert 'metrics' in res

    metrics = res['metrics']
    assert isinstance(metrics, dict)
    assert 'execution_time_seconds' in metrics
    assert metrics['execution_time_seconds'] > 0
    assert 'device_used' in metrics
    assert 'backend_used' in metrics

    assert 'jac_mean' in metrics
    assert 'folding_pct' in metrics
    assert 'smooth_1st' in metrics
    assert 'smooth_2nd' in metrics
    assert 'lncc_score' in metrics
    assert 'mse_score' in metrics
    assert 'mattes_mi_score' in metrics

    # Folding percentage should be very small (allow up to 0.1% for MPS non-determinism)
    assert metrics['folding_pct'] < 0.1

def test_auto_reg_syn_transform_2d():
    fi = ants.image_read(ants.get_data('r16'))
    mi = ants.image_read(ants.get_data('r64'))

    res = syntx.auto_reg(fixed=fi, moving=mi, type_of_transform='SyNTo', reg_iterations=[20, 10], affine_iterations=[20, 10], verbose=False)

    assert 'warpedmovout' in res
    assert 'metrics' in res
    assert res['metrics']['type_of_transform_used'] in ('SyNTo', 'SyN (Eulerian Sobolev)')
    assert res['metrics']['folding_pct'] < 0.1

def test_auto_reg_docstring_explicit_defaults():
    doc = syntx.auto_reg.__doc__
    assert doc is not None
    assert "SyNTo" in doc or "TVF" in doc
    assert "grad_step" in doc
    assert "flow_sigma" in doc
    assert "interpolator" in doc
    assert "metrics" in doc


def test_auto_reg_guided_sulcal_2d():
    fi = ants.image_read(ants.get_data('r16'))
    mi = ants.image_read(ants.get_data('r64'))

    res = syntx.auto_reg(
        fixed=fi,
        moving=mi,
        guided='sulcal',
        cohort_type='inter',
        reg_iterations=[15, 5],
        affine_iterations=[5, 5],
        levels=[2, 1],
        verbose=False
    )

    assert 'warpedmovout' in res
    assert 'metrics' in res
    assert "Guided" in res['metrics']['type_of_transform_used']
    assert res['metrics']['folding_pct'] < 0.1


def test_auto_reg_with_labels_2d():
    fi = ants.image_read(ants.get_data('r16'))
    mi = ants.image_read(ants.get_data('r64'))
    fl = ants.threshold_image(fi, "Otsu", 3).threshold_image(2, 2)
    ml = ants.threshold_image(mi, "Otsu", 3).threshold_image(2, 2)

    res = syntx.auto_reg(
        fixed=fi,
        moving=mi,
        fixed_label=fl,
        moving_label=ml,
        reg_iterations=[10],
        affine_iterations=[5],
        levels=[1],
        verbose=False
    )

    metrics = res['metrics']
    assert 'dice_fixed' in metrics
    assert 'dice_moving' in metrics
    assert 'dice_sym' in metrics
    assert 0.0 < metrics['dice_sym'] <= 1.0


def test_auto_reg_robust_affine_toggle_2d():
    fi = ants.image_read(ants.get_data('r16'))
    mi = ants.image_read(ants.get_data('r64'))

    res = syntx.auto_reg(
        fixed=fi,
        moving=mi,
        robust_affine=True,
        reg_iterations=[10],
        levels=[1],
        verbose=False
    )

    assert 'fwdtransforms' in res
    assert len(res['fwdtransforms']) >= 1

