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

def test_auto_reg_docstring_explicit_defaults():
    doc = syntx.auto_reg.__doc__
    assert doc is not None
    assert "SyNTo" in doc
    assert "grad_step" in doc
    assert "flow_sigma" in doc
    assert "interpolator" in doc
    assert "metrics" in doc
