import tempfile
import numpy as np
import pytest
import ants
import torch

from syntx.qc import RegistrationQCReport, evaluate_registration_qc


def test_qc_report_dataclass():
    report = RegistrationQCReport(
        status='PASS',
        is_diffeomorphic=True,
        folding_pct=0.005,
        min_jacobian=0.45,
        harmonic_energy=0.12,
        bending_energy=0.03,
        inverse_consistency_error_max_mm=0.25,
        inverse_consistency_error_mean_mm=0.08,
        target_dice_symmetric=0.88,
        failure_flags=[],
        warnings=[],
        recommended_remedy='none'
    )
    d = report.to_dict()
    assert d['status'] == 'PASS'
    assert d['is_diffeomorphic'] is True
    assert d['folding_pct'] == 0.005
    assert d['ice_max_mm'] == 0.25
    assert d['dice'] == 0.88


def test_qc_render_html():
    report = RegistrationQCReport(
        status='WARNING',
        is_diffeomorphic=True,
        folding_pct=0.010,
        min_jacobian=0.008,
        harmonic_energy=0.25,
        bending_energy=0.05,
        inverse_consistency_error_max_mm=1.2,
        target_dice_symmetric=0.65,
        warnings=['LOW_MIN_JACOBIAN: min det(J)=0.0080 is close to singularity'],
        recommended_remedy='reduce_cfl_step'
    )
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
        html = report.render_html(output_path=tmp.name)
        assert "<html" in html
        assert "syntx Registration QC Report" in html
        assert "WARNING" in html
        assert "LOW_MIN_JACOBIAN" in html
        assert "reduce_cfl_step" in html
        with open(tmp.name, 'r', encoding='utf-8') as f:
            content = f.read()
            assert len(content) > 0


def test_evaluate_registration_qc_diffeomorphic():
    # Synthetic 2D image
    arr_f = np.zeros((32, 32), dtype=np.float32)
    arr_f[8:24, 8:24] = 1.0
    fi = ants.from_numpy(arr_f, origin=(0.0, 0.0), spacing=(1.0, 1.0))
    mi = fi.clone()

    # Small smooth warp (diffeomorphic)
    warp_arr = np.zeros((32, 32, 2), dtype=np.float32)
    warp_img = ants.from_numpy(warp_arr, origin=(0.0, 0.0), spacing=(1.0, 1.0), has_components=True)

    reg_result = {
        'warpedmovout': mi,
        'fwdtransforms': [warp_img],
        'invtransforms': [warp_img],
        'inv_identity_error': {'max_error': 0.05, 'mean_error': 0.01},
    }

    report = evaluate_registration_qc(fixed=fi, moving=mi, registration_result=reg_result)
    assert report.status == 'PASS'
    assert report.is_diffeomorphic is True
    assert report.folding_pct == 0.0
    assert report.min_jacobian >= 0.99
    assert report.recommended_remedy == 'none'


def test_evaluate_registration_qc_folding():
    # Synthetic 2D image
    arr_f = np.ones((32, 32), dtype=np.float32)
    fi = ants.from_numpy(arr_f, origin=(0.0, 0.0), spacing=(1.0, 1.0))
    mi = fi.clone()

    # Severe folding displacement field (crossing grid lines)
    warp_arr = np.zeros((32, 32, 2), dtype=np.float32)
    # Violent gradient causing coordinate inversion:
    warp_arr[10:20, 10:20, 0] = -15.0
    warp_arr[10:20, 10:20, 1] = 15.0
    warp_img = ants.from_numpy(warp_arr, origin=(0.0, 0.0), spacing=(1.0, 1.0), has_components=True)

    reg_result = {
        'warpedmovout': mi,
        'fwdtransforms': [warp_img],
        'invtransforms': [warp_img],
    }

    report = evaluate_registration_qc(fixed=fi, moving=mi, registration_result=reg_result)
    assert report.status == 'FAIL'
    assert report.is_diffeomorphic is False
    assert any('TOPOLOGY_FOLDING' in f for f in report.failure_flags)
    assert report.recommended_remedy == 'increase_fluid_smoothing'
