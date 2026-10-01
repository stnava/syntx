"""
Regression tests for behaviour bugs found by the 2026-10-01 docstring audit
(docs/DOCSTRING_AUDIT.md). CPU only, each test well under a second.
"""
import numpy as np
import pytest
import torch


# 1. core/jacobian.py 'bspline' derivative kernel ------------------------------------------
@pytest.mark.parametrize("dim", [2, 3])
def test_bspline_derivative_of_linear_field_is_exact(dim):
    from syntx.core.jacobian import _spatial_jacobian_nd
    shape = (9,) * dim
    axes = torch.meshgrid(*[torch.arange(n, dtype=torch.float64) for n in shape], indexing="ij")
    field = torch.stack([0.3 * a for a in reversed(axes)], dim=-1).unsqueeze(0)   # u_k = 0.3 x_k
    J = _spatial_jacobian_nd(field, physical_spacing=[1.0] * dim, method="bspline")
    interior = (0,) + (slice(2, -2),) * dim
    expected = 0.3 * torch.eye(dim, dtype=torch.float64)
    assert torch.allclose(J[interior], expected.expand_as(J[interior]), atol=1e-10)
    # agrees with the central-difference path
    Jc = _spatial_jacobian_nd(field, physical_spacing=[1.0] * dim, method="central")
    assert torch.allclose(J[interior], Jc[interior], atol=1e-10)


# 2. viz/reports.py: no hard-coded benchmark numbers ----------------------------------------
def _rec(idx, dice, t=10.0, aff_t=1.25, cfg=None):
    r = {"status": "SUCCESS", "pair_idx": idx, "syntx_dice_sym": dice, "syntx_time": t,
         "syntx_fold": 0.0, "syntx_affine_dice_sym": 0.5, "syntx_affine_time": aff_t,
         "ants_baseline": {"dice_sym": 0.55, "runtime_seconds": 50.0}}
    if cfg:
        r["config"] = cfg
    return r


def test_affine_report_uses_recorded_times_not_constants(tmp_path, monkeypatch):
    from syntx.viz.reports import create_affine_benchmark_report
    monkeypatch.chdir(tmp_path)                       # no results/ files -> ANTs values unknown
    out = create_affine_benchmark_report({"gaussian_results": {"0": _rec(0, 0.6), "1": _rec(1, 0.7)}},
                                         output_html=str(tmp_path / "a.html"))
    html = open(out).read()
    assert "1.2s" in html or "1.3s" in html           # recorded syntx_affine_time (1.25 s)
    for fabricated in ("2.8s", "28.5s", "0.3472", "10.2&times;", "90 / 90", "0.5303"):
        assert fabricated not in html, fabricated
    assert "n/a" in html                              # unknown ANTs time / speedup


def test_population_report_counts_from_data(tmp_path):
    from syntx.viz.reports import create_population_benchmark_report
    tvf = {str(i): _rec(i, 0.65, cfg={"grad_step": 2.0}) for i in range(3)}
    sob = {str(i): _rec(i, 0.60 if i < 2 else 0.70) for i in range(3)}
    import json
    src = tmp_path / "summary.json"
    src.write_text(json.dumps({"tvf_results": tvf, "sobolev_results": sob}))
    out = create_population_benchmark_report(str(src), output_html=str(tmp_path / "p.html"))
    html = open(out).read()
    assert "88/90" not in html and "/ 90" not in html and "TVF 0.6466" not in html
    assert "TVF beats Sobolev in 2/3" in html
    assert "grad_step" in html                        # recorded config, not a static table
    assert "lr=1.2" not in html
