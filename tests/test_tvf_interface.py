"""
syntx.tvf interface contract: the benchmark configuration blocks mirror syntx.tvf()'s effective
defaults, and the advanced options are the declared set (nothing silently swallowed).
Per-parameter behaviour (accepted -> has an effect; inapplicable -> raises) is pinned in
tests/test_parameter_sensitivity.py::TestTVFParameterSensitivity.
"""

import inspect
import json
import os

import pytest

import syntx
from syntx.tvf import TVF_ADVANCED_OPTIONS, TVF_DEFAULT_ALPHA, tvf_registration

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _effective_defaults(dim=3):
    sig = inspect.signature(tvf_registration)
    d = {k: p.default for k, p in sig.parameters.items() if p.default is not inspect.Parameter.empty}
    d["alpha"] = TVF_DEFAULT_ALPHA[dim] if d["alpha"] is None else d["alpha"]
    if d["optimizer"] == "cfl":            # optimiser-family parameters resolve per family
        d["grad_step"] = 1.0 if d["grad_step"] is None else d["grad_step"]
        d["cfl_momentum"] = 0.9 if d["cfl_momentum"] is None else d["cfl_momentum"]
        del d["optimizer_lr"], d["max_step_norm"]
    else:
        d["optimizer_lr"] = 1.2 if d["optimizer_lr"] is None else d["optimizer_lr"]
        if d["optimizer"] == "reg_adam":
            d["max_step_norm"] = 0.5 if d["max_step_norm"] is None else d["max_step_norm"]
        del d["grad_step"], d["cfl_momentum"]
    d["fast_smooth"] = False if d["fast_smooth"] is None else d["fast_smooth"]
    d["reg_iterations"] = [100, 100, 20] if d["reg_iterations"] is None else d["reg_iterations"]
    d["multipoint_loss"] = list(d["multipoint_loss"])
    return d


@pytest.mark.parametrize("source", ["config.py", "run_config.json"])
def test_tvf_config_blocks_mirror_the_defaults(source):
    if source == "config.py":
        from syntx.benchmark.config import DEFAULT_BENCHMARK_CONFIG
        block = DEFAULT_BENCHMARK_CONFIG["tvf_config"]
    else:
        block = json.load(open(os.path.join(ROOT, "docs/provenance/run_config.json")))["tvf_config"]
    eff = _effective_defaults(3)
    assert {k: v for k, v in block.items() if eff.get(k) != v} == {}, (
        f"{source} tvf_config differs from syntx.tvf defaults")
    assert set(block) <= set(eff), f"{source} tvf_config has keys syntx.tvf does not take: {set(block) - set(eff)}"


def test_tvf_accepts_every_declared_advanced_option_name():
    # the declared options are exactly what the front end forwards; spot-check one that the
    # model reads and one that it forwards to the model constructor
    assert "constant_speed" in TVF_ADVANCED_OPTIONS and "solver" in TVF_ADVANCED_OPTIONS
    assert syntx.tvf is tvf_registration


def test_tvf_defaults_on_2d_pair():
    """Regression guard for the 2026-09-30 study: TVF defaults on r16 -> r64 are diffeomorphic by
    the exact (Liouville) determinant with bounded compression, beat SyN's Dice there (0.783),
    and the path is efficient (max speed / max displacement)."""
    import numpy as np
    import ants
    from syntx.benchmark.tune import METHODS, twod_evaluator
    store = {}
    orig = syntx.tvf

    def capture(*a, **k):
        store["res"] = orig(*a, **k)
        return store["res"]

    syntx.tvf = capture
    try:
        m, _ = twod_evaluator(METHODS["tvf"])(0, {})
    finally:
        syntx.tvf = orig
    assert m["folding_pct"] == 0.0 and m["jac_min"] > 0.01, m          # Liouville measure
    assert m["dice_sym"] > 0.785, m
    assert m["inv_interior_max_mm"] < 5.0, m
    v = store["res"]["model"].velocity.detach().squeeze(1).cpu().numpy()
    disp = ants.image_read(next(x for x in store["res"]["fwdtransforms"] if x.endswith(".nii.gz"))).numpy()
    assert np.linalg.norm(v, axis=-1).max() / np.linalg.norm(disp, axis=-1).max() < 1.6


def test_liouville_jacobian_is_exact_where_smooth_and_positive_everywhere():
    """TVFModel.jacobian_determinant (Liouville: exp int div v dt along the flow) agrees with a
    finite-difference Jacobian of the exported map where that is reliable, and is > 0 everywhere."""
    import numpy as np
    import ants
    import torch
    fi = ants.image_read(ants.get_ants_data("r16")).resample_image((2, 2), 0, 0)
    mi = ants.image_read(ants.get_ants_data("r64")).resample_image((2, 2), 0, 0)
    r = syntx.tvf(fixed=fi, moving=mi, initial_transform="identity", reg_iterations=[20, 10, 5], device="cpu")
    m = r["model"]
    det = m.jacobian_determinant(image_shape=m.image_shape)[0].numpy()
    with torch.no_grad():
        u = m.integrate(0.0, 1.0, image_shape=m.image_shape).squeeze(0).numpy()
    sp = np.array(fi.spacing)[::-1]                        # tensor (z,y,x) order
    J = np.empty(u.shape[:2] + (2, 2))
    for c in range(2):
        for a in range(2):
            J[..., c, a] = (1.0 if c == a else 0.0) + np.gradient(u[..., c], axis=a) / sp[a]
    fd = np.linalg.det(J)
    assert det.min() > 0
    ok = fd > 0.3
    assert abs(np.median(det[ok] / fd[ok]) - 1.0) < 0.01
    assert np.corrcoef(np.log(det[ok]), np.log(fd[ok]))[0, 1] > 0.95
