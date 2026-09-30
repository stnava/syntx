# Energy Gap Analysis and Parameter Sensitivity Tests for syntx.syn AND syntx.tvf
#
# This module verifies that changing input parameters actually changes outputs,
# catching bugs like the use_analytical_gradients parameter being silently ignored,
# or the double-sqrt sigma bug where fit() applied sqrt() on top of tvf_registration()'s sqrt().
#
# For continuous parameters, we additionally verify that the change produces a
# measurable difference in registration PERFORMANCE (Dice score), not just a
# numerical difference in displacement field values. This catches bugs where a
# parameter changes the field by floating-point noise but has no functional effect.

import pytest
import numpy as np
import torch
import ants


def _get_r16_images():
    """Get the r16/r64 benchmark pair as ANTsImages."""
    import syntx
    res = syntx.benchmark_data('r16_r64')
    return res['fixed'], res['moving'], res['fixed_label'], res['moving_label']


def _quick_syn(fixed, moving, **kwargs):
    """Run a minimal syntx.syn and return the registration result dict."""
    import syntx
    defaults = dict(
        flow_sigma=3.0,
        grad_step=0.25,
        total_sigma=0.0,
        reg_iterations=[10],
        syn_sampling=2,
    )
    defaults.update(kwargs)
    return syntx.syn(fixed, moving, **defaults)


def _quick_tvf(fixed, moving, **kwargs):
    """Minimal syntx.tvf run (CPU: deterministic, so 'no change' is exact)."""
    import syntx
    defaults = dict(initial_transform="identity", reg_iterations=[10], device="cpu", verbose=False)
    defaults.update(kwargs)
    return syntx.tvf(fixed, moving, **defaults)


def _get_warp(reg):
    """Extract the forward warp displacement field as numpy array."""
    return ants.image_read(reg['fwdtransforms'][0]).numpy()


def _compute_dice(reg, fixed_label, moving_label, fixed, moving):
    """Compute symmetric Dice from a registration result."""
    from syntx.benchmark.worker import compute_bidirectional_dice
    _, _, d_sym = compute_bidirectional_dice(
        fixed_label, moving_label, fixed, moving,
        reg['fwdtransforms'], reg['invtransforms']
    )
    return d_sym


def _assert_different_field(warp_a, warp_b, param_name, val_a, val_b, engine="syn"):
    """Assert two warps differ in displacement values."""
    max_diff = np.abs(warp_a - warp_b).max()
    assert max_diff > 1e-6, (
        f"[{engine}] {param_name}={val_a} vs {val_b} produced identical fields "
        f"(max_diff={max_diff}). Parameter is being silently ignored."
    )


def _assert_different_dice(dice_a, dice_b, param_name, val_a, val_b, engine="syn", min_gap=0.00005):
    """Assert two registrations produce measurably different Dice scores.

    A continuous parameter that changes the field but not the Dice by at least
    `min_gap` is functionally inert — it changes outputs by numerical noise only.
    """
    gap = abs(dice_a - dice_b)
    assert gap > min_gap, (
        f"[{engine}] {param_name}={val_a} (Dice={dice_a:.5f}) vs {val_b} (Dice={dice_b:.5f}) "
        f"produced a Dice gap of only {gap:.6f} < {min_gap}. "
        f"Parameter has no functional effect on registration quality."
    )


# =============================================================================
# SyN Parameter Sensitivity Tests
# =============================================================================
class TestSyNParameterSensitivity:
    """Verify that changing key parameters produces measurably different outputs
    for syntx.syn (SyN registration).
    """

    @pytest.fixture(autouse=True)
    def setup_images(self):
        self.fi, self.mi, self.fl, self.ml = _get_r16_images()

    def test_flow_sigma_changes_performance(self):
        """Different flow_sigma values must produce different Dice scores."""
        reg_a = _quick_syn(self.fi, self.mi, flow_sigma=0.5)
        reg_b = _quick_syn(self.fi, self.mi, flow_sigma=5.0)
        _assert_different_field(_get_warp(reg_a), _get_warp(reg_b), "flow_sigma", 0.5, 5.0, "syn")
        dice_a = _compute_dice(reg_a, self.fl, self.ml, self.fi, self.mi)
        dice_b = _compute_dice(reg_b, self.fl, self.ml, self.fi, self.mi)
        _assert_different_dice(dice_a, dice_b, "flow_sigma", 0.5, 5.0, "syn")

    def test_grad_step_changes_performance(self):
        """Different grad_step values must produce different Dice scores."""
        reg_a = _quick_syn(self.fi, self.mi, grad_step=0.05)
        reg_b = _quick_syn(self.fi, self.mi, grad_step=0.50)
        _assert_different_field(_get_warp(reg_a), _get_warp(reg_b), "grad_step", 0.05, 0.50, "syn")
        dice_a = _compute_dice(reg_a, self.fl, self.ml, self.fi, self.mi)
        dice_b = _compute_dice(reg_b, self.fl, self.ml, self.fi, self.mi)
        _assert_different_dice(dice_a, dice_b, "grad_step", 0.05, 0.50, "syn")

    def test_total_sigma_changes_performance(self):
        """Different total_sigma values must produce different Dice scores."""
        reg_a = _quick_syn(self.fi, self.mi, total_sigma=0.0)
        reg_b = _quick_syn(self.fi, self.mi, total_sigma=5.0)
        _assert_different_field(_get_warp(reg_a), _get_warp(reg_b), "total_sigma", 0.0, 5.0, "syn")
        dice_a = _compute_dice(reg_a, self.fl, self.ml, self.fi, self.mi)
        dice_b = _compute_dice(reg_b, self.fl, self.ml, self.fi, self.mi)
        _assert_different_dice(dice_a, dice_b, "total_sigma", 0.0, 5.0, "syn")



    def test_syn_sampling_changes_performance(self):
        """Different LNCC radius must produce different Dice scores."""
        reg_a = _quick_syn(self.fi, self.mi, syn_sampling=1)
        reg_b = _quick_syn(self.fi, self.mi, syn_sampling=4)
        _assert_different_field(_get_warp(reg_a), _get_warp(reg_b), "syn_sampling", 1, 4, "syn")
        dice_a = _compute_dice(reg_a, self.fl, self.ml, self.fi, self.mi)
        dice_b = _compute_dice(reg_b, self.fl, self.ml, self.fi, self.mi)
        _assert_different_dice(dice_a, dice_b, "syn_sampling", 1, 4, "syn")

    def test_use_analytical_gradients_changes_output(self):
        """Analytical vs autograd modes must produce different fields."""
        warp_a = _get_warp(_quick_syn(self.fi, self.mi, use_analytical_gradients=True))
        warp_b = _get_warp(_quick_syn(self.fi, self.mi, use_analytical_gradients=False))
        _assert_different_field(warp_a, warp_b, "use_analytical_gradients", True, False, "syn")

    def test_formulation_changes_output(self):
        """Eulerian vs lagrangian formulation must produce different fields."""
        warp_a = _get_warp(_quick_syn(self.fi, self.mi, formulation='eulerian'))
        warp_b = _get_warp(_quick_syn(self.fi, self.mi, formulation='lagrangian'))
        _assert_different_field(warp_a, warp_b, "formulation", "eulerian", "lagrangian", "syn")

    def test_antisymmetric_changes_output(self):
        """antisymmetric=True vs False must produce different fields."""
        warp_a = _get_warp(_quick_syn(self.fi, self.mi, antisymmetric=True))
        warp_b = _get_warp(_quick_syn(self.fi, self.mi, antisymmetric=False))
        _assert_different_field(warp_a, warp_b, "antisymmetric", True, False, "syn")


# =============================================================================
# TVF: every accepted parameter has an effect; every inapplicable one raises
# =============================================================================
# (base configuration, parameter, value): changing `parameter` from the base must change the warp.
_TVF_EFFECTIVE = [
    ({}, "alpha", 0.5),
    ({}, "total_alpha", 0.05),
    ({"regularizer": "dsti1"}, "alpha", 0.5),
    ({"regularizer": "gaussian"}, "flow_sigma", 1.0),
    ({"regularizer": "gaussian"}, "total_sigma", 0.5),
    ({}, "regularizer", "dsti1"),
    ({}, "regularizer", "gaussian"),
    ({}, "syn_sampling", 4),
    ({}, "n_time_steps", 5),
    ({}, "multipoint_loss", [0.5]),
    ({"optimizer": "cfl"}, "fast_smooth", True),
    ({}, "max_step_norm", 0.1),
    ({"max_step_norm": 1e6}, "optimizer_lr", 0.3),
    ({}, "cfl_max", 0.01),
    ({}, "constant_speed", False),
    ({}, "constant_speed_relaxation", 0.5),
    ({"optimizer": "cfl"}, "grad_step", 0.05),
    ({"optimizer": "cfl"}, "cfl_momentum", 0.0),
    ({}, "optimizer", "cfl"),
]

# (configuration, expected exception, message fragment): inapplicable / unknown / removed.
_TVF_REJECTED = [
    ({"flow_sigma": 1.0}, ValueError, "flow_sigma is only used"),
    ({"total_sigma": 0.1}, ValueError, "total_sigma is only used"),
    ({"regularizer": "gaussian", "alpha": 0.5}, ValueError, "alpha is only used"),
    ({"regularizer": "gaussian", "total_alpha": 0.5}, ValueError, "total_alpha is only used"),
    ({"grad_step": 0.2}, ValueError, "only used by optimizer='cfl'"),
    ({"cfl_momentum": 0.5}, ValueError, "only used by optimizer='cfl'"),
    ({"optimizer": "cfl", "optimizer_lr": 1.0}, ValueError, "not used by optimizer='cfl'"),
    ({"optimizer": "cfl", "max_step_norm": 0.3}, ValueError, "not used by optimizer='cfl'"),
    ({"optimizer": "adam", "max_step_norm": 0.3}, ValueError, "only used by optimizer='reg_adam'"),
    ({"regularizer": "gaussain"}, ValueError, "unknown regularizer"),
    ({"optimizer": "sobolev_adam"}, ValueError, "unknown optimizer"),
    ({"multipoint_loss": [1.5]}, ValueError, "multipoint_loss"),
    ({"use_analytical_gradients": True}, ValueError, "n_time_steps == 1"),
    ({"antisymmetric": True}, TypeError, "multipoint_loss"),
    ({"sobolev_alpha": 0.1}, TypeError, "use alpha"),
    ({"similarity_metric": "cc2"}, TypeError, "use syn_metric"),
    ({"interpolator": "linear"}, TypeError, "not used by TVF"),
    ({"fast_smooth": True}, ValueError, "reg_adam' smooths its own step"),
    ({"smooth_every_n": 2}, ValueError, "reg_adam' smooths its own step"),
    ({"no_such_option": 1}, TypeError, "unknown keyword"),
]


class TestTVFParameterSensitivity:
    """syntx.tvf: a parameter it accepts must change the result; one it would ignore must raise."""

    @pytest.fixture(autouse=True, scope="class")
    def images(self, request):
        request.cls.fi, request.cls.mi, request.cls.fl, request.cls.ml = _get_r16_images()

    @pytest.mark.parametrize("base,param,value", _TVF_EFFECTIVE,
                             ids=[f"{p}={v}|{b}" for b, p, v in _TVF_EFFECTIVE])
    def test_parameter_changes_result(self, base, param, value):
        warp_a = _get_warp(_quick_tvf(self.fi, self.mi, **base))
        warp_b = _get_warp(_quick_tvf(self.fi, self.mi, **{**base, param: value}))
        _assert_different_field(warp_a, warp_b, param, base.get(param, "default"), value, "tvf")

    @pytest.mark.parametrize("cfg,exc,match", _TVF_REJECTED, ids=[str(c) for c, _, _ in _TVF_REJECTED])
    def test_inapplicable_parameter_raises(self, cfg, exc, match):
        with pytest.raises(exc, match=match):
            _quick_tvf(self.fi, self.mi, **cfg)


# =============================================================================
# Cross-Engine Consistency: SyN and TVF Should Produce Different Results
# =============================================================================
class TestCrossEngineDifference:
    """Verify that syntx.syn and syntx.tvf produce meaningfully different results,
    confirming they are distinct algorithms and not accidentally calling each other.
    """

    @pytest.fixture(autouse=True)
    def setup_images(self):
        self.fi, self.mi, self.fl, self.ml = _get_r16_images()

    def test_syn_vs_tvf_different(self):
        """SyN and TVF must produce different displacement fields."""
        warp_syn = _get_warp(_quick_syn(self.fi, self.mi))
        warp_tvf = _get_warp(_quick_tvf(self.fi, self.mi))
        _assert_different_field(warp_syn, warp_tvf, "engine", "syn", "tvf", "cross")
