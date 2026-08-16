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
    """Run a minimal syntx.tvf and return the registration result dict."""
    import syntx
    defaults = dict(
        flow_sigma=0.5,
        grad_step=0.25,
        total_sigma=0.0,
        reg_iterations=[10],
        syn_sampling=2,
        affine_iterations=0,
        n_time_steps=3,
        constant_speed=False,
        use_analytical_gradients=False,
        verbose=0,
    )
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
# TVF Parameter Sensitivity Tests
# =============================================================================
class TestTVFParameterSensitivity:
    """Verify that changing key parameters produces measurably different outputs
    for syntx.tvf (Time-Varying Velocity Field registration).
    """

    @pytest.fixture(autouse=True)
    def setup_images(self):
        self.fi, self.mi, self.fl, self.ml = _get_r16_images()

    # --- Continuous parameters: test both field difference AND Dice difference ---

    def test_flow_sigma_changes_performance(self):
        """Different flow_sigma (fluid regularization) must change Dice.

        Regression test for the double-sqrt bug where fit() applied sqrt() on a value
        already sqrt()'d by tvf_registration(), making sigma quartic-rooted.
        """
        reg_a = _quick_tvf(self.fi, self.mi, flow_sigma=0.1)
        reg_b = _quick_tvf(self.fi, self.mi, flow_sigma=3.0)
        _assert_different_field(_get_warp(reg_a), _get_warp(reg_b), "flow_sigma", 0.1, 3.0, "tvf")
        dice_a = _compute_dice(reg_a, self.fl, self.ml, self.fi, self.mi)
        dice_b = _compute_dice(reg_b, self.fl, self.ml, self.fi, self.mi)
        _assert_different_dice(dice_a, dice_b, "flow_sigma", 0.1, 3.0, "tvf")

    def test_total_sigma_changes_performance(self):
        """Different total_sigma (elastic regularization) must change Dice."""
        reg_a = _quick_tvf(self.fi, self.mi, total_sigma=0.0, reg_iterations=[30])
        reg_b = _quick_tvf(self.fi, self.mi, total_sigma=2.0, reg_iterations=[30])
        _assert_different_field(_get_warp(reg_a), _get_warp(reg_b), "total_sigma", 0.0, 2.0, "tvf")
        dice_a = _compute_dice(reg_a, self.fl, self.ml, self.fi, self.mi)
        dice_b = _compute_dice(reg_b, self.fl, self.ml, self.fi, self.mi)
        _assert_different_dice(dice_a, dice_b, "total_sigma", 0.0, 2.0, "tvf")

    def test_grad_step_changes_performance(self):
        """Different grad_step (CFL step size) must change Dice."""
        reg_a = _quick_tvf(self.fi, self.mi, grad_step=0.02)
        reg_b = _quick_tvf(self.fi, self.mi, grad_step=0.50)
        _assert_different_field(_get_warp(reg_a), _get_warp(reg_b), "grad_step", 0.02, 0.50, "tvf")
        dice_a = _compute_dice(reg_a, self.fl, self.ml, self.fi, self.mi)
        dice_b = _compute_dice(reg_b, self.fl, self.ml, self.fi, self.mi)
        _assert_different_dice(dice_a, dice_b, "grad_step", 0.02, 0.50, "tvf")

    def test_syn_sampling_changes_performance(self):
        """Different LNCC radius must change Dice."""
        reg_a = _quick_tvf(self.fi, self.mi, syn_sampling=1)
        reg_b = _quick_tvf(self.fi, self.mi, syn_sampling=4)
        _assert_different_field(_get_warp(reg_a), _get_warp(reg_b), "syn_sampling", 1, 4, "tvf")
        dice_a = _compute_dice(reg_a, self.fl, self.ml, self.fi, self.mi)
        dice_b = _compute_dice(reg_b, self.fl, self.ml, self.fi, self.mi)
        _assert_different_dice(dice_a, dice_b, "syn_sampling", 1, 4, "tvf")

    def test_cfl_momentum_changes_performance(self):
        """Different cfl_momentum values must change displacement field."""
        warp_a = _get_warp(_quick_tvf(self.fi, self.mi, cfl_momentum=0.0))
        warp_b = _get_warp(_quick_tvf(self.fi, self.mi, cfl_momentum=0.95))
        _assert_different_field(warp_a, warp_b, "cfl_momentum", 0.0, 0.95, "tvf")

    def test_constant_speed_relaxation_changes_performance(self):
        """Different constant_speed_relaxation must change Dice."""
        reg_a = _quick_tvf(self.fi, self.mi, constant_speed=True, constant_speed_relaxation=0.01, reg_iterations=[30])
        reg_b = _quick_tvf(self.fi, self.mi, constant_speed=True, constant_speed_relaxation=0.50, reg_iterations=[30])
        _assert_different_field(_get_warp(reg_a), _get_warp(reg_b), "constant_speed_relaxation", 0.01, 0.50, "tvf")
        dice_a = _compute_dice(reg_a, self.fl, self.ml, self.fi, self.mi)
        dice_b = _compute_dice(reg_b, self.fl, self.ml, self.fi, self.mi)
        _assert_different_dice(dice_a, dice_b, "constant_speed_relaxation", 0.01, 0.50, "tvf")

    def test_cfl_max_changes_performance(self):
        """Different cfl_max values must produce different fields when velocity is clamped."""
        warp_a = _get_warp(_quick_tvf(self.fi, self.mi, cfl_max=0.001, grad_step=0.50))
        warp_b = _get_warp(_quick_tvf(self.fi, self.mi, cfl_max=5.0, grad_step=0.50))
        _assert_different_field(warp_a, warp_b, "cfl_max", 0.001, 5.0, "tvf")

    # --- Discrete/categorical parameters: test field difference ---

    def test_regularizer_gaussian_vs_dsti(self):
        """Gaussian vs DSTI regularizer must produce different fields."""
        warp_a = _get_warp(_quick_tvf(self.fi, self.mi, regularizer='gaussian'))
        warp_b = _get_warp(_quick_tvf(self.fi, self.mi, regularizer='dsti'))
        _assert_different_field(warp_a, warp_b, "regularizer", "gaussian", "dsti", "tvf")

    def test_regularizer_gaussian_vs_sobolev(self):
        """Gaussian vs Sobolev regularizer must produce different fields."""
        warp_a = _get_warp(_quick_tvf(self.fi, self.mi, regularizer='gaussian'))
        warp_b = _get_warp(_quick_tvf(self.fi, self.mi, regularizer='sobolev'))
        _assert_different_field(warp_a, warp_b, "regularizer", "gaussian", "sobolev", "tvf")

    def test_use_analytical_gradients_changes_output(self):
        """Analytical vs autograd gradient modes must produce different fields or raise NotImplementedError when n_time_steps > 1."""
        try:
            res_a = _quick_tvf(self.fi, self.mi, use_analytical_gradients=True)
            res_b = _quick_tvf(self.fi, self.mi, use_analytical_gradients=False)
            _assert_different_field(_get_warp(res_a), _get_warp(res_b), "use_analytical_gradients", True, False, "tvf")
        except NotImplementedError:
            pytest.skip("Analytical gradients raise NotImplementedError for n_time_steps > 1")

    def test_antisymmetric_changes_output(self):
        """antisymmetric=True vs False must produce different fields."""
        warp_a = _get_warp(_quick_tvf(self.fi, self.mi, antisymmetric=True))
        warp_b = _get_warp(_quick_tvf(self.fi, self.mi, antisymmetric=False))
        _assert_different_field(warp_a, warp_b, "antisymmetric", True, False, "tvf")

    def test_constant_speed_changes_output(self):
        """constant_speed=True vs False must produce different fields."""
        warp_a = _get_warp(_quick_tvf(self.fi, self.mi, constant_speed=True))
        warp_b = _get_warp(_quick_tvf(self.fi, self.mi, constant_speed=False))
        _assert_different_field(warp_a, warp_b, "constant_speed", True, False, "tvf")

    def test_multipoint_loss_changes_output(self):
        """Different multipoint_loss evaluation points must produce different fields."""
        warp_a = _get_warp(_quick_tvf(self.fi, self.mi, multipoint_loss=[0.5], antisymmetric=True))
        warp_b = _get_warp(_quick_tvf(self.fi, self.mi, multipoint_loss=[0.0, 0.5, 1.0], antisymmetric=True))
        _assert_different_field(warp_a, warp_b, "multipoint_loss", [0.5], [0.0, 0.5, 1.0], "tvf")

    def test_n_time_steps_changes_output(self):
        """Different n_time_steps must produce different fields."""
        warp_a = _get_warp(_quick_tvf(self.fi, self.mi, n_time_steps=2))
        warp_b = _get_warp(_quick_tvf(self.fi, self.mi, n_time_steps=6))
        _assert_different_field(warp_a, warp_b, "n_time_steps", 2, 6, "tvf")

    def test_fast_smooth_changes_output(self):
        """fast_smooth=True vs False must produce different fields."""
        warp_a = _get_warp(_quick_tvf(self.fi, self.mi, fast_smooth=True))
        warp_b = _get_warp(_quick_tvf(self.fi, self.mi, fast_smooth=False))
        _assert_different_field(warp_a, warp_b, "fast_smooth", True, False, "tvf")


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
