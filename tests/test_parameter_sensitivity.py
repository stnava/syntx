# Energy Gap Analysis and Parameter Sensitivity Tests for syntx.syn
#
# This module verifies that changing input parameters actually changes outputs,
# catching bugs like the use_analytical_gradients parameter being silently ignored.

import pytest
import numpy as np
import torch
import ants

from syntx.syn import (
    SyNTo,
    separable_gaussian_filter,
    registration,
)


def _get_r16_images():
    """Get the r16/r64 benchmark pair as ANTsImages."""
    import syntx
    res = syntx.benchmark_data('r16')
    return res['fixed'], res['moving']


def _quick_syn(fixed, moving, **kwargs):
    """Run a minimal 1-iteration syntx.syn and return the fwd warp displacement field."""
    import syntx
    defaults = dict(
        flow_sigma=3.0,
        grad_step=0.25,
        total_sigma=0.0,
        reg_iterations=[1],
        syn_sampling=2,
    )
    defaults.update(kwargs)
    reg = syntx.syn(fixed, moving, **defaults)
    warp_img = ants.image_read(reg['fwdtransforms'][0])
    return warp_img.numpy()


class TestParameterSensitivity:
    """Verify that changing key parameters produces measurably different outputs.
    
    Each test runs two 1-iteration registrations with different parameter values
    and asserts the output displacement fields are NOT identical. This catches
    bugs where parameters are silently ignored (e.g., the use_analytical_gradients
    bug where kwargs.get() overrode the explicit function argument).
    """

    @pytest.fixture(autouse=True)
    def setup_images(self):
        self.fi, self.mi = _get_r16_images()

    def test_flow_sigma_changes_output(self):
        """Different flow_sigma values must produce different displacement fields."""
        warp_a = _quick_syn(self.fi, self.mi, flow_sigma=1.0)
        warp_b = _quick_syn(self.fi, self.mi, flow_sigma=5.0)
        max_diff = np.abs(warp_a - warp_b).max()
        assert max_diff > 1e-6, (
            f"flow_sigma=1.0 vs 5.0 produced identical fields (max_diff={max_diff}). "
            "Parameter is being silently ignored."
        )

    def test_grad_step_changes_output(self):
        """Different grad_step values must produce different displacement fields."""
        warp_a = _quick_syn(self.fi, self.mi, grad_step=0.1)
        warp_b = _quick_syn(self.fi, self.mi, grad_step=0.5)
        max_diff = np.abs(warp_a - warp_b).max()
        assert max_diff > 1e-6, (
            f"grad_step=0.1 vs 0.5 produced identical fields (max_diff={max_diff}). "
            "Parameter is being silently ignored."
        )

    def test_use_analytical_gradients_changes_output(self):
        """Analytical vs autograd gradient modes must produce different displacement fields.
        
        This is a regression test for the bug where kwargs.get('use_analytical_gradients', True)
        silently overrode the explicit use_analytical_gradients=False parameter.
        """
        warp_analytical = _quick_syn(self.fi, self.mi, use_analytical_gradients=True)
        warp_autograd = _quick_syn(self.fi, self.mi, use_analytical_gradients=False)
        max_diff = np.abs(warp_analytical - warp_autograd).max()
        assert max_diff > 1e-6, (
            f"use_analytical_gradients=True vs False produced identical fields (max_diff={max_diff}). "
            "The parameter is being silently ignored — this is a known regression."
        )

    def test_syn_sampling_changes_output(self):
        """Different LNCC radius values must produce different displacement fields."""
        warp_a = _quick_syn(self.fi, self.mi, syn_sampling=2)
        warp_b = _quick_syn(self.fi, self.mi, syn_sampling=4)
        max_diff = np.abs(warp_a - warp_b).max()
        assert max_diff > 1e-6, (
            f"syn_sampling=2 vs 4 produced identical fields (max_diff={max_diff}). "
            "Parameter is being silently ignored."
        )

    def test_formulation_changes_output(self):
        """Eulerian vs lagrangian formulation must produce different displacement fields."""
        warp_a = _quick_syn(self.fi, self.mi, formulation='eulerian')
        warp_b = _quick_syn(self.fi, self.mi, formulation='lagrangian')
        max_diff = np.abs(warp_a - warp_b).max()
        assert max_diff > 1e-6, (
            f"formulation='eulerian' vs 'lagrangian' produced identical fields (max_diff={max_diff}). "
            "Parameter is being silently ignored."
        )
