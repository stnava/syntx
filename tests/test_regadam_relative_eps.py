"""
RegAdam's relative eps: near-zero gradient components must not be normalised into
full-size steps (they amplify rounding noise -- on MPS, grid_sample's input-gradient
scatter-add is non-deterministic -- into run-to-run differences).
"""

import numpy as np
import pytest
import torch
import ants

import syntx
from syntx.core.optimizers import RegAdam


def _one_step(eps_rel, grad):
    p = torch.zeros_like(grad, requires_grad=True)
    opt = RegAdam([p], lr=1.0, regularizer="none", gaussian_sigma=0.0, max_step_norm=1e9,
                  eps_rel=eps_rel)  # no smoothing: isolates the Adam denominator
    p.grad = grad.clone()
    opt.step()
    return p.detach()


def test_relative_eps_keeps_noise_components_small():
    g = torch.full((1, 1, 16, 16), 1e-7)       # "background": rounding-noise-sized gradient
    g[..., 8, 8] = 1.0                          # one real signal
    old = _one_step(0.0, g)
    new = _one_step(1e-2, g)
    # absolute eps: the 1e-7 components get a step as large as the real signal
    assert float(old[..., 0, 0].abs()) > 0.5 * float(old[..., 8, 8].abs())
    # relative eps: they stay ~1e-5 of the signal's step
    assert float(new[..., 0, 0].abs()) < 1e-3 * float(new[..., 8, 8].abs())
    assert float(new[..., 8, 8].abs()) > 0.5          # the real signal still moves


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS only")
def test_syngs_repeats_on_mps():
    fi = ants.image_read(ants.get_ants_data("r16"))
    mi = ants.image_read(ants.get_ants_data("r64"))
    outs = [syntx.syngs(fixed=fi, moving=mi, initial_transform="identity",
                        reg_iterations=[20, 10, 5, 2], device="mps", verbose=False)["warpedmovout"].numpy()
            for _ in range(2)]
    # was 0.06-0.22 (intensity units, 0-255 image) with the absolute eps
    assert np.abs(outs[1] - outs[0]).max() < 1e-2
