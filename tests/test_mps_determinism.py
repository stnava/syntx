"""
Run-to-run determinism on GPU/MPS.

The stock GPU/MPS backward kernels of grid_sample (input AND, on MPS, grid gradient) and
avg_pool (the LNCC box filter) accumulate with float atomics, so identical calls return slightly
different gradients and iterative registration (syngs, tvf) amplified that into different
results. core.grid.DeterministicGridSample (Metal kernel on MPS, fixed-point PyTorch fallback
elsewhere) and core.losses.box_mean_nd replace them; these tests pin (a) the gradients equal the
stock ones to float rounding, (b) they are bit-reproducible, (c) syngs/tvf repeat exactly on MPS.
Also: RegAdam's opt-in relative eps.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import ants

import syntx
from syntx.core.grid import DeterministicGridSample, _deterministic_grid_sample_backward
from syntx.core.losses import _DeterministicBoxMean, _box_pool
from syntx.core.optimizers import RegAdam

MPS = torch.backends.mps.is_available()
needs_mps = pytest.mark.skipif(not MPS, reason="MPS only")
CASES = [((1, 3, 12, 14, 10), "border"), ((1, 3, 12, 14, 10), "zeros"),
         ((2, 2, 17, 19), "border"), ((2, 1, 17, 19), "zeros")]


def _problem(shape, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    nd = len(shape) - 2
    x = torch.randn(shape, generator=g)
    idg = F.affine_grid(torch.eye(nd, nd + 1).unsqueeze(0).repeat(shape[0], 1, 1), shape, align_corners=True)
    # includes points outside the FOV (clamped / zero-padded) and exact-boundary cases
    grid = (idg + 0.15 * torch.randn(idg.shape, generator=g)).clamp(-1.2, 1.2)
    go = torch.randn(shape[:2] + tuple(grid.shape[1:-1]), generator=g)
    return (x.to(device).requires_grad_(), grid.to(device).requires_grad_(), go.to(device))


def _stock_grads(x, grid, go, pm):
    return torch.autograd.grad(F.grid_sample(x, grid, mode="bilinear", padding_mode=pm, align_corners=True),
                               (x, grid), go)


def _close(a, b, rtol=2e-6):
    return float((a - b).abs().max()) <= rtol * max(1.0, float(b.abs().max()))


@pytest.mark.parametrize("shape,pm", CASES)
def test_fallback_backward_matches_stock_on_cpu(shape, pm):
    x, grid, go = _problem(shape, "cpu")
    ri, rg = _stock_grads(x, grid, go, pm)
    gi, gg = _deterministic_grid_sample_backward(go, x.detach(), grid.detach(), pm)
    assert _close(gi, ri) and _close(gg, rg)


@needs_mps
@pytest.mark.parametrize("shape,pm", CASES)
def test_mps_kernel_matches_stock_and_repeats(shape, pm):
    x, grid, go = _problem(shape, "mps")
    ri, rg = _stock_grads(x, grid, go, pm)
    outs = [torch.autograd.grad(DeterministicGridSample.apply(x, grid, pm), (x, grid), go) for _ in range(3)]
    assert _close(outs[0][0], ri) and _close(outs[0][1], rg)
    for gi, gg in outs[1:]:
        assert torch.equal(gi, outs[0][0]) and torch.equal(gg, outs[0][1])


@needs_mps
def test_mps_kernel_many_contributions_to_one_voxel():
    """Border padding sends every out-of-FOV point to the edge voxels: 20k points into one bin."""
    x = torch.randn(1, 2, 6, 6, 6, device="mps", requires_grad=True)
    grid = torch.full((1, 20, 32, 32, 3), 5.0, device="mps", requires_grad=True)  # all -> corner (5,5,5)
    go = torch.rand(1, 2, 20, 32, 32, device="mps")
    gi, = torch.autograd.grad(DeterministicGridSample.apply(x, grid, "border"), x, go)
    ref = go.sum(dim=(2, 3, 4))
    assert torch.allclose(gi[0, :, 5, 5, 5], ref[0], rtol=1e-5)
    assert float(gi.abs().sum() - gi[0, :, 5, 5, 5].abs().sum()) == 0.0


@needs_mps
def test_mps_kernel_propagates_nan():
    x, grid, go = _problem((1, 1, 8, 8, 8), "mps")
    go[0, 0, 1, 1, 1] = float("nan")
    gi, _ = torch.autograd.grad(DeterministicGridSample.apply(x, grid, "zeros"), (x, grid), go)
    assert torch.isnan(gi).all()


@pytest.mark.parametrize("shape,k", [((1, 1, 12, 13, 11), 5), ((2, 1, 15, 17), 3), ((1, 1, 9, 10, 11), 9)])
def test_box_mean_backward_matches_stock(shape, k):
    dev = "mps" if MPS else "cpu"
    x = torch.randn(shape, device=dev, requires_grad=True)
    g = torch.randn(shape, device=dev)
    ref, = torch.autograd.grad(_box_pool(x, k, False), x, g)
    outs = [torch.autograd.grad(_DeterministicBoxMean.apply(x, k), x, g)[0] for _ in range(3)]
    assert _close(outs[0], ref)
    assert all(torch.equal(o, outs[0]) for o in outs[1:])


def _one_step(eps_rel, grad):
    p = torch.zeros_like(grad, requires_grad=True)
    opt = RegAdam([p], lr=1.0, regularizer="none", gaussian_sigma=0.0, max_step_norm=1e9,
                  eps_rel=eps_rel)  # no smoothing: isolates the Adam denominator
    p.grad = grad.clone()
    opt.step()
    return p.detach()


def test_relative_eps_keeps_noise_components_small():
    g = torch.full((1, 1, 16, 16), 1e-7)       # "background": noise-sized gradient
    g[..., 8, 8] = 1.0                          # one real signal
    old = _one_step(0.0, g)
    new = _one_step(1e-2, g)
    # absolute eps (default): the 1e-7 components get a step as large as the real signal
    assert float(old[..., 0, 0].abs()) > 0.5 * float(old[..., 8, 8].abs())
    # relative eps: they stay ~1e-5 of the signal's step
    assert float(new[..., 0, 0].abs()) < 1e-3 * float(new[..., 8, 8].abs())
    assert float(new[..., 8, 8].abs()) > 0.5          # the real signal still moves


@pytest.mark.parametrize("reg", ["dsti", "dsti1"])
def test_regadam_spectral_modes_use_their_own_operator(reg):
    """gaussian_sigma must not hijack the dsti / dsti1 branches (it used to: any mode other than
    'sobolev' with gaussian_sigma > 0 was Gaussian-smoothed)."""
    g = torch.randn(1, 12, 12, 12, 3, generator=torch.Generator().manual_seed(0))
    steps = []
    for gs in (0.0, 1.5, 4.0):
        p = torch.zeros_like(g, requires_grad=True)
        opt = RegAdam([p], lr=1.0, regularizer=reg, sobolev_alpha=0.5, gaussian_sigma=gs, max_step_norm=1e9)
        p.grad = g.clone()
        opt.step()
        steps.append(p.detach().clone())
    assert torch.equal(steps[0], steps[1]) and torch.equal(steps[1], steps[2])


def _r16_r64():
    return ants.image_read(ants.get_ants_data("r16")), ants.image_read(ants.get_ants_data("r64"))


@needs_mps
def test_syngs_repeats_exactly_on_mps():
    fi, mi = _r16_r64()
    outs = [syntx.syngs(fixed=fi, moving=mi, initial_transform="identity",
                        reg_iterations=[20, 10, 5, 2], device="mps", verbose=False)["warpedmovout"].numpy()
            for _ in range(2)]
    # was 0.06-0.22 (0-255 intensity units) with the stock kernels
    assert np.array_equal(outs[0], outs[1])


@needs_mps
def test_tvf_repeats_exactly_on_mps():
    fi, mi = _r16_r64()
    outs = [syntx.tvf(fixed=fi, moving=mi, initial_transform="identity",
                      reg_iterations=[20, 10, 5], device="mps", verbose=False)["warpedmovout"].numpy()
            for _ in range(2)]
    assert np.array_equal(outs[0], outs[1])
