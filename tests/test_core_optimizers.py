import pytest
import torch
import numpy as np

from syntx.core.optimizers import (
    LARS,
    get_cfl_max_norm,
    compute_cfl_step,
    check_convergence,
)


def test_lars_optimizer():
    p = torch.nn.Parameter(torch.ones(10, requires_grad=True))
    optimizer = LARS([p], lr=0.1, trust_coefficient=0.05)

    loss = torch.sum(p ** 2)
    loss.backward()

    optimizer.step()
    # Parameters should decrease
    assert torch.all(p.data < 1.0)


def test_get_cfl_max_norm():
    velocity = torch.zeros(1, 10, 10, 2)
    velocity[0, 5, 5] = torch.tensor([3.0, 4.0])  # norm = 5.0
    spacing = [1.0, 1.0]

    max_norm = get_cfl_max_norm(velocity, spacing)
    assert np.isclose(max_norm, 5.0)


def test_compute_cfl_step():
    kwargs = {'grad_step': 0.25}
    step = compute_cfl_step(kwargs, shrink_ratio=4.0)
    assert np.isclose(step, 0.50)


def test_check_convergence():
    # Strictly decreasing losses -> not converged
    decreasing = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    assert not check_convergence(decreasing, window_size=5)

    # Flat losses -> converged
    flat = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert check_convergence(flat, window_size=5)
