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
