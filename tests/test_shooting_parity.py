#!/usr/bin/env python3
"""
Explicit Shooting Parity Test File:
Verifies PyTorch and JAX GeodesicShootingModel parity.
"""
import pytest
from tests.test_shooting import (
    test_epdiff_advection_parity,
    test_shooting_forward_loss_parity,
    test_shooting_warp_parity,
    test_shooting_model_fit_parity
)

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
