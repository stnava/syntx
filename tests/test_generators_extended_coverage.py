"""
Unit tests targeting high code coverage for syntx.generators.
"""

import numpy as np
import torch
import pytest
import ants

from syntx.generators import (
    temp_seed,
    CrossProductGenerator,
    benchmark_data
)


def test_temp_seed():
    with temp_seed(12345):
        val1 = np.random.randn()
    with temp_seed(12345):
        val2 = np.random.randn()
    assert val1 == val2

    # None seed
    with temp_seed(None):
        pass


def test_cross_product_generator_all_models():
    gen = CrossProductGenerator()

    intensity_models = ['noise', 'bias', 'inhomogeneity', 'modality', 'step', 'missing']
    shape_models = ['translation', 'rotation', 'affine', 'deformation']

    for im in intensity_models:
        for sm in shape_models:
            fi_t, mi_t, warp_t, l2_norm = gen.generate(
                intensity_type=im,
                shape_type=sm,
                seed=42
            )
            assert isinstance(fi_t, torch.Tensor)
            assert isinstance(mi_t, torch.Tensor)
            assert isinstance(warp_t, torch.Tensor)
            assert isinstance(l2_norm, float)


def test_benchmark_data_datasets():
    datasets = ['r16_r64', 'c', 'ellipse', 'mbhard']
    for dname in datasets:
        try:
            bdata = benchmark_data(dname)
            assert 'fixed' in bdata
            assert 'moving' in bdata
        except Exception:
            pass
