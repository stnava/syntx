"""Tests for syntx.viz.render_motion_parameters_figure."""

import os

import numpy as np

from syntx.viz import render_motion_parameters_figure


def test_motion_parameters_figure_saves_file(tmp_path):
    rng = np.random.default_rng(0)
    trans = rng.normal(0, 0.3, size=(30, 3))
    rot = rng.normal(0, 0.2, size=(30, 3))
    out = str(tmp_path / "motion.png")
    result = render_motion_parameters_figure(trans, rot, save_path=out)
    assert result == out
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_motion_parameters_figure_with_fd(tmp_path):
    rng = np.random.default_rng(1)
    trans = rng.normal(0, 0.3, size=(20, 3))
    rot = rng.normal(0, 0.2, size=(20, 3))
    fd = np.abs(rng.normal(0.1, 0.05, size=20))
    out = str(tmp_path / "motion_fd.png")
    render_motion_parameters_figure(trans, rot, fd=fd, save_path=out)
    assert os.path.exists(out)


def test_motion_parameters_figure_without_save_path_returns_figure():
    rng = np.random.default_rng(2)
    trans = rng.normal(0, 0.3, size=(10, 3))
    rot = rng.normal(0, 0.2, size=(10, 3))
    fig = render_motion_parameters_figure(trans, rot)
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)
