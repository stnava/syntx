"""Tests for syntx.viz.render_carpet_plot_figure."""

import os

import numpy as np

from syntx.viz import render_carpet_plot_figure


def test_carpet_plot_saves_file(tmp_path):
    rng = np.random.default_rng(0)
    ts = rng.normal(1000, 20, size=(40, 200))
    out = str(tmp_path / "carpet.png")
    result = render_carpet_plot_figure(ts, title="test", save_path=out)
    assert result == out
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_carpet_plot_with_tissue_labels_and_fd(tmp_path):
    rng = np.random.default_rng(1)
    ts = rng.normal(1000, 20, size=(30, 150))
    tissue = np.concatenate([np.ones(100), np.full(50, 2)])
    fd = np.abs(rng.normal(0.1, 0.05, size=30))
    out = str(tmp_path / "carpet_full.png")
    render_carpet_plot_figure(ts, tissue_labels=tissue, fd=fd, save_path=out)
    assert os.path.exists(out)


def test_carpet_plot_without_save_path_returns_figure():
    rng = np.random.default_rng(2)
    ts = rng.normal(1000, 20, size=(20, 50))
    fig = render_carpet_plot_figure(ts)
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)
