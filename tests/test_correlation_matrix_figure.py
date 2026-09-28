"""Tests for syntx.viz.render_correlation_matrix_figure."""

import numpy as np

from syntx.viz import render_correlation_matrix_figure


def test_render_correlation_matrix_figure_saves_file(tmp_path):
    rng = np.random.default_rng(0)
    mat = np.corrcoef(rng.normal(size=(50, 10)), rowvar=False)
    out = str(tmp_path / "corr.png")
    result = render_correlation_matrix_figure(mat, title="test matrix", save_path=out)
    assert result == out
    import os

    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_render_correlation_matrix_figure_returns_figure_without_save_path():
    rng = np.random.default_rng(1)
    mat = np.corrcoef(rng.normal(size=(50, 6)), rowvar=False)
    fig = render_correlation_matrix_figure(mat, title="no save")
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_render_correlation_matrix_figure_with_labels_small_matrix(tmp_path):
    rng = np.random.default_rng(2)
    n = 5
    mat = np.corrcoef(rng.normal(size=(50, n)), rowvar=False)
    labels = [f"ROI{i}" for i in range(n)]
    out = str(tmp_path / "corr_labeled.png")
    render_correlation_matrix_figure(mat, roi_labels=labels, save_path=out)
    import os

    assert os.path.exists(out)


def test_render_correlation_matrix_figure_large_matrix_omits_labels(tmp_path):
    rng = np.random.default_rng(3)
    n = 60  # above the 40-ROI label threshold
    mat = np.corrcoef(rng.normal(size=(200, n)), rowvar=False)
    labels = [f"ROI{i}" for i in range(n)]
    out = str(tmp_path / "corr_large.png")
    # Must not raise despite requesting labels on a matrix larger than the label threshold.
    render_correlation_matrix_figure(mat, roi_labels=labels, save_path=out)
    import os

    assert os.path.exists(out)
