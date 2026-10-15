"""Unit tests for syntx.curves visualization and reporting across dimensions."""

from pathlib import Path
import numpy as np
import pytest

from syntx.curves.matching import match_curves
from syntx.curves.report import generate_curve_matching_html_report
from syntx.curves.viz import compute_dec_colors, fig_to_base64, plot_curve_match


def test_compute_dec_colors():
    t = np.linspace(0, 1, 50)[:, None]
    pts = np.hstack([t * 30.0, np.zeros_like(t), np.zeros_like(t)])
    dec = compute_dec_colors(pts, mode="open")
    assert dec.shape == (50, 3)
    # Pure X tangent should be pure Red [1, 0, 0]
    assert np.allclose(dec, [1.0, 0.0, 0.0], atol=1e-3)


def test_plot_curve_match_dimensions(tmp_path: Path):
    # Test 2-D match plot
    t = np.linspace(0, 1, 60)[:, None]
    p_2d = np.hstack([t * 50.0, 10.0 * np.sin(np.pi * t)])
    q_2d = p_2d.copy()
    q_2d[:, 1] += 3.0 * np.sin(2 * np.pi * t[:, 0])
    m_2d = match_curves(p_2d, q_2d, mode="open", n_points=40, rigid=False)

    fig_2d = plot_curve_match(m_2d, title="2-D Planar Curve Match")
    out_2d = tmp_path / "match_2d.png"
    fig_2d.savefig(out_2d)
    assert out_2d.exists() and out_2d.stat().st_size > 10_000

    # Test 3-D match plot
    p_3d = np.hstack([t * 50.0, 10.0 * np.sin(np.pi * t), 5.0 * np.cos(np.pi * t)])
    m_3d = match_curves(p_3d, p_3d, mode="open", n_points=40, rigid=False)
    fig_3d = plot_curve_match(m_3d, title="3-D Space Curve Match")
    out_3d = tmp_path / "match_3d.png"
    fig_3d.savefig(out_3d)
    assert out_3d.exists() and out_3d.stat().st_size > 10_000

    # Test 4-D match plot (PCA projection)
    p_4d = np.hstack([t * 40.0, 8.0 * np.sin(np.pi * t), 5.0 * np.cos(np.pi * t), 3.0 * t])
    m_4d = match_curves(p_4d, p_4d, mode="open", n_points=40, rigid=False)
    fig_4d = plot_curve_match(m_4d, title="4-D Hypercurve Match")
    out_4d = tmp_path / "match_4d.png"
    fig_4d.savefig(out_4d)
    assert out_4d.exists() and out_4d.stat().st_size > 10_000


def test_html_report_generation(tmp_path: Path):
    t = np.linspace(0, 1, 60)[:, None]
    p = np.hstack([t * 60.0, 12.0 * np.sin(np.pi * t), 6.0 * np.cos(np.pi * t)])
    q = p.copy()
    q[:, 1] += 4.0 * np.sin(np.pi * t[:, 0] ** 0.5)

    m = match_curves(p, q, mode="open", n_points=45, rigid=False, bidirectional=True)
    report_file = tmp_path / "curve_matching_report.html"

    out_path = generate_curve_matching_html_report(
        report_file,
        matches=[("Synthetic S-Curve Match", m)],
        title="N-D Curve Matching Report",
    )
    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 40_000
    text = Path(out_path).read_text(encoding="utf-8")
    assert "Synthetic S-Curve Match" in text
    assert "data:image/png;base64," in text
    assert "Elastic RMS Distance" in text
