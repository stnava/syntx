"""Tests for syntx.viz.modality_report, syntx.viz.qc_sections, and the new triplanar
figure functions (render_label_overlay_figure, render_checkerboard_figure) -- centralized
here after being independently prototyped (and having real bugs found and fixed) in a
downstream package (antsxfunctional), per the goal of consolidating shared antsx*
reporting/visualization infrastructure into syntx.
"""

import json
import os

import numpy as np
import pytest


def test_write_modality_report_title_reflects_caller_supplied_modality(tmp_path):
    from syntx.viz import kpi_card, write_modality_report

    out = write_modality_report(
        str(tmp_path / "report.html"),
        modality_title="Neuromelanin-Sensitive MRI",
        session_label="sub-01",
        kpis_html=kpi_card("NM-CNR", 0.076, "mean left/right"),
        figure_paths={},
    )
    html = open(out).read()
    assert "Neuromelanin-Sensitive MRI" in html
    assert "<title>Neuromelanin-Sensitive MRI report" in html


def test_write_modality_report_qc_sections_render_as_cards_not_a_table(tmp_path):
    from syntx.viz import write_modality_report

    out = write_modality_report(
        str(tmp_path / "report.html"),
        modality_title="PET",
        session_label="sub-01",
        kpis_html="",
        figure_paths={},
        qc_sections={
            "Registration": {"mutual_information": -0.4},
            "Motion": {"fd_mean": {"value": 0.3, "status": "warn", "note": "test"}},
        },
    )
    html = open(out).read()
    assert "Quality Control" in html
    assert "mutual information" in html
    assert "<table" not in html.split("Quality Control")[1].split("Processing Steps")[0]


def test_write_modality_report_caveats_go_to_json_sidecar(tmp_path):
    from syntx.viz import write_modality_report

    out = write_modality_report(
        str(tmp_path / "report.html"),
        modality_title="PET",
        session_label="sub-01",
        kpis_html="",
        figure_paths={},
        caveats=["A long methodology note.", "Second note."],
    )
    html = open(out).read()
    assert "A long methodology note" not in html
    assert "2 methodology notes" in html
    notes = json.loads((tmp_path / "report_notes.json").read_text())
    assert notes["caveats"] == ["A long methodology note.", "Second note."]


def test_write_modality_report_highlight_figure_after_kpis(tmp_path):
    from syntx.viz import write_modality_report

    fig_path = tmp_path / "eq.png"
    fig_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 20)
    out = write_modality_report(
        str(tmp_path / "report.html"),
        modality_title="Perfusion",
        session_label="sub-01",
        kpis_html='<div id="kpi-marker"></div>',
        figure_paths={},
        highlight_figure=str(fig_path),
        highlight_caption="Regression equation",
    )
    html = open(out).read()
    assert html.find("kpi-marker") < html.find("Regression equation")


def test_equation_figure_renders_mathtext(tmp_path):
    from syntx.viz import equation_figure

    out = equation_figure(
        r"$Y = \beta_0 + \beta_1 X$", ["term definitions"], str(tmp_path / "eq.png"), title="Model",
    )
    assert os.path.exists(out)


def test_registration_qc_section_grades_dice():
    from syntx.viz import registration_qc_section

    section = registration_qc_section(-0.4, dice=0.83, dice_note="test")
    assert section["dice_overlap"]["status"] == "ok"


def test_motion_qc_section_grades_fd():
    from syntx.viz import motion_qc_section

    section = motion_qc_section(fd_mean=0.6, fd_max=1.2, motion_corrected=True)
    assert section["fd_mean"]["status"] == "fail"


def test_ai_model_qc_section_selects_antstorch_engine():
    from syntx.contract import ProvenanceEntry
    from syntx.viz import ai_model_qc_section

    provenance = [
        ProvenanceEntry(step="tissue_priors", engine="antstorch", device="cpu", seconds=1.0, caller="x"),
        ProvenanceEntry(step="bandpass", engine="torch", device="cpu", seconds=0.5, caller="x"),
    ]
    section = ai_model_qc_section(provenance)
    assert "tissue_priors" in section
    assert "bandpass" not in section


def test_render_label_overlay_figure_shows_background_not_flat_color(tmp_path):
    """Real bug this replaces (found in a downstream antsx* package): a label image
    rendered alone with no background maps rank-0 through the same colormap as real
    labels, producing a flat saturated color -- confirmed hard to interpret. Background
    must show real grayscale variation, not a uniform color."""
    ants = pytest.importorskip("ants")
    from syntx.viz import render_label_overlay_figure

    rng = np.random.default_rng(0)
    bg = ants.from_numpy(rng.uniform(20, 200, size=(32, 32, 32)).astype(np.float32))
    label_arr = np.zeros((32, 32, 32), dtype=np.float32)
    label_arr[14:18, 14:18, 14:18] = 7
    labels = ants.from_numpy(label_arr)

    save_path = str(tmp_path / "overlay.png")
    render_label_overlay_figure(bg, labels, "test", save_path)
    assert os.path.exists(save_path)


def test_render_label_overlay_figure_omits_legend_when_too_many_labels(tmp_path):
    ants = pytest.importorskip("ants")
    from syntx.viz import render_label_overlay_figure

    bg = ants.from_numpy(np.full((32, 32, 32), 100.0, dtype=np.float32))
    label_arr = np.zeros((32, 32, 32), dtype=np.float32)
    for i in range(25):
        label_arr[i, i, i] = float(i + 1)
    labels = ants.from_numpy(label_arr)

    save_path = str(tmp_path / "overlay_nolegend.png")
    render_label_overlay_figure(bg, labels, "test", save_path, max_legend_labels=20)
    assert os.path.exists(save_path)


def test_render_checkerboard_figure_rejects_mismatched_grid():
    ants = pytest.importorskip("ants")
    from syntx.viz import render_checkerboard_figure

    a = ants.from_numpy(np.zeros((32, 32, 32), dtype=np.float32))
    b = ants.from_numpy(np.zeros((16, 16, 16), dtype=np.float32))
    with pytest.raises(ValueError):
        render_checkerboard_figure(a, b, "test", "/tmp/unused.png")


def test_render_checkerboard_figure_restricts_to_requested_views(tmp_path):
    ants = pytest.importorskip("ants")
    from syntx.viz import render_checkerboard_figure

    a = ants.from_numpy(np.random.default_rng(0).uniform(0, 100, size=(20, 20, 20)).astype(np.float32))
    b = ants.from_numpy(np.random.default_rng(1).uniform(0, 100, size=(20, 20, 20)).astype(np.float32))
    save_path = str(tmp_path / "checker.png")
    render_checkerboard_figure(a, b, "test", save_path, views=("axial",))
    assert os.path.exists(save_path)


def test_write_modality_report_brand_prefixes_title_and_h1(tmp_path):
    from syntx.viz import write_modality_report

    out = write_modality_report(
        str(tmp_path / "report.html"),
        modality_title="Perfusion / ASL",
        session_label="sub-01",
        kpis_html="",
        figure_paths={},
        brand="antsxfunctional",
    )
    html = open(out).read()
    assert "<title>antsxfunctional Perfusion / ASL report" in html
    assert "antsxfunctional — Perfusion / ASL Report" in html


def test_write_modality_report_no_brand_by_default(tmp_path):
    from syntx.viz import write_modality_report

    out = write_modality_report(
        str(tmp_path / "report.html"),
        modality_title="Perfusion / ASL",
        session_label="sub-01",
        kpis_html="",
        figure_paths={},
    )
    html = open(out).read()
    assert "<title>Perfusion / ASL report" in html


def test_write_modality_report_header_badges_and_open_helper(tmp_path):
    from syntx.viz import write_modality_report

    out = write_modality_report(
        str(tmp_path / "report.html"),
        modality_title="DTI",
        session_label="sub-01",
        kpis_html="",
        figure_paths={},
        header_badges=[("Subject", "sub-01"), ("Session", "ses-01"), ("Engine", "syntx")],
    )
    html = open(out).read()
    assert "Subject:" in html
    assert "sub-01" in html
    assert "Engine:" in html
    assert "syntx" in html
    assert "open" in html


def test_write_modality_report_stage_sections_and_artifacts(tmp_path):
    from syntx.viz import write_modality_report

    fig_png = tmp_path / "fig.png"
    fig_png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 20)
    fig_jpg = tmp_path / "fig.jpg"
    fig_jpg.write_bytes(b"\xff\xd8\xff" + b"0" * 20)

    stages = [
        {
            "title": "Stage 1: Motion Correction",
            "badge": "Threshold 0.5 mm",
            "badge_status": "ok",
            "description": "Rigid realignment across 4D volumes.",
            "figures": [(str(fig_png), "Figure 1: Motion Trace")],
        },
        {
            "title": "Stage 4: Dewarping",
            "badge": "Multi-PE SyN",
            "badge_status": "ok",
            "description": "Diffeomorphic distortion correction with CC=0.93.",
            "figures": [(str(fig_jpg), "Figure 4: Reverse-PE Alignment")],
        },
        {
            "title": "Stage 6: Tractography",
            "badge": "Empty Test",
            "badge_status": None,
            "description": "Narrative with no figures renders header and text.",
            "figures": [],
        },
    ]

    artifacts = {
        "dwi_dewarped": "/path/to/dwi_dewarped.nii.gz",
        "fa": "/path/to/dtifa.nii.gz",
    }

    provenance = [
        {"step": "dewarp", "engine": "syntx", "device": "cpu", "seconds": 12.3, "extra": {"ncc": 0.93}},
    ]

    out = write_modality_report(
        str(tmp_path / "report.html"),
        modality_title="Diffusion MRI",
        session_label="sub-01",
        kpis_html="",
        figure_paths={},
        stage_sections=stages,
        artifacts=artifacts,
        provenance=provenance,
        provenance_json=True,
    )
    html = open(out).read()

    # Stages
    assert "Stage 1: Motion Correction" in html
    assert "Threshold 0.5 mm" in html
    assert "Figure 1: Motion Trace" in html
    assert "data:image/png;base64," in html
    assert "Stage 4: Dewarping" in html
    assert "Multi-PE SyN" in html
    assert "Figure 4: Reverse-PE Alignment" in html
    assert "data:image/jpeg;base64," in html
    assert "Stage 6: Tractography" in html
    assert "Empty Test" in html

    # Artifacts
    assert "Generated Pipeline Artifacts" in html
    assert "dwi_dewarped" in html
    assert "dwi_dewarped.nii.gz" in html
    assert "dtifa.nii.gz" in html

    # Provenance JSON
    assert "Complete Execution Provenance" in html
    assert "&quot;step&quot;: &quot;dewarp&quot;" in html



def test_equations_figure_saves_file(tmp_path):
    from syntx.viz import equations_figure

    eqs = [
        {"title": "A", "equation": r"$y=mx+b$", "definitions": ["m: slope"]},
        {"title": "B", "equation": r"$E=mc^2$" + "\n" + r"$F=ma$", "definitions": []},
    ]
    out = str(tmp_path / "eqs.png")
    result = equations_figure(eqs, out, header_title="Test Equations")
    assert result == out
    import os

    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_equations_figure_single_entry(tmp_path):
    from syntx.viz import equations_figure

    out = str(tmp_path / "eq1.png")
    equations_figure([{"title": "Only", "equation": r"$x^2$", "definitions": []}], out)
    import os

    assert os.path.exists(out)
