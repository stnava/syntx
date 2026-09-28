"""syntx.viz.modality_report — generic, any-modality self-contained HTML QC report shell.

Unlike ``create_registration_report``/``create_population_benchmark_report`` (which are
registration/benchmark-specific), this module makes NO assumption about what pipeline
produced its content: title, KPI cards, parameter table, categorized QC sections,
figures, and caveats are all supplied by the caller. Centralized here (rather than
reimplemented per downstream package -- confirmed duplicated independently in both
antsxfunctional and antsxdwi) so every antsx* pipeline's report looks and behaves the
same way, and a fix made once (e.g. the two real bugs below) benefits every consumer.

Two real, confirmed usability bugs already found and fixed in this design, in an
upstream (antsxfunctional) prototype of this exact module, are baked in from the start:
  1. Categorized QC metrics render as a compact colored card grid (reusing the same
     visual language as the headline KPI cards), NOT a flat table -- long QC tables read
     as a "wall of text" nobody actually reads.
  2. Free-text caveats/methodology notes are written to a JSON sidecar next to the
     report (not rendered as HTML prose), with only a short pointer line on the page --
     keeping the full text machine-readable and available without cluttering the page.
"""

from __future__ import annotations

import base64
import datetime
import html as _html
import json
from pathlib import Path
from typing import Any

_STATUS_COLORS = {"ok": "#22c55e", "warn": "#f59e0b", "fail": "#ef4444", "unknown": "#64748b"}


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _img(path: str, caption: str, width: str = "100%") -> str:
    try:
        data = _b64(path)
        img_style = (
            f"max-width:{width};border:1px solid #1e293b;border-radius:6px;"
            "box-shadow:0 4px 6px -1px rgba(0,0,0,0.3)"
        )
        cap_style = "font-size:0.82rem;color:#94a3b8;margin-top:0.4rem;font-weight:500"
        return (
            f'<figure style="margin:0 0 1.25rem 0;text-align:center">'
            f'<img src="data:image/png;base64,{data}" style="{img_style}" alt="{_html.escape(caption)}"/>'
            f'<figcaption style="{cap_style}">{_html.escape(caption)}</figcaption>'
            f"</figure>"
        )
    except Exception:
        return f'<p style="color:#fb7185;font-size:0.85rem">Figure missing: {_html.escape(path)}</p>'


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return _html.escape(str(value))


def kpi_card(label: str, value: Any, note: str = "", status_color: str = "#38bdf8") -> str:
    """Build one KPI headline card. Callers assemble the KPI row for their own modality --
    nothing here is hardcoded to any specific metric."""
    return (
        '<div style="background:#1e293b;border-radius:8px;padding:0.9rem 1.1rem;min-width:150px;'
        f'border-left:4px solid {status_color}">'
        f'<div style="font-size:0.72rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.04em">{_html.escape(label)}</div>'
        f'<div style="font-size:1.5rem;color:#f1f5f9;font-weight:600;margin-top:0.15rem">{_fmt(value)}</div>'
        f'<div style="font-size:0.72rem;color:#64748b;margin-top:0.15rem">{_html.escape(note)}</div>'
        "</div>"
    )


def _status_pill(status: str) -> str:
    color = _STATUS_COLORS.get(status, "#64748b")
    return (
        f'<span style="display:inline-block;padding:0.1rem 0.55rem;border-radius:999px;'
        f'background:{color}22;color:{color};font-size:0.72rem;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:0.03em">{_html.escape(status)}</span>'
    )


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "".join(
        f'<th style="text-align:left;padding:0.4rem 0.8rem;color:#94a3b8;font-size:0.78rem;'
        f'text-transform:uppercase;letter-spacing:0.03em;border-bottom:1px solid #1e293b">{_html.escape(c)}</th>'
        for c in columns
    )

    def _cell(row: dict[str, Any], col: str) -> str:
        value = row.get(col, "")
        if col == "Status" and isinstance(value, str) and value in _STATUS_COLORS:
            return _status_pill(value)
        return _fmt(value)

    body = "".join(
        "<tr>"
        + "".join(
            f'<td style="padding:0.4rem 0.8rem;border-bottom:1px solid #1e293b;color:#e2e8f0;font-size:0.85rem">{_cell(row, c)}</td>'
            for c in columns
        )
        + "</tr>"
        for row in rows
    )
    return (
        '<table style="border-collapse:collapse;width:100%;margin-bottom:1.5rem">'
        f"<thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"
    )


def _qc_metric_card(name: str, value: Any) -> str:
    """Render one QC metric as a small colored card (reuses kpi_card's exact styling)
    instead of a table row -- see module docstring, bug #1."""
    if isinstance(value, dict):
        color = _STATUS_COLORS.get(value.get("status", ""), "#38bdf8")
        return kpi_card(name.replace("_", " "), value.get("value"), "", color)
    return kpi_card(name.replace("_", " "), value, "")


def equation_figure(equation: str, definitions: list[str], save_path: str, title: str = "") -> str:
    """Render a mathematical model equation as a standalone figure, for use as a report's
    ``highlight_figure``. Uses matplotlib's mathtext (no system LaTeX install, no MathJax/
    KaTeX CDN dependency) so the equation renders with real mathematical typesetting while
    keeping the report fully self-contained and offline-viewable.

    Parameters
    ----------
    equation : str
        A matplotlib mathtext expression, e.g. ``r"$Y(t,v) = \\beta_0(v) + ...$"``.
    definitions : list of str
        Short term-definition lines shown below the equation (mathtext allowed in each).
    save_path : str
    title : str, optional
        Small label shown above the equation.

    Returns
    -------
    str
        ``save_path``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_lines = max(len(definitions), 1)
    fig_height = 1.5 + 0.34 * n_lines
    fig, ax = plt.subplots(figsize=(9, fig_height), facecolor="#0f172a")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    top = 0.97
    if title:
        ax.text(0.5, top, title, ha="center", va="top", color="#94a3b8", fontsize=11.5, transform=ax.transAxes)
        top -= 0.16
    ax.text(0.5, top, equation, ha="center", va="top", color="#f1f5f9", fontsize=16, transform=ax.transAxes)
    top -= 0.22

    for i, line in enumerate(definitions):
        y = top - i * (top / n_lines if n_lines else 0)
        ax.text(0.03, y, line, ha="left", va="top", color="#cbd5e1", fontsize=10, transform=ax.transAxes)

    fig.patch.set_facecolor("#0f172a")
    fig.savefig(save_path, facecolor="#0f172a", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return save_path


def provenance_table_rows(provenance: list[Any]) -> list[dict[str, Any]]:
    """Convert a list of ``syntx.contract.ProvenanceEntry`` objects into table rows for the
    "Processing Steps" section -- one place that knows how to render provenance, reused by
    every modality instead of each pipeline hand-rolling its own table."""
    rows = []
    for p in provenance:
        extra = ", ".join(f"{k}={v}" for k, v in (p.extra or {}).items())
        rows.append(
            {
                "Step": p.step,
                "Engine": p.engine,
                "Device": p.device,
                "Seconds": round(p.seconds, 2),
                "Details": extra,
            }
        )
    return rows


def write_modality_report(
    output_path: str,
    modality_title: str,
    session_label: str,
    kpis_html: str,
    figure_paths: dict[str, str],
    parameters: dict[str, Any] | None = None,
    quantitative_qc: dict[str, Any] | None = None,
    qc_sections: dict[str, dict[str, Any]] | None = None,
    highlight_figure: str | None = None,
    highlight_caption: str = "",
    provenance: list[Any] | None = None,
    caveats: list[str] | None = None,
) -> str:
    """Write a self-contained, any-modality QC HTML report.

    Parameters
    ----------
    output_path : str
        Destination ``.html`` path.
    modality_title : str
        e.g. "Perfusion/ASL", "Resting-State fMRI", "DTI" -- rendered in the page
        ``<title>`` and ``<h1>``. Never hardcoded by this function.
    session_label : str
        Human-readable session identifier.
    kpis_html : str
        Pre-rendered HTML for the headline KPI cards (build with :func:`kpi_card` per
        card, concatenate the results) -- modality-specific, supplied by the caller.
    figure_paths : dict
        Mapping of figure name -> PNG path.
    parameters : dict, optional
        Modeling/acquisition parameters rendered as a "Parameters & Modeling" table.
    quantitative_qc : dict, optional
        Flat QC metrics rendered as a single "Quantitative QC" table (legacy/simple
        path). Each value is either a plain scalar, or ``{"value", "status", "note"}``.
    qc_sections : dict, optional
        Categorized QC (e.g. {"Registration": {...}, "Motion": {...}, "Segmentation":
        {...}, "Image Quality": {...}, "AI / Deep-Learning Models": {...}}), each
        rendered as its own "Quality Control — <category>" card grid (see
        ``_qc_metric_card``) -- preferred over the flat ``quantitative_qc`` for new
        callers. Renders alongside (not instead of) ``quantitative_qc`` if both given.
    highlight_figure : str, optional
        Path to one figure shown prominently right after the KPI row, before
        Parameters -- for content that must not get buried below a long page (e.g. a
        rendered equation, or a registration-staging figure).
    highlight_caption : str
    provenance : list of ProvenanceEntry, optional
        Rendered as a "Processing Steps" table via :func:`provenance_table_rows`.
    caveats : list of str, optional
        Free-text methodology notes. NOT rendered as prose on the page -- written to a
        JSON sidecar (``<output_path stem>_notes.json``) next to the report, with only
        a short pointer line shown.

    Returns
    -------
    str
        ``output_path``.
    """
    caveat_html = ""
    if caveats:
        notes_path = Path(output_path).with_name(Path(output_path).stem + "_notes.json")
        notes_path.write_text(json.dumps({"caveats": caveats}, indent=2))
        caveat_html = (
            '<div style="background:#1e293b;border-left:4px solid #38bdf8;border-radius:6px;'
            'padding:0.6rem 1.1rem;margin-bottom:1.5rem;color:#94a3b8;font-size:0.85rem">'
            f"{len(caveats)} methodology notes kept out of this page to stay readable -- "
            f'see <code style="color:#e2e8f0">{_html.escape(notes_path.name)}</code>'
            "</div>"
        )

    def _section(heading: str, body: str) -> str:
        if not body:
            return ""
        return (
            f'<h2 style="font-size:1.1rem;border-bottom:1px solid #1e293b;padding-bottom:0.4rem;'
            f'margin-top:1.8rem">{_html.escape(heading)}</h2>{body}'
        )

    params_html = ""
    if parameters:
        rows = [{"Parameter": k, "Value": v} for k, v in parameters.items()]
        params_html = _section("Parameters & Modeling", _table(rows, ["Parameter", "Value"]))

    qc_html = ""
    if quantitative_qc:
        rows = []
        any_status = False
        for k, v in quantitative_qc.items():
            if isinstance(v, dict):
                row = {"Metric": k, "Value": v.get("value"), "Status": v.get("status", ""), "Note": v.get("note", "")}
                any_status = any_status or bool(v.get("status"))
            else:
                row = {"Metric": k, "Value": v, "Status": "", "Note": ""}
            rows.append(row)
        columns = ["Metric", "Value", "Status", "Note"] if any_status else ["Metric", "Value"]
        qc_html = _section("Quantitative QC", _table(rows, columns))

    qc_sections_html = ""
    if qc_sections:
        parts = []
        for category, metrics in qc_sections.items():
            if not metrics:
                continue
            cards = "".join(_qc_metric_card(k, v) for k, v in metrics.items())
            parts.append(
                _section(
                    f"Quality Control — {category}",
                    f'<div style="display:flex;flex-wrap:wrap;gap:0.6rem;margin-bottom:0.5rem">{cards}</div>',
                )
            )
        qc_sections_html = "".join(parts)

    highlight_html = ""
    if highlight_figure:
        highlight_html = f'<div style="margin-bottom:1rem">{_img(highlight_figure, caption=highlight_caption, width="70%")}</div>'

    prov_html = ""
    if provenance:
        rows = provenance_table_rows(provenance)
        prov_html = _section("Processing Steps", _table(rows, ["Step", "Engine", "Device", "Seconds", "Details"]))

    figures_html = "".join(
        _img(path, caption=name.replace("_", " ").title()) for name, path in figure_paths.items() if path is not None
    )
    maps_html = _section("Maps", f'<div style="display:flex;flex-direction:column;gap:0.5rem">{figures_html}</div>')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>syntx {_html.escape(modality_title)} report — {_html.escape(session_label)}</title>
</head>
<body style="margin:0;padding:2rem;background:#0f172a;color:#e2e8f0;font-family:-apple-system,Segoe UI,sans-serif">
<div style="max-width:1100px;margin:0 auto">
<h1 style="font-size:1.4rem;margin-bottom:0.1rem">{_html.escape(modality_title)} Report</h1>
<div style="color:#64748b;font-size:0.85rem;margin-bottom:1.5rem">
  {_html.escape(session_label)} &middot; generated {datetime.datetime.now().isoformat(timespec="seconds")}
</div>
{caveat_html}
<div style="display:flex;flex-wrap:wrap;gap:0.9rem;margin-bottom:1rem">{kpis_html}</div>
{highlight_html}
{params_html}
{qc_html}
{qc_sections_html}
{prov_html}
{maps_html}
</div>
</body>
</html>"""

    Path(output_path).write_text(html)
    return output_path
