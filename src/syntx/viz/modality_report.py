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
        ext = Path(path).suffix.lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        img_style = (
            f"max-width:{width};border:1px solid #1e293b;border-radius:6px;"
            "box-shadow:0 4px 6px -1px rgba(0,0,0,0.3)"
        )
        cap_style = "font-size:0.82rem;color:#94a3b8;margin-top:0.4rem;font-weight:500"
        return (
            f'<figure style="margin:0 0 1.25rem 0;text-align:center">'
            f'<img src="data:{mime};base64,{data}" style="{img_style}" alt="{_html.escape(caption)}"/>'
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
        status_key = value.get("status", "")
        badge_map = {
            "badge-optimal": "#22c55e", "badge-nominal": "#38bdf8",
            "badge-warning": "#f59e0b", "badge-flagged": "#ef4444", "badge-neutral": "#64748b",
        }
        color = _STATUS_COLORS.get(status_key, badge_map.get(status_key, str(status_key) if str(status_key).startswith("#") else "#38bdf8"))
        note = value.get("note", "")
        label = value.get("label", name.replace("_", " "))
        return kpi_card(label, value.get("value"), note, color)
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


def equations_figure(equations: list[dict[str, Any]], save_path: str, header_title: str = "") -> str:
    """Render 2-3 core processing equations as one stacked figure, for use as a report's
    ``highlight_figure`` -- so the "key processing equations" for a modality's pipeline
    are visible at the top of the report, not buried in prose or omitted entirely. Uses
    matplotlib mathtext (no system LaTeX, no MathJax/KaTeX CDN dependency), the same
    rendering approach as :func:`equation_figure`, generalized to multiple entries.

    This is meant as a compact summary panel (the 2-3 MOST central equations for a
    method), not an exhaustive derivation -- pick the equations a reader most needs to
    understand what the pipeline actually computed.

    Parameters
    ----------
    equations : list of dict
        Each dict: ``{"title": str, "equation": str, "definitions": list[str]}``.
        ``equation`` is a mathtext expression (may contain embedded ``\\n`` to stack
        multiple related formulas within one entry, e.g. ALFF and fALFF together).
    save_path : str
    header_title : str, optional
        Overall figure title (e.g. "Key Processing Equations").

    Returns
    -------
    str
        ``save_path``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _tall_element_weight(equation: str) -> float:
        """Weighted count of mathtext constructs that render substantially taller than a
        single text line. Sum/integral/product with explicit over/under limits (the
        common case in practice) render even taller than a plain stacked fraction, so
        they are weighted more heavily rather than assuming the lighter no-limits case."""
        weight = 0.0
        weight += equation.count(r"\frac") * 1.0
        weight += equation.count(r"\dfrac") * 1.0
        weight += equation.count(r"\sum") * 1.6
        weight += equation.count(r"\int") * 1.6
        weight += equation.count(r"\prod") * 1.6
        return weight

    # Fixed, absolute (inches) vertical budgets per element. Each panel's total height is
    # literally the sum of the budgets used to place its own content (title, equation,
    # gap, one line per definition, plus top/bottom margins) -- this is what guarantees
    # definitions text can never overlap the equation above it, regardless of how tall
    # any individual equation renders: unlike an earlier version of this function (which
    # subtracted fixed AXIS-FRACTION offsets from a fixed 0-1 budget), a real bug found
    # by visual inspection of a generated report -- a tall equation (e.g. one with a
    # \sum with over/under limits) could subtract more than the entire available
    # fraction, driving the cursor negative and overlapping definitions on top of the
    # equation. Sizing and placement now share one source of truth instead of two.
    TITLE_H = 0.38
    EQ_BASE_H = 0.60
    EQ_TALL_H = 0.42  # per unit of _tall_element_weight
    EQ_NEWLINE_H = 0.34  # per embedded newline in the equation
    DEF_LINE_H = 0.30
    TOP_MARGIN = 0.28
    BOTTOM_MARGIN = 0.18
    GAP_BEFORE_DEFS = 0.18

    panel_specs = []
    for eq in equations:
        title = eq.get("title", "")
        equation = eq.get("equation", "")
        definitions = eq.get("definitions", [])
        title_h = TITLE_H if title else 0.0
        eq_h = EQ_BASE_H + EQ_TALL_H * _tall_element_weight(equation) + EQ_NEWLINE_H * equation.count("\n")
        defs_h = DEF_LINE_H * len(definitions)
        gap_h = GAP_BEFORE_DEFS if definitions else 0.0
        panel_h = TOP_MARGIN + title_h + eq_h + gap_h + defs_h + BOTTOM_MARGIN
        panel_specs.append({"panel_h": panel_h, "title_h": title_h, "eq_h": eq_h, "gap_h": gap_h})

    heights = [spec["panel_h"] for spec in panel_specs]
    header_h = 0.55 if header_title else 0.0
    total_height = sum(heights) + header_h

    fig, axes = plt.subplots(len(equations), 1, figsize=(9, total_height), facecolor="#0f172a", gridspec_kw={"height_ratios": heights})
    if len(equations) == 1:
        axes = [axes]

    if header_title:
        fig.suptitle(header_title, color="#f8fafc", fontsize=13, y=0.995)

    for ax, eq, spec in zip(axes, equations, panel_specs):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        panel_h = spec["panel_h"]

        cursor = 1.0 - TOP_MARGIN / panel_h

        title = eq.get("title", "")
        if title:
            ax.text(0.5, cursor, title, ha="center", va="top", color="#94a3b8", fontsize=11, transform=ax.transAxes)
            cursor -= spec["title_h"] / panel_h

        equation = eq.get("equation", "")
        ax.text(0.5, cursor, equation, ha="center", va="top", color="#f1f5f9", fontsize=14, transform=ax.transAxes)
        cursor -= spec["eq_h"] / panel_h

        definitions = eq.get("definitions", [])
        if definitions:
            cursor -= spec["gap_h"] / panel_h
            for line in definitions:
                ax.text(0.03, cursor, line, ha="left", va="top", color="#cbd5e1", fontsize=9.5, transform=ax.transAxes)
                cursor -= DEF_LINE_H / panel_h

    fig.patch.set_facecolor("#0f172a")
    if header_title:
        fig.subplots_adjust(top=1.0 - header_h / total_height)
    fig.savefig(save_path, facecolor="#0f172a", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return save_path


def provenance_table_rows(provenance: list[Any]) -> list[dict[str, Any]]:
    """Convert a list of ``syntx.contract.ProvenanceEntry`` objects into table rows for the
    "Processing Steps" section -- one place that knows how to render provenance, reused by
    every modality instead of each pipeline hand-rolling its own table."""
    rows = []
    for p in provenance:
        d = p.__dict__ if hasattr(p, "__dict__") else dict(p) if isinstance(p, dict) else {}
        step = d.get("step", getattr(p, "step", ""))
        engine = d.get("engine", getattr(p, "engine", ""))
        dev = d.get("device", getattr(p, "device", ""))
        sec = float(d.get("seconds", getattr(p, "seconds", 0.0)))
        extra_val = d.get("extra", getattr(p, "extra", None))
        extra = ", ".join(f"{k}={v}" for k, v in (extra_val or {}).items()) if isinstance(extra_val, dict) else str(extra_val or "")
        rows.append(
            {
                "Step": step,
                "Engine": engine,
                "Device": dev,
                "Seconds": round(sec, 2),
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
    brand: str = "",
    header_badges: list[tuple[str, str]] | None = None,
    stage_sections: list[dict[str, Any]] | None = None,
    artifacts: dict[str, str] | None = None,
    provenance_json: bool = False,
    provenance_title: str = "Processing Steps",
    title_override: str | None = None,
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
    brand : str, optional
        Optional caller package name prefixed onto the ``<h1>``/``<title>`` (e.g.
        "antsxfunctional", "antsxdwi") -- this function itself has no default branding.
    kpis_html : str
        Pre-rendered HTML for the headline KPI cards (build with :func:`kpi_card` per
        card, concatenate the results) -- modality-specific, supplied by the caller.
    figure_paths : dict
        Mapping of figure name -> PNG/JPEG path.
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
    header_badges : list of tuple, optional
        List of ``(label, value)`` tuples displayed as a metadata badge row under the
        header (e.g. Subject, Session, Run, Resolution, Engine, Device, Runtime).
    stage_sections : list of dict, optional
        Ordered list of structured pipeline stage sections. Each dict specifies:
        ``title`` (str), ``badge`` (str | None), ``badge_status`` (str | None,
        one of "ok"/"warn"/"fail"/"unknown"/None), ``description`` (str | None,
        narrative text/HTML), and ``figures`` (list of (path, caption) tuples).
    artifacts : dict, optional
        Mapping of artifact name -> filesystem path, rendered as a clickable
        "Generated Pipeline Artifacts" table.
    provenance_json : bool, default False
        When True, renders an interactive ``<details>`` dropdown containing the full
        machine-readable execution provenance JSON.

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

    badges_html = ""
    if header_badges:
        badges = []
        for item in header_badges:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                label, val = item
                label_esc = _html.escape(str(label))
                val_esc = _html.escape(str(val))
                if label:
                    badge_text = f'<span style="color:#94a3b8">{label_esc}:</span> <strong style="color:#e2e8f0">{val_esc}</strong>'
                else:
                    badge_text = f'<strong style="color:#e2e8f0">{val_esc}</strong>'
            else:
                badge_text = f'<strong style="color:#e2e8f0">{_html.escape(str(item))}</strong>'
            badges.append(
                f'<span style="display:inline-block;padding:0.25rem 0.65rem;background:#1e293b;border-radius:999px;'
                f'font-size:0.75rem;border:1px solid #334155">{badge_text}</span>'
            )
        badges_html = f'<div style="display:flex;flex-wrap:wrap;gap:0.4rem;margin-bottom:1.2rem">{"".join(badges)}</div>'

    stage_sections_html = ""
    if stage_sections:
        stage_parts = []
        for s in stage_sections:
            title = s.get("title", "")
            badge = s.get("badge", None)
            badge_status = s.get("badge_status", None)
            desc = s.get("description", None)
            s_figs = s.get("figures", [])

            badge_html = ""
            if badge:
                color = _STATUS_COLORS.get(badge_status, "#64748b") if badge_status else "#64748b"
                badge_html = (
                    f'<span style="display:inline-block;padding:0.15rem 0.6rem;border-radius:999px;'
                    f'background:{color}22;color:{color};font-size:0.75rem;font-weight:600;'
                    f'border:1px solid {color}44">{_html.escape(str(badge))}</span>'
                )

            header_html = (
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'border-bottom:1px solid #1e293b;padding-bottom:0.4rem;margin-bottom:0.8rem">'
                f'  <h2 style="font-size:1.1rem;margin:0;color:#f1f5f9">{_html.escape(str(title))}</h2>'
                f'  {badge_html}'
                f'</div>'
            )

            desc_html = ""
            if desc:
                desc_html = f'<div style="font-size:0.88rem;color:#cbd5e1;margin-bottom:1rem;line-height:1.5">{desc}</div>'

            figs_html = ""
            if s_figs:
                figs_html = "".join(_img(fpath, caption=fcap) for fpath, fcap in s_figs)

            stage_parts.append(
                f'<section style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;'
                f'padding:1.2rem 1.4rem;margin-bottom:1.5rem">'
                f'{header_html}{desc_html}{figs_html}'
                f'</section>'
            )
        stage_sections_html = "".join(stage_parts)

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
        prov_html = _section(provenance_title, _table(rows, ["Step", "Engine", "Device", "Seconds", "Details"]))

    figures_html = "".join(
        _img(path, caption=name.replace("_", " ").title()) for name, path in figure_paths.items() if path is not None
    )
    maps_html = _section("Maps", f'<div style="display:flex;flex-direction:column;gap:0.5rem">{figures_html}</div>') if figures_html else ""

    artifacts_html = ""
    if artifacts:
        art_rows = "".join(
            f'<tr><td style="padding:0.4rem 0.8rem;border-bottom:1px solid #1e293b;color:#e2e8f0;font-size:0.85rem"><strong>{_html.escape(str(k))}</strong></td>'
            f'<td style="padding:0.4rem 0.8rem;border-bottom:1px solid #1e293b;font-family:monospace;font-size:0.8rem;word-break:break-all">'
            f'<a href="{_html.escape(str(v))}" style="color:#38bdf8;text-decoration:none">{_html.escape(Path(v).name)}</a></td></tr>'
            for k, v in sorted(artifacts.items())
        )
        art_table = (
            f'<table style="border-collapse:collapse;width:100%;margin-bottom:1.5rem">'
            f'<thead><tr><th style="text-align:left;padding:0.4rem 0.8rem;color:#94a3b8;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.03em;border-bottom:1px solid #1e293b">Artifact Key</th>'
            f'<th style="text-align:left;padding:0.4rem 0.8rem;color:#94a3b8;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.03em;border-bottom:1px solid #1e293b">File System Path</th></tr></thead>'
            f'<tbody>{art_rows}</tbody></table>'
        )
        artifacts_html = _section("Generated Pipeline Artifacts", art_table)

    prov_json_html = ""
    if provenance_json and provenance:
        def _to_json_serializable(obj):
            if hasattr(obj, "__dict__"):
                return {k: _to_json_serializable(val) for k, val in obj.__dict__.items() if not k.startswith("_")}
            elif isinstance(obj, dict):
                return {k: _to_json_serializable(val) for k, val in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_to_json_serializable(val) for val in obj]
            return str(obj) if not isinstance(obj, (int, float, bool, type(None))) else obj

        prov_data = [_to_json_serializable(p) for p in provenance]
        prov_json_str = _html.escape(json.dumps(prov_data, indent=2))
        prov_json_html = (
            f'<details style="margin-top:1rem;margin-bottom:1.5rem;background:#1e293b;'
            f'padding:0.8rem 1.1rem;border-radius:6px;font-size:0.82rem">'
            f'<summary style="cursor:pointer;color:#94a3b8;font-weight:600">'
            f'Complete Execution Provenance &amp; Parameters JSON</summary>'
            f'<pre style="color:#cbd5e1;overflow-x:auto;max-height:400px;margin-top:0.6rem;font-size:0.75rem">{prov_json_str}</pre>'
            f'</details>'
        )

    abs_out = str(Path(output_path).resolve())
    open_helper_html = (
        f'<div style="background:#1e293b;border-left:4px solid #38bdf8;border-radius:6px;'
        f'padding:0.6rem 1.1rem;margin-top:1.5rem;margin-bottom:1.5rem;color:#94a3b8;font-size:0.85rem">'
        f'💡 <strong style="color:#cbd5e1">View Report:</strong> Open on macOS: '
        f'<code style="background:#0f172a;color:#38bdf8;padding:0.15rem 0.4rem;border-radius:4px">'
        f'open &quot;{_html.escape(abs_out)}&quot;</code>'
        f'</div>'
    )
    footer_text = f"Generated by <strong>{_html.escape(brand)}</strong> — Advanced Medical Processing Verification Engine" if brand else "Generated by <strong>syntx</strong>"
    footer_html = f'<footer style="margin-top:2rem;padding-top:1rem;border-top:1px solid #1e293b;color:#64748b;font-size:0.8rem;text-align:center"><p>{footer_text}</p></footer>'

    brand_prefix = f"{brand} " if brand else ""
    brand_h1_prefix = f"{brand} — " if brand else ""
    page_title = title_override if title_override is not None else f"{brand_prefix}{modality_title} report — {session_label}"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{_html.escape(page_title)}</title>
</head>
<body style="margin:0;padding:2rem;background:#0f172a;color:#e2e8f0;font-family:-apple-system,Segoe UI,sans-serif">
<div style="max-width:1100px;margin:0 auto">
<h1 style="font-size:1.4rem;margin-bottom:0.1rem">{_html.escape(brand_h1_prefix)}{_html.escape(modality_title)} Report</h1>
<div style="color:#64748b;font-size:0.85rem;margin-bottom:1rem">
  {_html.escape(session_label)} &middot; generated {datetime.datetime.now().isoformat(timespec="seconds")}
</div>
{badges_html}
{caveat_html}
<div style="display:flex;flex-wrap:wrap;gap:0.9rem;margin-bottom:1.5rem">{kpis_html}</div>
{highlight_html}
{stage_sections_html}
{params_html}
{qc_html}
{qc_sections_html}
{prov_html}
{maps_html}
{artifacts_html}
{prov_json_html}
{open_helper_html}
{footer_html}
</div>
</body>
</html>"""

    Path(output_path).write_text(html)
    return output_path

