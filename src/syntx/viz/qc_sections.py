"""syntx.viz.qc_sections — generic, modality-agnostic categorized QC-section builders.

Each function builds one category's worth of metrics for
:func:`syntx.viz.modality_report.write_modality_report`'s ``qc_sections`` parameter, in
the shared ``{"metric_name": value_or_{"value","status","note"}}`` shape. Centralized
here (rather than reimplemented per downstream package) so the same kind of metric --
a registration MI/Dice, a motion FD, a "which steps used a deep-learning model" summary
-- is graded and categorized identically across every antsx* pipeline.

Deliberately NOT included here: metrics that assume a specific vendored QC library's
naming convention (e.g. a package-specific "blind QC" wide-row prefix scheme) -- those
stay in the downstream package that owns that convention.
"""

from __future__ import annotations

from typing import Any


def grade_edge_dice_tol1(edge_dice_tol1: float) -> str:
    """Threshold a tolerant edge-boundary Dice overlap (top-gradient-percentile voxels
    only, e.g. syntx/antsxfunctional's edge_overlap_metrics) into ok/warn/fail.

    NOT the same scale as grade_dice_overlap's whole-mask Dice convention -- this compares
    only the sparsest, top ~15% gradient-magnitude voxels between two images, a much
    harder criterion than whole-region overlap, so real good cross-modal registrations
    score much lower here than a typical brain-mask Dice. Thresholds calibrated against
    real multi-dataset T1-to-PET registration evidence (ds004856, SOCOM, FPA; Sept 2026 -
    see antsxfunctional's pet/registration.py history) rather than a literature constant:
    observed edge_dice_tol1 for visually-confirmed-good real registrations clustered
    ~0.35-0.45; a genuinely misaligned pair scores far lower (edges don't coincide at all)."""
    if edge_dice_tol1 != edge_dice_tol1:  # NaN
        return "unknown"
    if edge_dice_tol1 >= 0.30:
        return "ok"
    if edge_dice_tol1 >= 0.15:
        return "warn"
    return "fail"


def grade_dice_overlap(dice: float) -> str:
    """Threshold Dice/mask-overlap into ok/warn/fail. Thresholds follow the common
    neuroimaging registration-QC convention (Dice >= 0.7 good overlap, 0.5-0.7 marginal,
    <0.5 poor) -- a heuristic bin, not a universal physical constant, but a defensible,
    disclosed default rather than leaving every caller to invent (or omit) one."""
    if dice != dice:  # NaN
        return "unknown"
    if dice >= 0.7:
        return "ok"
    if dice >= 0.5:
        return "warn"
    return "fail"


def grade_framewise_displacement_mean(fd_mean: float) -> str:
    """Threshold mean framewise displacement into ok/warn/fail (mm). Follows common
    resting-state fMRI QC convention (mean FD < 0.2mm low motion, 0.2-0.5mm moderate,
    >0.5mm high motion) -- disclosed heuristic bins."""
    if fd_mean != fd_mean:
        return "unknown"
    if fd_mean < 0.2:
        return "ok"
    if fd_mean < 0.5:
        return "warn"
    return "fail"


def registration_qc_section(
    mutual_information: float,
    dice: float | None = None,
    dice_note: str = "brain/foreground mask overlap",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a generic 'Registration' QC section. Reused by every modality's report --
    the metric names and grading are identical everywhere, only the caller-supplied
    values differ."""
    section: dict[str, Any] = {"mutual_information": mutual_information}
    if dice is not None:
        section["dice_overlap"] = {"value": dice, "status": grade_dice_overlap(dice), "note": dice_note}
    if extra:
        section.update(extra)
    return section


def motion_qc_section(
    fd_mean: float | None = None,
    fd_max: float | None = None,
    motion_corrected: bool | None = None,
) -> dict[str, Any]:
    """Build a generic 'Motion' QC section. Reused by every modality that performs motion
    correction -- fd_mean grading is identical everywhere."""
    section: dict[str, Any] = {}
    if motion_corrected is not None:
        section["motion_corrected"] = motion_corrected
    if fd_mean is not None:
        section["fd_mean"] = {
            "value": fd_mean, "status": grade_framewise_displacement_mean(fd_mean), "note": "mean framewise displacement (mm)",
        }
    if fd_max is not None:
        section["fd_max"] = fd_max
    return section


def segmentation_qc_section(
    mask_volume_mm3: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a generic 'Segmentation' QC section (mask/label volume and any caller-
    supplied extras). Deliberately minimal -- reports only what is genuinely known
    (volume), not a fabricated confidence score, unless the caller supplies one."""
    section: dict[str, Any] = {}
    if mask_volume_mm3 is not None:
        section["mask_volume_mm3"] = mask_volume_mm3
    if extra:
        section.update(extra)
    return section


def ai_model_qc_section(provenance: list[Any], extra_steps: tuple[str, ...] = ()) -> dict[str, Any]:
    """Build an 'AI / Deep-Learning Models' QC section listing which processing steps in
    ``provenance`` (a list of ``syntx.contract.ProvenanceEntry``) used a deep-learning
    model, and where they ran. A step counts as AI/deep-learning if its ``engine`` is
    ``"antstorch"`` (the antsx* convention for antstorch's deep models: brain_extraction,
    deep_atropos, DKT labeling, ...), or if its ``step`` name is in ``extra_steps`` (for
    steps that must be tagged with a different ``engine`` value for
    ``ProvenanceEntry``'s own Literal-type reasons, but are genuinely a deep model)."""
    section: dict[str, Any] = {}
    for p in provenance:
        if p.engine == "antstorch" or p.step in extra_steps:
            note = p.extra.get("source") if getattr(p, "extra", None) else None
            section[p.step] = {"value": f"{p.engine} on {p.device}", "note": f"{p.seconds:.2f}s" + (f"; {note}" if note else "")}
    return section
