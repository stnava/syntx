"""syntx.viz.metric_sections — modality-independent QC-section builders (thresholds live in ``antsxqc.specs``).

Each ``*_qc_section`` function returns one category of metrics for the ``qc_sections``
argument of :func:`syntx.viz.modality_report.write_modality_report`: a dict
``{metric_name: value}`` where a value is either a plain value or a dict
``{"value", "status", "note"}`` (``status`` one of "pass", "warn", "fail", "not_assessable"; see :func:`antsxqc.grade`). Keeping
the builders here lets the antsx* pipelines grade the same metric the same way.

Metrics tied to one package's own naming scheme are left to that package.
"""

from __future__ import annotations

from typing import Any

from antsxqc import grade


def registration_qc_section(
    mutual_information: float | None = None,
    dice: float | None = None,
    dice_note: str = "brain/foreground mask overlap",
    folding_pct: float | None = None,
    min_jacobian: float | None = None,
    jac_measure: str | None = None,
    inverse_identity_max_mm: float | None = None,
    inverse_identity_mean_mm: float | None = None,
    qc_report: Any | None = None,
    extra: dict[str, Any] | None = None,
    reference: Any | None = None,
) -> dict[str, Any]:
    """Build a 'Registration' QC section.

    Parameters
    ----------
    mutual_information : float, optional
        Stored ungraded under ``"mutual_information"``.
    dice : float, optional
        If given, stored as ``"dice_overlap": {"value", "status": grade("dice_overlap", dice),
        "note": dice_note}``.
    dice_note : str, default "brain/foreground mask overlap"
        Note for the Dice entry.
    folding_pct : float, optional
        Topology folding percentage (graded by ``antsxqc`` spec ``folding_pct``).
    min_jacobian : float, optional
        Minimum Jacobian determinant (graded by ``antsxqc`` spec ``min_jacobian``).
    jac_measure : str, optional
        Method used for Jacobian (e.g. "Liouville determinant" or "finite difference").
    inverse_identity_max_mm : float, optional
        Maximum inverse identity error in mm (graded by ``antsxqc`` spec ``inverse_identity_error``).
    inverse_identity_mean_mm : float, optional
        Mean inverse identity error in mm.
    qc_report : RegistrationQCReport or dict, optional
        Optional RegistrationQCReport object to populate metrics from.
    extra : dict, optional
        Merged in last (can overwrite the keys above).

    Returns
    -------
    dict
    """
    section: dict[str, Any] = {}
    if mutual_information is not None:
        section["mutual_information"] = mutual_information

    if qc_report is not None:
        d = qc_report.to_dict() if hasattr(qc_report, 'to_dict') else (qc_report if isinstance(qc_report, dict) else {})
        if folding_pct is None and 'folding_pct' in d:
            folding_pct = float(d['folding_pct'])
        if min_jacobian is None and 'min_jacobian' in d:
            min_jacobian = float(d['min_jacobian'])
        if jac_measure is None and 'jac_measure' in d:
            jac_measure = str(d['jac_measure'])
        if inverse_identity_max_mm is None and d.get('ice_max_mm') is not None:
            inverse_identity_max_mm = float(d['ice_max_mm'])
        if inverse_identity_mean_mm is None and d.get('ice_mean_mm') is not None:
            inverse_identity_mean_mm = float(d['ice_mean_mm'])
        if dice is None and d.get('dice') is not None:
            dice = float(d['dice'])

    if dice is not None:
        section["dice_overlap"] = {"value": dice, "status": grade("dice_overlap", dice, reference=reference), "note": dice_note}
    if folding_pct is not None:
        section["folding_pct"] = {
            "value": folding_pct,
            "status": grade("folding_pct", folding_pct, reference=reference),
            "note": f"folding voxels det(J)<=0% ({jac_measure or 'Liouville'})"
        }
    if min_jacobian is not None:
        section["min_jacobian"] = {
            "value": min_jacobian,
            "status": grade("min_jacobian", min_jacobian, reference=reference),
            "note": f"min det(J) ({jac_measure or 'Liouville'})"
        }
    if inverse_identity_max_mm is not None:
        section["inverse_identity_max_mm"] = {
            "value": inverse_identity_max_mm,
            "status": grade("inverse_identity_error", inverse_identity_max_mm, reference=reference),
            "note": "max ICE (mm)" + (f"; mean {inverse_identity_mean_mm:.3f}mm" if inverse_identity_mean_mm is not None else "")
        }
    if extra:
        section.update(extra)
    return section


def motion_qc_section(
    fd_mean: float | None = None,
    fd_max: float | None = None,
    motion_corrected: bool | None = None,
    reference: Any | None = None,
) -> dict[str, Any]:
    """Build a 'Motion' QC section with only the arguments that are not None.

    Keys: ``"motion_corrected"`` (as given), ``"fd_mean"`` (``{"value", "status":
    grade("fd_mean", fd_mean, reference=reference), "note"}``; mm) and ``"fd_max"`` (ungraded).

    Returns
    -------
    dict
    """
    section: dict[str, Any] = {}
    if motion_corrected is not None:
        section["motion_corrected"] = motion_corrected
    if fd_mean is not None:
        section["fd_mean"] = {
            "value": fd_mean, "status": grade("fd_mean", fd_mean, reference=reference), "note": "mean framewise displacement (mm)",
        }
    if fd_max is not None:
        section["fd_max"] = fd_max
    return section


def segmentation_qc_section(
    mask_volume_mm3: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a 'Segmentation' QC section: ``"mask_volume_mm3"`` (ungraded, if not None)
    plus the entries of ``extra`` merged in. Returns a dict (possibly empty)."""
    section: dict[str, Any] = {}
    if mask_volume_mm3 is not None:
        section["mask_volume_mm3"] = mask_volume_mm3
    if extra:
        section.update(extra)
    return section


def ai_model_qc_section(provenance: list[Any], extra_steps: tuple[str, ...] = ()) -> dict[str, Any]:
    """Build an 'AI / Deep-Learning Models' QC section from provenance records.

    A step is listed if its ``engine`` is ``"antstorch"`` (the antsx* convention for
    antstorch's deep models) or its ``step`` name is in ``extra_steps`` (deep-model steps
    recorded under another engine).

    Parameters
    ----------
    provenance : list of syntx.contract.ProvenanceEntry
        Needs ``step``, ``engine``, ``device``, ``seconds`` and (optionally) ``extra``.
    extra_steps : tuple of str, default ()
        Additional step names to list.

    Returns
    -------
    dict
        ``{step: {"value": "<engine> on <device>", "note": "<seconds>s[; <extra['source']>]"}}``;
        a later entry with the same step name replaces an earlier one.
    """
    section: dict[str, Any] = {}
    for p in provenance:
        if p.engine == "antstorch" or p.step in extra_steps:
            note = p.extra.get("source") if getattr(p, "extra", None) else None
            section[p.step] = {"value": f"{p.engine} on {p.device}", "note": f"{p.seconds:.2f}s" + (f"; {note}" if note else "")}
    return section
