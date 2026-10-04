"""syntx.viz.qc_sections — modality-independent QC-section builders and grading thresholds.

Each ``*_qc_section`` function returns one category of metrics for the ``qc_sections``
argument of :func:`syntx.viz.modality_report.write_modality_report`: a dict
``{metric_name: value}`` where a value is either a plain value or a dict
``{"value", "status", "note"}`` (``status`` one of "ok", "warn", "fail", "unknown"). Keeping
the builders here lets the antsx* pipelines grade the same metric the same way.

Metrics tied to one package's own naming scheme are left to that package.
"""

from __future__ import annotations

from typing import Any


def grade_edge_dice_tol1(edge_dice_tol1: float) -> str:
    """Grade an edge-map Dice with 1-voxel tolerance: >= 0.30 "ok", >= 0.15 "warn", else
    "fail"; NaN gives "unknown".

    The input is an overlap of high-gradient (edge) voxels between two images, as produced
    by antsxfunctional's edge-overlap metrics. It is on a much lower scale than a whole-mask
    Dice, so do not use ``grade_dice_overlap`` for it. The thresholds were set empirically
    from T1-to-PET registrations judged good by eye (which scored about 0.35-0.45).
    """
    if edge_dice_tol1 != edge_dice_tol1:  # NaN
        return "unknown"
    if edge_dice_tol1 >= 0.30:
        return "ok"
    if edge_dice_tol1 >= 0.15:
        return "warn"
    return "fail"


def grade_dice_overlap(dice: float) -> str:
    """Grade a Dice / mask overlap: >= 0.7 "ok", >= 0.5 "warn", else "fail"; NaN gives
    "unknown". The bins are a common heuristic for registration QC, not a standard."""
    if dice != dice:  # NaN
        return "unknown"
    if dice >= 0.7:
        return "ok"
    if dice >= 0.5:
        return "warn"
    return "fail"


def grade_framewise_displacement_mean(fd_mean: float) -> str:
    """Grade mean framewise displacement in mm: < 0.2 "ok", < 0.5 "warn", else "fail"; NaN
    gives "unknown". The bins are a common resting-state fMRI heuristic."""
    if fd_mean != fd_mean:
        return "unknown"
    if fd_mean < 0.2:
        return "ok"
    if fd_mean < 0.5:
        return "warn"
    return "fail"


def grade_folding_pct(folding_pct: float) -> str:
    """Grade Jacobian topology folding percentage: <= 0.0 "ok", <= 0.015 "warn", > 0.015 "fail"; NaN gives "unknown"."""
    if folding_pct != folding_pct:
        return "unknown"
    if folding_pct <= 0.0:
        return "ok"
    if folding_pct <= 0.015:
        return "warn"
    return "fail"


def grade_min_jacobian(min_j: float) -> str:
    """Grade minimum Jacobian determinant: >= 0.05 "ok", > 0.0 "warn", <= 0.0 "fail"; NaN gives "unknown"."""
    if min_j != min_j:
        return "unknown"
    if min_j >= 0.05:
        return "ok"
    if min_j > 0.0:
        return "warn"
    return "fail"


def grade_inverse_identity_error(ice_max_mm: float) -> str:
    """Grade max inverse identity error in mm: < 0.5 "ok", < 1.0 "warn", else "fail"; NaN gives "unknown"."""
    if ice_max_mm != ice_max_mm:
        return "unknown"
    if ice_max_mm < 0.5:
        return "ok"
    if ice_max_mm < 1.0:
        return "warn"
    return "fail"


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
) -> dict[str, Any]:
    """Build a 'Registration' QC section.

    Parameters
    ----------
    mutual_information : float, optional
        Stored ungraded under ``"mutual_information"``.
    dice : float, optional
        If given, stored as ``"dice_overlap": {"value", "status": grade_dice_overlap(dice),
        "note": dice_note}``.
    dice_note : str, default "brain/foreground mask overlap"
        Note for the Dice entry.
    folding_pct : float, optional
        Topology folding percentage (graded: <= 0.0 "ok", <= 0.015 "warn", > 0.015 "fail").
    min_jacobian : float, optional
        Minimum Jacobian determinant (graded: >= 0.05 "ok", > 0.0 "warn", <= 0.0 "fail").
    jac_measure : str, optional
        Method used for Jacobian (e.g. "Liouville determinant" or "finite difference").
    inverse_identity_max_mm : float, optional
        Maximum inverse identity error in mm (graded: < 0.5 "ok", < 1.0 "warn", else "fail").
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
        section["dice_overlap"] = {"value": dice, "status": grade_dice_overlap(dice), "note": dice_note}
    if folding_pct is not None:
        section["folding_pct"] = {
            "value": folding_pct,
            "status": grade_folding_pct(folding_pct),
            "note": f"folding voxels det(J)<=0% ({jac_measure or 'Liouville'})"
        }
    if min_jacobian is not None:
        section["min_jacobian"] = {
            "value": min_jacobian,
            "status": grade_min_jacobian(min_jacobian),
            "note": f"min det(J) ({jac_measure or 'Liouville'})"
        }
    if inverse_identity_max_mm is not None:
        section["inverse_identity_max_mm"] = {
            "value": inverse_identity_max_mm,
            "status": grade_inverse_identity_error(inverse_identity_max_mm),
            "note": "max ICE (mm)" + (f"; mean {inverse_identity_mean_mm:.3f}mm" if inverse_identity_mean_mm is not None else "")
        }
    if extra:
        section.update(extra)
    return section


def motion_qc_section(
    fd_mean: float | None = None,
    fd_max: float | None = None,
    motion_corrected: bool | None = None,
) -> dict[str, Any]:
    """Build a 'Motion' QC section with only the arguments that are not None.

    Keys: ``"motion_corrected"`` (as given), ``"fd_mean"`` (``{"value", "status":
    grade_framewise_displacement_mean(fd_mean), "note"}``; mm) and ``"fd_max"`` (ungraded).

    Returns
    -------
    dict
    """
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
