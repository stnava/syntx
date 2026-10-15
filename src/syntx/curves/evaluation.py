"""Evaluation metrics and ground-truth validation for N-dimensional curve matching."""

from __future__ import annotations

from typing import Sequence
import numpy as np

from syntx.curves.matching import CurveMatch


def correspondence_error(
    match: CurveMatch,
    gt_P: np.ndarray | Sequence[float],
    gt_Q: np.ndarray | Sequence[float],
) -> dict:
    """Evaluate correspondence accuracy against ground-truth landmark locations.

    Parameters
    ----------
    match : CurveMatch
        Fitted curve match outcome from ``match_curves(P, Q)``.
    gt_P : array_like of float
        Known landmark arc fractions along curve P in [0, 1] (or [0, 1) for closed).
    gt_Q : array_like of float
        True matching landmark arc fractions along curve Q.

    Returns
    -------
    dict
        Dictionary containing landmark-wise errors and summary metrics:
        - ``gt_P``: Ground-truth positions on P.
        - ``gt_Q``: Ground-truth positions on Q.
        - ``est_Q``: Estimated positions on Q via ``match.map_P_to_Q(gt_P)``.
        - ``error_fraction``: Arc-fraction absolute errors.
        - ``mean_error_fraction``: Mean absolute arc-fraction error.
        - ``max_error_fraction``: Maximum absolute arc-fraction error.
        - ``mean_error_mm``: Mean physical error along curve Q (mm).
        - ``max_error_mm``: Maximum physical error along curve Q (mm).
    """
    p_arr = np.asarray(gt_P, dtype=np.float64)
    q_true = np.asarray(gt_Q, dtype=np.float64)
    q_est = np.asarray(match.map_P_to_Q(p_arr), dtype=np.float64)

    if match.mode == "closed":
        diff = np.abs(q_est - q_true)
        diff = np.minimum(diff, 1.0 - diff)
    else:
        diff = np.abs(q_est - q_true)

    total_len_Q = float(match.spacing_mm * match.n_points)
    err_mm = diff * total_len_Q

    return {
        "gt_P": p_arr,
        "gt_Q": q_true,
        "est_Q": q_est,
        "error_fraction": diff,
        "mean_error_fraction": float(np.mean(diff)),
        "max_error_fraction": float(np.max(diff)),
        "mean_error_mm": float(np.mean(err_mm)),
        "max_error_mm": float(np.max(err_mm)),
    }
