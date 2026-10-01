"""syntx.perf_tracking — metric history and regression flags for downstream antsx* pipelines
(antsxfunctional, antsxdwi, ...).

Not :mod:`syntx.benchmark` (which benchmarks syntx's own registration methods). Here a
downstream script records the metrics of each run (FA_mean, Dice, tSNR, runtime, ...) as one
line of an append-only JSON-lines log, and a new run is compared with the median of the
recent history for the same (modality, dataset_id).

- ``record_run``: append one record (timestamp, metrics, git commit, package versions).
- ``load_history``: read records, optionally filtered by modality / dataset_id.
- ``detect_regressions``: relative difference of each metric from the median of the last
  ``window`` matching runs; returns ``RegressionFlag`` objects, never raises on a regression.
- ``record_run_and_check``: check first, then record.

Conventions (not enforced): one log file per (modality, dataset), kept in the calling repo
(e.g. ``reports/performance_history/<name>.jsonl``); metrics are a flat name -> float dict
whose meaning this module does not interpret.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def _best_effort_git_commit(path: str) -> str | None:
    """Short git commit hash (``git rev-parse --short HEAD``, run in ``path`` if it is a
    directory, else in its parent directory; 5 s timeout), or None on any failure (not a repo,
    no git, directory does not exist yet, ...). Never raises."""
    try:
        directory = path if os.path.isdir(path) else os.path.dirname(path) or "."
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=directory, capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() or None if result.returncode == 0 else None
    except Exception:
        return None


def _package_versions(package_names: tuple[str, ...]) -> dict[str, str]:
    """Best-effort installed-version lookup for a list of package names (e.g.
    ``("antsxfunctional", "syntx", "antstorch")``) via ``importlib.metadata``. Missing or
    unversioned packages are simply omitted, never raise."""
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    for name in package_names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
        except Exception:
            continue
    return versions


@dataclass
class RegressionFlag:
    """One metric that moved further from its own history than ``tolerance`` allows.

    Attributes
    ----------
    metric : str
    current : float
        Value of the run being checked.
    baseline : float
        Median of the metric over the recent history.
    relative_difference : float
        ``(current - baseline) / abs(baseline)`` (signed).
    tolerance : float
        Tolerance that was exceeded.
    direction : str
        "increase" if ``relative_difference > 0``, else "decrease".
    """

    metric: str
    current: float
    baseline: float
    relative_difference: float
    tolerance: float
    direction: str  # "increase" or "decrease" -- which way the metric moved

    def to_dict(self) -> dict[str, Any]:
        """Return the six fields as a plain dict."""
        return {
            "metric": self.metric,
            "current": self.current,
            "baseline": self.baseline,
            "relative_difference": self.relative_difference,
            "tolerance": self.tolerance,
            "direction": self.direction,
        }


def record_run(
    log_path: str,
    modality: str,
    dataset_id: str,
    metrics: dict[str, float],
    package_names: tuple[str, ...] = (),
    git_commit: str | None = None,
    versions: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one run's metrics to ``log_path`` as a single JSON line.

    Parameters
    ----------
    log_path : str
        Path to a ``.jsonl`` file; the file and its parent directories are created if
        missing, and the record is appended. Convention: one file per (modality, dataset), e.g.
        ``reports/performance_history/rsfmri_socom_blast01.jsonl``, checked into the
        calling repo like any other tracked file.
    modality : str
        Free-text modality label (e.g. ``"rsfmri"``, ``"dwi"``, ``"pet"``) -- stored, not
        interpreted.
    dataset_id : str
        Free-text dataset/session label (e.g. ``"SOCOM_sub-Blast-01_ses-01"``) -- stored,
        not interpreted. Regression detection scopes its history to matching
        (modality, dataset_id) pairs within the same log file.
    metrics : dict[str, float]
        Metrics to track. Every value is converted with ``float()``; values that raise
        TypeError / ValueError are dropped and their names listed in ``dropped_metrics``
        (NaN / inf pass and are written as the non-standard JSON tokens ``NaN`` /
        ``Infinity``).
    package_names : tuple of str, default ()
        Package names to look up installed versions for via ``importlib.metadata`` (e.g.
        ``("antsxfunctional", "syntx", "antstorch", "torch")``), merged into ``versions``.
    git_commit : str, optional
        Explicit git commit hash for the calling repo. If omitted, best-effort auto-
        detected via ``git rev-parse --short HEAD`` run in ``log_path``'s directory (None if
        that fails, including when the directory does not exist before this call).
    versions : dict[str, str], optional
        Explicit package-name -> version-string overrides, merged over the
        ``package_names`` lookup (explicit values win).
    extra : dict, optional
        Any additional free-form metadata to store alongside the record (e.g.
        ``{"device": "mps", "notes": "--fast 40-volume subset"}``).

    Returns
    -------
    dict
        The record written: ``timestamp`` (UTC ISO 8601), ``modality``, ``dataset_id``,
        ``metrics``, ``git_commit``, ``versions``, and when non-empty ``dropped_metrics`` and
        ``extra``.
    """
    clean_metrics: dict[str, float] = {}
    dropped: list[str] = []
    for key, value in metrics.items():
        try:
            clean_metrics[key] = float(value)
        except (TypeError, ValueError):
            dropped.append(key)

    resolved_versions = _package_versions(package_names)
    if versions:
        resolved_versions.update(versions)

    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modality": modality,
        "dataset_id": dataset_id,
        "metrics": clean_metrics,
        "git_commit": git_commit if git_commit is not None else _best_effort_git_commit(log_path),
        "versions": resolved_versions,
    }
    if dropped:
        record["dropped_metrics"] = dropped
    if extra:
        record["extra"] = extra

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    return record


def load_history(log_path: str, modality: str | None = None, dataset_id: str | None = None) -> list[dict[str, Any]]:
    """Return the records of ``log_path`` in file order, optionally only those whose
    ``modality`` / ``dataset_id`` equal the given values (None = no filter on that field).

    Blank lines are skipped; a malformed line raises ``json.JSONDecodeError``. A missing file
    gives ``[]``."""
    if not os.path.exists(log_path):
        return []
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if modality is not None and record.get("modality") != modality:
                continue
            if dataset_id is not None and record.get("dataset_id") != dataset_id:
                continue
            records.append(record)
    return records


def detect_regressions(
    log_path: str,
    modality: str,
    dataset_id: str,
    current_metrics: dict[str, float],
    tolerance: float | dict[str, float] = 0.15,
    window: int = 5,
    higher_is_better: dict[str, bool] | None = None,
) -> list[RegressionFlag]:
    """Flag metrics of a new run that differ from their recent median by more than ``tolerance``.

    History is the last ``window`` records of ``log_path`` with the same (modality,
    dataset_id). For each metric in ``current_metrics``: baseline = median of its values in
    those records (skipped if fewer than 2 of them have it, or if the median is 0); relative
    difference = (current - baseline) / |baseline|; flagged if its absolute value exceeds the
    metric's tolerance and, when ``higher_is_better`` lists the metric, it moved the bad way.

    Parameters
    ----------
    log_path : str
        Same log file :func:`record_run` writes to.
    modality, dataset_id : str
        Scopes history to matching runs only.
    current_metrics : dict[str, float]
        The just-computed metrics to check -- call this BEFORE :func:`record_run` for
        the same run, so the new run does not contaminate its own baseline. Values must be
        numeric (not converted); a NaN value is never flagged.
    tolerance : float or dict[str, float], default 0.15
        Maximum allowed absolute relative difference from the historical median before a
        metric is flagged (e.g. 0.15 = 15%). A single float applies to every metric; a
        dict overrides specific metric names, falling back to a dict's own ``"default"``
        key or 0.15 if a metric is not listed.
    window : int, default 5
        Number of most recent matching runs used for the median. With fewer than 2 matching
        runs in total nothing is flagged; a metric present in fewer than 2 of the last
        ``window`` runs is not checked.
    higher_is_better : dict[str, bool], optional
        For metrics where only one direction of change is actually a regression (e.g. a
        Dice overlap score getting worse only if it goes DOWN, not up), map the metric
        name to True (flag only decreases) or False (flag only increases). Metrics not
        listed are flagged on movement in either direction (the default, appropriate for
        most QC metrics where "moved a lot" is itself the signal worth a human look).

    Returns
    -------
    list of RegressionFlag
        Empty if nothing is flagged (including when there is no baseline history yet).
    """
    history = load_history(log_path, modality=modality, dataset_id=dataset_id)
    if len(history) < 2:
        return []

    recent = history[-window:]
    higher_is_better = higher_is_better or {}
    flags: list[RegressionFlag] = []

    for metric_name, current_value in current_metrics.items():
        baseline_values = [
            r["metrics"][metric_name] for r in recent if metric_name in r.get("metrics", {})
        ]
        if len(baseline_values) < 2:
            continue

        baseline = statistics.median(baseline_values)
        if baseline == 0:
            continue

        relative_diff = (current_value - baseline) / abs(baseline)
        metric_tolerance = (
            tolerance if isinstance(tolerance, (int, float)) else tolerance.get(metric_name, tolerance.get("default", 0.15))
        )

        direction = "increase" if relative_diff > 0 else "decrease"
        if metric_name in higher_is_better:
            # higher_is_better[metric] True -> only a decrease is a regression.
            # higher_is_better[metric] False -> only an increase is a regression.
            wanted_direction = "decrease" if higher_is_better[metric_name] else "increase"
            if direction != wanted_direction:
                continue

        if abs(relative_diff) > metric_tolerance:
            flags.append(
                RegressionFlag(
                    metric=metric_name,
                    current=current_value,
                    baseline=baseline,
                    relative_difference=relative_diff,
                    tolerance=metric_tolerance,
                    direction=direction,
                )
            )

    return flags


def record_run_and_check(
    log_path: str,
    modality: str,
    dataset_id: str,
    metrics: dict[str, float],
    tolerance: float | dict[str, float] = 0.15,
    window: int = 5,
    higher_is_better: dict[str, bool] | None = None,
    package_names: tuple[str, ...] = (),
    git_commit: str | None = None,
    versions: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> tuple[list[RegressionFlag], dict[str, Any]]:
    """Run :func:`detect_regressions` on the existing history, then :func:`record_run` the
    new metrics (appending to ``log_path``). Returns ``(regression_flags, record)``.

    Checking first keeps the new run out of its own baseline. Parameters are those of the two
    functions.
    """
    flags = detect_regressions(
        log_path, modality, dataset_id, metrics, tolerance=tolerance, window=window, higher_is_better=higher_is_better
    )
    record = record_run(
        log_path, modality, dataset_id, metrics,
        package_names=package_names, git_commit=git_commit, versions=versions, extra=extra,
    )
    return flags, record
