"""syntx.perf_tracking — cross-version performance-regression tracking for downstream
antsx* pipelines (antsxfunctional, antsxdwi, ...).

Not to be confused with :mod:`syntx.benchmark`, which benchmarks syntx's own registration
algorithms against Mindboggle/MSD reference data. This module instead lets a downstream
package's real-data demo/acceptance scripts record their own real QC/scientific metrics
(FA_mean, registration Dice, tSNR, CBF, runtime, ...) each time they run, append-only, to a
small JSON-lines log the caller checks into their own repo, and flag when a metric on a new
run has moved further than expected relative to its own history -- catching a real
regression (a code change that quietly makes registration worse, or a dependency bump that
silently changes a model's output) before it ships, the same way this session's own gate-
based validation reports do, but automatically and over time rather than one comparison at
a time.

Design principles:
  - One JSON-lines file per (modality, dataset) pair, checked into the calling repo (not
    stored here) -- diffable in PRs, tied to the commit that produced it, no new infra.
  - Metrics are an arbitrary flat dict[str, float] the caller supplies (this module has no
    opinion on what "FA_mean" or "tsnr" mean) -- any modality can use this the same way.
  - Regression detection compares a new run's metrics against the median of its own
    (modality, dataset)-scoped history, not a single previous run, so one noisy run does
    not permanently poison the baseline.
  - Flags by default, never blocks -- this module never raises on a detected regression;
    callers decide what to do with the returned list (log it, fail a test, ask a human).
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
    """Short git commit hash of the repo containing ``path``, or None if unavailable
    (not a git repo, git not installed, etc.) -- never raises."""
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
    """One metric that moved further from its own history than ``tolerance`` allows."""

    metric: str
    current: float
    baseline: float
    relative_difference: float
    tolerance: float
    direction: str  # "increase" or "decrease" -- which way the metric moved

    def to_dict(self) -> dict[str, Any]:
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
        Path to a ``.jsonl`` file (created if it does not exist; parent directory must
        exist). Convention: one file per (modality, dataset), e.g.
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
        Whatever real QC/scientific/runtime metrics the caller wants tracked. Any
        non-numeric value is coerced with ``float()``; values that cannot be coerced are
        dropped with a note in the returned record's ``dropped_metrics`` list, not raised.
    package_names : tuple of str
        Package names to look up installed versions for via ``importlib.metadata`` (e.g.
        ``("antsxfunctional", "syntx", "antstorch", "torch")``), merged into ``versions``.
    git_commit : str, optional
        Explicit git commit hash for the calling repo. If omitted, best-effort auto-
        detected via ``git rev-parse --short HEAD`` run in ``log_path``'s directory
        (never raises if unavailable).
    versions : dict[str, str], optional
        Explicit package-name -> version-string overrides, merged over the
        ``package_names`` lookup (explicit values win).
    extra : dict, optional
        Any additional free-form metadata to store alongside the record (e.g.
        ``{"device": "mps", "notes": "--fast 40-volume subset"}``).

    Returns
    -------
    dict
        The exact record written (also containing ``timestamp``, ``git_commit``,
        ``versions``, and ``dropped_metrics`` if any).
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
    """Load all records from ``log_path``, optionally filtered to a single
    (modality, dataset_id) pair. Returns an empty list if the file does not exist yet
    (a fresh log, not an error)."""
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
    """Compare ``current_metrics`` (a run NOT yet recorded via :func:`record_run`) against
    the median of the last ``window`` historical runs for the same (modality, dataset_id)
    in ``log_path``, and flag any metric whose relative difference from that median
    exceeds ``tolerance``.

    Parameters
    ----------
    log_path : str
        Same log file :func:`record_run` writes to.
    modality, dataset_id : str
        Scopes history to matching runs only.
    current_metrics : dict[str, float]
        The just-computed metrics to check -- call this BEFORE :func:`record_run` for
        the same run, so the new run does not contaminate its own baseline.
    tolerance : float or dict[str, float]
        Maximum allowed absolute relative difference from the historical median before a
        metric is flagged (e.g. 0.15 = 15%). A single float applies to every metric; a
        dict overrides specific metric names, falling back to a dict's own ``"default"``
        key or 0.15 if a metric is not listed.
    window : int
        Number of most recent matching historical runs to compute the median baseline
        from (default 5). If fewer than 2 historical runs exist, nothing is flagged (there
        is no meaningful baseline yet) -- this is the expected, non-error state for a
        metric's first and second runs.
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
    """Convenience wrapper: check for regressions against existing history FIRST, then
    record the new run. Returns ``(regression_flags, record)``.

    This ordering matters: checking after recording would compare the new run against a
    baseline that already includes itself.
    """
    flags = detect_regressions(
        log_path, modality, dataset_id, metrics, tolerance=tolerance, window=window, higher_is_better=higher_is_better
    )
    record = record_run(
        log_path, modality, dataset_id, metrics,
        package_names=package_names, git_commit=git_commit, versions=versions, extra=extra,
    )
    return flags, record
