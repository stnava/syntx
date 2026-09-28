"""Tests for syntx.perf_tracking."""

from syntx.perf_tracking import detect_regressions, load_history, record_run, record_run_and_check


def test_record_run_writes_jsonl(tmp_path):
    log_path = str(tmp_path / "rsfmri_socom.jsonl")
    record = record_run(log_path, "rsfmri", "SOCOM_sub-Blast-01", {"tsnr": 15.0, "fd_mean": 0.12})
    assert record["modality"] == "rsfmri"
    assert record["metrics"]["tsnr"] == 15.0
    history = load_history(log_path)
    assert len(history) == 1
    assert history[0]["dataset_id"] == "SOCOM_sub-Blast-01"


def test_record_run_drops_non_numeric_metrics(tmp_path):
    log_path = str(tmp_path / "log.jsonl")
    record = record_run(log_path, "dwi", "d1", {"fa_mean": 0.4, "notes": "not a number"})
    assert "fa_mean" in record["metrics"]
    assert "notes" not in record["metrics"]
    assert record["dropped_metrics"] == ["notes"]


def test_load_history_filters_by_modality_and_dataset(tmp_path):
    log_path = str(tmp_path / "log.jsonl")
    record_run(log_path, "rsfmri", "d1", {"x": 1.0})
    record_run(log_path, "rsfmri", "d2", {"x": 2.0})
    record_run(log_path, "pet", "d1", {"x": 3.0})
    assert len(load_history(log_path)) == 3
    assert len(load_history(log_path, modality="rsfmri")) == 2
    assert len(load_history(log_path, dataset_id="d1")) == 2
    assert len(load_history(log_path, modality="rsfmri", dataset_id="d1")) == 1


def test_detect_regressions_needs_at_least_two_prior_runs(tmp_path):
    log_path = str(tmp_path / "log.jsonl")
    record_run(log_path, "rsfmri", "d1", {"tsnr": 15.0})
    # Only 1 prior run -- no baseline yet, nothing should be flagged.
    flags = detect_regressions(log_path, "rsfmri", "d1", {"tsnr": 2.0})
    assert flags == []


def test_detect_regressions_flags_large_relative_change(tmp_path):
    log_path = str(tmp_path / "log.jsonl")
    for v in (15.0, 14.5, 15.5):
        record_run(log_path, "rsfmri", "d1", {"tsnr": v})
    # Median baseline ~15.0; 8.0 is a >15% drop.
    flags = detect_regressions(log_path, "rsfmri", "d1", {"tsnr": 8.0}, tolerance=0.15)
    assert len(flags) == 1
    assert flags[0].metric == "tsnr"
    assert flags[0].direction == "decrease"


def test_detect_regressions_respects_per_metric_tolerance(tmp_path):
    log_path = str(tmp_path / "log.jsonl")
    for v in (15.0, 15.0, 15.0):
        record_run(log_path, "rsfmri", "d1", {"tsnr": v})
    # 20% drop: flagged under a strict 0.05 tolerance for this metric, not under 0.5 default.
    flags = detect_regressions(
        log_path, "rsfmri", "d1", {"tsnr": 12.0}, tolerance={"tsnr": 0.05, "default": 0.5}
    )
    assert len(flags) == 1


def test_detect_regressions_higher_is_better_ignores_improvements(tmp_path):
    log_path = str(tmp_path / "log.jsonl")
    for v in (0.80, 0.81, 0.80):
        record_run(log_path, "rsfmri", "d1", {"dice": v})
    # Dice went UP a lot -- an improvement, not a regression, and higher_is_better=True
    # means only a DECREASE should ever be flagged.
    flags = detect_regressions(
        log_path, "rsfmri", "d1", {"dice": 0.99}, tolerance=0.05, higher_is_better={"dice": True}
    )
    assert flags == []
    flags = detect_regressions(
        log_path, "rsfmri", "d1", {"dice": 0.50}, tolerance=0.05, higher_is_better={"dice": True}
    )
    assert len(flags) == 1
    assert flags[0].direction == "decrease"


def test_record_run_and_check_does_not_contaminate_own_baseline(tmp_path):
    log_path = str(tmp_path / "log.jsonl")
    for v in (15.0, 15.0, 15.0):
        record_run(log_path, "rsfmri", "d1", {"tsnr": v})
    flags, record = record_run_and_check(log_path, "rsfmri", "d1", {"tsnr": 5.0}, tolerance=0.15)
    assert len(flags) == 1
    # The regressed run is still recorded (never blocks), and is now part of history.
    assert len(load_history(log_path)) == 4


def test_record_run_captures_git_commit_and_versions(tmp_path):
    log_path = str(tmp_path / "log.jsonl")
    record = record_run(log_path, "rsfmri", "d1", {"x": 1.0}, versions={"antsxfunctional": "0.1.0"})
    assert record["versions"]["antsxfunctional"] == "0.1.0"
    # git_commit is best-effort; either a short hash string or None, never raises.
    assert record["git_commit"] is None or isinstance(record["git_commit"], str)
