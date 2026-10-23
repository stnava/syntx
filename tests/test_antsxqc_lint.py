"""Architectural lint check: ensure syntx delegates QC decisions to antsxqc."""

from __future__ import annotations

from pathlib import Path

from antsxqc.lint import assert_clean


def test_antsxqc_architectural_lint() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "syntx"
    assert_clean(str(root))
