"""Guard: probe-displacement pose calculation lives in ``src/syntx/pose.py`` and nowhere else.

Previous work had four separate copies of probe-displacement helpers across tests and benchmarks
(e.g. test_robust_affine_anisotropy.py, test_slab_capture.py, pyramid_compare.py, run_four.py).
This guard test scans Python source files across ``src/syntx``, ``tests``, and ``benchmarks``
(with comments and string literals removed) to ensure that no new local copy of the probe
generator, displacement loop, or max displacement metric is introduced outside ``syntx.pose``.

``ALLOW`` maps ``(relative_file, pattern)`` to ``(max_occurrences, reason)``. A new occurrence
fails the test: call ``syntx.pose`` (or ``syntx.pose_error`` / ``syntx.pose_error_report`` /
``syntx.probe_points`` / ``syntx.probe_displacements``) instead, or add an allow-list entry with
a documented reason.
"""

from __future__ import annotations

import io
from pathlib import Path
import re
import tokenize

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "syntx"
TESTS = ROOT / "tests"
BENCHMARKS = ROOT / "benchmarks"

CENTRAL_MODULES = {
    "src/syntx/pose.py",
    "tests/test_pose.py",
}

PATTERNS = {
    # Probe points corner generation (+-15, +-15, +-5 mm)
    "probe_corners_loop": r"for\s+x\s+in\s+\(\s*-\s*15\s*,\s*15\s*\)\s+for\s+y\s+in\s+\(\s*-\s*15\s*,\s*15\s*\)",
    # apply_to_point displacement loop over probe points: [tx.apply_to_point(q) - q for q in probes]
    "apply_to_point_displacement": r"\.apply_to_point\s*\([^)]*\)[^-\n]*-\s*[qp]\b",
    # Local probe loop comprehension: [...apply_to_point... for ... in probes]
    "probe_eval_loop": r"\[\s*[^\]]*\.apply_to_point\s*\([^\]]*\)[^\]]*for\s+[a-zA-Z_]\w*\s+in\s+probes\s*\]",
    # Local definitions of probes, displacements, or pose_error
    "def_local_pose": r"\bdef\s+(?:probes|displacements|pose_error)\s*\(",
}

ALLOW: dict[tuple[str, str], tuple[int, str]] = {
    # No external copies currently allowed; everything must use syntx.pose.
}


def _code_only(text: str) -> str:
    """Source with comments and string literals blanked out to preserve line numbers."""
    lines = text.splitlines(keepends=True)
    starts = [0]
    for ln in lines:
        starts.append(starts[-1] + len(ln))
    chars = list(text)
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                lo = starts[tok.start[0] - 1] + tok.start[1]
                hi = starts[tok.end[0] - 1] + tok.end[1]
                for i in range(lo, hi):
                    if chars[i] != "\n":
                        chars[i] = " "
    except Exception:
        return text
    return "".join(chars)


def scan() -> dict[tuple[str, str], list[int]]:
    """{(relative_path, pattern_name): [line_numbers]} for occurrences outside central modules."""
    found: dict[tuple[str, str], list[int]] = {}
    for folder in (SRC, TESTS, BENCHMARKS):
        if not folder.exists():
            continue
        for f in sorted(folder.rglob("*.py")):
            rel = f.relative_to(ROOT).as_posix()
            if rel in CENTRAL_MODULES:
                continue
            code = _code_only(f.read_text(encoding="utf-8"))
            for name, rx in PATTERNS.items():
                for m in re.finditer(rx, code):
                    line_no = code.count("\n", 0, m.start()) + 1
                    found.setdefault((rel, name), []).append(line_no)
    return found


def violations(found: dict[tuple[str, str], list[int]]) -> list[str]:
    bad = []
    for (rel, name), lines in sorted(found.items()):
        allowed, _ = ALLOW.get((rel, name), (0, ""))
        if len(lines) > allowed:
            bad.append(
                f"{rel}: {len(lines)} x '{name}' at line(s) {lines} (allowed {allowed}); "
                f"probe-displacement pose logic belongs in syntx.pose (pose_error, pose_error_report, "
                f"probe_points, probe_displacements) -- or add an ALLOW entry with a documented reason"
            )
    return bad


def test_no_probe_displacement_copies_outside_pose_module():
    """Verify no unauthorized probe displacement calculation exists outside syntx.pose."""
    bad = violations(scan())
    assert not bad, "\n".join(bad)


def test_allow_list_entries_are_valid():
    """Verify that all entries in ALLOW are valid and non-stale."""
    found = scan()
    for key, (n, reason) in ALLOW.items():
        assert reason and len(reason) > 10, f"{key}: allow-list entry needs a descriptive reason"
        assert (ROOT / key[0]).exists(), f"{key}: file no longer exists"
        have = len(found.get(key, []))
        assert have >= n or n == 0, f"{key}: allows {n} but only {have} found; lower allowance"


@pytest.mark.parametrize(
    "snippet,pattern",
    [
        (
            "probes = [c + np.array([x, y, z]) for x in (-15, 15) for y in (-15, 15) for z in (-5, 5)]",
            "probe_corners_loop",
        ),
        (
            "disp = [np.array(tx.apply_to_point(tuple(q))) - q for q in probes]",
            "apply_to_point_displacement",
        ),
        (
            "disp = [tx.apply_to_point(tuple(p)) for p in probes]",
            "probe_eval_loop",
        ),
        (
            "def pose_error(fixed, a, b):\n    pass",
            "def_local_pose",
        ),
    ],
)
def test_guard_catches_seeded_violations(tmp_path, snippet, pattern):
    """The guard would have caught the old duplicate copies."""
    test_file = tmp_path / "violation.py"
    test_file.write_text(snippet + "\n")
    code = _code_only(test_file.read_text())
    assert re.search(PATTERNS[pattern], code)


def test_comments_and_docstrings_ignored(tmp_path):
    """Prose mentioning probe points and apply_to_point in docstrings/comments is ignored."""
    code = '''"""
Documentation discussing [tx.apply_to_point(q) - q for q in probes] and
def pose_error(fixed, a, b) in docstring.
"""
# for x in (-15, 15) for y in (-15, 15) in comment
x = 1
'''
    stripped = _code_only(code)
    for rx in PATTERNS.values():
        assert not re.search(rx, stripped)
