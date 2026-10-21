"""Guard: no hard-coded phase-encode axis/direction defaults in ``src/syntx`` outside ``phase_encode.py``.

The audit (2026-10-11) found silent ``(0, 1, 0)`` / ``"j"`` / ``pe_axis=1`` defaults in several repos, each turning
missing or malformed metadata into "y". The phase-encode direction is resolved in ONE module, ``syntx/phase_encode.py``;
everything else must pass a resolved ``PhaseEncode`` / explicit weights / an explicit ``'unrestricted'`` policy.

Comments and docstrings (triple-quoted strings) are removed before scanning. ``ALLOW`` maps ``(file, pattern)`` to
``(maximum occurrences, reason)``; a new occurrence fails the test: resolve through ``syntx.phase_encode`` instead, or
add an entry whose reason says why it is NOT a phase-encode default.
"""
from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "syntx"
CENTRAL = "phase_encode.py"

PATTERNS = {
    "tuple_010": r"\(\s*0(?:\.0*)?\s*,\s*1(?:\.0*)?\s*,\s*0(?:\.0*)?\s*\)",
    "list_010": r"\[\s*0(?:\.0*)?\s*,\s*1(?:\.0*)?\s*,\s*0(?:\.0*)?\s*\]",
    "pe_axis_1": r"\bpe_axis\s*(?::\s*[\w\[\], |.]+)?=\s*1\b",
    # an assignment / default of a bare BIDS label to something named like a phase-encode variable
    "pe_label_default": r"\b\w*(?:pe|phase)\w*\s*(?::\s*[\w\[\], |.]+)?=\s*['\"][ijk][-+]?['\"]",
    "anatomical_axis_default": r"\banatomical_axis\s*(?::\s*[\w\[\], |.]+)?=\s*['\"]",
}

ALLOW = {
    ("transforms.py", "tuple_010"): (1, "unit y vector that probes the linear part of an affine transform in a geometry self-check: not a phase-encode default"),
    ("robust_affine.py", "list_010"): (5, "rotation-matrix rows of a rotation about y (np.array([[c,0,s],[0,1,0],[-s,0,c]])), not a phase-encode axis"),
    ("slab_capture.py", "list_010"): (1, "rotation-matrix row of a rotation about y, not a phase-encode axis"),
    ("landmarks/sift3d.py", "list_010"): (2, "unit-vector table of the six axis neighbours of the SIFT3D descriptor"),
}


def _strip(text: str) -> str:
    out = []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError):
        return text
    for t in toks:
        if t.type == tokenize.COMMENT:
            continue
        if t.type == tokenize.STRING and t.string.lstrip("rbfuRBFU")[:3] in ('"""', "'''"):
            out.append((t.start, '""'))
            continue
        out.append((t.start, t.string))
    # rebuild preserving only the code text (positions irrelevant for counting)
    return " ".join(s for _, s in out)


def _scan():
    hits = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel == CENTRAL:
            continue
        code = _strip(path.read_text(encoding="utf-8"))
        for name, pat in PATTERNS.items():
            n = len(re.findall(pat, code))
            if n:
                hits[(rel, name)] = n
    return hits


def test_no_hardcoded_phase_encode_defaults_outside_phase_encode_module():
    hits = _scan()
    bad = []
    for key, n in hits.items():
        allowed = ALLOW.get(key, (0, ""))[0]
        if n > allowed:
            bad.append(f"{key[0]}: {n} x {key[1]} (allowed {allowed})")
    assert not bad, (
        "hard-coded phase-encode default(s) found; resolve through syntx.phase_encode "
        "(resolve_phase_encode / restrict_for_image / explicit pe_policy='unrestricted'):\n  " + "\n  ".join(bad)
    )


def test_allow_list_entries_are_still_needed_and_have_reasons():
    hits = _scan()
    for key, (n, reason) in ALLOW.items():
        assert reason.strip(), f"{key} has no reason"
        assert hits.get(key, 0) == n, f"{key}: allow-list says {n}, found {hits.get(key, 0)}; update ALLOW"


def test_patterns_catch_seeded_violations():
    seeded = {
        "tuple_010": "def f(restrict_transformation=(0, 1, 0)): pass",
        "list_010": "w = [0, 1, 0]",
        "pe_axis_1": "def f(pe_axis: int = 1): pass",
        "pe_label_default": "def f(phase_encoding_direction='j'): pass",
        "anatomical_axis_default": "def f(anatomical_axis='AP'): pass",
    }
    for name, code in seeded.items():
        assert re.search(PATTERNS[name], _strip(code)), name
    # docstrings and comments are ignored
    assert not re.search(PATTERNS["tuple_010"], _strip('"""e.g. (0, 1, 0)"""\n# (0, 1, 0)\nx = 1\n'))
