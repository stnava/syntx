"""Grep-based enforcement of docs/antsx_implementation_standards.md.

A prose standard nobody checks gets silently violated. This test greps source files for
the specific patterns the standards doc names and fails if a new, unlisted violation
appears. Each exception below must have a one-line reason -- an unexplained exception is
itself a review finding.

Adapted from antsxdwi's tests/test_implementation_standards.py. Inside syntx itself, rule 2
("registration goes through syntx, not plain ants") applies to syntx's OWN internal
primitives: a plain ants.registration/ants.motion_correction call is only acceptable as an
explicit, named legacy mode, a benchmark baseline, or a documented not-yet-migrated
primitive -- never an unmarked default working path.
"""
from pathlib import Path

SRC = Path(__file__).parent.parent / "src" / "syntx"

# (relative path, line-content substring) -> reason this exception is allowed.
# Every entry here must be a genuine, narrow, documented exception -- not a way to make
# this test pass without fixing the underlying call site.
ALLOWED_REGISTRATION_EXCEPTIONS: dict[tuple[str, str], str] = {
    ("motion.py", 'reg = ants.registration('):
        "motion_correction's inner per-frame loop: a direct port of ants.motion_correction, "
        "not yet a torch-native reimplementation -- documented architectural gap, see "
        "standards doc rule 2",
    ("template.py", 'w1 = ants.registration('):
        "build_template is a documented 'Direct port of ants.build_template' (see its own "
        "docstring) -- not yet a torch-native deformable template solver",
    ("robust_affine.py", "reg_a = ants.registration("):
        "mode='ants'/'ants_fast': an explicitly named, non-default legacy affine mode "
        "(see the function's own docstring), not a silent fallback",
    ("benchmark/high_level.py", "res = ants.registration(**reg_args)"):
        "benchmark harness's explicit 'ants'/'ants_syn' baseline model, run only when asked "
        "to compare syntx against legacy ANTs -- the entire point of this code path",
    ("benchmark/evaluate.py", "res_reg = ants.registration("):
        "benchmark harness's explicit 'ants'/'ants_syn' baseline model, same as high_level.py",
    ("robust_affine.py", "selection + `ants.registration(type_of_transform='Affine')`).  Not"):
        "docstring prose describing the mode='ants_fast' legacy path, not a call site",
    ("viz/reports.py", "<td><strong>Standard ANTs Affine</strong><br><code>ants.registration('Affine')</code></td>"):
        "HTML report template text describing a comparison row label, not a call site",
}

ALLOWED_REORIENT_EXCEPTIONS: dict[tuple[str, str], str] = {
    ("viz/core.py", '- Canonical LPI space reorientation (`reorient_image2("LPI")`).'):
        "module docstring prose describing the visualizer's behavior, not a call site",
    ("viz/core.py", 'img_proc = img.reorient_image2("LPI")'):
        "AnatomicalVisualizer canonical display reorientation; result feeds only the "
        "plotting path, never returned to the caller as computed data",
    ("viz/core.py", 'img_ants = img_ants.reorient_image2("LPI")'):
        "same visualizer display path, numpy-array-rebuilt-image branch",
    ("viz/figures.py", 'try: fixed_img = fixed.reorient_image2("LPI")'):
        "registration-report figure rendering only; result -> matplotlib, not reused",
    ("viz/figures.py", 'try: moving_img = moving.reorient_image2("LPI")'):
        "registration-report figure rendering only; result -> matplotlib, not reused",
    ("viz/figures.py", 'try: fl_img = fixed_labels.reorient_image2("LPI")'):
        "label-overlay figure rendering only; result -> matplotlib, not reused",
    ("viz/figures.py", 'try: wl_img = warped_labels.reorient_image2("LPI")'):
        "label-overlay figure rendering only; result -> matplotlib, not reused",
    ("viz/figures.py", 'try: fi_img = fixed_image.reorient_image2("LPI")'):
        "label-overlay figure rendering only; result -> matplotlib, not reused",
}

ALLOWED_PREPROCESS_EXCEPTIONS: dict[tuple[str, str], str] = {
    ("landmarks/preprocess.py", "img = ants.n4_bias_field_correction(img, shrink_factor=4)"):
        "fallback after antstorch.n4_bias_field_correction raises -- current except clause "
        "is a bare Exception (too broad, matches the 'junk fallback' pattern already fixed "
        "in antsxslowflow/antsxdwi); recorded here as a known gap, not silently reintroduced",
}


def _grep(pattern: str) -> list[tuple[Path, int, str]]:
    hits = []
    for f in SRC.rglob("*.py"):
        for i, line in enumerate(f.read_text().splitlines(), start=1):
            if pattern in line:
                hits.append((f, i, line.strip()))
    return hits


def _check(pattern: str, allowed: dict[tuple[str, str], str], rule_name: str):
    hits = _grep(pattern)
    unexplained = []
    for path, lineno, line in hits:
        rel = str(path.relative_to(SRC))
        key = (rel, line)
        if key not in allowed:
            unexplained.append(f"{rel}:{lineno}: {line}")
    assert not unexplained, (
        f"Unlisted violation(s) of standards rule '{rule_name}' "
        f"(docs/antsx_implementation_standards.md):\n" + "\n".join(unexplained) +
        "\n\nEither fix the call site, or add a reasoned exception to "
        "ALLOWED_*_EXCEPTIONS in this test with a one-line justification."
    )


def test_no_unexplained_ants_registration_calls():
    """Rule 2: syntx's own primitives should not silently regress to plain ants.registration."""
    _check("ants.registration(", ALLOWED_REGISTRATION_EXCEPTIONS, "syntx-internal-registration")


def test_no_unexplained_ants_motion_correction_calls():
    """Rule 2: no unmarked ants.motion_correction usage inside syntx."""
    _check("ants.motion_correction(", ALLOWED_REGISTRATION_EXCEPTIONS, "syntx-internal-registration")


def test_no_unexplained_reorient_calls():
    """Rule 1: reorient_image* only at a documented, named boundary."""
    hits = _grep("reorient_image")
    unexplained = [
        f"{p.relative_to(SRC)}:{ln}: {line}"
        for p, ln, line in hits
        if (str(p.relative_to(SRC)), line) not in ALLOWED_REORIENT_EXCEPTIONS
    ]
    assert not unexplained, (
        "reorient_image* call(s) without a documented reason (see rule 1 in "
        "docs/antsx_implementation_standards.md -- every call site needs a one-line "
        "comment naming the specific downstream consumer that requires it):\n"
        + "\n".join(unexplained)
    )


def test_no_unexplained_ants_preprocessing_fallbacks():
    """Rule 3: ants.n4_bias_field_correction/ants.denoise_image only as a documented,
    narrowly-scoped fallback after antstorch, never the unmarked default."""
    hits = _grep("ants.n4_bias_field_correction(") + _grep("ants.denoise_image(")
    unexplained = [
        f"{p.relative_to(SRC)}:{ln}: {line}"
        for p, ln, line in hits
        if (str(p.relative_to(SRC)), line) not in ALLOWED_PREPROCESS_EXCEPTIONS
    ]
    assert not unexplained, (
        "ants.n4_bias_field_correction/ants.denoise_image call(s) without a documented "
        "reason (see rule 3 in docs/antsx_implementation_standards.md):\n"
        + "\n".join(unexplained)
    )
