"""Guard: pyramid / downsample code lives in ``src/syntx/pyramid.py`` and nowhere else.

The thin-slab neuromelanin failure came from several independent pyramid implementations that disagreed about thin or anisotropic
images (a floor in one copy that never reached the others, a voxel pool that collapsed an 11-slice slab to one level, ...). This test
scans every module of ``src/syntx`` (comments and string literals removed) for the operators a pyramid level is built from, and for
re-implementations of the level spacing, and fails on any occurrence outside ``pyramid.py`` that is not in ``ALLOW``.

``ALLOW`` maps ``(file, pattern)`` to ``(maximum number of occurrences, reason)``. A new occurrence in an allowed file raises the count
and fails the test: add the entry (with a reason that says why this is NOT an image pyramid level) or, better, call
``syntx.pyramid``. Image levels are built with ``build_pyramid`` / ``plan_levels``; level spacings come from ``level_spacing``.
"""
from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "syntx"
CENTRAL = "pyramid.py"

PATTERNS = {
    # the operator of a voxel-pooled level (kernel = stride = level)
    "avg_pool": r"\bavg_pool[123]d\b",
    # a resampled level (Gaussian + F.interpolate, align_corners)
    "interpolate": r"\binterpolate\s*\(",
    "jax_resize": r"\bjax\.image\.resize\b",
    "zoom": r"\bzoom\s*\(",
    # ANTs-side resampling used to cap or shrink resolution
    "ants_resample_image": r"\bresample_image\s*\(",
    "ants_resample_to_target": r"\bresample_image_to_target\s*\(",
    # per-file level-spacing / level-builder re-implementations
    "def_level_spacing": r"\bdef\s+_?level_spacing\b",
    "def_pyramid_builder": r"\bdef\s+\w*pyramid\w*\s*\(",
    "corner_spacing_formula": r"\(\s*\w+\s*-\s*1\s*\)\s*/\s*\(\s*\w+\s*-\s*1\s*\)",
    "log2_level_sigma": r"\blog2\s*\(\s*(?:level|lvl|s|scale|factor)\b",
}

ALLOW = {
    # ---- avg_pool: box windows / SSIM / MS-SSIM, none builds a registration pyramid level
    ("core/losses/cross_correlation.py", "avg_pool"): (8, "stride-1 box windows of the LNCC / NCC losses (not a downsample)"),
    ("landmarks/mind.py", "avg_pool"): (3, "stride-1 box means of the MIND descriptor"),
    ("joint_label_fusion.py", "avg_pool"): (2, "stride-1 box sum (patch similarity)"),
    ("image_compare.py", "avg_pool"): (6, "SSIM windows and the factor-2 downsample of ms_ssim (a metric's own scale space, not a registration level)"),
    # ---- interpolate
    ("scattered/solver.py", "interpolate"): (4, "scattered-point solver: levels are caller-prescribed grid shapes (a resize to a prescribed grid, not an image pyramid level); nearest-neighbour mask resizes"),
    ("core/grid.py", "interpolate"): (1, "resize_field: the one resize of a displacement / velocity field to a prescribed grid"),
    ("core/regularizers.py", "interpolate"): (1, "resizes a user mask to the field grid"),
    ("core/losses/deep_features.py", "interpolate"): (6, "resizes 2-D slices to the input size of a pretrained network"),
    ("classifier.py", "interpolate"): (1, "resizes an image to the classifier's training shape"),
    ("generators.py", "interpolate"): (2, "synthetic low-resolution displacement / bias field upsampling"),
    ("tvf.py", "interpolate"): (3, "velocity-field resize between levels (field, not image) and the fast-smooth gradient down/up-sample"),
    ("tvf_adj.py", "interpolate"): (1, "velocity-field resize between levels (field, not image)"),
    ("syn.py", "interpolate"): (23, "warp-field resizes between levels, 2.5-D deep-feature slices, and the CoM-init low-resolution grid (corner-aligned, spacing from level_spacing)"),
    ("tvf_jax.py", "jax_resize"): (2, "fast-smooth gradient down/up-sample inside the update (a gradient field, not an image level)"),
    # ---- ANTs resampling
    ("dewarp.py", "ants_resample_image"): (1, "isotropic motion reference for the dewarp (a registration target, not a pyramid level)"),
    ("dewarp.py", "ants_resample_to_target"): (1, "_apply_optional_mask: nearest-neighbour resampling of a caller mask onto the image grid (not a pyramid level)"),
    ("benchmark/msd.py", "ants_resample_image"): (2, "benchmark data loader: resamples MSD volumes to the benchmark spacing"),
    ("data/msd.py", "ants_resample_image"): (2, "data loader: resamples to the target shape"),
    ("transforms.py", "ants_resample_to_target"): (3, "resample_image_to_target's ANTs fallbacks and the identity-chain resample of TransformChain.apply: geometry onto a target grid, not a pyramid level"),
    ("viz/figures.py", "zoom"): (1, "figure compositing: shape-match for display"),
    # ---- thin wrappers over the central pyramid (they call build_pyramid / plan_levels; they do not implement a level)
    ("motion_batched.py", "def_pyramid_builder"): (1, "get_pyramid_level: caches central build_pyramid levels per level"),
    ("syn_jax.py", "def_pyramid_builder"): (1, "pyramid_level_jax: wraps pyramid.build_pyramid_jax for one level"),
    # ---- not a pyramid: entropy
    ("image_compare.py", "log2_level_sigma"): (0, "(no occurrence expected)"),
}


def _code_only(text: str) -> str:
    """Source with comments and string literals (incl. docstrings) blanked to spaces (newlines kept, so line numbers stay true)."""
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
    except (tokenize.TokenError, IndentationError):
        return text
    return "".join(chars)


def scan(root: Path = SRC, central: str = CENTRAL) -> dict:
    """{(relative file, pattern name): [line numbers]} for every occurrence outside the central module."""
    found: dict = {}
    for f in sorted(root.rglob("*.py")):
        rel = f.relative_to(root).as_posix()
        if rel == central:
            continue
        code = _code_only(f.read_text())
        for name, rx in PATTERNS.items():
            for m in re.finditer(rx, code):
                found.setdefault((rel, name), []).append(code.count("\n", 0, m.start()) + 1)
    return found


def violations(found: dict) -> list:
    bad = []
    for (rel, name), lines in sorted(found.items()):
        allowed, _ = ALLOW.get((rel, name), (0, ""))
        if len(lines) > allowed:
            bad.append(f"{rel}: {len(lines)} x '{name}' at lines {lines} (allowed {allowed}); pyramid / downsample code belongs in "
                       f"syntx/pyramid.py (build_pyramid, plan_levels, level_spacing) -- or add an ALLOW entry with a reason")
    return bad


def test_no_pyramid_or_downsample_code_outside_the_central_module():
    bad = violations(scan())
    assert not bad, "\n".join(bad)


def test_allow_list_has_reasons_and_no_stale_entries():
    found = scan()
    for key, (n, reason) in ALLOW.items():
        assert reason and len(reason) > 10, f"{key}: an allow-list entry needs a reason"
        assert (SRC / key[0]).exists(), f"{key}: the file no longer exists"
        have = len(found.get(key, []))
        assert have >= n or n == 0, f"{key}: allows {n} but only {have} occur; lower the allowance so a new copy cannot hide in the slack"


@pytest.mark.parametrize("snippet,pattern", [
    ("x = F.avg_pool3d(x, kernel_size=level, stride=level)", "avg_pool"),
    ("y = F.interpolate(x, size=shape, mode='trilinear', align_corners=True)", "interpolate"),
    ("img = ants.resample_image(img, (2, 2, 2), use_voxels=False, interp_type=0)", "ants_resample_image"),
    ("def level_spacing(sp, n, m):\n    return sp", "def_level_spacing"),
    ("def build_image_pyramid(x, levels):\n    return x", "def_pyramid_builder"),
    ("sp2 = [s * (a - 1) / (b - 1) for s, a, b in zip(sp, n, m)]", "corner_spacing_formula"),
    ("sig = math.log2(level)", "log2_level_sigma"),
])
def test_the_scan_detects_each_kind_of_copy(tmp_path, snippet, pattern):
    """The guard would have caught the earlier copies: each pattern fires on a representative violation in a scratch package."""
    (tmp_path / "newmodel.py").write_text(snippet + "\n")
    found = scan(tmp_path)
    assert ("newmodel.py", pattern) in found and violations(found)


def test_prose_and_comments_do_not_trip_the_scan(tmp_path):
    (tmp_path / "doc.py").write_text('"""Uses avg_pool3d and F.interpolate( in prose."""\n# resample_image( in a comment\nx = 1\n')
    assert not scan(tmp_path)
