"""Guard: compute-device selection lives in ONE module, ``syntx/device.py``.

The device audit (2026-10-12, docs/DEVICE_POLICY.md) found a dozen local ``cuda > mps > cpu`` copies with different
precedences (some mps-first, some cuda-first) and several ``device="cpu"`` defaults that silently pinned whole
pipelines to the CPU (a SOCOM rsfMRI T1->BOLD SyN ran in 236 s on CPU vs 89 s on MPS). Everything must resolve through
``syntx.device.resolve_device``; "cpu" is only legitimate where stated.

Comments and docstrings are removed before scanning. ``ALLOW`` maps ``(file, pattern)`` to ``(maximum occurrences,
reason)``. A new occurrence fails the test: resolve through ``syntx.device`` instead, or add an entry whose reason says
why CPU / a direct availability probe is correct there.

Patterns
--------
avail        ``cuda.is_available()`` / ``mps.is_available()`` outside ``device.py`` (device *choice* belongs to the
             resolver; cache clearing, RNG-state capture and environment reports are allow-listed).
cpu_literal  a ``device``-like name set to ``"cpu"`` (default argument, keyword argument or assignment).
reg_cpu      ``reg_device`` / ``model_device`` set to ``"cpu"`` -- never allowed in ``src``.
"""
from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "syntx"
CENTRAL = "device.py"

_NAMES = (r"(?:reg_device|model_device|device|dev|device_str|target_device|torch_device|csd_device|cmp_device|"
          r"device_obj|chosen_device|fit_device|dev_str)")
PATTERNS = {
    "avail": r"\b(?:cuda|mps)\s*\.\s*is_available\s*\(\s*\)",
    # ``_strip`` joins tokens with spaces, so ``torch.device("cpu")`` is scanned as ``torch . device ( "cpu" )``: the
    # wrapper must tolerate whitespace (it did not before 6.6.1, which hid every ``device = torch.device("cpu")``).
    "cpu_literal": rf"\b{_NAMES}\s*(?::\s*[\w\[\]|. ]+)?=\s*(?:torch\s*\.\s*device\s*\(\s*)?['\"]cpu['\"]",
    "reg_cpu": r"\b(?:reg_device|model_device)\s*(?::\s*[\w\[\]|. ]+)?=\s*['\"]cpu['\"]",
}

_CLEAR = "cache clearing / RNG-state capture / environment report -- not device choice"
_LOWLEVEL = ("low-level tensor helper: the device is the caller's choice, 'cpu' is only the default for building a "
             "small constant tensor / grid")
ALLOW = {
    ("benchmark/evaluate.py", "avail"): (2, _CLEAR),
    ("benchmark/high_level.py", "avail"): (1, _CLEAR),
    ("cli.py", "avail"): (2, "`syntx info` environment report"),
    ("core/mps_kernels.py", "avail"): (1, "capability probe for the Metal shader kernels"),
    ("generators.py", "avail"): (2, "RNG-state capture"),
    ("provenance.py", "avail"): (2, "environment report"),
    ("robust_affine.py", "avail"): (3, "seeding and cache clearing"),
    ("syn.py", "avail"): (2, _CLEAR),
    ("benchmark/high_level.py", "cpu_literal"): (1, "explicit CPU baseline of a benchmark"),
    ("benchmark/tune.py", "cpu_literal"): (2, "explicit CPU probe / documented exploratory evaluator default"),
    ("core/pipeline.py", "cpu_literal"): (1, "normalize_and_tensorize low-level default; JAX backend accepts only cpu"),
    ("core/smoothing.py", "cpu_literal"): (1, _LOWLEVEL),
    ("generators.py", "cpu_literal"): (1, _LOWLEVEL),
    ("motion_batched.py", "cpu_literal"): (2, "torch.Generator(device='cpu') seeds: RNG reproducibility across devices"),
    ("robust_affine.py", "cpu_literal"): (2, "one-time small-tensor setup, commented (avoids per-call MPS overhead)"),
    ("scattered/solver.py", "cpu_literal"): (2, _LOWLEVEL + "; plain-numpy inputs build on cpu when no device is given"),
    ("spatial.py", "cpu_literal"): (11, _LOWLEVEL + "; one is the documented mps->cpu redirect (150x slower on MPS)"),
    ("syn_jax.py", "cpu_literal"): (5, "JAX backend accepts only cpu; SyNToTransform grids are stored on cpu"),
    ("core/losses/structural.py", "cpu_literal"): (1, "exact Euclidean distance transform runs in SciPy (CPU by construction)"),
    ("scattered/bspline.py", "cpu_literal"): (1, _LOWLEVEL + "; default when no tensor input carries a device"),
    ("scattered/projection.py", "cpu_literal"): (1, _LOWLEVEL + "; default when no tensor input carries a device"),
    ("syngs.py", "cpu_literal"): (3, "JAX path accepts only cpu; physical-grid reference matrices built on cpu"),
    ("tabulate.py", "cpu_literal"): (3, "tiny ROI x time float64 correlation matrices (MPS has no float64); documented"),
    ("transform.py", "cpu_literal"): (1, "TransformCollection default storage device; follows .to()"),
    ("tvf.py", "cpu_literal"): (3, "JAX path accepts only cpu; physical-grid reference matrices built on cpu"),
    ("tvf_adj.py", "cpu_literal"): (3, "adjoint-TVF research class: explicit device argument at call sites"),
    ("viz/reports.py", "cpu_literal"): (1, "report renderer default; follows caller"),
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
            out.append('""')
            continue
        out.append(t.string)
    return " ".join(out)


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


def test_no_new_local_device_resolution_or_cpu_defaults():
    offenders = []
    for (rel, name), n in _scan().items():
        allowed = ALLOW.get((rel, name), (0, ""))[0]
        if n > allowed:
            offenders.append(f"{rel}: {name} x{n} (allowed {allowed})")
    assert not offenders, (
        "New local device resolution / 'cpu' device defaults. Resolve through syntx.device.resolve_device "
        "(ComputeContext resolves 'auto' itself), or add an ALLOW entry with a reason:\n  " + "\n  ".join(offenders)
    )


def test_reg_and_model_device_never_default_to_cpu_in_src():
    assert not [k for k in _scan() if k[1] == "reg_cpu"]


def test_allow_entries_have_reasons():
    assert all(reason.strip() for _, reason in ALLOW.values())


def test_guard_detects_a_local_resolver(tmp_path):
    code = _strip("device = 'cuda' if torch.cuda.is_available() else 'cpu'\nreg_device = 'cpu'\n")
    assert re.search(PATTERNS["avail"], code)
    assert re.search(PATTERNS["reg_cpu"], code)


def test_guard_detects_torch_device_cpu_assignment():
    """Regression (6.6.1): ``device = torch.device("cpu")`` slipped through because the tokenizer round trip inserts spaces."""
    for line in ('device = torch.device("cpu")\n', "dev = torch.device('cpu')\n", "reg_device = torch.device( 'cpu' )\n"):
        assert re.search(PATTERNS["cpu_literal"], _strip(line)), line
