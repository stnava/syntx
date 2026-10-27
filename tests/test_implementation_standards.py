"""ANTsX implementation standards tests for syntx."""

import ast
import pathlib
import re

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "syntx"

FORBIDDEN_DATA_TOKENS = [
    "antspynet.get_antsxnet_data",
    "antspyt1w.get_data",
    "urllib.request.urlretrieve",
    "requests.get(",
    "huggingface_hub",
    "~/.antsx",
]

ALLOWED_DATA_EXCEPTIONS: dict[str, str] = {
    "data/msd.py:275": (
        "Medical Segmentation Decathlon public S3 benchmark tarball fetch; "
        "optional external benchmark helper, not an ANTsX runtime/template asset"
    ),
}

ALLOWED_REGISTRATION_EXCEPTIONS: dict[str, str] = {
    "benchmark/evaluate.py:589": (
        "Affine benchmark baseline runner measuring external ANTs C++ affine against syntx"
    ),
    "benchmark/high_level.py:373": (
        "2D r16/r64 benchmark baseline comparison against external ANTs C++ SyN"
    ),
    "motion.py:1067": (
        "Legacy mode='ants' C++ motion correction baseline comparator inside syntx.motion"
    ),
    "robust_affine.py:2470": (
        "mode='ants_fast' explicit C++ affine backend inside syntx (the registration engine itself)"
    ),
    "robust_affine.py:2479": (
        "mode='ants_fast' explicit C++ affine backend inside syntx (the registration engine itself)"
    ),
    "robust_affine.py:2547": (
        "mode='ants_fast' explicit C++ affine backend inside syntx (the registration engine itself)"
    ),
    "robust_affine.py:2639": (
        "mode='ants_fast' explicit C++ affine backend inside syntx (the registration engine itself)"
    ),
    "template.py:394": (
        "Template builder C++ fallback backend when explicitly requested via backend='ants'"
    ),
    "viz/reports.py:2328": (
        "Prose / code snippet inside HTML report template string describing ANTs baseline"
    ),
}

ALLOWED_REORIENT_EXCEPTIONS: dict[str, str] = {}

ALLOWED_PREPROCESS_EXCEPTIONS: dict[str, str] = {
    "device.py:146": (
        "Central device resolver setting PYTORCH_ENABLE_MPS_FALLBACK=1 when selecting MPS"
    ),
    "syn_jax.py:28": (
        "JAX backend module-level XLA_PYTHON_CLIENT_PREALLOCATE=false required before importing jax"
    ),
    "syn_jax.py:29": (
        "JAX backend module-level XLA_FLAGS CPU multi-thread configuration required before importing jax"
    ),
}

FORBIDDEN_IMPORTS = {
    "antspynet",
    "antspyt1w",
    "nibabel",
    "SimpleITK",
    "nilearn",
    "tensorflow",
    "keras",
}


def _py_files():
    return sorted(SRC.rglob("*.py"))


def _rel(p: pathlib.Path) -> str:
    return str(p.relative_to(SRC))


def test_data_policy_all_assets_through_antsxdata():
    hits = []
    for p in _py_files():
        for i, line in enumerate(p.read_text().splitlines(), 1):
            for tok in FORBIDDEN_DATA_TOKENS:
                if tok in line:
                    key = f"{_rel(p)}:{i}"
                    if key not in ALLOWED_DATA_EXCEPTIONS:
                        hits.append(f"{key}: {tok}")
    assert not hits, "Direct download / ad-hoc cache found (use antsxdata):\n" + "\n".join(hits)


def test_no_raw_ants_registration():
    pat = re.compile(r"\bants\.registration\s*\(")
    hits = []
    for p in _py_files():
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if pat.search(line):
                key = f"{_rel(p)}:{i}"
                if key not in ALLOWED_REGISTRATION_EXCEPTIONS:
                    hits.append(key)
    assert not hits, "Raw ants.registration calls without documented exception:\n" + "\n".join(hits)


def test_no_raw_ants_motion_correction():
    pat = re.compile(r"\bants\.motion_correction\s*\(")
    hits = [
        f"{_rel(p)}:{i}"
        for p in _py_files()
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if pat.search(line)
    ]
    assert not hits, "Raw ants.motion_correction calls (use syntx.motion_correction):\n" + "\n".join(hits)


def test_no_unexplained_reorient_image():
    pat = re.compile(r"\breorient_image2?\b")
    hits = []
    for p in _py_files():
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if pat.search(line):
                key = f"{_rel(p)}:{i}"
                if key not in ALLOWED_REORIENT_EXCEPTIONS:
                    hits.append(key)
    assert not hits, "reorient_image call without documented physical-space justification:\n" + "\n".join(hits)


def test_no_unexplained_ants_preprocessing_or_env_mutation():
    pat = re.compile(r"\bants\.(n4_bias_field_correction|n3_bias_field_correction|denoise_image)\s*\(")
    env_pat = re.compile(r"\bos\.environ(?:\.setdefault|\s*\[)")
    hits = []
    for p in _py_files():
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if pat.search(line) or env_pat.search(line):
                key = f"{_rel(p)}:{i}"
                if key not in ALLOWED_PREPROCESS_EXCEPTIONS:
                    hits.append(key)
    assert not hits, (
        "CPU ants preprocessing or module-level os.environ mutation found:\n" + "\n".join(hits)
    )


def test_no_forbidden_imports():
    hits = []
    for p in _py_files():
        tree = ast.parse(p.read_text(), filename=str(p))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            for n in names:
                if n in FORBIDDEN_IMPORTS:
                    hits.append(f"{_rel(p)}:{node.lineno} imports {n}")
    assert not hits, "Forbidden imports:\n" + "\n".join(hits)


@pytest.mark.parametrize(
    "table",
    [
        ALLOWED_DATA_EXCEPTIONS,
        ALLOWED_REGISTRATION_EXCEPTIONS,
        ALLOWED_REORIENT_EXCEPTIONS,
        ALLOWED_PREPROCESS_EXCEPTIONS,
    ],
)
def test_exception_rationales_are_nonempty(table):
    for loc, why in table.items():
        assert isinstance(why, str) and len(why.strip()) >= 20, f"Missing rationale at {loc}"
