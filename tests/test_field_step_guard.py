"""Guard & unit tests for centralized regularizer metadata and field-step optimization.

Enforces the centralized architecture in ``core/regularizers.py`` and
``core/optimizers.py`` across all registration solvers (``greedy``, ``syn``, ``tvf``,
``syngs``) so continuum PDE coupling, physical-space chain rules, and single-pass
RegAdam quotient smoothing cannot drift or regress:

1. Zero local ``_*_EXTRA_REG_KEYS`` or ``_*_CONTINUUM*`` regularizer sets outside
   ``core/regularizers.py``.
2. Zero ``sobolev_alpha=0.035`` signature/fallback defaults outside
   ``core/optimizers.py``'s legacy ``'sobolev'``-only branch.
3. Zero inline ``exp_avg_sq`` manual Adam loops in ``greedy.py`` (must delegate to
   ``core/optimizers.py``).
4. Zero ``min(spacing)`` isotropic step-bound shortcuts in ``core/optimizers.py``.
"""
from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

import ants
import numpy as np
import pytest
import torch

import syntx

SRC = Path(__file__).resolve().parents[1] / "src" / "syntx"


def _code_only(text: str) -> str:
    """Return source with comments and string literals (incl. docstrings) blanked."""
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


# ---------------------------------------------------------------------------
# Static architectural guards
# ---------------------------------------------------------------------------
def test_no_local_continuum_or_extra_reg_sets_outside_regularizers():
    """Zero local ``_*_EXTRA_REG_KEYS`` or ``_*_CONTINUUM*`` sets outside core/regularizers.py."""
    pat = re.compile(r"\b_\w*(?:EXTRA_REG_KEYS|CONTINUUM\w*)\s*=")
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel == "core/regularizers.py":
            continue
        code = _code_only(path.read_text(encoding="utf-8"))
        for m in pat.finditer(code):
            line = code.count("\n", 0, m.start()) + 1
            offenders.append(f"{rel}:{line}: {m.group(0).strip()}")
    assert not offenders, (
        "Continuum regularizer sets and extra-key allowlists belong in "
        "src/syntx/core/regularizers.py:\n  " + "\n  ".join(offenders)
    )


def test_no_sobolev_alpha_0035_default_outside_legacy_optimizers():
    """Zero ``sobolev_alpha=0.035`` signature/fallback defaults outside core/optimizers.py."""
    pat = re.compile(r"\bsobolev_alpha\s*(?::[^=]+)?=\s*0\.035\b")
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        code = _code_only(path.read_text(encoding="utf-8"))
        for m in pat.finditer(code):
            line = code.count("\n", 0, m.start()) + 1
            offenders.append(f"{rel}:{line}: {m.group(0).strip()}")
    assert not offenders, (
        "sobolev_alpha must default to None (legacy 0.035 fallback lives only inside "
        "core/optimizers.py's sobolev-only branch):\n  " + "\n  ".join(offenders)
    )


def test_greedy_has_no_inline_manual_adam_loop():
    """greedy.py must delegate field stepping to core/optimizers.py (no inline exp_avg_sq)."""
    code = _code_only((SRC / "greedy.py").read_text(encoding="utf-8"))
    hits = [
        code.count("\n", 0, m.start()) + 1
        for m in re.finditer(r"\bexp_avg_sq\b", code)
    ]
    assert not hits, (
        f"greedy.py still contains inline exp_avg_sq Adam loop at lines {hits}; "
        "use FieldStepOptimizer from core/optimizers.py"
    )


def test_optimizers_has_no_min_spacing_shortcut():
    """core/optimizers.py must use per-axis anisotropic voxel norms, never min(spacing)."""
    code = _code_only((SRC / "core" / "optimizers.py").read_text(encoding="utf-8"))
    hits = [
        code.count("\n", 0, m.start()) + 1
        for m in re.finditer(r"\bmin\s*\(\s*spacing\s*\)", code)
    ]
    assert not hits, (
        f"core/optimizers.py uses isotropic min(spacing) shortcut at lines {hits}; "
        "use _scaled_sq_sum per-axis voxel norm"
    )


# ---------------------------------------------------------------------------
# Behavioral & unit tests
# ---------------------------------------------------------------------------
def _tiny_pair(shape=(16, 16, 12), spacing=(1.0, 1.0, 2.0)):
    z, y, x = np.ogrid[:shape[2], :shape[1], :shape[0]]
    cx, cy, cz = shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0
    fi_np = np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / 12.0).T.astype(np.float32)
    mi_np = np.exp(-((x - (cx - 1.2)) ** 2 + (y - (cy + 0.8)) ** 2 + (z - cz) ** 2) / 12.0).T.astype(np.float32)
    fi = ants.from_numpy(fi_np, spacing=spacing, origin=(0.0, 0.0, 0.0))
    mi = ants.from_numpy(mi_np, spacing=spacing, origin=(0.0, 0.0, 0.0))
    return fi, mi


@pytest.mark.parametrize("bad_opt", ["adam", "adamw", "rmsprop", "rprop"])
def test_validate_regularizer_optimizer_pair_rejects_componentwise_continuum(bad_opt):
    from syntx.core.optimizers import validate_regularizer_optimizer_pair

    with pytest.raises(ValueError, match="incompatible with coupled continuum"):
        validate_regularizer_optimizer_pair("navier", bad_opt, caller="test")
    assert validate_regularizer_optimizer_pair("navier", None, caller="test") == "cfl"
    assert validate_regularizer_optimizer_pair("gaussian", None, caller="test") == "adam"
    assert validate_regularizer_optimizer_pair("navier", "cfl", caller="test") == "cfl"
    assert validate_regularizer_optimizer_pair("navier", "reg_adam", caller="test") == "reg_adam"


def test_all_four_solvers_reject_explicit_navier_plus_adam():
    fi, mi = _tiny_pair(shape=(12, 12, 10), spacing=(1.0, 1.0, 1.0))

    with pytest.raises(ValueError, match="incompatible with coupled continuum"):
        syntx.greedy(fi, mi, regularizer="navier", optimizer="adam",
                     initial_transform=False, reg_iterations=[1], scales=[1], device="cpu")

    with pytest.raises(ValueError, match="incompatible with coupled continuum"):
        syntx.syn(fi, mi, regularizer="navier", optimizer="adam",
                  initial_transform="identity", reg_iterations=[1], levels=[1], device="cpu")

    with pytest.raises(ValueError, match="incompatible with coupled continuum"):
        syntx.tvf(fi, mi, regularizer="navier", optimizer="adam",
                  initial_transform="identity", reg_iterations=[1], levels=[1], device="cpu")

    with pytest.raises(ValueError, match="incompatible with coupled continuum"):
        syntx.syngs(fi, mi, regularizer="navier", optimizer="adam",
                    initial_transform="identity", reg_iterations=[1], levels=[1], device="cpu")


def test_greedy_navier_autoselects_cfl_and_converges_anisotropic_noncubic():
    fi, mi = _tiny_pair(shape=(20, 18, 12), spacing=(1.0, 1.2, 2.5))
    init_mse = float(np.mean((fi.numpy() - mi.numpy()) ** 2))
    res = syntx.greedy(
        fi,
        mi,
        regularizer="navier",
        reg_iterations=[15, 10],
        scales=[2, 1],
        initial_transform=False,
        device="cpu",
        verbose=False,
    )
    assert res["provenance"]["optimizer"] == "cfl"
    assert res["model"].optimizer == "cfl"
    assert res["model"].sobolev_alpha is None
    final_mse = float(np.mean((fi.numpy() - res["warpedmovout"].numpy()) ** 2))
    assert final_mse < init_mse
    from syntx.deformation_metrics import compute_jacobian_metrics
    jac = compute_jacobian_metrics(fi, res["fwdtransforms"][0])
    assert jac["folding_pct"] == 0.0


def test_synto_reg_adam_applies_greens_operator_once_not_twice(monkeypatch):
    """Verify SyNTo.fit with optimizer_type='reg_adam' filters once per field per step."""
    import importlib
    syn_mod = importlib.import_module("syntx.syn")

    call_counts = {"sobolev": 0, "navier": 0}
    orig_sobolev = syn_mod.SyNTo._apply_sobolev_green_operator

    def spy_sobolev(self, m, *args, **kwargs):
        call_counts["sobolev"] += 1
        return orig_sobolev(self, m, *args, **kwargs)

    monkeypatch.setattr(syn_mod.SyNTo, "_apply_sobolev_green_operator", spy_sobolev)

    fi, mi = _tiny_pair(shape=(12, 12, 10), spacing=(1.0, 1.0, 1.0))
    syntx.syn(
        fi,
        mi,
        regularizer="sobolev",
        optimizer="reg_adam",
        fast_smooth=True,
        reg_iterations=[2],
        levels=[1],
        in_loop_inv_steps=0,
        polish_inv_steps=0,
        compose_polish_tol=0.0,
        initial_transform="identity",
        device="cpu",
    )
    # 2 epochs * 2 half-fields (warp_l2r + warp_r2l) = 4 calls (NOT 8 from H^4 double smoothing)
    assert call_counts["sobolev"] == 4

    orig_build = syn_mod.build_level_regularizer

    def spy_build(*args, **kwargs):
        fn = orig_build(*args, **kwargs)
        def wrapped(v):
            call_counts["navier"] += 1
            return fn(v)
        return wrapped

    monkeypatch.setattr(syn_mod, "build_level_regularizer", spy_build)
    syntx.syn(
        fi,
        mi,
        regularizer="navier",
        optimizer="reg_adam",
        reg_iterations=[2],
        levels=[1],
        in_loop_inv_steps=0,
        polish_inv_steps=0,
        compose_polish_tol=0.0,
        initial_transform="identity",
        device="cpu",
    )
    assert call_counts["navier"] == 4


def test_regadam_anisotropic_step_bound_and_direction_origin_forwarding(monkeypatch):
    from syntx.core.optimizers import RegAdam

    # 1. Anisotropic step bound: spacing=(1.0, 4.0) -> zyx=(4.0, 1.0)
    # Component 0 is z (sp=4.0), component 1 is y (sp=1.0).
    p = torch.nn.Parameter(torch.zeros(1, 8, 8, 2))
    opt = RegAdam([p], lr=1.0, regularizer="none", max_step_norm=0.25, spacing=(1.0, 4.0))
    p.grad = torch.zeros_like(p)
    p.grad[..., 0] = 10.0  # z-component with spacing=4.0
    opt.step()
    # Exact anisotropic voxel norm: |dz|/4.0 <= 0.25 => |dz| <= 1.0 (whereas min(spacing)=1.0 clipped to 0.25!)
    dz = float(p.detach()[..., 0].abs().max())
    assert 0.99 <= dz <= 1.0 + 1e-5

    # 2. Forwarding of origin, direction, and extra_reg_kwargs to get_regularizer
    seen = {}
    import syntx.core.optimizers as opt_mod
    orig_get_reg = opt_mod.get_regularizer

    def spy_get_reg(name, **kw):
        seen.update(kw)
        seen["name"] = name
        return orig_get_reg(name, **kw)

    monkeypatch.setattr(opt_mod, "get_regularizer", spy_get_reg)
    p2 = torch.nn.Parameter(torch.zeros(1, 8, 8, 2))
    oblique = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
    opt2 = RegAdam(
        [p2],
        lr=0.1,
        regularizer="navier",
        max_step_norm=0.5,
        spacing=(1.0, 2.0),
        origin=(10.0, -5.0),
        direction=oblique,
        extra_reg_kwargs={"poisson_ratio": 0.35},
    )
    p2.grad = torch.randn_like(p2)
    opt2.step()
    assert seen["name"] == "navier"
    assert seen["alpha"] is None
    assert seen["origin"] == (10.0, -5.0)
    assert np.allclose(seen["direction"], oblique)
    assert seen["poisson_ratio"] == 0.35
