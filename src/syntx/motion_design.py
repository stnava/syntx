"""Design-driven motion correction: ONE entry point for every acquisition design (a menu).

An acquisition design says which frames share a contrast (the design LEVELS: ASL control / label, DWI shells, slow-flow
encoding states; BOLD has one level) and which per-frame columns can carry an acquisition-locked pose bias ``beta``
(the encoding columns: ASL tag/control +-1, slow-flow state labels; DWI direction terms). Head motion and ``beta`` are
estimated jointly: ``pose(t) = smooth head motion + D(t) beta + residual`` in overlapping Hann windows with Tukey
weights (``motion_batched._project_encoding_surround_tukey_np``), with design-specific references.

Menu (``DESIGNS``; ``make_design(name, ...)`` builds a :class:`Design` from the acquisition metadata):

- ``none``             BOLD: one level, no ``beta`` (the estimator reduces to registration to the mean).
- ``tag_control``      ASL: ``volume_types`` (aslcontext); control/label carry the +-1 column; other types (M0) are
                       their own levels.
- ``shells``           DWI: levels = ``shell_groups(bvals)``; per-volume eddy term along phase encode tied to the
                       gradient (``pe_affine_covariates``), the five l = 2 direction terms as bias columns.
- ``encoding_states``  slow flow: integer encoding-state labels.
- ``custom``           caller-supplied ``levels`` and ``encoding`` columns.

``engine="dispatch"`` reproduces, for each design, the call each modality makes today (ASL: antsxfunctional
``syntx_motion_correction``; DWI: antsxdwi ``motion_correct``; slow flow: antsxslowflow ``preset="slowflow"``; BOLD:
``preset="bold"``), so the menu can replace those call sites without changing results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Design:
    """An acquisition design: per-frame ``levels`` (0 = anchor) and the optional ``beta`` columns."""

    name: str
    levels: np.ndarray  # (T,) int, 0..k-1; frames of one level share a contrast (and a reference)
    encoding: Optional[np.ndarray] = None  # (T,) or (T, K): columns that may carry an acquisition-locked pose bias
    pe_affine_covariates: Optional[np.ndarray] = None  # (T, 3) DWI: eddy term along phase encode tied to the gradient
    bias_covariates: Optional[np.ndarray] = None  # (T, 5) DWI: l = 2 direction terms (post-hoc, per group)
    pe_axis: Optional[np.ndarray] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(len(self.levels))

    @property
    def n_levels(self) -> int:
        return int(len(np.unique(self.levels)))


def _real_sh_l2(bvecs: np.ndarray, dw: np.ndarray) -> np.ndarray:
    """The five l = 2 real symmetric (Descoteaux) spherical harmonics of the unit gradient directions (rows ``dw``;
    other rows 0), the basis antsxdwi uses for its direction-bias covariates."""
    import scipy.special as sps

    g = np.asarray(bvecs, dtype=np.float64)
    out = np.zeros((len(g), 5))
    if not np.any(dw):
        return out
    gu = g[dw] / np.linalg.norm(g[dw], axis=1, keepdims=True)
    theta = np.arccos(np.clip(gu[:, 2], -1.0, 1.0))[:, None]
    phi = np.arctan2(gu[:, 1], gu[:, 0])[:, None]
    m = np.arange(-2, 3)
    deg = np.full(5, 2)
    if hasattr(sps, "sph_harm_y"):
        y = sps.sph_harm_y(deg, np.abs(m), theta, phi)
    else:  # scipy < 1.15
        y = sps.sph_harm(np.abs(m), deg, phi, theta)
    real = np.where(m > 0, y.imag, y.real) * np.where(m == 0, 1.0, np.sqrt(2.0))
    out[dw] = real
    return out


def _design_none(n_frames: int) -> Design:
    return Design("none", np.zeros(int(n_frames), dtype=int))


def _design_tag_control(volume_types: Sequence[str]) -> Design:
    vt = [str(v).lower().replace("tag", "label") for v in volume_types]
    others = sorted(set(vt) - {"control", "label"})
    levels = np.array([0 if v in ("control", "label") else 1 + others.index(v) for v in vt], dtype=int)
    s_tc = np.array([1.0 if v == "control" else (-1.0 if v == "label" else 0.0) for v in vt])
    n_tc = int(np.sum(np.abs(s_tc) > 0))
    enc = s_tc if (n_tc >= 6 and {"control", "label"} <= set(vt)) else None
    return Design("tag_control", levels, encoding=enc,
                  detail={"other_volume_types": others, "n_control_label": n_tc,
                          "encoding": "tag_control_pm1" if enc is not None else
                          f"none ({n_tc} control/label volumes of types {sorted(set(vt))})"})


def _design_shells(bvals, bvecs=None, pe_axis=None, b0_threshold: float = 50.0) -> Design:
    from .motion_reference import shell_groups

    bv = np.asarray(bvals, dtype=np.float64).reshape(-1)
    levels = np.asarray(shell_groups(bv, b0_threshold=b0_threshold), dtype=int)
    pe_cov = bias = None
    if bvecs is not None:
        g = np.asarray(bvecs, dtype=np.float64).reshape(len(bv), 3)
        dw = (bv > b0_threshold) & (np.linalg.norm(g, axis=1) > 0)
        bias = _real_sh_l2(g, dw)
        if pe_axis is not None:
            pe_cov = g
    return Design("shells", levels, pe_affine_covariates=pe_cov, bias_covariates=bias,
                  pe_axis=None if pe_axis is None else np.asarray(pe_axis, dtype=np.float64),
                  detail={"b0_threshold": float(b0_threshold), "n_per_level": np.bincount(levels).tolist()})


def _design_encoding_states(state_labels) -> Design:
    lab = np.asarray(state_labels).reshape(-1)
    _, levels = np.unique(lab, return_inverse=True)
    return Design("encoding_states", levels.astype(int), encoding=lab.astype(np.float64),
                  detail={"n_states": int(levels.max() + 1)})


def _design_custom(levels, encoding=None) -> Design:
    lev = np.asarray(levels).reshape(-1)
    _, levels_i = np.unique(lev, return_inverse=True)
    return Design("custom", levels_i.astype(int), encoding=None if encoding is None else np.asarray(encoding, dtype=np.float64))


DESIGNS: Dict[str, Callable[..., Design]] = {
    "none": _design_none,
    "tag_control": _design_tag_control,
    "shells": _design_shells,
    "encoding_states": _design_encoding_states,
    "custom": _design_custom,
}


def make_design(name: str, **metadata) -> Design:
    """Build a :class:`Design` from its menu entry; unknown names raise."""
    if name not in DESIGNS:
        raise ValueError(f"unknown acquisition design {name!r}; available: {sorted(DESIGNS)}")
    return DESIGNS[name](**metadata)


def _normalized(result: Any, design: Design, engine: str, call: str) -> dict:
    out = dict(result)
    out["design"] = {"name": design.name, "engine": engine, "call": call, "n_levels": design.n_levels,
                     "levels": np.asarray(design.levels).tolist(), **design.detail}
    return out


def _dispatch(image, design: Design, **kwargs) -> dict:
    """Today's per-modality calls (see the module docstring), unchanged."""
    from .motion import motion_correction
    from .motion_presets import get_motion_preset
    from .motion_reference import motion_correct_grouped

    if design.name == "none":
        r = motion_correction(image, reference="mean", type_of_transform="Rigid", preset="bold", **kwargs)
        return _normalized(r, design, "dispatch", "motion_correction(preset='bold')")
    if design.name == "tag_control":
        if design.encoding is None:
            r = motion_correction(image, reference="mean", type_of_transform="Rigid", preset="bold", **kwargs)
            return _normalized(r, design, "dispatch", "motion_correction(preset='bold')")
        if design.n_levels == 1:
            r = motion_correction(image, reference="mean", type_of_transform="Rigid", preset="asl",
                                  encoding_covariates=design.encoding, **kwargs)
            return _normalized(r, design, "dispatch", "motion_correction(preset='asl', encoding_covariates=s_tc)")
        knobs = {k: v for k, v in get_motion_preset("asl").items() if k != "window_size"}
        r = motion_correct_grouped(image, design.levels, encoding_covariates=design.encoding, **knobs, **kwargs)
        return _normalized(r, design, "dispatch", "motion_correct_grouped(asl knobs, encoding_covariates=s_tc)")
    if design.name == "shells":
        kw = dict(preset="dwi", registration_direction="forward", metric_smoothing_mm=0.0)
        if design.pe_affine_covariates is not None:
            kw.update(pe_affine_covariates=design.pe_affine_covariates, pe_axis=design.pe_axis)
        if design.bias_covariates is not None:
            kw["bias_covariates"] = design.bias_covariates
        r = motion_correct_grouped(image, design.levels, **kw, **kwargs)
        return _normalized(r, design, "dispatch", "motion_correct_grouped(preset='dwi')")
    if design.name == "encoding_states":
        r = motion_correction(image, reference="mean", preset="slowflow", encoding_covariates=design.encoding, **kwargs)
        return _normalized(r, design, "dispatch", "motion_correction(preset='slowflow', encoding_covariates=states)")
    if design.name == "custom":
        if design.n_levels == 1:
            r = motion_correction(image, reference="mean", type_of_transform="Rigid", preset="bold",
                                  **({"encoding_covariates": design.encoding} if design.encoding is not None else {}), **kwargs)
            return _normalized(r, design, "dispatch", "motion_correction(preset='bold')")
        r = motion_correct_grouped(image, design.levels,
                                   **({"encoding_covariates": design.encoding} if design.encoding is not None else {}), **kwargs)
        return _normalized(r, design, "dispatch", "motion_correct_grouped")
    raise ValueError(f"no dispatch for design {design.name!r}")


ENGINES = {"dispatch": _dispatch}


def motion_correct_design(image, design: Design, *, engine: str = "dispatch", **kwargs) -> dict:
    """Motion correction of a 4-D series under an acquisition :class:`Design` (``make_design``).

    Returns the underlying result (``motion_corrected``, ``fwdtransforms``, ``fd``, ...) plus ``design`` (name, engine,
    the call made, levels). ``kwargs`` go to that call (``device``, ``outprefix``, ``sample_seed``, ...)."""
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; available: {sorted(ENGINES)}")
    T = int(image.shape[-1])
    if design.n_frames != T:
        raise ValueError(f"design {design.name!r} has {design.n_frames} frames, the series {T}")
    return ENGINES[engine](image, design, **kwargs)
