"""Design-driven motion correction: ONE entry point for every acquisition design (a menu).

An acquisition design says which frames share a contrast (the design LEVELS: ASL control / label, DWI shells, slow-flow
encoding states; BOLD has one level) and which per-frame columns can carry an acquisition-locked pose bias ``beta``
(DWI: the l = 2 direction terms and the eddy term per shell). Model: ``pose(t) = smooth head motion + D(t) beta +
residual``. The ``unified`` engine registers every level to a reference of its own contrast with its own sample points,
estimates the in-level design columns in the optimizer (overlapping Hann windows, Tukey weights), links the levels, and
removes a per-level offset ``beta`` estimated from local (surround) differences of the composed trajectory with Tukey
weights (``motion_reference._group_offsets``): smooth drift is in its nullspace, so it cannot absorb real head motion
that is not locked to the design. Head motion that IS locked to the design (e.g. a period-2 jitter in an alternating
design) is indistinguishable from ``beta`` and is removed with it.

Menu (``DESIGNS``; ``make_design(name, ...)`` builds a :class:`Design` from the acquisition metadata):

- ``none``             BOLD: one level, no ``beta`` (the estimator reduces to registration to the mean).
- ``tag_control``      ASL: ``volume_types`` (aslcontext); levels control (anchor) / label / each other type (M0);
                       control/label carry the +-1 column.
- ``shells``           DWI: levels = ``shell_groups(bvals)``; per-volume eddy term along phase encode tied to the
                       gradient (``pe_affine_covariates``), the five l = 2 direction terms as bias columns.
- ``encoding_states``  slow flow: integer encoding-state labels.
- ``custom``           caller-supplied ``levels`` and ``encoding`` columns.

``engine="unified"`` is the design-driven estimator; ``engine="dispatch"`` reproduces, for each design, the call each
modality makes today (ASL: antsxfunctional
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
    levels: np.ndarray  # (T,) int, 0..k-1 (0 = anchor): frames of one level share a contrast (and a reference)
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
    levels = np.array([0 if v == "control" else (1 if v == "label" else 2 + others.index(v)) for v in vt], dtype=int)
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
        if design.n_levels == 2:  # control and label only
            r = motion_correction(image, reference="mean", type_of_transform="Rigid", preset="asl",
                                  encoding_covariates=design.encoding, **kwargs)
            return _normalized(r, design, "dispatch", "motion_correction(preset='asl', encoding_covariates=s_tc)")
        knobs = {k: v for k, v in get_motion_preset("asl").items() if k != "window_size"}
        groups = np.where(design.levels <= 1, 0, design.levels - 1)  # production: control and label share group 0
        r = motion_correct_grouped(image, groups, encoding_covariates=design.encoding, **knobs, **kwargs)
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


MIN_MIXED_FOR_BETA = 4  # frames with a neighbour of another level needed to estimate a level's offset locally
FRAMES_PER_BETA_COLUMN = 6  # a design column is estimated only with at least this many frames per column


def level_layout(levels: np.ndarray) -> Dict[int, dict]:
    """Per level: how many of its frames have a neighbour of another level (``mixed``). A level is INTERLEAVED (spread
    through the series, so its mean shares the series' mean head position) when it has at least 2 frames and at least
    half of them are mixed; otherwise it is BLOCKED (its offset is not separable from drift). Its offset to the others is
    estimable locally (``beta_estimable``) with at least ``MIN_MIXED_FOR_BETA`` mixed frames."""
    lev = np.asarray(levels).reshape(-1)
    T = lev.size
    out = {}
    for k in np.unique(lev):
        idx = np.flatnonzero(lev == k)
        mixed = sum(1 for t in idx if (t > 0 and lev[t - 1] != k) or (t < T - 1 and lev[t + 1] != k))
        out[int(k)] = {"n": int(idx.size), "mixed": int(mixed),
                       "interleaved": bool(idx.size >= 2 and mixed >= max(2, (idx.size + 1) // 2)),
                       "beta_estimable": bool(mixed >= MIN_MIXED_FOR_BETA),
                       "mean_time": float(idx.mean() / max(T - 1, 1))}
    return out


def _knobs(design: Design) -> dict:
    """Registration settings per design (the modality presets; ``dwi``'s registration knobs without its grouped-only
    keys)."""
    from .motion_presets import get_motion_preset

    name = {"tag_control": "asl", "encoding_states": "slowflow", "shells": "dwi"}.get(design.name, "bold")
    cfg = dict(get_motion_preset(name))
    cfg.pop("cross_num_bins", None)
    return cfg


def _corrected_level_mean(frames, group_index=None, frame_indices=None):
    """``reference_predictor`` for the reference iteration: every frame of the level gets the mean of the level's
    corrected frames (sharper than the first-pass low-motion mean)."""
    import ants

    m = ants.new_image_like(frames[0], np.mean([f.numpy() for f in frames], axis=0).astype(np.float32))
    return [m] * len(frames)


def _unified(image, design: Design, *, beta: str = "levels+design", iterations: int = 0, **kwargs) -> dict:
    """ONE path for every design. Each design level is registered in its own pass to its own reference, with its own
    sample points (``motion_correct_grouped``: the level's low-motion mean; a reference, and a sampling region, of the
    frame's own contrast); the design's in-level columns are estimated in the optimizer (DWI: the l = 2 direction terms
    per shell, and the eddy term along phase encode tied to the gradient per shell). Adjacent levels are linked by identity
    when both are interleaved in time (their raw means share the series' mean head position) and by a rigid registration of
    the corrected means otherwise; the remaining per-level offset ``beta`` is the surround-Tukey projection of the
    composed trajectory on the level indicators, applied to the transforms (``cross_group_debias``). A one-level design is
    the plain registration to the series mean (BOLD). ``beta``: ``"levels+design"``, ``"levels"`` (no in-level design
    columns) or ``"none"`` (no offset removal). ``iterations``: passes re-registering each level to the mean of its
    corrected frames (``motion_correct_grouped(reference_predictor=..., reference_iterations=...)``)."""
    from .motion import motion_correction
    from .motion_reference import motion_correct_grouped

    if beta not in ("levels+design", "levels", "none"):
        raise ValueError(f"beta must be 'levels+design', 'levels' or 'none', got {beta!r}")
    levels = np.asarray(design.levels, dtype=int)
    knobs = _knobs(design)
    backend = knobs.pop("backend")
    window_size = knobs.pop("window_size", None)
    if design.n_levels == 1:
        kw = {**knobs, **kwargs}
        if design.encoding is not None:
            kw["encoding_covariates"] = design.encoding
            kw["window_size"] = window_size
        r = motion_correction(image, reference="mean", type_of_transform="Rigid", backend=backend, **kw)
        return _normalized(r, design, "unified", "motion_correction(reference='mean')")

    layout = level_layout(levels)
    ks = sorted(layout)
    # identity link where the level's offset is then measured locally (beta on, >= MIN_MIXED_FOR_BETA mixed frames: any
    # mean-pose difference between the raw level means is part of that offset) or where both levels share the series'
    # mean head position (interleaved); otherwise a rigid registration of the corrected means
    links = ["identity" if ((beta != "none" and layout[ks[i]]["beta_estimable"])
                            or (layout[ks[i - 1]]["interleaved"] and layout[ks[i]]["interleaved"])) else "registration"
             for i in range(1, len(ks))]
    kw = dict(kwargs)
    dropped = []
    if design.name == "shells":
        kw.update(preset="dwi", registration_direction="forward", metric_smoothing_mm=0.0)
        if design.pe_affine_covariates is not None:
            kw.update(pe_affine_covariates=design.pe_affine_covariates, pe_axis=design.pe_axis)
        if beta == "levels+design" and design.bias_covariates is not None:
            enc = np.array(design.bias_covariates, dtype=np.float64)
            for k in ks:
                live = int(np.sum(np.ptp(enc[levels == k], axis=0) > 0)) if np.any(levels == k) else 0
                if live and layout[k]["n"] < FRAMES_PER_BETA_COLUMN * live:
                    enc[levels == k] = 0.0
                    dropped.append(f"l2 terms of level {k} ({layout[k]['n']} frames < {FRAMES_PER_BETA_COLUMN} x {live})")
            if np.any(enc):
                kw["encoding_covariates"] = enc
                if window_size is not None:
                    kw["window_size"] = window_size
    else:
        kw.update(backend=backend, **knobs)
    if iterations > 0:
        kw.update(reference_predictor=_corrected_level_mean, reference_iterations=int(iterations))
    r = motion_correct_grouped(image, levels, cross_link=links, cross_group_debias=(beta != "none"), **kw)
    out = _normalized(r, design, "unified", "motion_correct_grouped(per-level references, links, cross_group_debias)")
    xo = r.get("cross_group_offset")
    out["design"].update(layout=layout, links=links, beta=beta, beta_dropped=dropped, iterations=int(iterations),
                         beta_levels=None if xo is None else {k: v for k, v in xo.items() if k != "correction_paths"},
                         beta_levels_estimable=xo is not None or beta == "none")
    return out


ENGINES = {"dispatch": _dispatch, "unified": _unified}


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
