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

    def __post_init__(self):
        lev = np.asarray(self.levels)
        if lev.ndim != 1:
            raise ValueError(f"levels must be 1D, got ndim={lev.ndim}")
        unique_levs = set(lev.tolist())
        expected = set(range(len(unique_levs)))
        if unique_levs != expected:
            raise ValueError(f"levels must be contiguous integers 0..k-1; got {sorted(unique_levs)}")
        T = len(lev)
        if self.encoding is not None:
            enc = np.asarray(self.encoding)
            if enc.shape[0] != T:
                raise ValueError(f"encoding rows ({enc.shape[0]}) must match levels length ({T})")
        if self.pe_affine_covariates is not None:
            pe_cov = np.asarray(self.pe_affine_covariates)
            if pe_cov.shape[0] != T:
                raise ValueError(f"pe_affine_covariates rows ({pe_cov.shape[0]}) must match levels length ({T})")
        if self.bias_covariates is not None:
            bias = np.asarray(self.bias_covariates)
            if bias.shape[0] != T:
                raise ValueError(f"bias_covariates rows ({bias.shape[0]}) must match levels length ({T})")
        if self.pe_axis is not None:
            p = np.asarray(self.pe_axis, dtype=np.float64).reshape(-1)
            norm = float(np.linalg.norm(p))
            if p.size != 3 or norm < 1e-8:
                raise ValueError(f"pe_axis must be a non-zero 3-vector, got {self.pe_axis!r}")
            object.__setattr__(self, "pe_axis", p / norm)

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


def _design_none(n_frames: int, pe_axis=None) -> Design:
    return Design("none", np.zeros(int(n_frames), dtype=int), pe_axis=pe_axis)


def _design_tag_control(volume_types: Sequence[str]) -> Design:
    vt = [str(v).lower().replace("tag", "label") for v in volume_types]
    others = sorted(set(vt) - {"control", "label"})
    seen = []
    for v in vt:
        if v not in seen:
            seen.append(v)
    if "control" in seen:
        seen.remove("control")
        seen.insert(0, "control")
    if len(seen) > 1 and seen[0] == "control" and "label" in seen:
        seen.remove("label")
        seen.insert(1, "label")
    type_to_level = {v: i for i, v in enumerate(seen)}
    levels = np.array([type_to_level[v] for v in vt], dtype=int)
    anchor_type = seen[0] if seen else "control"

    s_tc = np.array([1.0 if v == "control" else (-1.0 if v == "label" else 0.0) for v in vt])
    n_tc = int(np.sum(np.abs(s_tc) > 0))
    enc = s_tc if (n_tc >= 6 and {"control", "label"} <= set(vt)) else None
    return Design("tag_control", levels, encoding=enc,
                  detail={"other_volume_types": others, "anchor_volume_type": anchor_type,
                          "n_control_label": n_tc,
                          "encoding": "tag_control_pm1" if enc is not None else
                          f"none ({n_tc} control/label volumes of types {sorted(set(vt))})"})


def _design_shells(bvals, bvecs=None, pe_axis=None, b0_threshold: float = 50.0) -> Design:
    from .motion_reference import shell_groups

    bv = np.asarray(bvals, dtype=np.float64).reshape(-1)
    levels = np.asarray(shell_groups(bv, b0_threshold=b0_threshold), dtype=int)
    pe_cov = bias = None
    if bvecs is not None:
        bvecs_arr = np.asarray(bvecs, dtype=np.float64)
        if bvecs_arr.ndim != 2:
            raise ValueError(f"bvecs must be 2-dimensional; received shape {bvecs_arr.shape}")
        if bvecs_arr.shape == (3, len(bv)) and len(bv) != 3:
            raise ValueError("bvecs must have shape (T, 3); received (3, T). Transpose before passing.")
        if bvecs_arr.shape != (len(bv), 3):
            raise ValueError(f"bvecs shape {bvecs_arr.shape} does not match bvals length ({len(bv)}, 3)")
        g = bvecs_arr
        dw = (bv > b0_threshold) & (np.linalg.norm(g, axis=1) > 0)
        bias = _real_sh_l2(g, dw)
        if pe_axis is not None:
            pe_cov = np.zeros_like(g)
            if np.any(dw):
                norms = np.linalg.norm(g[dw], axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                pe_cov[dw] = g[dw] / norms
    return Design("shells", levels, pe_affine_covariates=pe_cov, bias_covariates=bias,
                  pe_axis=pe_axis,
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
    out = type(result)(result)   # keeps MotionCorrectionResult / GroupedMotionCorrectionResult (attribute access)
    out["design"] = {"name": design.name, "engine": engine, "call": call, "n_levels": design.n_levels,
                     "levels": np.asarray(design.levels).tolist(), **design.detail}
    return out


def _dispatch(image, design: Design, **kwargs) -> dict:
    """Today's per-modality calls (see the module docstring), unchanged."""
    from .motion import motion_correction
    from .motion_presets import get_motion_preset
    from .motion_reference import motion_correct_grouped

    if design.name == "none":
        kw = dict(kwargs)
        if design.pe_axis is not None:
            kw.setdefault("pe_axis", design.pe_axis)
        r = motion_correction(image, reference="mean", type_of_transform="Rigid", preset="bold", **kw)
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
        kw = dict(preset="dwi", **_DWI_DEFAULTS)
        if design.pe_affine_covariates is not None:
            kw.update(pe_affine_covariates=design.pe_affine_covariates, pe_axis=design.pe_axis)
        if design.bias_covariates is not None:
            kw["bias_covariates"] = design.bias_covariates
        r = motion_correct_grouped(image, design.levels, **{**kw, **kwargs})
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


# antsxdwi's registration defaults for the shells (caller values win)
_DWI_DEFAULTS = {"registration_direction": "forward", "metric_smoothing_mm": 0.0}

MIN_MIXED_FOR_BETA = 4  # frames with a neighbour of another level needed to estimate a level's offset locally
FRAMES_PER_BETA_COLUMN = 6  # a design column is estimated only with at least this many frames per column


def level_layout(levels: np.ndarray) -> Dict[int, dict]:
    """Per level: how many of its frames have a neighbour of another level (``mixed``). A level is INTERLEAVED (spread
    through the series, so its mean shares the series' mean head position) when it has at least 2 frames and at least
    half of them are mixed; otherwise it is BLOCKED (its offset is not separable from drift). Its offset to the others is
    estimable locally (``beta_estimable``) with at least ``MIN_MIXED_FOR_BETA`` mixed frames."""
    from .motion_reference import offset_identifiable

    lev = np.asarray(levels).reshape(-1)
    T = lev.size
    info, _ = offset_identifiable(lev)
    out = {}
    for k in np.unique(lev):
        k = int(k)
        idx = np.flatnonzero(lev == k)
        k_info = info.get(k, {})
        out[k] = {
            "n": int(idx.size),
            "mixed": int(k_info.get("mixed", 0)),
            "interleaved": bool(k_info.get("interleaved", False)),
            "beta_estimable": bool(k_info.get("identifiable", False)),
            "mean_time": float(idx.mean() / max(T - 1, 1)),
            "root": int(k_info.get("root", k)),
            "component": k_info.get("component", [k]),
        }
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
    """``reference_predictor`` for the reference iteration: returns the single mean 3D image of the level's
    corrected frames (sharper than the first-pass low-motion mean), without stacking T copies in memory."""
    import ants

    return ants.new_image_like(frames[0], np.mean([f.numpy() for f in frames], axis=0).astype(np.float32))


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
        if design.pe_axis is not None:
            kw.setdefault("pe_axis", design.pe_axis)
        if design.encoding is not None:
            kw["encoding_covariates"] = design.encoding
            kw["window_size"] = window_size
        r = motion_correction(image, reference="mean", type_of_transform="Rigid", backend=backend, **kw)
        return _normalized(r, design, "unified", "motion_correction(reference='mean')")

    layout = level_layout(levels)
    ks = sorted(layout)
    # identity link where the levels belong to the same connected component and the level's offset is measured locally
    # (beta on, identifiable) or where both levels share the series' mean head position (interleaved);
    # otherwise a rigid registration of the corrected means
    links = ["identity" if (layout[ks[i]]["root"] == layout[ks[i - 1]]["root"]
                            and ((beta != "none" and layout[ks[i]]["beta_estimable"])
                                 or (layout[ks[i - 1]]["interleaved"] and layout[ks[i]]["interleaved"]))) else "registration"
             for i in range(1, len(ks))]
    kw = dict(kwargs)
    dropped = []
    if design.name == "shells":
        kw = {"preset": "dwi", **_DWI_DEFAULTS, **kw}
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
        if design.name == "custom" and design.encoding is not None:
            kw["encoding_covariates"] = design.encoding
            if window_size is not None:
                kw["window_size"] = window_size
    if iterations > 0:
        kw.update(reference_predictor=_corrected_level_mean, reference_iterations=int(iterations))
    r = motion_correct_grouped(image, levels, cross_link=links, cross_group_debias=(beta != "none"), **kw)
    out = _normalized(r, design, "unified", "motion_correct_grouped(per-level references, links, cross_group_debias)")
    xo = r.get("cross_group_offset")
    out["design"].update(layout=layout, links=links, beta=beta, beta_dropped=dropped, iterations=int(iterations),
                         beta_levels=None if xo is None else {k: v for k, v in xo.items() if k != "correction_paths"},
                         beta_levels_estimable=xo is not None or beta == "none")
    return out



# Charbonnier knee of the head-motion prior in the joint engine: the registration's own precision floor (0.05 mm), so
# unexplained frame-to-frame jitter stays in the pose at nearly full amplitude and only a design column can lower it.
JOINT_SURROUND_DELTA_MM = 0.05
# Weight of that prior: OFF by default. Phantoms with known motion (simulate_dwi_series seeds 42 / 7, simulate_asl_series
# seed 2; true FD 0.156 / 0.155 / 0.160 mm): FD error 0.21 / 0.26 / 0.06 mm at 0, 0.11 / 0.20 / 0.05 at 0.02, 0.11 / 0.17 /
# 0.07 at 0.05 (where ASL's mean FD falls below the truth: jitter shrunk). But the weight is absolute: its force
# (lambda x delta per mm) competes with the image term's curvature, which depends on the image. On a featureless
# Gaussian-blob phantom with NO motion (antsxfunctional test_asl_tag_control_motion_debias) 0.02 displaced frames by
# 0.1 mm (1e-6 and 0: exact zero). A curvature-normalised prior is the next work package; until then the split between
# motion and the design columns is made by the robust re-split after every stage (lambda = 0), which already gives
# most of the gain above. Real-data gate: benchmarks/motion_cohort_gate.py.
JOINT_SURROUND_WEIGHT = 0.0


def joint_design_columns(design: Design, layout: Dict[int, dict]) -> tuple:
    """The acquisition columns ``D_t`` of the joint pass over the whole timeline, and the eddy covariates per shell.

    Returns ``(encoding (T, K) or None, pe_affine (T, 3 * n_shells) or None, dropped: list)``. Columns: one indicator per
    non-anchor level (the constant level offset), and for ``shells`` the five l = 2 direction terms of each shell in its
    own block (zero on the other frames), each block kept only with ``FRAMES_PER_BETA_COLUMN`` frames per column. The
    eddy covariates are the unit gradients in one block per shell, so the tied eddy matrix is per shell as in the
    grouped call (the eddy amplitude depends on b)."""
    levels = np.asarray(design.levels, dtype=int)
    T = levels.size
    ks = sorted(layout)
    cols, dropped = [], []
    for k in ks[1:]:
        if not layout[k]["beta_estimable"]:
            # a constant offset of a level that never alternates with another cannot be told from head motion between
            # the blocks (offset_identifiable); the level keeps the pose of its link
            dropped.append(f"offset of level {k} (not identifiable: {layout[k]['mixed']} mixed frames)")
            continue
        cols.append((levels == k).astype(np.float64)[:, None])
    pe_cov = None
    if design.name == "shells":
        if design.bias_covariates is not None:
            l2 = np.asarray(design.bias_covariates, dtype=np.float64)
            for k in ks[1:]:
                blk = np.zeros_like(l2)
                blk[levels == k] = l2[levels == k]
                live = int(np.sum(np.ptp(blk[levels == k], axis=0) > 0)) if np.any(levels == k) else 0
                if live == 0:
                    continue
                if layout[k]["n"] < FRAMES_PER_BETA_COLUMN * live:
                    dropped.append(f"l2 terms of level {k} ({layout[k]['n']} frames < {FRAMES_PER_BETA_COLUMN} x {live})")
                    continue
                cols.append(blk)
        if design.pe_affine_covariates is not None:
            g = np.asarray(design.pe_affine_covariates, dtype=np.float64)
            blocks = []
            for k in ks[1:]:
                blk = np.zeros_like(g)
                blk[levels == k] = g[levels == k]
                if np.any(blk != 0):
                    blocks.append(blk)
            pe_cov = np.hstack(blocks) if blocks else None
    enc = np.hstack(cols) if cols else None
    return enc, pe_cov, dropped


def _joint(image, design: Design, *, init: str = "levels", surround_weight: Optional[float] = None,
           surround_delta_mm: float = JOINT_SURROUND_DELTA_MM, window_size: Optional[int] = None,
           sample_mask_fn=None, level_weights: bool = False, schedule_mode: str = "refine", **kwargs) -> dict:
    """ONE optimisation over the whole acquisition timeline (docs/MOTION_JOINT_OPTIMIZATION_PLAN.md).

    H0: every level registered to its own mean and the level means linked by an explicit rigid registration of the
    corrected means (``motion_correct_grouped(cross_link="registration")``, no design columns): the DW-to-b0 link is a
    measurement. H1: all T frames in one batched pass against the per-frame reference of their own level (the corrected
    level means, all in the anchor space), each frame weighting the shared sample pool by its own level's edges
    (``frame_levels``), with every design column of the timeline in the optimiser (``joint_design_columns``: level
    offsets and the l = 2 terms per shell as contrast gauge, removed from the pose, never applied; the eddy affine per
    shell, applied) and the Charbonnier prior on the second differences of the head trajectory (knee
    ``surround_delta_mm``). The joint pass estimates the residual pose of the H0-corrected frames; the written chain is
    ``[eddy, residual rigid] + H0 chain`` and every frame is resampled ONCE from the original. A one-level design
    (BOLD) is the plain registration to the mean (no column: nothing the prior could trade against)."""
    import ants

    from .motion import jenkinson_framewise_displacement, motion_correction, _extract_rigid_parameters
    from .motion_reference import GroupedMotionCorrectionResult, _mean_image, motion_correct_grouped

    levels = np.asarray(design.levels, dtype=int)
    T = int(levels.size)
    knobs = _knobs(design)
    backend = knobs.pop("backend")
    knobs.pop("window_size", None)
    if design.n_levels == 1:
        kw = {**knobs, **kwargs}
        if design.pe_axis is not None:
            kw.setdefault("pe_axis", design.pe_axis)
        r = motion_correction(image, reference="mean", type_of_transform="Rigid", backend=backend, **kw)
        out = _normalized(r, design, "joint", "motion_correction(reference='mean')")
        out["design"].update(columns=[], surround_weight=0.0, surround_delta_mm=float(surround_delta_mm))
        return out
    if init != "levels":
        raise ValueError(f"init must be 'levels', got {init!r}")
    layout = level_layout(levels)
    enc, pe_cov, dropped = joint_design_columns(design, layout)
    lam = JOINT_SURROUND_WEIGHT if surround_weight is None else float(surround_weight)

    # H0: per-level references, no design columns. Links as in ``unified``: identity where the level's offset is
    # identifiable from its interleaving (its mean shares the series' head position, the column takes the offset),
    # a rigid registration of the corrected means otherwise (a blocked level, DWI's b0 block: an explicit measurement)
    ks = sorted(layout)
    links = ["identity" if (layout[ks[i]]["root"] == layout[ks[i - 1]]["root"]
                            and (layout[ks[i]]["beta_estimable"]
                                 or (layout[ks[i - 1]]["interleaved"] and layout[ks[i]]["interleaved"]))) else "registration"
             for i in range(1, len(ks))]
    kw0 = dict(kwargs)
    kw0.pop("sample_mask", None)
    if design.name == "shells":
        kw0 = {"preset": "dwi", **_DWI_DEFAULTS, **kw0}
    else:
        kw0.update(backend=backend, **knobs)
    if kw0.get("outprefix") is not None:
        kw0["outprefix"] = f"{kw0['outprefix']}h0_"
    h0 = motion_correct_grouped(image, levels, cross_link=links, cross_group_debias=False, **kw0)
    corrected0 = h0["motion_corrected"]
    frames0 = [ants.slice_image(corrected0, axis=3, idx=t) for t in range(T)]
    ref_level = {int(k): _mean_image([frames0[t] for t in np.flatnonzero(levels == k)]) for k in np.unique(levels)}
    refs = [ref_level[int(levels[t])] for t in range(T)]

    # H1: the joint pass on the ORIGINAL frames against their level's reference in the anchor space, started from the
    # H0 poses (the composed chain): the TOTAL pose is solved, so the design columns see the whole trajectory
    anchor = h0["group_means"][0]
    com = np.asarray(anchor.get_center_of_mass(), dtype=np.float64)
    from scipy.spatial.transform import Rotation as _Rot

    def _compose(paths):
        h = np.eye(4)
        for f in paths:
            h = _extract_rigid_parameters(ants.read_transform(f), 3)[2] @ h
        return h
    H0 = [_compose(h0["fwdtransforms"][t]) for t in range(T)]
    t_init = np.array([H[:3, 3] + (H[:3, :3] - np.eye(3)) @ com for H in H0])
    om_init = np.array([_Rot.from_matrix(H[:3, :3]).as_rotvec() for H in H0])
    kw1 = dict(kwargs)
    kw1.update(backend=backend, **{k: v for k, v in knobs.items() if k != "schedule_mode"})
    if design.name == "shells":
        kw1 = {k: v for k, v in kw1.items() if k not in ("registration_direction", "metric_smoothing_mm", "cross_num_bins")}
    if kw1.get("outprefix") is not None:
        kw1["outprefix"] = f"{kw1['outprefix']}h1_"
    if enc is not None:
        kw1["encoding_covariates"] = enc
        if window_size is not None:
            kw1["window_size"] = int(window_size)
    if pe_cov is not None:
        kw1.update(pe_affine_covariates=pe_cov, pe_axis=design.pe_axis)
    if sample_mask_fn is not None:
        kw1["sample_mask"] = sample_mask_fn(ref_level[0])
    kw1.update(surround_weight=lam, surround_delta_mm=float(surround_delta_mm), init_pose=(t_init, om_init), schedule_mode=schedule_mode)
    if level_weights:
        kw1["frame_levels"] = levels
    kw1 = {k: v for k, v in kw1.items() if k not in ("preset",)}
    r1 = motion_correction(image, reference=refs, type_of_transform="Rigid", **kw1)

    chains = [list(r1.fwdtransforms[t]) for t in range(T)]
    motion_corrected = r1["motion_corrected"]
    pe1 = r1.get("pe_affine")
    jac = np.ones(T) if pe1 is None else np.asarray(pe1["jacobian"], dtype=float)
    composed_all = [_compose(c) for c in chains]
    composed_rigid = [_compose(c[-1:]) for c in chains]
    fd_radius = float(kwargs.get("fd_radius", 50.0))
    fd_raw = jenkinson_framewise_displacement(composed_all, fd_radius, com)
    fd = jenkinson_framewise_displacement(composed_rigid, fd_radius, com)
    translations_mm = np.zeros((T, 3)); rotations_deg = np.zeros((T, 3))
    for i, H in enumerate(composed_rigid):
        R_i = H[:3, :3]
        translations_mm[i] = H[:3, 3] + (R_i - np.eye(3)) @ com
        rotations_deg[i] = np.degrees(_Rot.from_matrix(R_i).as_euler("xyz"))

    enc_info = r1.get("encoding_pose_bias_mm_rad")
    beta_levels = None
    est_levels = [k for k in ks[1:] if layout[k]["beta_estimable"]]
    if enc_info is not None and est_levels:
        dY = np.asarray(enc_info["delta_Y_mm_rad"], dtype=float)
        by = {int(k): np.mean(dY[levels == k], axis=0) - np.mean(dY[levels == 0], axis=0) for k in est_levels}
        beta_levels = {"method": "joint_in_optimizer_level_columns",
                       "offset_mm_rad_by_group": {k: v.tolist() for k, v in by.items()},
                       "offset_mm_by_group": {k: float(np.linalg.norm(v[:3])) for k, v in by.items()},
                       "offset_deg_by_group": {k: float(np.degrees(np.linalg.norm(v[3:]))) for k, v in by.items()},
                       "not_identifiable_groups": [k for k in ks[1:] if k not in est_levels]}
    pe_affine = None
    if pe1 is not None:
        p_unit = np.asarray(design.pe_axis, dtype=np.float64) / np.linalg.norm(design.pe_axis)
        a_all = np.asarray(pe1["a"], dtype=float)
        pe_affine = {"axis": [float(x) for x in p_unit], "coefficients_by_group": [pe1.get("coefficients")],
                     "a": a_all, "jacobian": jac, "shift_mm": a_all[:, 0].copy()}

    out = GroupedMotionCorrectionResult()
    out.update(motion_corrected=motion_corrected, fd=fd, fd_raw=fd_raw, translations_mm=translations_mm, rotations_deg=rotations_deg,
               nuisance=None, pe_affine=pe_affine, reference_modes=["joint: corrected level means"],
               masked_metric=sample_mask_fn is not None, bias=[r1.get("bias")], encoding_pose_bias_mm_rad=[enc_info],
               cross_group_offset=beta_levels, group_offset_fd=None, pe_pseudomotion_debias=None,
               frame_converged=(np.asarray(r1["convergence"]["frame_converged"], dtype=bool) if r1.get("convergence") else np.ones(T, dtype=bool)),
               cross_converged=list(h0.get("cross_converged") or []), cross_link=list(h0.get("cross_link") or []),
               fwdtransforms=chains, group_results=[r1], group_indices=[np.arange(T)], group_means=[ref_level[int(k)] for k in sorted(ref_level)],
               cross_registrations=h0.get("cross_registrations"), preset=kw0.get("preset"), h0=h0)
    out = _normalized(out, design, "joint", "motion_correct_grouped(H0 links) + motion_correction(per-frame refs, D_t, prior)")
    out["design"].update(layout=layout, links=list(h0.get("cross_link") or []), beta="columns", beta_dropped=dropped,
                         columns=([f"level{k}" for k in est_levels] + ([f"l2 x {design.n_levels - 1} shells"] if (enc is not None and design.name == "shells" and enc.shape[1] > len(est_levels)) else [])),
                         n_columns=0 if enc is None else int(enc.shape[1]), surround_weight=lam, surround_delta_mm=float(surround_delta_mm),
                         beta_levels=beta_levels, beta_levels_estimable=beta_levels is not None)
    return out


ENGINES = {"dispatch": _dispatch, "unified": _unified, "joint": _joint}


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
