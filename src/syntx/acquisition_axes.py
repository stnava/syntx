"""Acquisition axes of an EPI / ASL series from metadata or, failing that, from the image itself.

One implementation, used wherever a slice or phase-encode axis is needed (the phase-encode B0 / Nyquist debias of the
motion trajectory, ``resolve_phase_encode(allow_inference=True, inference=infer_phase_encode_axis)``).

Slice axis (``infer_slice_axis``), first decisive source wins:

1. ``slice_axis`` stated by the caller (BIDS ``i/j/k[-]`` label or voxel index);
2. the sidecar's ``SliceEncodingDirection``;
3. the sidecar's ``SliceTiming`` length matching exactly one voxel axis size;
4. voxel thickness: one axis at least ``THICK_RATIO`` x the median spacing of the other two (thick slices / 3D
   partitions);
5. slice interleave: in an interleaved acquisition neighbouring slices are acquired about TR/2 apart while slices two
   apart are close in time, so the plane-mean time series correlate LESS with their neighbour than with the plane two
   away along the slice axis only (in-plane axes: spatial smoothness makes the neighbour the most similar).
   Score per axis ``median_i[corr(i, i+2) - corr(i, i+1)]``; the slice axis must lead the runner-up by at least
   ``INTERLEAVE_MARGIN`` in the whole series and in both halves of the frames. The per-level mean is removed first
   when ``levels`` is given (DWI shells, ASL tag/control: the contrast swing is common to all planes and hides the
   interleave). Needs at least ``MIN_FRAMES_INTERLEAVE`` frames.

Otherwise ``AcquisitionAxisUndecidable`` (no silent default).

Phase-encode axis (``infer_phase_encode_axis``), among the two in-plane axes (the slice axis excluded): the Nyquist
N/2 ghost puts signal in the background where the head lands after a half field-of-view shift along phase encode.
Score per axis: median background intensity at those positions over the median of the remaining background. The
winner must lead by ``GHOST_RATIO``; with at least ``MIN_FRAMES_SHIFT`` frames the cross-check is the frame-to-frame
spread of the apparent shift along each axis (B0 fluctuations displace the image along phase encode) and it must name
the same axis. Only the AXIS is inferred (unsigned).

Thresholds are provisional, set at gaps in measurements on FPA sub-BLAST079 (BOLD MB8, DWI MB4, 3D PCASL; slice k,
phase encode j): interleave margin 0.076 (BOLD), 0.040 (DWI, levels removed), none in 3D ASL (thickness ratio 2.13);
ghost ratio 3.2 (BOLD), 2.3 (DWI), 130 (ASL).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

THICK_RATIO = 1.25
INTERLEAVE_MARGIN = 0.02
MIN_FRAMES_INTERLEAVE = 20
GHOST_RATIO = 1.5
MIN_FRAMES_SHIFT = 50


class AcquisitionAxisUndecidable(ValueError):
    """Neither metadata nor the image names the axis with the required margin."""


@dataclass(frozen=True)
class AxisInference:
    """A voxel axis (0/1/2 = BIDS i/j/k) with its source and the evidence behind it."""

    voxel_axis: int
    source: str  # 'explicit' | 'sidecar' | 'slice_timing' | 'voxel_thickness' | 'interleave' | 'nyquist_ghost'
    scores: dict = field(default_factory=dict)


def _label_axis(raw: Any) -> int:
    if isinstance(raw, (int, np.integer)) and 0 <= int(raw) <= 2:
        return int(raw)
    if isinstance(raw, str) and raw.strip().rstrip("-+") in ("i", "j", "k"):
        return "ijk".index(raw.strip().rstrip("-+"))
    raise ValueError(f"slice axis {raw!r}: expected a voxel index 0/1/2 or a BIDS label i/j/k[-]")


def _sidecar_dict(sidecar: Any) -> dict:
    if sidecar is None:
        return {}
    if isinstance(sidecar, dict):
        return sidecar
    return json.loads(Path(str(sidecar)).read_text())


def _array4d(image: Any) -> np.ndarray:
    a = np.asarray(image.numpy(), dtype=np.float32)
    if a.ndim != 4:
        raise ValueError(f"expected a 4-D series, got {a.ndim}-D")
    return a


def _head_mask(mean3: np.ndarray) -> np.ndarray:
    import ants

    m = ants.get_mask(ants.from_numpy(np.ascontiguousarray(mean3, dtype=np.float32)))
    return ants.iMath(m, "MD", 2).numpy() > 0


def _remove_levels(a4: np.ndarray, levels: Optional[np.ndarray]) -> np.ndarray:
    if levels is None:
        return a4
    lev = np.asarray(levels).reshape(-1)
    if lev.size != a4.shape[-1]:
        raise ValueError(f"levels has {lev.size} entries for {a4.shape[-1]} frames")
    out = a4.copy()
    for k in np.unique(lev):
        sel = lev == k
        out[..., sel] -= out[..., sel].mean(-1, keepdims=True)
    return out + a4.mean(-1, keepdims=True)


def interleave_scores(a4: np.ndarray, mask: np.ndarray) -> list:
    """Per voxel axis: ``median_i[corr(i, i+2) - corr(i, i+1)]`` of the detrended plane-mean time series."""
    T = a4.shape[-1]
    t = np.linspace(-1.0, 1.0, T)
    X = np.c_[np.ones(T), t, t ** 2]
    out = []
    for ax in range(3):
        ts = []
        for i in range(a4.shape[ax]):
            sl = [slice(None)] * 3
            sl[ax] = i
            mk = mask[tuple(sl)]
            if mk.sum() < 20:
                ts.append(None)
                continue
            y = a4[tuple(sl)][mk].mean(0).astype(np.float64)
            y = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
            sd = y.std()
            ts.append(y / sd if sd > 0 else None)
        d = [np.mean(ts[i] * ts[i + 2]) - np.mean(ts[i] * ts[i + 1]) for i in range(len(ts) - 2)
             if ts[i] is not None and ts[i + 1] is not None and ts[i + 2] is not None]
        out.append(float(np.median(d)) if d else float("nan"))
    return out


def ghost_scores(mean3: np.ndarray, mask: np.ndarray) -> list:
    """Per voxel axis: background intensity where the head lands after a half-FOV shift, over the rest of the
    background (medians)."""
    bg = ~mask
    out = []
    for a in range(3):
        sh = np.roll(mask, mean3.shape[a] // 2, axis=a) & bg
        rest = bg & ~sh
        if sh.sum() < 50 or rest.sum() < 50:
            out.append(float("nan"))
            continue
        out.append(float(np.median(mean3[sh]) / max(float(np.median(mean3[rest])), 1e-6)))
    return out


def shift_step_spread(a4: np.ndarray, mask: np.ndarray, spacing) -> list:
    """Per voxel axis: frame-to-frame spread (mm) of the apparent shift along that axis (first-order flow on the mean
    over the strongest-gradient quarter of the head, cubic trend removed)."""
    m = a4.mean(-1)
    g = np.gradient(m, *[float(s) for s in spacing[:3]])
    gm = np.sqrt(sum(gi ** 2 for gi in g))
    mk = mask & (gm > np.percentile(gm[mask], 75))
    G = np.stack([gi[mk] for gi in g], 1).astype(np.float64)
    D = (a4[mk] - m[mk][:, None]).astype(np.float64)
    v = -np.linalg.lstsq(G, D, rcond=None)[0]
    T = a4.shape[-1]
    t = np.linspace(-1.0, 1.0, T)
    X = np.c_[np.ones(T), t, t ** 2, t ** 3]
    r = v.T - X @ np.linalg.lstsq(X, v.T, rcond=None)[0]
    return [float(np.std(np.diff(r[:, a]))) for a in range(3)]


def infer_slice_axis(image: Any, *, slice_axis: Any = None, sidecar: Any = None,
                     levels: Optional[np.ndarray] = None) -> AxisInference:
    """The acquired slice (or 3D partition) voxel axis of a 4-D series; see the module docstring for the sources."""
    if slice_axis is not None:
        return AxisInference(_label_axis(slice_axis), "explicit", {"raw": str(slice_axis)})
    meta = _sidecar_dict(sidecar)
    if meta.get("SliceEncodingDirection"):
        return AxisInference(_label_axis(meta["SliceEncodingDirection"]), "sidecar",
                             {"SliceEncodingDirection": meta["SliceEncodingDirection"]})
    shape = tuple(int(n) for n in image.shape[:3])
    st = meta.get("SliceTiming")
    if st:
        hits = [a for a in range(3) if shape[a] == len(st)]
        if len(hits) == 1:
            return AxisInference(hits[0], "slice_timing", {"n_slice_timing": len(st), "shape": shape})
    if meta.get("dcmmeta_slice_dim") in (0, 1, 2):
        sd = int(meta["dcmmeta_slice_dim"])
        return AxisInference(sd, "dcmmeta_slice_dim", {"dcmmeta_slice_dim": sd, "shape": shape})
    spacing = np.asarray(image.spacing[:3], dtype=np.float64)
    thick = [float(spacing[a] / np.median(np.delete(spacing, a))) for a in range(3)]
    scores: dict = {"thickness_ratio": thick, "shape": shape}
    a4 = _array4d(image)
    T = a4.shape[-1]
    il = None
    if T >= MIN_FRAMES_INTERLEAVE:
        mask = _head_mask(a4.mean(-1))
        r = _remove_levels(a4, levels)
        il = interleave_scores(r, mask)
        halves = [interleave_scores(r[..., :T // 2], mask), interleave_scores(r[..., T // 2:], mask)]
        scores.update(interleave=il, interleave_halves=halves)
    ithick = int(np.argmax(thick))
    if thick[ithick] >= THICK_RATIO:
        if il is not None:
            lead = int(np.nanargmax(il))
            margin = il[lead] - max(v for a, v in enumerate(il) if a != lead)
            if lead != ithick and margin >= INTERLEAVE_MARGIN:
                raise AcquisitionAxisUndecidable(
                    f"slice axis: the thick axis {'ijk'[ithick]} and the interleave axis {'ijk'[lead]} disagree "
                    f"(scores {scores}); pass slice_axis")
        return AxisInference(ithick, "voxel_thickness", scores)
    if il is None:
        raise AcquisitionAxisUndecidable(
            f"slice axis: no metadata, near-isotropic voxels {np.round(spacing, 3).tolist()} and {T} frames "
            f"(< {MIN_FRAMES_INTERLEAVE}) for the interleave test; pass slice_axis or the sidecar")

    def lead_margin(s):
        k = int(np.nanargmax(s))
        return k, s[k] - max(v for a, v in enumerate(s) if a != k)

    k, margin = lead_margin(il)
    hk = [lead_margin(h) for h in scores["interleave_halves"]]
    scores["interleave_margin"] = float(margin)
    if margin >= INTERLEAVE_MARGIN and all(h[0] == k and h[1] > 0 for h in hk):
        return AxisInference(k, "interleave", scores)
    other_dims = [shape[a] for a in range(3) if a != k]
    if margin > 0 and all(h[0] == k and h[1] > 0 for h in hk) and other_dims[0] == other_dims[1] and shape[k] < other_dims[0]:
        return AxisInference(k, "interleave_and_slab_shape", scores)
    raise AcquisitionAxisUndecidable(
        f"slice axis: interleave test not decisive (lead {'ijk'[k]}, margin {margin:.3f} < {INTERLEAVE_MARGIN} or "
        f"halves disagree: {hk}); pass slice_axis or the sidecar")


def infer_phase_encode_axis_detail(image: Any, *, slice_axis: Optional[int] = None,
                                   levels: Optional[np.ndarray] = None) -> AxisInference:
    """The in-plane voxel axis carrying the Nyquist ghost (unsigned phase-encode axis)."""
    if slice_axis is None:
        slice_axis = infer_slice_axis(image, levels=levels).voxel_axis
    a4 = _array4d(image)
    mean3 = a4.mean(-1)
    mask = _head_mask(mean3)
    cand = [a for a in range(3) if a != int(slice_axis)]
    gh = ghost_scores(mean3, mask)
    g = {a: gh[a] for a in cand}
    if any(not np.isfinite(v) for v in g.values()):
        raise AcquisitionAxisUndecidable(f"phase-encode axis: ghost test needs background on both in-plane axes ({gh})")
    win = max(cand, key=g.get)
    lose = min(cand, key=g.get)
    ratio = g[win] / max(g[lose], 1e-6)
    scores: dict = {"slice_axis": int(slice_axis), "ghost": gh, "ghost_ratio": float(ratio)}
    if ratio < GHOST_RATIO:
        raise AcquisitionAxisUndecidable(f"phase-encode axis: ghost ratio {ratio:.2f} < {GHOST_RATIO} ({scores})")
    if a4.shape[-1] >= MIN_FRAMES_SHIFT:
        sh = shift_step_spread(_remove_levels(a4, levels), mask, image.spacing)
        scores["shift_step_mm"] = sh
        if max(cand, key=lambda a: sh[a]) != win:
            raise AcquisitionAxisUndecidable(
                f"phase-encode axis: the ghost names {'ijk'[win]} but the frame-to-frame shift names the other "
                f"in-plane axis ({scores})")
    return AxisInference(win, "nyquist_ghost", scores)


def phase_encode_inference(
    image: Any = None,
    *,
    levels: Optional[np.ndarray] = None,
    slice_axis: Any = None,
    sidecar: Any = None,
    **kwargs,
):
    """Infer the in-plane phase-encode axis and return a :class:`PhaseEncode` object.

    Contains exact voxel axis, physical unit vector (preserving oblique orientation),
    physical axis index, alignment, and evidence scores.
    """
    from .phase_encode import PhaseEncode

    img = image if image is not None else kwargs.get("image")
    if img is None:
        raise ValueError("phase_encode_inference requires an image")

    sl_inf = infer_slice_axis(img, slice_axis=slice_axis, sidecar=sidecar, levels=levels)
    pe_detail = infer_phase_encode_axis_detail(img, slice_axis=sl_inf.voxel_axis, levels=levels)
    vax = int(pe_detail.voxel_axis)

    # Calculate physical vector from direction matrix
    d = np.asarray(getattr(img, "direction", np.eye(3)), dtype=np.float64)[:3, :3]
    d = d / np.maximum(np.linalg.norm(d, axis=0, keepdims=True), 1e-12)
    p_vec = d[:, vax]
    phys_ax = int(np.argmax(np.abs(p_vec)))
    alignment = float(np.abs(p_vec[phys_ax]))

    evidence = dict(pe_detail.scores)
    evidence["slice_axis_source"] = sl_inf.source
    evidence["slice_axis_scores"] = sl_inf.scores

    return PhaseEncode(
        voxel_axis=vax,
        sign=None,
        physical_vector=tuple(float(x) for x in p_vec),
        physical_axis=phys_ax,
        alignment=alignment,
        label=None,
        anatomical=None,
        source="inferred",
        source_detail=f"acquisition_axes:{pe_detail.source}",
        evidence=evidence,
    )


def infer_phase_encode_axis(image: Any) -> int:
    """``resolve_phase_encode(allow_inference=True, inference=infer_phase_encode_axis)`` callable: the PHYSICAL axis
    (0/1/2 = x/y/z) of the inferred in-plane phase-encode voxel axis."""
    pe = phase_encode_inference(image)
    return pe.physical_axis


def encoding_frame(direction: Any, pe_axis: Any, slice_axis: int) -> tuple:
    """Unit physical vectors ``(p_pe, p_sl, p_ro)``: phase encode (voxel index 0/1/2 or physical 3-vector), the slice
    voxel axis orthogonalised against it, and readout = ``p_pe x p_sl``."""
    D = np.asarray(direction, dtype=np.float64)[:3, :3].copy()
    D /= np.maximum(np.linalg.norm(D, axis=0, keepdims=True), 1e-12)
    arr = np.asarray(pe_axis, dtype=np.float64).reshape(-1)
    if arr.size == 1 and float(arr[0]) in (0.0, 1.0, 2.0):
        p_pe = D[:, int(arr[0])].copy()
    elif arr.size == 3 and float(np.linalg.norm(arr)) > 1e-8:
        p_pe = arr / float(np.linalg.norm(arr))
    else:
        raise ValueError(f"pe_axis must be a voxel axis index (0, 1, 2) or a non-zero 3-vector, got {pe_axis!r}")
    j_pe = int(np.argmax(np.abs(D.T @ p_pe)))
    k = int(slice_axis)
    if k == j_pe:
        raise ValueError(f"slice axis {'ijk'[k]} is the phase-encode axis")
    p_sl = D[:, k] - float(np.dot(D[:, k], p_pe)) * p_pe
    p_sl /= max(float(np.linalg.norm(p_sl)), 1e-12)
    p_ro = np.cross(p_pe, p_sl)
    p_ro /= max(float(np.linalg.norm(p_ro)), 1e-12)
    return p_pe, p_sl, p_ro
