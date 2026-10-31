"""Slice-set clock: is a breathing-band pose displacement volume-locked or continuous? (WIP, opt-in; 2026-10-31)

An interleaved 2-D EPI acquisition records one set of slices (odd or even index) about half a TR before the other.
Registering the two half-volumes separately gives two samples of the pose per volume, about TR/2 apart. A displacement
that the scanner applies between volumes (a prospective field-of-view update, e.g. PACE) is identical in both sets:
in phase, coherent. A displacement that evolves continuously (bed, head) is sampled at two times and the sets differ
by a predictable phase, 360 f dt degrees at the band frequency f. On breacher resting BOLD (TR 2.0 s, sets 0.47 s
and 1.46 s) a volume-locked synthetic pair reads 3 to 5 degrees and a continuous one 69 degrees
(``docs/BREACHER_PSEUDOMOTION_SINGLE_SUBJECT_PLAN.md``, M1).

Stage 1 of that plan, implemented here: per degree of freedom, the share of the band-limited full-volume pose that the
two half-sets hold in common and in phase (coherent, phase near zero) is taken as not head motion and removed from
the FD used for censoring (``fd_head_vl``). The applied transforms and the resampled series are never modified. A
white-jitter trajectory is incoherent between the sets, so nothing is removed (``tests/test_motion_slice_sets.py``).
The per-run calibration on a synthetic pair from the run's own mean image (plan step 3) is not implemented; fixed
thresholds from the ten-run calibration are used and recorded.

Limits: two sets give the volume-locked share, not the head trajectory (that is Stage 2, three or more sets); a
band-limited component incoherent between the sets stays in FD (it is not rigid head motion, but it may be aliased
pulsation); the band is fixed or taken from the caller; the early set is read from the sidecar ``SliceTiming`` or
assumed (recorded in ``early_set_source``).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import ants
import numpy as np

DOF = ("t_ro", "t_pe", "t_sl", "pitch", "roll", "yaw")
DEFAULT_BAND_HZ = (0.15, 0.25)
VOLUME_LOCKED_MAX_PHASE_DEG = 35.0   # ten-run calibration: volume-locked 3-5 deg, continuous 69 deg
CONTINUOUS_MIN_PHASE_DEG = 55.0
MIN_COHERENCE = 0.4
MIN_FRAMES = 64


def _bandpass(x: np.ndarray, tr: float, lo: float, hi: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(x), d=tr)
    X[(f < lo) | (f > hi)] = 0.0
    return np.fft.irfft(X, n=len(x))


def _clock_stats(early: np.ndarray, late: np.ndarray, tr: float, band: Tuple[float, float], dt_sets: float) -> Dict[str, float]:
    """Phase of ``late`` relative to ``early`` in ``band`` (degrees) and coherence, both weighted by cross-power, and the
    band centroid (Hz)."""
    from scipy.signal import coherence, csd

    nper = min(64, len(early))
    f, C = coherence(early - early.mean(), late - late.mean(), fs=1.0 / tr, nperseg=nper)
    _, S = csd(early - early.mean(), late - late.mean(), fs=1.0 / tr, nperseg=nper)
    sel = (f >= band[0]) & (f <= band[1])
    if not sel.any():
        return {"phase_deg": float("nan"), "coherence": 0.0, "f_centroid_hz": float("nan"), "continuous_pred_deg": float("nan")}
    pw = np.abs(S[sel])                                   # cross-power: bins without signal carry no weight
    w = C[sel] * pw
    unit = S[sel] / np.maximum(pw, 1e-30)
    phase = float(np.degrees(np.angle(np.sum(w * unit))))
    fc = float(np.sum(w * f[sel]) / max(float(w.sum()), 1e-12))
    coh = float(np.sum(C[sel] * pw) / max(float(pw.sum()), 1e-30))
    return {"phase_deg": phase, "coherence": coh, "f_centroid_hz": fc, "continuous_pred_deg": 360.0 * fc * dt_sets}


def _reading(stats: Dict[str, float]) -> str:
    if not np.isfinite(stats["phase_deg"]) or stats["coherence"] < MIN_COHERENCE:
        return "no_coherent_band"
    if abs(stats["phase_deg"]) <= VOLUME_LOCKED_MAX_PHASE_DEG:
        return "volume_locked"
    if abs(stats["phase_deg"]) >= CONTINUOUS_MIN_PHASE_DEG:
        return "continuous"
    return "mixed"


def _power_fd(X6: np.ndarray) -> np.ndarray:
    """Power FD of a (T, 6) pose array whose rotations are already in mm at the FD radius."""
    return np.r_[0.0, np.abs(np.diff(X6, axis=0)).sum(axis=1)]


def slice_set_clock_from_poses(
    full: np.ndarray,
    early: np.ndarray,
    late: np.ndarray,
    tr_s: float,
    band_hz: Tuple[float, float] = DEFAULT_BAND_HZ,
    dt_sets_s: float = 0.99,
) -> Dict[str, Any]:
    """Numeric core. ``full``, ``early``, ``late``: (T, 6) poses in one frame, rotations in mm at the FD radius.

    Returns per-DOF clock statistics and readings, the removed volume-locked component ``x_vl`` (T, 6), the head pose
    ``x_head = full - x_vl``, and ``fd`` (Power FD of ``full``) and ``fd_head_vl`` (Power FD of ``x_head``). Only a DOF
    whose reading is ``volume_locked`` has anything removed; the removed part is the band-limited full-volume pose scaled
    by r^2(early, late) in the band, r clipped at zero.
    """
    full = np.asarray(full, dtype=np.float64)
    early = np.asarray(early, dtype=np.float64)
    late = np.asarray(late, dtype=np.float64)
    if full.shape != early.shape or full.shape != late.shape or full.ndim != 2 or full.shape[1] != 6:
        raise ValueError(f"full, early, late must share shape (T, 6); got {full.shape}, {early.shape}, {late.shape}")
    T = full.shape[0]
    lo, hi = float(band_hz[0]), float(band_hz[1])
    nyq = 0.5 / float(tr_s)
    if hi > nyq + 1e-9:
        raise ValueError(f"band {band_hz} exceeds the Nyquist frequency {nyq:.3f} Hz at TR {tr_s} s")
    x_vl = np.zeros_like(full)
    per_dof: Dict[str, Dict[str, Any]] = {}
    for k, name in enumerate(DOF):
        if T < MIN_FRAMES:
            stats = {"phase_deg": float("nan"), "coherence": 0.0, "f_centroid_hz": float("nan"), "continuous_pred_deg": float("nan")}
        else:
            stats = _clock_stats(early[:, k], late[:, k], float(tr_s), (lo, hi), float(dt_sets_s))
        reading = _reading(stats)
        eb, lb, fb = (_bandpass(v[:, k], float(tr_s), lo, hi) for v in (early, late, full))
        r = float(np.corrcoef(eb, lb)[0, 1]) if eb.std() > 1e-12 and lb.std() > 1e-12 else 0.0
        share = float(max(r, 0.0) ** 2) if reading == "volume_locked" else 0.0
        x_vl[:, k] = share * fb
        per_dof[name] = {**{a: round(float(b), 4) for a, b in stats.items()}, "reading": reading, "r_early_late": round(r, 4),
                         "removed_share": round(share, 4), "band_sd_full_mm": round(float(fb.std()), 4),
                         "removed_rms_mm": round(float(np.sqrt(np.mean(x_vl[:, k] ** 2))), 4)}
    x_head = full - x_vl
    fd = _power_fd(full)
    fd_head = _power_fd(x_head)
    return {
        "per_dof": per_dof,
        "readings": {n: per_dof[n]["reading"] for n in DOF},
        "x_vl": x_vl, "x_head": x_head,
        "fd": fd, "fd_head_vl": fd_head,
        "fd_mean": float(fd[1:].mean()) if T > 1 else 0.0,
        "fd_head_vl_mean": float(fd_head[1:].mean()) if T > 1 else 0.0,
        "band_hz": (lo, hi), "tr_s": float(tr_s), "dt_sets_s": float(dt_sets_s),
        "thresholds": {"volume_locked_max_phase_deg": VOLUME_LOCKED_MAX_PHASE_DEG,
                       "continuous_min_phase_deg": CONTINUOUS_MIN_PHASE_DEG, "min_coherence": MIN_COHERENCE},
    }


def _early_set_from_sidecar(sidecar: Any) -> Tuple[Optional[int], Optional[float], str]:
    """(parity of the early set, mean time difference late - early in s, source) from a sidecar's SliceTiming."""
    if sidecar is None:
        return None, None, "none"
    meta = sidecar if isinstance(sidecar, dict) else json.loads(Path(sidecar).read_text(encoding="utf-8"))
    st = meta.get("SliceTiming")
    if not st or len(st) < 4:
        return None, None, "sidecar_without_slice_timing"
    st = np.asarray(st, dtype=np.float64)
    even, odd = st[0::2], st[1::2]
    # a two-set interleave puts the two parities in disjoint time ranges (odd 0 to 0.94 s, even 0.99 to 1.93 s on
    # breacher); sequential and multiband orders overlap them, so the parity split is not the timing split
    tol = 0.02 * (st.max() - st.min() + 1e-9)
    if not (odd.max() < even.min() + tol or even.max() < odd.min() + tol):
        return None, None, "sidecar_not_interleaved_by_parity"
    early = 1 if odd.mean() < even.mean() else 0
    return early, float(abs(even.mean() - odd.mean())), "sidecar:SliceTiming"


def _half_volume(img: ants.ANTsImage, k0: int) -> ants.ANTsImage:
    arr = img.numpy()
    sp = list(img.spacing)
    D = img.direction
    org = list(img.origin)
    sub = np.ascontiguousarray(arr[:, :, k0::2, :])
    o3 = np.asarray(org[:3], dtype=np.float64) + D[:3, :3] @ np.array([0.0, 0.0, k0 * sp[2]])
    return ants.from_numpy(sub, origin=[float(v) for v in o3] + [org[3]], spacing=[sp[0], sp[1], 2 * sp[2], sp[3]], direction=D)


def _poses_in_frame(translations: np.ndarray, rotations_deg: np.ndarray, P: Sequence[np.ndarray], radius: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rot

    p_pe, p_sl, p_ro = P
    t = np.asarray(translations, dtype=np.float64)
    rv = Rot.from_euler("xyz", np.asarray(rotations_deg, dtype=np.float64), degrees=True).as_rotvec() * float(radius)
    return np.column_stack([t @ p_ro, t @ p_pe, t @ p_sl, rv @ p_ro, rv @ p_pe, rv @ p_sl])


def slice_set_clock(
    image: ants.ANTsImage,
    result: Dict[str, Any],
    *,
    sidecar: Any = None,
    early_set: Optional[int] = None,
    dt_sets_s: Optional[float] = None,
    band_hz: Tuple[float, float] = DEFAULT_BAND_HZ,
    pe_axis: Any = None,
    slice_axis: Any = None,
    fd_radius: float = 50.0,
    device: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Register the two interleaved slice sets of ``image`` and attach ``fd_head_vl`` to ``result`` (opt-in, in place).

    ``result`` is a ``motion_correction`` / ``motion_correct_design`` result for ``image`` (4-D, slice axis = third
    voxel axis); its raw poses (``motion_parameters_raw`` when a PE debias ran, else ``motion_parameters``) are the
    full-volume series. ``early_set``: parity (0 or 1) of the slice set acquired first; read from the sidecar
    ``SliceTiming`` when not given. With neither, parity 1 (odd) is assumed with a warning and
    ``early_set_source='assumed_odd'``. A sidecar whose SliceTiming is not interleaved by parity (sequential or
    multiband) raises: the clock needs the two parities about TR/2 apart. ``dt_sets_s``: mean acquisition-time difference of the sets (sidecar, or TR/2).

    Adds to ``result``: ``slice_set_clock`` (the numeric core's dict with the half-set poses, readings, thresholds and
    provenance) and ``fd_head_vl`` (T,). Returns ``result['slice_set_clock']``. Costs two half-volume registrations.
    """
    from .acquisition_axes import encoding_frame, infer_slice_axis
    from .motion_design import make_design, motion_correct_design
    from .phase_encode import resolve_phase_encode

    if image.dimension != 4:
        raise ValueError("slice_set_clock needs a 4-D series")
    tr = float(image.spacing[3])
    if not tr > 0:
        raise ValueError("slice_set_clock needs a positive time spacing (TR) in the image header")
    parity, dt_side, source = _early_set_from_sidecar(sidecar)
    if early_set is not None:
        parity, source = int(early_set), "argument"
    elif source == "sidecar_not_interleaved_by_parity":
        raise ValueError(
            "slice_set_clock: the sidecar SliceTiming shows the odd and even slices are not acquired about TR/2 apart "
            "(sequential or multiband acquisition); the two-set clock does not apply. Pass early_set explicitly only if "
            "you know the parity split is the timing split."
        )
    if parity is None:
        warnings.warn("slice_set_clock: no SliceTiming; assuming the odd-index slices are acquired first", stacklevel=2)
        parity, source = 1, "assumed_odd"
    dt = float(dt_sets_s) if dt_sets_s is not None else (dt_side if dt_side is not None else 0.5 * tr)

    if pe_axis is None:
        pe_axis = np.asarray(resolve_phase_encode(image, sidecar=sidecar, on_oblique="warn").physical_vector, dtype=np.float64)
    sl_ax = infer_slice_axis(image, sidecar=sidecar).voxel_axis if slice_axis is None else int(slice_axis)
    if sl_ax != 2:
        raise ValueError(f"slice_set_clock splits the third voxel axis; the slice axis is {sl_ax}")
    P = encoding_frame(image.direction[:3, :3], np.asarray(pe_axis, dtype=np.float64), sl_ax)

    mp = result.get("motion_parameters_raw")
    if mp is None:
        mp = result.get("motion_parameters")
    if mp is None:
        raise ValueError("result has no motion_parameters")
    full = _poses_in_frame(mp.translations, mp.rotations_deg, P, fd_radius)

    kw = {"engine": "joint"}
    if device is not None:
        kw["device"] = device
    halves = {}
    for name, k0 in (("early", parity), ("late", 1 - parity)):
        h = _half_volume(image, k0)
        r = motion_correct_design(h, make_design("none", n_frames=int(h.shape[-1])), verbose=verbose, **kw)
        hm = r["motion_parameters"]
        halves[name] = _poses_in_frame(hm.translations, hm.rotations_deg, P, fd_radius)
    core = slice_set_clock_from_poses(full, halves["early"], halves["late"], tr, band_hz=band_hz, dt_sets_s=dt)
    core.update(early_set_parity=int(parity), early_set_source=source, poses_full=full, poses_early=halves["early"],
                poses_late=halves["late"], dof=list(DOF), frame="acquisition (ro, pe, sl); rotations mm at fd_radius",
                fd_radius=float(fd_radius), pe_axis=[float(v) for v in np.asarray(pe_axis)])
    result["slice_set_clock"] = core
    result["fd_head_vl"] = core["fd_head_vl"]
    if isinstance(result.get("summary"), dict):
        result["summary"].update(fd_head_vl_mean=core["fd_head_vl_mean"], slice_set_clock_reading_t_sl=core["readings"]["t_sl"])
    return core
