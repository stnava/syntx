"""Phase-encode (PE) direction: the one place it is parsed, resolved against a header and turned into a restriction.

Why this module exists
----------------------
Every antsx* repo needs "which physical axis was this EPI series distorted along, and in which
direction?". The answer is a function of (a) what the metadata states and (b) the image header, and
getting either wrong gives a *worse* registration than none (a restriction along the wrong axis
forbids the real distortion). The rules here are therefore strict:

* There is **no default axis**. No source and no inference -> :class:`PhaseEncodeUndecidable`.
* Several sources must **agree after being mapped through the header**, else :class:`PhaseEncodeConflict`.
* An unrecognised word (including DICOM ``ROW`` / ``COL``, and ``x``/``y``/``z``) is a
  :class:`PhaseEncodeVocabularyError`, never silently mapped.
* Opting out of the restriction is explicit (``pe_policy="unrestricted"``) and is recorded.

Conventions
-----------
Physical coordinates are ITK / ANTs **LPS**: +x = Left, +y = Posterior, +z = Superior.
``image.direction`` columns are the voxel axes i, j, k expressed in LPS. (A NIfTI that is RAS in
nibabel appears here with x and y negated by ANTs; this module never looks at nibabel.)

* BIDS ``PhaseEncodingDirection`` ``i``/``j``/``k`` (optionally ``-``): the encoding runs along
  increasing (``+``) or decreasing (``-``) voxel index. Physical vector = ``sign * direction[:, axis]``.
* Anatomical names ``"XY"`` mean "acquired from X toward Y": ``AP`` = +y, ``PA`` = -y,
  ``LR`` = -x, ``RL`` = +x, ``SI`` = -z, ``IS`` = +z (LPS). They name a physical vector and are
  mapped back to a voxel axis through the header; the result is the same ``PhaseEncode`` a BIDS
  label would give.
* A leading ``dir-`` (the BIDS entity) is stripped from labels.
* ``x``, ``y``, ``z`` are **rejected**: they name an axis without a direction and BIDS has no such
  value. Use an anatomical name (``LR``/``AP``/``SI``) for a physical statement.
* DICOM ``InPlanePhaseEncodingDirection`` (``ROW``/``COL``) is **not** BIDS and has no sign; rejected.

Restriction weights are one-hot in physical (x, y, z) order, length = number of spatial dimensions
(a 4-D image gives 3), the layout ``syntx.syn(restrict_transformation=...)`` consumes (1 = free axis).

Inference
---------
``allow_inference=True`` only calls a caller-supplied ``inference`` callable (``image -> physical
axis 0/1/2`` or a ``PhaseEncode``). The data-driven estimator (T1-anchored, in antsxdwi
``infer_pe_axis``) is not part of syntx; without a callable the call raises
:class:`PhaseEncodeUndecidable`. An inferred axis has ``sign=None`` (unsigned), is usable for a
restriction and is refused by :func:`relate_phase_encodes`.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

__all__ = [
    "PhaseEncode",
    "ParsedLabel",
    "PhaseEncodeError",
    "PhaseEncodeUndecidable",
    "PhaseEncodeConflict",
    "PhaseEncodeVocabularyError",
    "PhaseEncodeObliqueError",
    "PhaseEncodeRelationError",
    "PE_UNRESTRICTED",
    "parse_pe_label",
    "pe_entity_from_path",
    "resolve_phase_encode",
    "restrict_transformation",
    "restrict_for_image",
    "restrict_with_provenance",
    "relate_phase_encodes",
    "pe_to_dict",
    "pe_from_dict",
]

_LOG = logging.getLogger("syntx.phase_encode")

PE_UNRESTRICTED = "unrestricted"
"""Explicit opt-out policy: no restriction is applied (returns ``None``) and the choice is recorded."""

SCHEMA = "syntx.phase_encode/1"

# "XY" = acquired from X toward Y, in LPS (+x Left, +y Posterior, +z Superior).
_NAMED_VECTORS = {
    "LR": (-1.0, 0.0, 0.0),
    "RL": (1.0, 0.0, 0.0),
    "AP": (0.0, 1.0, 0.0),
    "PA": (0.0, -1.0, 0.0),
    "SI": (0.0, 0.0, -1.0),
    "IS": (0.0, 0.0, 1.0),
}
# (physical axis, sign of that component) -> anatomical name
_ANATOMICAL_BY_AXIS_SIGN = {(0, 1): "RL", (0, -1): "LR", (1, 1): "AP", (1, -1): "PA", (2, 1): "IS", (2, -1): "SI"}

_BIDS_RE = re.compile(r"^([ijk])([+-]?)$")
_NAMED_RE = re.compile(r"^(AP|PA|LR|RL|SI|IS)$", re.IGNORECASE)
_DICOM_WORDS = {"ROW", "COL", "COLUMN", "COLS", "ROWS"}
# BIDS entity values are alphanumeric only. The value therefore ends at the first non-alphanumeric
# character, so ``dir-LR-dwi`` (PPMI-style: entity ``dir-LR`` + suffix ``-dwi``) yields ``LR``.
_ENTITY_RE = re.compile(r"(?:^|[_-])dir-([A-Za-z0-9]+)(?=[-_.]|$)", re.IGNORECASE)


# --------------------------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------------------------
class PhaseEncodeError(ValueError):
    """Base class of every phase-encode failure (a ValueError, so legacy ``except ValueError`` still sees it)."""


class PhaseEncodeUndecidable(PhaseEncodeError):
    """Nothing states the phase-encode direction and inference was not allowed (or not possible)."""


class PhaseEncodeConflict(PhaseEncodeError):
    """Two sources, mapped through the header, name different directions; or a policy contradicts a source."""


class PhaseEncodeVocabularyError(PhaseEncodeError):
    """A label that is not in the accepted vocabulary (e.g. DICOM ``ROW``/``COL``, ``x``, ``''``, ``None``)."""


class PhaseEncodeObliqueError(PhaseEncodeError):
    """The phase-encode vector is not aligned with a voxel/physical axis within ``oblique_tol``."""


class PhaseEncodeRelationError(PhaseEncodeError):
    """Two phase encodes are neither opposed nor the same (orthogonal, ambiguous, or unsigned)."""


# --------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ParsedLabel:
    """Result of :func:`parse_pe_label`: a pure string parse, no image involved."""

    kind: str  # "bids" | "named"
    raw: str
    voxel_axis: int | None = None  # bids only
    sign: int | None = None  # bids only (+1/-1)
    name: str | None = None  # named only, upper case e.g. "AP"


@dataclass(frozen=True)
class PhaseEncode:
    """A resolved phase-encode direction on a given header.

    Attributes
    ----------
    voxel_axis : int
        0/1/2 = BIDS i/j/k.
    sign : int or None
        +1 / -1 along the voxel axis; ``None`` only for an inferred (unsigned) axis.
    physical_vector : tuple of 3 floats
        Unit vector in ITK/ANTs LPS physical coordinates (see module docstring). For an unsigned
        inferred axis it is the ``+`` voxel column (sign unknown).
    physical_axis : int
        argmax |physical_vector|: 0 = x (LR), 1 = y (AP), 2 = z (SI).
    alignment : float
        |cos| of the vector with its nearest axis (physical for BIDS labels; the voxel column for
        named labels). 1.0 is axis aligned; ``< oblique_tol`` triggers ``on_oblique``.
    label : str or None
        Canonical BIDS label ``'i','i-','j','j-','k','k-'``; ``None`` for an unsigned inferred axis.
    anatomical : str or None
        ``'AP','PA','LR','RL','SI','IS'`` when the vector is axis aligned within the tolerance.
    source : str
        ``'sidecar' | 'label' | 'entity' | 'inferred' | 'explicit'`` of the decisive (highest precedence) source.
    source_detail : str
        Path or raw string of the decisive source.
    evidence : dict
        JSON-safe record of every source examined and how it mapped.
    """

    voxel_axis: int
    sign: int | None
    physical_vector: tuple[float, float, float]
    physical_axis: int
    alignment: float
    label: str | None
    anatomical: str | None
    source: str
    source_detail: str
    evidence: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------------------------
def parse_pe_label(raw: Any) -> ParsedLabel:
    """Parse one label string into a :class:`ParsedLabel`. No image, no defaults.

    Accepted (whole-string match only, so ``map``/``path``/``pa`` inside words never match):
    BIDS ``i j k`` with optional ``-`` (``+`` tolerated), and ``AP PA LR RL SI IS`` (any case),
    each optionally prefixed ``dir-``. Everything else raises :class:`PhaseEncodeVocabularyError`.
    """
    if raw is None or not isinstance(raw, str) or not raw.strip():
        raise PhaseEncodeVocabularyError(
            f"phase-encode label {raw!r} is empty or not a string; expected BIDS i/j/k[-] or AP/PA/LR/RL/SI/IS"
        )
    key = raw.strip()
    if key.lower().startswith("dir-"):
        key = key[4:]
    m = _BIDS_RE.match(key)
    if m:
        return ParsedLabel("bids", raw, voxel_axis="ijk".index(m.group(1)), sign=-1 if m.group(2) == "-" else 1)
    if _NAMED_RE.match(key):
        return ParsedLabel("named", raw, name=key.upper())
    if key.upper() in _DICOM_WORDS:
        raise PhaseEncodeVocabularyError(
            f"{raw!r} is a DICOM InPlanePhaseEncodingDirection value (in-plane axis, no sign), not a BIDS "
            "PhaseEncodingDirection; it cannot be used as a phase-encode label. Supply BIDS i/j/k[-] "
            "(e.g. from the dcm2niix sidecar) or an anatomical name AP/PA/LR/RL/SI/IS."
        )
    if key.lower() in {"x", "y", "z", "x-", "y-", "z-"}:
        raise PhaseEncodeVocabularyError(
            f"{raw!r}: BIDS has no x/y/z phase-encode value and a bare axis has no direction. Use BIDS "
            "i/j/k[-] (voxel axis, mapped through the header) or an anatomical name LR/RL/AP/PA/SI/IS "
            "(physical LPS vector)."
        )
    raise PhaseEncodeVocabularyError(
        f"unrecognised phase-encode label {raw!r}; expected BIDS i/j/k with optional '-', or AP/PA/LR/RL/SI/IS"
    )


def pe_entity_from_path(path: Any) -> str | None:
    """The ``dir-<value>`` BIDS entity of a file *name* (basename only), or ``None`` when absent.

    Strict: the entity key ``dir`` must start the name or follow ``_``/``-``; the value is the run of
    alphanumerics after ``dir-`` (BIDS entity values are alphanumeric), ending at ``-``, ``_``, ``.`` or
    end of name. So ``dir-LR_dwi``, ``dir-LR-dwi`` and ``dir-LR.nii.gz`` all give ``LR``. A bare
    ``ap``/``pa`` anywhere else is not an entity. ``dir-UNK`` is "unknown", i.e. no statement, and is
    returned as ``None``. A value that is not a phase-encode word (``dir-run-01`` -> ``run``,
    ``dir-none``) is returned as is; :func:`resolve_phase_encode` then raises
    :class:`PhaseEncodeVocabularyError` for it (never a silent skip).
    """
    name = Path(str(path)).name
    m = _ENTITY_RE.search(name)
    if not m:
        return None
    val = m.group(1)
    return None if val.upper() == "UNK" else val


# --------------------------------------------------------------------------------------------
# header handling
# --------------------------------------------------------------------------------------------
def _spatial_direction(image: Any) -> tuple[np.ndarray, int]:
    """(3x3 direction in LPS, number of spatial dimensions) from an ANTs image or a direction matrix.

    A 4-D image yields its spatial 3x3 block; a 2-D image is embedded in an identity 3x3.
    """
    d = np.asarray(getattr(image, "direction", image), dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or d.shape[0] not in (2, 3, 4):
        raise PhaseEncodeError(f"cannot read a 2-/3-/4-D direction matrix from {type(image).__name__} (shape {d.shape})")
    n = d.shape[0]
    if n == 4:
        return d[:3, :3].copy(), 3
    if n == 2:
        d3 = np.eye(3)
        d3[:2, :2] = d
        return d3, 2
    return d.copy(), 3


def _fl(v: Any) -> list[float]:
    return [float(x) for x in v]


def _check_oblique(alignment: float, tol: float, on_oblique: str, what: str) -> None:
    if alignment >= tol:
        return
    msg = (
        f"{what}: phase-encode direction is oblique (alignment {alignment:.3f} < {tol}); an axis-aligned "
        "restriction is only an approximation"
    )
    if on_oblique == "raise":
        raise PhaseEncodeObliqueError(msg + ". Pass on_oblique='warn' to accept the approximation explicitly.")
    warnings.warn(msg, UserWarning, stacklevel=4)


def _build(
    parsed: ParsedLabel,
    d3: np.ndarray,
    ndim: int,
    *,
    source: str,
    detail: str,
    oblique_tol: float,
    on_oblique: str,
) -> PhaseEncode:
    if parsed.kind == "bids":
        axis, sign = int(parsed.voxel_axis), int(parsed.sign)
        if axis >= ndim:
            raise PhaseEncodeError(
                f"{parsed.raw!r} names voxel axis {'ijk'[axis]} but the image has {ndim} spatial dimensions"
            )
        col = d3[:, axis]
        v = sign * col / np.linalg.norm(col)
        paxis = int(np.argmax(np.abs(v)))
        alignment = float(abs(v[paxis]))
    else:
        v = np.asarray(_NAMED_VECTORS[parsed.name], dtype=float)
        paxis = int(np.argmax(np.abs(v)))
        if paxis >= ndim:
            raise PhaseEncodeError(f"{parsed.raw!r} is along physical axis {'xyz'[paxis]}, absent in a {ndim}-D image")
        cos = d3[:, :ndim].T @ v
        axis = int(np.argmax(np.abs(cos)))
        sign = 1 if cos[axis] > 0 else -1
        alignment = float(abs(cos[axis]))
    _check_oblique(alignment, oblique_tol, on_oblique, f"{detail} ({parsed.raw!r})")
    pvec = tuple(float(x) for x in v)
    anat = None
    if abs(pvec[paxis]) >= oblique_tol:
        anat = _ANATOMICAL_BY_AXIS_SIGN[(paxis, 1 if pvec[paxis] > 0 else -1)]
    return PhaseEncode(
        voxel_axis=axis,
        sign=sign,
        physical_vector=pvec,
        physical_axis=paxis,
        alignment=alignment,
        label="ijk"[axis] + ("-" if sign < 0 else ""),
        anatomical=anat,
        source=source,
        source_detail=detail,
        evidence={},
    )


def _read_sidecar(sidecar: Any) -> tuple[str, Any, str | None]:
    """(detail, raw PhaseEncodingDirection or None when the key is absent, error)."""
    if isinstance(sidecar, Mapping):
        return "sidecar dict", sidecar.get("PhaseEncodingDirection"), None
    p = Path(str(sidecar))
    if not p.is_file():
        raise PhaseEncodeError(f"sidecar {str(p)!r} does not exist (pass sidecar=None when there is none)")
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PhaseEncodeError(f"sidecar {str(p)!r} is not readable JSON: {exc}") from exc
    if not isinstance(meta, dict):
        raise PhaseEncodeError(f"sidecar {str(p)!r} is not a JSON object")
    return str(p), meta.get("PhaseEncodingDirection"), None


def resolve_phase_encode(
    image: Any,
    *,
    sidecar: Any = None,
    label: str | None = None,
    entity_path: Any = None,
    anatomical: str | None = None,
    allow_inference: bool = False,
    inference: Callable[[Any], Any] | None = None,
    oblique_tol: float = 0.85,
    on_oblique: str = "raise",
) -> PhaseEncode:
    """Resolve the phase-encode direction of ``image`` from every stated source, or raise.

    Parameters
    ----------
    image : ANTsImage or direction matrix
        Supplies the voxel -> LPS mapping (2-D, 3-D or 4-D; 4-D uses its spatial block).
    sidecar : str, Path or dict, optional
        BIDS JSON sidecar (path) or its parsed dict; ``PhaseEncodingDirection`` is read. A missing
        file or unreadable JSON raises; an absent key is "no statement from this source".
    label : str, optional
        A label in any accepted vocabulary (see :func:`parse_pe_label`).
    entity_path : str or Path, optional
        A file name carrying the ``dir-XX`` entity (basename only is read; ``dir-UNK`` and no entity
        are "no statement"). Pass the REAL file, never a staged/renamed link.
    anatomical : str, optional
        An anatomical name ``AP/PA/LR/RL/SI/IS`` stated by the caller (source ``'explicit'``).
    allow_inference, inference
        Explicit opt-in to a data-driven estimator when nothing states the direction; ``inference``
        must be a callable ``image -> physical axis (0/1/2) | PhaseEncode``. Without it: Undecidable.
    oblique_tol : float, default 0.85
        Minimum |cos| to an axis (0.85 is about 31.8 degrees, accommodating clinical AC-PC slice tilt).
    on_oblique : {'raise', 'warn'}, default 'raise'

    Precedence of the decisive source: sidecar > label > explicit > entity. All stated sources must
    agree (same voxel axis, sign and physical axis); otherwise :class:`PhaseEncodeConflict`.
    """
    if on_oblique not in ("raise", "warn"):
        raise ValueError("on_oblique must be 'raise' or 'warn'")
    d3, ndim = _spatial_direction(image)

    cands: list[PhaseEncode] = []
    skipped: list[dict] = []

    def add(parsed_raw: Any, source: str, detail: str) -> None:
        parsed = parse_pe_label(parsed_raw)
        pe = _build(parsed, d3, ndim, source=source, detail=detail, oblique_tol=oblique_tol, on_oblique=on_oblique)
        cands.append(pe)
        evidence_items.append({"source": source, "detail": detail, "raw": str(parsed_raw), "label": pe.label,
                               "anatomical": pe.anatomical, "physical_vector": _fl(pe.physical_vector)})

    evidence_items: list[dict] = []
    if sidecar is not None:
        det, raw, _ = _read_sidecar(sidecar)
        if raw is None:
            skipped.append({"source": "sidecar", "detail": det, "reason": "no PhaseEncodingDirection key"})
        else:
            add(raw, "sidecar", det)
    if label is not None:
        add(label, "label", f"label {label!r}")
    if anatomical is not None:
        add(anatomical, "explicit", f"anatomical {anatomical!r}")
    if entity_path is not None:
        ent = pe_entity_from_path(entity_path)
        if ent is None:
            skipped.append({"source": "entity", "detail": str(entity_path), "reason": "no dir- entity (or dir-UNK)"})
        else:
            add(ent, "entity", f"dir-{ent} in {Path(str(entity_path)).name}")

    if not cands:
        if allow_inference and inference is not None:
            got = inference(image)
            if isinstance(got, PhaseEncode):
                return got
            paxis = int(got)
            if not 0 <= paxis < ndim:
                raise PhaseEncodeError(f"inference returned physical axis {got!r}, outside a {ndim}-D image")
            cos = d3[:, :ndim].T @ np.eye(3)[paxis]
            vax = int(np.argmax(np.abs(cos)))
            alignment = float(abs(cos[vax]))
            _check_oblique(alignment, oblique_tol, on_oblique, f"inferred axis {'xyz'[paxis]}")
            col = d3[:, vax]
            return PhaseEncode(
                voxel_axis=vax, sign=None, physical_vector=tuple(float(x) for x in col / np.linalg.norm(col)),
                physical_axis=paxis, alignment=alignment, label=None, anatomical=None, source="inferred",
                source_detail=getattr(inference, "__name__", "inference"),
                evidence={"skipped": skipped, "sources": [], "unsigned": True},
            )
        why = ("allow_inference=True but no inference callable was supplied (the data-driven estimator "
               "lives in antsxdwi.infer_pe_axis; pass it as inference=)") if allow_inference else (
               "inference is not allowed (allow_inference=False)")
        raise PhaseEncodeUndecidable(
            "no source states the phase-encode direction (sidecar/label/entity/anatomical); "
            f"{why}. Skipped: {skipped}. To skip the restriction on purpose use pe_policy='unrestricted'."
        )

    order = {"sidecar": 0, "label": 1, "explicit": 2, "entity": 3}
    cands_sorted = sorted(cands, key=lambda p: order[p.source])
    top = cands_sorted[0]
    for other in cands_sorted[1:]:
        same = (other.voxel_axis, other.sign, other.physical_axis) == (top.voxel_axis, top.sign, top.physical_axis)
        if not same:
            raise PhaseEncodeConflict(
                "phase-encode sources disagree after mapping through the header: "
                + "; ".join(f"{e['source']} ({e['detail']}) -> {e['label']} / {e['anatomical']} / "
                            f"vector {[round(x, 3) for x in e['physical_vector']]}" for e in evidence_items)
            )
    from dataclasses import replace

    ev = {"sources": evidence_items, "skipped": skipped, "oblique_tol": float(oblique_tol), "ndim": ndim,
          "direction": [_fl(r) for r in d3]}
    return replace(top, evidence=ev)


# --------------------------------------------------------------------------------------------
# restriction
# --------------------------------------------------------------------------------------------
def _coerce_pe(pe: Any) -> Any:
    if isinstance(pe, PhaseEncode):
        return pe
    if isinstance(pe, Mapping):
        return pe_from_dict(pe)
    return pe


def restrict_transformation(pe: Any, *, dim: int = 3) -> tuple[float, ...] | None:
    """One-hot physical-axis weights (1 = free axis) for ``syntx.syn(restrict_transformation=...)``.

    ``pe`` is a :class:`PhaseEncode` (or its dict) or the explicit :data:`PE_UNRESTRICTED` policy,
    which returns ``None``. ``dim`` is the number of spatial dimensions (3 even for 4-D images).
    """
    if isinstance(pe, str) and pe == PE_UNRESTRICTED:
        _LOG.info("phase-encode restriction explicitly disabled (pe_policy='unrestricted')")
        return None
    pe = _coerce_pe(pe)
    if not isinstance(pe, PhaseEncode):
        raise TypeError(f"restrict_transformation needs a PhaseEncode, its dict or {PE_UNRESTRICTED!r}; got {pe!r}")
    if dim not in (2, 3) or pe.physical_axis >= dim:
        raise PhaseEncodeError(f"physical axis {'xyz'[pe.physical_axis]} does not exist in a {dim}-D restriction")
    w = [0.0] * dim
    w[pe.physical_axis] = 1.0
    return tuple(w)


_SOURCE_KEYS = ("sidecar", "label", "entity_path", "anatomical")


def _spatial_dim(image: Any) -> int:
    return _spatial_direction(image)[1]


def restrict_with_provenance(
    image: Any,
    *,
    pe: Any = None,
    pe_policy: str | None = None,
    dim: int | None = None,
    **resolve_kwargs: Any,
) -> tuple[tuple[float, ...] | None, dict]:
    """Restriction tuple plus the JSON-safe provenance record of how it was obtained.

    ``pe_policy`` is ``None``/``'require'`` (a direction must be resolved; the default behaviour is
    to raise when it cannot) or ``'unrestricted'`` (explicit opt-out; contradicting sources raise
    :class:`PhaseEncodeConflict`). Record the returned dict in the run's provenance so the applied
    tuple is never a constant.
    """
    if pe_policy not in (None, "require", PE_UNRESTRICTED):
        raise ValueError(f"pe_policy must be 'require' or {PE_UNRESTRICTED!r}, got {pe_policy!r}")
    n = _spatial_dim(image) if dim is None else int(dim)
    src_given = any(resolve_kwargs.get(k) is not None for k in _SOURCE_KEYS)
    given = src_given or pe is not None
    if pe_policy == PE_UNRESTRICTED:
        if given:
            raise PhaseEncodeConflict(
                "pe_policy='unrestricted' contradicts the phase-encode sources that were also supplied; "
                "give either the sources or the explicit opt-out"
            )
        return None, {"schema": SCHEMA, "policy": PE_UNRESTRICTED, "restrict_transformation": None, "phase_encode": None}
    if pe is not None:
        if src_given:
            raise PhaseEncodeError("give a resolved pe or source arguments, not both")
        pe = _coerce_pe(pe)
        if isinstance(pe, str) and pe == PE_UNRESTRICTED:
            return restrict_with_provenance(image, pe_policy=PE_UNRESTRICTED, dim=dim)
        if not isinstance(pe, PhaseEncode):
            raise TypeError(f"pe must be a PhaseEncode or its dict, got {type(pe).__name__}")
    else:
        pe = resolve_phase_encode(image, **resolve_kwargs)
    w = restrict_transformation(pe, dim=n)
    return w, {"schema": SCHEMA, "policy": "resolved", "restrict_transformation": list(w), "phase_encode": pe_to_dict(pe)}


def restrict_for_image(image: Any, *, pe: Any = None, pe_policy: str | None = None, dim: int | None = None,
                       **resolve_kwargs: Any) -> tuple[float, ...] | None:
    """``restrict_transformation`` weights for ``image`` (length = spatial dims; 4-D gives 3).

    Same arguments as :func:`restrict_with_provenance` (use that to also get the provenance record).
    Returns ``None`` only for the explicit ``pe_policy='unrestricted'``.
    """
    return restrict_with_provenance(image, pe=pe, pe_policy=pe_policy, dim=dim, **resolve_kwargs)[0]


# --------------------------------------------------------------------------------------------
# relation
# --------------------------------------------------------------------------------------------
def relate_phase_encodes(a: PhaseEncode, b: PhaseEncode, *, min_alignment: float = 0.95) -> str:
    """``'opposed'`` (anti-parallel physical vectors) or ``'same'`` (parallel); anything else raises.

    Orthogonal/ambiguous pairs and unsigned (inferred) encodes raise :class:`PhaseEncodeRelationError`.
    """
    for x in (a, b):
        if not isinstance(x, PhaseEncode):
            raise TypeError("relate_phase_encodes needs PhaseEncode objects")
        if x.sign is None:
            raise PhaseEncodeRelationError(
                f"phase encode from {x.source!r} is unsigned (inferred axis only); opposed/same cannot be decided"
            )
    dot = float(np.dot(a.physical_vector, b.physical_vector))
    if dot <= -min_alignment:
        return "opposed"
    if dot >= min_alignment:
        return "same"
    raise PhaseEncodeRelationError(
        f"phase encodes {a.label!r} ({a.anatomical}) and {b.label!r} ({b.anatomical}) are on different or "
        f"ambiguous physical axes (dot={dot:.3f}); they cannot be paired or merged"
    )


# --------------------------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------------------------
def pe_to_dict(pe: PhaseEncode) -> dict:
    """JSON-safe dict of ``pe`` (provenance, manifests, ``ModalityUnit.phase_encode``)."""
    if not isinstance(pe, PhaseEncode):
        raise TypeError("pe_to_dict needs a PhaseEncode")
    return {
        "schema": SCHEMA,
        "voxel_axis": int(pe.voxel_axis),
        "sign": None if pe.sign is None else int(pe.sign),
        "physical_vector": _fl(pe.physical_vector),
        "physical_axis": int(pe.physical_axis),
        "alignment": float(pe.alignment),
        "label": pe.label,
        "anatomical": pe.anatomical,
        "source": pe.source,
        "source_detail": pe.source_detail,
        "evidence": json.loads(json.dumps(pe.evidence, default=str)),
    }


def pe_from_dict(d: Mapping) -> PhaseEncode:
    """Inverse of :func:`pe_to_dict`; raises :class:`PhaseEncodeError` on a malformed record."""
    need = {"voxel_axis", "sign", "physical_vector", "physical_axis", "alignment", "label", "anatomical",
            "source", "source_detail", "evidence"}
    if not isinstance(d, Mapping) or need - set(d):
        raise PhaseEncodeError(f"not a phase-encode record (missing {sorted(need - set(d)) if isinstance(d, Mapping) else 'mapping'})")
    if d.get("schema", SCHEMA) != SCHEMA:
        raise PhaseEncodeError(f"unsupported phase-encode record schema {d.get('schema')!r}")
    vec = tuple(float(x) for x in d["physical_vector"])
    if len(vec) != 3 or d["voxel_axis"] not in (0, 1, 2) or d["physical_axis"] not in (0, 1, 2) or d["sign"] not in (1, -1, None):
        raise PhaseEncodeError(f"malformed phase-encode record {dict(d)!r}")
    return PhaseEncode(
        voxel_axis=int(d["voxel_axis"]), sign=None if d["sign"] is None else int(d["sign"]),
        physical_vector=vec, physical_axis=int(d["physical_axis"]), alignment=float(d["alignment"]),
        label=d["label"], anatomical=d["anatomical"], source=str(d["source"]),
        source_detail=str(d["source_detail"]), evidence=dict(d["evidence"]),
    )
