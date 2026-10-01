"""syntx.contract — shared ANTsX modality-plugin contract.

Plain dataclasses describing what a modality plugin (DWI, rsfMRI, ...) receives and returns:

- ``ModalityUnit``: one input acquisition (paths + BIDS-style identifiers).
- ``SessionContext``: per-session inputs shared by plugins (output prefix, T1 products, flags).
- ``ComputeContext``: device / thread / seed choices made by the caller, not the plugin.
- ``ProvenanceEntry``: one timed processing step.
- ``ModalityResult``: what a plugin returns; ``to_dict`` gives a JSON-serialisable dict.

The types were first written inside antsxdwi's own ``contract.py`` and copied here so that
packages that already depend on syntx can share one definition (syntx imports none of them).
antsxmm, antsxfunctional, antsxstructural and antsxdwi all use these types (``antsxdwi.contract``
re-exports them; its ``SessionContext`` subclass only changes the fixels / tracking defaults
for standalone antsxdwi runs).

Contract rules (design doc §4; conventions, not enforced by this module): a plugin never raises
through ``run``; never chooses a device; never writes outside ``output_prefix``; declares a
noise-floor family per wide-CSV column; pre-fetches and hashes weights.

Module constants: ``CONTRACT_VERSION`` ("0.1"); ``Status``, ``Family`` and ``Engine`` are
``typing.Literal`` aliases (type hints only; values are not validated at run time).
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any, Literal

CONTRACT_VERSION = "0.1"

Status = Literal["success", "failed", "skipped"]
Family = Literal["bitwise", "registration", "deep_model", "derived"]
Engine = Literal["syntx", "antstorch", "torch", "numpy", "ants", "dipy"]


@dataclass
class ModalityUnit:
    """One acquisition to process.

    Attributes
    ----------
    project, subject, session, modality, run : str
        Identifiers (BIDS-style) of the acquisition.
    input_paths : list of str
        Image files (e.g. one or more NIfTI series).
    sidecar_paths : list of str, default []
        JSON sidecars, usually parallel to ``input_paths``.
    phase_encoding : list of str, default []
        Phase-encoding direction per input, e.g. ``["i", "i-"]``, parallel to ``input_paths``.
    bval_paths, bvec_paths : list of str, default []
        Diffusion gradient files (DWI only).
    """

    project: str
    subject: str
    session: str
    modality: str
    run: str
    input_paths: list[str]
    sidecar_paths: list[str] = field(default_factory=list)
    phase_encoding: list[str] = field(default_factory=list)  # e.g. ["i", "i-"] parallel to input_paths
    bval_paths: list[str] = field(default_factory=list)
    bvec_paths: list[str] = field(default_factory=list)


@dataclass
class SessionContext:
    """Per-session inputs and options handed to a plugin's ``run``.

    Attributes
    ----------
    output_prefix : str
        Path prefix; a plugin writes only below it.
    separator : str, default "+"
        Separator used when composing output names / column names.
    t1_brain : ants.ANTsImage or None
        Brain-extracted T1 image.
    t1_hierarchical : dict or None
        Outputs of the T1 hierarchical (structural) pipeline.
    brain_mask : ants.ANTsImage or None
        Brain mask in T1 space; the plugin projects it to modality space itself.
    resume : bool, default True
        Reuse existing outputs where the plugin supports it.
    bids_root : str or None
        BIDS dataset root.
    antsxbids_version : str or None
        Version string of the antsxbids layout in use.
    processing_resolution : float or None
        Isotropic target spacing in mm; None keeps native spacing.
    dewarp_params : DewarpParams, dict or None
        Distortion-correction parameters (interpreted by the plugin).
    fixels, tracking : bool, default False
        antsxdwi options: fixel analysis (stages F1/F2/F4) and streamline tractography (F5a).
    denoise : bool, default False
        antsxdwi option: SANLM denoising of the raw 4-D series before motion correction.
    """

    output_prefix: str
    separator: str = "+"
    t1_brain: Any | None = None  # ants.ANTsImage
    t1_hierarchical: dict[str, Any] | None = None
    brain_mask: Any | None = None  # ants.ANTsImage in T1 space (projected to modality space inside run)
    resume: bool = True
    bids_root: str | None = None
    antsxbids_version: str | None = None
    processing_resolution: float | None = None  # isotropic target mm; None = native
    dewarp_params: Any | None = None  # DewarpParams or dict for distortion correction parameters
    fixels: bool = False  # antsxdwi Stage F1/F2/F4 fixel-based analysis
    tracking: bool = False  # antsxdwi Stage F5a physical streamline tractography
    denoise: bool = False  # antsxdwi: SANLM denoise of the raw 4-D series before motion correction


@dataclass
class ComputeContext:
    """Compute resources chosen by the caller (plugins must not choose a device).

    Attributes
    ----------
    model_device : str, default "cpu"
        Device for antstorch / torch model inference or fitting.
    reg_device : str, default "cpu"
        Device for syntx registration.
    reg_device_source : {"cache", "benchmark", "override", "default"}, default "default"
        How ``reg_device`` was chosen (recorded only).
    threads : int, default 8
        CPU thread budget.
    seed : int, default 1234
        Random seed.
    determinism : {"fast", "strict"}, default "fast"
        Requested determinism level (meaning is up to the consumer).
    """

    model_device: str = "cpu"  # antstorch / torch fitting
    reg_device: str = "cpu"  # syntx registration; chosen by benchmark-once-and-cache or user override
    reg_device_source: Literal["cache", "benchmark", "override", "default"] = "default"
    threads: int = 8
    seed: int = 1234
    determinism: Literal["fast", "strict"] = "fast"


@dataclass
class ProvenanceEntry:
    """Record of one processing step.

    Attributes
    ----------
    step : str
        Step name.
    engine : str
        One of ``Engine`` ("syntx", "antstorch", "torch", "numpy", "ants", "dipy"); not
        validated.
    device : str
        Device the step ran on.
    seconds : float
        Wall-clock time of the step.
    caller : str
        Function / module that ran the step.
    weights_sha256 : str or None
        SHA-256 of model weights used, if any.
    transform_chain : list of dict or None
        Transforms applied, as string-valued dicts.
    extra : dict, default {}
        Free-form additional fields.
    timestamp : str
        UTC time at construction, ``"%Y-%m-%dT%H:%M:%SZ"``.
    """

    step: str
    engine: Engine
    device: str
    seconds: float
    caller: str
    weights_sha256: str | None = None
    transform_chain: list[dict[str, str]] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


@dataclass
class ModalityResult:
    """What a plugin's ``run`` returns.

    Attributes
    ----------
    status : {"success", "failed", "skipped"}
    modality : str
    error : str or None
        Error message when ``status == "failed"``.
    wide_row : dict, default {}
        Column -> value for the wide (one row per session) CSV.
    artifacts : dict of str, default {}
        Name -> output file path.
    provenance : list of ProvenanceEntry, default []
    figures : dict of str, default {}
        Name -> figure file path.
    noise_floor_family : dict, default {}
        ``wide_row`` column -> ``Family`` ("bitwise", "registration", "deep_model", "derived").
    contract_version : str, default ``CONTRACT_VERSION``
    """

    status: Status
    modality: str
    error: str | None = None
    wide_row: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    provenance: list[ProvenanceEntry] = field(default_factory=list)
    figures: dict[str, str] = field(default_factory=dict)
    noise_floor_family: dict[str, Family] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return ``dataclasses.asdict(self)`` (nested ``ProvenanceEntry`` become dicts)."""
        return dataclasses.asdict(self)

    @staticmethod
    def failed(modality: str, error: str, provenance: list[ProvenanceEntry] | None = None) -> ModalityResult:
        """Build a ``status="failed"`` result with ``error`` set and other fields empty."""
        return ModalityResult(status="failed", modality=modality, error=error, provenance=provenance or [])
