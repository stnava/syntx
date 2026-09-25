"""syntx.contract — shared ANTsX modality-plugin contract.

Plain dataclasses, JSON-serialisable via ``to_dict``. Originally defined independently
inside antsxdwi's own ``contract.py`` (whose docstring said "when a third plugin appears
these move to antsxcore"); centralised here instead once antsxfunctional became that third
plugin, since syntx is the one dependency every ANTsX-family modality package and antsxmm
itself already share, with no circular-import direction problem (syntx depends on none of
them). antsxmm, antsxdwi, antsxslowflow, and antsxfunctional all import these types rather
than each defining a structurally-compatible copy.

Rules (design doc §4, unchanged from the original antsxdwi contract): a plugin never raises
through ``run``; never chooses a device; never writes outside ``output_prefix``; declares a
noise-floor family per wide-CSV column; pre-fetches and hashes weights.
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


@dataclass
class ComputeContext:
    model_device: str = "cpu"  # antstorch / torch fitting
    reg_device: str = "cpu"  # syntx registration; chosen by benchmark-once-and-cache or user override
    reg_device_source: Literal["cache", "benchmark", "override", "default"] = "default"
    threads: int = 8
    seed: int = 1234
    determinism: Literal["fast", "strict"] = "fast"


@dataclass
class ProvenanceEntry:
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
        return dataclasses.asdict(self)

    @staticmethod
    def failed(modality: str, error: str, provenance: list[ProvenanceEntry] | None = None) -> ModalityResult:
        return ModalityResult(status="failed", modality=modality, error=error, provenance=provenance or [])
