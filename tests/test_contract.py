"""Tests for syntx.contract -- the shared ANTsX modality-plugin dataclasses."""

from syntx.contract import (
    CONTRACT_VERSION,
    ComputeContext,
    ModalityResult,
    ModalityUnit,
    ProvenanceEntry,
    SessionContext,
)


def test_modality_unit_minimal_construction():
    unit = ModalityUnit(
        project="TEST",
        subject="sub-01",
        session="ses-01",
        modality="DTI",
        run="run-01",
        input_paths=["/tmp/a.nii.gz"],
    )
    assert unit.sidecar_paths == []
    assert unit.phase_encoding == []


def test_session_context_defaults():
    ctx = SessionContext(output_prefix="/tmp/out")
    assert ctx.separator == "+"
    assert ctx.resume is True
    assert ctx.fixels is False


def test_compute_context_defaults():
    ctx = ComputeContext()
    assert ctx.model_device == "cpu"
    assert ctx.reg_device_source == "default"
    assert ctx.seed == 1234


def test_provenance_entry_has_timestamp():
    prov = ProvenanceEntry(step="fit", engine="torch", device="cpu", seconds=1.0, caller="test")
    assert prov.timestamp.endswith("Z")
    assert prov.extra == {}


def test_modality_result_to_dict_round_trips_nested_provenance():
    prov = ProvenanceEntry(step="fit", engine="torch", device="cpu", seconds=1.0, caller="test")
    result = ModalityResult(status="success", modality="DTI", wide_row={"FA_mean": 0.4}, provenance=[prov])
    d = result.to_dict()
    assert d["status"] == "success"
    assert d["wide_row"]["FA_mean"] == 0.4
    assert d["provenance"][0]["step"] == "fit"
    assert d["contract_version"] == CONTRACT_VERSION


def test_modality_result_failed_helper():
    result = ModalityResult.failed("PET", "convergence error")
    assert result.status == "failed"
    assert result.error == "convergence error"
    assert result.provenance == []
