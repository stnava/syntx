"""Stage 0 Challenger Empirical Verification Suite.

Tests real-data probes from blast cohorts (FPA DWI, breacher BOLD, ExpArt ASL),
verifies absence of NaNs, exceptions, and physical space metadata corruption,
and stress-tests boundary invariants (BF1-BF7, C1-C3).
"""
import json
from pathlib import Path
import ants
import numpy as np
import pytest

import syntx
from syntx.motion_design import make_design, motion_correct_design, Design
from syntx.motion_reference import level_adjacency, offset_identifiable


def test_blast_cohort_fpa_dwi_probe():
    """Empirically test FPA/sub-BLAST070/ses-01 DWI dir-AP probe."""
    dpath = Path("/Users/stnava/data/blast_cohorts/BIDS/FPA/sub-BLAST070/ses-01/dwi")
    dwi_nii = dpath / "sub-BLAST070_ses-01_run-001_dir-AP_dwi.nii.gz"
    dwi_bval = dpath / "sub-BLAST070_ses-01_run-001_dir-AP_dwi.bval"
    dwi_bvec = dpath / "sub-BLAST070_ses-01_run-001_dir-AP_dwi.bvec"
    dwi_json = dpath / "sub-BLAST070_ses-01_run-001_dir-AP_dwi.json"

    assert dwi_nii.exists(), f"DWI file missing: {dwi_nii}"
    bvals = np.loadtxt(dwi_bval)[:25]
    bvecs = np.loadtxt(dwi_bvec)[:, :25].T
    sidecar = json.loads(dwi_json.read_text()) if dwi_json.exists() else None

    dwi_img = ants.image_read(str(dwi_nii))
    dwi_sub = ants.from_numpy(
        dwi_img.numpy()[..., :25],
        origin=dwi_img.origin,
        spacing=dwi_img.spacing,
        direction=dwi_img.direction,
    )

    pe_rec = syntx.resolve_phase_encode(dwi_sub, sidecar=sidecar, on_oblique="warn") if sidecar else None
    pe_axis = pe_rec.physical_vector if pe_rec else [0.0, 1.0, 0.0]

    design = make_design("shells", bvals=bvals, bvecs=bvecs, pe_axis=pe_axis)
    res = motion_correct_design(dwi_sub, design, engine="unified", device="cpu")

    # 1. Verify clean execution & expected keys
    assert "fd" in res
    assert "fwdtransforms" in res
    assert "motion_corrected" in res
    assert "translations_mm" in res
    assert "rotations_deg" in res

    # 2. Verify absence of NaNs and Infs
    fd = np.asarray(res["fd"], dtype=float)
    assert fd.shape == (25,)
    assert not np.isnan(fd).any(), "NaN found in DWI FD"
    assert not np.isinf(fd).any(), "Inf found in DWI FD"
    assert (fd >= 0.0).all(), "Negative FD value found"
    assert fd[0] == 0.0, "Initial FD frame must be 0.0"

    t_params = np.asarray(res["translations_mm"], dtype=float)
    assert not np.isnan(t_params).any(), "NaN found in translations_mm"
    assert not np.isinf(t_params).any(), "Inf found in translations_mm"

    r_params = np.asarray(res["rotations_deg"], dtype=float)
    assert not np.isnan(r_params).any(), "NaN found in rotations_deg"
    assert not np.isinf(r_params).any(), "Inf found in rotations_deg"

    # 3. Verify physical space metadata integrity (no corruption)
    corrected = res["motion_corrected"]
    assert np.allclose(corrected.spacing[:3], dwi_sub.spacing[:3], atol=1e-6), "Spacing mismatch"
    assert np.allclose(corrected.origin[:3], dwi_sub.origin[:3], atol=1e-6), "Origin mismatch"
    assert np.allclose(corrected.direction[:3, :3], dwi_sub.direction[:3, :3], atol=1e-6), "Direction mismatch"
    assert not np.isnan(corrected.numpy()).any(), "NaN found in corrected image"


def test_blast_cohort_breacher_bold_probe_default():
    """Empirically test breacher/sub-0102/ses-initial-day2 BOLD probe under default debias_pe_pseudomotion=False."""
    bpath = Path("/Users/stnava/data/blast_cohorts/BIDS/breacher/sub-0102/ses-initial-day2/func")
    bold_nii = bpath / "sub-0102_ses-initial-day2_task-rest_run-01_bold.nii.gz"
    bold_json = bpath / "sub-0102_ses-initial-day2_task-rest_run-01_bold.json"

    assert bold_nii.exists(), f"BOLD file missing: {bold_nii}"
    sidecar = json.loads(bold_json.read_text()) if bold_json.exists() else None

    bold_img = ants.image_read(str(bold_nii))
    n_frames = 30
    bold_sub = ants.from_numpy(
        bold_img.numpy()[..., :n_frames],
        origin=bold_img.origin,
        spacing=bold_img.spacing,
        direction=bold_img.direction,
    )

    pe_rec = syntx.resolve_phase_encode(bold_sub, sidecar=sidecar, on_oblique="warn")
    pe_axis = pe_rec.physical_vector

    design = make_design("none", n_frames=n_frames)
    res = motion_correct_design(
        bold_sub,
        design,
        engine="unified",
        device="cpu",
        pe_axis=pe_axis,
        debias_pe_pseudomotion=False,
    )

    # 1. Verify clean execution
    assert "fd" in res
    assert "motion_corrected" in res

    # 2. Verify absence of NaNs and Infs
    fd = np.asarray(res["fd"], dtype=float)
    assert fd.shape == (n_frames,)
    assert not np.isnan(fd).any(), "NaN found in BOLD FD"
    assert not np.isinf(fd).any(), "Inf found in BOLD FD"
    assert (fd >= 0.0).all(), "Negative FD value found"
    assert fd[0] == 0.0, "Initial FD frame must be 0.0"

    # 3. Verify physical space metadata integrity
    corrected = res["motion_corrected"]
    assert np.allclose(corrected.spacing[:3], bold_sub.spacing[:3], atol=1e-6), "Spacing mismatch"
    assert np.allclose(corrected.origin[:3], bold_sub.origin[:3], atol=1e-6), "Origin mismatch"
    assert np.allclose(corrected.direction[:3, :3], bold_sub.direction[:3, :3], atol=1e-6), "Direction mismatch"
    assert not np.isnan(corrected.numpy()).any(), "NaN found in corrected image"


def test_blast_cohort_breacher_bold_probe_optin_debias_success():
    """Empirically verify that opt-in debias_pe_pseudomotion=True in separate_bold_pe_pseudomotion runs cleanly without NameError."""
    bpath = Path("/Users/stnava/data/blast_cohorts/BIDS/breacher/sub-0102/ses-initial-day2/func")
    bold_nii = bpath / "sub-0102_ses-initial-day2_task-rest_run-01_bold.nii.gz"
    bold_json = bpath / "sub-0102_ses-initial-day2_task-rest_run-01_bold.json"

    sidecar = json.loads(bold_json.read_text())
    bold_img = ants.image_read(str(bold_nii))
    n_frames = 25
    bold_sub = ants.from_numpy(
        bold_img.numpy()[..., :n_frames],
        origin=bold_img.origin,
        spacing=bold_img.spacing,
        direction=bold_img.direction,
    )
    pe_rec = syntx.resolve_phase_encode(bold_sub, sidecar=sidecar, on_oblique="warn")
    design = make_design("none", n_frames=n_frames)

    # Calling with debias_pe_pseudomotion=True triggers separate_bold_pe_pseudomotion
    # which previously had an unresolved NameError ('sig_ref'); now fixed and returns successfully.
    res = motion_correct_design(
        bold_sub,
        design,
        engine="unified",
        device="cpu",
        pe_axis=pe_rec.physical_vector,
        debias_pe_pseudomotion=True,
        sidecar=sidecar,
    )
    assert res is not None
    assert "pe_pseudomotion_debias" in res
    assert res["pe_pseudomotion_debias"] is not None


def test_blast_cohort_expart_asl_probe():
    """Empirically test ExpArt/sub-EAS024/ses-01 ASL probe."""
    asl_path = Path("/Users/stnava/data/blast_cohorts/BIDS/ExpArt/sub-EAS024/ses-01/asl")
    asl_nii = asl_path / "sub-EAS024_ses-01_run-001_dir-PA_asl.nii.gz"
    asl_ctx = asl_path / "sub-EAS024_ses-01_run-001_dir-PA_aslcontext.tsv"
    asl_json = asl_path / "sub-EAS024_ses-01_run-001_dir-PA_asl.json"

    assert asl_nii.exists(), f"ASL file missing: {asl_nii}"
    assert asl_ctx.exists(), f"aslcontext missing: {asl_ctx}"

    lines = [line.strip() for line in asl_ctx.read_text().splitlines() if line.strip()]
    header = lines[0]
    volume_types = lines[1:]

    asl_img = ants.image_read(str(asl_nii))
    n_frames = asl_img.shape[-1]
    assert n_frames == len(volume_types), f"Frame count mismatch: {n_frames} vs {len(volume_types)}"

    sidecar = json.loads(asl_json.read_text()) if asl_json.exists() else None
    pe_rec = syntx.resolve_phase_encode(asl_img, sidecar=sidecar, on_oblique="warn")

    design = make_design("tag_control", volume_types=volume_types)
    assert design.detail["anchor_volume_type"] == "control", "BF1 anchor_volume_type check"

    res = motion_correct_design(
        asl_img,
        design,
        engine="unified",
        device="cpu",
        pe_axis=pe_rec.physical_vector,
    )

    # 1. Verify clean execution
    assert "fd" in res
    assert "motion_corrected" in res

    # 2. Verify absence of NaNs and Infs
    fd = np.asarray(res["fd"], dtype=float)
    assert fd.shape == (n_frames,)
    assert not np.isnan(fd).any(), "NaN found in ASL FD"
    assert not np.isinf(fd).any(), "Inf found in ASL FD"
    assert (fd >= 0.0).all(), "Negative FD value found"
    assert fd[0] == 0.0, "Initial FD frame must be 0.0"

    # 3. Verify physical space metadata integrity
    corrected = res["motion_corrected"]
    assert np.allclose(corrected.spacing[:3], asl_img.spacing[:3], atol=1e-6), "Spacing mismatch"
    assert np.allclose(corrected.origin[:3], asl_img.origin[:3], atol=1e-6), "Origin mismatch"
    assert np.allclose(corrected.direction[:3, :3], asl_img.direction[:3, :3], atol=1e-6), "Direction mismatch"
    assert not np.isnan(corrected.numpy()).any(), "NaN found in corrected image"


def test_bf7_connected_component_isolation_invariants():
    """Verify that level_adjacency and offset_identifiable properly isolate disconnected components."""
    # Disconnected sequence: 5 b0 frames, then 20 alternating shell 1 and shell 2 frames
    levels = np.array([0] * 5 + [1, 2] * 10)
    adj = level_adjacency(levels)
    info, comps = offset_identifiable(levels, min_mixed=4)

    assert comps == [[0], [1, 2]], f"Unexpected components: {comps}"
    assert not info[0]["identifiable"], "Component root 0 should not be offset-identifiable"
    assert not info[1]["identifiable"], "Component root 1 should not be offset-identifiable"
    assert info[2]["identifiable"], "Non-root level 2 in interleaved component [1, 2] must be identifiable"
    assert info[2]["root"] == 1, "Root of level 2 must be 1"


def test_bf4_design_validation():
    """Verify BF4 validation: non-contiguous levels or bad covariates raise ValueError."""
    # Non-contiguous levels: 0, 2
    with pytest.raises(ValueError, match="contiguous"):
        Design("invalid", levels=np.array([0, 2, 0, 2]))

    # Covariate length mismatch
    with pytest.raises(ValueError, match="match levels length"):
        Design("invalid", levels=np.array([0, 1, 0, 1]), encoding=np.array([1.0, -1.0, 1.0]))
