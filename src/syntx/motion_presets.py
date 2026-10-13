"""syntx.motion_presets -- modality-specific parameter sets for ``syntx.motion_correct*``.

One place for every preset, so a modality's motion-correction settings are never scattered across callers.
A preset is chosen explicitly by the caller (``motion_correct_grouped(..., preset="dwi")``) and recorded in the
result; the generic defaults of ``motion_correction`` are not changed by any preset here.

``dwi``
    Measured on already-corrected, still travelling-subject DWI (true motion about zero; docs/MOTION_DWI.md):
    the pipeline default (``pytorch_batched_adaptive`` against the mean of all diffusion volumes) reported
    1.3-1.7 mm of spurious motion per DW volume. Causes: (1) the adaptive backend registers at 8 then 4 mm
    resolution caps, (2) one isotropic mean is a poor target for volumes whose contrast depends on gradient
    direction. The preset therefore uses the full-precision batched backend and registers every DW volume to
    the leave-one-out Gaussian-process prediction of its own direction from the other volumes of its shell
    (``motion_reference.dwi_direction_targets``). ``kappa`` is the concentration of the antipodally symmetric
    Watson-type kernel exp(kappa * (cos^2 - 1)); ``ridge`` the kernel nugget; ``n_iter`` the number of
    prediction/registration rounds after the first mean-target pass (a second round measured no gain in recovery of
    injected motion); ``loo_seed_mode`` / ``loo_schedule_mode`` the registrar settings of the per-direction pass (narrow seeds with the
    full schedule agree with wide/full to 0.014 mm on a high-motion scan at 0.7x the time; the fast schedule differs by
    up to 1.4 mm / 2.5 deg and is not used); ``resolution_cap_mm`` the finest resolution registrations run at (2 mm gave the same error as native
    1.5 mm on a still scan: 0.052 vs 0.048 mm; 2.5 and 3 mm gave 0.077 and 0.069 mm); ``interpolator`` the resampling kernel (a random 0.25 mm / 0.15 deg resample of a still scan
    lost 5% of white-matter FA with linear and 1% with b-spline).
"""

DWI_MOTION_PRESET = {
    "backend": "pytorch_batched",
    "kappa": 6.0,
    "ridge": 0.05,
    "n_iter": 1,
    "shell_tol": 150.0,
    "interpolator": "bSpline",
    "resolution_cap_mm": 2.0,
    "loo_seed_mode": "narrow",
    "loo_schedule_mode": "full",
}

MOTION_PRESETS = {"dwi": DWI_MOTION_PRESET}


def get_motion_preset(name: str) -> dict:
    """The preset ``name`` (a copy); unknown names raise (no silent fallback to generic defaults)."""
    if name not in MOTION_PRESETS:
        raise ValueError(f"unknown motion preset {name!r}; available: {sorted(MOTION_PRESETS)}")
    return dict(MOTION_PRESETS[name])
