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
    injected motion); ``interpolator`` the resampling kernel (a random 0.25 mm / 0.15 deg resample of a still scan
    lost 5% of white-matter FA with linear and 1% with b-spline).
"""

DWI_MOTION_PRESET = {
    "backend": "pytorch_batched",
    "kappa": 6.0,
    "ridge": 0.05,
    "n_iter": 1,
    "shell_tol": 150.0,
    "interpolator": "bSpline",
}

MOTION_PRESETS = {"dwi": DWI_MOTION_PRESET}


def get_motion_preset(name: str) -> dict:
    """The preset ``name`` (a copy); unknown names raise (no silent fallback to generic defaults)."""
    if name not in MOTION_PRESETS:
        raise ValueError(f"unknown motion preset {name!r}; available: {sorted(MOTION_PRESETS)}")
    return dict(MOTION_PRESETS[name])
