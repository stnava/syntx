"""syntx.motion_presets -- modality-specific parameter sets for ``syntx.motion_correct*``.

One place for every preset, so a modality's motion-correction settings are never scattered across callers.
A preset is chosen explicitly by the caller (``motion_correct_grouped(..., preset="dwi")``,
``motion_correction(..., preset="bold")``) and recorded in the result; the generic defaults of ``motion_correction`` are not
changed by any preset here. Evidence for every value: docs/MOTION_DWI.md, docs/MOTION_BOLD.md.

``dwi`` (b0 group + diffusion group; ``motion_correct_grouped``)
    ``b0_mean`` and ``dwi_mean`` are the two channels; every b0 frame is registered to ``b0_mean`` and every diffusion frame to
    ``dwi_mean`` (full-precision batched backend, registration at ``resolution_cap_mm``); both means are then recomputed from
    all corrected frames and registered to each other once (``cross_num_bins`` histogram bins: on a still scan, true
    cross-registration 0, the bias fell from 0.29 mm at the default 32 bins to 0.19 mm at 128 bins, and from 0.61 mm for the
    mean of 10 low-motion volumes the older code used); the diffusion frames are resampled once from the originals through
    the composed chain with ``interpolator`` (a random 0.25 mm / 0.15 deg resample of a still scan lost 5% of white-matter FA
    with linear and 1% with b-spline). Registration at 2 mm gave the same error as native 1.5 mm (0.052 vs 0.048 mm).

``bold`` (one reference; ``motion_correction(preset="bold")``, also ASL)
    Full-precision batched registration, narrow seeds, ``medium`` schedule: on SRPBS travelling-subject BOLD it removes the
    constant ~0.15 mm FD offset of ``pytorch_batched_adaptive`` and halves the injected-motion error (0.083 -> 0.045 mm).
    Power FD (Power et al. 2012, scrubbing at 0.5 mm) stays the primary FD.
"""

DWI_MOTION_PRESET = {
    "backend": "pytorch_batched",
    "resolution_cap_mm": 2.0,
    "interpolator": "bSpline",
    "cross_num_bins": 128,
}

BOLD_MOTION_PRESET = {
    "backend": "pytorch_batched",
    "seed_mode": "narrow",
    "schedule_mode": "medium",
}

MOTION_PRESETS = {"dwi": DWI_MOTION_PRESET, "bold": BOLD_MOTION_PRESET}


def get_motion_preset(name: str) -> dict:
    """The preset ``name`` (a copy); unknown names raise (no silent fallback to generic defaults)."""
    if name not in MOTION_PRESETS:
        raise ValueError(f"unknown motion preset {name!r}; available: {sorted(MOTION_PRESETS)}")
    return dict(MOTION_PRESETS[name])
