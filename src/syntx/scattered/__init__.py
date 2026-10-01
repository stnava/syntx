"""
syntx.scattered -- registration and resampling for scattered points (point clouds).

- ``projection``: points + per-point values -> regular grid, by normalised Gaussian kernel
  regression (Nadaraya-Watson) or ANTsTorch multi-level B-spline fitting; distance maps.
- ``mapping``: sample a grid field at points; move points through a displacement field.
- ``transport``: pull grid values back to warped points, push point values onto a grid,
  move point values to another point set.
- ``solver``: ``SyNScattered`` / ``syn_scattered``, a SyN-style symmetric registration that
  projects both point sets to grids and optimises two half-warps on that grid.
- ``bspline``: ANTsTorch B-spline helpers (landmark warp fit, B-spline velocity smoothing,
  ``bspline_syn_scattered``).

Coordinate conventions used throughout: ``coord_convention='xyz'`` means point component 0
runs along the LAST grid tensor axis (same as ``F.grid_sample``); ``'zyx'`` means component k
runs along tensor axis k. Displacement fields produced by the solver are in normalised
[-1, 1] grid units with (x, y, z) components.
"""

from .projection import (
    ProjectionConfig,
    project_scattered_to_grid,
    compute_distance_transform_to_grid,
    ScatteredProjector,
    differentiable_grid_projection,
    compute_adaptive_sigma,
)
from .mapping import (
    evaluate_field_at_scattered,
    warp_scattered_coordinates,
    ScatteredWarper,
)
from .transport import (
    pullback_grid_to_scattered,
    pushforward_scattered_to_grid,
    transport_scattered_to_scattered,
)
from .solver import (
    ScatteredRegistrationConfig,
    ScatteredRegistrationResult,
    SyNScattered,
    syn_scattered,
)
from .bspline import (
    has_antstorch,
    fit_bspline_landmark_warp,
    apply_bspline_fluid_regularizer,
    BSplineScatteredProjector,
    bspline_syn_scattered,
)

__all__ = [
    "ProjectionConfig",
    "project_scattered_to_grid",
    "compute_distance_transform_to_grid",
    "ScatteredProjector",
    "differentiable_grid_projection",
    "compute_adaptive_sigma",
    "evaluate_field_at_scattered",
    "warp_scattered_coordinates",
    "ScatteredWarper",
    "pullback_grid_to_scattered",
    "pushforward_scattered_to_grid",
    "transport_scattered_to_scattered",
    "ScatteredRegistrationConfig",
    "ScatteredRegistrationResult",
    "SyNScattered",
    "syn_scattered",
    "has_antstorch",
    "fit_bspline_landmark_warp",
    "apply_bspline_fluid_regularizer",
    "BSplineScatteredProjector",
    "bspline_syn_scattered",
]
