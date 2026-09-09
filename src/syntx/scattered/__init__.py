"""
syntx.scattered — Generalized Scattered Data Diffeomorphic Registration
========================================================================

Provides differentiable scattered-to-grid projection (Nadaraya-Watson kernel regression),
Lagrangian coordinate mapping, feature transport, and symmetric diffeomorphic
point cloud registration.
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
]
