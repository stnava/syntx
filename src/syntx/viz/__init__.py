"""
syntx.viz — Visualization Suite & Verification Infrastructure
=============================================================

Sub-package providing standard figure generators and interactive HTML report tools:
- render_input_pair_figure (Figure 1: Fixed Top / Moving Bottom for 3D, Side-by-Side for 2D)
- render_standard_4panel (Figure 2: Mesh Grid, Jacobian Map, Inverse Error Map, Edge Overlap)
- plot_deformation_grid & plot_edge_overlay
- create_registration_report
- build_engine_provenance
"""

from .figures import (
    extract_2d_slice,
    plot_deformation_grid,
    plot_edge_overlay,
    render_input_pair_figure,
    render_standard_4panel,
)
from .reports import (
    _parse_image_metadata,
    _compute_jacobian_stats,
    build_engine_provenance,
    create_registration_report,
)

__all__ = [
    "extract_2d_slice",
    "plot_deformation_grid",
    "plot_edge_overlay",
    "render_input_pair_figure",
    "render_standard_4panel",
    "create_registration_report",
    "build_engine_provenance",
    "_parse_image_metadata",
    "_compute_jacobian_stats",
]
