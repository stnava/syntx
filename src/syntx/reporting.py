"""
Legacy Reporting Alias Module for syntx.

Re-exports visualization and reporting infrastructure from the dedicated syntx.viz sub-package.
"""

from .viz import (
    _parse_image_metadata,
    _compute_jacobian_stats,
    build_engine_provenance,
    create_registration_report,
    render_input_pair_figure,
    render_standard_4panel,
    extract_2d_slice,
    plot_deformation_grid,
    plot_edge_overlay,
)

__all__ = [
    "_parse_image_metadata",
    "_compute_jacobian_stats",
    "build_engine_provenance",
    "create_registration_report",
    "render_input_pair_figure",
    "render_standard_4panel",
    "extract_2d_slice",
    "plot_deformation_grid",
    "plot_edge_overlay",
]
