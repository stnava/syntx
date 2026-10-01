"""
syntx.reporting — legacy alias for report helpers that now live in ``syntx.viz``.

Re-exports, unchanged, ``build_engine_provenance``, ``create_registration_report``,
``render_input_pair_figure``, ``render_standard_4panel``, ``extract_2d_slice``,
``plot_deformation_grid``, ``plot_edge_overlay`` and the private helpers
``_parse_image_metadata`` / ``_compute_jacobian_stats`` (all from ``syntx.viz``; the report
helpers are defined in ``syntx.viz.reports``, the figures in ``syntx.viz.figures``). Kept so
that older ``from syntx.reporting import ...`` code keeps working; new code should import from
``syntx.viz``.
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
