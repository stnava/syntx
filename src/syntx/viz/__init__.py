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
    extract_oriented_slice,
    plot_deformation_grid,
    plot_edge_overlay,
    plot_correspondence_vectors,
    plot_vector_field,
    compute_deformation_tensor_rgb,
    plot_deformation_tensor_rgb,
    get_dkt_colormap,
    dkt_colormap,
    render_input_pair_figure,
    render_standard_4panel,
    render_label_alignment_figure,
)
from .stats import (
    plot_label_overlap_stats,
    plot_jacobian_distribution,
)
from .reports import (
    _parse_image_metadata,
    _compute_jacobian_stats,
    build_engine_provenance,
    create_registration_report,
)
from .gallery import (
    create_visualization_gallery,
)

__all__ = [
    "extract_2d_slice",
    "extract_oriented_slice",
    "plot_deformation_grid",
    "plot_edge_overlay",
    "plot_correspondence_vectors",
    "plot_vector_field",
    "compute_deformation_tensor_rgb",
    "plot_deformation_tensor_rgb",
    "get_dkt_colormap",
    "dkt_colormap",
    "render_input_pair_figure",
    "render_standard_4panel",
    "render_label_alignment_figure",
    "plot_label_overlap_stats",
    "plot_jacobian_distribution",
    "create_registration_report",
    "build_engine_provenance",
    "create_visualization_gallery",
    "_parse_image_metadata",
    "_compute_jacobian_stats",
]
