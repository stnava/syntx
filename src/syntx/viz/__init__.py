"""
syntx.viz — figures and HTML reports for registration and imaging QC
====================================================================

- ``core``: ``AnatomicalVisualizer`` (oriented slice extraction), ``corner_watermark``.
- ``figures``: matplotlib figures -- input pair, 4-panel registration QC, deformation grid /
  vectors / tensor RGB, label alignment / overlay, checkerboard, TVF keyframes, correlation
  matrix, carpet plot, motion parameters.
- ``colormaps``: label colours (``get_dkt_colormap``, ``build_dkt_label_palette``, ...).
- ``stats``: Dice and det(J) distribution plots.
- ``reports``: ``create_registration_report``, benchmark reports, ``build_engine_provenance``.
- ``gallery``: ``create_visualization_gallery`` (all figures on one HTML page).
- ``qc_sections`` / ``modality_report``: graded QC sections and a generic single-file HTML
  report (``write_modality_report``).
"""

from .core import (
    corner_watermark,
    AnatomicalSlice,
    AnatomicalVisualizer,
)
from .figures import (
    extract_2d_slice,
    extract_oriented_slice,
    plot_deformation_grid,
    plot_edge_overlay,
    plot_correspondence_vectors,
    plot_vector_field,
    compute_deformation_tensor_rgb,
    plot_deformation_tensor_rgb,
    render_input_pair_figure,
    render_standard_4panel,
    render_label_alignment_figure,
    render_label_overlay_figure,
    render_checkerboard_figure,
    render_correlation_matrix_figure,
    render_carpet_plot_figure,
    render_motion_parameters_figure,
    plot_time_varying_velocity_grid,
)
from .colormaps import (
    get_dkt_colormap,
    get_dkt_label_color_dict,
    build_dkt_label_palette,
    dkt_colormap,
)
from .stats import (
    plot_label_overlap_stats,
    plot_jacobian_distribution,
    plot_loss_convergence,
)
from .reports import (
    _parse_image_metadata,
    _compute_jacobian_stats,
    build_engine_provenance,
    create_registration_report,
    create_population_benchmark_report,
    create_affine_benchmark_report,
)
from .gallery import (
    create_visualization_gallery,
)
from .qc_sections import (
    grade_dice_overlap,
    grade_edge_dice_tol1,
    grade_framewise_displacement_mean,
    registration_qc_section,
    motion_qc_section,
    segmentation_qc_section,
    ai_model_qc_section,
)
from .modality_report import (
    kpi_card,
    equation_figure,
    equations_figure,
    provenance_table_rows,
    write_modality_report,
)

__all__ = [
    "corner_watermark",
    "AnatomicalSlice",
    "AnatomicalVisualizer",
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
    "render_label_overlay_figure",
    "render_checkerboard_figure",
    "render_correlation_matrix_figure",
    "render_carpet_plot_figure",
    "render_motion_parameters_figure",
    "plot_label_overlap_stats",
    "plot_jacobian_distribution",
    "plot_loss_convergence",
    "create_registration_report",
    "create_population_benchmark_report",
    "create_affine_benchmark_report",
    "build_engine_provenance",
    "create_visualization_gallery",
    "_parse_image_metadata",
    "_compute_jacobian_stats",
    "grade_dice_overlap",
    "grade_edge_dice_tol1",
    "grade_framewise_displacement_mean",
    "registration_qc_section",
    "motion_qc_section",
    "segmentation_qc_section",
    "ai_model_qc_section",
    "kpi_card",
    "equation_figure",
    "equations_figure",
    "provenance_table_rows",
    "write_modality_report",
]
