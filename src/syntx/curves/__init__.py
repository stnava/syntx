"""syntx.curves — Continuous elastic curve matching in N dimensions.

Provides symmetric scale-space continuous curve matching (Avants & Gee 2003) for:
- Open curves: Centerlines, sulcal fundi, fibers, and open polylines.
- Closed curves: Cortical boundary loops, anatomical contours, and closed ROIs.
- Arbitrary dimension d >= 1: 2-D planar contours, 3-D space curves, and higher-D trajectories.
- Evaluation metrics: Elastic RMS distance, Elastic gain, Membrane and bending strain,
  SO(d) rigid pose, and bidirectional inverse consistency.
"""

from syntx.curves.geometry import (
    as_curve_array,
    count_curvature_extrema,
    curve_arc_fraction,
    curve_curvature,
    curve_length,
    curve_segment_lengths,
    curve_tangents,
    curvature_scale_space,
    resample_curve,
    select_matching_scale,
    smooth_curve,
    tangent_scale_space,
)
from syntx.curves.matching import (
    CurveMatch,
    dp_match_grid,
    enforce_bounded_increments,
    enforce_monotone,
    kabsch_rigid,
    match_curves,
    path_to_map_subsample,
    project_monotone_l2,
)
from syntx.curves.batched import (
    can_batch_open_match,
    match_open_curves_batch,
    open_tangents_and_curvature,
    resample_open_curves_batch,
    smooth_open_curves_batch,
)
from syntx.curves.evaluation import correspondence_error
from syntx.curves.viz import (
    compute_dec_colors,
    fig_to_base64,
    plot_curve_match,
)
from syntx.curves.report import generate_curve_matching_html_report

__all__ = [
    # Geometry
    "as_curve_array",
    "curve_segment_lengths",
    "curve_length",
    "curve_arc_fraction",
    "resample_curve",
    "smooth_curve",
    "curve_tangents",
    "curve_curvature",
    "curvature_scale_space",
    "tangent_scale_space",
    "count_curvature_extrema",
    "select_matching_scale",
    # Matching
    "CurveMatch",
    "kabsch_rigid",
    "dp_match_grid",
    "enforce_bounded_increments",
    "enforce_monotone",
    "path_to_map_subsample",
    "project_monotone_l2",
    "match_curves",
    "can_batch_open_match",
    "match_open_curves_batch",
    "open_tangents_and_curvature",
    "resample_open_curves_batch",
    "smooth_open_curves_batch",
    # Evaluation
    "correspondence_error",
    # Viz & Reporting
    "compute_dec_colors",
    "fig_to_base64",
    "plot_curve_match",
    "generate_curve_matching_html_report",
]
