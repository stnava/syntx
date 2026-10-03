"""
syntx — diffeomorphic image registration in PyTorch (with JAX ports of some models)
====================================================================================

2-D / 3-D registration of ``ants.ANTsImage`` pairs. Results are returned in the ANTsPy
``ants.registration`` shape (``warpedmovout``, ``fwdtransforms``, ``invtransforms``, ...), so
transforms can be applied with ``ants.apply_transforms``.

Registration entry points
-------------------------
syntx.syn / syntx.registration
    Symmetric normalization (SyN, model ``SyNTo``): forward and inverse fields meeting at a
    midpoint, optional deep-feature loss.
syntx.tvf / syntx.tvf_registration
    Time-varying velocity field (``TVFModel``), integrated with Euler steps over t in [0, 1].
syntx.syngs / syntx.syngs_registration
    "Geodesic shooting" (``GeodesicShootingModel``): an initial velocity integrated with Euler
    steps; in the default mode this is the flow of a stationary velocity (EPDiff momentum
    transport is not implemented, see ``syntx.syngs``).
syntx.greedy / syntx.greedy_registration
    One-directional greedy SyN (no inverse unless requested).
syntx.auto_reg
    Diagnoses the pair (``diagnose_pair`` / ``synthesize_policy``), picks a method and runs it.
syntx.robust_affine
    Initial rigid / affine alignment used by the deformable methods.

Other public helpers include ``motion_correction``, ``build_template``, ``image_compare``,
``liouville_determinant``, the deformation metrics, landmark detection / matching
(``syntx.landmarks``), scattered-data registration (``syntx.scattered``), figures and reports
(``syntx.viz``) and the modality-plugin dataclasses (``syntx.contract``).

Side effect of import: sets the environment variable ``PYTORCH_MPS_HIGH_WATERMARK_RATIO`` to
"0.0" (no MPS allocator limit) unless it is already set.

Quick start
-----------
>>> import syntx
>>> reg = syntx.syn(fixed=fi, moving=mi)
>>> warped = reg['warpedmovout']
>>> transforms = reg['fwdtransforms']
"""

import os

# Force MPS allocator to be unconstrained for large 3D operations
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")  # keep a user-set value

from .syn import (
    registration,
    SyNTo,
    calculate_inverse_identity_error,
    auto_reg,
    plot_deformation_grid,
    plot_edge_overlay,
    render_standard_4panel,
)
from .core import normalize_tensor, normalize_image
from .syn_jax import SyNTo as SyNToJax
from .transform import SyNToTransform
from .features import (
    FeatureSpaceLoss,
    VGG19Extractor,
    DINOv2Extractor,
    ResNet10Extractor,
    SwinUNETRExtractor,
)
from .image_compare import image_compare, correlation
from .generators import CrossProductGenerator, benchmark_data
from .tvf import TVFModel, tvf_registration
from .tvf_jax import TVFModelJAX
from .syngs import (
    GeodesicShootingModel,
    syngs_registration,
    integrate_momentum,
    shoot_geodesic,
    momentum_to_deformation,
)
from .syngs_jax import (
    GeodesicShootingModelJAX,
    integrate_momentum_jax,
    shoot_geodesic_jax,
    momentum_to_deformation_jax,
)
from . import spatial
from .spatial import deformation_gradient
from . import contract
from . import tabulate
from . import viz
from .viz import (
    render_input_pair_figure,
    render_standard_4panel,
    plot_deformation_grid,
    plot_edge_overlay,
    create_registration_report,
    create_population_benchmark_report,
    extract_2d_slice,
)
from .reporting import build_engine_provenance
from .robust_affine import robust_affine, robust_center_of_mass, compute_center_of_mass, robust_cross_modal_rigid
from . import benchmark
from .benchmark import (
    run_benchmark_suite,
    high_level_benchmark_run,
    evaluate_affine_benchmark,
)
from .pyramid import build_image_pyramid
from .deformation_metrics import (
    compute_harmonic_energy,
    compute_bending_energy,
    compute_jacobian_metrics,
    compute_bidirectional_dice,
)
from .qc import RegistrationQCReport, evaluate_registration_qc
from .benchmark.metrics import compute_pair_metrics
from . import scattered
from . import landmarks
from .landmarks import (
    detect_blobs_log,
    detect_blobs_dog,
    detect_sift2d,
    detect_sift3d,
    compute_mind,
    extract_mind_at_points,
    match_landmarks,
    ransac_filter,
    compute_tre,
    sinkhorn_matching,
    weighted_procrustes,
    sampled_optimal_transport_affine,
    score_rotation_candidates_sampled,
)
from .scattered import (
    project_scattered_to_grid,
    ScatteredProjector,
    ProjectionConfig,
    differentiable_grid_projection,
    evaluate_field_at_scattered,
    warp_scattered_coordinates,
    ScatteredWarper,
    pullback_grid_to_scattered,
    pushforward_scattered_to_grid,
    transport_scattered_to_scattered,
    ScatteredRegistrationConfig,
    ScatteredRegistrationResult,
    SyNScattered,
    syn_scattered,
)
from .greedy import (
    greedy,
    greedy_registration,
    GreedyRegistrationModel,
)

from .surface import (
    compute_surface_classes,
    generate_surface_channels,
    extract_sulcal_probability_map,
)
from .diagnose import (
    ImageDiagnosis,
    PairDiagnosis,
    diagnose_image,
    diagnose_pair,
)
from .policy import (
    RegistrationPolicy,
    synthesize_policy,
    auto_policy_for_images,
)
from .classifier import (
    DiagnosticClassifier3D,
    predict_diagnosis_deep,
)
from . import data
from . import perf_tracking
from .perf_tracking import detect_regressions, load_history, record_run, record_run_and_check
from .template import build_template
from .imaging_utils import cap_resolution_for_registration, clip_cervical_spine_fov
from .image_utils import reflect_image
from .motion import (
    motion_correction,
    MotionParameters,
    TransformCollection,
    MotionCorrectionResult,
    calculate_framewise_displacement,
    calculate_dvars,
)
from .motion_batched import (
    batched_rigid_register_pass,
    batched_rigid_register_pass_temporal,
    batched_group_bias_register_pass,
)
from . import dewarp
from .dewarp import (
    dewarp_to_anatomical,
    rigid_align_anatomy_to_reference,
    rigid_align_masked_anatomy_to_motion_reference,
    syn_only_reference_to_anatomy_rigid,
    syn_only_motion_reference_to_t1_rigid,
    compose_native_frame_to_anatomy_rigid_transform,
    compose_native_frame_to_t1_rigid_transform,
    compose_anatomy_to_t1_rigid_transform,
    compose_anatomy_to_native_frame_transform,
    compose_native_to_anatomy_transform,
    build_motion_reference,
    apply_transform_chain,
    invert_transformlist,
    compose_transformlists,
    normalize_transformlist,
    DewarpResult,
    AnatomyRigidResult,
    CanonicalSynResult,
    TransformApplication,
)


# Expose syn, registration, auto_reg, and tvf
syn = registration
tvf = tvf_registration
from .liouville import liouville_determinant, determinant_summary
syngs = syngs_registration
affine_benchmark = evaluate_affine_benchmark

__version__ = "6.0.1"


__all__ = [
    "dewarp",
    "dewarp_to_anatomical",
    "rigid_align_anatomy_to_reference",
    "rigid_align_masked_anatomy_to_motion_reference",
    "syn_only_reference_to_anatomy_rigid",
    "syn_only_motion_reference_to_t1_rigid",
    "compose_native_frame_to_anatomy_rigid_transform",
    "compose_native_frame_to_t1_rigid_transform",
    "compose_anatomy_to_t1_rigid_transform",
    "compose_anatomy_to_native_frame_transform",
    "compose_native_to_anatomy_transform",
    "build_motion_reference",
    "apply_transform_chain",
    "invert_transformlist",
    "compose_transformlists",
    "normalize_transformlist",
    "DewarpResult",
    "AnatomyRigidResult",
    "CanonicalSynResult",
    "TransformApplication",
    "spatial",
    "contract",
    "tabulate",
    "viz",
    "syn",
    "registration",
    "auto_reg",
    "normalize_tensor",
    "normalize_image",
    "plot_deformation_grid",
    "plot_edge_overlay",
    "render_standard_4panel",
    "render_input_pair_figure",
    "create_registration_report",
    "create_population_benchmark_report",
    "build_engine_provenance",
    "SyNTo",
    "SyNToJax",
    "SyNToTransform",
    "calculate_inverse_identity_error",
    "FeatureSpaceLoss",
    "VGG19Extractor",
    "DINOv2Extractor",
    "ResNet10Extractor",
    "SwinUNETRExtractor",
    "image_compare",
    "correlation",
    "cap_resolution_for_registration",
    "clip_cervical_spine_fov",
    "CrossProductGenerator",
    "benchmark_data",
    "TVFModel",
    "TVFModelJAX",
    "tvf_registration",
    "tvf",
    "GeodesicShootingModel",
    "GeodesicShootingModelJAX",
    "syngs_registration",
    "syngs",
    "integrate_momentum",
    "shoot_geodesic",
    "momentum_to_deformation",
    "integrate_momentum_jax",
    "shoot_geodesic_jax",
    "momentum_to_deformation_jax",
    "robust_affine",
    "robust_center_of_mass",
    "compute_center_of_mass",
    "extract_2d_slice",
    "run_benchmark_suite",
    "high_level_benchmark_run",
    "compute_pair_metrics",
    "scattered",
    "landmarks",
    "detect_blobs_log",
    "detect_blobs_dog",
    "detect_sift2d",
    "detect_sift3d",
    "compute_mind",
    "extract_mind_at_points",
    "match_landmarks",
    "ransac_filter",
    "compute_tre",
    "project_scattered_to_grid",
    "ScatteredProjector",
    "ProjectionConfig",
    "differentiable_grid_projection",
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
    "greedy",
    "greedy_registration",
    "GreedyRegistrationModel",
    "ImageDiagnosis",
    "PairDiagnosis",
    "diagnose_image",
    "diagnose_pair",
    "RegistrationPolicy",
    "synthesize_policy",
    "auto_policy_for_images",
    "data",
    "build_template",
    "reflect_image",
    "motion_correction",
    "MotionParameters",
    "TransformCollection",
    "MotionCorrectionResult",
    "calculate_framewise_displacement",
    "calculate_dvars",
    "batched_rigid_register_pass",
    "batched_rigid_register_pass_temporal",
    "batched_group_bias_register_pass",
    "deformation_gradient",
    "RegistrationQCReport",
    "evaluate_registration_qc",
    "__version__",
]
