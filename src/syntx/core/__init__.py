"""
syntx.core — Shared algorithmic infrastructure for SyN, TVF, and SyNGS.
"""

from .affine import (
    get_rotation_matrix,
    HierarchicalAffine,
    grid_to_physical_affine_torch,
    physical_to_grid_affine,
    grid_to_physical_affine,
    parse_ants_affine,
    compute_initial_grid,
)
from .grid import (
    grid_sample_bspline_torch,
    AnalyticalGridSample,
    grid_sample_nd,
    compose_grids,
    get_physical_grid_torch,
    physical_to_normalized_torch,
    physical_to_normalized_torch_cached,
    prepare_mid_images_and_gradients_torch,
)
from .smoothing import (
    separable_gaussian_filter,
    get_cached_gaussian_kernel_1d,
    apply_sobolev_green_operator,
    apply_dsti_green_operator,
    apply_dsti1_green_operator,
    get_boundary_mask,
)
from .losses import (
    AnalyticalLNCC,
    ANTsPseudoLNCC,
    local_ncc_loss_nd,
    b_spline_3,
    mattes_mi_loss_core,
    mattes_mi_loss_nd,
)
from .jacobian import (
    _spatial_jacobian_nd,
    compute_jacobian_determinant_nd,
    compute_physical_jacobian_determinant,
)
from .inverse import (
    update_inverse_field_nd_hybrid_lm,
    integrate_time_varying_velocity_field,
    update_inverse_field_nd_anderson,
    update_inverse_field_nd,
    compute_inverse_identity_error_nd,
    calculate_inverse_identity_error,
)
from .optimizers import (
    LARS,
    SobolevAdam,
    get_cfl_max_norm,
    compute_cfl_step,
    check_convergence,
)
from .pipeline import (
    auto_detect_device,
    normalize_and_tensorize,
    cleanup_gpu,
)
from .utils import (
    normalize_tensor,
    normalize_image,
)

__all__ = [
    'get_rotation_matrix',
    'HierarchicalAffine',
    'grid_to_physical_affine_torch',
    'physical_to_grid_affine',
    'grid_to_physical_affine',
    'parse_ants_affine',
    'compute_initial_grid',
    'grid_sample_bspline_torch',
    'AnalyticalGridSample',
    'grid_sample_nd',
    'compose_grids',
    'get_physical_grid_torch',
    'physical_to_normalized_torch',
    'physical_to_normalized_torch_cached',
    'prepare_mid_images_and_gradients_torch',
    'separable_gaussian_filter',
    'get_cached_gaussian_kernel_1d',
    'apply_sobolev_green_operator',
    'apply_dsti_green_operator',
    'apply_dsti1_green_operator',
    'get_boundary_mask',
    'AnalyticalLNCC',
    'ANTsPseudoLNCC',
    'local_ncc_loss_nd',
    'b_spline_3',
    'mattes_mi_loss_core',
    'mattes_mi_loss_nd',
    '_spatial_jacobian_nd',
    'compute_jacobian_determinant_nd',
    'compute_physical_jacobian_determinant',
    'update_inverse_field_nd_hybrid_lm',
    'integrate_time_varying_velocity_field',
    'update_inverse_field_nd_anderson',
    'update_inverse_field_nd',
    'compute_inverse_identity_error_nd',
    'calculate_inverse_identity_error',
    'LARS',
    'get_cfl_max_norm',
    'compute_cfl_step',
    'check_convergence',
    'auto_detect_device',
    'normalize_and_tensorize',
    'cleanup_gpu',
    'normalize_tensor',
    'normalize_image',
]
