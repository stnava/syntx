import collections
import concurrent.futures
import math
import pytest
import torch
import numpy as np

from syntx.core.smoothing import (
    separable_gaussian_filter,
    apply_sobolev_green_operator,
    apply_dsti_green_operator,
    apply_dsti1_green_operator,
    smooth_displacement_field_dst,
    clear_dst_cache,
    clear_dsti_filter_cache,
    get_dst_cache_info,
    get_dsti_cache_info,
    get_dsti_filter_cache_size,
    _get_dsti_filter_cached,
    _get_dst_filter_cached,
    _DST_FILTER_CACHE,
    get_boundary_mask,
    CacheInfo,
)


# ==============================================================================
# Baseline Reference Implementations for Parity Verification
# ==============================================================================

def _baseline_meshgrid_filter(spatial_shape, alpha_val, s, spacing, device, dtype):
    """Un-cached baseline computing DST-I filter via torch.meshgrid."""
    dim = len(spatial_shape)
    spacing_zyx = tuple(reversed(spacing)) if spacing is not None else None
    k_axes = []
    for d in range(dim):
        n_d = spatial_shape[d]
        sp_d = float(spacing_zyx[d]) if (spacing_zyx is not None and d < len(spacing_zyx)) else 1.0
        k_vec = torch.arange(1, n_d + 1, device=device, dtype=torch.float32)
        scale = (sp_d ** 2) if spacing is not None else 1.0
        lambda_d = (4.0 * (torch.sin(math.pi * k_vec / (2.0 * (n_d + 1))) ** 2)) / scale
        k_axes.append(lambda_d)

    if dim == 1:
        lambda_sq = k_axes[0]
    else:
        k_mesh = torch.meshgrid(*k_axes, indexing="ij")
        lambda_sq = sum(k_j for k_j in k_mesh)

    return (
        (1.0 / ((1.0 + alpha_val * lambda_sq) ** s))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .contiguous()
    )


def _baseline_apply_dsti(m, fluid_sigma=3.0, alpha=None, spacing=None, s=2.0):
    """Un-cached baseline smoothing function using torch.meshgrid."""
    if fluid_sigma <= 0:
        return m
    device = m.device
    dtype = m.dtype
    spatial_shape = m.shape[1:-1]
    dim = len(spatial_shape)
    alpha_val = float(alpha) if alpha is not None else float(fluid_sigma) / 2.0

    K_dst = _baseline_meshgrid_filter(spatial_shape, alpha_val, s, spacing, device, dtype)

    curr = m.movedim(-1, 1).to(torch.float32)
    for d in range(dim):
        axis = 2 + d
        n_d = spatial_shape[d]
        z_shape = list(curr.shape)
        z_shape[axis] = 1
        z = torch.zeros(z_shape, device=device, dtype=torch.float32)
        rev = -torch.flip(curr, dims=[axis])
        padded = torch.cat([z, curr, z, rev], dim=axis)
        F = torch.fft.fft(padded, dim=axis)
        curr = -torch.imag(F.narrow(axis, 1, n_d)).contiguous()

    curr = curr * K_dst

    for d in range(dim):
        axis = 2 + d
        n_d = spatial_shape[d]
        z_shape = list(curr.shape)
        z_shape[axis] = 1
        z = torch.zeros(z_shape, device=device, dtype=torch.float32)
        rev = -torch.flip(curr, dims=[axis])
        padded = torch.cat([z, curr, z, rev], dim=axis)
        F = torch.fft.fft(padded, dim=axis)
        curr = (-torch.imag(F.narrow(axis, 1, n_d)) / (2.0 * (n_d + 1))).contiguous()

    return curr.to(dtype=dtype).movedim(1, -1)


# ==============================================================================
# Existing Core Smoothing Tests (Preserved)
# ==============================================================================

def test_separable_gaussian_filter_constant():
    # Gaussian filter of constant tensor should be identical constant
    grid = torch.ones(1, 16, 16, 2)
    filtered = separable_gaussian_filter(grid, sigma=2.0)
    assert torch.allclose(filtered, grid, atol=1e-5)


def test_get_boundary_mask():
    mask = get_boundary_mask((8, 8), device="cpu", dtype=torch.float32, rim_size=1)
    assert mask.shape == (1, 8, 8, 1)
    # Boundaries should be 0, interior should be 1
    assert (mask[0, 0, :, 0] == 0).all()
    assert (mask[0, -1, :, 0] == 0).all()
    assert (mask[0, :, 0, 0] == 0).all()
    assert (mask[0, :, -1, 0] == 0).all()
    assert mask[0, 2:6, 2:6, 0].mean() == 1.0


def test_sobolev_green_operator_2d():
    m = torch.randn(1, 16, 16, 2)
    v = apply_sobolev_green_operator(m, fluid_sigma=3.0)
    assert v.shape == m.shape
    # Smoothing reduces energy/variance
    assert torch.std(v) < torch.std(m)


def test_dsti_green_operator_2d_and_3d():
    m_2d = torch.randn(1, 16, 16, 2)
    v_dst_2d = apply_dsti_green_operator(m_2d, fluid_sigma=3.0)
    assert v_dst_2d.shape == m_2d.shape

    v_dst1_2d = apply_dsti1_green_operator(m_2d, fluid_sigma=3.0)
    assert v_dst1_2d.shape == m_2d.shape

    m_3d = torch.randn(1, 8, 8, 8, 3)
    v_dst_3d = apply_dsti_green_operator(m_3d, fluid_sigma=2.0)
    assert v_dst_3d.shape == m_3d.shape


# ==============================================================================
# Milestone 1: Stationary Operator Caching & Interface Conformance Tests
# ==============================================================================

def test_dsti_cache_hit_matching_geometry():
    """Verify that repeated calls with identical geometry return the exact cached tensor instance."""
    clear_dst_cache()
    assert get_dsti_filter_cache_size() == 0
    info = get_dst_cache_info()
    assert info.currsize == 0
    assert info.hits == 0
    assert info.misses == 0

    k1 = _get_dsti_filter_cached((16, 16), alpha_val=1.5, s=2.0, spacing=None, device="cpu", dtype=torch.float32)
    assert get_dsti_filter_cache_size() == 1
    info = get_dst_cache_info()
    assert info.currsize == 1
    assert info.hits == 0
    assert info.misses == 1

    k2 = _get_dsti_filter_cached((16, 16), alpha_val=1.5, s=2.0, spacing=None, device="cpu", dtype=torch.float32)
    assert get_dsti_filter_cache_size() == 1
    info = get_dst_cache_info()
    assert info.currsize == 1
    assert info.hits == 1
    assert info.misses == 1
    assert k1 is k2, "Expected identical cached tensor object identity on cache hit"

    # Pre-shaped check: shape must be (1, 1, 16, 16)
    assert k1.shape == (1, 1, 16, 16)

    # Verify through high-level smoothing wrappers
    m = torch.randn(1, 16, 16, 2)
    v1 = apply_dsti_green_operator(m, fluid_sigma=3.0)  # alpha_val = 3.0 / 2.0 = 1.5, s = 2.0
    v2 = smooth_displacement_field_dst(m, fluid_sigma=3.0)
    assert get_dsti_filter_cache_size() == 1
    info = get_dst_cache_info()
    assert info.hits == 3  # k2 was 1st hit, v1 was 2nd, v2 was 3rd
    assert v1.shape == m.shape
    assert v2.shape == m.shape


def test_dsti_cache_miss_parameter_variations():
    """Verify that varying shape, spacing, alpha, s, or dtype triggers cache misses with distinct entries."""
    clear_dst_cache()

    k_base = _get_dsti_filter_cached((16, 16), 1.5, 2.0, None, "cpu", torch.float32)
    assert get_dsti_filter_cache_size() == 1

    # 1. Alter spatial shape
    k_shape = _get_dsti_filter_cached((20, 16), 1.5, 2.0, None, "cpu", torch.float32)
    assert k_shape is not k_base
    assert get_dsti_filter_cache_size() == 2

    # 2. Alter alpha
    k_alpha = _get_dsti_filter_cached((16, 16), 2.5, 2.0, None, "cpu", torch.float32)
    assert k_alpha is not k_base
    assert get_dsti_filter_cache_size() == 3

    # 3. Alter exponent s
    k_s = _get_dsti_filter_cached((16, 16), 1.5, 1.0, None, "cpu", torch.float32)
    assert k_s is not k_base
    assert get_dsti_filter_cache_size() == 4

    # 4. Alter spacing
    k_spacing = _get_dsti_filter_cached((16, 16), 1.5, 2.0, (1.0, 2.0), "cpu", torch.float32)
    assert k_spacing is not k_base
    assert get_dsti_filter_cache_size() == 5

    # 5. Spacing canonicalization (tuple vs list of matching values should hit cache)
    k_spacing_list = _get_dsti_filter_cached((16, 16), 1.5, 2.0, [1.0, 2.0], "cpu", torch.float32)
    assert k_spacing_list is k_spacing
    assert get_dsti_filter_cache_size() == 5

    # 6. Alter dtype
    k_dtype = _get_dsti_filter_cached((16, 16), 1.5, 2.0, None, "cpu", torch.float64)
    assert k_dtype is not k_base
    assert get_dsti_filter_cache_size() == 6


def test_dsti_cache_lru_eviction_and_clear():
    """Verify that the cache bounds growth at maxsize=64, evicts least recently used, and clears cleanly."""
    clear_dst_cache()
    assert get_dsti_filter_cache_size() == 0

    # Insert 70 distinct entries
    for i in range(70):
        _get_dsti_filter_cached((10 + i, 10), 1.5, 2.0, None, "cpu", torch.float32)

    assert get_dsti_filter_cache_size() == 64, "Cache size must be capped at _MAX_DST_CACHE_SIZE = 64"

    # Oldest entry (index 0: shape (10, 10)) should have been evicted
    # Cache key format: (tuple(spatial_shape), sp_tuple, str(device), str(dtype), float(alpha_val), float(s))
    old_key = ((10, 10), None, "cpu", "torch.float32", 1.5, 2.0)
    assert old_key not in _DST_FILTER_CACHE

    # Clean clearance test
    clear_dsti_filter_cache()
    assert get_dsti_filter_cache_size() == 0
    info = get_dst_cache_info()
    assert info.currsize == 0
    assert info.hits == 0
    assert info.misses == 0


def test_dsti_bitwise_equivalence_cached_vs_meshgrid_baseline():
    """Verify bitwise equivalence (L_inf = 0.0) between broadcasted cached filter and meshgrid baseline."""
    test_cases = [
        # (spatial_shape, spacing, alpha_val, s)
        ((32,), None, 1.0, 2.0),
        ((16, 20), None, 1.5, 2.0),
        ((24, 18), (1.5, 0.8), 2.0, 2.0),
        ((12, 12, 12), None, 1.0, 2.0),
        ((10, 14, 16), (2.0, 1.0, 0.5), 2.5, 1.5),
        ((8, 8, 8, 8), None, 1.0, 2.0),
    ]

    for shape, spacing, alpha_val, s in test_cases:
        k_cached = _get_dsti_filter_cached(shape, alpha_val, s, spacing, "cpu", torch.float32)
        k_baseline = _baseline_meshgrid_filter(shape, alpha_val, s, spacing, "cpu", torch.float32)
        max_diff = torch.max(torch.abs(k_cached - k_baseline)).item()
        assert max_diff == 0.0, f"Failed bitwise parity on shape {shape}: max_diff={max_diff}"


def test_dsti_output_bitwise_equivalence_against_baseline():
    """Verify bitwise equivalence (L_inf = 0.0) between smoothed displacement fields."""
    torch.manual_seed(42)

    # 2D test
    m_2d = torch.randn(2, 18, 22, 2)
    v_cached_2d = smooth_displacement_field_dst(m_2d, fluid_sigma=3.2, spacing=(1.2, 0.8))
    v_base_2d = _baseline_apply_dsti(m_2d, fluid_sigma=3.2, spacing=(1.2, 0.8))
    diff_2d = torch.max(torch.abs(v_cached_2d - v_base_2d)).item()
    assert diff_2d == 0.0, f"2D smoothing output discrepancy: {diff_2d}"

    # 3D test
    m_3d = torch.randn(1, 10, 12, 14, 3)
    v_cached_3d = smooth_displacement_field_dst(m_3d, fluid_sigma=2.4, spacing=(1.5, 1.0, 0.75))
    v_base_3d = _baseline_apply_dsti(m_3d, fluid_sigma=2.4, spacing=(1.5, 1.0, 0.75))
    diff_3d = torch.max(torch.abs(v_cached_3d - v_base_3d)).item()
    assert diff_3d == 0.0, f"3D smoothing output discrepancy: {diff_3d}"


def test_dsti_concurrent_multithreading_safety():
    """Verify thread-safety of cache lookups and smoothing operations under concurrent execution."""
    clear_dst_cache()

    def worker_same_geometry(idx):
        m = torch.randn(1, 16, 16, 2)
        return smooth_displacement_field_dst(m, fluid_sigma=3.0)

    # Scenario A: 16 concurrent threads on identical geometry
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker_same_geometry, i) for i in range(16)]
        results = [f.result() for f in futures]

    assert len(results) == 16
    assert get_dsti_filter_cache_size() == 1, "Cache size must be exactly 1 under identical geometry contention"

    # Scenario B: 32 concurrent threads on varied geometries
    def worker_varied_geometry(idx):
        shape = (14 + (idx % 6), 14 + (idx % 6))
        m = torch.randn(1, *shape, 2)
        return smooth_displacement_field_dst(m, fluid_sigma=2.0 + (idx % 3))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker_varied_geometry, i) for i in range(32)]
        results_varied = [f.result() for f in futures]

    assert len(results_varied) == 32
    assert get_dsti_filter_cache_size() <= 64


def test_dsti_2d_and_3d_tensor_compatibility():
    """Verify proper smoothing, energy reduction, and shape preservation across 2D/3D non-square shapes."""
    # 2D non-square, batch=2
    m_2d = torch.randn(2, 17, 23, 2)
    v_2d = smooth_displacement_field_dst(m_2d, fluid_sigma=3.0)
    assert v_2d.shape == m_2d.shape
    assert torch.var(v_2d) < torch.var(m_2d), "Smoothing must reduce field variance"

    # 3D non-cubic, batch=2
    m_3d = torch.randn(2, 9, 11, 13, 3)
    v_3d = smooth_displacement_field_dst(m_3d, fluid_sigma=2.5)
    assert v_3d.shape == m_3d.shape
    assert torch.var(v_3d) < torch.var(m_3d), "Smoothing must reduce field variance"


def test_dsti_zero_fluid_sigma_passthrough():
    """Verify that fluid_sigma <= 0 immediately returns the original field unchanged."""
    m = torch.randn(1, 16, 16, 2)
    v_zero = smooth_displacement_field_dst(m, fluid_sigma=0.0)
    assert v_zero is m, "Expected direct tensor return when fluid_sigma <= 0"

    v_neg = smooth_displacement_field_dst(m, fluid_sigma=-1.0)
    assert v_neg is m, "Expected direct tensor return when fluid_sigma < 0"


def test_dsti_float64_dtype_preservation():
    """Verify that float64 inputs are smoothly handled and return float64 outputs."""
    m_f64 = torch.randn(1, 16, 16, 2, dtype=torch.float64)
    v_f64 = smooth_displacement_field_dst(m_f64, fluid_sigma=2.0)
    assert v_f64.dtype == torch.float64
    assert v_f64.shape == m_f64.shape


def test_smooth_displacement_field_dst_alias_conformance():
    """Verify all argument binding conventions for the public alias smooth_displacement_field_dst."""
    m = torch.randn(1, 16, 16, 2)

    # 1. Positional m and fluid_sigma
    out1 = smooth_displacement_field_dst(m, 3.0)
    ref1 = apply_dsti_green_operator(m, fluid_sigma=3.0)
    assert torch.equal(out1, ref1)

    # 2. Keyword displacement and alpha_val
    out2 = smooth_displacement_field_dst(displacement=m, alpha_val=1.5, s=2.0, spacing=(1.0, 1.0))
    ref2 = apply_dsti_green_operator(m, alpha=1.5, s=2.0, spacing=(1.0, 1.0))
    assert torch.equal(out2, ref2)

    # 3. Keyword alpha
    out3 = smooth_displacement_field_dst(m=m, alpha=1.2)
    ref3 = apply_dsti_green_operator(m, alpha=1.2)
    assert torch.equal(out3, ref3)

    # 4. Missing tensor exception
    with pytest.raises(ValueError, match="Must provide either 'm' or 'displacement'"):
        smooth_displacement_field_dst(fluid_sigma=3.0)


def test_dst_cache_info_telemetry():
    """Verify CacheInfo namedtuple telemetry behavior and aliases."""
    clear_dst_cache()
    info = get_dst_cache_info()
    assert isinstance(info, CacheInfo)
    assert info.hits == 0
    assert info.misses == 0
    assert info.maxsize == 64
    assert info.currsize == 0

    # Unpacking check
    hits, misses, maxsize, currsize = get_dsti_cache_info()
    assert hits == 0
    assert misses == 0
    assert maxsize == 64
    assert currsize == 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_dsti_mps_compatibility():
    """Verify 2D and 3D DST-I smoothing executes cleanly on Apple Silicon MPS device."""
    device = torch.device("mps")

    # 2D on MPS
    m_mps_2d = torch.randn(1, 16, 16, 2, device=device)
    v_mps_2d = smooth_displacement_field_dst(m_mps_2d, fluid_sigma=3.0)
    assert v_mps_2d.device.type == "mps"
    assert v_mps_2d.shape == m_mps_2d.shape

    # Numerical agreement with CPU
    m_cpu_2d = m_mps_2d.cpu()
    v_cpu_2d = smooth_displacement_field_dst(m_cpu_2d, fluid_sigma=3.0)
    assert torch.allclose(v_cpu_2d, v_mps_2d.cpu(), atol=1e-5)

    # 3D on MPS
    m_mps_3d = torch.randn(1, 8, 8, 8, 3, device=device)
    v_mps_3d = smooth_displacement_field_dst(m_mps_3d, fluid_sigma=2.0)
    assert v_mps_3d.device.type == "mps"
    assert v_mps_3d.shape == m_mps_3d.shape
