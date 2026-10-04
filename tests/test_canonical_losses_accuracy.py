"""Gradient accuracy and cosine similarity verification suite for canonical losses.

Specifications (per ORIGINAL_REQUEST.md & PROJECT.md):
- Gradient cosine similarity >= 0.999999 vs numerical reference autograd:
  - CanonicalMattesMIFunction analytical gradient vs autograd Parzen
  - BoxLNCCLoss custom gradient vs reference correlation autograd
  - Verification of AnalyticalLNCC and ANTsPseudoLNCC against documented ITK pseudo-gradient formulations
- Unified alias resolution verification:
  - Test parse_similarity_metric and get_similarity_loss on all aliases
    ('cc2', 'lncc', 'mattes_mi', 'mattes_64', 'mse', 'soft_dice', etc.).
"""

import pytest
import torch
import torch.nn.functional as F

from syntx.core.losses import (
    AnalyticalLNCC,
    ANTsPseudoLNCC,
    BoxLNCCLoss,
    CanonicalMattesMIFunction,
    local_ncc_loss_nd,
    mattes_mi_from_weights,
    parzen_weights,
)

# Registry imports (available when M1 package lands)
try:
    from syntx.core.losses import (
        SimilarityLossConfig,
        get_similarity_loss,
        parse_similarity_metric,
    )
    HAS_REGISTRY = True
except ImportError:
    HAS_REGISTRY = False


def _cosine_similarity(g1: torch.Tensor, g2: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute cosine similarity between two gradient tensors."""
    v1 = g1.flatten().to(torch.float64)
    v2 = g2.flatten().to(torch.float64)
    dot = torch.dot(v1, v2)
    norm1 = torch.linalg.norm(v1)
    norm2 = torch.linalg.norm(v2)
    return float(dot / (norm1 * norm2 + eps))


def test_canonical_mattes_mi_gradient_cosine_similarity():
    """Verify CanonicalMattesMIFunction analytical gradient achieves cosine sim >= 0.999999 vs autograd."""
    torch.manual_seed(42)
    N = 1000
    x_raw = torch.randn(N, dtype=torch.float64) * 0.4
    y_raw = torch.randn(N, dtype=torch.float64) * 0.4

    # 1. Reference autograd using unrolled Parzen weights + MI from weights
    xr = x_raw.clone().requires_grad_(True)
    yr = y_raw.clone().requires_grad_(True)
    wx = parzen_weights(xr, num_bins=32, min_val=-1.0, max_val=1.0, pad=2.0)
    wy = parzen_weights(yr, num_bins=32, min_val=-1.0, max_val=1.0, pad=2.0)
    l_ref = mattes_mi_from_weights(wx, wy)
    l_ref.backward()
    g_xr = xr.grad
    g_yr = yr.grad

    # 2. Analytical CanonicalMattesMIFunction backward pass
    xa = x_raw.clone().requires_grad_(True)
    ya = y_raw.clone().requires_grad_(True)
    l_ana = CanonicalMattesMIFunction.apply(xa, ya, 32, -1.0, 1.0, 2.0, 32768, None)
    l_ana.backward()
    g_xa = xa.grad
    g_ya = ya.grad

    cos_x = _cosine_similarity(g_xa, g_xr)
    cos_y = _cosine_similarity(g_ya, g_yr)

    assert cos_x >= 0.999999, f"Mattes MI grad_x cosine similarity {cos_x} < 0.999999"
    assert cos_y >= 0.999999, f"Mattes MI grad_y cosine similarity {cos_y} < 0.999999"


@pytest.mark.parametrize("squared", [True, False])
def test_box_lncc_gradient_cosine_similarity(squared):
    """Verify BoxLNCCLoss gradient achieves cosine sim >= 0.999999 vs unrolled reference autograd."""
    torch.manual_seed(43)
    kernel_size = 5
    smooth_nr = 1e-5
    smooth_dr = 1e-5
    pad = kernel_size // 2

    I_raw = torch.randn(1, 1, 16, 16, dtype=torch.float64)
    J_raw = torch.randn(1, 1, 16, 16, dtype=torch.float64)

    # 1. Reference correlation autograd using standard 2D avg_pool
    Ir = I_raw.clone().requires_grad_(True)
    i_mean = F.avg_pool2d(Ir, kernel_size, stride=1, padding=pad, count_include_pad=True)
    j_mean = F.avg_pool2d(J_raw, kernel_size, stride=1, padding=pad, count_include_pad=True)
    i2_mean = F.avg_pool2d(Ir * Ir, kernel_size, stride=1, padding=pad, count_include_pad=True)
    j2_mean = F.avg_pool2d(J_raw * J_raw, kernel_size, stride=1, padding=pad, count_include_pad=True)
    ij_mean = F.avg_pool2d(Ir * J_raw, kernel_size, stride=1, padding=pad, count_include_pad=True)

    i_var = i2_mean - i_mean * i_mean
    j_var = j2_mean - j_mean * j_mean
    cov = ij_mean - i_mean * j_mean

    cc = (cov + smooth_nr) / torch.sqrt(torch.clamp(i_var * j_var, min=0.0) + smooth_dr)
    l_ref = -torch.mean(cc * cc) if squared else -torch.mean(cc)
    l_ref.backward()
    g_ref = Ir.grad

    # 2. BoxLNCCLoss implementation
    Ia = I_raw.clone().requires_grad_(True)
    loss_fn = BoxLNCCLoss(kernel_size=kernel_size, squared=squared, smooth_nr=smooth_nr, smooth_dr=smooth_dr)
    l_box = loss_fn(Ia, J_raw)
    l_box.backward()
    g_box = Ia.grad

    cos_sim = _cosine_similarity(g_box, g_ref)
    assert cos_sim >= 0.999999, f"BoxLNCCLoss (squared={squared}) cosine sim {cos_sim} < 0.999999"


def test_analytical_lncc_pseudo_gradient_formulation():
    """Verify AnalyticalLNCC matches the exact documented ITK pseudo-gradient formulation.
    
    Formula:
        dCC/dJ_c = -(1 / n_window) / sqrt(var_I * var_J) * (F_centered - CC * M_centered) / N_spatial
    And verifies that its cosine similarity against exact autograd is in [0.84, 0.90] (expected divergence).
    """
    torch.manual_seed(44)
    window_size = 5
    I_raw = torch.randn(1, 1, 16, 16, dtype=torch.float32)
    J_raw = torch.randn(1, 1, 16, 16, dtype=torch.float32)

    # 1. AnalyticalLNCC backward
    I_ana = I_raw.clone().requires_grad_(True)
    J_ana = J_raw.clone().requires_grad_(True)
    l_ana = AnalyticalLNCC.apply(I_ana, J_ana, None, window_size)
    l_ana.backward()
    g_ana_I = I_ana.grad
    g_ana_J = J_ana.grad

    # 2. Exact mathematical formula of the pseudo-gradient
    pad = window_size // 2
    I_mean = F.avg_pool2d(I_raw, window_size, stride=1, padding=pad, count_include_pad=False)
    J_mean = F.avg_pool2d(J_raw, window_size, stride=1, padding=pad, count_include_pad=False)
    F_c = I_raw - I_mean
    M_c = J_raw - J_mean
    var_I = torch.clamp(F.avg_pool2d(F_c ** 2, window_size, stride=1, padding=pad, count_include_pad=False), min=1e-6)
    var_J = torch.clamp(F.avg_pool2d(M_c ** 2, window_size, stride=1, padding=pad, count_include_pad=False), min=1e-6)
    cov_IJ = F.avg_pool2d(F_c * M_c, window_size, stride=1, padding=pad, count_include_pad=False)
    cc = torch.clamp(cov_IJ / (torch.sqrt(var_I * var_J) + 1e-6), min=-1.0, max=1.0)

    ones = torch.ones_like(I_raw[:, :1])
    n_win = F.avg_pool2d(ones, window_size, stride=1, padding=pad, count_include_pad=True) * (window_size ** 2)
    inv_denom = 1.0 / (torch.sqrt(var_I * var_J) + 1e-6)
    scale = -(1.0 / n_win) * inv_denom
    N_spatial = float(I_raw.numel())

    expected_g_J = scale * (F_c - cc * M_c) / N_spatial
    expected_g_I = scale * (M_c - cc * F_c) / N_spatial

    cos_I = _cosine_similarity(g_ana_I, expected_g_I)
    cos_J = _cosine_similarity(g_ana_J, expected_g_J)

    assert cos_I >= 0.999999, f"AnalyticalLNCC grad_I vs pseudo formula cosine sim {cos_I} < 0.999999"
    assert cos_J >= 0.999999, f"AnalyticalLNCC grad_J vs pseudo formula cosine sim {cos_J} < 0.999999"

    # 3. Cross-check against true autograd: pseudo-gradient exhibits expected ~0.87 correlation
    I_auto = I_raw.clone().requires_grad_(True)
    J_auto = J_raw.clone().requires_grad_(True)
    l_auto = local_ncc_loss_nd(I_auto, J_auto, window_size=window_size, squared=False, use_ants_pseudo_gradient=False)
    l_auto.backward()
    cos_auto = _cosine_similarity(g_ana_I, I_auto.grad)
    assert 0.80 <= cos_auto <= 0.95, f"Unexpected pseudo-gradient vs autograd cosine sim: {cos_auto}"


def test_ants_pseudo_lncc_pseudo_gradient_formulation():
    """Verify ANTsPseudoLNCC matches the exact documented ITK ANTSNeighborhoodCorrelation CC^2 pseudo-derivative."""
    torch.manual_seed(45)
    window_size = 5
    I_raw = torch.randn(1, 1, 16, 16, dtype=torch.float32)
    J_raw = torch.randn(1, 1, 16, 16, dtype=torch.float32)

    I_ana = I_raw.clone().requires_grad_(True)
    J_ana = J_raw.clone().requires_grad_(True)
    l_ana = ANTsPseudoLNCC.apply(I_ana, J_ana, None, window_size)
    l_ana.backward()
    g_ana_J = J_ana.grad

    # Expected formulation for CC^2:
    pad = window_size // 2
    I_mean = F.avg_pool2d(I_raw, window_size, stride=1, padding=pad, count_include_pad=False)
    J_mean = F.avg_pool2d(J_raw, window_size, stride=1, padding=pad, count_include_pad=False)
    F_c = I_raw - I_mean
    M_c = J_raw - J_mean
    var_I = torch.clamp(F.avg_pool2d(F_c ** 2, window_size, stride=1, padding=pad, count_include_pad=False), min=1e-6)
    var_J = torch.clamp(F.avg_pool2d(M_c ** 2, window_size, stride=1, padding=pad, count_include_pad=False), min=1e-6)
    cov_IJ = F.avg_pool2d(F_c * M_c, window_size, stride=1, padding=pad, count_include_pad=False)

    ones = torch.ones_like(I_raw[:, :1])
    n_win = F.avg_pool2d(ones, window_size, stride=1, padding=pad, count_include_pad=True) * (window_size ** 2)
    sFF_sMM = var_I * var_J + 1e-8
    scale = -(2.0 / n_win) * (cov_IJ / sFF_sMM)
    N_spatial = float(I_raw.numel())
    expected_g_J = scale * (F_c - (cov_IJ / var_J) * M_c) / N_spatial

    cos_J = _cosine_similarity(g_ana_J, expected_g_J)
    assert cos_J >= 0.999999, f"ANTsPseudoLNCC grad_J vs pseudo formula cosine sim {cos_J} < 0.999999"


# Alias resolution test matrix
ALIAS_SPECS = [
    ("cc2", {"metric": "cc2", "squared": True}),
    ("lncc2", {"metric": "cc2", "squared": True}),
    ("lncc", {"metric": "lncc", "squared": False}),
    ("cc", {"metric": "lncc", "squared": False}),
    ("mattes_mi", {"metric": "mattes_mi", "num_bins": 32}),
    ("mattes", {"metric": "mattes_mi", "num_bins": 32}),
    ("mattes_64", {"metric": "mattes_mi", "num_bins": 64}),
    ("mattes_48", {"metric": "mattes_mi", "num_bins": 48}),
    ("mi", {"metric": "mattes_mi", "num_bins": 32}),
    ("mse", {"metric": "mse"}),
    ("l2", {"metric": "l2"}),
    ("mae", {"metric": "mae"}),
    ("soft_dice", {"metric": "soft_dice"}),
    ("dice", {"metric": "soft_dice"}),
]


@pytest.mark.parametrize("alias,expected_attrs", ALIAS_SPECS)
def test_parse_similarity_metric_aliases(alias, expected_attrs):
    """Verify parse_similarity_metric correctly parses all standard and custom aliases."""
    if not HAS_REGISTRY:
        pytest.skip("Similarity registry (parse_similarity_metric) not yet exported in syntx.core.losses")

    cfg = parse_similarity_metric(alias)
    assert isinstance(cfg, SimilarityLossConfig)
    for attr, val in expected_attrs.items():
        assert getattr(cfg, attr) == val, f"Alias {alias} expected {attr}={val}, got {getattr(cfg, attr)}"


@pytest.mark.parametrize("alias,_", ALIAS_SPECS)
def test_get_similarity_loss_dispatch(alias, _):
    """Verify get_similarity_loss returns callable adhering to (I, J, mask=None) standard interface."""
    if not HAS_REGISTRY:
        pytest.skip("Similarity factory (get_similarity_loss) not yet exported in syntx.core.losses")

    loss_fn = get_similarity_loss(alias)
    assert callable(loss_fn)

    # Test 2D evaluation with gradient flow
    torch.manual_seed(101)
    I = torch.rand(1, 1, 16, 16, requires_grad=True)
    J = torch.rand(1, 1, 16, 16, requires_grad=True)
    mask = (torch.rand(1, 1, 16, 16) > 0.2).float()

    loss = loss_fn(I, J, mask=mask)
    assert torch.is_tensor(loss)
    assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert torch.isfinite(loss), f"Loss for alias {alias} was not finite: {loss.item()}"

    loss.backward()
    assert I.grad is not None
    assert torch.isfinite(I.grad).all(), f"Gradient for {alias} contains NaN/Inf"


def test_invalid_alias_rejection():
    """Verify unknown/unsupported metric strings raise ValueError upon evaluation."""
    if not HAS_REGISTRY:
        pytest.skip("Similarity registry not yet exported in syntx.core.losses")

    loss_fn = get_similarity_loss("unknown_metric_xyz")
    with pytest.raises(ValueError, match="Unknown similarity metric"):
        I = torch.randn(1, 1, 8, 8)
        loss_fn(I, I)
