"""
Multi-resolution image pyramids: the CENTRAL pyramid of syntx (``plan_levels`` + ``build_pyramid``) and the public
``build_image_pyramid`` delegate.

The contract is ``docs/PYRAMID.md``; the evidence behind the defaults is ``docs/PYRAMID_EVIDENCE.md``.

Central API (Pass 2)
--------------------
``plan_levels``      pure-Python planner (no torch, no numpy arrays): per level the target shape, working spacing, per-axis
                     factor ``f`` and offset ``off`` (index j on axis i sits at full-resolution continuous index
                     ``j * f + off``), the per-axis operator and the planned smoothing. The torch and the JAX operators
                     execute the same plan.
``build_pyramid``    torch operators; returns ``Level`` objects (tensor, f, off, spacing, shape, optional valid mask and mask
                     tensor, physical-coordinate helper).
``build_pyramid_jax`` the same plan on JAX arrays (jax is optional and imported lazily).
``warp_to_moving_norm``, ``level_spacing``  shared helpers (replace the per-solver copies).
``cap_ants_image``   the ANTs-image wrapper behind ``imaging_utils.cap_resolution_for_registration``.
Presets: ``mm_iso`` (default for affine / motion), ``mm_iso_margin``, ``legacy_avgpool``, ``legacy_interp``,
``legacy_floor16`` (see ``PRESETS``).

``build_image_pyramid`` is the public list-of-tensors delegate of ``build_pyramid(..., 'legacy_interp')``. The former
``build_anti_aliased_pyramid`` (no production caller) was deleted; ``tests/legacy_pyramid_reference.py`` keeps the exact code of both
incumbents as the golden references of ``tests/test_pyramid_central.py``.
"""
import math
from dataclasses import dataclass, field, replace
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def build_image_pyramid(image, spacing, levels, smoothing_sigmas=None, sigma_mode='voxel'):
    """Gaussian-smoothed multi-resolution pyramid of ``(B, C, *spatial)`` as a list of tensors: a thin delegate of the central
    ``build_pyramid`` with the ``legacy_interp`` preset (sigma = ``log2(level)`` voxels unless ``smoothing_sigmas`` is given, then
    ``F.interpolate(align_corners=True)`` to ``floor(n / level)``). Public API, kept for ``syn`` / ``syngs`` / ``tvf`` and outside callers.

    ``smoothing_sigmas`` None, a float, or one value per level; ``sigma_mode`` 'voxel' or 'physical' (``spacing`` is used only
    for 'physical'). Results are bit-identical to the previous implementation, except that a thin axis no longer crashes or
    collapses to 0 / 1 voxel (it keeps at least 2 voxels, or its native count when that is smaller).
    Raises ``ValueError`` when ``smoothing_sigmas`` does not match ``levels``, ``PyramidLevelError`` for an invalid level."""
    if smoothing_sigmas is not None and not isinstance(smoothing_sigmas, (int, float)) and len(smoothing_sigmas) != len(levels):
        raise ValueError(f"Length of smoothing_sigmas ({len(smoothing_sigmas)}) must match levels ({len(levels)})")
    dim = image.dim() - 2
    if sigma_mode == 'physical':
        if spacing is None:
            raise ValueError("sigma_mode='physical' needs the voxel spacing")
        sp = spacing
    else:
        sp = (1.0,) * dim                       # legacy_interp's grid and voxel sigma never use the spacing
    return [lv.tensor for lv in build_pyramid(image, sp, levels, 'legacy_interp', sigma=smoothing_sigmas, sigma_mode=sigma_mode)]


# =====================================================================================================================
# The central pyramid
# =====================================================================================================================
MIN_VOXELS_PER_AXIS = 16
"""THE minimum-voxel floor constant (value of greedy's floor and of ``imaging_utils.DEFAULT_MIN_VOXELS_PER_AXIS``). A preset or
caller opts in with ``floor=MIN_VOXELS_PER_AXIS`` (``legacy_floor16`` does). Presets that reproduce an incumbent without a floor
(``mm_iso``, ``legacy_avgpool``, ``legacy_interp``) default to the structural minimum only; the affine solver keeps its own
coarsest-level clamp (8 finest voxels) until Pass 4 chooses between the two."""

HARD_MIN_VOXELS = 2
"""Structural minimum, never tunable: no level of an axis that has >= 2 samples ever has fewer than 2 (never 0 or 1), and no
call raises on a thin axis. An input axis of 1 sample (a 2-D slice stored as 3-D) passes through unchanged."""

ANISOTROPY_RATIO = 1.25     # same threshold as robust_affine.ANISOTROPY_RESAMPLE_RATIO: max / min spacing above it = anisotropic
_EPS = 1e-6


class PyramidLevelError(RuntimeError):
    """A requested level cannot be built (invalid policy, levels, sigma or geometry); never a silent fallback."""


@dataclass(frozen=True)
class PyramidPolicy:
    """How a level grid is planned and which operator builds it (see ``PRESETS``)."""
    name: str
    grid: str                       # 'mm_iso' | 'voxel' | 'interp_size' | 'shape_floor'
    operator: str                   # 'matrix' | 'avgpool' | 'interp'
    sigma_rule: str                 # default smoothing: 'none' | 'log2_voxel' | 'compact_ratio'
    kernel: str                     # 'bessel' (replicate padding) | 'compact' (erf kernel, zero padding: legacy greedy)
    floor: Optional[int]            # default per-axis minimum voxels (None: HARD_MIN_VOXELS only)
    smooth_uncoarsened: bool        # legacy presets smooth every axis when sigma > 0; mm presets never smooth an axis that is not coarsened
    margin: int = 0                 # valid-region margin (native slices) for an anisotropic image
    align_corners: Optional[bool] = None   # interp operators only
    sigma_clip: Optional[Tuple[float, float]] = None   # legacy_interp 'physical' sigma: voxels clipped to this range
    floor_upsamples: bool = False   # legacy_floor16: an axis shorter than the floor is UP-sampled to it (greedy); others keep native sampling


PRESETS = {
    # millimetre-isotropic working grid (factor x finest spacing), centroid-exact area average on coarser axes, linear on finer
    'mm_iso': PyramidPolicy('mm_iso', 'mm_iso', 'matrix', 'none', 'bessel', None, False),
    # mm_iso + one-slice valid-region margin on an anisotropic image (use for the FIXED image of rigid / affine / motion)
    'mm_iso_margin': PyramidPolicy('mm_iso_margin', 'mm_iso', 'matrix', 'none', 'bessel', None, False, margin=1),
    # voxel average-pooling by the level (robust_affine._pool_level, motion_batched)
    'legacy_avgpool': PyramidPolicy('legacy_avgpool', 'voxel', 'avgpool', 'none', 'bessel', None, True),
    # Gaussian sigma = log2(level) voxels, then F.interpolate(align_corners=True) to floor(n / level) (build_image_pyramid)
    'legacy_interp': PyramidPolicy('legacy_interp', 'interp_size', 'interp', 'log2_voxel', 'bessel', None, True,
                                   align_corners=True, sigma_clip=(0.5, 10.0)),
    # shape max(int(n / level), 16), compact Gaussian 0.5 n / m, align_corners (greedy, build_anti_aliased_pyramid)
    'legacy_floor16': PyramidPolicy('legacy_floor16', 'shape_floor', 'interp', 'compact_ratio', 'compact', MIN_VOXELS_PER_AXIS, True,
                                    align_corners=True, floor_upsamples=True),
}
DEFAULT_POLICY = 'mm_iso'


def get_policy(policy: Union[str, PyramidPolicy]) -> PyramidPolicy:
    if isinstance(policy, PyramidPolicy):
        return policy
    if policy not in PRESETS:
        raise PyramidLevelError(f"unknown pyramid policy {policy!r}; presets: {sorted(PRESETS)}")
    return PRESETS[policy]


@dataclass(frozen=True)
class LevelPlan:
    """One planned level (all pure Python). Tuples in the order of their name: ``shape_zyx`` is tensor order, the rest ANTs (x, y, z)."""
    level: float
    shape_zyx: Tuple[int, ...]
    spacing_xyz: Tuple[float, ...]            # working spacing = f * native spacing
    f_xyz: Tuple[float, ...]                  # full-res voxels per working voxel; working index j <-> full-res index j*f + off
    off_xyz: Tuple[float, ...]
    modes_xyz: Tuple[str, ...]                # per axis: 'identity' | 'area' | 'linear' | 'pool' | 'interp'
    sigma_vox_zyx: Tuple[float, ...]          # planned smoothing before the resampling, native voxels per axis (0 = none)
    native_shape_zyx: Tuple[int, ...]
    native_spacing_xyz: Tuple[float, ...]
    is_identity: bool = False

    @property
    def shape_xyz(self):
        return tuple(self.shape_zyx[::-1])


def _eff_floor(floor: Optional[int]) -> int:
    fl = HARD_MIN_VOXELS if floor is None else int(floor)
    if fl < HARD_MIN_VOXELS:
        raise PyramidLevelError(f"floor must be >= {HARD_MIN_VOXELS} (the structural minimum), got {floor}")
    return fl


def _norm_levels(levels) -> List[float]:
    try:
        lv = [float(x) for x in levels]
    except TypeError:
        raise PyramidLevelError(f"levels must be a sequence of numbers >= 1, got {levels!r}")
    if len(lv) == 0 or any(not (x >= 1.0) for x in lv):
        raise PyramidLevelError(f"levels must be a non-empty sequence of numbers >= 1, got {list(levels)!r}")
    return lv


def _snap(x: float) -> float:
    r = round(x)
    return float(r) if abs(x - r) < 1e-9 else x


def _plan_mm_axes(n_xyz, sp, t, floor_eff, centre_to_centre):
    """Per axis (n, spacing, target mm) -> (m, f, off, mode). Never below min(floor, n) voxels (a thin axis keeps its native sampling)."""
    out = []
    for n, s in zip(n_xyz, sp):
        if n < HARD_MIN_VOXELS:                                   # a single sample: nothing to resample
            out.append((n, 1.0, 0.0, 'identity')); continue
        coarser = t > s * (1.0 + _EPS)
        if coarser or not centre_to_centre:                       # cell-aligned: the voxel extents tile the field of view
            m = max(max(1, int(round(n * s / t))), min(floor_eff, n))
            f = n / m; off = (f - 1.0) / 2.0
            mode = 'area' if f > 1.0 else 'linear'
        else:                                                      # finer or equal axis: centre-to-centre linear
            m = max(max(2, int(round((n - 1) * s / t)) + 1), min(floor_eff, n))
            f = (n - 1.0) / (m - 1.0); off = 0.0; mode = 'linear'
        if m == n and abs(f - 1.0) < 1e-12:
            out.append((n, 1.0, 0.0, 'identity')); continue
        out.append((m, f, off, mode))
    return out


def plan_levels(shape_zyx: Sequence[int], spacing_xyz: Sequence[float], levels: Sequence[float],
                policy: Union[str, PyramidPolicy] = DEFAULT_POLICY, floor: Optional[int] = None,
                sigma=None, sigma_mode: str = 'voxel', max_voxels: Optional[int] = None,
                reference_spacing_xyz: Optional[Sequence[float]] = None, centre_to_centre: bool = True,
                align_corners: Optional[bool] = None) -> List[LevelPlan]:
    """Plan every level of a pyramid (pure Python, shared by the torch and the JAX operators).

    Parameters
    ----------
    shape_zyx : spatial shape of the full-resolution image in TENSOR order (z, y, x).
    spacing_xyz : voxel spacing in ANTs order (x, y, z); ``len`` must equal ``len(shape_zyx)``.
    levels : shrink factors >= 1 (``[8, 4, 2, 1]``). Meaning depends on the policy grid: ``mm_iso`` working spacing =
        level x (finest spacing of ``reference_spacing_xyz`` or of the image), never below the image's own finest spacing;
        ``voxel`` pools by the level; ``interp_size`` gives ``floor(n / level)``; ``shape_floor`` gives ``max(int(n / level), floor)``.
    policy : a preset name of ``PRESETS`` or a ``PyramidPolicy``.
    floor : per-axis minimum voxels (default: the policy's). Raised to at least ``HARD_MIN_VOXELS``; an axis shorter than the floor
        keeps its native sampling (except ``legacy_floor16``, which up-samples it as greedy does). Never 0 or 1 voxel.
    sigma : None (the policy's rule), a float, or a per-level sequence; units ``sigma_mode`` ('voxel' = native voxels, 'physical' = mm,
        converted per axis with the image's own spacing). Per-axis sigmas <= 0 are skipped (no NaN); policies with
        ``smooth_uncoarsened=False`` never smooth an axis that is not coarsened.
    max_voxels : ``mm_iso`` only: raise the target spacing just enough to keep the level within this many voxels.
    reference_spacing_xyz : ``mm_iso`` only: the finest-spacing reference (the FIXED image's spacing) when planning a moving image
        on its own grid.
    centre_to_centre : ``mm_iso`` only: False keeps the cell-aligned (area) convention on the finer axes too.
    align_corners : interp policies only (default: the policy's); False gives cell-centred coordinates ``f = n / m``, ``off = (f - 1) / 2``.
    """
    pol = get_policy(policy)
    n_zyx = tuple(int(x) for x in shape_zyx)
    sp = tuple(float(x) for x in spacing_xyz)
    dim = len(n_zyx)
    if dim not in (2, 3) or len(sp) != dim:
        raise PyramidLevelError(f"need 2 or 3 spatial axes with one spacing each, got shape {n_zyx} and spacing {sp}")
    if any(n < 1 for n in n_zyx) or any(not (s > 0) for s in sp):
        raise PyramidLevelError(f"shape must be >= 1 and spacing > 0, got shape {n_zyx}, spacing {sp}")
    if sigma_mode not in ('voxel', 'physical'):
        raise PyramidLevelError(f"sigma_mode must be 'voxel' or 'physical', got {sigma_mode!r}")
    if align_corners is not None and pol.operator != 'interp':
        raise PyramidLevelError(f"align_corners applies to the interpolating presets only, not {pol.name!r}")
    ac = pol.align_corners if align_corners is None else bool(align_corners)
    lv = _norm_levels(levels)
    fl = floor if floor is not None else pol.floor
    floor_eff = _eff_floor(fl)
    if sigma is not None and not isinstance(sigma, (int, float)):
        sigma = [float(x) for x in sigma]
        if len(sigma) != len(lv):
            raise PyramidLevelError(f"sigma has {len(sigma)} entries for {len(lv)} levels")
    n_xyz = n_zyx[::-1]
    ref = tuple(float(x) for x in reference_spacing_xyz) if reference_spacing_xyz is not None else sp
    if len(ref) != dim:
        raise PyramidLevelError("reference_spacing_xyz must have one entry per axis")
    plans = []
    for li, L in enumerate(lv):
        # ---- grid -------------------------------------------------------------------------------------------------
        if pol.grid == 'mm_iso':
            t = max(L * min(ref), min(sp))
            axes = _plan_mm_axes(n_xyz, sp, t, floor_eff, centre_to_centre)
            nvox = float(np.prod([a[0] for a in axes]))
            if max_voxels is not None and nvox > max_voxels:
                axes = _plan_mm_axes(n_xyz, sp, t * (nvox / max_voxels) ** (1.0 / 3.0), floor_eff, centre_to_centre)
        elif pol.grid == 'voxel':
            axes = []
            for n in n_xyz:
                if n < HARD_MIN_VOXELS:
                    axes.append((n, 1.0, 0.0, 'identity')); continue
                k = max(1, min(int(L), n // min(floor_eff, n)))          # largest pool <= L that keeps min(floor, n) voxels
                m = n // k
                axes.append((m, float(k), (k - 1) / 2.0, 'pool') if k > 1 else (n, 1.0, 0.0, 'identity'))
        else:                                                              # interp_size / shape_floor
            axes = []
            for n in n_xyz:
                if L <= 1.0 or n < HARD_MIN_VOXELS:
                    axes.append((n, 1.0, 0.0, 'identity')); continue
                if pol.grid == 'interp_size':
                    m = int(math.floor(n * (1.0 / L)))
                    m = max(m, min(floor_eff, n))
                else:
                    m = max(int(n / L), floor_eff)
                    if not pol.floor_upsamples:
                        m = min(m, n)
                if m == n:
                    axes.append((n, 1.0, 0.0, 'identity')); continue
                if ac:
                    f = (n - 1.0) / (m - 1.0); off = 0.0
                else:
                    f = n / m; off = (f - 1.0) / 2.0
                axes.append((m, f, off, 'interp'))
        m_xyz = tuple(a[0] for a in axes); f_xyz = tuple(a[1] for a in axes)
        off_xyz = tuple(a[2] for a in axes); modes = tuple(a[3] for a in axes)
        # ---- smoothing --------------------------------------------------------------------------------------------
        if sigma is None:
            if pol.sigma_rule == 'none' or L <= 1.0:
                sig_zyx = [0.0] * dim
            elif pol.sigma_rule == 'log2_voxel':
                sig_zyx = [float(math.log2(L))] * dim
            elif pol.sigma_rule == 'compact_ratio':
                sig_zyx = [0.5 * (n / m) if m > 0 else 0.0 for n, m in zip(n_xyz, m_xyz)][::-1]
            else:
                raise PyramidLevelError(f"unknown sigma_rule {pol.sigma_rule!r}")
        else:
            sv = float(sigma if isinstance(sigma, (int, float)) else sigma[li])
            if sv < 0 or not math.isfinite(sv):
                raise PyramidLevelError(f"sigma must be finite and >= 0, got {sv}")
            if sigma_mode == 'physical':
                per = [sv / s for s in sp]
                if pol.sigma_clip is not None and sv > 0:
                    per = [min(max(p, pol.sigma_clip[0]), pol.sigma_clip[1]) for p in per]
            else:
                per = [sv] * dim
            sig_zyx = per[::-1]
        if not pol.smooth_uncoarsened:
            sig_zyx = [sg if f > 1.0 + _EPS else 0.0 for sg, f in zip(sig_zyx, f_xyz[::-1])]
        sig_zyx = [sg if sg > 0 else 0.0 for sg in sig_zyx]
        plans.append(LevelPlan(
            level=L, shape_zyx=tuple(m_xyz[::-1]), spacing_xyz=tuple(f * s for f, s in zip(f_xyz, sp)),
            f_xyz=f_xyz, off_xyz=off_xyz, modes_xyz=modes, sigma_vox_zyx=tuple(sig_zyx),
            native_shape_zyx=n_zyx, native_spacing_xyz=sp, is_identity=all(md == 'identity' for md in modes)))
    return plans


def plan_resample(shape_zyx: Sequence[int], spacing_xyz: Sequence[float], target_mm: float, floor: Optional[int] = MIN_VOXELS_PER_AXIS) -> LevelPlan:
    """Plan ONE linear resample onto an isotropic grid of spacing ``target_mm`` that keeps the image origin (voxel j at
    ``origin + j * target``, i.e. ``f = target / spacing``, ``off = 0``), the grid of ``ants.resample_image``. The voxel count is
    ``floor(n * spacing / target + 0.5)`` (as ANTs), raised on an axis shorter than ``floor`` to ``min(floor, n)`` samples: a thin
    axis keeps at least its native sampling and is never collapsed. Working spacing is ``target`` on every axis whose count is
    unchanged by the floor, ``n * spacing / m`` where the floor bound."""
    n_zyx = tuple(int(x) for x in shape_zyx); sp = tuple(float(x) for x in spacing_xyz)
    if not (target_mm > 0):
        raise PyramidLevelError(f"target_mm must be > 0, got {target_mm}")
    fl = _eff_floor(floor)
    axes = []
    for n, s in zip(n_zyx[::-1], sp):
        if n < HARD_MIN_VOXELS:
            axes.append((n, 1.0, 0.0, 'identity')); continue
        m = int(math.floor(n * s / target_mm + 0.5))
        floored = m < min(fl, n)
        m = max(m, min(fl, n), HARD_MIN_VOXELS)
        f = (target_mm / s) if not floored else (n * s / m) / s
        if m == n and abs(f - 1.0) < 1e-9:
            axes.append((n, 1.0, 0.0, 'identity')); continue
        axes.append((m, f, 0.0, 'linear'))
    f_xyz = tuple(a[1] for a in axes)
    return LevelPlan(level=target_mm, shape_zyx=tuple(a[0] for a in axes)[::-1], spacing_xyz=tuple(f * s for f, s in zip(f_xyz, sp)),
                     f_xyz=f_xyz, off_xyz=tuple(a[2] for a in axes), modes_xyz=tuple(a[3] for a in axes),
                     sigma_vox_zyx=(0.0,) * len(n_zyx), native_shape_zyx=n_zyx, native_spacing_xyz=sp,
                     is_identity=all(a[3] == 'identity' for a in axes))


# ---------------------------------------------------------------------------------------------- 1-D weights (numpy)
def area_matrix(n_old: int, n_new: int, f: float) -> np.ndarray:
    """(n_new, n_old) centroid-exact area-average weights; ``f = n_old / n_new > 1``; rows sum to 1.

    Working voxel j averages the full-resolution window [j f, (j + 1) f) (cell coordinates). The plain overlap-weighted box
    (the one of ``robust_affine._axis_resample_matrix``) is exact for an integer ``f`` but with fractional end weights the centroid
    of its staircase is not the window centre, a bias of up to ~0.06 working voxel. Here the two end-voxel weights are solved so
    the row has the same total weight (``f``) AND its centroid is exactly the window centre ``j f + (f - 1) / 2``; interior
    weights stay 1. For an integer ``f`` the end weights come out 1 and the matrix is the old one exactly. A row whose solved
    end weight is negative raises ``PyramidLevelError`` (it does not clamp).
    """
    W = np.zeros((n_new, n_old), dtype=np.float64)
    for j in range(n_new):
        lo, hi = _snap(j * f), _snap(min((j + 1) * f, float(n_old)))
        iL, iR = int(math.floor(lo)), int(math.ceil(hi)) - 1
        if iR <= iL:                                  # window inside one voxel (f < 1): not an area average
            raise PyramidLevelError(f"area_matrix needs f > 1, got f={f}")
        k = iR - iL - 1
        c = 0.5 * (lo + hi) - 0.5
        wL = ((f - k) * iR - f * c + k * (iL + iR) / 2.0) / (iR - iL)
        wR = (f - k) - wL
        if wL < -1e-9 or wR < -1e-9:
            raise PyramidLevelError(f"area_matrix: negative end weight at row {j} (f={f}, wL={wL}, wR={wR})")
        W[j, iL] = max(wL, 0.0); W[j, iR] = max(wR, 0.0)
        if k > 0:
            W[j, iL + 1:iR] = 1.0
        W[j] /= W[j].sum()
    return W


def linear_matrix(n_old: int, n_new: int, f: float, off: float) -> np.ndarray:
    """(n_new, n_old) linear interpolation of a signal sampled at centres to the positions ``j f + off`` (clamped to [0, n_old - 1])."""
    W = np.zeros((n_new, n_old), dtype=np.float64)
    for j in range(n_new):
        c = min(max(j * f + off, 0.0), n_old - 1.0)
        i0 = int(math.floor(c)); i1 = min(i0 + 1, n_old - 1); w1 = c - i0
        W[j, i0] += 1.0 - w1
        W[j, i1] += w1
    return W


def axis_weights(mode: str, n_old: int, n_new: int, f: float, off: float) -> np.ndarray:
    if mode == 'area':
        return area_matrix(n_old, n_new, f)
    if mode in ('linear', 'interp'):
        return linear_matrix(n_old, n_new, f, off)
    if mode == 'pool':
        k = int(round(f)); W = np.zeros((n_new, n_old))
        for j in range(n_new):
            W[j, j * k:(j + 1) * k] = 1.0 / k
        return W
    raise PyramidLevelError(f"no weights for axis mode {mode!r}")


# ---------------------------------------------------------------------------------------------- Level object
@dataclass
class Level:
    """One pyramid level. ``tensor`` is ``(B, C, *shape)``; the geometry fields are in ANTs (x, y, z) order."""
    level: float
    tensor: Any
    shape: Tuple[int, ...]                    # tensor (z, y, x) order
    spacing: Tuple[float, ...]                # working spacing, mm
    f: Tuple[float, ...]
    off: Tuple[float, ...]
    native_shape: Tuple[int, ...]             # tensor order
    native_spacing: Tuple[float, ...]
    origin: Tuple[float, ...]
    direction: Tuple[Tuple[float, ...], ...]
    plan: LevelPlan
    valid: Any = None                         # bool (1, 1, *shape) selecting samples that enter the metric, or None
    mask: Any = None                          # the mask tensor on this level grid (not thresholded), or None

    def full_res_index(self, idx_xyz) -> np.ndarray:
        """Working continuous index (..., dim) in x, y, z order -> full-resolution continuous index ``j f + off``."""
        return np.asarray(idx_xyz, dtype=np.float64) * np.asarray(self.f) + np.asarray(self.off)

    def physical_coordinates(self, idx_xyz=None) -> np.ndarray:
        """Physical (x, y, z) coordinates (float64 numpy) of working indices ``idx_xyz`` (..., dim; default every voxel, shape
        ``(*shape_zyx, dim)``): ``origin + direction @ ((j f + off) * native_spacing)``."""
        if idx_xyz is None:
            ax = [np.arange(n, dtype=np.float64) for n in self.shape]
            mesh = np.meshgrid(*ax, indexing='ij')
            idx_xyz = np.stack(mesh[::-1], axis=-1)
        full = self.full_res_index(idx_xyz) * np.asarray(self.native_spacing)
        return np.asarray(self.origin) + full @ np.asarray(self.direction, dtype=np.float64).T


def _geometry(dim, origin, direction):
    o = tuple(float(x) for x in origin) if origin is not None else (0.0,) * dim
    d = np.asarray(direction, dtype=np.float64).reshape(dim, dim) if direction is not None else np.eye(dim)
    if len(o) != dim:
        raise PyramidLevelError(f"origin needs {dim} entries, got {len(o)}")
    return o, tuple(tuple(float(x) for x in row) for row in d)


def warp_to_moving_norm(y_phys, level: Level):
    """Physical points ``y_phys (..., dim)`` (x, y, z) -> normalised ``grid_sample`` coordinates (``align_corners=True``) of the
    tensor of ``level`` (a possibly differently built moving-image level; ``Level`` of a native image has ``f = 1``, ``off = 0``).
    One implementation replacing the copies in ``robust_affine`` and ``motion_batched``:
    ``y_vox = (y - origin) @ inv(direction).T / native_spacing``; ``j = (y_vox - off) / f``; ``2 j / (shape_xyz - 1) - 1``."""
    dev, dt = y_phys.device, y_phys.dtype
    dim = len(level.f)
    t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=dt, device=dev)
    dir_inv_t = t(np.linalg.inv(np.asarray(level.direction, dtype=np.float64)).T)
    y_vox = (y_phys - t(level.origin)) @ dir_inv_t / t(level.native_spacing)
    j = (y_vox - t(level.off)) / t(level.f)
    return 2.0 * (j / (t(level.shape[::-1]) - 1.0)) - 1.0


def level_spacing(spacing_xyz, orig_shape_zyx, curr_shape_zyx):
    """ITK (x, y, z)-order spacing of a resampled ``align_corners`` grid that spans the same box:
    ``sp * (n_orig - 1) / (n_curr - 1)`` per axis (a one-sample axis keeps ``sp``). The one implementation replacing
    ``syngs._level_spacing`` and ``syn_jax.level_spacing_itk``."""
    return [float(sp) * (float(o - 1) / float(c - 1)) if c > 1 else float(sp)
            for sp, o, c in zip(spacing_xyz, reversed(tuple(orig_shape_zyx)), reversed(tuple(curr_shape_zyx)))]


def valid_region(plan: LevelPlan, margin: int, device=None):
    """Bool ``(1, 1, *shape)`` tensor, False where a working sample's centre lies within ``margin`` native slices of either end of the
    thick axis (``c < margin`` or ``c > n - 1 - margin``, ``c = j f + off``). None for an (almost) isotropic image or ``margin <= 0``."""
    sp = plan.native_spacing_xyz
    if margin <= 0 or max(sp) / min(sp) <= ANISOTROPY_RATIO:
        return None
    a = int(np.argmax(sp))
    n = int(plan.native_shape_zyx[::-1][a])
    if n < 2 * margin + 2:
        return None
    j = torch.arange(int(plan.shape_zyx[::-1][a]), dtype=torch.float64)
    c = j * float(plan.f_xyz[a]) + float(plan.off_xyz[a])
    ok = (c >= margin - 1e-6) & (c <= n - 1 - margin + 1e-6)
    shape = [1] * (2 + len(plan.shape_zyx)); shape[2 + (len(plan.shape_zyx) - 1 - a)] = -1
    full = torch.ones(1, 1, *plan.shape_zyx, dtype=torch.bool)
    return (full & ok.reshape(shape)).to(device) if device is not None else (full & ok.reshape(shape))


# ---------------------------------------------------------------------------------------------- torch operators
def _smooth_torch(x: torch.Tensor, sig_zyx, kernel: str) -> torch.Tensor:
    """Per-axis Gaussian of a channel-first tensor through ``core.smoothing.separable_gaussian_filter`` (axes with sigma <= 0 skipped)."""
    if not any(s > 0 for s in sig_zyx):
        return x
    from .core.smoothing import separable_gaussian_filter
    kt = 'compact' if kernel == 'compact' else 'bessel'
    return separable_gaussian_filter(x.movedim(1, -1), list(sig_zyx), kernel_type=kt).movedim(-1, 1)


def _apply_torch(x: torch.Tensor, plan: LevelPlan, pol: PyramidPolicy, align_corners: bool, kernel: str) -> torch.Tensor:
    x = _smooth_torch(x, plan.sigma_vox_zyx, kernel)
    if plan.is_identity:
        return x
    dim = x.dim() - 2
    modes_zyx = plan.modes_xyz[::-1]; f_zyx = plan.f_xyz[::-1]; off_zyx = plan.off_xyz[::-1]
    n_zyx = plan.native_shape_zyx
    if pol.operator == 'avgpool':
        ks = tuple(int(round(f)) if md == 'pool' else 1 for f, md in zip(f_zyx, modes_zyx))
        return (F.avg_pool3d if dim == 3 else F.avg_pool2d)(x, kernel_size=ks, stride=ks).contiguous()
    if pol.operator == 'interp':
        mode = 'trilinear' if dim == 3 else 'bilinear'
        return F.interpolate(x, size=list(plan.shape_zyx), mode=mode, align_corners=align_corners)
    out = x                                                                   # 'matrix' (and resample): one weight matrix per axis
    for ax, (md, f, off, n_old, n_new) in enumerate(zip(modes_zyx, f_zyx, off_zyx, n_zyx, plan.shape_zyx), start=2):
        if md == 'identity':
            continue
        W = torch.as_tensor(axis_weights(md, n_old, n_new, f, off), dtype=out.dtype, device=out.device)
        out = torch.movedim(torch.movedim(out, ax, -1) @ W.t(), -1, ax)
    return out.contiguous()


def _check_image(image, spacing):
    if not isinstance(image, torch.Tensor) or not image.is_floating_point():
        raise PyramidLevelError("image must be a floating-point torch tensor of shape (B, C, *spatial)")
    dim = image.dim() - 2
    if dim not in (2, 3):
        raise PyramidLevelError(f"image must be (B, C, *spatial) with 2 or 3 spatial axes, got shape {tuple(image.shape)}")
    if len(spacing) != dim:
        raise PyramidLevelError(f"spacing needs {dim} entries (x, y, z), got {len(spacing)}")
    return dim


def build_pyramid(image: torch.Tensor, spacing: Sequence[float], levels: Sequence[float],
                  policy: Union[str, PyramidPolicy] = DEFAULT_POLICY, sigma=None, sigma_mode: str = 'voxel',
                  floor: Optional[int] = None, mask: Optional[torch.Tensor] = None,
                  origin: Optional[Sequence[float]] = None, direction=None, align_corners: Optional[bool] = None,
                  max_voxels: Optional[int] = None, reference_spacing: Optional[Sequence[float]] = None,
                  valid_margin: Optional[int] = None, centre_to_centre: bool = True) -> List[Level]:
    """Build the pyramid of ``image`` ``(B, C, *spatial)`` (torch order z, y, x) with ``spacing`` in ANTs (x, y, z) order.

    ``policy`` is a preset name (``PRESETS``: ``mm_iso`` default, ``mm_iso_margin``, ``legacy_avgpool``, ``legacy_interp``,
    ``legacy_floor16``) or a ``PyramidPolicy``; ``sigma`` / ``sigma_mode`` / ``floor`` / ``max_voxels`` / ``centre_to_centre`` as in
    ``plan_levels`` (a per-axis sigma is computed from THIS image's own shape and spacing, so a moving image on its own grid gets its
    own sigma and size; pass the fixed image's spacing as ``reference_spacing`` to share the millimetre target). ``mask`` ``(B', C',
    *spatial)`` on the same grid goes through the same smoothing and operator (a float tensor: threshold it yourself; never nearest).
    ``valid_margin`` (native slices; default the policy's) fills ``Level.valid`` for an anisotropic image: use it for the FIXED image only.
    ``origin`` / ``direction`` (ANTs convention, default zeros / identity) feed ``Level.physical_coordinates`` and ``warp_to_moving_norm``.
    A level that is the identity returns ``image`` itself (shared storage).
    """
    spacing = tuple(float(x) for x in spacing)
    dim = _check_image(image, spacing)
    pol = get_policy(policy)
    shape_zyx = tuple(image.shape[2:])
    plans = plan_levels(shape_zyx, spacing, levels, pol, floor=floor, sigma=sigma, sigma_mode=sigma_mode, max_voxels=max_voxels,
                        reference_spacing_xyz=reference_spacing, centre_to_centre=centre_to_centre, align_corners=align_corners)
    ac = pol.align_corners if align_corners is None else bool(align_corners)
    mg = pol.margin if valid_margin is None else int(valid_margin)
    if mask is not None and (mask.dim() != image.dim() or tuple(mask.shape[2:]) != shape_zyx):
        raise PyramidLevelError(f"mask must share the image grid: image {tuple(image.shape)}, mask {tuple(mask.shape)}")
    o, d = _geometry(dim, origin, direction)
    out = []
    for p in plans:
        t = _apply_torch(image, p, pol, ac, pol.kernel)
        m = _apply_torch(mask.to(image.dtype), p, pol, ac, pol.kernel) if mask is not None else None
        out.append(Level(level=p.level, tensor=t, shape=p.shape_zyx, spacing=p.spacing_xyz, f=p.f_xyz, off=p.off_xyz,
                         native_shape=shape_zyx, native_spacing=spacing, origin=o, direction=d, plan=p,
                         valid=valid_region(p, mg, image.device), mask=m))
    return out


# ---------------------------------------------------------------------------------------------- JAX operators
def _import_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as e:                                                    # pragma: no cover - environment dependent
        raise ImportError("build_pyramid_jax needs jax (an optional dependency of syntx): pip install jax") from e
    return jax, jnp


def _kernel_np(sig: float, kernel: str) -> Tuple[np.ndarray, str]:
    if kernel == 'compact':
        from .core.smoothing import gaussian_1d_compact
        return gaussian_1d_compact(sig, truncated=2.0, device='cpu', dtype=torch.float32).numpy(), 'constant'
    from .core.smoothing import get_cached_gaussian_kernel_1d
    return get_cached_gaussian_kernel_1d(sig, 'cpu', torch.float32).reshape(-1).numpy(), 'edge'


def _smooth_jax(x, sig_zyx, kernel: str):
    jax, jnp = _import_jax()
    for i, sg in enumerate(sig_zyx):
        if sg <= 0:
            continue
        k, pad_mode = _kernel_np(sg, kernel)
        r = len(k) // 2
        ax = 2 + i
        pads = [(0, 0)] * x.ndim; pads[ax] = (r, r)
        xp = jnp.pad(x, pads, mode=pad_mode)
        n = x.shape[ax]
        acc = 0
        for t in range(len(k)):
            sl = [slice(None)] * x.ndim; sl[ax] = slice(t, t + n)
            acc = acc + float(k[t]) * xp[tuple(sl)]
        x = acc
    return x


def _apply_jax(x, plan: LevelPlan, pol: PyramidPolicy, kernel: str):
    _, jnp = _import_jax()
    x = _smooth_jax(x, plan.sigma_vox_zyx, kernel)
    if plan.is_identity:
        return x
    for ax, (md, f, off, n_old, n_new) in enumerate(zip(plan.modes_xyz[::-1], plan.f_xyz[::-1], plan.off_xyz[::-1],
                                                        plan.native_shape_zyx, plan.shape_zyx), start=2):
        if md == 'identity':
            continue
        W = jnp.asarray(axis_weights(md, n_old, n_new, f, off), dtype=x.dtype)
        x = jnp.moveaxis(jnp.moveaxis(x, ax, -1) @ W.T, -1, ax)
    return x


def build_pyramid_jax(image, spacing: Sequence[float], levels: Sequence[float], policy: Union[str, PyramidPolicy] = DEFAULT_POLICY,
                      sigma=None, sigma_mode: str = 'voxel', floor: Optional[int] = None, mask=None,
                      origin=None, direction=None, align_corners: Optional[bool] = None, max_voxels: Optional[int] = None,
                      reference_spacing=None, valid_margin: Optional[int] = None, centre_to_centre: bool = True) -> List[Level]:
    """``build_pyramid`` on a JAX array ``(B, C, *spatial)``: the SAME ``plan_levels`` plan, executed with jnp. Every operator is a
    separable weight matrix (average pooling, area, linear and align-corners interpolation all are), so results agree with the torch
    operators to float32 rounding (the torch ``F.interpolate`` / ``avg_pool`` paths of the legacy presets are their exact references).
    ``Level.valid`` is a torch bool tensor (a metric mask, not a JAX computation). jax is imported lazily; without it ``ImportError``."""
    _, jnp = _import_jax()
    spacing = tuple(float(x) for x in spacing)
    if image.ndim not in (4, 5) or len(spacing) != image.ndim - 2:
        raise PyramidLevelError(f"image must be (B, C, *spatial) with 2 or 3 spatial axes and one spacing per axis, got {image.shape}")
    pol = get_policy(policy)
    shape_zyx = tuple(int(x) for x in image.shape[2:])
    plans = plan_levels(shape_zyx, spacing, levels, pol, floor=floor, sigma=sigma, sigma_mode=sigma_mode, max_voxels=max_voxels,
                        reference_spacing_xyz=reference_spacing, centre_to_centre=centre_to_centre, align_corners=align_corners)
    mg = pol.margin if valid_margin is None else int(valid_margin)
    o, d = _geometry(len(spacing), origin, direction)
    out = []
    for p in plans:
        t = _apply_jax(jnp.asarray(image), p, pol, pol.kernel)
        m = _apply_jax(jnp.asarray(mask, dtype=t.dtype), p, pol, pol.kernel) if mask is not None else None
        out.append(Level(level=p.level, tensor=t, shape=p.shape_zyx, spacing=p.spacing_xyz, f=p.f_xyz, off=p.off_xyz,
                         native_shape=shape_zyx, native_spacing=spacing, origin=o, direction=d, plan=p,
                         valid=valid_region(p, mg), mask=m))
    return out


# ---------------------------------------------------------------------------------------------- ANTs wrapper
def cap_ants_image(image, max_resolution_mm: float, min_voxels_per_axis: int = MIN_VOXELS_PER_AXIS):
    """Planner implementation of ``imaging_utils.cap_resolution_for_registration`` (3-D ANTs image, linear interpolation).

    The target spacing is ``min(max_resolution_mm, shortest extent / min_voxels_per_axis)`` exactly as before (so every existing
    caller whose shortest axis keeps >= ``min_voxels_per_axis`` samples gets the same grid). When that target is not coarser than the
    finest spacing (a thin slab: the shortest extent is under ``min_voxels_per_axis`` finest voxels) the old code returned the image
    untouched and the cap was silently defeated; now the cap itself (``max_resolution_mm``) is planned and the floor is applied per
    axis (``plan_resample``: an axis never drops below ``min(min_voxels_per_axis, n)`` samples, so the thin axis keeps its native
    sampling). An image whose finest spacing already reaches the cap is returned unchanged (as before)."""
    import ants
    if image.dimension != 3:
        raise PyramidLevelError(f"cap_ants_image supports 3-D images, got {image.dimension}-D")
    spc = tuple(float(s) for s in image.spacing[:3])
    if min(spc) >= max_resolution_mm:
        return image
    shape = tuple(int(n) for n in image.shape[:3])
    safe = min(max_resolution_mm, min(n * s for n, s in zip(shape, spc)) / min_voxels_per_axis)
    target = safe if safe > min(spc) else float(max_resolution_mm)
    plan = plan_resample(shape[::-1], spc, target, floor=min_voxels_per_axis)
    if plan.is_identity:
        return image
    arr = torch.from_numpy(np.ascontiguousarray(image.numpy().astype(np.float32).transpose(2, 1, 0)))[None, None]
    out = _apply_torch(arr, plan, PRESETS['mm_iso'], True, 'bessel')[0, 0].numpy().transpose(2, 1, 0)
    return ants.from_numpy(np.ascontiguousarray(out).astype(image.numpy().dtype),
                           origin=image.origin, spacing=plan.spacing_xyz, direction=image.direction)
