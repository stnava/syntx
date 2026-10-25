"""Metric sample-point selection: one implementation, a menu of named strategies, used by every point-sampled metric.

A strategy decides WHERE the metric looks (its candidate region) and HOW OFTEN each voxel is picked (its weights); the
shared machinery here does the rest -- robust normalisation, the without-replacement weighted draw, the within-voxel jitter,
the pool / per-evaluation budget and the provenance record. Call sites (``motion_batched``) never implement selection.

Speed: the region and weights are computed once (ANTs, on the CPU) and uploaded once as float32; every later draw, jitter
and index stays on the compute device with a generator on that device (no host round trip per draw, no float64).
Draws are reproducible per device (a seeded MPS generator repeats on MPS; CPU and MPS draws differ).

Menu (``SAMPLING_STRATEGIES``; ``get_sampling_strategy(name)``; an unknown name raises; default ``DEFAULT_SAMPLING`` =
``"gradient_floor"``, see its comment for the evidence):

Candidate region (shared by every entry, ``background_floor``): voxels above the reference's own noise floor,
``raw > mu_bg + z * sd_bg`` (z = 4), and below the saturated top (``n < upper``, 0.98, of the 2-98 % normalised image),
intersected with a caller mask; the 2-98 % percentiles are taken over the voxels above the floor. ``mu_bg`` / ``sd_bg`` are the median / 1.4826 MAD of the positive voxels at or below three
times the low-intensity histogram mode; exact zeros (zero-filled or clipped voxels) are excluded. On EPI references the
background is structured (ghosts, non-stationary noise), so part of it stays above the floor (FPA sub-BLAST079: 40-58 % of
the kept region outside a rough head mask; NM slab 23 %); the head itself is kept (< 3 % lost).

``"gradient"``
    Points drawn without replacement with probability proportional to ``|grad n|`` (Gaussian-derivative gradient magnitude
    in physical units, sigma in mm).

``"uniform"``
    The same candidate region, equal weights.

``"gradient_floor"``
    The same candidate region, probability proportional to ``floor + |grad n| / max |grad n|`` (floor 0.1).

Dense-grid metrics (the JAX SyN / TVF Mattes MI on whole warped grids) use the two grid entries below, so their
sampling rule also lives here: ``regular_stride`` (``"regular"``: every k-th voxel of the flattened grid, unchanged from
the former inline stride) and ``uniform_cube_points`` (seeded uniform points in the normalised grid cube; formerly an
unseeded ``np.random.uniform``). Restricting those metrics to a strategy region is a separate, separately validated step.

Further strategies are added as menu entries, one at a time, each with its own validation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import ants
import numpy as np
import torch

MIN_REGION_VOXELS = 500          # candidate voxels required: min(MIN_REGION_VOXELS, MIN_REGION_FRACTION of the grid)
MIN_REGION_FRACTION = 0.02       # tiny images (a 12^3 test phantom has 1728 voxels) need fewer
MIN_REGION_VOXELS_TINY = 25
# Absolute cap on the points one metric evaluation sees: Mattes MI precision depends on the number of points (a 32 x 32
# joint histogram is well filled by 1e4-1e5), not on the fraction of a large grid. Real DWI (FPA sub-BLAST079 AP,
# 140 x 140 x 104, 100 vols, MPS): no cap 170 s / mean FD 0.116 mm; 50 k 115 s / 0.109; 20 k 78 s / 0.125; former
# sampler 65 s / 0.163. 20 k keeps most of the gain at near the former cost. The phantoms of the gate never reach it.
MAX_EVAL_POINTS = 20000

# The default menu entry, chosen on the ground-truth motion phantoms (syntx.motion_simulation, 5 seeds; ASL, slow flow
# with / without eddy, DWI, BOLD) plus the tied eddy PE-affine recovery test, with the noise-floor region:
# "gradient_floor" is best or tied in 4 of 5 cases (rotation error deg: ASL 0.113, slow flow + eddy 0.135, DWI 0.392 with
# the smallest worst seed 0.57, BOLD 0.138; former fixed "> 0.01" sampler 0.181 / 0.183 / 0.672 / 0.192) and recovers the
# eddy PE scale (0.0322 for 0.03, rigid FD 0.049 mm); "uniform": DWI 0.430 (worst seed 0.83), eddy scale 0.0362.
# (With the earlier 0.01 < n < 0.98 region on whole-grid percentiles the gradient strategies biased that scale to 0.0165;
# the noise-floor region removed the bias.)
DEFAULT_SAMPLING = "gradient_floor"


def background_floor(arr: np.ndarray, z: float = 4.0, bins: int = 200) -> dict:
    """Noise floor of a magnitude image from its own background: ``threshold = mu + z * sd``.

    Three backgrounds, told apart by the exact zeros and the low positive population:

    Exact zeros in contiguous blocks (padding, zero-filling outside the reconstruction) are set aside first (a
    morphological opening keeps blocks and removes isolated zeros); only scattered zeros count below.

    * ``"noise"`` (few scattered zeros): the background sample is the positive voxels at or below three times the mode of the lower
      half of the positive histogram (Rayleigh: mode = sigma, 3 sigma holds 99 %); ``mu`` = median, ``sd`` = 1.4826 MAD
      (1.2533 x mean absolute deviation when the MAD is 0, e.g. integer data with sigma ~1). For Rayleigh noise the
      floor at z = 4 sits near 3.8 sigma, a tail of about 8e-4 of the background voxels.
    * ``"clipped_noise"`` (zeros at least half as many as that low positive population, and tiny next to tissue): noise
      around 0 whose negative half was clipped to 0, so the ``n_zero`` smallest positive voxels are its positive half
      (half-normal): ``sd = median / 0.6745``, ``mu = 0``. Real 3D PCASL (FPA sub-BLAST079: 81727 zeros, 70231 low
      positives) and noisy synthetic pairs are of this kind.
    * ``"zero-filled"`` (zero blocks over >= 20 % of the grid and the positives hold no noise population -- the
      ``no_background`` test below -- or scattered zeros with no tiny positive population: the scanner zeroed the background): there is no noise to measure; the floor is 0 (every
      non-zero voxel is signal), with a ``UserWarning``.
    * ``"no_background"`` (the object fills the field of view: the "noise" sample is more than 3x noisier than the
      image's own noise from neighbour differences, or its floor reaches the 98th percentile): nothing to exclude, floor 0.
    """
    arr = np.asarray(arr, dtype=np.float32)
    a = arr.ravel()
    pos = a[a > 0]
    # Exact zeros are either PADDING / zero-filling (contiguous blocks: a morphological opening keeps them) or CLIPPED noise
    # (isolated zeros among small positive noise values: the opening removes them). Only the latter describe the noise.
    from scipy.ndimage import binary_opening

    zmask = arr == 0
    padding = binary_opening(zmask) if zmask.any() else zmask
    n_padding = int(padding.sum())
    n_zero = int(zmask.sum()) - n_padding
    if pos.size < 2 * MIN_REGION_VOXELS_TINY:
        raise ValueError(f"{pos.size} positive voxels: no image to sample")
    p98 = float(np.percentile(pos, 98.0))
    low = pos[pos <= np.percentile(pos, 50.0)]
    h, e = np.histogram(low, bins=int(bins))
    i = int(h.argmax())
    mode = 0.5 * float(e[i] + e[i + 1])
    bg = pos[pos <= 3.0 * mode]
    rec = {"zero_fraction": float(zmask.mean()), "padding_zero_fraction": n_padding / a.size, "mode": mode, "z": float(z)}

    if n_zero >= 0.5 * max(bg.size, 1):
        s_half = np.partition(pos, min(n_zero, pos.size) - 1)[:min(n_zero, pos.size)]
        med_half = float(np.median(s_half))
        if med_half < 0.05 * p98:                      # a tiny positive half next to tissue: clipped noise
            sd = med_half / 0.6745
            return {**rec, "background": "clipped_noise", "mu": 0.0, "sd": sd, "threshold": float(z) * sd,
                    "background_voxels": int(s_half.size)}
        warnings.warn(f"background is zero-filled ({n_zero} scattered exact zeros, no tiny positive population): sampling region = "
                      "voxels > 0 below the saturated top; pass sample_mask to restrict it", UserWarning, stacklevel=3)
        return {**rec, "background": "zero-filled", "mu": None, "sd": None, "threshold": 0.0, "background_voxels": 0}
    mu = float(np.median(bg))
    sd = 1.4826 * float(np.median(np.abs(bg - mu)))
    rec["sd_estimator"] = "mad"
    if not sd > 0.0:
        # integer-valued low-noise data (rounded noise of sigma ~0.7-1: most values equal the median, MAD = 0): the mean
        # absolute deviation still sees the spread (sigma = 1.2533 MeanAD for Gaussian-like noise)
        sd = 1.2533 * float(np.mean(np.abs(bg - mu)))
        rec["sd_estimator"] = "mean_abs_dev"
    if not sd > 0.0:
        raise ValueError(f"background spread is zero (median {mu:.4g}): constant background; pass sample_mask")
    # the low half of the histogram is background only if the image HAS one: an object filling the field of view puts
    # dark tissue there. Check its spread against the image's own noise (neighbour differences, robust to edges): a
    # "background" more than 3x noisier than the image is tissue, and there is no background to exclude.
    sd_img = _noise_from_differences(arr)
    rec["noise_sd_from_differences"] = sd_img
    if (sd_img > 0.0 and sd > 3.0 * sd_img) or mu + float(z) * sd >= p98:
        # also: a floor at or above the bright end means the "background" model spans the whole image (e.g. pure noise
        # or uniform texture, no air): nothing separable to exclude among the positives. With zero blocks over >= 20 % of
        # the grid that is a zero-filled background (the scanner zeroed it): every non-zero voxel is signal, announced.
        if n_padding >= 0.2 * a.size:
            warnings.warn(f"background is zero-filled ({n_padding} voxels in zero blocks, no positive noise population): "
                          "sampling region = voxels > 0 below the saturated top; pass sample_mask to restrict it",
                          UserWarning, stacklevel=3)
            return {**rec, "background": "zero-filled", "mu": None, "sd": None, "threshold": 0.0, "background_voxels": 0}
        return {**rec, "background": "no_background", "mu": None, "sd": sd_img, "threshold": 0.0,
                "background_voxels": 0}
    return {**rec, "background": "noise", "mu": mu, "sd": sd, "threshold": mu + float(z) * sd,
            "background_voxels": int(bg.size)}


def _noise_from_differences(arr: np.ndarray) -> float:
    """Noise standard deviation from neighbour differences along the first axis over the positive voxels:
    ``1.4826 MAD(diff) / sqrt(2)``; edges are a minority of differences, so the median ignores them."""
    a = np.asarray(arr, dtype=np.float32)
    d = np.diff(a, axis=0)
    both = (a[1:] > 0) & (a[:-1] > 0)
    d = d[both]
    if d.size < 2 * MIN_REGION_VOXELS_TINY:
        return 0.0
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))


def normalize_robust(img: ants.ANTsImage, lower_pct: float = 2.0, upper_pct: float = 98.0,
                     mask: Optional[np.ndarray] = None) -> ants.ANTsImage:
    """Rescale to [0, 1] between the ``lower_pct`` / ``upper_pct`` percentiles of the positive voxels (within ``mask``
    when given), clipped.

    If the two percentiles coincide (``p_hi <= p_lo + 1e-4``) the range [0, max positive] is used; with no positive voxel,
    the image min / max. Returns a new float32 image with the same header (origin, spacing, direction).
    """
    arr = img.numpy()
    pos = arr[(arr > 0) & (np.asarray(mask, dtype=bool) if mask is not None else True)]
    if len(pos) > 0:
        p_lo = float(np.percentile(pos, lower_pct))
        p_hi = float(np.percentile(pos, upper_pct))
        if p_hi <= p_lo + 1e-4:
            p_lo, p_hi = 0.0, float(pos.max())
    else:
        p_lo, p_hi = float(arr.min()), float(arr.max())
    return img.new_image_like(np.clip((arr - p_lo) / (p_hi - p_lo + 1e-6), 0.0, 1.0).astype(np.float32))


def gradient_magnitude_mm(img: ants.ANTsImage, sigma_mm: float) -> np.ndarray:
    """|grad| of ``img`` after a Gaussian of ``sigma_mm`` (mm per axis = sigma_mm / spacing voxels), derivatives per mm
    (central differences with the image spacing). Works on any grid, including axes of 1-3 voxels (ITK's recursive
    Gaussian aborts the process below 4 voxels on an axis)."""
    from scipy.ndimage import gaussian_filter

    a = img.numpy().astype(np.float32)
    sp = [float(x) for x in img.spacing]
    sm = gaussian_filter(a, sigma=[float(sigma_mm) / s_ if n > 1 else 0.0 for s_, n in zip(sp, a.shape)], mode="nearest")
    g2 = np.zeros_like(sm)
    for ax, (s_, n) in enumerate(zip(sp, a.shape)):
        if n > 1:
            g2 += np.gradient(sm, s_, axis=ax) ** 2
    return np.sqrt(g2).astype(np.float32)


class SamplingStrategy:
    """Base class of a menu entry: ``region`` (candidate voxels) and ``weights`` (None = uniform over the region)."""

    name: str = ""

    def params(self) -> dict:
        return {}

    def region(self, reference: ants.ANTsImage, normalized: ants.ANTsImage, sample_mask: Optional[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def weights(self, normalized: ants.ANTsImage, region: np.ndarray, sigma_mm: float) -> Optional[np.ndarray]:
        raise NotImplementedError


@dataclass
class GradientSampling(SamplingStrategy):
    """Candidates above the noise floor (``background_floor``, ``z``) and below ``upper`` of the normalised reference;
    probability proportional to ``|grad n|``."""

    z: float = 4.0
    upper: float = 0.98
    name: str = field(default="gradient", init=False)
    background: Optional[dict] = field(default=None, init=False, repr=False)
    normalized: Optional[ants.ANTsImage] = field(default=None, init=False, repr=False)

    def params(self) -> dict:
        return {"z": float(self.z), "upper": float(self.upper)}

    def region(self, reference, normalized, sample_mask):
        self.background = background_floor(reference.numpy(), z=self.z)
        above = reference.numpy() > self.background["threshold"]
        # the saturated top is the brightest 2 % of the FOREGROUND: percentiles over every positive voxel let the
        # background set them, and with a small head in a large field of view "top 2 % of the grid" is a large share of
        # the tissue (head = 11 % of the grid: 22 % of it cut)
        self.normalized = normalize_robust(reference, mask=above)
        reg = above & (self.normalized.numpy() < self.upper)
        # the saturated top is a thin tail (2 % by construction); when more than 5 % of the region ties at the top it is
        # a plateau of real signal (a constant-intensity object), not saturation: no cut
        cut = int(above.sum()) - int(reg.sum())
        self.background["upper_cut"] = bool(cut <= 0.05 * max(int(above.sum()), 1))
        if not self.background["upper_cut"]:
            reg = above
        if sample_mask is not None:
            reg &= np.asarray(sample_mask, dtype=bool)
        return reg

    def weights(self, normalized, region, sigma_mm):
        g = gradient_magnitude_mm(normalized, sigma_mm)
        return np.where(region, np.maximum(g, 0.0), 0.0).astype(np.float32)


@dataclass
class UniformSampling(GradientSampling):
    """The shared candidate region (noise floor, saturated top), equal weights."""

    name: str = field(default="uniform", init=False)

    def weights(self, normalized, region, sigma_mm):
        return None


@dataclass
class GradientFloorSampling(GradientSampling):
    """The ``gradient`` candidate region; probability proportional to ``floor + |grad n| / max |grad n|``: edges are
    favoured but flat regions keep a share (``floor`` 0.1 as in ``robust_affine``'s gradient weighting)."""

    floor: float = 0.1
    name: str = field(default="gradient_floor", init=False)

    def params(self) -> dict:
        return {**super().params(), "floor": float(self.floor)}

    def weights(self, normalized, region, sigma_mm):
        g = super().weights(normalized, region, sigma_mm)
        gmax = float(g.max())
        rel = g / gmax if gmax > 0 else g
        return np.where(region, float(self.floor) + rel, 0.0).astype(np.float32)


SAMPLING_STRATEGIES = {"gradient": GradientSampling, "uniform": UniformSampling, "gradient_floor": GradientFloorSampling}


def get_sampling_strategy(strategy: Union[str, SamplingStrategy] = DEFAULT_SAMPLING, **params) -> SamplingStrategy:
    """A strategy object from its menu name (with ``params``) or the object itself; an unknown name raises."""
    if isinstance(strategy, SamplingStrategy):
        if params:
            raise ValueError("params are only accepted with a strategy name, not with a strategy object")
        return strategy
    key = str(strategy).lower()
    if key not in SAMPLING_STRATEGIES:
        raise ValueError(f"unknown sampling strategy {strategy!r}; menu: {sorted(SAMPLING_STRATEGIES)}")
    return SAMPLING_STRATEGIES[key](**params)


def sample_budget(sample_fraction: float, stage_sampling: float) -> float:
    """Pool fraction of the candidate region such that a stage drawing ``stage_sampling`` of the pool evaluates
    ``sample_fraction`` of the region (``sample_fraction >= 1``: the whole region, every stage)."""
    sf = float(sample_fraction)
    if not 0.0 < sf <= 1.0:
        raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction!r}")
    if not 0.0 < float(stage_sampling) <= 1.0:
        raise ValueError(f"stage_sampling must be in (0, 1], got {stage_sampling!r}")
    return 1.0 if sf >= 1.0 else min(1.0, sf / float(stage_sampling))


def draw_indices(weights: Optional[torch.Tensor], n: int, k: int, generator: torch.Generator) -> torch.Tensor:
    """``k`` distinct indices of ``range(n)`` on ``generator``'s device: uniform (``weights`` None) or with probability
    proportional to ``weights`` (Gumbel top-k, without replacement; zero-weight entries are never drawn): ``min(k, number
    of positive weights)`` int64 indices, sorted; weights are used in float32.
    ``weights`` must already be on the generator's device (no transfer here)."""
    dev = generator.device
    if weights is None:
        return torch.randperm(n, generator=generator, device=dev)[:k].sort().values
    if weights.device != dev:
        raise ValueError(f"weights on {weights.device}, generator on {dev}: upload the weights once, not per draw")
    w = weights.detach().to(torch.float32)
    pos = w > 0
    k = min(int(k), int(pos.sum()))
    u = torch.rand(w.shape, generator=generator, device=dev, dtype=torch.float32).clamp(min=torch.finfo(torch.float32).tiny, max=1.0 - 1e-7)
    keys = torch.where(pos, torch.log(w.clamp_min(torch.finfo(torch.float32).tiny)) - torch.log(-torch.log(u)),
                       torch.full_like(w, -float("inf")))
    return torch.topk(keys, k).indices.sort().values


def jitter_within_voxel(vox_xyz: torch.Tensor, shape_xyz: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Move each point uniformly within its voxel (clamped to the grid). On voxel centres the warped moving frame is sampled
    exactly at grid nodes at the identity and between nodes otherwise, a pose-dependent interpolation bias (0.5 mm per b3000
    frame on a known-truth phantom); off-centre points make it pose-independent. On ``generator``'s device."""
    j = vox_xyz + (torch.rand(vox_xyz.shape, generator=generator, device=generator.device) - 0.5)
    return torch.minimum(j.clamp(min=0.0), shape_xyz.to(vox_xyz.device) - 1.0)


class PointSampler:
    """Physical sample points of one reference image, shared by every scale of one point-sampled metric.

    Built once per registration: robust normalisation and the strategy's candidate region. ``pool()`` draws ONE set of
    physical points (mm), uniformly over the candidate region (``sample_budget`` of it, at least ``min_points``; the caller's
    ``sample_mask`` is not applied to the pool), jittered within their voxels. Every scale evaluates its own images at these
    same points. ``weights(sigma_mm, use_sample_mask)`` gives the strategy's weights AT the pool points at a scale (the
    level's voxel size in mm; 0 outside the candidate region or, with ``use_sample_mask``, outside the caller's mask): a
    scale draws its points from the pool with these weights, so the strategy acts at every scale on one point set.
    """

    def __init__(self, reference: ants.ANTsImage, *, strategy: Union[str, SamplingStrategy] = DEFAULT_SAMPLING,
                 sample_mask=None, sample_fraction: float = 0.15, stage_sampling: float = 0.2, min_points: int = 2000,
                 max_eval_points: Optional[int] = None,
                 seed: int = 42, device: Union[str, torch.device] = "cpu"):
        self.strategy = get_sampling_strategy(strategy)
        self.reference = reference
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.seed = int(seed)
        self.sample_fraction = float(sample_fraction)
        self.pool_fraction = sample_budget(sample_fraction, stage_sampling)
        self.stage_sampling = float(stage_sampling)
        cap = MAX_EVAL_POINTS if max_eval_points is None else int(max_eval_points)   # None: the module cap; 0: no cap
        self.max_eval_points = None if not cap else int(cap)
        self.mask = None
        if sample_mask is not None:
            self.mask = np.asarray(sample_mask.numpy() if hasattr(sample_mask, "numpy") else sample_mask) > 0
            if tuple(self.mask.shape) != tuple(reference.shape):
                raise ValueError(f"sample_mask shape {self.mask.shape} differs from the reference grid {tuple(reference.shape)}")
        self.normalized = normalize_robust(reference)
        self.region = self.strategy.region(reference, self.normalized, None)
        # this sampler's own copy: a strategy object shared by several samplers keeps only the last call's record
        self.background = None if getattr(self.strategy, "background", None) is None else dict(self.strategy.background)
        if getattr(self.strategy, "normalized", None) is not None:
            self.normalized = self.strategy.normalized     # foreground-normalised: the weights use the same scale
        n_region = int(self.region.sum())
        n_masked = n_region if self.mask is None else int((self.region & self.mask).sum())
        need = max(MIN_REGION_VOXELS_TINY, min(MIN_REGION_VOXELS, int(MIN_REGION_FRACTION * self.region.size)))
        if min(n_region, n_masked) < need:
            raise ValueError(f"sampling strategy {self.strategy.name!r} leaves {min(n_region, n_masked)} candidate voxels "
                             f"(< {need}) in the reference: empty or saturated reference, or a too small sample_mask")
        self.sigma_mm = float(min(reference.spacing))
        self.idx_region = torch.from_numpy(np.flatnonzero(self.region.reshape(-1))).to(self.device)   # uploaded once
        self.min_points = int(min_points)
        self.n_pool = 0
        self.vox_xyz: Optional[torch.Tensor] = None
        self._weights_cache: dict = {}

    def pool(self) -> dict:
        """The shared pool: ``vox_xyz`` (continuous full-resolution voxel coordinates) and ``phys`` (S, 3, mm) on ``device``."""
        n_region = int(self.idx_region.numel())
        n = min(n_region, max(self.min_points, int(self.pool_fraction * n_region)))
        if self.max_eval_points is not None and self.sample_fraction < 1.0:
            # a stage evaluates stage_sampling of the pool: cap the pool so an evaluation sees at most max_eval_points
            # (independently of the pool fraction: sample_fraction >= stage_sampling makes the pool the whole region)
            n = min(n, max(self.min_points, int(self.max_eval_points / self.stage_sampling)))
        sel = self.idx_region[draw_indices(None, n_region, n, self.generator)]
        self.n_pool = int(sel.numel())
        shape = tuple(int(s) for s in self.reference.shape)
        vox = torch.stack(torch.unravel_index(sel, shape), dim=-1).to(torch.float32)
        self.vox_xyz = jitter_within_voxel(vox, torch.tensor(shape, dtype=torch.float32, device=self.device), self.generator)
        return {"vox_xyz": self.vox_xyz, "phys": self.to_physical(self.vox_xyz)}

    def weights(self, sigma_mm: Optional[float] = None, use_sample_mask: bool = True) -> torch.Tensor:
        """(n_pool,) float32 weights of the pool points at scale ``sigma_mm`` (default: the reference's finest spacing).
        Computed once per (scale, mask) and cached; a uniform strategy gives the region indicator."""
        if self.vox_xyz is None:
            raise RuntimeError("PointSampler.weights before pool()")
        sig = self.sigma_mm if sigma_mm is None else float(sigma_mm)
        key = (round(sig, 6), bool(use_sample_mask) and self.mask is not None)
        if key not in self._weights_cache:
            region = self.region if not key[1] else (self.region & self.mask)
            w = self.strategy.weights(self.normalized, region, sig)
            field_np = region.astype(np.float32) if w is None else w.astype(np.float32)
            if key[1]:
                field_np = field_np * self.mask
            self._weights_cache[key] = self._at_pool(field_np)
        return self._weights_cache[key]

    def _at_pool(self, field_np: np.ndarray) -> torch.Tensor:
        """The field's value in each pool point's OWN voxel (nearest; the jitter keeps a point within +-0.5 voxel of its
        voxel centre). Bilinear interpolation would mix in the zeros of neighbouring voxels outside the region and
        under-weight every point near the region's edge (uniform strategy: 0.43-1.0 instead of 1)."""
        t = torch.from_numpy(np.ascontiguousarray(np.transpose(field_np, (2, 1, 0)))).to(self.device)
        shape_xyz = torch.tensor(list(self.reference.shape), dtype=torch.float32, device=self.device)
        norm = 2.0 * self.vox_xyz / (shape_xyz - 1.0).clamp_min(1.0) - 1.0
        out = torch.nn.functional.grid_sample(t[None, None], norm.view(1, 1, 1, -1, 3), mode="nearest",
                                              align_corners=True).reshape(-1)
        return out.clamp_min(0.0)

    def eval_points(self, fraction: float, n_available: int, minimum: int = 500) -> int:
        """Points one metric evaluation draws: ``fraction`` of ``n_available`` (at least ``minimum``), never more than
        ``max_eval_points`` unless the caller asked for dense sampling (``sample_fraction >= 1``)."""
        if float(fraction) >= 1.0 or self.sample_fraction >= 1.0:
            return int(n_available)
        k = min(int(n_available), max(int(minimum), int(float(fraction) * n_available)))
        return k if self.max_eval_points is None else min(k, max(int(minimum), self.max_eval_points))

    def draw(self, k: int, sigma_mm: Optional[float] = None, use_sample_mask: bool = True) -> torch.Tensor:
        """``k`` indices into the pool drawn with the strategy's weights at scale ``sigma_mm`` (device tensor)."""
        return draw_indices(self.weights(sigma_mm, use_sample_mask), self.n_pool, int(k), self.generator)

    def to_physical(self, vox_xyz: torch.Tensor) -> torch.Tensor:
        ref = self.reference
        f32 = dict(dtype=torch.float32, device=vox_xyz.device)
        sp = torch.tensor(ref.spacing, **f32)
        orig = torch.tensor(ref.origin, **f32)
        dirm = torch.tensor(np.asarray(ref.direction), **f32)
        return orig + (vox_xyz * sp) @ dirm.t()

    def provenance(self) -> dict:
        return {"strategy": self.strategy.name, "params": self.strategy.params(), "seed": self.seed,
                "background": self.background,
                "sample_fraction": self.sample_fraction, "pool_fraction": self.pool_fraction,
                "max_eval_points": self.max_eval_points,
                "region_voxels": int(self.idx_region.numel()),
                "masked_region_voxels": None if self.mask is None else int((self.region & self.mask).sum()),
                "pool_points": self.n_pool, "pool_is_physical": True, "sigma_mm": self.sigma_mm}


def regular_stride(sampling_fraction: Optional[float]) -> int:
    """``"regular"`` dense-grid entry: keep every ``k``-th voxel of the flattened grid, ``k = int(1 / fraction)``
    (``None`` or ``>= 1``: every voxel)."""
    if sampling_fraction is None or float(sampling_fraction) >= 1.0:
        return 1
    if not float(sampling_fraction) > 0.0:
        raise ValueError(f"sampling fraction must be in (0, 1], got {sampling_fraction!r}")
    return max(1, int(1.0 / float(sampling_fraction)))


def uniform_cube_points(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` points uniform in the normalised grid cube ``[-1, 1]^dim`` (float32), from a seeded numpy generator."""
    return rng.uniform(-1.0, 1.0, (int(n), int(dim))).astype(np.float32)


def strategy_on_grid(arr_xyz: np.ndarray, spacing, origin, direction,
                     strategy: Union[str, SamplingStrategy] = DEFAULT_SAMPLING, weights_on_region: bool = True) -> tuple:
    """A strategy's candidate region and weights on an arbitrary grid (e.g. one pyramid level of a registration):
    ``(region, weights, background_record)`` as numpy arrays on ``arr_xyz``'s grid (weights None for a uniform strategy).
    The level's physical header (``spacing``, ``origin``, ``direction``) is kept so the noise floor and the gradient
    sigma (the level's finest spacing, mm) are those of the level. ``weights_on_region=False``: weights over every
    voxel (a caller that samples outside the region)."""
    st = get_sampling_strategy(strategy)
    img = ants.from_numpy(np.ascontiguousarray(arr_xyz, dtype=np.float32), origin=tuple(float(x) for x in origin),
                          spacing=tuple(float(x) for x in spacing), direction=np.asarray(direction, dtype=np.float64))
    norm = normalize_robust(img)
    region = st.region(img, norm, None)
    if getattr(st, "normalized", None) is not None:
        norm = st.normalized
    w = st.weights(norm, region if weights_on_region else np.ones_like(region), float(min(spacing)))
    return region, w, (None if getattr(st, "background", None) is None else dict(st.background))
