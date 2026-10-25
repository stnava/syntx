"""Metric sample-point selection: one implementation, a menu of named strategies, used by every point-sampled metric.

A strategy decides WHERE the metric looks (its candidate region) and HOW OFTEN each voxel is picked (its weights); the
shared machinery here does the rest -- robust normalisation, the without-replacement weighted draw, the within-voxel jitter,
the pool / per-evaluation budget and the provenance record. Call sites (``motion_batched``) never implement selection.

Speed: the region and weights are computed once (ANTs, on the CPU) and uploaded once as float32; every later draw, jitter
and index stays on the compute device with a generator on that device (no host round trip per draw, no float64).
Draws are reproducible per device (a seeded MPS generator repeats on MPS; CPU and MPS draws differ).

Menu (``SAMPLING_STRATEGIES``; ``get_sampling_strategy(name)``; an unknown name raises; default ``DEFAULT_SAMPLING`` =
``"uniform"``, see its comment for the evidence):

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

Further strategies (regular stride, dense) are added as menu entries, one at a time, each with its own validation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import ants
import numpy as np
import torch

MIN_REGION_VOXELS = 500

# The default menu entry, chosen on the ground-truth motion phantoms (syntx.motion_simulation, 5 seeds; ASL, slow flow
# with / without eddy, DWI, BOLD) plus the tied eddy PE-affine recovery test: "uniform" keeps nearly all of the gain over
# the former fixed "> 0.01" sampler (DWI rotation 0.67 -> 0.42 deg, slow flow 0.28 +- 0.14 -> 0.16 deg) and recovers the
# eddy PE scale (0.0288 for 0.03); "gradient" / "gradient_floor" improve rigid rotation by a further 0.01-0.06 deg but
# bias that scale to 0.0165 (an edge-weighted joint histogram moves the optimum of a scale, which needs signal across the
# field). The eddy term is a displacement applied to the data, so an unbiased default wins.
DEFAULT_SAMPLING = "uniform"


def background_floor(arr: np.ndarray, z: float = 4.0, bins: int = 200) -> dict:
    """Noise floor of a magnitude image from its own background: ``threshold = mu + z * sd``.

    Exact zeros are excluded (zero-filled / clipped voxels carry no noise). The background sample is the positive voxels at
    or below three times the mode of the lower half of the positive histogram (for Rayleigh noise the mode is sigma and
    3 sigma holds 99 % of it); ``mu`` = its median, ``sd`` = 1.4826 x its MAD. A zero-filled background (exact zeros
    outnumber the positive background sample, or no spread) has no noise to measure: the floor is then 0 (every non-zero
    voxel is signal), with a ``UserWarning`` and ``"background": "zero-filled"`` in the record.
    """
    a = np.asarray(arr, dtype=np.float32).ravel()
    pos = a[a > 0]
    n_zero = int((a == 0).sum())
    if pos.size < MIN_REGION_VOXELS:
        raise ValueError(f"{pos.size} positive voxels: no image to sample")
    low = pos[pos <= np.percentile(pos, 50.0)]
    h, e = np.histogram(low, bins=int(bins))
    i = int(h.argmax())
    mode = 0.5 * float(e[i] + e[i + 1])
    bg = pos[pos <= 3.0 * mode]
    mu = float(np.median(bg)) if bg.size else 0.0
    sd = 1.4826 * float(np.median(np.abs(bg - mu))) if bg.size else 0.0
    if bg.size < MIN_REGION_VOXELS or n_zero > bg.size or not sd > 0.0:
        # zero-filled background (e.g. 3D PCASL with background suppression: 81727 exact zeros vs 70231 low positive
        # voxels): there is no noise to measure; every non-zero voxel is signal by construction, so the floor is 0.
        # Announced and recorded, never silent.
        warnings.warn(f"background is zero-filled ({n_zero} exact zeros, {bg.size} low positive voxels): sampling region "
                      "= voxels > 0 below the saturated top; pass sample_mask to restrict it", UserWarning, stacklevel=3)
        return {"background": "zero-filled", "zero_fraction": n_zero / a.size, "mode": mode, "mu": None, "sd": None,
                "z": float(z), "threshold": 0.0, "background_voxels": int(bg.size)}
    return {"background": "noise", "zero_fraction": n_zero / a.size, "mode": mode, "mu": mu, "sd": sd, "z": float(z),
            "threshold": mu + float(z) * sd, "background_voxels": int(bg.size)}


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
        if sample_mask is not None:
            reg &= np.asarray(sample_mask, dtype=bool)
        return reg

    def weights(self, normalized, region, sigma_mm):
        # ITK GradientMagnitudeRecursiveGaussian: derivatives in physical units, sigma in mm (image spacing used)
        g = ants.iMath(normalized, "Grad", float(sigma_mm)).numpy().astype(np.float32)
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
    proportional to ``weights`` (Gumbel top-k, without replacement; zero-weight entries are never drawn). float32, sorted.
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
                 seed: int = 42, device: Union[str, torch.device] = "cpu"):
        self.strategy = get_sampling_strategy(strategy)
        self.reference = reference
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.seed = int(seed)
        self.sample_fraction = float(sample_fraction)
        self.pool_fraction = sample_budget(sample_fraction, stage_sampling)
        self.mask = None
        if sample_mask is not None:
            self.mask = np.asarray(sample_mask.numpy() if hasattr(sample_mask, "numpy") else sample_mask) > 0
            if tuple(self.mask.shape) != tuple(reference.shape):
                raise ValueError(f"sample_mask shape {self.mask.shape} differs from the reference grid {tuple(reference.shape)}")
        self.normalized = normalize_robust(reference)
        self.region = self.strategy.region(reference, self.normalized, None)
        if getattr(self.strategy, "normalized", None) is not None:
            self.normalized = self.strategy.normalized     # foreground-normalised: the weights use the same scale
        n_region = int(self.region.sum())
        n_masked = n_region if self.mask is None else int((self.region & self.mask).sum())
        if min(n_region, n_masked) < MIN_REGION_VOXELS:
            raise ValueError(f"sampling strategy {self.strategy.name!r} leaves {min(n_region, n_masked)} candidate voxels "
                             f"(< {MIN_REGION_VOXELS}) in the reference: empty or saturated reference, or a too small sample_mask")
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
        norm = 2.0 * self.vox_xyz / (shape_xyz - 1.0) - 1.0
        out = torch.nn.functional.grid_sample(t[None, None], norm.view(1, 1, 1, -1, 3), mode="nearest",
                                              align_corners=True).reshape(-1)
        return out.clamp_min(0.0)

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
                "background": getattr(self.strategy, "background", None),
                "sample_fraction": self.sample_fraction, "pool_fraction": self.pool_fraction,
                "region_voxels": int(self.idx_region.numel()),
                "masked_region_voxels": None if self.mask is None else int((self.region & self.mask).sum()),
                "pool_points": self.n_pool, "pool_is_physical": True, "sigma_mm": self.sigma_mm}
