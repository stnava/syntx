"""Normalized-gradient-field (NGF) alignment: a contrast-agnostic similarity between two images (numpy, CPU).

Edge-weighted mean of ``cos^2`` between the gradients of two images, in [0, 1] (1 = identical edge directions wherever
there are edges, 0 = orthogonal). The sign of a gradient does not matter (an inverted T1 vs b0 edge counts as aligned)
and the score is insensitive to intensity scale. Used as an acceptance gate and objective for T1-anchored refinement
(antsxdwi); a score, not a differentiable training loss.
"""

from __future__ import annotations

import numpy as np


class NGFReference:
    """Fixed image whose gradient and noise floor are computed once; ``alignment(moving)`` scores many moving images.

    Gradients are taken in physical units (``spacing``). The edge-noise floor is ``noise_fraction`` times the MEAN
    gradient magnitude inside ``mask`` (the mean, not the median: a mostly flat masked image has a zero median, a zero
    floor and a 0/0 at flat voxels). Raises ValueError for an empty mask or an image without gradient in the mask.
    """

    def __init__(self, fixed, spacing, mask=None, noise_fraction: float = 0.02):
        self.spacing = tuple(float(s) for s in spacing)
        self.noise_fraction = float(noise_fraction)
        self.g = self._gradient(fixed)
        shape = self.g.shape[:-1]
        self.mask = np.ones(shape, bool) if mask is None else np.asarray(mask) > 0
        if not self.mask.any():
            raise ValueError("ngf_alignment: empty mask")
        self.n, self.e = self._norm_floor(self.g)
        self.w = self.n**2 / (self.n**2 + self.e**2)

    def _gradient(self, arr):
        return np.stack(np.gradient(np.asarray(arr, dtype=np.float64), *self.spacing), axis=-1)

    def _norm_floor(self, g):
        n = np.linalg.norm(g, axis=-1)
        e = self.noise_fraction * float(n[self.mask].mean())
        if not e > 0.0:
            raise ValueError("ngf_alignment: an image has no gradient inside the mask")
        return n, e

    def alignment(self, moving) -> float:
        gb = self._gradient(moving)
        nb, eb = self._norm_floor(gb)
        cos2 = (self.g * gb).sum(-1) ** 2 / ((self.n**2 + self.e**2) * (nb**2 + eb**2))
        # edge weight: ~1 where both gradients are well above the noise floor, ~0 on flat voxels, so flat background
        # neither helps nor hurts and identical edges score exactly 1
        w = self.w * nb**2 / (nb**2 + eb**2)
        out = float(cos2[self.mask].sum() / w[self.mask].sum())
        if not np.isfinite(out):
            raise ValueError("ngf_alignment: non-finite result")
        return out


def ngf_alignment(a, b, spacing, mask=None, noise_fraction: float = 0.02) -> float:
    """NGF alignment of two equally shaped arrays on a grid with ``spacing`` (see ``NGFReference``)."""
    return NGFReference(a, spacing, mask, noise_fraction).alignment(b)
