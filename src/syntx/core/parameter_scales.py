"""Parameter scales and metric-manifold probing for registration optimisers (after ITK's RegistrationParameterScalesEstimator).

Two problems with hand-set optimiser constants are solved here, in physical units, independently of image spacing and of
the size of the object in the field of view:

1. *Scales.* A parameter is measured by how far it moves the sampled points: ``scale_mm`` = RMS physical displacement of
   the sample points per unit of the parameter (a translation: 1 mm per mm; a rotation: the RMS radius of the points per
   radian; an affine matrix element: the RMS coordinate of the points along the column). Expressed in voxels, one unit of
   a scaled parameter is the same physical effect for every parameter and for every image, so a single step size in
   voxels tunes a rigid, an affine, a mouse and a human registration alike.
2. *Manifold probe.* The metric is evaluated at known perturbations of each parameter (``+-probe_vox`` voxels of
   displacement) around the current pose. The central differences give, per frame and per parameter, the gradient and the
   curvature in voxel units; their ratio, the diagonal Newton step ``-g / h``, estimates the remaining distance to the
   optimum along that axis. A step-size test cannot see that distance (a slow valley moves little per iteration and is
   still far from the minimum); the probe can, and it is cheap: ``2 P + 1`` loss evaluations for a P-parameter model, for
   all frames of a batch at once, and only when the step test already suggests convergence.
"""

from __future__ import annotations

from typing import List, NamedTuple, Sequence, Tuple

import torch


class ParamAxis(NamedTuple):
    """One scalar parameter axis of a registration model.

    ``tensor`` holds the parameter, ``index`` selects the axis in it (e.g. ``(slice(None), 2)`` = column 2 of every frame of a
    (B, 3) tensor, perturbed for all frames at once; ``(2,)`` = element 2 of a single (3,) tensor) and ``scale_mm`` is the
    physical displacement of the sample points per unit of this parameter (see ``rms_shift_scale``).
    """

    tensor: torch.Tensor
    index: Tuple
    scale_mm: float


def rotation_axis_scale(points: torch.Tensor, center: torch.Tensor, axis: int) -> float:
    """Displacement (mm, RMS) of the sample points per radian of rotation about coordinate ``axis`` through ``center``: the RMS
    distance of the points from that axis (a rotation moves a point by its perpendicular distance times the angle)."""
    rel = points - center
    others = [j for j in range(3) if j != axis]
    return float(rel[:, others].pow(2).sum(-1).mean().sqrt())


def rigid_axes(t_param: torch.Tensor, omega: torch.Tensor, points: torch.Tensor, center: torch.Tensor) -> List[ParamAxis]:
    """The six axes of a per-frame rigid model (translation in mm, axis-angle rotation in radians) on (B, 3) tensors, each
    with its physical-shift scale measured on the sample ``points``."""
    axes = [ParamAxis(t_param, (slice(None), j), 1.0) for j in range(3)]
    axes += [ParamAxis(omega, (slice(None), j), rotation_axis_scale(points, center, j)) for j in range(3)]
    return axes


def matrix_element_scale(points: torch.Tensor, center: torch.Tensor, column: int) -> float:
    """Displacement (mm, RMS) of the sample points per unit change of an affine-matrix element in ``column``: the RMS of
    that coordinate of the points about the centre."""
    return float((points - center)[:, column].pow(2).mean().sqrt())


class ManifoldProbe(NamedTuple):
    """Per-frame, per-axis metric manifold around the current pose, in voxel units.

    ``grad`` (loss per voxel of displacement), ``curv`` (loss per voxel^2), ``newton_vox`` (|-g / h| in voxels; ``inf`` where
    the curvature is not positive, i.e. on a plateau or at a maximum along that axis) and ``remaining_vox`` (the largest
    Newton step over the axes: how far the frame still is from the optimum, to first order)."""

    grad: torch.Tensor
    curv: torch.Tensor
    newton_vox: torch.Tensor
    remaining_vox: torch.Tensor


@torch.no_grad()
def probe_manifold(loss_fn, axes: Sequence[ParamAxis], voxel_mm: float, probe_vox: float = 0.5, min_curv: float = 1e-12,
                   rel_floor: float = 1e-3) -> ManifoldProbe:
    """Perturb every axis by ``+-probe_vox`` voxels of physical displacement and read the loss manifold.

    ``loss_fn()`` returns the per-frame loss with the CURRENT parameter values, shape (B,) (or a scalar for a single model);
    it must be deterministic (the same sample points for every call) so the finite differences are exact derivatives of one
    objective. The parameters are restored exactly. The cost is ``2 * len(axes) + 1`` calls of ``loss_fn``.
    """
    f0 = torch.atleast_1d(loss_fn()).detach()
    grads, curvs = [], []
    for ax in axes:
        delta = float(probe_vox) * float(voxel_mm) / float(ax.scale_mm)   # parameter change that displaces the points by probe_vox voxels
        saved = ax.tensor[ax.index].clone()
        ax.tensor[ax.index] = saved + delta
        fp = torch.atleast_1d(loss_fn()).detach()
        ax.tensor[ax.index] = saved - delta
        fm = torch.atleast_1d(loss_fn()).detach()
        ax.tensor[ax.index] = saved
        grads.append((fp - fm) / (2.0 * probe_vox))
        curvs.append((fp + fm - 2.0 * f0) / (probe_vox ** 2))
    grad, curv = torch.stack(grads, dim=1), torch.stack(curvs, dim=1)
    # An axis the data barely constrains (curvature ~ 0, e.g. a shear or scale at a coarse level) must not block convergence for ever:
    # its curvature is floored at ``rel_floor`` x the largest curvature of the frame -- the same saddle-free floor the Newton step
    # uses -- so a flat axis with a negligible gradient reports a negligible distance, and one with a real gradient a large one.
    # A frame with no positive curvature on any axis (a maximum or a plateau everywhere) reports ``inf``.
    top = curv.clamp_min(0.0).max(dim=1, keepdim=True).values
    floor = torch.maximum(top * float(rel_floor), torch.full_like(top, min_curv))
    newton = torch.where(top > min_curv, (grad / torch.maximum(curv, floor)).abs(), torch.full_like(curv, float("inf")))
    return ManifoldProbe(grad=grad, curv=curv, newton_vox=newton, remaining_vox=newton.max(dim=1).values)
