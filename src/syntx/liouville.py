"""
Jacobian determinants of fitted syntx registrations -- the measure used for folding.

``liouville_determinant(result, fixed)`` returns det(D phi) of the deformable forward map of a
registration result (affine excluded) as an ANTsImage on the fixed grid, computed the way each
method builds its map, after the fit, from what the result carries:

- ``tvf`` / ``syngs`` (flows of a velocity field): the product of the step Jacobians
  det(I + dt grad v(phi_k(x))) along the Euler trajectories that produce the exported map -- the
  Jacobian of the map itself (its continuous limit is Liouville's exp int div v dt). A
  finite-difference Jacobian of the exported displacement is unreliable where the map varies
  sharply within a voxel: it reports det <= 0 for maps that are locally invertible.
- ``syn`` (two half transforms meeting at the midpoint, phi = phi_r2l o phi_l2r^-1): the
  finite-difference Jacobian of each half field, combined exactly:
  det phi(x) = det D phi_r2l(y) / det D phi_l2r(y), y = phi_l2r^-1(x). Each half carries half
  the deformation, so its finite difference is far more reliable than that of the composition.
- ``greedy`` (one composed displacement): the finite-difference Jacobian of the full field.
"""

from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

METHODS = ("tvf", "syngs", "syn", "greedy")


# ---------------------------------------------------------------------------------------------
# building blocks for flows
# ---------------------------------------------------------------------------------------------
def velocity_gradient_at(sample: Callable[[torch.Tensor], torch.Tensor], points: torch.Tensor,
                         h: float) -> torch.Tensor:
    """grad v at ``points`` (..., dim) by central differences of ``sample`` (points -> v(points),
    same frame) at +-h along each axis of that frame. Returns (..., dim, dim), [i, k] = dv_i/dx_k.
    """
    dim = points.shape[-1]
    cols = []
    for k in range(dim):
        e = torch.zeros(dim, device=points.device, dtype=points.dtype)
        e[k] = h
        cols.append((sample(points + e) - sample(points - e)) / (2.0 * h))
    return torch.stack(cols, dim=-1)


def step_log_det(grad_v: torch.Tensor, dt: float):
    """(log |det(I + dt grad v)|, 1 where det < 0 else 0) for one Euler step."""
    dim = grad_v.shape[-1]
    eye = torch.eye(dim, device=grad_v.device, dtype=grad_v.dtype)
    d = torch.linalg.det(eye + dt * grad_v)
    return torch.log(d.abs().clamp(min=1e-30)), (d <= 0).to(grad_v.dtype)


@torch.no_grad()
def syngs_jacobian_determinant(model, image_shape=None) -> torch.Tensor:
    """det D phi of syngs' exported forward map (transport mode: Euler steps of the stationary,
    pre-smoothed v0), as the product of the step Jacobians. Shape (1, *spatial)."""
    from .core.grid import sample_field_cf, resize_field
    from .spatial import get_physical_grid_torch, reverse_metadata
    if getattr(model, "transport_mode", "transport") != "transport" or \
            str(getattr(model, "solver", "euler")).lower() != "euler":
        raise NotImplementedError("syngs determinant: transport mode with the Euler solver only")
    target_shape = tuple(image_shape) if image_shape is not None else tuple(model.image_shape)
    device = model.velocity_0_fwd.device
    dtype = model.velocity_0_fwd.dtype
    spacing = list(model.spacing)
    spacing_rev, origin_rev, direction_rev = reverse_metadata(spacing, model.origin, model.direction)
    phys_grid = get_physical_grid_torch(target_shape, spacing, model.origin, model.direction,
                                        device=device, dtype=dtype)
    shape_t = torch.tensor(list(target_shape), device=device, dtype=dtype)
    spacing_t = torch.tensor(spacing_rev, device=device, dtype=dtype)
    origin_t = torch.tensor(origin_rev, device=device, dtype=dtype)
    direction_t = torch.tensor(direction_rev, device=device, dtype=dtype)
    v = model.velocity_0_fwd
    v_up = resize_field(v, target_shape) if tuple(v.shape[1:-1]) != target_shape else v
    v0_cf = torch.movedim(model.apply_green_operator(v_up, target_shape, spacing_rev), -1, 1)
    # physical -> normalised, exactly as GeodesicShootingModel.shoot
    scale_t = 2.0 / (spacing_t * (shape_t - 1.0))
    M = direction_t * scale_t.unsqueeze(0)
    b = -(origin_t @ M) - 1.0
    M_norm, b_norm = torch.flip(M, dims=[1]), torch.flip(b, dims=[0])
    dim = model.dim

    def sample(p):
        return sample_field_cf(v0_cf, (p.view(-1, dim) @ M_norm + b_norm).view(p.shape))

    h = 0.5 * float(spacing_t.min())
    dt = 1.0 / model.n_steps
    phi = phys_grid.clone()
    log_jac = torch.zeros(phi.shape[:-1], device=device, dtype=dtype)
    n_neg = torch.zeros_like(log_jac)
    for _ in range(model.n_steps):
        ld, neg = step_log_det(velocity_gradient_at(sample, phi, h), dt)
        log_jac, n_neg = log_jac + ld, n_neg + neg
        phi = phi + dt * sample(phi)
    # a step with det <= 0 folds the composed map there (even if another step flips it back)
    return torch.where(n_neg > 0, -torch.exp(log_jac), torch.exp(log_jac))


# ---------------------------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------------------------
def _detect_method(result: Dict[str, Any]) -> str:
    model = result.get("model")
    name = type(model).__name__ if model is not None else ""
    return {"TVFModel": "tvf", "GeodesicShootingModel": "syngs", "SyNTo": "syn",
            "GreedyRegistrationModel": "greedy"}.get(name) or \
        str((result.get("provenance") or {}).get("algorithm", "")).replace("syntx.", "")


def _tensor_to_image(arr: np.ndarray, fixed):
    import ants
    arr = np.transpose(arr, tuple(range(arr.ndim))[::-1])          # tensor (z,y,x) -> ANTs (x,y,z)
    return ants.from_numpy(arr.astype(np.float32), origin=fixed.origin, spacing=fixed.spacing,
                           direction=fixed.direction)


def _fd_det(fixed, disp_tensor):
    """Finite-difference Jacobian determinant (ANTs) of a physical displacement tensor
    (1, *spatial, dim) on the fixed grid."""
    import ants
    from .spatial import disp_tensor_to_itk
    return ants.create_jacobian_determinant_image(fixed, disp_tensor_to_itk(disp_tensor, fixed), do_log=False)


def liouville_determinant(result: Dict[str, Any], fixed, method: Optional[str] = None,
                          return_details: bool = False):
    """Jacobian determinant of a registration's deformable forward map, on the fixed grid.

    Parameters
    ----------
    result : dict
        What ``syntx.tvf`` / ``syntx.syngs`` / ``syntx.syn`` / ``syntx.greedy`` returned.
    fixed : ANTsImage
        The fixed image of that registration (grid and geometry of the output).
    method : {'tvf', 'syngs', 'syn', 'greedy'} or None
        None: detected from ``result['model']``.
    return_details : bool
        Also return ``{'measure': ..., 'method': ..., [halves for syn]}``.

    Returns
    -------
    ANTsImage (det, > 0 = locally invertible), or (ANTsImage, dict) with ``return_details``.
    See the module docstring for how each method is handled.
    """
    import ants
    method = (method or _detect_method(result)).lower()
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS} (got {method!r})")
    model = result.get("model")
    details: Dict[str, Any] = {"method": method}

    if method == "tvf":
        det = model.jacobian_determinant(image_shape=model.image_shape)[0].detach().cpu().numpy()
        img = _tensor_to_image(det, fixed)
        details["measure"] = "flow (product of Euler step Jacobians)"
    elif method == "syngs":
        det = syngs_jacobian_determinant(model)[0].detach().cpu().numpy()
        img = _tensor_to_image(det, fixed)
        details["measure"] = "flow (product of Euler step Jacobians)"
    elif method == "syn":
        need = ("midpoint_warp_l2r", "midpoint_warp_r2l", "midpoint_warp_l2r_inv")
        if model is None or not all(hasattr(model, a) for a in need):
            raise ValueError("syn result lacks the half fields (midpoint_warp_l2r / _r2l / _l2r_inv)")
        det_l2r = _fd_det(fixed, model.midpoint_warp_l2r.detach().cpu())
        det_r2l = _fd_det(fixed, model.midpoint_warp_r2l.detach().cpu())
        l2r, r2l = det_l2r.numpy(), det_r2l.numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(l2r != 0, r2l / l2r, -np.inf)
        ratio_img = ants.from_numpy(ratio.astype(np.float32), origin=fixed.origin,
                                    spacing=fixed.spacing, direction=fixed.direction)
        from .spatial import disp_tensor_to_itk
        inv = disp_tensor_to_itk(model.midpoint_warp_l2r_inv.detach().cpu(), fixed)
        import tempfile
        f = tempfile.NamedTemporaryFile(suffix="_l2r_inv_Warp.nii.gz", delete=False).name
        ants.image_write(inv, f)
        img = ants.apply_transforms(fixed=fixed, moving=ratio_img, transformlist=[f], interpolator="linear")
        # a half that folds makes the composition fold wherever it is used
        both_pos = ants.apply_transforms(fixed=fixed, moving=ants.from_numpy(
            ((l2r > 0) & (r2l > 0)).astype(np.float32), origin=fixed.origin, spacing=fixed.spacing,
            direction=fixed.direction), transformlist=[f], interpolator="nearestNeighbor").numpy()
        img = img.new_image_like(np.where(both_pos > 0.5, img.numpy(), -np.abs(img.numpy())))
        details.update(measure="finite difference of each half field", det_l2r=det_l2r, det_r2l=det_r2l)
    else:  # greedy
        warp = next((x for x in result["fwdtransforms"] if isinstance(x, str) and x.endswith(".nii.gz")), None)
        if warp is None:
            raise ValueError("greedy result has no displacement field")
        img = ants.create_jacobian_determinant_image(fixed, warp, do_log=False)
        details["measure"] = "finite difference of the full field"
    return (img, details) if return_details else img


def determinant_summary(det_img, fixed=None) -> Dict[str, float]:
    """min / max / mean / std / folding % (det <= 0) of a determinant image, restricted to the
    fixed image's foreground mask when ``fixed`` is given."""
    import ants
    d = det_img.numpy()
    if fixed is not None:
        m = ants.get_mask(fixed).numpy() > 0
        if m.any():
            d = d[m]
    d = d[np.isfinite(d)] if np.isfinite(d).any() else d
    return {"min": float(d.min()), "max": float(d.max()), "mean": float(d.mean()),
            "std": float(d.std()), "folding_pct": float((d <= 0).mean() * 100.0)}
