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
    """(log |det(I + dt grad v)|, 1 where det < 0 else 0) for one Euler step.

    Uses explicit 2-D / 3-D cofactor expansion via ``_det_identity_plus`` rather than
    ``torch.linalg.det`` so large 3-D volumes on Apple MPS never trigger batched LU
    ``IOGPUDeviceShmem`` exhaustion.
    """
    from .core.jacobian import _det_identity_plus

    d = _det_identity_plus(dt * grad_v)
    return torch.log(d.abs().clamp(min=1e-30)), (d <= 0).to(grad_v.dtype)


def step_log_det_at_norm(
    v_cf: torch.Tensor,
    phi_norm: torch.Tensor,
    M_norm: torch.Tensor,
    h: float,
    dt: float,
    q_buf: Optional[torch.Tensor] = None,
):
    """Fused Euler-step ``(log |det(I + dt grad v)|, neg_flag)`` directly from ``phi_norm``.

    Exploits linearity ``(p +- h e_k) @ M_norm + b_norm = phi_norm +- h M_norm[k]`` to avoid 6
    full-volume ``(N, 3) @ (3, 3)`` matrix multiplications per Euler step, and evaluates the 2-D
    or 3-D determinant directly on channels-first columns without allocating a 6-D ``(..., 3, 3)``
    Jacobian tensor (~453 MB at 192x256x256).
    """
    import torch.nn.functional as F

    dim = phi_norm.shape[-1]
    if q_buf is None or q_buf.shape != phi_norm.shape:
        q_buf = torch.empty_like(phi_norm)
    scale = float(dt) / (2.0 * float(h))
    cols = []
    for k in range(dim):
        delta_k = float(h) * M_norm[k]
        torch.add(phi_norm, delta_k, out=q_buf)
        vp = F.grid_sample(v_cf, q_buf, mode="bilinear", padding_mode="border", align_corners=True)
        torch.sub(phi_norm, delta_k, out=q_buf)
        vm = F.grid_sample(v_cf, q_buf, mode="bilinear", padding_mode="border", align_corners=True)
        col_k = vp.sub_(vm).mul_(scale)
        del vm
        col_k[:, k].add_(1.0)
        cols.append(col_k)

    if dim == 3:
        c0, c1, c2 = cols
        cross_12 = torch.cross(c1, c2, dim=1)
        del c1, c2, cols
        d = (c0 * cross_12).sum(dim=1)
        del c0, cross_12
    elif dim == 2:
        c0, c1 = cols
        d = c0[:, 0] * c1[:, 1] - c0[:, 1] * c1[:, 0]
        del c0, c1, cols
    else:
        grad_mat = torch.stack([torch.movedim(c, 1, -1) for c in cols], dim=-1)
        d = torch.linalg.det(grad_mat)
    return torch.log(d.abs().clamp(min=1e-30)), (d <= 0).to(v_cf.dtype)


@torch.no_grad()
def syngs_jacobian_determinant(model, image_shape=None) -> torch.Tensor:
    """det D phi of syngs' exported forward map (transport mode: Euler steps of the stationary,
    pre-smoothed v0), as the product of the step Jacobians. Shape (1, *spatial)."""
    import torch.nn.functional as F
    from .core.grid import resize_field
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
    v0_cf = torch.movedim(model.apply_green_operator(v_up, target_shape, spacing_rev), -1, 1).contiguous()
    # physical -> normalised, exactly as GeodesicShootingModel.shoot
    scale_t = 2.0 / (spacing_t * (shape_t - 1.0))
    M = direction_t * scale_t.unsqueeze(0)
    b = -(origin_t @ M) - 1.0
    M_norm, b_norm = torch.flip(M, dims=[1]), torch.flip(b, dims=[0])
    dim = model.dim

    h = 0.5 * float(spacing_t.min())
    dt = 1.0 / model.n_steps
    phi = phys_grid.clone()
    q_buf = torch.empty_like(phi)
    log_jac = torch.zeros(phi.shape[:-1], device=device, dtype=dtype)
    n_neg = torch.zeros_like(log_jac)
    for _ in range(model.n_steps):
        phi_norm = (phi.view(-1, dim) @ M_norm + b_norm).view(phi.shape)
        ld, neg = step_log_det_at_norm(v0_cf, phi_norm, M_norm, h, dt, q_buf=q_buf)
        log_jac.add_(ld)
        n_neg.add_(neg)
        del ld, neg
        v_step = torch.movedim(
            F.grid_sample(v0_cf, phi_norm, mode="bilinear", padding_mode="border", align_corners=True),
            1,
            -1,
        )
        del phi_norm
        phi.add_(v_step, alpha=dt)
        del v_step
    del q_buf
    # a step with det <= 0 folds the composed map there (even if another step flips it back)
    return torch.where(n_neg > 0, -torch.exp(log_jac), torch.exp(log_jac))


# ---------------------------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------------------------
def _detect_method(result: Dict[str, Any]) -> str:
    model = result.get("model")
    name = type(model).__name__ if model is not None else ""
    return {
        "TVFModel": "tvf",
        "TVFModelJAX": "tvf",
        "GeodesicShootingModel": "syngs",
        "GeodesicShootingModelJAX": "syngs",
        "SyNTo": "syn",
        "SyNJAX": "syn",
        "GreedyRegistrationModel": "greedy",
    }.get(name) or str((result.get("provenance") or {}).get("algorithm", "")).replace("syntx.", "")


def _to_cpu_tensor(x: Any) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.detach().cpu().float()
    return torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)


def _model_ref_image(model: Any, spatial_shape_tensor: tuple, fixed: Any):
    """Build an ANTsImage matching ``model``'s native physical grid (falling back to ``fixed``)."""
    import ants

    ants_shape = tuple(int(s) for s in reversed(spatial_shape_tensor))
    m_spacing = getattr(model, "spacing", None)
    m_origin = getattr(model, "origin", None)
    m_dir = getattr(model, "direction", None)

    spacing = tuple(float(s) for s in (m_spacing if m_spacing is not None else fixed.spacing))
    origin = tuple(float(o) for o in (m_origin if m_origin is not None else fixed.origin))
    if m_dir is not None:
        d_raw = m_dir.detach().cpu().numpy() if hasattr(m_dir, "detach") else np.asarray(m_dir)
        direction = np.asarray(d_raw, dtype=np.float64)
    else:
        direction = np.asarray(fixed.direction, dtype=np.float64)

    if (
        tuple(fixed.shape) == ants_shape
        and np.allclose(fixed.spacing, spacing, atol=1e-4, rtol=1e-4)
        and np.allclose(fixed.origin, origin, atol=1e-4, rtol=1e-4)
        and np.allclose(fixed.direction, direction, atol=1e-4, rtol=1e-4)
    ):
        return fixed, True

    ref = ants.from_numpy(
        np.zeros(ants_shape, dtype=np.float32),
        origin=origin,
        spacing=spacing,
        direction=direction,
    )
    return ref, False


def _resample_det_to_fixed(det_model_img: Any, fixed: Any, same_grid: bool):
    """Resample ``det_model_img`` onto ``fixed`` in physical space while preserving fold signs."""
    if same_grid:
        return det_model_img
    import ants
    from .transforms import apply_transforms as _apply_transforms

    img_lin = _apply_transforms(
        fixed=fixed,
        moving=det_model_img,
        transformlist=["identity"],
        interpolator="linear",
        defaultvalue=1.0,
    )
    det_np = det_model_img.numpy()
    if bool((det_np <= 0).any()):
        pos_flag = ants.from_numpy(
            (det_np > 0).astype(np.float32),
            origin=det_model_img.origin,
            spacing=det_model_img.spacing,
            direction=det_model_img.direction,
        )
        pos_fix = _apply_transforms(
            fixed=fixed,
            moving=pos_flag,
            transformlist=["identity"],
            interpolator="nearestNeighbor",
            defaultvalue=1.0,
        ).numpy()
        arr_lin = img_lin.numpy()
        img_lin = img_lin.new_image_like(
            np.where(pos_fix > 0.5, np.maximum(arr_lin, 1e-6), -np.abs(arr_lin))
        )
    return img_lin


def _tensor_to_image(arr: np.ndarray, fixed):
    import ants
    arr = np.transpose(arr, tuple(range(arr.ndim))[::-1])          # tensor (z,y,x) -> ANTs (x,y,z)
    return ants.from_numpy(arr.astype(np.float32), origin=fixed.origin, spacing=fixed.spacing,
                           direction=fixed.direction)


def _fd_det(fixed, disp_tensor):
    """Finite-difference Jacobian determinant (ANTs) of a physical displacement tensor
    (1, *spatial, dim) on the fixed grid."""
    from .spatial import disp_tensor_to_itk
    from .transforms import create_jacobian_determinant_image as _create_jacobian_determinant_image
    return _create_jacobian_determinant_image(fixed, disp_tensor_to_itk(disp_tensor, fixed), do_log=False)


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
    import torch.nn.functional as F
    from .transforms import create_jacobian_determinant_image as _create_jacobian_determinant_image
    method = (method or _detect_method(result)).lower()
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS} (got {method!r})")
    model = result.get("model")
    details: Dict[str, Any] = {"method": method}

    if method == "tvf":
        if model is None or not hasattr(model, "jacobian_determinant"):
            raise ValueError("tvf result lacks a flow model with jacobian_determinant")
        det = model.jacobian_determinant(image_shape=model.image_shape)[0].detach().cpu().numpy()
        model_ref, same_grid = _model_ref_image(model, det.shape, fixed)
        img = _resample_det_to_fixed(_tensor_to_image(det, model_ref), fixed, same_grid)
        details["measure"] = "flow (product of Euler step Jacobians)"
    elif method == "syngs":
        if model is None:
            raise ValueError("syngs result lacks a shooting model")
        det = syngs_jacobian_determinant(model)[0].detach().cpu().numpy()
        model_ref, same_grid = _model_ref_image(model, det.shape, fixed)
        img = _resample_det_to_fixed(_tensor_to_image(det, model_ref), fixed, same_grid)
        details["measure"] = "flow (product of Euler step Jacobians)"
    elif method == "syn":
        need = ("midpoint_warp_l2r", "midpoint_warp_r2l", "midpoint_warp_l2r_inv")
        if model is None or not all(hasattr(model, a) for a in need):
            raise ValueError("syn result lacks the half fields (midpoint_warp_l2r / _r2l / _l2r_inv)")
        w_l2r_cpu = _to_cpu_tensor(model.midpoint_warp_l2r)
        w_r2l_cpu = _to_cpu_tensor(model.midpoint_warp_r2l)
        w_l2r_inv_cpu = _to_cpu_tensor(model.midpoint_warp_l2r_inv)

        sp_t = tuple(int(s) for s in w_l2r_cpu.shape[1:-1])
        model_ref, same_grid = _model_ref_image(model, sp_t, fixed)
        det_l2r = _fd_det(model_ref, w_l2r_cpu)
        det_r2l = _fd_det(model_ref, w_r2l_cpu)
        l2r, r2l = det_l2r.numpy(), det_r2l.numpy()

        # Pull back det_r2l(y) and det_l2r(y) at y = X_phys + w_l2r_inv(X_phys) using
        # border clamping, matching syn.py's forward composition (F.grid_sample padding_mode='border')
        # so out-of-FOV boundary pullbacks never inject 0.0 -> -0.0 synthetic folds.
        from .spatial import (
            get_physical_grid_torch,
            physical_to_normalized_torch_cached,
            reverse_metadata,
        )

        from .device import resolve_device

        dim = len(sp_t)
        rev_axes = tuple(range(dim))[::-1]
        n_vox = int(np.prod(sp_t))
        eval_dev = (
            torch.device(resolve_device(None, task="model"))
            if n_vox >= 65_536
            else w_l2r_inv_cpu.device
        )
        l2r_t = torch.from_numpy(np.ascontiguousarray(np.transpose(l2r, rev_axes))).to(eval_dev)
        r2l_t = torch.from_numpy(np.ascontiguousarray(np.transpose(r2l, rev_axes))).to(eval_dev)
        half_min_grid = torch.minimum(l2r_t, r2l_t).unsqueeze(0).unsqueeze(0)
        pair_t = torch.stack([r2l_t, l2r_t], dim=0).unsqueeze(0)
        del l2r_t, r2l_t

        m_spacing = list(model_ref.spacing)
        m_origin = list(model_ref.origin)
        m_dir = np.asarray(model_ref.direction, dtype=np.float64)
        X_phys = get_physical_grid_torch(
            sp_t, m_spacing, m_origin, m_dir, device=eval_dev, dtype=torch.float32
        )
        sp_rev, org_rev, dir_rev = reverse_metadata(m_spacing, m_origin, m_dir)
        coords_norm = physical_to_normalized_torch_cached(
            X_phys.add_(w_l2r_inv_cpu.to(eval_dev)),
            torch.tensor(list(sp_t), device=eval_dev, dtype=torch.float32),
            torch.tensor(sp_rev, device=eval_dev, dtype=torch.float32),
            torch.tensor(org_rev, device=eval_dev, dtype=torch.float32),
            torch.tensor(dir_rev, device=eval_dev, dtype=torch.float32),
        )
        del X_phys

        sampled_pair = F.grid_sample(pair_t, coords_norm, mode="bilinear", padding_mode="border", align_corners=True)
        del pair_t
        r2l_y = sampled_pair[0, 0]
        l2r_y = sampled_pair[0, 1]
        half_min_y = F.grid_sample(half_min_grid, coords_norm, mode="nearest", padding_mode="border", align_corners=True)[0, 0]
        del coords_norm, half_min_grid

        valid_pos = (half_min_y > 0.0) & (l2r_y > 1e-4)
        if bool((~valid_pos).any().item()):
            # Where a half folds or l2r -> 0, the inverse-function ratio r2l / l2r diverges;
            # evaluate the signed chain-rule product det D phi_r2l(y) * det D phi_l2r^-1(x)
            # bounded above by the folded half determinant (<= 0) without negating positive expansions.
            det_l2r_inv = _fd_det(model_ref, w_l2r_inv_cpu)
            l2r_inv_t = torch.from_numpy(np.ascontiguousarray(np.transpose(det_l2r_inv.numpy(), rev_axes))).to(eval_dev)
            chain_det = r2l_y * l2r_inv_t
            folded_val = torch.minimum(
                chain_det, torch.minimum(half_min_y, torch.minimum(l2r_y, r2l_y))
            ).clamp(max=0.0)
            det_t = torch.where(valid_pos, r2l_y / l2r_y.clamp(min=1e-4), folded_val)
        else:
            det_t = r2l_y / l2r_y
        del sampled_pair

        det_ants = np.transpose(det_t.detach().cpu().numpy(), rev_axes).astype(np.float32)
        img_model = ants.from_numpy(det_ants, origin=model_ref.origin, spacing=model_ref.spacing, direction=model_ref.direction)
        img = _resample_det_to_fixed(img_model, fixed, same_grid)
        details.update(measure="finite difference of each half field", det_l2r=det_l2r, det_r2l=det_r2l)
    else:  # greedy
        warp = next((x for x in result["fwdtransforms"] if isinstance(x, str) and x.endswith(".nii.gz")), None)
        if warp is None:
            raise ValueError("greedy result has no displacement field")
        img = _create_jacobian_determinant_image(fixed, warp, do_log=False)
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
