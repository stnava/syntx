"""Population-scale GPU cohort registration primitives: ``FixedTarget``, ``batch_robust_affine``, and ``batch_greedy``.

Eliminates per-subject fixed-template H2D transfers, pyramid/LNCC rebuilds, post-solve D2H
materialization, and `/tmp` gzip serialization while saturating GPU compute across multi-subject
batches (`B = 2..16`) with exact per-subject physical-space and step-norm invariance.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import ants
import numpy as np
import torch
import torch.nn.functional as F

from .benchmark.evaluate import normalize_intensity
from .core.convergence import DEFAULT_SLOPE_THRESHOLD
from .core.initial_warp import resolve_initial_transform_and_warp
from .core.inverse import _LazyKeyDict, calculate_inverse_identity_error, update_inverse_field_nd_anderson
from .core.losses import parse_similarity_metric
from .core.losses.cross_correlation import _box_pool
from .core.losses.mutual_information import (
    batched_mattes_mi_loss_shared_fixed,
    mattes_sample_indices,
    parzen_weights,
)
from .core.optimizers import validate_regularizer_optimizer_pair
from .core.pipeline import _same_device, cleanup_gpu, is_unclipped_unit_image
from .core.regularizers import REGULARIZER_EXTRA_KEYS, list_regularizers
from .core.utils import normalize_image, require_finite_images
from .device import resolve_device
from .greedy import GreedyRegistrationModel, _build_torch_affine_matrix
from .pyramid import build_pyramid
from .robust_affine import (
    ANISOTROPY_RESAMPLE_RATIO,
    RobustAffineSolverError,
    _analyze_volume_support_3d,
    _build_schedule_from_levels,
    _cluster_candidates_se3,
    _default_affine_schedule,
    _phase_correlation_translation_candidates,
    _spacing_ratio,
    compute_center_of_mass,
    compute_fov_center,
    robust_affine,
    robust_center_of_mass,
)
from .spatial import (
    disp_itk_to_tensor,
    disp_tensor_to_itk,
    export_ants_affine_transform,
    image_to_tensor,
    normalized_to_physical_disp,
)
from .transforms import (
    ContinuousField,
    TransformChain,
    TransformElement,
    _cache_disp_image_and_field,
    _write_ants_images_fast,
    apply_transforms as _apply_transforms,
)


def _select_cohort_batch_size(
    batch_size: Union[int, str],
    n_subjects: int,
    n_voxels: int,
    *,
    stage: str = "deformable",
) -> int:
    if n_subjects <= 1:
        return 1
    if isinstance(batch_size, (int, np.integer)):
        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be >= 1 or 'auto', got {batch_size!r}")
        return min(int(batch_size), n_subjects)
    if str(batch_size).lower() != "auto":
        raise ValueError(f"batch_size must be a positive int or 'auto', got {batch_size!r}")
    if stage == "affine":
        if n_voxels > 16_777_216:
            return min(2, n_subjects)
        if n_voxels >= 8_000_000:
            return min(8, n_subjects)
        return min(16, n_subjects)
    if n_voxels > 16_777_216:
        return 1
    if n_voxels >= 8_000_000:
        return min(4, n_subjects)
    if n_voxels >= 2_000_000:
        return min(6, n_subjects)
    return min(8, n_subjects)


class FixedTarget:
    """GPU-pinned reference/atlas target reusable across `N`-subject cohort registrations and `build_template`.

    Precomputes and retains on GPU:
    - foreground-normalized fixed `ANTsImage` and `(1, 1, *spatial_zyx)` float32 tensor,
    - `robust_affine` percentile-normalized tensor, physical CoM, FOV center, and 3-D volume support,
    - multi-scale deformable pyramid levels and per-level fixed LNCC mean-subtracted / variance buffers,
    - multi-scale affine pyramid levels and physical voxel coordinate grids.
    """

    _is_syntx_fixed_target: bool = True

    def __init__(
        self,
        *,
        image: ants.ANTsImage,
        norm_image: ants.ANTsImage,
        tensor_zyx: torch.Tensor,
        affine_norm_image: ants.ANTsImage,
        affine_tensor_zyx: torch.Tensor,
        affine_support: Dict[str, Any],
        is_unclipped_unit: bool,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.image = image
        self.norm_image = norm_image
        self.tensor_zyx = tensor_zyx
        self.affine_norm_image = affine_norm_image
        self.affine_tensor_zyx = affine_tensor_zyx
        self.affine_support = affine_support
        self.is_unclipped_unit = bool(is_unclipped_unit)
        self.device = device
        self.dtype = dtype
        self.dimension = int(image.dimension)
        self.shape = tuple(int(s) for s in image.shape[: self.dimension])
        self.spacing = tuple(float(s) for s in image.spacing[: self.dimension])
        self.origin = tuple(float(o) for o in image.origin[: self.dimension])
        self.direction = np.asarray(image.direction, dtype=np.float64)[: self.dimension, : self.dimension]
        self._deformable_cache: Dict[Tuple[Tuple[int, ...], int, bool], Tuple[List[Any], List[Any]]] = {}
        self._affine_level_cache: Dict[Tuple[int, bool, float, int, int, float], Tuple[Any, Any]] = {}
        self._affine_subset_cache: Dict[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @classmethod
    def from_image(
        cls,
        image: ants.ANTsImage,
        *,
        scales: Optional[Sequence[int]] = (4, 2, 1),
        window_size: int = 5,
        device: Union[str, torch.device, None] = "auto",
        dtype: torch.dtype = torch.float32,
    ) -> "FixedTarget":
        require_finite_images(fixed=image)
        dev_req = None if (device is None or str(device).lower() == "auto") else str(device)
        dev = torch.device(resolve_device(dev_req, task="reg"))
        dim = int(image.dimension)
        rev_sp = tuple(range(dim - 1, -1, -1))
        is_unit = is_unclipped_unit_image(image)

        norm_img = normalize_intensity(image)
        t_zyx = (
            torch.from_numpy(np.ascontiguousarray(norm_img.numpy(), dtype=np.float32))
            .to(device=dev, dtype=dtype)
            .permute(*rev_sp)
            .contiguous()
            .unsqueeze(0)
            .unsqueeze(0)
        )

        aff_norm = normalize_image(image, method="auto", force=True)
        aff_t_zyx = image_to_tensor(aff_norm, device=dev, to_zyx=True).to(dtype=dtype)

        com_f_pos, com_f_neg = robust_center_of_mass(aff_norm, weighted=True)
        com_f = np.asarray(
            com_f_pos
            if com_f_pos is not None
            else (com_f_neg if com_f_neg is not None else compute_center_of_mass(aff_norm, weighted=False)),
            dtype=np.float64,
        )
        fov_f = np.asarray(compute_fov_center(aff_norm), dtype=np.float64)
        ext_f_world = np.abs(np.asarray(image.direction, dtype=np.float64).reshape(dim, dim)) @ (
            np.asarray(image.shape[:dim], dtype=np.float64) * np.asarray(image.spacing[:dim], dtype=np.float64)
        )
        support_f = _analyze_volume_support_3d(image, aff_norm) if dim == 3 else None
        aniso_f = bool(dim == 3 and _spacing_ratio(image.spacing) > ANISOTROPY_RESAMPLE_RATIO)
        aff_sup = {
            "com_f_pos": com_f_pos,
            "com_f_neg": com_f_neg,
            "com_f": com_f,
            "fov_f": fov_f,
            "ext_f_world": ext_f_world,
            "support_f": support_f,
            "aniso_f": aniso_f,
        }

        obj = cls(
            image=image,
            norm_image=norm_img,
            tensor_zyx=t_zyx,
            affine_norm_image=aff_norm,
            affine_tensor_zyx=aff_t_zyx,
            affine_support=aff_sup,
            is_unclipped_unit=is_unit,
            device=dev,
            dtype=dtype,
        )
        if scales is not None and len(scales) > 0:
            obj.get_deformable_pyramid(scales=scales, window_size=window_size, use_fast_lncc=True)
        return obj

    def get_deformable_pyramid(
        self,
        scales: Sequence[int],
        window_size: int = 5,
        use_fast_lncc: bool = True,
    ) -> Tuple[List[Any], List[Any]]:
        sc_tup = tuple(int(s) for s in scales)
        ws_int = int(window_size)
        key = (sc_tup, ws_int, bool(use_fast_lncc))
        cached = self._deformable_cache.get(key)
        if cached is not None:
            return cached
        fixed_lv = build_pyramid(
            self.tensor_zyx,
            self.spacing,
            list(sc_tup),
            "deformable",
            origin=self.origin,
            direction=self.direction,
        )
        lncc_list: List[Any] = []
        with torch.no_grad():
            for lv in fixed_lv:
                fi_down = lv.tensor
                size_down = list(lv.shape)
                if use_fast_lncc and int(fi_down[0, 0].numel()) <= 16_777_216:
                    ws_lvl = min(ws_int, int(min(size_down)))
                    if ws_lvl % 2 == 0:
                        ws_lvl = max(1, ws_lvl - 1)
                    fi_f = fi_down.float() if fi_down.dtype in (torch.float16, torch.bfloat16) else fi_down
                    j_mean = _box_pool(fi_f, ws_lvl, count_include_pad=False)
                    m_c = fi_f - j_mean
                    del j_mean
                    j_var = torch.clamp(_box_pool(m_c.square(), ws_lvl, count_include_pad=False), min=0.0)
                    safe_j_var = torch.clamp(j_var, min=1e-6)
                    del j_var
                    lncc_list.append((ws_lvl, m_c, safe_j_var))
                else:
                    lncc_list.append(None)
        out = (fixed_lv, lncc_list)
        self._deformable_cache[key] = out
        return out

    def get_affine_level(
        self,
        *,
        level: int,
        aniso_any: bool,
        smooth_sigma_per_level: float,
        max_pyramid_voxels: int,
        valid_margin_slices: int,
        fg_level: float,
        device: torch.device,
    ) -> Tuple[Any, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        key = (
            int(level),
            bool(aniso_any),
            round(float(smooth_sigma_per_level), 6),
            int(max_pyramid_voxels),
            int(valid_margin_slices),
            round(float(fg_level), 6),
        )
        if _same_device(device, self.device) and key in self._affine_level_cache:
            return self._affine_level_cache[key]

        fi_arr = self.affine_tensor_zyx if _same_device(device, self.device) else self.affine_tensor_zyx.to(device=device)
        sp_min_f = min(self.spacing)
        sig_mm = float(smooth_sigma_per_level) * max(int(level) - 1, 0) * sp_min_f
        kw = dict(origin=self.origin, direction=self.direction)
        if aniso_any:
            lf = build_pyramid(
                fi_arr,
                self.spacing,
                [int(level)],
                policy="mm_iso",
                sigma=sig_mm,
                sigma_mode="physical",
                reference_spacing=self.spacing,
                max_voxels=int(max_pyramid_voxels),
                valid_margin=int(valid_margin_slices),
                **kw,
            )[0]
        else:
            lf = build_pyramid(
                fi_arr,
                self.spacing,
                [int(level)],
                policy="mm_iso",
                sigma=float(smooth_sigma_per_level) * max(int(level) - 1, 0),
                **kw,
            )[0]
        dim = self.dimension
        f32 = dict(dtype=torch.float32, device=device)
        sp_xyz = torch.tensor(self.spacing, **f32)
        orig_xyz = torch.tensor(self.origin, **f32)
        dir_xyz = torch.tensor(self.direction, **f32)
        axes = [torch.arange(n, **f32) for n in lf.shape]
        mesh = torch.meshgrid(*axes, indexing="ij")
        f_t = torch.tensor(list(lf.f), **f32)
        o_t = torch.tensor(list(lf.off), **f32)
        vox_xyz = torch.stack(list(reversed(mesh)), dim=-1).reshape(-1, dim) * f_t + o_t
        phys_xyz = orig_xyz + (vox_xyz * sp_xyz) @ dir_xyz.t()
        fmask = lf.tensor > float(fg_level)
        valid_lev = lf.valid.to(device=device, dtype=torch.bool).reshape(fmask.shape) if lf.valid is not None else None
        geom = (vox_xyz, phys_xyz, fmask, valid_lev)
        if _same_device(device, self.device):
            self._affine_level_cache[key] = (lf, geom)
        return lf, geom


def _batched_rodrigues_3d(omega: torch.Tensor) -> torch.Tensor:
    """Differentiable batched Rodrigues ``so(3) -> SO(3)`` for ``omega`` ``(M, 3) -> (M, 3, 3)``."""
    theta_sq = (omega * omega).sum(dim=-1)
    small = theta_sq < 1e-8
    theta_sq_safe = torch.where(small, torch.ones_like(theta_sq), theta_sq)
    theta_safe = torch.sqrt(theta_sq_safe)
    c1 = torch.where(small, 1.0 - theta_sq * (1.0 / 6.0), torch.sin(theta_safe) / theta_safe)
    c2 = torch.where(small, 0.5 - theta_sq * (1.0 / 24.0), (1.0 - torch.cos(theta_safe)) / theta_sq_safe)

    z = torch.zeros_like(theta_sq)
    w0, w1, w2 = omega[:, 0], omega[:, 1], omega[:, 2]
    K = torch.stack(
        [
            torch.stack([z, -w2, w1], dim=-1),
            torch.stack([w2, z, -w0], dim=-1),
            torch.stack([-w1, w0, z], dim=-1),
        ],
        dim=-2,
    )
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0)
    return eye + c1[:, None, None] * K + c2[:, None, None] * torch.bmm(K, K)


def _moving_level_norm_affine_3d(
    origin: Sequence[float],
    direction: np.ndarray,
    native_spacing: Sequence[float],
    f_xyz: Sequence[float],
    off_xyz: Sequence[float],
    shape_zyx: Sequence[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(P_mov, q_mov)`` (`(3, 3)` and `(3,)`) such that ``y_norm = y_phys @ P_mov + q_mov``
    matches ``pyramid.prebuild_warp`` (`align_corners=True`)."""
    dir_inv_t = np.linalg.inv(np.asarray(direction, dtype=np.float64)[:3, :3]).T
    sp = np.asarray(native_spacing[:3], dtype=np.float64)
    f_arr = np.asarray(f_xyz[:3], dtype=np.float64)
    off_arr = np.asarray(off_xyz[:3], dtype=np.float64)
    den = np.maximum(np.asarray(shape_zyx[::-1], dtype=np.float64) - 1.0, 1e-8)
    scale = 2.0 / (sp * f_arr * den)
    p_np = (dir_inv_t * scale[None, :]).astype(np.float32)
    o_np = np.asarray(origin[:3], dtype=np.float64)
    q_np = (-(o_np @ (dir_inv_t * scale[None, :])) - (2.0 * off_arr / (f_arr * den)) - 1.0).astype(np.float32)
    return torch.as_tensor(p_np, device=device), torch.as_tensor(q_np, device=device)


class _BatchedAffinePoses3D:
    """Vectorized ``M``-pose 3-D Lie-algebra parameterization matching ``robust_affine._AffinePath``."""

    def __init__(
        self,
        B_init: torch.Tensor,
        t_init: torch.Tensor,
        subj_idx: torch.Tensor,
        expected_ov: torch.Tensor,
        w_scale: torch.Tensor,
        w_shear: torch.Tensor,
    ) -> None:
        dev = B_init.device
        m_sz = int(B_init.shape[0])
        self.B = B_init.detach().to(torch.float32)
        self.subj_idx = subj_idx.to(device=dev, dtype=torch.long)
        self.expected_ov = expected_ov.to(device=dev, dtype=torch.float32)
        self.w_scale = w_scale.to(device=dev, dtype=torch.float32)
        self.w_shear = w_shear.to(device=dev, dtype=torch.float32)
        self.t = t_init.detach().to(device=dev, dtype=torch.float32).clone().requires_grad_(True)
        self.omega = torch.zeros((m_sz, 3), device=dev, dtype=torch.float32, requires_grad=True)
        self.scale = torch.zeros((m_sz, 3), device=dev, dtype=torch.float32, requires_grad=True)
        self.shear = torch.zeros((m_sz, 3), device=dev, dtype=torch.float32, requires_grad=True)

    def matrices(self, dof: str) -> torch.Tensor:
        R = _batched_rodrigues_3d(self.omega)
        A = torch.bmm(R, self.B)
        if dof == "affine":
            s_vec = torch.exp(torch.clamp(self.scale, -0.4, 0.4))
            A_scaled = A * s_vec.unsqueeze(1)
            z = torch.zeros_like(self.shear[:, 0])
            one = torch.ones_like(z)
            sh0, sh1, sh2 = self.shear[:, 0], self.shear[:, 1], self.shear[:, 2]
            Sh = torch.stack(
                [
                    torch.stack([one, sh0, sh1], dim=-1),
                    torch.stack([z, one, sh2], dim=-1),
                    torch.stack([z, z, one], dim=-1),
                ],
                dim=-2,
            )
            A = torch.bmm(A_scaled, Sh)
        return A

    def reg_losses(self, dof: str, lambda_shear: float, lambda_scale: float) -> torch.Tensor:
        if dof != "affine" or (lambda_shear <= 0.0 and lambda_scale <= 0.0):
            return torch.zeros(int(self.t.shape[0]), device=self.t.device, dtype=torch.float32)
        return lambda_scale * (self.w_scale.unsqueeze(0) * self.scale.square()).sum(dim=-1) + lambda_shear * (
            self.w_shear.unsqueeze(0) * self.shear.square()
        ).sum(dim=-1)

    @torch.no_grad()
    def clamp_(self) -> None:
        self.scale.clamp_(-0.35, 0.35)
        self.shear.clamp_(-0.35, 0.35)
        self.omega.clamp_(-math.pi / 3.0, math.pi / 3.0)

    @torch.no_grad()
    def select_indices(self, keep_idx: torch.Tensor) -> None:
        idx = keep_idx.to(device=self.t.device, dtype=torch.long)
        self.B = self.B[idx].contiguous()
        self.subj_idx = self.subj_idx[idx].contiguous()
        self.expected_ov = self.expected_ov[idx].contiguous()
        self.t = self.t.detach()[idx].contiguous().requires_grad_(True)
        self.omega = self.omega.detach()[idx].contiguous().requires_grad_(True)
        self.scale = self.scale.detach()[idx].contiguous().requires_grad_(True)
        self.shear = self.shear.detach()[idx].contiguous().requires_grad_(True)


def _eval_batched_poses_3d(
    A_m: torch.Tensor,
    t_m: torch.Tensor,
    subj_idx: torch.Tensor,
    expected_ov_m: torch.Tensor,
    C_phys: torch.Tensor,
    phys_p: torch.Tensor,
    fw: torch.Tensor,
    mi_lev_list: List[torch.Tensor],
    P_mov_b: torch.Tensor,
    q_mov_b: torch.Tensor,
    same_mov_shape: bool,
    mi_lev_stacked: Optional[torch.Tensor],
    num_bins: int = 32,
    fixed_range: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """Evaluate per-pose Mattes MI + overlap guard for ``M`` poses across ``B`` subjects simultaneously."""
    m_sz = int(A_m.shape[0])
    n_pts = int(phys_p.shape[0])
    b_sz = len(mi_lev_list)

    teff = t_m + C_phys.unsqueeze(0) - torch.matmul(A_m, C_phys)
    P_sel = P_mov_b[subj_idx]
    q_sel = q_mov_b[subj_idx]
    M_tot = torch.bmm(A_m.transpose(1, 2), P_sel)
    b_tot = torch.bmm(teff.unsqueeze(1), P_sel).squeeze(1) + q_sel

    y_all = torch.matmul(phys_p.unsqueeze(0), M_tot) + b_tot.unsqueeze(1)  # (M, P, 3)

    if same_mov_shape and mi_lev_stacked is not None and m_sz % b_sz == 0:
        k_per = m_sz // b_sz
        # Check subj_idx is contiguous blocks [0..0, 1..1, ...]
        grid_bkp = y_all.view(b_sz, 1, k_per, n_pts, 3)
        w_all = F.grid_sample(
            mi_lev_stacked, grid_bkp, mode="bilinear", padding_mode="zeros", align_corners=True
        ).reshape(m_sz, n_pts)
    else:
        w_parts: List[torch.Tensor] = []
        for b in range(b_sz):
            sel_b = subj_idx == b
            y_b = y_all[sel_b]
            k_b = int(y_b.shape[0])
            if k_b == 0:
                continue
            grid_b = y_b.view(1, 1, k_b, n_pts, 3)
            wb = F.grid_sample(
                mi_lev_list[b], grid_b, mode="bilinear", padding_mode="zeros", align_corners=True
            ).view(k_b, n_pts)
            w_parts.append((sel_b, wb))
        if len(w_parts) == 1 and bool(w_parts[0][0].all()):
            w_all = w_parts[0][1]
        else:
            w_all = torch.empty((m_sz, n_pts), device=y_all.device, dtype=y_all.dtype)
            for sel_b, wb in w_parts:
                w_all[sel_b] = wb

    mi_losses = batched_mattes_mi_loss_shared_fixed(w_all, fw, num_bins=num_bins, fixed_range=fixed_range)
    max_abs_y = y_all.abs().amax(dim=-1)
    smooth_in = torch.sigmoid(20.0 * (1.02 - max_abs_y)).mean(dim=-1)
    ov_req = 0.65 * expected_ov_m
    ov_deficit = F.relu(ov_req - smooth_in)
    has_def = (ov_deficit.detach() > 0.0).to(mi_losses.dtype)
    ov_pen = 1.8 * ov_deficit + (0.05 * has_def) * F.relu(max_abs_y - 1.0).mean(dim=-1)
    return mi_losses + ov_pen


def batch_robust_affine(
    fixed: Union[ants.ANTsImage, FixedTarget],
    moving_list: Sequence[ants.ANTsImage],
    *,
    dof: str = "affine",
    mode: str = "auto",
    preset: str = "default",
    batch_size: Union[int, str] = "auto",
    n_starts: int = 3,
    multi_start: bool = True,
    cone_angles_deg: Optional[List[float]] = None,
    levels: Optional[Sequence[int]] = None,
    iterations: Optional[Union[int, Sequence[int]]] = None,
    sampling_percentage: Optional[float] = None,
    num_bins: int = 32,
    lambda_shear: float = 0.02,
    lambda_scale: float = 0.01,
    cluster_threshold: float = 0.35,
    device: Union[str, torch.device, None] = "auto",
    seed: int = 42,
    return_warped: bool = False,
    write_transforms: bool = False,
    outprefixes: Optional[Sequence[Optional[str]]] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Vectorized multi-subject GPU Mattes-MI affine/rigid registration onto a shared ``fixed`` target.

    Evaluates coarse start candidates and multi-start Lie-algebra optimization across ``B`` subjects
    simultaneously against precomputed fixed-template Parzen histograms, automatically delegating
    2-D, non-PyTorch, or slab-truncated subjects to ``syntx.robust_affine``.
    """
    mov_seq = list(moving_list)
    n_subj = len(mov_seq)
    if n_subj == 0:
        return []
    if outprefixes is not None and len(outprefixes) != n_subj:
        raise ValueError(f"len(outprefixes)={len(outprefixes)} must match len(moving_list)={n_subj}")

    dev_req = None if (device is None or str(device).lower() == "auto") else str(device)
    dev_str = resolve_device(dev_req, task="affine")
    dev_obj = torch.device(dev_str)

    target = (
        fixed
        if getattr(fixed, "_is_syntx_fixed_target", False)
        else FixedTarget.from_image(fixed, scales=None, device=dev_obj)
    )
    fixed_img = target.image
    dim = target.dimension
    for m_img in mov_seq:
        require_finite_images(moving=m_img)

    # Delegate 2-D, non-PyTorch modes, rigid L-BFGS schedules, or exotic kwargs to FixedTarget-accelerated robust_affine
    use_vectorized_3d = (
        dim == 3
        and mode in ("auto", "pytorch", "gpu", "pytorch_gpu")
        and not kwargs
        and not target.affine_support["aniso_f"]
        and not bool(target.affine_support["support_f"] and target.affine_support["support_f"]["is_internal_fov"])
    )
    if not use_vectorized_3d:
        out_fallback: List[Dict[str, Any]] = []
        for idx_s, m_img in enumerate(mov_seq):
            op = outprefixes[idx_s] if outprefixes is not None else None
            do_w = bool(write_transforms or op is not None)
            res_i = robust_affine(
                target,
                m_img,
                dof=dof,
                mode=mode,
                preset=preset,
                n_starts=n_starts,
                multi_start=multi_start,
                cone_angles_deg=cone_angles_deg,
                levels=levels,
                iterations=iterations,
                sampling_percentage=sampling_percentage,
                num_bins=num_bins,
                lambda_shear=lambda_shear,
                lambda_scale=lambda_scale,
                cluster_threshold=cluster_threshold,
                device=dev_str,
                seed=seed,
                return_warped=return_warped,
                write_transforms=do_w,
                verbose=verbose,
                **kwargs,
            )
            if op is not None:
                saved = res_i["fwd_chain"].save(op, compresslevel=4)
                res_i["fwdtransforms"] = saved
                res_i["invtransforms"] = saved
            out_fallback.append(res_i)
        return out_fallback

    # Build schedule (using pure Adam blocks for vectorized multi-subject execution; rigid L1 uses 40 Adam steps)
    if levels is not None:
        schedule = _build_schedule_from_levels(levels, iterations=iterations, dof=dof, preset=preset)
    elif dof == "rigid":
        schedule = [
            dict(level=4, iters=50, dof="rigid", step=(0.50, 0.24, 0.0, 0.0), eta_frac=0.05, select=True, full_grid=True, sampling=1.0),
            dict(level=2, iters=80, dof="rigid", step=(0.15, 0.15, 0.0, 0.0), eta_frac=0.067, select=False, full_grid=True, sampling=0.10),
            dict(level=1, iters=40, dof="rigid", step=(0.03, 0.06, 0.0, 0.0), eta_frac=0.02, select=False, full_grid=True, sampling=0.02),
        ]
        if iterations is not None:
            it_l = [int(iterations)] * len(schedule) if isinstance(iterations, (int, np.integer)) else [int(x) for x in iterations]
            for s_d, it_v in zip(schedule, it_l):
                s_d["iters"] = it_v
    else:
        schedule = _default_affine_schedule(3, preset, dof="affine")
        if iterations is not None:
            it_l = [int(iterations)] * len(schedule) if isinstance(iterations, (int, np.integer)) else [int(x) for x in iterations]
            for s_d, it_v in zip(schedule, it_l):
                s_d["iters"] = it_v

    min_shape = min(int(s) for s in fixed_img.shape[:3])
    max_level = max(1, min_shape // 8)
    if any(int(s_["level"]) > max_level for s_ in schedule):
        clamped = []
        for s_ in schedule:
            orig_l = max(1, int(s_["level"]))
            new_l = min(orig_l, max_level)
            ns_ = dict(s_, level=new_l)
            if new_l < orig_l and ns_.get("step") is not None:
                st = list(ns_["step"])
                st[0] = float(st[0]) * (float(new_l) / float(orig_l))
                ns_["step"] = tuple(st)
            clamped.append(ns_)
        dedup = []
        for s_ in clamped:
            if dedup and int(dedup[-1]["level"]) == int(s_["level"]) and dedup[-1].get("dof") == s_.get("dof"):
                merged = dict(s_)
                merged["select"] = bool(dedup[-1].get("select", False) or s_.get("select", False))
                dedup[-1] = merged
            else:
                dedup.append(s_)
        schedule = dedup

    if sampling_percentage is None:
        sampling_percentage = 0.01 if preset == "fast" else (0.20 if preset == "accurate" else 0.02)

    sched_levels = sorted({int(s_["level"]) for s_ in schedule}, reverse=True)
    coarse = sched_levels[0]
    sp_min_f = min(target.spacing)
    coarse_mm = float(coarse) * sp_min_f
    domain_diam = float(np.linalg.norm(np.asarray(target.spacing) * np.asarray(target.shape)))
    trans_basin_diam = min(domain_diam, max(80.0, 8.0 * coarse_mm))

    com_f = target.affine_support["com_f"]
    fov_f = target.affine_support["fov_f"]
    ext_f_world = target.affine_support["ext_f_world"]
    support_f = target.affine_support["support_f"]
    C_phys = torch.tensor(com_f, dtype=torch.float32, device=dev_obj)

    sp_f_arr = np.asarray(target.spacing, dtype=np.float32)
    w_sc_np = sp_f_arr / max(float(sp_f_arr.min()), 1e-4)
    w_scale_t = torch.tensor(w_sc_np, dtype=torch.float32, device=dev_obj)
    w_shear_t = torch.tensor(
        [w_sc_np[0] * w_sc_np[1], w_sc_np[0] * w_sc_np[2], w_sc_np[1] * w_sc_np[2]],
        dtype=torch.float32,
        device=dev_obj,
    )
    is_thin_slab_f = min(float(n) * float(s) for n, s in zip(target.shape, target.spacing)) < 90.0 or min(target.shape) < 24
    eff_l_shear = float(lambda_shear) if (is_thin_slab_f or lambda_shear != 0.02) else 0.0
    eff_l_scale = float(lambda_scale) if (is_thin_slab_f or lambda_scale != 0.01) else 0.0

    # Precompute fixed level subsets, Parzen weights, and foreground lever-arm parameter scales ONCE
    fix_lvl_data: Dict[int, Dict[str, Any]] = {}
    for lvl in sched_levels:
        lf, (_, phys_xyz, _, _) = target.get_affine_level(
            level=lvl,
            aniso_any=False,
            smooth_sigma_per_level=0.0,
            max_pyramid_voxels=2**24,
            valid_margin_slices=1,
            fg_level=0.01,
            device=dev_obj,
        )
        fvals_full = lf.tensor.reshape(-1)
        fg_sel = fvals_full > 0.01
        pts_fg = phys_xyz[fg_sel] if int(fg_sel.sum().item()) >= 64 else phys_xyz
        pts_chk = pts_fg[:: max(1, int(pts_fg.shape[0]) // 2000)]
        rel_sq = (pts_chk - C_phys).pow(2)
        col_ms = rel_sq.mean(dim=0)
        rot_ms = torch.stack([col_ms[1] + col_ms[2], col_ms[0] + col_ms[2], col_ms[0] + col_ms[1]])
        sc_vals = [float(v) for v in torch.cat([rot_ms.sqrt(), col_ms.sqrt()]).tolist()]
        r_sc, c_sc = sc_vals[:3], sc_vals[3:]
        groups_scale = (
            1.0,
            float(np.mean(r_sc)),
            float(np.mean(c_sc)),
            float(np.mean([c_sc[1], c_sc[2], c_sc[2]])),
        )
        fix_lvl_data[lvl] = {
            "phys_full": phys_xyz,
            "fvals_full": fvals_full,
            "groups_scale": groups_scale,
            "subsets": {},
        }

    def _get_fix_subset(lvl: int, samp: Optional[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        d_lvl = fix_lvl_data[lvl]
        fv_full = d_lvl["fvals_full"]
        n_tot = int(fv_full.numel())
        if samp is not None and float(samp) < 1.0:
            eff_s = min(1.0, max(float(samp), 1000.0 / max(n_tot, 1)))
            k_key = max(1, round(float(eff_s) * n_tot)) if eff_s < 1.0 else n_tot
        else:
            eff_s = 1.0
            k_key = n_tot
        cached_sub = d_lvl["subsets"].get(k_key)
        if cached_sub is not None:
            return cached_sub
        if k_key < n_tot:
            idx = mattes_sample_indices(n_tot, eff_s, dev_obj)
            phys_p = d_lvl["phys_full"][idx]
            fv = fv_full[idx]
        else:
            phys_p = d_lvl["phys_full"]
            fv = fv_full
        with torch.no_grad():
            y_sc = fv.to(torch.float32) * 2.0 - 1.0
            fw = parzen_weights(y_sc, num_bins=num_bins, min_val=-1.0, max_val=1.0, pad=2.0)
        d_lvl["subsets"][k_key] = (phys_p, fw)
        return phys_p, fw

    cone_degs = list(cone_angles_deg) if cone_angles_deg is not None else [-24.0, -18.0, -12.0, -6.0, 6.0, 12.0, 18.0, 24.0]
    cone_rots: List[Tuple[str, np.ndarray]] = []
    for deg in cone_degs:
        if abs(deg) < 1e-3:
            continue
        rad = math.radians(float(deg))
        c_ang, s_ang = math.cos(rad), math.sin(rad)
        r_y = np.eye(3, dtype=np.float64)
        r_y[0, 0] = c_ang
        r_y[0, 2] = s_ang
        r_y[2, 0] = -s_ang
        r_y[2, 2] = c_ang
        cone_rots.append((f"pitch_{deg:+.0f}deg", np.array([[1.0, 0.0, 0.0], [0.0, c_ang, -s_ang], [0.0, s_ang, c_ang]])))
        cone_rots.append((f"roll_{deg:+.0f}deg", r_y))
        cone_rots.append((f"yaw_{deg:+.0f}deg", np.array([[c_ang, -s_ang, 0.0], [s_ang, c_ang, 0.0], [0.0, 0.0, 1.0]])))

    n_vox = int(np.prod(target.shape))
    eff_b = _select_cohort_batch_size(batch_size, n_subj, n_vox, stage="affine")
    results: List[Optional[Dict[str, Any]]] = [None] * n_subj

    # Partition subjects: standard isotropic-compatible subjects run in vectorized chunks; anisotropic/special subjects delegate
    vec_indices: List[int] = []
    for idx_s, m_img in enumerate(mov_seq):
        aniso_m = _spacing_ratio(m_img.spacing) > ANISOTROPY_RESAMPLE_RATIO
        min_ext_m = min(float(n) * float(s) for n, s in zip(m_img.shape[:3], m_img.spacing[:3]))
        if aniso_m or min_ext_m < 90.0 or min(int(n) for n in m_img.shape[:3]) < 24:
            op = outprefixes[idx_s] if outprefixes is not None else None
            do_w = bool(write_transforms or op is not None)
            res_s = robust_affine(
                target,
                m_img,
                dof=dof,
                mode=mode,
                preset=preset,
                n_starts=n_starts,
                multi_start=multi_start,
                cone_angles_deg=cone_angles_deg,
                levels=levels,
                iterations=iterations,
                sampling_percentage=sampling_percentage,
                num_bins=num_bins,
                lambda_shear=lambda_shear,
                lambda_scale=lambda_scale,
                cluster_threshold=cluster_threshold,
                device=dev_str,
                seed=seed,
                return_warped=return_warped,
                write_transforms=do_w,
                verbose=verbose,
            )
            if op is not None:
                saved = res_s["fwd_chain"].save(op, compresslevel=4)
                res_s["fwdtransforms"] = saved
                res_s["invtransforms"] = saved
            results[idx_s] = res_s
        else:
            vec_indices.append(idx_s)

    for chunk_start in range(0, len(vec_indices), eff_b):
        chunk_ids = vec_indices[chunk_start : chunk_start + eff_b]
        t0_chunk = time.time()
        b_c = len(chunk_ids)

        mov_norms: List[ants.ANTsImage] = []
        mi_tensors: List[torch.Tensor] = []
        cand_lists: List[List[Tuple[str, np.ndarray, np.ndarray]]] = []
        base_starts_list: List[List[Tuple[str, np.ndarray]]] = []
        expected_ovs: List[float] = []

        for g_idx in chunk_ids:
            m_img = mov_seq[g_idx]
            m_norm = normalize_image(m_img, method="auto", force=True)
            mov_norms.append(m_norm)
            mi_tensors.append(image_to_tensor(m_norm, device=dev_obj, to_zyx=True))

            com_m_pos, com_m_neg = robust_center_of_mass(m_norm, weighted=True)
            com_m = np.asarray(
                com_m_pos
                if com_m_pos is not None
                else (com_m_neg if com_m_neg is not None else compute_center_of_mass(m_norm, weighted=False)),
                dtype=np.float64,
            )
            t_init = com_m - com_f
            fov_m = np.asarray(compute_fov_center(m_norm), dtype=np.float64)
            t_fov = fov_m - fov_f
            ext_m_world = np.abs(np.asarray(m_img.direction, dtype=np.float64).reshape(3, 3)) @ (
                np.asarray(m_img.shape[:3], dtype=np.float64) * np.asarray(m_img.spacing[:3], dtype=np.float64)
            )
            expected_ovs.append(float(np.clip(np.prod(np.minimum(1.0, ext_m_world / np.maximum(ext_f_world, 1e-3))), 0.05, 1.0)))

            supp_m = _analyze_volume_support_3d(m_img, m_norm)
            t_trunk = supp_m["center_lps"] - support_f["center_lps"] if support_f is not None else t_init

            t_pc_list: List[np.ndarray] = []
            if multi_start:
                try:
                    t_pc_list = list(_phase_correlation_translation_candidates(fixed_img, m_img, topk=3))
                except Exception:
                    t_pc_list = []

            cands_b: List[Tuple[str, np.ndarray, np.ndarray]] = [("Identity_CoM", np.eye(3), t_init)]
            bases_b: List[Tuple[str, np.ndarray]] = [("Identity_CoM", t_init), ("Identity", np.zeros(3))]
            if np.linalg.norm(t_fov - t_init) >= 1.0:
                cands_b.append(("Identity_FOV", np.eye(3), t_fov))
                bases_b.append(("Identity_FOV", t_fov))
            if np.linalg.norm(t_trunk - t_init) >= 4.0:
                cands_b.append(("Identity_TrunkBox", np.eye(3), t_trunk))
            if np.linalg.norm(t_init) > 1e-6:
                cands_b.append(("Identity", np.eye(3), np.zeros(3)))
            for k_pc, t_pc in enumerate(t_pc_list):
                cands_b.append((f"Identity_PhaseCorr{k_pc}", np.eye(3), t_pc))
            if multi_start:
                fov_sw = [("FOV", t_fov)] if np.linalg.norm(t_fov - t_init) >= 1.5 * coarse_mm else []
                rot_bases = [("CoM", t_init)] + fov_sw + [(f"PhaseCorr{k_pc}", t_pc) for k_pc, t_pc in enumerate(t_pc_list)]
                for b_nm, t_bs in rot_bases:
                    for r_nm, r_mat in cone_rots:
                        cands_b.append((f"{b_nm}_{r_nm}", r_mat, t_bs))
            cand_lists.append(cands_b)
            base_starts_list.append(bases_b)

        # Build moving pyramids per level (batched in one build_pyramid call when shapes/spacings match!)
        same_mov_grid = all(
            tuple(mov_seq[g].shape[:3]) == tuple(mov_seq[chunk_ids[0]].shape[:3])
            and np.allclose(mov_seq[g].spacing[:3], mov_seq[chunk_ids[0]].spacing[:3], atol=1e-5)
            for g in chunk_ids[1:]
        )
        mov_pyr_by_lvl: Dict[int, Dict[str, Any]] = {}
        if same_mov_grid and b_c > 1:
            mi_cat = torch.cat(mi_tensors, dim=0)
            m0 = mov_seq[chunk_ids[0]]
            lvs_cat = build_pyramid(
                mi_cat, m0.spacing, sched_levels, policy="mm_iso", sigma=0.0, origin=m0.origin, direction=m0.direction
            )
            del mi_cat
            for lv_obj in lvs_cat:
                lvl_i = int(lv_obj.level)
                P_list, q_list = [], []
                for bi, g_idx in enumerate(chunk_ids):
                    mg = mov_seq[g_idx]
                    Pm, qm = _moving_level_norm_affine_3d(
                        mg.origin, mg.direction, mg.spacing, lv_obj.f, lv_obj.off, lv_obj.shape, dev_obj
                    )
                    P_list.append(Pm)
                    q_list.append(qm)
                mov_pyr_by_lvl[lvl_i] = {
                    "stacked": lv_obj.tensor,
                    "list": [lv_obj.tensor[bi : bi + 1] for bi in range(b_c)],
                    "P_b": torch.stack(P_list, dim=0),
                    "q_b": torch.stack(q_list, dim=0),
                }
        else:
            per_subj_lvs = [
                {
                    int(lv_o.level): lv_o
                    for lv_o in build_pyramid(
                        mi_tensors[bi],
                        mov_seq[g_idx].spacing,
                        sched_levels,
                        policy="mm_iso",
                        sigma=0.0,
                        origin=mov_seq[g_idx].origin,
                        direction=mov_seq[g_idx].direction,
                    )
                }
                for bi, g_idx in enumerate(chunk_ids)
            ]
            for lvl_i in sched_levels:
                t_list = [per_subj_lvs[bi][lvl_i].tensor for bi in range(b_c)]
                same_lvl_sh = all(t.shape == t_list[0].shape for t in t_list[1:])
                P_list, q_list = [], []
                for bi, g_idx in enumerate(chunk_ids):
                    mg = mov_seq[g_idx]
                    lo = per_subj_lvs[bi][lvl_i]
                    Pm, qm = _moving_level_norm_affine_3d(
                        mg.origin, mg.direction, mg.spacing, lo.f, lo.off, lo.shape, dev_obj
                    )
                    P_list.append(Pm)
                    q_list.append(qm)
                mov_pyr_by_lvl[lvl_i] = {
                    "stacked": torch.cat(t_list, dim=0) if (same_lvl_sh and b_c > 1) else (t_list[0] if b_c == 1 else None),
                    "list": t_list,
                    "P_b": torch.stack(P_list, dim=0),
                    "q_b": torch.stack(q_list, dim=0),
                }
        del mi_tensors

        # Batched coarse candidate scoring across ALL subjects & candidates in ONE GPU evaluation
        phys_coarse, fw_coarse = _get_fix_subset(coarse, 1.0)
        mp_coarse = mov_pyr_by_lvl[coarse]
        all_A_c: List[np.ndarray] = []
        all_t_c: List[np.ndarray] = []
        all_subj_c: List[int] = []
        all_ov_c: List[float] = []
        subj_slices: List[Tuple[int, int]] = []
        cursor = 0
        for bi in range(b_c):
            c_len = len(cand_lists[bi])
            subj_slices.append((cursor, cursor + c_len))
            cursor += c_len
            for _, B_mat, t_vec in cand_lists[bi]:
                all_A_c.append(np.asarray(B_mat, dtype=np.float32))
                all_t_c.append(np.asarray(t_vec, dtype=np.float32))
                all_subj_c.append(bi)
                all_ov_c.append(expected_ovs[bi])
            for _, t_bs in base_starts_list[bi]:
                all_A_c.append(np.eye(3, dtype=np.float32))
                all_t_c.append(np.asarray(t_bs, dtype=np.float32))
                all_subj_c.append(bi)
                all_ov_c.append(expected_ovs[bi])

        with torch.no_grad():
            A_cand_t = torch.as_tensor(np.stack(all_A_c, axis=0), device=dev_obj, dtype=torch.float32)
            t_cand_t = torch.as_tensor(np.stack(all_t_c, axis=0), device=dev_obj, dtype=torch.float32)
            s_cand_t = torch.as_tensor(all_subj_c, device=dev_obj, dtype=torch.long)
            ov_cand_t = torch.as_tensor(all_ov_c, device=dev_obj, dtype=torch.float32)
            cand_scores_all = _eval_batched_poses_3d(
                A_cand_t,
                t_cand_t,
                s_cand_t,
                ov_cand_t,
                C_phys,
                phys_coarse,
                fw_coarse,
                mp_coarse["list"],
                mp_coarse["P_b"],
                mp_coarse["q_b"],
                same_mov_shape=False,
                mi_lev_stacked=None,
                num_bins=num_bins,
            ).tolist()

        # Select top `n_starts` SE(3)-diverse starts per subject
        eff_k = max(1, int(n_starts))
        kept_B: List[np.ndarray] = []
        kept_t: List[np.ndarray] = []
        kept_names: List[List[str]] = []
        kept_init_scores: List[float] = []
        scored_records: List[List[Tuple[str, float]]] = []
        baselines_per_subj: List[Dict[str, float]] = []
        base_cursor = sum(len(cl) for cl in cand_lists)

        cur_c = 0
        cur_b = base_cursor
        for bi in range(b_c):
            cl = cand_lists[bi]
            bl = base_starts_list[bi]
            sc_b = [
                (float(cand_scores_all[cur_c + j]), cl[j][0], cl[j][1], cl[j][2])
                for j in range(len(cl))
            ]
            cur_c += len(cl)
            sc_b.sort(key=lambda x: x[0])
            scored_records.append([(nm, sv) for sv, nm, _, _ in sc_b])

            bs_dict = {bl[j][0]: float(cand_scores_all[cur_b + j]) for j in range(len(bl))}
            cur_b += len(bl)
            baselines_per_subj.append(bs_dict)

            keep_b = _cluster_candidates_se3(
                sc_b, eff_k, cluster_threshold=cluster_threshold, domain_diam=trans_basin_diam
            )
            if eff_k >= 2 and not any(
                s[1] in ("Identity", "Identity_CoM", "Identity_FOV") or s[1].startswith("CoM_") for s in keep_b
            ):
                com_pool = [
                    s for s in sc_b if s[1] in ("Identity", "Identity_CoM", "Identity_FOV") or s[1].startswith("CoM_")
                ]
                if com_pool:
                    keep_b = list(keep_b[:-1]) + [com_pool[0]]
            while len(keep_b) < eff_k:
                keep_b.append(keep_b[0])
            keep_b = keep_b[:eff_k]
            kept_init_scores.append(float(keep_b[0][0]))
            kept_names.append([s[1] for s in keep_b])
            for _, _, Bm, tv in keep_b:
                kept_B.append(np.asarray(Bm, dtype=np.float32))
                kept_t.append(np.asarray(tv, dtype=np.float32))

        subj_idx_init = torch.arange(b_c, device=dev_obj, dtype=torch.long).repeat_interleave(eff_k)
        ov_init = torch.tensor(expected_ovs, device=dev_obj, dtype=torch.float32).repeat_interleave(eff_k)
        poses = _BatchedAffinePoses3D(
            torch.as_tensor(np.stack(kept_B, axis=0), device=dev_obj, dtype=torch.float32),
            torch.as_tensor(np.stack(kept_t, axis=0), device=dev_obj, dtype=torch.float32),
            subj_idx_init,
            ov_init,
            w_scale_t,
            w_shear_t,
        )
        cur_k_per_subj = eff_k
        winner_names: List[str] = [kept_names[bi][0] for bi in range(b_c)]

        # Run vectorized multi-stage schedule across all `b_c * cur_k_per_subj` poses simultaneously (0 inner syncs!)
        last_losses_t: Optional[torch.Tensor] = None
        for stg in schedule:
            lvl = int(stg["level"])
            iters_s = int(stg["iters"])
            s_dof = str(stg.get("dof", dof))
            samp_s = stg.get("sampling", sampling_percentage)
            if iters_s <= 0:
                continue
            phys_p, fw = _get_fix_subset(lvl, samp_s)
            mp = mov_pyr_by_lvl[lvl]
            gs = fix_lvl_data[lvl]["groups_scale"]
            if stg.get("step") is not None:
                step_t = stg["step"]
                lrs = tuple(float(st_v) * sp_min_f / max(gs_v, 1e-12) for st_v, gs_v in zip(step_t, gs))
            else:
                lrs = tuple(float(x) for x in stg["lr"])
            eta_f = float(stg.get("eta_frac", 0.05))
            p_groups = [
                {"params": [poses.t], "lr": lrs[0]},
                {"params": [poses.omega], "lr": lrs[1]},
            ]
            if s_dof == "affine":
                p_groups += [
                    {"params": [poses.scale], "lr": lrs[2]},
                    {"params": [poses.shear], "lr": lrs[3]},
                ]
            opt = torch.optim.Adam(p_groups, foreach=True)
            sched_lr = [
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(iters_s, 1), eta_min=lrs[0] * eta_f
                )
            ]
            # Vectorized cosine schedule across groups via manual multiplier
            base_lrs = [g["lr"] for g in opt.param_groups]
            del sched_lr
            use_stacked_stage = mp["stacked"] is not None

            for it_idx in range(iters_s):
                if iters_s > 1:
                    cos_f = eta_f + 0.5 * (1.0 - eta_f) * (1.0 + math.cos(math.pi * it_idx / max(iters_s, 1)))
                    for pg, blr in zip(opt.param_groups, base_lrs):
                        pg["lr"] = blr * cos_f
                opt.zero_grad(set_to_none=True)
                A_m = poses.matrices(s_dof)
                pose_losses = _eval_batched_poses_3d(
                    A_m,
                    poses.t,
                    poses.subj_idx,
                    poses.expected_ov,
                    C_phys,
                    phys_p,
                    fw,
                    mp["list"],
                    mp["P_b"],
                    mp["q_b"],
                    same_mov_shape=use_stacked_stage,
                    mi_lev_stacked=mp["stacked"],
                    num_bins=num_bins,
                )
                if s_dof == "affine" and (eff_l_shear > 0.0 or eff_l_scale > 0.0):
                    pose_losses = pose_losses + poses.reg_losses(s_dof, eff_l_shear, eff_l_scale)
                total_loss = pose_losses.sum()
                total_loss.backward()
                opt.step()
                poses.clamp_()
                last_losses_t = pose_losses.detach()

            if bool(stg.get("select", False)) and cur_k_per_subj > 1:
                with torch.no_grad():
                    phys_sel, fw_sel = _get_fix_subset(lvl, 1.0)
                    A_sel = poses.matrices(s_dof)
                    sel_scores = _eval_batched_poses_3d(
                        A_sel,
                        poses.t,
                        poses.subj_idx,
                        poses.expected_ov,
                        C_phys,
                        phys_sel,
                        fw_sel,
                        mp["list"],
                        mp["P_b"],
                        mp["q_b"],
                        same_mov_shape=use_stacked_stage,
                        mi_lev_stacked=mp["stacked"],
                        num_bins=num_bins,
                    ).view(b_c, cur_k_per_subj)
                    best_k = sel_scores.argmin(dim=1)
                    best_k_list = best_k.tolist()
                    winner_names = [kept_names[bi][int(best_k_list[bi])] for bi in range(b_c)]
                    keep_global = torch.arange(b_c, device=dev_obj, dtype=torch.long) * cur_k_per_subj + best_k
                    poses.select_indices(keep_global)
                    cur_k_per_subj = 1

        if cur_k_per_subj > 1:
            with torch.no_grad():
                A_sel = poses.matrices(dof)
                sel_scores = _eval_batched_poses_3d(
                    A_sel,
                    poses.t,
                    poses.subj_idx,
                    poses.expected_ov,
                    C_phys,
                    phys_coarse,
                    fw_coarse,
                    mp_coarse["list"],
                    mp_coarse["P_b"],
                    mp_coarse["q_b"],
                    same_mov_shape=(mp_coarse["stacked"] is not None),
                    mi_lev_stacked=mp_coarse["stacked"],
                    num_bins=num_bins,
                ).view(b_c, cur_k_per_subj)
                best_k = sel_scores.argmin(dim=1)
                best_k_list = best_k.tolist()
                winner_names = [kept_names[bi][int(best_k_list[bi])] for bi in range(b_c)]
                keep_global = torch.arange(b_c, device=dev_obj, dtype=torch.long) * cur_k_per_subj + best_k
                poses.select_indices(keep_global)
                cur_k_per_subj = 1

        # Batched self-check at coarse level + single D2H transfer for all `b_c` subjects
        with torch.no_grad():
            A_fin_b = poses.matrices("affine")
            coarse_dof = str(schedule[0].get("dof", dof))
            A_chk_b = poses.matrices(coarse_dof)
            final_coarse_t = _eval_batched_poses_3d(
                A_chk_b,
                poses.t,
                poses.subj_idx,
                poses.expected_ov,
                C_phys,
                phys_coarse,
                fw_coarse,
                mp_coarse["list"],
                mp_coarse["P_b"],
                mp_coarse["q_b"],
                same_mov_shape=(mp_coarse["stacked"] is not None),
                mi_lev_stacked=mp_coarse["stacked"],
                num_bins=num_bins,
            )
            A_fin_np = A_fin_b.detach().cpu().numpy().astype(np.float64)
            t_fin_np = poses.t.detach().cpu().numpy().astype(np.float64)
            final_coarse_vals = final_coarse_t.tolist()
            final_loss_vals = (
                last_losses_t.reshape(b_c, -1)[:, 0].tolist()
                if last_losses_t is not None
                else final_coarse_vals
            )

        elapsed_chunk = (time.time() - t0_chunk) / max(b_c, 1)
        kind_lin = "rigid" if dof == "rigid" else "affine"

        for bi, g_idx in enumerate(chunk_ids):
            best_base = min(baselines_per_subj[bi].values())
            slack = max(0.015, 0.05 * abs(best_base))
            fc_val = float(final_coarse_vals[bi])
            op = outprefixes[g_idx] if outprefixes is not None else None
            do_w = bool(write_transforms or op is not None)

            if (not math.isfinite(fc_val)) or fc_val > best_base + slack:
                # Re-run full single-pair robust_affine fallback retry protocol on any pathological pair
                res_retry = robust_affine(
                    target,
                    mov_seq[g_idx],
                    dof=dof,
                    mode=mode,
                    preset=preset,
                    n_starts=n_starts,
                    multi_start=multi_start,
                    cone_angles_deg=cone_angles_deg,
                    levels=levels,
                    iterations=iterations,
                    sampling_percentage=sampling_percentage,
                    num_bins=num_bins,
                    lambda_shear=lambda_shear,
                    lambda_scale=lambda_scale,
                    cluster_threshold=cluster_threshold,
                    device=dev_str,
                    seed=seed,
                    return_warped=return_warped,
                    write_transforms=do_w,
                    verbose=verbose,
                )
                if op is not None:
                    saved = res_retry["fwd_chain"].save(op, compresslevel=4)
                    res_retry["fwdtransforms"] = saved
                    res_retry["invtransforms"] = saved
                results[g_idx] = res_retry
                continue

            A_i = A_fin_np[bi]
            t_i = t_fin_np[bi]
            b_phys_i = com_f + t_i - A_i @ com_f
            if do_w:
                if op is not None:
                    os.makedirs(os.path.dirname(str(op)) or ".", exist_ok=True)
                    tx_path = f"{op}0GenericAffine.mat"
                else:
                    out_d = tempfile.mkdtemp(prefix="robust_affine_batch_")
                    tx_path = os.path.join(out_d, "affine.mat")
                tx_obj = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3)
                tx_obj.set_parameters(np.concatenate([A_i.ravel(), t_i]))
                tx_obj.set_fixed_parameters(com_f)
                ants.write_transform(tx_obj, tx_path)
                fwd_tx = [tx_path]
                fwd_ch = TransformChain.from_paths([tx_path])
                inv_ch = TransformChain.from_paths([tx_path], [True])
            else:
                fwd_tx = []
                fwd_ch = TransformChain([TransformElement.from_affine(A_i, b_phys_i, invert=False, kind=kind_lin)])
                inv_ch = TransformChain([TransformElement.from_affine(A_i, b_phys_i, invert=True, kind=kind_lin)])

            warped_mov = (
                _apply_transforms(
                    fixed=fixed_img,
                    moving=mov_seq[g_idx],
                    transformlist=fwd_tx if do_w else fwd_ch,
                    interpolator="linear",
                    device=dev_str,
                )
                if return_warped
                else None
            )
            results[g_idx] = {
                "warpedmovout": warped_mov,
                "warpedfixout": fixed_img,
                "fwdtransforms": fwd_tx,
                "invtransforms": fwd_tx,
                "whichtoinvert_inv": [True],
                "fwd_chain": fwd_ch,
                "inv_chain": inv_ch,
                "affine_matrix": (A_i, b_phys_i),
                "runtime_seconds": elapsed_chunk,
                "time": elapsed_chunk,
                "init_candidate": winner_names[bi],
                "init_score": kept_init_scores[bi],
                "final_loss": float(final_loss_vals[bi]),
                "candidates_scored": scored_records[bi],
                "status": "SUCCESS",
                "solver": "pytorch",
                "self_check": {
                    "final_loss_coarse": fc_val,
                    "baseline_losses_coarse": baselines_per_subj[bi],
                    "coarse_level": coarse,
                },
            }
        del mov_pyr_by_lvl, poses

    if not getattr(fixed, "_is_syntx_fixed_target", False):
        del target
        cleanup_gpu(dev_str)

    return [r for r in results if r is not None]


def batch_greedy(
    fixed: Union[ants.ANTsImage, FixedTarget],
    moving_list: Sequence[ants.ANTsImage],
    *,
    initial_transforms: Optional[Sequence[Any]] = None,
    batch_size: Union[int, str] = "auto",
    scales: Optional[Sequence[int]] = None,
    levels: Optional[Sequence[int]] = None,
    reg_iterations: Optional[Union[int, Sequence[int]]] = None,
    iterations: Optional[Union[int, Sequence[int]]] = None,
    learning_rate: float = 1.5,
    flow_sigma: float = 1.62,
    total_sigma: float = 0.0,
    optimizer: Optional[str] = "cfl",
    regadam_sigma: float = 0.8,
    cfl_momentum: float = 0.1875,
    regularizer: str = "navier",
    sobolev_alpha: Optional[float] = 2.625,
    similarity_metric: str = "cc2",
    lncc_radius: int = 2,
    return_inverse: bool = False,
    return_warped: bool = False,
    write_transforms: bool = False,
    amp: bool = False,
    sync_free: bool = True,
    convergence_slope: Optional[float] = None,
    affine_mode: str = "auto",
    fixed_mask: Optional[ants.ANTsImage] = None,
    restrict_transformation: Optional[Sequence[float]] = None,
    digital_diffeomorphism: bool = True,
    digital_diffeomorphism_floor: float = 0.05,
    digital_diffeomorphism_steps: int = 25,
    interpolator: str = "linear",
    device: Union[str, torch.device, None] = "auto",
    seed: int = 42,
    outprefixes: Optional[Sequence[Optional[str]]] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Vectorized multi-subject GPU greedy diffeomorphic cohort registration onto a shared ``fixed`` target.

    Shares one ``FixedTarget`` (pyramid + precomputed fixed LNCC local variance) across all subjects and
    evaluates ``B`` subjects per GPU launch with exact per-subject CFL/RegAdam step-size invariance.
    Defaults to ``write_transforms=False, return_warped=False, sync_free=True`` for maximum cohort throughput
    and zero `/tmp` disk I/O (returning GPU-resident ``res['fwd_chain']`` ready for ``syntx.apply_transforms``).
    """
    mov_seq = list(moving_list)
    n_subj = len(mov_seq)
    if n_subj == 0:
        return []
    if initial_transforms is not None and len(initial_transforms) != n_subj:
        raise ValueError(
            f"len(initial_transforms)={len(initial_transforms)} must match len(moving_list)={n_subj}"
        )
    if outprefixes is not None and len(outprefixes) != n_subj:
        raise ValueError(f"len(outprefixes)={len(outprefixes)} must match len(moving_list)={n_subj}")

    t0_all = time.time()
    dev_req = None if (device is None or str(device).lower() == "auto") else str(device)
    dev_str = resolve_device(dev_req, task="reg")
    dev_obj = torch.device(dev_str)

    caller_owned_target = bool(getattr(fixed, "_is_syntx_fixed_target", False))
    fixed_img = fixed.image if caller_owned_target else fixed
    require_finite_images(fixed=fixed_img)
    for m_img in mov_seq:
        require_finite_images(moving=m_img)

    dim = int(fixed_img.dimension)
    if reg_iterations is None:
        reg_iterations = iterations
    if reg_iterations is None:
        reg_iters_list = [100, 100, 20] if dim == 3 else [100, 100, 100, 50]
    elif isinstance(reg_iterations, int):
        reg_iters_list = [int(reg_iterations)]
    else:
        reg_iters_list = [int(x) for x in reg_iterations]

    if scales is None and levels is not None:
        scales = levels
    if scales is None:
        n_lv = len(reg_iters_list)
        scales_list = [2**i for i in range(n_lv)][::-1] if n_lv > 0 else ([4, 2, 1] if dim == 3 else [8, 4, 2, 1])
    elif isinstance(scales, int):
        scales_list = [int(scales)]
    else:
        scales_list = [int(x) for x in scales]

    extra_reg_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in REGULARIZER_EXTRA_KEYS}
    if "alpha" in extra_reg_kwargs:
        sobolev_alpha = float(extra_reg_kwargs.pop("alpha"))
    squared = bool(kwargs.pop("squared", True))
    padding_mode = str(kwargs.pop("padding_mode", "border"))
    num_bins_val = kwargs.pop("num_bins", None)
    samp_pct = kwargs.pop("sampling_percentage", None)
    raw_mc = kwargs.pop("metric_chunks", 1) or 1
    _return_warped_tensor = bool(kwargs.pop("_return_warped_tensor", False))
    _return_disp_tensor = bool(kwargs.pop("_return_disp_tensor", False))
    if kwargs:
        raise TypeError(f"syntx.batch_greedy() got unexpected keyword(s) {sorted(kwargs)}")

    reg_mode = str(regularizer).lower()
    valid_regs = set(list_regularizers()) | {"gaussian", "gauss", "sobolev"}
    if reg_mode not in valid_regs:
        raise ValueError(f"syntx.batch_greedy: unknown regularizer {reg_mode!r}")
    opt_resolved = validate_regularizer_optimizer_pair(reg_mode, optimizer, caller="syntx.batch_greedy")
    cfg = parse_similarity_metric(
        similarity_metric,
        lncc_radius=lncc_radius,
        num_bins=int(num_bins_val) if num_bins_val is not None else 32,
        squared=squared,
        sampling_percentage=samp_pct,
        metric_chunks=int(raw_mc),
    )

    target = (
        fixed
        if caller_owned_target
        else FixedTarget.from_image(
            fixed_img, scales=scales_list, window_size=2 * int(lncc_radius) + 1, device=dev_obj
        )
    )

    # 1. Resolve initial transforms (via batch_robust_affine when omitted or for None entries)
    t0_aff = time.time()
    if initial_transforms is None:
        aff_results = batch_robust_affine(
            target,
            mov_seq,
            mode=affine_mode,
            batch_size=batch_size,
            device=dev_str,
            seed=seed,
            return_warped=False,
            write_transforms=False,
            verbose=verbose,
        )
        raw_inits = [ar["fwd_chain"] for ar in aff_results]
    else:
        raw_inits = list(initial_transforms)
        none_ids = [idx_n for idx_n, tx_n in enumerate(raw_inits) if tx_n is None]
        if none_ids:
            sub_affs = batch_robust_affine(
                target,
                [mov_seq[idx_n] for idx_n in none_ids],
                mode=affine_mode,
                batch_size=batch_size,
                device=dev_str,
                seed=seed,
                return_warped=False,
                write_transforms=False,
                verbose=verbose,
            )
            for idx_n, ar in zip(none_ids, sub_affs):
                raw_inits[idx_n] = ar["fwd_chain"]
    t_aff_per_subj = (time.time() - t0_aff) / max(n_subj, 1)

    n_vox = int(np.prod(target.shape))
    eff_b = _select_cohort_batch_size(batch_size, n_subj, n_vox, stage="deformable")

    rev_sp = tuple(range(dim - 1, -1, -1))
    mask_t = (
        torch.from_numpy(fixed_mask.numpy())
        .to(device=dev_obj, dtype=torch.float32)
        .permute(*rev_sp)
        .contiguous()
        .unsqueeze(0)
        .unsqueeze(0)
        if fixed_mask is not None
        else None
    )

    # Group subjects by native moving shape & spacing so every batch uses exact native moving pyramids
    groups_by_geom: Dict[Tuple[Tuple[int, ...], Tuple[float, ...]], List[int]] = {}
    for idx_s, m_img in enumerate(mov_seq):
        g_key = (
            tuple(int(x) for x in m_img.shape[:dim]),
            tuple(round(float(x), 5) for x in m_img.spacing[:dim]),
        )
        groups_by_geom.setdefault(g_key, []).append(idx_s)

    results: List[Optional[Dict[str, Any]]] = [None] * n_subj

    for _, subj_indices in groups_by_geom.items():
        for c_start in range(0, len(subj_indices), eff_b):
            chunk_ids = subj_indices[c_start : c_start + eff_b]
            b_c = len(chunk_ids)

            mi_t_list: List[torch.Tensor] = []
            theta_list: List[torch.Tensor] = []
            init_warps: List[Optional[torch.Tensor]] = []
            resolved_list = []

            for g_idx in chunk_ids:
                m_img = mov_seq[g_idx]
                mi_norm = normalize_intensity(m_img)
                mi_t_i = (
                    torch.from_numpy(np.ascontiguousarray(mi_norm.numpy(), dtype=np.float32))
                    .to(device=dev_obj, dtype=torch.float32)
                    .permute(*rev_sp)
                    .contiguous()
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                mi_t_list.append(mi_t_i)

                res_init = resolve_initial_transform_and_warp(
                    target.norm_image,
                    mi_norm,
                    initial_transform=raw_inits[g_idx],
                    need_inverse=False,
                    device=dev_str,
                )
                resolved_list.append(res_init)
                th_i, _, _ = _build_torch_affine_matrix(
                    target.norm_image, mi_norm, res_init.M_phys, res_init.t_phys, dev_str
                )
                theta_list.append(th_i)
                init_warps.append(res_init.u0_norm)

            mi_batch = torch.cat(mi_t_list, dim=0)
            theta_batch = torch.cat(theta_list, dim=0)
            if any(iw is not None for iw in init_warps):
                z_shape = (1, *target.shape[::-1], dim)
                init_warp_batch: Optional[torch.Tensor] = torch.cat(
                    [
                        iw if iw is not None else torch.zeros(z_shape, device=dev_obj, dtype=torch.float32)
                        for iw in init_warps
                    ],
                    dim=0,
                )
            else:
                init_warp_batch = None

            eff_slope = None if sync_free else (convergence_slope if convergence_slope is not None else DEFAULT_SLOPE_THRESHOLD)
            model = GreedyRegistrationModel(
                dim=dim,
                learning_rate=learning_rate,
                flow_sigma=flow_sigma,
                total_sigma=total_sigma,
                optimizer=opt_resolved,
                regularizer=reg_mode,
                extra_reg_kwargs=extra_reg_kwargs,
                regadam_sigma=regadam_sigma,
                sobolev_alpha=sobolev_alpha,
                cfl_momentum=cfl_momentum,
                lncc_radius=lncc_radius,
                similarity_metric=similarity_metric,
                squared=squared,
                num_bins=cfg.num_bins,
                sampling_percentage=cfg.sampling_percentage,
                padding_mode=padding_mode,
                restrict_transformation=restrict_transformation,
                device=dev_obj,
                convergence_slope=eff_slope,
                digital_diffeomorphism=digital_diffeomorphism,
                digital_diffeomorphism_floor=digital_diffeomorphism_floor,
                digital_diffeomorphism_steps=digital_diffeomorphism_steps,
                metric_chunks=int(cfg.metric_chunks),
                amp=amp,
                sync_free=sync_free,
            )

            pre_pyr = target.get_deformable_pyramid(
                scales=scales_list,
                window_size=model.window_size,
                use_fast_lncc=bool(getattr(model.loss_fn, "_is_default_greedy_lncc", False)),
            )

            m_rep = mov_seq[chunk_ids[0]]
            t0_opt = time.time()
            warps_b = model.fit(
                fixed_tensor=target.tensor_zyx,
                moving_tensor=mi_batch,
                theta=theta_batch,
                scales=scales_list,
                iterations=reg_iters_list,
                fixed_mask_tensor=mask_t,
                verbose=verbose,
                spacing=target.spacing,
                moving_spacing=tuple(float(x) for x in m_rep.spacing[:dim]),
                origin=target.origin,
                direction=target.direction,
                moving_origin=tuple(float(x) for x in m_rep.origin[:dim]),
                moving_direction=np.asarray(m_rep.direction, dtype=np.float64)[:dim, :dim],
                initial_warp=init_warp_batch,
                _precomputed_fixed_pyramid=pre_pyr,
            )
            t_opt_per_subj = (time.time() - t0_opt) / max(b_c, 1)

            full_shape_zyx = target.tensor_zyx.shape[2:]
            restr_t_zyx = (
                torch.tensor(
                    list(reversed([0.0 if float(w) == 0.0 else 1.0 for w in restrict_transformation])),
                    device=dev_obj,
                    dtype=torch.float32,
                ).view(1, *([1] * dim), dim)
                if restrict_transformation is not None
                else None
            )

            from .core.digital_diffeomorphism import enforce_digital_diffeomorphism_nd

            for bi, g_idx in enumerate(chunk_ids):
                m_img = mov_seq[g_idx]
                res_init = resolved_list[bi]
                op = outprefixes[g_idx] if outprefixes is not None else None
                do_w = bool(write_transforms or op is not None)

                with torch.no_grad():
                    disp_i = normalized_to_physical_disp(
                        warps_b[bi : bi + 1],
                        full_shape_zyx,
                        target.spacing,
                        target.direction,
                        device=dev_str,
                        dtype=torch.float32,
                    )
                    if restr_t_zyx is not None:
                        disp_i = disp_i * restr_t_zyx

                if digital_diffeomorphism and (sum(reg_iters_list) > 0 or res_init.has_initial_warp):
                    disp_i = enforce_digital_diffeomorphism_nd(
                        disp_i,
                        None,
                        spacing=target.spacing,
                        origin=target.origin,
                        direction=target.direction,
                        jac_floor=digital_diffeomorphism_floor,
                        steps=digital_diffeomorphism_steps,
                        stationary_boundary=False,
                    )
                    if restr_t_zyx is not None:
                        with torch.no_grad():
                            disp_i = disp_i * restr_t_zyx

                inv_disp_i = None
                inv_err_dict: Dict[str, Any] = {}
                t_inv_i = 0.0
                if return_inverse:
                    t0_inv = time.time()
                    inv_disp_i = update_inverse_field_nd_anderson(
                        W_disp=disp_i,
                        W_inv_disp=res_init.u0_inv_phys,
                        steps=15,
                        m=5,
                        spacing=target.spacing,
                        origin=target.origin,
                        direction=target.direction,
                        check_after=3,
                        check_interval=4,
                        stationary_boundary=False,
                    )
                    if digital_diffeomorphism and (sum(reg_iters_list) > 0 or res_init.has_initial_warp):
                        inv_disp_i = enforce_digital_diffeomorphism_nd(
                            inv_disp_i,
                            None,
                            spacing=target.spacing,
                            origin=target.origin,
                            direction=target.direction,
                            jac_floor=digital_diffeomorphism_floor,
                            steps=digital_diffeomorphism_steps,
                            stationary_boundary=False,
                        )
                    if restr_t_zyx is not None:
                        with torch.no_grad():
                            inv_disp_i = inv_disp_i * restr_t_zyx
                    with torch.no_grad():
                        ice_c = calculate_inverse_identity_error(
                            disp_i,
                            inv_disp_i,
                            spacing=target.spacing,
                            origin=target.origin,
                            direction=target.direction,
                            return_error_map=False,
                        )
                        _f_ref, _i_ref = disp_i, inv_disp_i

                        def _load_err(_fr=_f_ref, _ir=_i_ref):
                            with torch.no_grad():
                                return calculate_inverse_identity_error(
                                    _fr,
                                    _ir,
                                    spacing=target.spacing,
                                    origin=target.origin,
                                    direction=target.direction,
                                    return_error_map=True,
                                )["error_map"].cpu()

                        inv_err_dict = {"phi_1": _LazyKeyDict(ice_c, lazy_loaders={"error_map": _load_err})}
                    t_inv_i = time.time() - t0_inv

                fwd_tx_paths: List[str] = []
                inv_tx_paths: List[str] = []
                wti_inv: List[bool] = []
                M_p, t_p = res_init.M_phys, res_init.t_phys

                if do_w:
                    disp_img_i = disp_tensor_to_itk(disp_i, fixed_img)
                    inv_img_i = disp_tensor_to_itk(inv_disp_i, fixed_img) if inv_disp_i is not None else None
                    comp_lvl = 4 if op is not None else 1
                    if op is not None:
                        os.makedirs(os.path.dirname(str(op)) or ".", exist_ok=True)
                        fwd_f = f"{op}1Warp.nii.gz"
                        inv_f = f"{op}1InverseWarp.nii.gz" if inv_img_i is not None else None
                        aff_f = f"{op}0GenericAffine.mat" if res_init.has_linear_affine else None
                    else:
                        fwd_f = tempfile.NamedTemporaryFile(suffix="_fwd_warp.nii.gz", delete=False).name
                        inv_f = (
                            tempfile.NamedTemporaryFile(suffix="_inv_warp.nii.gz", delete=False).name
                            if inv_img_i is not None
                            else None
                        )
                        aff_f = (
                            tempfile.NamedTemporaryFile(suffix=".mat", delete=False).name
                            if res_init.has_linear_affine
                            else None
                        )
                    if aff_f is not None:
                        export_ants_affine_transform(M_p, t_p, dim=dim, filename=aff_f)
                    if inv_img_i is not None and inv_f is not None:
                        _write_ants_images_fast([(disp_img_i, fwd_f), (inv_img_i, inv_f)], compresslevel=comp_lvl)
                        _cache_disp_image_and_field(fwd_f, disp_img_i, dev_str, disp_tensor=disp_i)
                        _cache_disp_image_and_field(inv_f, inv_img_i, dev_str, disp_tensor=inv_disp_i)
                    else:
                        _write_ants_images_fast([(disp_img_i, fwd_f)], compresslevel=comp_lvl)
                        _cache_disp_image_and_field(fwd_f, disp_img_i, dev_str, disp_tensor=disp_i)
                    fwd_tx_paths = [fwd_f, aff_f] if aff_f is not None else [fwd_f]
                    if inv_f is not None:
                        inv_tx_paths = [aff_f, inv_f] if aff_f is not None else [inv_f]
                        wti_inv = [True, False] if aff_f is not None else [False]
                    fwd_chain = TransformChain.from_paths(fwd_tx_paths)
                    inv_chain = TransformChain.from_paths(inv_tx_paths, wti_inv) if inv_tx_paths else TransformChain([])
                    aff_out = aff_f
                else:
                    fwd_el = TransformElement.from_disp_tensor(disp_i, fixed_img, kind="displacement")
                    if res_init.has_linear_affine:
                        aff_el = TransformElement.from_affine(M_p, t_p, invert=False)
                        fwd_chain = TransformChain([fwd_el, aff_el])
                        aff_out = TransformChain([aff_el])
                    else:
                        fwd_chain = TransformChain([fwd_el])
                        aff_out = None
                    if inv_disp_i is not None:
                        inv_el = TransformElement.from_disp_tensor(inv_disp_i, fixed_img, kind="inverse_displacement")
                        if res_init.has_linear_affine:
                            aff_inv_el = TransformElement.from_affine(M_p, t_p, invert=True)
                            inv_chain = TransformChain([aff_inv_el, inv_el])
                            wti_inv = [True, False]
                        else:
                            inv_chain = TransformChain([inv_el])
                            wti_inv = [False]
                    else:
                        inv_chain = TransformChain([])

                warped_mov_i = None
                warped_mov_ten_i = None
                warped_fix_i = None
                if return_warped or _return_warped_tensor:
                    can_cf = dim == 3 and ants.is_image(fixed_img) and ants.is_image(m_img)
                    mov_cf = (
                        ContinuousField.from_tensor(
                            mi_t_list[bi],
                            spatial_shape=m_img.shape,
                            origin=m_img.origin,
                            spacing=m_img.spacing,
                            direction=m_img.direction,
                            is_vector=False,
                        )
                        if (can_cf and is_unclipped_unit_image(m_img))
                        else None
                    )
                    if _return_warped_tensor:
                        warped_mov_ten_i = _apply_transforms(
                            fixed=fixed_img,
                            moving=m_img,
                            transformlist=fwd_tx_paths if do_w else fwd_chain,
                            interpolator=interpolator,
                            device=dev_str,
                            _moving_field=mov_cf,
                            return_tensor=True,
                        )
                        if return_warped:
                            rev_w = tuple(range(warped_mov_ten_i.ndim - 3, -1, -1))
                            arr_w = (
                                warped_mov_ten_i[0, 0]
                                .permute(*rev_w)
                                .contiguous()
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.float32, copy=False)
                            )
                            warped_mov_i = fixed_img.new_image_like(arr_w)
                    else:
                        warped_mov_i = _apply_transforms(
                            fixed=fixed_img,
                            moving=m_img,
                            transformlist=fwd_tx_paths if do_w else fwd_chain,
                            interpolator=interpolator,
                            device=dev_str,
                            _moving_field=mov_cf,
                        )
                    if return_warped and return_inverse and len(inv_chain) > 0:
                        fix_cf = (
                            ContinuousField.from_tensor(
                                target.tensor_zyx,
                                spatial_shape=fixed_img.shape,
                                origin=fixed_img.origin,
                                spacing=fixed_img.spacing,
                                direction=fixed_img.direction,
                                is_vector=False,
                            )
                            if (can_cf and target.is_unclipped_unit)
                            else None
                        )
                        warped_fix_i = _apply_transforms(
                            fixed=m_img,
                            moving=fixed_img,
                            transformlist=inv_tx_paths if do_w else inv_chain,
                            whichtoinvert=wti_inv if do_w else None,
                            interpolator=interpolator,
                            device=dev_str,
                            _moving_field=fix_cf,
                        )

                # Attach per-subject model view for liouville_determinant / provenance inspection
                sub_model = GreedyRegistrationModel(
                    dim=dim,
                    learning_rate=learning_rate,
                    flow_sigma=flow_sigma,
                    total_sigma=total_sigma,
                    optimizer=opt_resolved,
                    regularizer=reg_mode,
                    extra_reg_kwargs=extra_reg_kwargs,
                    regadam_sigma=regadam_sigma,
                    sobolev_alpha=sobolev_alpha,
                    cfl_momentum=cfl_momentum,
                    lncc_radius=lncc_radius,
                    similarity_metric=similarity_metric,
                    squared=squared,
                    device=dev_obj,
                    convergence_slope=eff_slope,
                    digital_diffeomorphism=digital_diffeomorphism,
                    digital_diffeomorphism_floor=digital_diffeomorphism_floor,
                    digital_diffeomorphism_steps=digital_diffeomorphism_steps,
                    amp=amp,
                    sync_free=sync_free,
                )
                sub_model.warp = warps_b[bi : bi + 1]
                sub_model.theta = theta_list[bi]

                prov_i = {
                    "algorithm": "syntx.batch_greedy",
                    "formulation": "eulerian_compositive",
                    "batch_size": b_c,
                    "optimizer": opt_resolved,
                    "regularizer": reg_mode,
                    "sobolev_alpha": sobolev_alpha,
                    "reg_iterations": reg_iters_list,
                    "scales": scales_list,
                    "learning_rate": learning_rate,
                    "flow_sigma": flow_sigma,
                    "return_inverse": return_inverse,
                    "amp": bool(amp),
                    "sync_free": bool(sync_free),
                    "write_transforms": bool(do_w),
                    "digital_diffeomorphism": digital_diffeomorphism,
                    "runtime_affine_sec": round(t_aff_per_subj, 3),
                    "runtime_deformable_sec": round(t_opt_per_subj, 3),
                    "runtime_inverse_sec": round(t_inv_i, 3),
                    "device": dev_str,
                }
                res_entry = {
                    "warpedmovout": warped_mov_i,
                    "warpedfixout": warped_fix_i,
                    "fwdtransforms": fwd_tx_paths,
                    "invtransforms": inv_tx_paths,
                    "whichtoinvert_inv": wti_inv,
                    "fwd_chain": fwd_chain,
                    "inv_chain": inv_chain,
                    "model": sub_model,
                    "convergence": sub_model.stop.summary(),
                    "provenance": prov_i,
                    "inverse_identity_errors": inv_err_dict,
                    "affine_transform": aff_out,
                }
                if _return_warped_tensor and warped_mov_ten_i is not None:
                    res_entry["warpedmovout_tensor"] = warped_mov_ten_i
                if _return_disp_tensor:
                    res_entry["disp_tensor_zyx"] = disp_i
                results[g_idx] = res_entry
            del mi_batch, mi_t_list, warps_b

    if not caller_owned_target:
        del target
        cleanup_gpu(dev_str)

    _ = time.time() - t0_all
    return [r for r in results if r is not None]
