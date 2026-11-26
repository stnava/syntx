"""Population-scale GPU cohort registration primitives: ``FixedTarget``, ``batch_robust_affine``, ``batch_greedy``, and ``batch_syn``.

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
    if stage == "syn":
        if n_voxels > 16_777_216:
            return 1
        if n_voxels >= 8_000_000:
            return min(3, n_subjects)
        if n_voxels >= 2_000_000:
            return min(4, n_subjects)
        return min(8, n_subjects)
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
    initial_warps: Optional[Sequence[Any]] = None,
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
    control_stride: int = 1,
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
    if initial_warps is not None and len(initial_warps) != n_subj:
        raise ValueError(
            f"len(initial_warps)={len(initial_warps)} must match len(moving_list)={n_subj}"
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
                    initial_warp=initial_warps[g_idx] if initial_warps is not None else None,
                    need_inverse=return_inverse,
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
                control_stride=control_stride,
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
                    control_stride=control_stride,
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


def _normalize_fg_unit_zyx(img: ants.ANTsImage, dev_obj: torch.device) -> torch.Tensor:
    """Upload ``img`` and map foreground 2-98% quantiles to ``[0, 1]`` in ``(1, 1, *spatial_zyx)``."""
    from .core.pipeline import _compute_fg_norm_params

    dim = int(img.dimension)
    rev_sp = tuple(range(dim - 1, -1, -1))
    arr = np.ascontiguousarray(img.numpy(), dtype=np.float32)
    p02, denom, is_id_clamp = _compute_fg_norm_params(arr, 0.02, 0.98)
    t = torch.from_numpy(arr).to(device=dev_obj, dtype=torch.float32).permute(*rev_sp).contiguous()
    if is_id_clamp:
        t.clamp_(0.0, 1.0)
    else:
        t.sub_(p02).div_(denom).clamp_(0.0, 1.0)
    return t.unsqueeze(0).unsqueeze(0)


def batch_syn(
    fixed: Union[ants.ANTsImage, FixedTarget],
    moving_list: Sequence[ants.ANTsImage],
    *,
    initial_transforms: Optional[Sequence[Any]] = None,
    initial_warps: Optional[Sequence[Any]] = None,
    initial_inverse_warps: Optional[Sequence[Any]] = None,
    type_of_transform: str = "SyN",
    batch_size: Union[int, str] = "auto",
    levels: Optional[Sequence[int]] = None,
    scales: Optional[Sequence[int]] = None,
    reg_iterations: Optional[Union[int, Sequence[int]]] = None,
    iterations: Optional[Union[int, Sequence[int]]] = None,
    grad_step: float = 0.6,
    flow_sigma: float = 2.4,
    total_sigma: float = 0.0,
    optimizer: str = "cfl",
    regularizer: str = "navier",
    sobolev_alpha: Optional[float] = 1.5,
    fast_smooth: Optional[bool] = None,
    syn_metric: str = "cc2",
    syn_sampling: int = 2,
    in_loop_inv_steps: Union[int, Sequence[int]] = 10,
    inv_refresh_every: Union[int, Sequence[int]] = (1, 2, 4),
    polish_inv_steps: int = 30,
    polish_inv_tol: Optional[float] = 0.02,
    compose_polish_tol: Optional[float] = 0.30,
    inverse_steps: int = 30,
    inverse_method: str = "anderson",
    antisymmetric: bool = True,
    stationary_boundary: bool = True,
    control_stride: int = 1,
    return_warped: bool = False,
    write_transforms: bool = False,
    sync_free: bool = True,
    affine_mode: str = "auto",
    fixed_mask: Optional[ants.ANTsImage] = None,
    restrict_transformation: Optional[Sequence[float]] = None,
    digital_diffeomorphism: bool = True,
    digital_diffeomorphism_floor: float = 0.05,
    digital_diffeomorphism_steps: int = 50,
    interpolator: str = "linear",
    device: Union[str, torch.device, None] = "auto",
    seed: int = 42,
    outprefixes: Optional[Sequence[Optional[str]]] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Vectorized multi-subject GPU Symmetric Normalization (``SyN``) cohort registration onto a shared ``fixed`` target.

    Shares one ``FixedTarget`` across all subjects and evaluates ``B`` symmetric midpoint registrations
    (``warp_l2r``, ``warp_r2l``, ``warp_l2r_inv``, ``warp_r2l_inv``) per GPU launch with exact per-subject
    CFL step-norm invariance, batched continuum/Sobolev Green's regularization, batched Anderson inverse
    refreshes, and ``0.0000%``-folding digital diffeomorphism guarantees.
    Defaults to ``write_transforms=False, return_warped=False, sync_free=True`` for maximum cohort throughput.
    """
    mov_seq = list(moving_list)
    n_subj = len(mov_seq)
    if n_subj == 0:
        return []
    if initial_transforms is not None and len(initial_transforms) != n_subj:
        raise ValueError(
            f"len(initial_transforms)={len(initial_transforms)} must match len(moving_list)={n_subj}"
        )
    if initial_warps is not None and len(initial_warps) != n_subj:
        raise ValueError(
            f"len(initial_warps)={len(initial_warps)} must match len(moving_list)={n_subj}"
        )
    if initial_inverse_warps is not None and len(initial_inverse_warps) != n_subj:
        raise ValueError(
            f"len(initial_inverse_warps)={len(initial_inverse_warps)} must match len(moving_list)={n_subj}"
        )
    if outprefixes is not None and len(outprefixes) != n_subj:
        raise ValueError(f"len(outprefixes)={len(outprefixes)} must match len(moving_list)={n_subj}")

    if "similarity_metric" in kwargs:
        syn_metric = str(kwargs.pop("similarity_metric"))
    if "lncc_radius" in kwargs:
        syn_sampling = int(kwargs.pop("lncc_radius"))
    extra_reg_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in REGULARIZER_EXTRA_KEYS}
    if "alpha" in extra_reg_kwargs:
        sobolev_alpha = float(extra_reg_kwargs.pop("alpha"))
    _return_warped_tensor = bool(kwargs.pop("_return_warped_tensor", False))
    _return_disp_tensor = bool(kwargs.pop("_return_disp_tensor", False))
    if kwargs:
        raise TypeError(f"syntx.batch_syn() got unexpected keyword(s) {sorted(kwargs)}")

    t0_all = time.time()
    dev_req = None if (device is None or str(device).lower() == "auto") else str(device)
    dev_str = resolve_device(dev_req, task="syn")
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

    if levels is None and scales is not None:
        levels = scales
    if levels is None:
        n_lv = len(reg_iters_list)
        levels_list = [2**i for i in range(n_lv)][::-1] if n_lv > 0 else ([4, 2, 1] if dim == 3 else [8, 4, 2, 1])
    elif isinstance(levels, int):
        levels_list = [int(levels)]
    else:
        levels_list = [int(x) for x in levels]
    if len(reg_iters_list) < len(levels_list):
        reg_iters_list = [0] * (len(levels_list) - len(reg_iters_list)) + list(reg_iters_list)
    elif len(reg_iters_list) > len(levels_list):
        reg_iters_list = list(reg_iters_list)[-len(levels_list) :]

    tx_type_up = str(type_of_transform).upper()
    if tx_type_up not in ("SYN", "SYNTO", "SYNONLY", "BSPLINESYN"):
        raise ValueError(
            f"syntx.batch_syn: unsupported type_of_transform={type_of_transform!r}; expected 'SyN' or 'SyNOnly'"
        )
    is_syn_only = tx_type_up == "SYNONLY"

    target = (
        fixed
        if caller_owned_target
        else FixedTarget.from_image(
            fixed_img, scales=levels_list, window_size=2 * int(syn_sampling) + 1, device=dev_obj
        )
    )

    # 1. Resolve initial transforms across cohort
    t0_aff = time.time()
    if initial_transforms is None:
        if is_syn_only:
            raw_inits: List[Any] = ["identity"] * n_subj
        else:
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
            if is_syn_only:
                for idx_n in none_ids:
                    raw_inits[idx_n] = "identity"
            else:
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

    cfg_syn = parse_similarity_metric(syn_metric, lncc_radius=syn_sampling)
    metric_low = str(cfg_syn.metric).lower()
    reg_mode = str(regularizer).lower()
    opt_low = str(optimizer).lower()
    can_vectorize_3d = bool(
        dim == 3
        and metric_low in ("cc2", "lncc", "mattes_mi")
        and opt_low == "cfl"
        and tx_type_up != "BSPLINESYN"
        and reg_mode not in ("bspline", "bsplinesyn")
        and str(interpolator).lower() == "linear"
    )
    if not can_vectorize_3d:
        from .syn import syn as _single_syn

        out_fallback: List[Dict[str, Any]] = []
        for idx_s, m_img in enumerate(mov_seq):
            op_s = outprefixes[idx_s] if outprefixes is not None else None
            r_s = _single_syn(
                fixed_img,
                m_img,
                type_of_transform=type_of_transform,
                initial_transform=raw_inits[idx_s],
                initial_warp=initial_warps[idx_s] if initial_warps is not None else None,
                initial_inverse_warp=initial_inverse_warps[idx_s] if initial_inverse_warps is not None else None,
                levels=levels_list,
                reg_iterations=reg_iters_list,
                grad_step=grad_step,
                flow_sigma=flow_sigma,
                total_sigma=total_sigma,
                optimizer=optimizer,
                regularizer=reg_mode,
                sobolev_alpha=sobolev_alpha,
                fast_smooth=fast_smooth,
                syn_metric=syn_metric,
                syn_sampling=syn_sampling,
                in_loop_inv_steps=in_loop_inv_steps,
                inv_refresh_every=inv_refresh_every,
                polish_inv_steps=polish_inv_steps,
                polish_inv_tol=polish_inv_tol,
                compose_polish_tol=compose_polish_tol,
                inverse_steps=inverse_steps,
                inverse_method=inverse_method,
                antisymmetric=antisymmetric,
                stationary_boundary=stationary_boundary,
                control_stride=control_stride,
                return_warped=return_warped,
                write_transforms=bool(write_transforms or op_s is not None),
                export_midpoint=False,
                sync_free=sync_free,
                restrict_transformation=restrict_transformation,
                digital_diffeomorphism=digital_diffeomorphism,
                digital_diffeomorphism_floor=digital_diffeomorphism_floor,
                digital_diffeomorphism_steps=digital_diffeomorphism_steps,
                interpolator=interpolator,
                device=dev_str,
                seed=seed,
                outprefix=op_s,
                verbose=verbose,
                **extra_reg_kwargs,
            )
            out_fallback.append(r_s)
        if not caller_owned_target:
            del target
            cleanup_gpu(dev_str)
        return out_fallback

    from .core.digital_diffeomorphism import (
        enforce_digital_diffeomorphism_nd,
        pullback_digital_diffeomorphism_to_halves,
    )
    from .core.grid import resize_field
    from .core.inverse import (
        _scaled_sq_sum,
        update_inverse_field_nd,
        update_inverse_field_nd_hybrid_lm,
    )
    from .core.kernels import exact_lncc_2sided_loss_and_grad_3d, exact_mattes_mi_2sided_loss_and_grad_3d
    from .core.losses.cross_correlation import _get_window_count_views
    from .core.regularizers import COUPLED_CONTINUUM_REGULARIZERS, build_level_regularizer
    from .core.smoothing import (
        apply_sobolev_green_operator,
        get_boundary_mask,
        separable_gaussian_filter,
    )
    from .pyramid import cfl_shrink_ratio, level_spacing as _pyr_level_spacing
    from .spatial import (
        compute_autograd_physical_scale,
        get_physical_grid_torch,
        get_physical_to_normalized_affine,
        physical_to_normalized_fast,
        physical_to_normalized_torch_cached,
        reverse_metadata,
    )
    from .syn import SyNTo

    squared_metric = metric_low == "cc2"
    ws_syn = 2 * int(syn_sampling) + 1
    fluid_sigma_val = math.sqrt(float(flow_sigma)) if float(flow_sigma) > 0.0 else 0.0
    elastic_sigma_val = math.sqrt(float(total_sigma)) if float(total_sigma) > 0.0 else 0.0

    # Build shared fixed deformable pyramid once for the entire cohort
    fi_syn_t = _normalize_fg_unit_zyx(fixed_img, dev_obj)
    with torch.no_grad():
        i_min = fi_syn_t.amin()
        i_span = fi_syn_t.amax() - i_min
        if float(i_span.item()) > 1e-6:
            fi_syn_t = (fi_syn_t - i_min) / i_span.clamp(min=1e-6)
    I_lv = build_pyramid(fi_syn_t, target.spacing, levels_list, "deformable")
    del fi_syn_t
    full_shape_zyx = tuple(int(s) for s in target.tensor_zyx.shape[2:])
    finest_spacing_fixed = _pyr_level_spacing(target.spacing, full_shape_zyx, I_lv[-1].tensor.shape[2:])

    n_vox = int(np.prod(target.shape))
    eff_b = _select_cohort_batch_size(batch_size, n_subj, n_vox, stage="syn")
    raw_cstride = max(1, int(control_stride))

    restr_mask = (
        torch.tensor(
            list(reversed([float(w) for w in restrict_transformation])),
            device=dev_obj,
            dtype=torch.float32,
        ).view(1, *([1] * dim), dim)
        if restrict_transformation is not None
        else None
    )

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

            mi_raw_unit_list: List[Optional[torch.Tensor]] = []
            mi_syn_list: List[torch.Tensor] = []
            resolved_list = []
            M_zyx_list: List[torch.Tensor] = []
            t_zyx_list: List[torch.Tensor] = []

            for g_idx in chunk_ids:
                m_img = mov_seq[g_idx]
                mi_u = _normalize_fg_unit_zyx(m_img, dev_obj)
                if (return_warped or _return_warped_tensor) and is_unclipped_unit_image(m_img):
                    mi_raw_unit_list.append(mi_u)
                    mi_s = mi_u.clone()
                else:
                    mi_raw_unit_list.append(None)
                    mi_s = mi_u
                with torch.no_grad():
                    j_min = mi_s.amin()
                    j_span = mi_s.amax() - j_min
                    if float(j_span.item()) > 1e-6:
                        mi_s = (mi_s - j_min) / j_span.clamp(min=1e-6)
                mi_syn_list.append(mi_s)

                res_init = resolve_initial_transform_and_warp(
                    fixed_img,
                    m_img,
                    initial_transform=raw_inits[g_idx],
                    initial_warp=initial_warps[g_idx] if initial_warps is not None else None,
                    initial_inverse_warp=initial_inverse_warps[g_idx] if initial_inverse_warps is not None else None,
                    need_inverse=True,
                    device=dev_str,
                )
                resolved_list.append(res_init)
                m_xyz = np.asarray(res_init.M_phys, dtype=np.float32)
                t_xyz = np.asarray(res_init.t_phys, dtype=np.float32)
                M_zyx_list.append(torch.from_numpy(m_xyz[::-1, ::-1].copy()).to(device=dev_obj, dtype=torch.float32))
                t_zyx_list.append(torch.from_numpy(t_xyz[::-1].copy()).to(device=dev_obj, dtype=torch.float32))

            J_batch = torch.cat(mi_syn_list, dim=0)
            del mi_syn_list
            m_rep = mov_seq[chunk_ids[0]]
            J_lv = build_pyramid(
                J_batch,
                tuple(float(x) for x in m_rep.spacing[:dim]),
                levels_list,
                "deformable",
                reference_spacing=target.spacing,
            )
            del J_batch

            M_zyx_b = torch.stack(M_zyx_list, dim=0)  # (b_c, 3, 3)
            t_zyx_b = torch.stack(t_zyx_list, dim=0)  # (b_c, 3)

            t0_opt = time.time()
            init_sp0 = tuple(int(s) for s in I_lv[0].tensor.shape[2:])
            if int(levels_list[0]) == 1 and raw_cstride >= 2 and min(init_sp0) >= 16 * raw_cstride:
                init_sp0 = tuple(max(8, int(round(s / raw_cstride))) for s in init_sp0)

            with torch.no_grad():
                warp_l2r = torch.zeros((b_c, *init_sp0, dim), device=dev_obj, dtype=torch.float32)
                warp_l2r_inv = torch.zeros_like(warp_l2r)
                if any(r.u0_phys is not None for r in resolved_list):
                    wr_init_list = []
                    wri_init_list = []
                    z_u0 = torch.zeros((1, *init_sp0, dim), device=dev_obj, dtype=torch.float32)
                    for r in resolved_list:
                        if r.u0_phys is not None:
                            u0_c = (
                                resize_field(r.u0_phys, list(init_sp0), mode="trilinear")
                                if tuple(r.u0_phys.shape[1:-1]) != init_sp0
                                else r.u0_phys.clone()
                            )
                            u0i_src = r.u0_inv_phys if r.u0_inv_phys is not None else -r.u0_phys
                            u0i_c = (
                                resize_field(u0i_src, list(init_sp0), mode="trilinear")
                                if tuple(u0i_src.shape[1:-1]) != init_sp0
                                else u0i_src.clone()
                            )
                            wr_init_list.append(u0_c)
                            wri_init_list.append(u0i_c)
                        else:
                            wr_init_list.append(z_u0)
                            wri_init_list.append(z_u0)
                    warp_r2l = torch.cat(wr_init_list, dim=0)
                    warp_r2l_inv = torch.cat(wri_init_list, dim=0)
                else:
                    warp_r2l = torch.zeros_like(warp_l2r)
                    warp_r2l_inv = torch.zeros_like(warp_l2r)

                for level_idx, (scale, n_epochs) in enumerate(zip(levels_list, reg_iters_list)):
                    I_curr = I_lv[level_idx].tensor
                    J_curr = J_lv[level_idx].tensor
                    img_spatial = tuple(int(s) for s in I_curr.shape[2:])
                    img_sp_fixed = tuple(I_lv[level_idx].spacing)
                    mov_lvl_spatial = tuple(int(s) for s in J_curr.shape[2:])
                    mov_sp_lvl = tuple(J_lv[level_idx].spacing)

                    eff_cs = (
                        raw_cstride
                        if (int(scale) == 1 and raw_cstride >= 2 and min(img_spatial) >= 16 * raw_cstride)
                        else 1
                    )
                    curr_spatial = (
                        tuple(max(8, int(round(s / eff_cs))) for s in img_spatial)
                        if eff_cs > 1
                        else img_spatial
                    )
                    if tuple(warp_l2r.shape[1:-1]) != curr_spatial:
                        warp_l2r = resize_field(warp_l2r, list(curr_spatial), mode="trilinear")
                        warp_r2l = resize_field(warp_r2l, list(curr_spatial), mode="trilinear")
                        warp_l2r_inv = resize_field(warp_l2r_inv, list(curr_spatial), mode="trilinear")
                        warp_r2l_inv = resize_field(warp_r2l_inv, list(curr_spatial), mode="trilinear")

                    if n_epochs <= 0:
                        J_lv[level_idx] = None
                        continue

                    curr_sp_fixed = (
                        tuple(_pyr_level_spacing(img_sp_fixed, img_spatial, curr_spatial))
                        if eff_cs > 1
                        else img_sp_fixed
                    )
                    X_phys = get_physical_grid_torch(
                        curr_spatial, curr_sp_fixed, target.origin, target.direction, device=dev_obj, dtype=torch.float32
                    )
                    X_phys_img = (
                        get_physical_grid_torch(
                            img_spatial, img_sp_fixed, target.origin, target.direction, device=dev_obj, dtype=torch.float32
                        )
                        if eff_cs > 1
                        else X_phys
                    )
                    b_mask = get_boundary_mask(curr_spatial, dev_obj, torch.float32)

                    f_sp_rev, f_org_rev, f_dir_rev = reverse_metadata(curr_sp_fixed, target.origin, target.direction)
                    f_shape_t = torch.tensor(list(curr_spatial), device=dev_obj, dtype=torch.float32)
                    f_sp_t = torch.tensor(f_sp_rev, device=dev_obj, dtype=torch.float32)
                    f_org_t = torch.tensor(f_org_rev, device=dev_obj, dtype=torch.float32)
                    f_dir_t = torch.tensor(f_dir_rev, device=dev_obj, dtype=torch.float32)

                    if eff_cs > 1:
                        if_sp_rev, _, _ = reverse_metadata(img_sp_fixed, target.origin, target.direction)
                        if_shape_t = torch.tensor(list(img_spatial), device=dev_obj, dtype=torch.float32)
                        if_sp_t = torch.tensor(if_sp_rev, device=dev_obj, dtype=torch.float32)
                    else:
                        if_shape_t, if_sp_t = f_shape_t, f_sp_t

                    M_nf_lvl, b_nf_lvl = get_physical_to_normalized_affine(if_shape_t, if_sp_t, f_org_t, f_dir_t)
                    sc_f_lvl = compute_autograd_physical_scale(if_shape_t, if_sp_t, device=dev_obj, dtype=torch.float32)

                    # Per-subject moving physical-to-normalized composed affines
                    mov_shape_t = torch.tensor(list(mov_lvl_spatial), device=dev_obj, dtype=torch.float32)
                    A_m_list, d_m_list, B_grad_list = [], [], []
                    for bi, g_idx in enumerate(chunk_ids):
                        m_i = mov_seq[g_idx]
                        m_sp_rev, m_org_rev, m_dir_rev = reverse_metadata(mov_sp_lvl, m_i.origin, m_i.direction)
                        m_sp_t = torch.tensor(m_sp_rev, device=dev_obj, dtype=torch.float32)
                        m_org_t = torch.tensor(m_org_rev, device=dev_obj, dtype=torch.float32)
                        m_dir_t = torch.tensor(m_dir_rev, device=dev_obj, dtype=torch.float32)
                        M_nm_i, b_nm_i = get_physical_to_normalized_affine(mov_shape_t, m_sp_t, m_org_t, m_dir_t)
                        sc_m_i = compute_autograd_physical_scale(mov_shape_t, m_sp_t, device=dev_obj, dtype=torch.float32)
                        A_i = M_zyx_b[bi].t() @ M_nm_i
                        d_i = t_zyx_b[bi] @ M_nm_i + b_nm_i
                        B_i = (M_nm_i.t() @ M_zyx_b[bi]) * sc_m_i.view(1, 3)
                        A_m_list.append(A_i)
                        d_m_list.append(d_i.view(1, 1, 1, 3))
                        B_grad_list.append(B_i)
                    A_m_b = torch.stack(A_m_list, dim=0)        # (b_c, 3, 3)
                    d_m_b = torch.stack(d_m_list, dim=0)        # (b_c, 1, 1, 1, 3)
                    B_grad_b = torch.stack(B_grad_list, dim=0)  # (b_c, 3, 3)

                    ws_lvl = min(ws_syn, int(min(img_spatial)))
                    if ws_lvl % 2 == 0:
                        ws_lvl = max(1, ws_lvl - 1)
                    _, _, pad_ratios_lvl = _get_window_count_views(img_spatial, ws_lvl, dev_obj, torch.float32)

                    shrink_ratio = cfl_shrink_ratio(curr_sp_fixed, finest_spacing_fixed)
                    eff_cfl = float(grad_step) * math.sqrt(shrink_ratio)
                    sp_flat = torch.tensor(curr_sp_fixed, device=dev_obj, dtype=torch.float32).reshape(-1)
                    curr_fluid_sig = fluid_sigma_val / float(eff_cs)
                    raw_alpha = (
                        None
                        if sobolev_alpha is None
                        else float(sobolev_alpha) / float(eff_cs * eff_cs)
                    )
                    alpha_sob = float(raw_alpha) if raw_alpha is not None else float(curr_fluid_sig / 2.0)
                    fs_on = bool(fast_smooth) if fast_smooth is not None else False

                    lvl_reg_fn = (
                        build_level_regularizer(
                            reg_mode,
                            alpha=raw_alpha,
                            sobolev_alpha=raw_alpha,
                            spacing=curr_sp_fixed,
                            origin=target.origin,
                            direction=target.direction,
                            fluid_sigma=curr_fluid_sig,
                            param_domain="physical_mm",
                            extra_reg_kwargs=extra_reg_kwargs,
                        )
                        if reg_mode in COUPLED_CONTINUUM_REGULARIZERS
                        else None
                    )

                    lvl_inv_steps = (
                        int(in_loop_inv_steps[min(level_idx, len(in_loop_inv_steps) - 1)])
                        if isinstance(in_loop_inv_steps, (list, tuple))
                        else int(in_loop_inv_steps)
                    )
                    if inverse_steps <= 0:
                        lvl_inv_steps = 0
                    eff_inv_steps = (
                        min(lvl_inv_steps, 7 if level_idx == 0 else 4) if sync_free else lvl_inv_steps
                    )
                    curr_inv_every = max(
                        1,
                        int(inv_refresh_every[min(level_idx, len(inv_refresh_every) - 1)])
                        if isinstance(inv_refresh_every, (list, tuple))
                        else int(inv_refresh_every),
                    )
                    split_inv = bool(b_c * int(np.prod(curr_spatial)) >= 4_194_304)

                    best_loss_b = torch.full((b_c,), float("inf"), device=dev_obj, dtype=torch.float32)
                    best_w_l2r = warp_l2r.clone()
                    best_w_r2l = warp_r2l.clone()
                    sp_dims_w = tuple(range(1, warp_l2r.ndim - 1))

                    def _eval_and_grad_epoch():
                        wl_ev = resize_field(warp_l2r, list(img_spatial), mode="trilinear") if eff_cs > 1 else warp_l2r
                        wr_ev = resize_field(warp_r2l, list(img_spatial), mode="trilinear") if eff_cs > 1 else warp_r2l
                        c_norm = physical_to_normalized_fast(X_phys_img + wl_ev, M_nf_lvl, b_nf_lvl)
                        x_mid_r = (X_phys_img + wr_ev).reshape(b_c, -1, 3)
                        y_norm = torch.bmm(x_mid_r, A_m_b).view(b_c, *img_spatial, 3).add_(d_m_b)
                        del wl_ev, wr_ev, x_mid_r
                        in_b = (
                            ((c_norm >= -1.0) & (c_norm <= 1.0)).all(dim=-1)
                            & ((y_norm >= -1.0) & (y_norm <= 1.0)).all(dim=-1)
                        ).unsqueeze(1).to(torch.float32)
                        if metric_low == "mattes_mi":
                            l_vec, g_cn, g_yn = exact_mattes_mi_2sided_loss_and_grad_3d(
                                I_curr,
                                c_norm,
                                J_curr,
                                y_norm,
                                num_bins=int(cfg_syn.num_bins),
                                fixed_range=(0.0, 1.0),
                                in_bounds_mask=in_b,
                                padding_mode="border",
                                return_per_batch_loss=True,
                            )
                        else:
                            l_vec, g_cn, g_yn = exact_lncc_2sided_loss_and_grad_3d(
                                I_curr,
                                c_norm,
                                J_curr,
                                y_norm,
                                ws_lvl,
                                pad_ratios_lvl,
                                squared=squared_metric,
                                in_bounds_mask=in_b,
                                padding_mode="border",
                                return_per_batch_loss=True,
                            )
                        del c_norm, y_norm, in_b
                        gl_img = (g_cn @ M_nf_lvl.t()).mul_(sc_f_lvl)
                        gr_img = torch.bmm(g_yn.reshape(b_c, -1, 3), B_grad_b).view(b_c, *img_spatial, 3)
                        del g_cn, g_yn
                        if eff_cs > 1:
                            c_mult = float(np.prod(img_spatial)) / float(max(np.prod(curr_spatial), 1))
                            gl = resize_field(gl_img, list(curr_spatial), mode="trilinear").mul_(c_mult)
                            gr = resize_field(gr_img, list(curr_spatial), mode="trilinear").mul_(c_mult)
                            del gl_img, gr_img
                        else:
                            gl, gr = gl_img, gr_img
                        return l_vec, gl, gr

                    for epoch in range(n_epochs):
                        loss_vec, grad_l, grad_r = _eval_and_grad_epoch()
                        imp_mask = (loss_vec < best_loss_b).view(b_c, *([1] * (dim + 1)))
                        best_loss_b = torch.minimum(best_loss_b, loss_vec)
                        torch.where(imp_mask, warp_l2r, best_w_l2r, out=best_w_l2r)
                        torch.where(imp_mask, warp_r2l, best_w_r2l, out=best_w_r2l)

                        grad_l.mul_(b_mask)
                        grad_r.mul_(b_mask)
                        g_pair = torch.cat([grad_l, grad_r], dim=0)
                        del grad_l, grad_r
                        if lvl_reg_fn is not None:
                            g_sm = lvl_reg_fn(g_pair)
                        elif reg_mode == "sobolev":
                            g_sm = apply_sobolev_green_operator(
                                g_pair, fluid_sigma=curr_fluid_sig, alpha=alpha_sob, spacing=curr_sp_fixed
                            )
                            if not fs_on:
                                g_sm = separable_gaussian_filter(g_sm, curr_fluid_sig * 0.5, kernel_type="bessel")
                        else:
                            if fs_on:
                                g_sm = apply_sobolev_green_operator(
                                    g_pair, fluid_sigma=curr_fluid_sig, alpha=curr_fluid_sig / 2.0, spacing=curr_sp_fixed
                                )
                            else:
                                g_sm = separable_gaussian_filter(g_sm if False else g_pair, curr_fluid_sig, kernel_type="bessel")
                        del g_pair
                        grad_l = g_sm[:b_c]
                        grad_r = g_sm[b_c:]
                        del g_sm

                        if restr_mask is not None:
                            grad_l = grad_l.contiguous().mul_(restr_mask)
                            grad_r = grad_r.contiguous().mul_(restr_mask)

                        max_nl = _scaled_sq_sum(grad_l, sp_flat, divide=True).amax(dim=sp_dims_w).sqrt()
                        max_nr = _scaled_sq_sum(grad_r, sp_flat, divide=True).amax(dim=sp_dims_w).sqrt()
                        sc_l = torch.where(
                            max_nl > 1e-12, eff_cfl / torch.clamp(max_nl, min=1e-8), torch.zeros_like(max_nl)
                        ).view(b_c, *([1] * (dim + 1)))
                        sc_r = torch.where(
                            max_nr > 1e-12, eff_cfl / torch.clamp(max_nr, min=1e-8), torch.zeros_like(max_nr)
                        ).view(b_c, *([1] * (dim + 1)))
                        delta_l = grad_l.contiguous().mul_(sc_l)
                        delta_r = grad_r.contiguous().mul_(sc_r)
                        del grad_l, grad_r

                        if antisymmetric:
                            delta_l.sub_(delta_r).mul_(0.5)
                            delta_r = -delta_l

                        cn_l = physical_to_normalized_torch_cached(
                            X_phys + warp_l2r, f_shape_t, f_sp_t, f_org_t, f_dir_t
                        )
                        dl_pb = F.grid_sample(
                            delta_l.movedim(-1, 1), cn_l, padding_mode="border", align_corners=True
                        ).movedim(1, -1)
                        del cn_l, delta_l
                        warp_l2r.sub_(dl_pb)
                        del dl_pb

                        cn_r = physical_to_normalized_torch_cached(
                            X_phys + warp_r2l, f_shape_t, f_sp_t, f_org_t, f_dir_t
                        )
                        dr_pb = F.grid_sample(
                            delta_r.movedim(-1, 1), cn_r, padding_mode="border", align_corners=True
                        ).movedim(1, -1)
                        del cn_r, delta_r
                        warp_r2l.sub_(dr_pb)
                        del dr_pb

                        if elastic_sigma_val > 0.0:
                            warp_l2r.copy_(separable_gaussian_filter(warp_l2r, elastic_sigma_val, kernel_type="bessel"))
                            warp_r2l.copy_(separable_gaussian_filter(warp_r2l, elastic_sigma_val, kernel_type="bessel"))

                        if lvl_inv_steps > 0 and (((epoch + 1) % curr_inv_every == 0) or (epoch == n_epochs - 1)):
                            if split_inv:
                                warp_l2r_inv = update_inverse_field_nd(
                                    warp_l2r,
                                    warp_l2r_inv,
                                    steps=eff_inv_steps,
                                    method=inverse_method,
                                    spacing=curr_sp_fixed,
                                    origin=target.origin,
                                    direction=target.direction,
                                    X_phys=X_phys,
                                    max_error_threshold=-1.0,
                                    mean_error_threshold=-1.0,
                                )
                                warp_r2l_inv = update_inverse_field_nd(
                                    warp_r2l,
                                    warp_r2l_inv,
                                    steps=eff_inv_steps,
                                    method=inverse_method,
                                    spacing=curr_sp_fixed,
                                    origin=target.origin,
                                    direction=target.direction,
                                    X_phys=X_phys,
                                    max_error_threshold=-1.0,
                                    mean_error_threshold=-1.0,
                                )
                            else:
                                fwd_p = torch.cat([warp_l2r, warp_r2l], dim=0)
                                ini_p = torch.cat([warp_l2r_inv, warp_r2l_inv], dim=0)
                                out_p = update_inverse_field_nd(
                                    fwd_p,
                                    ini_p,
                                    steps=eff_inv_steps,
                                    method=inverse_method,
                                    spacing=curr_sp_fixed,
                                    origin=target.origin,
                                    direction=target.direction,
                                    X_phys=X_phys,
                                    max_error_threshold=-1.0,
                                    mean_error_threshold=-1.0,
                                )
                                del fwd_p, ini_p
                                warp_l2r_inv = out_p[:b_c]
                                warp_r2l_inv = out_p[b_c:]
                                del out_p

                        if stationary_boundary:
                            for _w_sb in (warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv):
                                _w_sb.mul_(b_mask)

                    # End-of-level per-subject best-state restore & inverse polish
                    fin_loss_vec, _gl_fin, _gr_fin = _eval_and_grad_epoch()
                    del _gl_fin, _gr_fin
                    revert_m = (fin_loss_vec >= best_loss_b).view(b_c, *([1] * (dim + 1)))
                    torch.where(revert_m, best_w_l2r, warp_l2r, out=warp_l2r)
                    torch.where(revert_m, best_w_r2l, warp_r2l, out=warp_r2l)
                    del best_w_l2r, best_w_r2l

                    if inverse_steps > 0 and polish_inv_steps > 0:
                        p_tol = float(polish_inv_tol) if polish_inv_tol is not None else 0.02
                        warp_l2r_inv = update_inverse_field_nd(
                            warp_l2r,
                            warp_l2r_inv,
                            steps=int(polish_inv_steps),
                            method=inverse_method,
                            spacing=curr_sp_fixed,
                            origin=target.origin,
                            direction=target.direction,
                            X_phys=X_phys,
                            check_interval=5,
                            max_error_threshold=p_tol,
                            mean_error_threshold=p_tol * 0.01,
                        )
                        warp_r2l_inv = update_inverse_field_nd(
                            warp_r2l,
                            warp_r2l_inv,
                            steps=int(polish_inv_steps),
                            method=inverse_method,
                            spacing=curr_sp_fixed,
                            origin=target.origin,
                            direction=target.direction,
                            X_phys=X_phys,
                            check_interval=5,
                            max_error_threshold=p_tol,
                            mean_error_threshold=p_tol * 0.01,
                        )
                    J_lv[level_idx] = None
                    del X_phys, X_phys_img, b_mask

            del J_lv
            t_opt_per_subj = (time.time() - t0_opt) / max(b_c, 1)

            X_phys_full = get_physical_grid_torch(
                full_shape_zyx, target.spacing, target.origin, target.direction, device=dev_obj, dtype=torch.float32
            )
            c_sp_rev, c_org_rev, c_dir_rev = reverse_metadata(target.spacing, target.origin, target.direction)
            c_shape_t = torch.tensor(list(full_shape_zyx), device=dev_obj, dtype=torch.float32)
            c_sp_t = torch.tensor(c_sp_rev, device=dev_obj, dtype=torch.float32)
            c_org_t = torch.tensor(c_org_rev, device=dev_obj, dtype=torch.float32)
            c_dir_t = torch.tensor(c_dir_rev, device=dev_obj, dtype=torch.float32)
            b_mask_full = get_boundary_mask(full_shape_zyx, dev_obj, torch.float32) if stationary_boundary else None

            for bi, g_idx in enumerate(chunk_ids):
                t0_comp = time.time()
                m_img = mov_seq[g_idx]
                res_init = resolved_list[bi]
                op = outprefixes[g_idx] if outprefixes is not None else None
                do_w = bool(write_transforms or op is not None)

                with torch.no_grad():
                    wl_i = warp_l2r[bi : bi + 1]
                    wr_i = warp_r2l[bi : bi + 1]
                    wli_i = warp_l2r_inv[bi : bi + 1]
                    wri_i = warp_r2l_inv[bi : bi + 1]
                    if tuple(wl_i.shape[1:-1]) != full_shape_zyx:
                        wl_i = resize_field(wl_i, list(full_shape_zyx), mode="trilinear")
                        wr_i = resize_field(wr_i, list(full_shape_zyx), mode="trilinear")
                        wli_i = resize_field(wli_i, list(full_shape_zyx), mode="trilinear")
                        wri_i = resize_field(wri_i, list(full_shape_zyx), mode="trilinear")
                    else:
                        wl_i = wl_i.clone()
                        wr_i = wr_i.clone()
                        wli_i = wli_i.clone()
                        wri_i = wri_i.clone()

                    if b_mask_full is not None:
                        wl_i.mul_(b_mask_full)
                        wr_i.mul_(b_mask_full)
                        wli_i.mul_(b_mask_full)
                        wri_i.mul_(b_mask_full)

                    run_def = sum(reg_iters_list) > 0 or res_init.has_initial_warp
                    dd_ice_tol = (
                        float(compose_polish_tol)
                        if compose_polish_tol is not None and float(compose_polish_tol) > 0.0
                        else 0.30
                    )
                    if digital_diffeomorphism and run_def and inverse_steps > 0:
                        wl_i, wli_i = enforce_digital_diffeomorphism_nd(
                            wl_i,
                            wli_i,
                            spacing=target.spacing,
                            origin=target.origin,
                            direction=target.direction,
                            X_phys=X_phys_full,
                            ice_tol=dd_ice_tol,
                            jac_floor=digital_diffeomorphism_floor,
                            steps=digital_diffeomorphism_steps,
                            stationary_boundary=stationary_boundary,
                        )
                        wr_i, wri_i = enforce_digital_diffeomorphism_nd(
                            wr_i,
                            wri_i,
                            spacing=target.spacing,
                            origin=target.origin,
                            direction=target.direction,
                            X_phys=X_phys_full,
                            ice_tol=dd_ice_tol,
                            jac_floor=digital_diffeomorphism_floor,
                            steps=digital_diffeomorphism_steps,
                            stationary_boundary=stationary_boundary,
                        )

                    cn_fwd = physical_to_normalized_torch_cached(
                        X_phys_full + wli_i, c_shape_t, c_sp_t, c_org_t, c_dir_t
                    )
                    fwd_comp = wli_i + F.grid_sample(
                        wr_i.movedim(-1, 1), cn_fwd, padding_mode="border", align_corners=True
                    ).movedim(1, -1)
                    del cn_fwd

                    cn_inv = physical_to_normalized_torch_cached(
                        X_phys_full + wri_i, c_shape_t, c_sp_t, c_org_t, c_dir_t
                    )
                    inv_comp = wri_i + F.grid_sample(
                        wl_i.movedim(-1, 1), cn_inv, padding_mode="border", align_corners=True
                    ).movedim(1, -1)
                    del cn_inv, wri_i

                    if b_mask_full is not None:
                        fwd_comp.mul_(b_mask_full)
                        inv_comp.mul_(b_mask_full)

                    if run_def and inverse_steps > 0 and compose_polish_tol is not None and float(compose_polish_tol) > 0.0:
                        cp_tol = float(compose_polish_tol)
                        inv_comp = update_inverse_field_nd_hybrid_lm(
                            fwd_comp,
                            inv_comp,
                            steps=max(int(inverse_steps), 25),
                            max_error_threshold=cp_tol,
                            mean_error_threshold=min(cp_tol * 0.5, 0.02),
                            spacing=target.spacing,
                            origin=target.origin,
                            direction=target.direction,
                            X_phys=X_phys_full,
                            stationary_boundary=stationary_boundary,
                        )

                    if digital_diffeomorphism and run_def and inverse_steps > 0:
                        fwd_pre = fwd_comp
                        fwd_comp, inv_comp = enforce_digital_diffeomorphism_nd(
                            fwd_comp,
                            inv_comp,
                            spacing=target.spacing,
                            origin=target.origin,
                            direction=target.direction,
                            X_phys=X_phys_full,
                            ice_tol=dd_ice_tol,
                            jac_floor=digital_diffeomorphism_floor,
                            steps=digital_diffeomorphism_steps,
                            stationary_boundary=stationary_boundary,
                        )
                        if fwd_comp is not fwd_pre:
                            d_dd = fwd_comp - fwd_pre
                            wl_i, wr_i, wli_i, _ = pullback_digital_diffeomorphism_to_halves(
                                d_dd,
                                wl_i,
                                wr_i,
                                wli_i,
                                None,
                                spacing=target.spacing,
                                origin=target.origin,
                                direction=target.direction,
                                X_phys=X_phys_full,
                                jac_floor=digital_diffeomorphism_floor,
                                stationary_boundary=stationary_boundary,
                            )
                            del d_dd
                        del fwd_pre

                    ice_c = calculate_inverse_identity_error(
                        fwd_comp,
                        inv_comp,
                        spacing=target.spacing,
                        origin=target.origin,
                        direction=target.direction,
                        return_error_map=False,
                    )
                    _fc_ref, _ic_ref = fwd_comp, inv_comp

                    def _load_comp_err(_f=_fc_ref, _i=_ic_ref):
                        with torch.no_grad():
                            return calculate_inverse_identity_error(
                                _f,
                                _i,
                                spacing=target.spacing,
                                origin=target.origin,
                                direction=target.direction,
                                return_error_map=True,
                            )["error_map"].cpu()

                    inv_err_dict = {"composed": _LazyKeyDict(ice_c, lazy_loaders={"error_map": _load_comp_err})}
                t_comp_i = time.time() - t0_comp

                M_p, t_p = res_init.M_phys, res_init.t_phys
                has_lin = bool(res_init.has_linear_affine and not is_syn_only)
                fwd_tx_paths: List[str] = []
                inv_tx_paths: List[str] = []
                wti_inv: List[bool] = []

                if do_w:
                    disp_img_i = disp_tensor_to_itk(fwd_comp, fixed_img)
                    inv_img_i = disp_tensor_to_itk(inv_comp, fixed_img)
                    comp_lvl = 4 if op is not None else 1
                    if op is not None:
                        os.makedirs(os.path.dirname(str(op)) or ".", exist_ok=True)
                        fwd_f = f"{op}1Warp.nii.gz"
                        inv_f = f"{op}1InverseWarp.nii.gz"
                        aff_f = f"{op}0GenericAffine.mat" if has_lin else None
                    else:
                        fwd_f = tempfile.NamedTemporaryFile(suffix="_fwd_warp.nii.gz", delete=False).name
                        inv_f = tempfile.NamedTemporaryFile(suffix="_inv_warp.nii.gz", delete=False).name
                        aff_f = tempfile.NamedTemporaryFile(suffix=".mat", delete=False).name if has_lin else None
                    if aff_f is not None:
                        export_ants_affine_transform(M_p, t_p, dim=dim, filename=aff_f)
                    _write_ants_images_fast([(disp_img_i, fwd_f), (inv_img_i, inv_f)], compresslevel=comp_lvl)
                    _cache_disp_image_and_field(fwd_f, disp_img_i, dev_str, disp_tensor=fwd_comp)
                    _cache_disp_image_and_field(inv_f, inv_img_i, dev_str, disp_tensor=inv_comp)
                    fwd_tx_paths = [fwd_f, aff_f] if aff_f is not None else [fwd_f]
                    inv_tx_paths = [aff_f, inv_f] if aff_f is not None else [inv_f]
                    wti_inv = [True, False] if aff_f is not None else [False]
                    fwd_chain = TransformChain.from_paths(fwd_tx_paths)
                    inv_chain = TransformChain.from_paths(inv_tx_paths, wti_inv)
                    aff_out: Any = aff_f
                else:
                    fwd_el = TransformElement.from_disp_tensor(fwd_comp, fixed_img, kind="displacement")
                    inv_el = TransformElement.from_disp_tensor(inv_comp, fixed_img, kind="inverse_displacement")
                    if has_lin:
                        aff_el = TransformElement.from_affine(M_p, t_p, invert=False)
                        aff_inv_el = TransformElement.from_affine(M_p, t_p, invert=True)
                        fwd_chain = TransformChain([fwd_el, aff_el])
                        inv_chain = TransformChain([aff_inv_el, inv_el])
                        wti_inv = [True, False]
                        aff_out = TransformChain([aff_el])
                    else:
                        fwd_chain = TransformChain([fwd_el])
                        inv_chain = TransformChain([inv_el])
                        wti_inv = [False]
                        aff_out = None

                warped_mov_i = None
                warped_mov_ten_i = None
                warped_fix_i = None
                if return_warped or _return_warped_tensor:
                    mov_cf = (
                        ContinuousField.from_tensor(
                            mi_raw_unit_list[bi],
                            spatial_shape=m_img.shape,
                            origin=m_img.origin,
                            spacing=m_img.spacing,
                            direction=m_img.direction,
                            is_vector=False,
                        )
                        if mi_raw_unit_list[bi] is not None
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
                    if return_warped:
                        fix_cf = (
                            ContinuousField.from_tensor(
                                target.tensor_zyx,
                                spatial_shape=fixed_img.shape,
                                origin=fixed_img.origin,
                                spacing=fixed_img.spacing,
                                direction=fixed_img.direction,
                                is_vector=False,
                            )
                            if target.is_unclipped_unit
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

                offload_mid = n_vox >= 4_194_304
                sub_model = SyNTo(
                    dim=dim,
                    grid_shape=full_shape_zyx,
                    spacing=target.spacing,
                    origin=target.origin,
                    direction=target.direction,
                    fluid_sigma=fluid_sigma_val,
                    elastic_sigma=elastic_sigma_val,
                    inverse_method=inverse_method,
                    inverse_steps=inverse_steps,
                    in_loop_inv_steps=in_loop_inv_steps,
                    interpolator=interpolator,
                    stationary_boundary=stationary_boundary,
                    antisymmetric=antisymmetric,
                    allocate_fields=False,
                )
                sub_model.formulation = "lagrangian"
                sub_model.warp_l2r = torch.nn.Parameter(fwd_comp, requires_grad=False)
                sub_model.warp_l2r.is_physical = True
                sub_model.warp_r2l = torch.nn.Parameter(inv_comp, requires_grad=False)
                sub_model.warp_r2l.is_physical = True
                sub_model.midpoint_warp_l2r = torch.nn.Parameter(
                    wl_i.cpu() if offload_mid else wl_i, requires_grad=False
                )
                sub_model.midpoint_warp_l2r.is_physical = True
                sub_model.midpoint_warp_r2l = torch.nn.Parameter(
                    wr_i.cpu() if offload_mid else wr_i, requires_grad=False
                )
                sub_model.midpoint_warp_r2l.is_physical = True
                sub_model.midpoint_warp_l2r_inv = torch.nn.Parameter(
                    wli_i.cpu() if offload_mid else wli_i, requires_grad=False
                )
                sub_model.midpoint_warp_l2r_inv.is_physical = True

                prov_i = {
                    "algorithm": "syntx.batch_syn",
                    "formulation": "lagrangian",
                    "batch_size": b_c,
                    "optimizer": "cfl",
                    "regularizer": reg_mode,
                    "sobolev_alpha": sobolev_alpha,
                    "reg_iterations": reg_iters_list,
                    "levels": levels_list,
                    "grad_step": grad_step,
                    "flow_sigma": flow_sigma,
                    "control_stride": raw_cstride,
                    "sync_free": bool(sync_free),
                    "write_transforms": bool(do_w),
                    "digital_diffeomorphism": digital_diffeomorphism,
                    "runtime_affine_sec": round(t_aff_per_subj, 3),
                    "runtime_deformable_sec": round(t_opt_per_subj, 3),
                    "runtime_compose_sec": round(t_comp_i, 3),
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
                    "provenance": prov_i,
                    "inverse_identity_errors": inv_err_dict,
                    "affine_transform": aff_out,
                }
                if _return_warped_tensor and warped_mov_ten_i is not None:
                    res_entry["warpedmovout_tensor"] = warped_mov_ten_i
                if _return_disp_tensor:
                    res_entry["disp_tensor_zyx"] = fwd_comp
                results[g_idx] = res_entry
            del warp_l2r, warp_r2l, warp_l2r_inv, warp_r2l_inv, X_phys_full, mi_raw_unit_list

    del I_lv
    if not caller_owned_target:
        del target
        cleanup_gpu(dev_str)

    _ = time.time() - t0_all
    return [r for r in results if r is not None]


def batch_mattes_mi(
    fixed: Union[ants.ANTsImage, FixedTarget, torch.Tensor, np.ndarray],
    moving: Union[Sequence[Any], torch.Tensor, np.ndarray],
    *,
    num_bins: int = 32,
    sampling_percentage: Optional[float] = None,
    mask: Optional[Union[ants.ANTsImage, torch.Tensor, np.ndarray]] = None,
    auto_mask: bool = False,
    fixed_range: Optional[Tuple[float, float]] = None,
    return_tensor: bool = False,
    device: Union[str, torch.device, None] = "auto",
) -> Union[np.ndarray, torch.Tensor]:
    """Vectorized GPU Batch Mattes Mutual Information evaluator across ``B`` moving images/tensors
    against a shared ``fixed`` reference (``FixedTarget``, ``ANTsImage``, ``ndarray``, or ``Tensor``).

    Prepacks the fixed image's 4-tap cubic B-spline Parzen weights once on GPU (`8x` smaller than
    dense ``(N, num_bins)`` weights) and evaluates all ``B`` joint histograms in parallel with
    deterministic zero-atomic register reduction (``RegisterParzenMattesMIFunction``).
    Differentiable with respect to ``moving`` when ``moving`` is a ``torch.Tensor`` with
    ``requires_grad=True``. Returns negative MI in nats (lower = higher similarity, matching
    ``syntx.image_compare(..., metric='mattes')``).
    """
    from .core.kernels import RegisterParzenMattesMIFunction, prepack_fixed_parzen_sparse4

    dev_req = None if (device is None or str(device).lower() == "auto") else str(device)
    if isinstance(moving, torch.Tensor) and device == "auto":
        dev_obj = moving.device
    elif bool(getattr(fixed, "_is_syntx_fixed_target", False)) and device == "auto":
        dev_obj = fixed.device
    elif isinstance(fixed, torch.Tensor) and device == "auto":
        dev_obj = fixed.device
    else:
        dev_obj = torch.device(resolve_device(dev_req, task="reg"))

    def _to_zyx_1d(item: Any) -> torch.Tensor:
        if bool(getattr(item, "_is_syntx_fixed_target", False)):
            return item.tensor_zyx.to(device=dev_obj, dtype=torch.float32).reshape(-1)
        if hasattr(item, "numpy") and hasattr(item, "dimension"):
            arr = np.ascontiguousarray(item.numpy(), dtype=np.float32)
            rev_axes = tuple(range(arr.ndim - 1, -1, -1))
            return torch.from_numpy(arr).to(device=dev_obj, dtype=torch.float32).permute(*rev_axes).reshape(-1)
        if isinstance(item, torch.Tensor):
            return item.to(device=dev_obj, dtype=torch.float32).reshape(-1)
        return torch.as_tensor(np.asarray(item, dtype=np.float32), device=dev_obj, dtype=torch.float32).reshape(-1)

    y_1d = _to_zyx_1d(fixed)
    n_elem = int(y_1d.numel())

    need_grad = isinstance(moving, torch.Tensor) and bool(moving.requires_grad)
    if isinstance(moving, torch.Tensor):
        if moving.ndim >= 2 and int(moving[0].numel()) == n_elem:
            x_2d = moving.to(device=dev_obj, dtype=torch.float32).reshape(int(moving.shape[0]), n_elem)
        elif int(moving.numel()) == n_elem:
            x_2d = moving.to(device=dev_obj, dtype=torch.float32).reshape(1, n_elem)
        else:
            raise ValueError(
                f"batch_mattes_mi: moving tensor shape {tuple(moving.shape)} incompatible with fixed numel={n_elem}"
            )
    elif isinstance(moving, np.ndarray) and moving.ndim >= 2 and int(moving[0].size) == n_elem:
        x_2d = torch.as_tensor(moving, device=dev_obj, dtype=torch.float32).reshape(int(moving.shape[0]), n_elem)
    else:
        mov_items = list(moving)
        if len(mov_items) == 0:
            empty_t = torch.empty((0,), device=dev_obj, dtype=torch.float32)
            return empty_t if return_tensor else np.empty((0,), dtype=np.float32)
        need_grad = any(isinstance(it, torch.Tensor) and bool(it.requires_grad) for it in mov_items)
        x_2d = torch.stack([_to_zyx_1d(it) for it in mov_items], dim=0)

    b_sz = int(x_2d.shape[0])
    if int(x_2d.shape[1]) != n_elem:
        raise ValueError(
            f"batch_mattes_mi: moving voxel count {x_2d.shape[1]} must match fixed voxel count {n_elem}"
        )

    with torch.no_grad():
        valid_mask: Optional[torch.Tensor] = None
        if mask is not None:
            valid_mask = _to_zyx_1d(mask) > 0.5
        if auto_mask:
            fg_y = y_1d.abs() > 0.01 * y_1d.detach().abs().max()
            fg_x = (x_2d.detach().abs() > 0.01 * x_2d.detach().abs().amax(dim=1, keepdim=True)).any(dim=0)
            fg_u = fg_y | fg_x
            valid_mask = (valid_mask & fg_u) if valid_mask is not None else fg_u
        sel_idx = valid_mask.nonzero(as_tuple=False).squeeze(-1) if valid_mask is not None else None
        if sampling_percentage is not None and float(sampling_percentage) < 1.0:
            base_cnt = int(sel_idx.numel()) if sel_idx is not None else n_elem
            if base_cnt > 0:
                sub_i = mattes_sample_indices(base_cnt, float(sampling_percentage), dev_obj)
                sel_idx = sel_idx.index_select(0, sub_i) if sel_idx is not None else sub_i

    if sel_idx is not None:
        if int(sel_idx.numel()) == 0:
            z_out = x_2d.sum(dim=1) * 0.0
            return z_out if (return_tensor or need_grad) else z_out.detach().cpu().numpy().astype(np.float32)
        y_sub = y_1d.index_select(0, sel_idx)
        x_sub = x_2d.index_select(1, sel_idx)
    else:
        y_sub = y_1d
        x_sub = x_2d

    if fixed_range is None:
        min_j = y_sub.amin().detach()
        max_j = y_sub.amax().detach()
        min_i = x_sub.amin(dim=1, keepdim=True).detach()
        max_i = x_sub.amax(dim=1, keepdim=True).detach()
    else:
        min_i = float(fixed_range[0])
        max_i = float(fixed_range[1])
        min_j = float(fixed_range[0])
        max_j = float(fixed_range[1])

    y_sc = (y_sub - min_j) / (max_j - min_j + 1e-8) * 2.0 - 1.0
    x_sc = (x_sub - min_i) / (max_i - min_i + 1e-8) * 2.0 - 1.0
    fi_idx4, fi_w4, _ = prepack_fixed_parzen_sparse4(
        y_sc.detach(), num_bins=int(num_bins), min_val=-1.0, max_val=1.0, pad=2.0
    )
    ctx_mgr = torch.enable_grad() if need_grad else torch.no_grad()
    with ctx_mgr:
        losses = RegisterParzenMattesMIFunction.apply(
            x_sc, fi_idx4, fi_w4, int(num_bins), -1.0, 1.0, 2.0, None
        )
    if return_tensor or need_grad:
        return losses
    return losses.detach().cpu().numpy().astype(np.float32)


def _prepare_batch_spatial_tensors(
    fixed: Union[ants.ANTsImage, FixedTarget, torch.Tensor, np.ndarray],
    moving: Union[Sequence[Any], torch.Tensor, np.ndarray],
    device: Union[str, torch.device, None],
) -> Tuple[torch.Tensor, torch.Tensor, bool, torch.device]:
    """Standardize ``fixed`` to ``(1, 1, *spatial_zyx)`` and ``moving`` to ``(B, 1, *spatial_zyx)`` on GPU."""
    dev_req = None if (device is None or str(device).lower() == "auto") else str(device)
    if isinstance(moving, torch.Tensor) and device == "auto":
        dev_obj = moving.device
    elif bool(getattr(fixed, "_is_syntx_fixed_target", False)) and device == "auto":
        dev_obj = fixed.device
    elif isinstance(fixed, torch.Tensor) and device == "auto":
        dev_obj = fixed.device
    else:
        dev_obj = torch.device(resolve_device(dev_req, task="reg"))

    def _to_5d_or_4d(item: Any) -> torch.Tensor:
        if bool(getattr(item, "_is_syntx_fixed_target", False)):
            if item.is_unclipped_unit:
                return item.tensor_zyx.to(device=dev_obj, dtype=torch.float32)
            raw_cached = getattr(item, "_raw_tensor_zyx", None)
            if raw_cached is None or raw_cached.device != dev_obj:
                raw_cached = image_to_tensor(item.image, device=dev_obj, to_zyx=True).to(torch.float32)
                item._raw_tensor_zyx = raw_cached
            return raw_cached
        if hasattr(item, "numpy") and hasattr(item, "dimension"):
            arr = np.ascontiguousarray(item.numpy(), dtype=np.float32)
            rev_axes = tuple(range(arr.ndim - 1, -1, -1))
            return (
                torch.from_numpy(arr)
                .to(device=dev_obj, dtype=torch.float32)
                .permute(*rev_axes)
                .contiguous()
                .unsqueeze(0)
                .unsqueeze(0)
            )
        t = item if isinstance(item, torch.Tensor) else torch.as_tensor(np.asarray(item, dtype=np.float32))
        t = t.to(device=dev_obj, dtype=torch.float32)
        if t.ndim in (2, 3):
            return t.unsqueeze(0).unsqueeze(0)
        if t.ndim in (4, 5):
            return t
        raise ValueError(f"Expected 2D/3D spatial image or (B, 1, *spatial) tensor, got shape {tuple(t.shape)}")

    fi_t = _to_5d_or_4d(fixed)
    if int(fi_t.shape[0]) != 1 or int(fi_t.shape[1]) != 1:
        raise ValueError(f"fixed must be single-subject single-channel (1, 1, *spatial), got {tuple(fi_t.shape)}")
    sp_shape = tuple(int(s) for s in fi_t.shape[2:])

    need_grad = isinstance(moving, torch.Tensor) and bool(moving.requires_grad)
    if isinstance(moving, torch.Tensor):
        if moving.ndim == len(sp_shape) + 2 and tuple(int(s) for s in moving.shape[2:]) == sp_shape:
            mi_t = moving.to(device=dev_obj, dtype=torch.float32)
        elif moving.ndim == len(sp_shape) + 1 and tuple(int(s) for s in moving.shape[1:]) == sp_shape:
            mi_t = moving.to(device=dev_obj, dtype=torch.float32).unsqueeze(1)
        elif moving.ndim == len(sp_shape) and tuple(int(s) for s in moving.shape) == sp_shape:
            mi_t = moving.to(device=dev_obj, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"moving tensor shape {tuple(moving.shape)} incompatible with fixed spatial {sp_shape}")
    elif isinstance(moving, np.ndarray) and moving.ndim == len(sp_shape) + 1 and tuple(moving.shape[1:]) == sp_shape:
        mi_t = torch.as_tensor(moving, device=dev_obj, dtype=torch.float32).unsqueeze(1)
    else:
        mov_items = list(moving)
        if len(mov_items) == 0:
            return fi_t, torch.empty((0, 1, *sp_shape), device=dev_obj, dtype=torch.float32), False, dev_obj
        need_grad = any(isinstance(it, torch.Tensor) and bool(it.requires_grad) for it in mov_items)
        mi_t = torch.cat([_to_5d_or_4d(it) for it in mov_items], dim=0)

    if tuple(int(s) for s in mi_t.shape[2:]) != sp_shape or int(mi_t.shape[1]) != 1:
        raise ValueError(
            f"moving shape {tuple(mi_t.shape)} must have channel=1 and spatial shape {sp_shape}"
        )
    return fi_t, mi_t, need_grad, dev_obj


def batch_cc(
    fixed: Union[ants.ANTsImage, FixedTarget, torch.Tensor, np.ndarray],
    moving: Union[Sequence[Any], torch.Tensor, np.ndarray],
    *,
    window_size: int = 5,
    lncc_radius: Optional[int] = None,
    squared: bool = False,
    mask: Optional[Union[ants.ANTsImage, torch.Tensor, np.ndarray]] = None,
    var_floor: float = 1e-6,
    return_tensor: bool = False,
    device: Union[str, torch.device, None] = "auto",
) -> Union[np.ndarray, torch.Tensor]:
    """Vectorized GPU Batch Local Normalized Cross-Correlation (``LNCC`` / ``cc``, or ``CC^2`` when ``squared=True``).

    Precomputes ``fixed``'s local mean-subtracted and local variance buffers once on GPU (and caches
    them on ``FixedTarget`` across repeated cohort calls), then evaluates all ``B`` moving volumes in
    memory-bounded GPU batches using 3-pass separable Metal ``fused_box_mean_3d`` (and exact
    ``BatchedLNCC3DFunction`` transpose backward when ``moving`` requires gradients).
    Returns negative LNCC in ``[-1, 1]`` (or ``[-1, 0]`` when ``squared=True``; lower = higher similarity,
    matching ``syntx.image_compare(..., 'cc')`` and ``syntx.core.losses.local_ncc_loss_nd``).
    """
    from .core.kernels import BatchedLNCC3DFunction, fused_box_mean_3d
    from .core.losses.cross_correlation import local_ncc_loss_nd

    ws = (2 * int(lncc_radius) + 1) if lncc_radius is not None else int(window_size)
    if ws < 1 or ws % 2 == 0:
        raise ValueError(f"batch_cc: window_size must be a positive odd integer, got {ws}")

    fi_t, mi_t, need_grad, dev_obj = _prepare_batch_spatial_tensors(fixed, moving, device)
    b_sz = int(mi_t.shape[0])
    if b_sz == 0:
        empty_t = torch.empty((0,), device=dev_obj, dtype=torch.float32)
        return empty_t if return_tensor else np.empty((0,), dtype=np.float32)

    dim = fi_t.ndim - 2
    min_sp = min(int(s) for s in fi_t.shape[2:])
    if ws > min_sp:
        ws = min_sp if (min_sp % 2 == 1) else max(1, min_sp - 1)
    fl = max(float(var_floor), 1e-6)

    mask_t: Optional[torch.Tensor] = None
    if mask is not None:
        _, mask_5d, _, _ = _prepare_batch_spatial_tensors(fixed, [mask], dev_obj)
        mask_t = mask_5d > 0.5

    if dim != 3:
        ctx_2d = torch.enable_grad() if need_grad else torch.no_grad()
        with ctx_2d:
            losses_2d = torch.stack(
                [
                    local_ncc_loss_nd(
                        mi_t[bi : bi + 1], fi_t, mask=mask_t, window_size=ws, squared=bool(squared), var_floor=fl
                    )
                    for bi in range(b_sz)
                ]
            )
        return losses_2d if (return_tensor or need_grad) else losses_2d.detach().cpu().numpy().astype(np.float32)

    # Precompute / reuse fixed local moments once on GPU
    is_ft = bool(getattr(fixed, "_is_syntx_fixed_target", False))
    cache_dict = getattr(fixed, "_raw_lncc_cache", None) if is_ft else None
    cache_key = (ws, fl, str(dev_obj))
    if cache_dict is not None and cache_key in cache_dict:
        m_c, j_var, safe_j_var = cache_dict[cache_key]
    else:
        with torch.no_grad():
            j_mean = fused_box_mean_3d(fi_t, ws, count_include_pad=False)
            m_c = fi_t - j_mean
            del j_mean
            j_var = torch.clamp(fused_box_mean_3d(m_c.square(), ws, count_include_pad=False), min=0.0)
            safe_j_var = torch.clamp(j_var, min=fl)
        if is_ft:
            if cache_dict is None:
                fixed._raw_lncc_cache = {}
            fixed._raw_lncc_cache[cache_key] = (m_c, j_var, safe_j_var)

    j_valid = ((j_var > fl) & mask_t) if mask_t is not None else ((j_var > fl) if False else None)
    n_vox = int(fi_t[0, 0].numel())
    chunk_b = max(1, min(b_sz, 67_108_864 // max(1, n_vox)))

    ctx_mgr = torch.enable_grad() if need_grad else torch.no_grad()
    with ctx_mgr:
        if chunk_b >= b_sz:
            losses = BatchedLNCC3DFunction.apply(mi_t, m_c, safe_j_var, ws, bool(squared), fl, j_valid)
        else:
            losses = torch.cat(
                [
                    BatchedLNCC3DFunction.apply(
                        mi_t[s : s + chunk_b], m_c, safe_j_var, ws, bool(squared), fl, j_valid
                    )
                    for s in range(0, b_sz, chunk_b)
                ],
                dim=0,
            )
    if return_tensor or need_grad:
        return losses
    return losses.detach().cpu().numpy().astype(np.float32)


def batch_cc2(
    fixed: Union[ants.ANTsImage, FixedTarget, torch.Tensor, np.ndarray],
    moving: Union[Sequence[Any], torch.Tensor, np.ndarray],
    *,
    window_size: int = 5,
    lncc_radius: Optional[int] = None,
    mask: Optional[Union[ants.ANTsImage, torch.Tensor, np.ndarray]] = None,
    var_floor: float = 1e-6,
    return_tensor: bool = False,
    device: Union[str, torch.device, None] = "auto",
) -> Union[np.ndarray, torch.Tensor]:
    """Vectorized GPU Batch Squared Local Cross-Correlation (``CC^2`` in ``[-1, 0]``; lower = higher similarity)."""
    return batch_cc(
        fixed,
        moving,
        window_size=window_size,
        lncc_radius=lncc_radius,
        squared=True,
        mask=mask,
        var_floor=var_floor,
        return_tensor=return_tensor,
        device=device,
    )


def batch_correlation(
    fixed: Union[ants.ANTsImage, FixedTarget, torch.Tensor, np.ndarray],
    moving: Union[Sequence[Any], torch.Tensor, np.ndarray],
    *,
    mask: Optional[Union[ants.ANTsImage, torch.Tensor, np.ndarray]] = None,
    return_tensor: bool = False,
    device: Union[str, torch.device, None] = "auto",
) -> Union[np.ndarray, torch.Tensor]:
    """Vectorized GPU Batch Global Pearson Correlation (``+1.0`` = identical, matching ``syntx.correlation``)."""
    fi_t, mi_t, need_grad, dev_obj = _prepare_batch_spatial_tensors(fixed, moving, device)
    b_sz = int(mi_t.shape[0])
    if b_sz == 0:
        empty_t = torch.empty((0,), device=dev_obj, dtype=torch.float32)
        return empty_t if return_tensor else np.empty((0,), dtype=np.float32)

    ctx_mgr = torch.enable_grad() if need_grad else torch.no_grad()
    with ctx_mgr:
        f_flat = fi_t.reshape(1, -1)
        m_flat = mi_t.reshape(b_sz, -1)
        if mask is not None:
            _, mask_5d, _, _ = _prepare_batch_spatial_tensors(fixed, [mask], dev_obj)
            sel = (mask_5d.reshape(-1) > 0.5).nonzero(as_tuple=False).squeeze(-1)
            f_flat = f_flat.index_select(1, sel)
            m_flat = m_flat.index_select(1, sel)
        fc = f_flat - f_flat.mean(dim=1, keepdim=True)
        mc = m_flat - m_flat.mean(dim=1, keepdim=True)
        num = (mc * fc).sum(dim=1)
        den = torch.sqrt(mc.square().sum(dim=1) * fc.square().sum(dim=1) + 1e-8)
        corrs = num / den
    if return_tensor or need_grad:
        return corrs
    return corrs.detach().cpu().numpy().astype(np.float32)


def batch_jacobian(
    fixed: Union[ants.ANTsImage, FixedTarget],
    transformlist: Sequence[Any],
    *,
    do_log: bool = True,
    include_affine: bool = False,
    return_tensors: bool = False,
    device: Union[str, torch.device, None] = "auto",
) -> Union[List[ants.ANTsImage], torch.Tensor]:
    """Vectorized GPU Batch Jacobian Determinant / Log-Jacobian for Tensor-Based Morphometry (TBM).

    Evaluates exact physical-space finite-difference Jacobian determinants across ``N`` subject
    transforms (``TransformChain``, registration result dicts with ``'fwd_chain'``, ``TransformElement``,
    or warp file paths) on ``fixed``'s physical grid using the fused single-pass ``fused_jacobian_det_3d``
    GPU kernel without writing temporary ``.nii.gz`` warps to disk.

    Parameters
    ----------
    fixed : ants.ANTsImage or FixedTarget
        Common template / reference image defining physical ``origin``, ``spacing``, and ``direction``.
    transformlist : sequence
        Per-subject transforms mapping ``fixed`` (template) -> ``moving`` (subject): each item may be
        a ``TransformChain``, a registration dict containing ``'fwd_chain'`` or ``'fwdtransforms'``,
        a ``TransformElement``, or a transform path / list of paths.
    do_log : bool, default True
        Return ``log(clamp(det J, min=1e-6))`` (standard TBM log-Jacobian: ``> 0`` local subject
        expansion relative to template, ``< 0`` local contraction/atrophy).
    include_affine : bool, default False
        When ``False`` (standard nonlinear TBM), factors out global affine brain/organ scaling and
        evaluates ``det(I + grad u_def(x))`` strictly on the nonlinear deformation field. When ``True``,
        multiplies by ``|det A_k|`` of all linear transforms in each subject's chain.
    return_tensors : bool, default False
        When ``True``, returns a GPU tensor of shape ``(N, 1, Z, Y, X)`` (3-D) or ``(N, 1, Y, X)`` (2-D)
        for direct GPU voxelwise GLM / z-score maps; when ``False``, returns ``list[ants.ANTsImage]``
        preserving ``fixed``'s physical header.
    """
    from .core.jacobian import compute_physical_jacobian_determinant
    from .transforms import TransformChain, TransformElement, _LINEAR, _resolve_element_linear

    ref_img = fixed.image if getattr(fixed, "_is_syntx_fixed_target", False) else fixed
    dev_str = resolve_device("auto" if device is None else str(device), task="reg")
    dev_obj = torch.device(dev_str)
    dim = int(ref_img.dimension)
    sp_zyx = tuple(int(s) for s in reversed(ref_img.shape[:dim]))
    n_items = len(transformlist)
    if n_items == 0:
        return torch.empty((0, 1, *sp_zyx), device=dev_obj, dtype=torch.float32) if return_tensors else []

    disp_list: List[torch.Tensor] = []
    aff_scales: List[float] = []
    for item in transformlist:
        if isinstance(item, dict):
            ch = item.get("fwd_chain")
            if ch is None:
                ch = TransformChain.from_paths(item.get("fwdtransforms", []))
        elif isinstance(item, TransformChain):
            ch = item
        elif isinstance(item, (str, TransformElement)):
            ch = TransformChain.from_paths([item])
        else:
            ch = TransformChain.from_paths(list(item))

        disp_elem = next((el for el in ch.elements if el.kind not in (*_LINEAR, "identity")), None)
        if disp_elem is not None:
            d_t = disp_elem.disp_tensor_zyx(device=dev_obj)
        else:
            d_t = torch.zeros((1, *sp_zyx, dim), device=dev_obj, dtype=torch.float32)
            d_t.is_physical = True
        disp_list.append(d_t)

        if include_affine:
            det_lin = 1.0
            for el in ch.elements:
                if el.kind in _LINEAR:
                    m_lin, _ = _resolve_element_linear(el)
                    det_lin *= abs(float(np.linalg.det(m_lin)))
            aff_scales.append(det_lin)
        else:
            aff_scales.append(1.0)

    n_vox = int(np.prod(sp_zyx))
    chunk_b = max(1, min(n_items, 67_108_864 // max(1, n_vox)))
    det_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for s in range(0, n_items, chunk_b):
            e = min(s + chunk_b, n_items)
            d_batch = torch.cat(disp_list[s:e], dim=0)
            d_batch.is_physical = True
            det_b = compute_physical_jacobian_determinant(
                d_batch, direction=ref_img.direction, spacing=ref_img.spacing
            )
            if include_affine:
                sc_t = torch.tensor(aff_scales[s:e], device=dev_obj, dtype=det_b.dtype).view(-1, *([1] * (det_b.ndim - 1)))
                det_b = det_b * sc_t
            if do_log:
                det_b = torch.log(torch.clamp(det_b, min=1e-6))
            det_chunks.append(det_b)
        out_t = torch.cat(det_chunks, dim=0).unsqueeze(1)

    if return_tensors:
        return out_t

    rev_sp = tuple(range(dim - 1, -1, -1))
    out_np = out_t[:, 0].permute(0, *tuple(ax + 1 for ax in rev_sp)).contiguous().detach().cpu().numpy()
    return [ref_img.new_image_like(np.ascontiguousarray(out_np[k], dtype=np.float32)) for k in range(n_items)]


def batch_tbm(
    fixed: Union[ants.ANTsImage, FixedTarget],
    moving_list: Sequence[ants.ANTsImage],
    *,
    method: str = "syn",
    similarity_metric: str = "cc2",
    do_log: bool = True,
    include_affine: bool = False,
    initial_transforms: Optional[Sequence[Any]] = None,
    affine_dof: str = "affine",
    scales: Sequence[int] = (4, 2, 1),
    reg_iterations: Sequence[int] = (100, 100, 20),
    batch_size: Union[int, str] = "auto",
    mask: Optional[ants.ANTsImage] = None,
    return_warped: bool = True,
    return_tensors: bool = False,
    sync_free: bool = True,
    device: Union[str, torch.device, None] = "auto",
    **kwargs: Any,
) -> Dict[str, Any]:
    """End-to-end GPU-resident Population Tensor-Based Morphometry (TBM) pipeline.

    Pins the study template once as a ``FixedTarget`` (caching multi-scale pyramids, LNCC local
    moments, and sparse 4-tap Mattes-MI Parzen weights on GPU), runs vectorized cohort affine
    alignment (``batch_robust_affine``) + deformable registration (``batch_syn`` or ``batch_greedy``)
    driven by the fused batched similarity kernels (``cc2``, ``cc``, or ``mattes_mi``), extracts
    per-subject physical-space Jacobian / log-Jacobian maps via ``batch_jacobian`` (`137x` fused
    ``fused_jacobian_det_3d``), and scores post-registration template alignment across the cohort
    in a single batched pass (``batch_cc2`` / ``batch_cc`` / ``batch_mattes_mi`` + ``batch_correlation``).
    """
    from .core.jacobian import compute_physical_jacobian_determinant

    mov_seq = list(moving_list)
    dev_str = resolve_device("auto" if device is None else str(device), task="reg")
    dev_obj = torch.device(dev_str)
    ref_img = fixed.image if getattr(fixed, "_is_syntx_fixed_target", False) else fixed
    dim = int(ref_img.dimension)

    if len(mov_seq) == 0:
        return {
            "jacobians": [],
            "mean_jacobian": None,
            "std_jacobian": None,
            "warped": [],
            "fwd_chains": [],
            "inv_chains": [],
            "similarity": np.empty((0,), dtype=np.float32),
            "correlation": np.empty((0,), dtype=np.float32),
            "min_jacobian": np.empty((0,), dtype=np.float32),
            "folding_pct": np.empty((0,), dtype=np.float32),
            "registrations": [],
        }

    ft = (
        fixed
        if getattr(fixed, "_is_syntx_fixed_target", False)
        else FixedTarget.from_image(ref_img, scales=scales, device=dev_str)
    )

    if initial_transforms is None:
        aff_res = batch_robust_affine(
            ft,
            mov_seq,
            dof=affine_dof,
            batch_size=batch_size,
            write_transforms=False,
            return_warped=False,
            device=dev_str,
        )
        init_txs = [a["fwd_chain"] for a in aff_res]
    else:
        init_txs = list(initial_transforms)

    m_low = str(method).lower()
    sim_low = str(similarity_metric).lower()
    if m_low in ("syn", "synto", "symmetric"):
        regs = batch_syn(
            ft,
            mov_seq,
            initial_transforms=init_txs,
            levels=scales,
            reg_iterations=reg_iterations,
            similarity_metric=sim_low,
            batch_size=batch_size,
            write_transforms=False,
            return_warped=return_warped and (not return_tensors),
            _return_warped_tensor=True,
            sync_free=sync_free,
            device=dev_str,
            fixed_mask=mask,
            **kwargs,
        )
    elif m_low in ("greedy", "syntx_greedy"):
        regs = batch_greedy(
            ft,
            mov_seq,
            initial_transforms=init_txs,
            scales=scales,
            reg_iterations=reg_iterations,
            similarity_metric=sim_low,
            batch_size=batch_size,
            return_inverse=False,
            write_transforms=False,
            return_warped=return_warped and (not return_tensors),
            _return_warped_tensor=True,
            sync_free=sync_free,
            device=dev_str,
            fixed_mask=mask,
            **kwargs,
        )
    else:
        raise ValueError(f"batch_tbm: method must be 'syn' or 'greedy', got {method!r}")

    fwd_chains = [r["fwd_chain"] for r in regs]
    inv_chains = [r.get("inv_chain") for r in regs]

    # Compute raw physical det(J) once on GPU to extract both TBM maps and diffeomorphism metrics
    raw_det_t = batch_jacobian(
        ft, fwd_chains, do_log=False, include_affine=include_affine, return_tensors=True, device=dev_obj
    )
    with torch.no_grad():
        min_jac_np = raw_det_t.flatten(1).amin(dim=1).detach().cpu().numpy().astype(np.float32)
        fold_pct_np = ((raw_det_t <= 0.0).float().flatten(1).mean(dim=1) * 100.0).detach().cpu().numpy().astype(np.float32)
        jac_t = torch.log(torch.clamp(raw_det_t, min=1e-6)) if do_log else raw_det_t
        mean_jac_t = jac_t.mean(dim=0, keepdim=True)
        std_jac_t = jac_t.std(dim=0, unbiased=False, keepdim=True)

    # Cohort post-registration similarity screening on GPU-resident warped tensors (zero re-upload)
    warped_tensors = torch.cat([r["warpedmovout_tensor"] for r in regs], dim=0)
    if sim_low in ("mi", "mattes", "mattes_mi"):
        sim_scores = batch_mattes_mi(ft, warped_tensors, mask=mask, fixed_range=(0.0, 1.0), device=dev_obj)
    elif sim_low in ("cc", "lncc"):
        sim_scores = batch_cc(ft, warped_tensors, mask=mask, squared=False, device=dev_obj)
    else:
        sim_scores = batch_cc2(ft, warped_tensors, mask=mask, device=dev_obj)
    corr_scores = batch_correlation(ft, warped_tensors, mask=mask, device=dev_obj)

    rev_sp = tuple(range(dim - 1, -1, -1))
    mean_np = mean_jac_t[0, 0].permute(*rev_sp).contiguous().detach().cpu().numpy().astype(np.float32)
    std_np = std_jac_t[0, 0].permute(*rev_sp).contiguous().detach().cpu().numpy().astype(np.float32)
    mean_img = ref_img.new_image_like(mean_np)
    std_img = ref_img.new_image_like(std_np)

    if return_tensors:
        jac_out: Any = jac_t
        warped_out: Any = warped_tensors if return_warped else None
    else:
        jac_np = jac_t[:, 0].permute(0, *tuple(ax + 1 for ax in rev_sp)).contiguous().detach().cpu().numpy()
        jac_out = [ref_img.new_image_like(np.ascontiguousarray(jac_np[k], dtype=np.float32)) for k in range(len(mov_seq))]
        warped_out = [r["warpedmovout"] for r in regs] if return_warped else []
        del warped_tensors

    return {
        "jacobians": jac_out,
        "mean_jacobian": mean_jac_t if return_tensors else mean_img,
        "std_jacobian": std_jac_t if return_tensors else std_img,
        "warped": warped_out,
        "fwd_chains": fwd_chains,
        "inv_chains": inv_chains,
        "similarity": sim_scores,
        "correlation": corr_scores,
        "min_jacobian": min_jac_np,
        "folding_pct": fold_pct_np,
        "registrations": regs,
    }
