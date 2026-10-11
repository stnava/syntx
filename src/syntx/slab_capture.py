"""Intermodality thin-slab capture: ``robust_affine(..., setting='intermodality_slab')``.

The problem. A thin anisotropic slab (a neuromelanin MRI: 0.43 x 0.43 x 3.5 mm, 11 slices) registered to a whole-brain T1: the
metric's capture range is about one coarse voxel in every translation axis, so a start that is a few millimetres off (centre of
mass, phase correlation) never reaches the right basin, and the solver's own coarse-level loss is a poor judge between the
poses that several starts end in (measured, docs/PYRAMID.md Section 13 and docs/RELEASE_NOTES_6.3.0.md).

The strategy (after ``antspymm.tra_initializer`` as used by ``antspymm.neuromelanin``):

1. TRANSLATION first. A translation-only registration (the same Mattes solver, rotation frozen, a large step size) captures the
   large offset, the z offset of a slab above all.
2. MULTI-START rotation. ``n_starts`` rigid starts: the identity and ``n_starts - 1`` random rotations (uniform in
   +-``max_rotation_deg`` about each axis, about the centre of the fixed field of view) composed on top of that translation; each is
   refined by the ordinary rigid solver schedule from that single start (no further candidate search).
3. NATIVE-RESOLUTION ranking. The winner is the start whose WARPED MOVING IMAGE ON THE FIXED IMAGE'S NATIVE GRID has the best Mattes
   mutual information, not the one with the best coarse-level loss of the solver. Same robust intensity normalisation as the solver
   (``normalize_image(method='auto', force=True)`` on both images), same Mattes implementation, same slab-edge valid-region margin
   (``valid_margin_slices``) on the fixed image.
4. BLANK warps (zero variance inside the valid region, e.g. a start that moved the moving image out of the field of view) are
   rejected. A start that makes the solver raise ``RobustAffineSolverError`` is rejected too and recorded; if no start survives,
   ``RobustAffineSolverError`` is raised (no fallback).

Everything is torch (CUDA / MPS / CPU through the solver's device handling); the native grid, the valid region and the warp of the
moving image come from ``syntx.pyramid`` (level 1 of the ``deformable`` policy IS the native grid).
"""
from __future__ import annotations

import logging
import tempfile
import os
import time

import ants
import numpy as np
import torch
import torch.nn.functional as F

from .core.losses import mattes_mi_loss_nd
from .pyramid import build_pyramid, warp_to_moving_norm
from .spatial import image_to_tensor

logger = logging.getLogger(__name__)

SETTINGS = ('default', 'intermodality_slab')

# a warped moving image whose variance inside the valid region is below this (intensities are in [0, 1]) is blank
BLANK_VARIANCE = 1e-8

# translation stage: a large Adam step (mm) so the stage can travel tens of millimetres, annealed per level. eta_min is 0 for every
# group, so the frozen rotation group (lr 0) stays at exactly 0 (a positive eta_min would anneal it UP to eta_min)
_TRANSLATION_SCHEDULE = [
    dict(level=4, iters=120, dof='rigid', lr=(0.8, 0.0, 0.0, 0.0), eta_min=0.0, select=True, sampling=1.0),
    dict(level=2, iters=80, dof='rigid', lr=(0.15, 0.0, 0.0, 0.0), eta_min=0.0, select=False, sampling=0.10),
    dict(level=1, iters=40, dof='rigid', lr=(0.03, 0.0, 0.0, 0.0), eta_min=0.0, select=False, sampling=0.02),
]


def check_setting(setting) -> str:
    """Validate the ``setting`` argument of ``robust_affine``; ``ValueError`` for anything but a known setting."""
    if not isinstance(setting, str) or setting not in SETTINGS:
        raise ValueError(f"robust_affine: setting must be one of {list(SETTINGS)}, got {setting!r}")
    return setting


# ---------------------------------------------------------------------------------------------- transforms
def _affine_from_file(path: str):
    """(A, t_eff) of an ITK affine file as ``y = A x + t_eff`` (float64): ITK stores ``y = A (x - c) + c + t``."""
    tx = ants.read_transform(path)
    p = np.asarray(tx.parameters, dtype=np.float64)
    c = np.asarray(tx.fixed_parameters, dtype=np.float64)
    A = p[:9].reshape(3, 3)
    return A, p[9:12] + c - A @ c


def _write_affine(A: np.ndarray, t_eff: np.ndarray, path: str) -> str:
    tx = ants.create_ants_transform(transform_type='AffineTransform', precision='float', dimension=3)
    tx.set_parameters(np.concatenate([np.asarray(A, dtype=np.float64).ravel(), np.asarray(t_eff, dtype=np.float64)]))
    tx.set_fixed_parameters(np.zeros(3))
    ants.write_transform(tx, path)
    return path


def random_rotation(rng: np.random.Generator, max_rotation_deg: float) -> np.ndarray:
    """Rotation Rz Ry Rx with each angle uniform in +-``max_rotation_deg`` (degrees)."""
    rx, ry, rz = np.radians(rng.uniform(-max_rotation_deg, max_rotation_deg, 3))
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _fov_center(img) -> np.ndarray:
    """Physical centre of the field of view of ``img`` (centre of its index box)."""
    idx = (np.asarray(img.shape, dtype=np.float64) - 1.0) / 2.0
    return np.asarray(img.origin, dtype=np.float64) + np.asarray(img.direction, dtype=np.float64).reshape(3, 3) @ (idx * np.asarray(img.spacing))


# ---------------------------------------------------------------------------------------------- native-resolution scoring
class NativeMIScorer:
    """Mattes mutual information of the moving image warped onto the FIXED image's native grid (every voxel inside the valid region).

    Both images get the solver's intensity normalisation; the fixed native grid, the valid region (``valid_margin_slices`` outermost
    native slices of the thick axis excluded) and the warp of the moving image are ``syntx.pyramid`` level 1 objects. ``loss`` is the
    negative MI in nats (lower is better), the same quantity and the same Mattes implementation the solver optimises, at the native
    level instead of its coarse levels.
    """

    def __init__(self, fixed, moving, device, valid_margin_slices: int = 1, num_bins: int = 32, fixed_range=(0.0, 1.0)):
        from .core.utils import normalize_image
        fn = normalize_image(fixed, method='auto', force=True)
        mn = normalize_image(moving, method='auto', force=True)
        fi = image_to_tensor(fn, device=device, to_zyx=True)
        mi = image_to_tensor(mn, device=device, to_zyx=True)
        self.device, self.num_bins, self.fixed_range = device, num_bins, fixed_range
        lf = build_pyramid(fi, fixed.spacing, [1], policy='deformable', valid_margin=int(valid_margin_slices),
                           origin=fixed.origin, direction=fixed.direction)[0]
        self.lm = build_pyramid(mi, moving.spacing, [1], policy='deformable', origin=moving.origin, direction=moving.direction)[0]
        if tuple(lf.shape) != tuple(fixed.shape[::-1]) or not lf.plan.is_identity:
            raise RuntimeError(f"native scoring needs the identity level-1 grid of the fixed image, got {lf.shape} for {fixed.shape[::-1]}")
        self.fv = lf.tensor.reshape(-1)
        self.phys = torch.as_tensor(lf.physical_coordinates().reshape(-1, 3), dtype=torch.float32, device=device)
        self.valid = (torch.ones_like(self.fv, dtype=torch.bool) if lf.valid is None
                      else lf.valid.to(device=device, dtype=torch.bool).reshape(-1))

    @torch.no_grad()
    def warped(self, A: np.ndarray, t_eff: np.ndarray) -> torch.Tensor:
        """Moving image on the fixed native grid for ``y = A x + t_eff``, flattened (zero outside the moving field of view)."""
        At = torch.as_tensor(np.asarray(A), dtype=torch.float32, device=self.device)
        tt = torch.as_tensor(np.asarray(t_eff), dtype=torch.float32, device=self.device)
        y = warp_to_moving_norm(self.phys @ At.t() + tt, self.lm)
        grid = y.reshape(1, 1, 1, -1, 3)
        return F.grid_sample(self.lm.tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=True).reshape(-1)

    @torch.no_grad()
    def loss(self, A: np.ndarray, t_eff: np.ndarray):
        """(negative MI, blank flag). A blank warp (variance inside the valid region <= ``BLANK_VARIANCE``) has loss ``inf``."""
        w = self.warped(A, t_eff)
        wv = w[self.valid]
        if wv.numel() == 0 or not bool(torch.isfinite(wv).all()) or float(wv.var()) <= BLANK_VARIANCE:
            return float('inf'), True
        loss = mattes_mi_loss_nd(w, self.fv, mask=self.valid, num_bins=self.num_bins, auto_mask=False, fixed_range=self.fixed_range)
        return float(loss.item()), False


# ---------------------------------------------------------------------------------------------- the strategy
def run_slab_capture(fixed, moving, *, initial_tx_path=None, device='auto', verbose=False, seed=42, n_starts=32,
                     max_rotation_deg=30.0, include_default_start=True, preset='default', dof='rigid', valid_margin_slices=1, num_bins=32, solver=None,
                     **solver_kwargs) -> dict:
    """Translation-first, multi-start rigid registration ranked by native-resolution MI (see the module docstring).

    ``solver`` is the single-start rigid solver (default ``robust_affine._run_pytorch_affine_solver``; tests substitute their own).
    ``include_default_start`` (default True) adds the result of the ordinary multi-start rigid solver (centre-of-mass, phase-correlation and cone
    candidates, no translation stage) to the pool that the native-MI ranking chooses from: the translation-first stage can converge to
    one translation basin from which every rotation start refines to the same wrong pose (real sub-01: 7.6 mm), a failure the default
    solver does not have. ``False`` is the pure translation-first strategy. ``n_starts`` counts the identity-rotation start, so ``n_starts - 1`` random rotations are drawn. Returns the ordinary
    ``robust_affine`` result dict of the winning start plus ``'setting'``, ``'native_mi'`` and ``'slab_capture'`` diagnostics.
    """
    import importlib
    RA = importlib.import_module('syntx.robust_affine')
    t0 = time.time()
    if fixed.dimension != 3 or moving.dimension != 3:
        raise ValueError("robust_affine(setting='intermodality_slab') is 3-D only")
    if dof != 'rigid':
        raise ValueError(f"robust_affine(setting='intermodality_slab') registers rigidly: pass dof='rigid', got {dof!r}")
    if not isinstance(n_starts, (int, np.integer)) or isinstance(n_starts, bool) or n_starts < 1:
        raise ValueError(f"slab capture: n_starts must be a positive integer, got {n_starts!r}")
    if not (0.0 <= float(max_rotation_deg) <= 180.0):
        raise ValueError(f"slab capture: max_rotation_deg must be in [0, 180], got {max_rotation_deg!r}")
    if solver is None:
        solver = RA._run_pytorch_affine_solver
    if device in ('auto', None):
        device_obj = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    else:
        device_obj = torch.device(device)
    seed = 42 if seed is None else int(seed)
    common = dict(device=str(device_obj), seed=seed, preset=preset, num_bins=num_bins, valid_margin_slices=valid_margin_slices,
                  dof='rigid', verbose=False, **solver_kwargs)
    work = tempfile.mkdtemp(prefix='slab_capture_')

    # 1. translation-only stage ---------------------------------------------------------------------------------------
    tr = solver(fixed, moving, initial_tx_path=None, multi_start=True, n_starts=3, cone_angles_deg=[], schedule=[dict(s) for s in _TRANSLATION_SCHEDULE],
                **{k: v for k, v in common.items() if k != 'preset'})
    A_tr, t_tr = _affine_from_file(tr['fwdtransforms'][0])
    if not np.allclose(A_tr, np.eye(3), atol=1e-6):
        raise RA.RobustAffineSolverError("robust_affine(setting='intermodality_slab'): the translation-only stage returned a rotation")
    if verbose:
        print(f"[slab_capture] translation stage: t = {np.round(t_tr, 2).tolist()} ({tr['init_candidate']}, loss {tr['final_loss']:.4f})", flush=True)

    # 2. multi-start rotations composed on the translation ---------------------------------------------------------------
    rng = np.random.default_rng(seed)
    C = _fov_center(fixed)
    starts = []                                     # (name, A, t_eff) : y = R (x - C) + C + t_tr
    for k in range(int(n_starts)):
        R = np.eye(3) if k == 0 else random_rotation(rng, float(max_rotation_deg))
        starts.append((f'start{k:02d}' + ('_translation' if k == 0 else ''), R, C - R @ C + t_tr))
    if initial_tx_path is not None:
        Ai, ti = _affine_from_file(str(initial_tx_path))
        starts.append(('provided_initial_transform', Ai, ti))

    schedule = [dict(s_, dof='rigid') for s_ in RA._default_affine_schedule(3, preset)]
    results, rejected = [], []
    for name, A0, t0_ in starts:
        path = _write_affine(A0, t0_, os.path.join(work, f'{name}_start.mat'))
        try:
            reg = solver(fixed, moving, initial_tx_path=path, multi_start=False, n_starts=1, schedule=[dict(s) for s in schedule], **common)
        except RA.RobustAffineSolverError as e:
            rejected.append((name, f'solver: {e}'))
            continue
        results.append((name, reg))

    if include_default_start:
        try:
            results.append(('default_solver', solver(fixed, moving, initial_tx_path=None if initial_tx_path is None else str(initial_tx_path),
                                                      multi_start=True, n_starts=3, schedule=[dict(s) for s in schedule], **common)))
        except RA.RobustAffineSolverError as e:
            rejected.append(('default_solver', f'solver: {e}'))

    # 3./4. native-resolution MI ranking, blank warps rejected -------------------------------------------------------------
    scorer = NativeMIScorer(fixed, moving, device_obj, valid_margin_slices=valid_margin_slices, num_bins=num_bins)
    scored = []
    for name, reg in results:
        A, t = _affine_from_file(reg['fwdtransforms'][0])
        loss, blank = scorer.loss(A, t)
        if blank:
            rejected.append((name, 'blank warped image'))
            continue
        scored.append((loss, name, reg))
    if not scored:
        raise RA.RobustAffineSolverError(
            f"robust_affine(setting='intermodality_slab'): none of the {len(starts)} starts gave a usable pose "
            f"(rejected: {rejected}); the slab could not be registered to this image.")
    scored.sort(key=lambda x: x[0])
    best_loss, best_name, best = scored[0]
    if verbose:
        print(f"[slab_capture] native MI ranking: {[(n, round(-l, 4)) for l, n, _ in scored[:6]]} ... winner {best_name}; "
              f"{len(rejected)} rejected; {time.time() - t0:.1f} s", flush=True)
    out = dict(best)
    out.update({
        'setting': 'intermodality_slab',
        'native_mi': -best_loss,
        'init_candidate': f'slab_capture:{best_name}',
        'runtime_seconds': time.time() - t0, 'time': time.time() - t0,
        'slab_capture': {
            'translation_mm': t_tr.tolist(), 'n_starts': len(starts), 'default_start': bool(include_default_start), 'winner': best_name,
            'ranking': [(n, -l) for l, n, _ in scored], 'rejected': rejected,
            'coarse_loss_ranking': sorted([(float(r['self_check']['final_loss_coarse']), n) for _, n, r in scored]),
        },
    })
    return out
