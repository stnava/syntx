"""Pyramid-policy comparison harness (Pass 1 of the central-pyramid plan; see docs/PYRAMID.md, docs/PYRAMID_EVIDENCE.md).

Candidate pyramid POLICIES are evaluated through the real solvers on the same data, back to back:

    DEFAULT  no builder: robust_affine's current behaviour (mm pyramid for an anisotropic fixed image, voxel pooling otherwise)
    P1  voxel avg_pool                     (robust_affine._pool_level, motion_batched)
    P2  Gaussian + F.interpolate           (pyramid.build_image_pyramid; sigma = log2(level) voxels, align_corners)
    P3  shape based, 16-voxel floor        (greedy / pyramid.build_anti_aliased_pyramid; compact Gaussian, align_corners)
    P4  millimetre isotropic area/linear   (robust_affine._mm_pyramid_level)
    P5  hybrid: P4 at the coarse levels, the NATIVE fixed grid at level 1
    each with an optional valid-region margin (``+m``): the outermost slab-edge slice(s) of the FIXED image do not enter
    the metric (the solver's ``valid`` mask; only meaningful for an anisotropic fixed image).

A policy is a *level builder*: ``builder(role, arr, spacing_xyz, ctx) -> (tensor, f_xyz, off_xyz[, valid])`` where ``arr`` is
``(B,1,z,y,x)``, working index j on axis i sits at full-resolution continuous index ``j*f[i] + off[i]`` and ``ctx`` carries
``level, sp_min_f, sig_mm, budget, device``. The builder attribute ``min_shape_rule`` ('voxel' | 'mm' | 'none') picks the
solver's coarsest-level clamp. The solvers accept a builder through the private ``_level_builder`` seam (default: None, which
leaves their behaviour bit-identical).

Sub-commands (all write JSON-lines under --out and are resumable):
    synthetic   thin-slab known-rigid sweep (tests/test_robust_affine_anisotropy.py::_slab_pair generalised over pose)
    real        real T1-slab -> NM pairs (pose error vs the ANTs rigid reference, solver loss at the ANTs pose and at the
                solved pose, a policy-independent native-grid MI)
    isotropic   back-to-back timing and accuracy on isotropic pairs
    motion      motion_batched known-transform cases, P1 vs P4 (and the others)
    properties  builder property tests (shapes, NaN at sigma 0, coordinates, determinism, CPU/MPS parity, time, memory)
    deformable  characterisation of the pyramids the deformable models use (shapes, intensities, coordinates, time)
    probe_truth coarse-level solver loss and native MI at the TRUE transform vs at the solved one for synthetic failures
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

POLICIES = ["P1", "P2", "P3", "P4", "P5"]
ANISO_RATIO = 1.25            # same threshold as syntx.robust_affine.ANISOTROPY_RESAMPLE_RATIO
SIXTEEN = 16                  # the floor of build_anti_aliased_pyramid / greedy


class PyramidLevelError(RuntimeError):
    """A policy cannot build the requested level (a zero- or one-voxel axis)."""


# ---------------------------------------------------------------------------------------------- level operators
def _check_shape(name: str, shape, n_in, level):
    if any(int(m) < 2 for m in shape):
        raise PyramidLevelError(f"{name}: level {level} of a {tuple(n_in)} tensor has shape {tuple(shape)} (an axis < 2 voxels)")


def level_p1(arr: torch.Tensor, level: int):
    """P1: average-pool by ``level`` voxels on every axis (no smoothing, as robust_affine's default)."""
    if level > 1:
        arr = F.avg_pool3d(arr, kernel_size=level, stride=level)
    _check_shape("P1", arr.shape[2:], arr.shape[2:], level)
    return arr, [float(level)] * 3, [(level - 1) / 2.0] * 3


def _legacy():
    """The incumbent ``build_image_pyramid`` / ``build_anti_aliased_pyramid`` as of Pass 1 (moved to tests/ in Pass 3b)."""
    import sys
    from pathlib import Path
    root = str(Path(__file__).resolve().parents[1] / "tests")
    if root not in sys.path:
        sys.path.insert(0, root)
    import legacy_pyramid_reference
    return legacy_pyramid_reference


def level_p2(arr: torch.Tensor, spacing_xyz, level: int):
    """P2: syntx.pyramid.build_image_pyramid itself (sigma = log2(level) voxels on every axis, then F.interpolate,
    align_corners=True). Working index j sits at full-res index j*(n-1)/(m-1) (corner aligned)."""
    build_image_pyramid = _legacy().build_image_pyramid       # the Pass-1 incumbent, now the test-only golden reference
    n = list(arr.shape[2:])
    if level <= 1:
        return arr, [1.0] * 3, [0.0] * 3
    out = build_image_pyramid(arr, spacing_xyz, [level])[0]
    m = list(out.shape[2:])
    _check_shape("P2", m, n, level)
    f = [(a - 1.0) / (b - 1.0) for a, b in zip(n, m)]
    return out, f[::-1], [0.0] * 3


def level_p3(arr: torch.Tensor, level: int, floor: int = SIXTEEN):
    """P3: shape-based (m = max(int(n/level), floor) per axis; a floor above n up-samples, as greedy does), compact Gaussian
    sigma = 0.5 n/m, F.interpolate align_corners. Sigma is computed from each image's OWN shape (greedy uses the fixed
    image's sigma for the moving image; that bug is not reproduced here)."""
    from syntx.core.smoothing import gaussian_1d_compact, separable_1d_filter
    n = list(arr.shape[2:])
    if level <= 1:
        return arr, [1.0] * 3, [0.0] * 3
    m = [max(int(sz / level), floor) for sz in n]
    sig = [0.5 * (sz / mm) for sz, mm in zip(n, m)]
    g = [gaussian_1d_compact(s, truncated=2.0, device=arr.device, dtype=arr.dtype) for s in sig]
    out = F.interpolate(separable_1d_filter(arr, g), size=m, mode="trilinear", align_corners=True)
    f = [(a - 1.0) / (b - 1.0) for a, b in zip(n, m)]
    return out, f[::-1], [0.0] * 3


def level_p4(arr: torch.Tensor, spacing_xyz, target_mm: float, sig_mm: float, budget: int):
    from syntx.robust_affine import _mm_pyramid_level
    return _mm_pyramid_level(arr, spacing_xyz, target_mm, sig_mm, budget)


def level_p5(arr: torch.Tensor, spacing_xyz, target_mm: float, sig_mm: float, budget: int, level: int, role: str):
    """P5: isotropic millimetre levels, the native fixed grid at level 1."""
    if level == 1 and role == "fixed":
        return arr, [1.0] * 3, [0.0] * 3
    return level_p4(arr, spacing_xyz, target_mm, sig_mm, budget)


# ---------------------------------------------------------------------------------------------- builders
class Policy:
    """A named level builder. ``margin`` (native slices) drops the outermost slab-edge slice(s) of the fixed image."""

    def __init__(self, name: str, margin: int = 0):
        if name not in POLICIES and name != "DEFAULT_EQUIV":
            raise ValueError(f"unknown policy {name!r}; known: {POLICIES}")
        self.name, self.margin = name, int(margin)
        # P1/P2 are used as the solver would ship them: with its 8-voxel coarsest-level clamp in voxels. P3 carries its own
        # floor; P4/P5 use the millimetre clamp of the solver's anisotropic path.
        self.min_shape_rule = {"P1": "voxel", "P2": "voxel", "P3": "none", "P4": "mm", "P5": "mm", "DEFAULT_EQUIV": "mm"}[name]

    @property
    def label(self) -> str:
        return self.name + ("+m" if self.margin else "")

    def __call__(self, role, arr, spacing_xyz, ctx):
        level = int(ctx["level"])
        sp_min = float(ctx["sp_min_f"])
        if self.name == "DEFAULT_EQUIV":
            # robust_affine's own default for an anisotropic fixed image: fixed = P4; moving = P4 only if it is itself
            # anisotropic, otherwise voxel pooling (used to prove the seam is default-preserving, not a candidate)
            if role == "fixed" or max(spacing_xyz) / min(spacing_xyz) > ANISO_RATIO:
                return Policy("P4")(role, arr, spacing_xyz, ctx)
            return level_p1(arr, level)
        if self.name == "P1":
            out = level_p1(arr, level)
        elif self.name == "P2":
            out = level_p2(arr, spacing_xyz, level)
        elif self.name == "P3":
            out = level_p3(arr, level)
        else:
            tgt = level * sp_min if role == "fixed" else max(level * sp_min, min(float(x) for x in spacing_xyz))
            if self.name == "P4":
                out = level_p4(arr, spacing_xyz, tgt, ctx["sig_mm"], ctx["budget"])
            else:
                out = level_p5(arr, spacing_xyz, tgt, ctx["sig_mm"], ctx["budget"], level, role)
        if role == "fixed" and self.margin > 0:
            v = valid_margin(arr.shape[2:], spacing_xyz, out[1], out[2], out[0].shape[2:], self.margin, arr.device)
            if v is not None:
                return (*out, v)
        return out


def valid_margin(n_zyx, spacing_xyz, f_xyz, off_xyz, m_zyx, margin: int, device):
    """Bool mask on the level grid: False where the working sample's centre lies within ``margin`` native slices of either end
    of the thick axis (c < margin or c > n-1-margin, c = j*f+off). None for an (almost) isotropic image."""
    sp = [float(x) for x in spacing_xyz]
    if max(sp) / min(sp) <= ANISO_RATIO:
        return None
    a = int(np.argmax(sp))                                   # thick axis, xyz index
    n = int(list(n_zyx)[::-1][a])
    if n < 2 * margin + 2:
        return None
    j = torch.arange(int(list(m_zyx)[::-1][a]), dtype=torch.float64)
    c = j * float(f_xyz[a]) + float(off_xyz[a])
    ok = (c >= margin - 1e-6) & (c <= n - 1 - margin + 1e-6)
    shape = [1, 1, 1, 1, 1]
    shape[2 + (2 - a)] = -1
    full = torch.ones(1, 1, *[int(x) for x in m_zyx], dtype=torch.bool)
    return (full & ok.reshape(shape)).to(device)


def get_policy(label: str) -> Optional[Policy]:
    """'DEFAULT' -> None (no builder); 'P3' / 'P3+m' -> Policy."""
    if label == "DEFAULT":
        return None
    return Policy(label.rstrip("+m"), margin=1 if label.endswith("+m") else 0)


def all_labels(with_margin: bool = True) -> List[str]:
    return ["DEFAULT"] + POLICIES + ([p + "+m" for p in POLICIES] if with_margin else [])


# ---------------------------------------------------------------------------------------------- synthetic thin slab
POSES = {   # (rotation z deg, rotation x deg, translation mm); "A" is the pose of tests/test_robust_affine_anisotropy.py
    "A": (6.0, -4.0, (4.0, -3.0, 5.0)),
    "D": (-5.0, 3.0, (-3.0, 2.0, -4.0)),
    "E": (4.0, 6.0, (2.0, -5.0, 3.0)),
    "F": (-3.0, -5.0, (5.0, 4.0, -2.0)),
    # Screened out: B (-8, 5, (-6, 2, -6)) and C (10, 7, (3, 5, 7)) were tried first and every policy, including DEFAULT, fails
    # them at 1:1 (TRE 5.7 and 7.4 mm), i.e. they lie outside the solver's capture range on this phantom regardless of the
    # pyramid; they cannot discriminate policies. D, E, F were the first three of a fixed list of five candidates that
    # DEFAULT solves at 1:1 (TRE < 0.11 mm); the choice used no thin-slab result.
    "B": (-8.0, 5.0, (-6.0, 2.0, -6.0)),
    "C": (10.0, 7.0, (3.0, 5.0, 7.0)),
}
_PHANTOM = None


def _phantom():
    global _PHANTOM
    if _PHANTOM is None:
        import ants, siq
        from scipy.ndimage import gaussian_filter
        np.random.seed(1234)
        arr = gaussian_filter(siq.simulate_brain_procedural((96, 104, 88)).numpy().astype(np.float32), 1.0)
        _PHANTOM = ants.from_numpy(arr, origin=(-48.0, -52.0, -44.0), spacing=(1.0, 1.0, 1.0))
    return _PHANTOM


def slab_pair(r: int, pose: str = "A", inplane: float = 0.75, fov: float = 54.0):
    """tests/test_robust_affine_anisotropy.py::_slab_pair generalised over the pose (pose 'A' reproduces it exactly)."""
    import ants
    mov = _phantom()
    a = mov.numpy(); a = (a - a.min()) / (a.max() - a.min())
    cmap = mov.new_image_like((1.0 - a ** 0.8).astype(np.float32))
    za, xa, t = POSES[pose]
    a1, a2 = np.deg2rad(za), np.deg2rad(xa)
    Rz = np.array([[np.cos(a1), -np.sin(a1), 0], [np.sin(a1), np.cos(a1), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(a2), -np.sin(a2)], [0, np.sin(a2), np.cos(a2)]])
    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
    tx.set_parameters(np.concatenate([(Rz @ Rx).ravel(), list(t)])); tx.set_fixed_parameters(np.zeros(3))
    n = int(round(fov / inplane)); nz = int(round(fov / (inplane * r)))
    grid = ants.make_image((n, n, nz * r), spacing=(inplane,) * 3, origin=(-fov / 2,) * 3)
    fine = ants.apply_ants_transform_to_image(tx, cmap, grid, interpolation="linear")
    fa = fine.numpy().reshape(n, n, nz, r).mean(axis=3)
    fixed = ants.from_numpy(fa.astype(np.float32), origin=(-fov / 2, -fov / 2, -fov / 2 + (r - 1) * inplane / 2),
                            spacing=(inplane, inplane, inplane * r))
    return fixed, mov, tx


def tre(reg, tx) -> float:
    import ants
    rec = ants.read_transform(reg["fwdtransforms"][0])
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-20, 20, 100), rng.uniform(-20, 20, 100), rng.uniform(-15, 15, 100)])
    return float(np.mean([np.linalg.norm(np.array(rec.apply_to_point(tuple(p))) - np.array(tx.apply_to_point(tuple(p))))
                          for p in pts]))


def _done_keys(path: str) -> set:
    keys = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                keys.add(json.loads(line)["key"])
            except Exception:
                pass
    return keys


def _append(path: str, rec: dict):
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def run_robust(fixed, moving, label: str, device: str, seed: int, **kw):
    """One robust_affine call under a policy. Returns (reg or None, seconds, error string or None)."""
    import syntx
    t = time.time()
    try:
        reg = syntx.robust_affine(fixed, moving, mode="pytorch", dof="rigid", device=device, seed=seed,
                                  _level_builder=get_policy(label), **kw)
        return reg, time.time() - t, None
    except Exception as e:     # a crash is a measured outcome of the policy, recorded verbatim
        return None, time.time() - t, f"{type(e).__name__}: {str(e)[:200]}"


def cmd_synthetic(a):
    out = os.path.join(a.out, "synthetic.jsonl")
    done = _done_keys(out)
    labels = a.policies or all_labels()
    for r in a.ratios:
        for pose in a.poses:
            fixed, mov, tx = slab_pair(r, pose)
            for seed in a.seeds:
                for label in labels:
                    if r == 1 and label.endswith("+m"):
                        continue                                  # no thick axis, no margin
                    key = f"{r}|{pose}|{seed}|{label}|{a.device}"
                    if key in done:
                        continue
                    reg, dt, err = run_robust(fixed, mov, label, a.device, seed)
                    rec = dict(key=key, ratio=r, pose=pose, seed=seed, policy=label, device=a.device, time=dt, error=err,
                               tre=None if reg is None else tre(reg, tx),
                               loss=None if reg is None else float(reg["final_loss"]))
                    _append(out, rec)
                    print(json.dumps(rec), flush=True)


# ---------------------------------------------------------------------------------------------- real NM pairs
def probes(fixed):
    import ants
    c = np.array(ants.get_center_of_mass(fixed * 0 + 1))
    return [c + np.array([x, y, z]) for x in (-15, 15) for y in (-15, 15) for z in (-5, 5)] + [c]


def displacements(fixed, tx_path):
    import ants
    tx = ants.read_transform(tx_path)
    return np.array([np.array(tx.apply_to_point(tuple(p))) - p for p in probes(fixed)])


def pose_error(fixed, tx_a, tx_b):
    d = displacements(fixed, tx_a) - displacements(fixed, tx_b)
    mag = np.linalg.norm(d, axis=1)
    return float(mag.max()), float(abs(d[:, 2].mean()))


def native_mi(fixed, moving, tx_path) -> float:
    """Policy-independent yardstick: ANTs histogram MI of the fixed image and the moving image warped onto it (all voxels)."""
    import ants
    w = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=[tx_path], interpolator="linear")
    return float(ants.image_mutual_information(fixed, w))


def loss_at_pose(fixed, moving, tx_path, label, device, seed):
    """The solver's own coarse-level full-grid loss AT a given pose: the solver is started from that transform with zero
    optimiser iterations (candidate = that pose only), so the loss comes out of its self-check under the same policy."""
    reg, dt, err = run_robust(fixed, moving, label, device, seed, initial_transform=tx_path, multi_start=False, iterations=0)
    if reg is None:
        return None, err
    return float(reg["self_check"]["final_loss_coarse"]), None


def cmd_real(a):
    import ants
    out = os.path.join(a.out, "real.jsonl")
    done = _done_keys(out)
    labels = a.policies or all_labels()
    for sub in a.subjects:
        d = os.path.join(a.subjects_dir, sub)
        fixed = ants.image_read(os.path.join(d, "fixed_nm_n4.nii.gz"))
        moving = ants.image_read(os.path.join(d, "moving_t1_slab_crop.nii.gz"))
        ref = os.path.join(d, "ants_rigid_reference.mat")
        mi_ref = native_mi(fixed, moving, ref)
        for label in labels:
            for seed in a.seeds:
                key = f"{sub}|{seed}|{label}|{a.device}"
                if key in done:
                    continue
                reg, dt, err = run_robust(fixed, moving, label, a.device, seed)
                rec = dict(key=key, subject=sub, seed=seed, policy=label, device=a.device, time=dt, error=err,
                           mi_ants=mi_ref)
                if reg is not None:
                    tx = reg["fwdtransforms"][0]
                    rec["pose_err_max"], rec["pose_err_z"] = pose_error(fixed, tx, ref)
                    rec["loss_solved"] = float(reg["self_check"]["final_loss_coarse"])
                    rec["loss_ants_pose"], rec["loss_err"] = loss_at_pose(fixed, moving, ref, label, a.device, seed)
                    rec["mi_solved"] = native_mi(fixed, moving, tx)
                _append(out, rec)
                print(json.dumps(rec), flush=True)


# ---------------------------------------------------------------------------------------------- isotropic timing/accuracy
def iso_pair(shape=(96, 104, 88), kind: str = "same_modality"):
    """Isotropic 1 mm pair: the phantom (or its intensity inverse) vs a rigidly moved copy; returns (fixed, moving, tx)."""
    import ants
    mov = _phantom()
    fixed_src = mov
    if kind == "cross_modal":
        a = mov.numpy(); a = (a - a.min()) / (a.max() - a.min())
        fixed_src = mov.new_image_like((1.0 - a ** 0.8).astype(np.float32))
    a1, a2 = np.deg2rad(6.0), np.deg2rad(-4.0)
    Rz = np.array([[np.cos(a1), -np.sin(a1), 0], [np.sin(a1), np.cos(a1), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(a2), -np.sin(a2)], [0, np.sin(a2), np.cos(a2)]])
    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
    tx.set_parameters(np.concatenate([(Rz @ Rx).ravel(), [4.0, -3.0, 5.0]])); tx.set_fixed_parameters(np.zeros(3))
    fixed = ants.apply_ants_transform_to_image(tx, fixed_src, mov, interpolation="linear")
    return fixed, mov, tx


def cmd_isotropic(a):
    """Back-to-back timing: each repeat runs every policy once in a rotating order; medians are reported."""
    out = os.path.join(a.out, "isotropic.jsonl")
    labels = ["DEFAULT"] + POLICIES
    for kind in ("same_modality", "cross_modal"):
        fixed, mov, tx = iso_pair(kind=kind)
        for rep in range(a.repeats):
            order = labels[rep % len(labels):] + labels[:rep % len(labels)]
            for label in order:
                for seed in a.seeds:
                    reg, dt, err = run_robust(fixed, mov, label, a.device, seed)
                    rec = dict(kind=kind, rep=rep, seed=seed, policy=label, device=a.device, time=dt, error=err,
                               tre=None if reg is None else tre(reg, tx),
                               loss=None if reg is None else float(reg["final_loss"]))
                    _append(out, rec)
                    print(json.dumps(rec), flush=True)


# ---------------------------------------------------------------------------------------------- motion
def _motion_series(shape, spacing, shifts_rot):
    """Frames = the same smooth blob volume resampled through KNOWN rigid transforms; returns (reference, frames, truths).
    truths[k] is the ANTs transform mapping fixed (reference) points to frame k points (the inverse of the one used to resample)."""
    import ants
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(7)
    arr = np.zeros(shape, np.float32)
    for _ in range(14):
        c = [rng.uniform(0.3, 0.7) * s for s in shape]
        g = np.ogrid[tuple(slice(0, s) for s in shape)]
        sig = rng.uniform(0.08, 0.16) * min(s * sp for s, sp in zip(shape, spacing)) / np.array(spacing)
        arr += rng.uniform(0.5, 1.0) * np.exp(-sum(((g[i] - c[i]) / sig[i]) ** 2 for i in range(3)) / 2).astype(np.float32)
    arr = gaussian_filter(arr, 0.7).astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    ref = ants.from_numpy(arr, spacing=tuple(spacing), origin=tuple(-0.5 * s * sp for s, sp in zip(shape, spacing)))
    frames, truths = [], []
    for (rz, rx, t) in shifts_rot:
        a1, a2 = np.deg2rad(rz), np.deg2rad(rx)
        Rz = np.array([[np.cos(a1), -np.sin(a1), 0], [np.sin(a1), np.cos(a1), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, np.cos(a2), -np.sin(a2)], [0, np.sin(a2), np.cos(a2)]])
        tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
        tx.set_parameters(np.concatenate([(Rz @ Rx).ravel(), list(t)])); tx.set_fixed_parameters(np.zeros(3))
        frames.append(ants.apply_ants_transform_to_image(tx, ref, ref, interpolation="linear"))
        truths.append(tx.invert())      # frame(x) = ref(T(x)): the fixed(ref)->moving(frame) map is T^-1
    return ref, frames, truths


def cmd_motion(a):
    """motion_batched on frames generated through known rigid transforms (small and large jumps), isotropic and anisotropic
    grids. Frame k = reference resampled by transform T_k, so the registration fwd transform of frame k must equal T_k."""
    import ants
    from syntx.motion_batched import _batched_rigid_register_pass_core
    out = os.path.join(a.out, "motion.jsonl")
    cases = {
        "iso_small": ((48, 52, 44), (2.0, 2.0, 2.0), [(1.0, -1.0, (1.5, -1.0, 1.0)), (2.5, 1.5, (-2.0, 1.5, 2.0)), (0.0, 0.0, (0.0, 0.0, 0.0))]),
        "iso_large": ((48, 52, 44), (2.0, 2.0, 2.0), [(8.0, -6.0, (8.0, -6.0, 7.0)), (-9.0, 5.0, (-7.0, 8.0, -6.0)), (5.0, 9.0, (6.0, 6.0, 8.0))]),
        "aniso_3x": ((64, 64, 22), (2.0, 2.0, 6.0), [(1.0, -1.0, (1.5, -1.0, 2.0)), (2.5, 1.5, (-2.0, 1.5, -3.0)), (3.0, 2.0, (3.0, 2.0, 5.0))]),
        "aniso_6x": ((96, 96, 16), (1.5, 1.5, 9.0), [(1.0, -1.0, (1.5, -1.0, 3.0)), (2.5, 1.5, (-2.0, 1.5, -5.0)), (3.0, 2.0, (3.0, 2.0, 7.0))]),
    }
    labels = a.policies or ["DEFAULT"] + POLICIES
    for cname, (shape, spacing, specs) in cases.items():
        ref, frames, truths = _motion_series(shape, spacing, specs)
        aniso = max(spacing) / min(spacing) > ANISO_RATIO
        c = np.array(ants.get_center_of_mass(ref * 0 + 1))
        pts = [c + np.array([x, y, z]) * np.array(spacing) * 3 for x in (-4, 4) for y in (-4, 4) for z in (-1.5, 1.5)]
        for label in labels:
            for rep in range(a.repeats):
                key = f"{cname}|{label}|{rep}|{a.device}"
                t = time.time()
                try:
                    fwd, inv, el = _batched_rigid_register_pass_core(ref, frames, device=a.device, seed_mode="narrow" if aniso else "wide",
                                                                      _level_builder=get_policy(label))
                    errs = []
                    for k, tx in enumerate(truths):
                        rec_tx = ants.read_transform(fwd[k][0])
                        errs.append(float(np.mean([np.linalg.norm(np.array(rec_tx.apply_to_point(tuple(p))) - np.array(tx.apply_to_point(tuple(p)))) for p in pts])))
                    rec = dict(key=key, case=cname, policy=label, rep=rep, device=a.device, time=time.time() - t, error=None,
                               tre_per_frame=errs, tre_mean=float(np.mean(errs)), tre_max=float(np.max(errs)))
                except Exception as e:
                    rec = dict(key=key, case=cname, policy=label, rep=rep, device=a.device, time=time.time() - t,
                               error=f"{type(e).__name__}: {str(e)[:200]}")
                _append(out, rec)
                print(json.dumps(rec), flush=True)


# ---------------------------------------------------------------------------------------------- property tests
def _build(label: str, arr: torch.Tensor, spacing, level: int, role: str = "fixed", sigma0: bool = True):
    pol = get_policy(label)
    ctx = dict(level=level, sp_min_f=min(spacing), sig_mm=0.0, budget=2 ** 24, device=arr.device)
    return pol(role, arr, spacing, ctx)


def _ramp(shape_zyx, dev, axis_t: int):
    idx = torch.arange(shape_zyx[axis_t], dtype=torch.float32, device=dev)
    sh = [1, 1, 1, 1, 1]; sh[2 + axis_t] = -1
    return (idx.reshape(sh) * torch.ones(1, 1, *shape_zyx, device=dev)).contiguous()


def _alloc_cpu_mb(label, big, lev) -> float:
    """Total CPU bytes allocated (MB) during one build (CPU run), from the torch profiler memory events: an upper bound on the peak."""
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        _build(label, big, (1.0, 1.0, 1.0), lev)
    tot = [e for e in prof.key_averages() if e.cpu_memory_usage > 0]
    return float(sum(e.cpu_memory_usage for e in tot)) / 1e6      # total bytes allocated during the build (upper bound on peak)


def cmd_properties(a):
    out = os.path.join(a.out, "properties.jsonl")
    dev_main = a.device
    shapes = [((3, 64, 64), (3.5, 0.43, 0.43)), ((11, 128, 128), (3.5, 0.43, 0.43)), ((11, 256, 256), (3.5, 0.43, 0.43)),
              ((64, 64, 64), (1.0, 1.0, 1.0)), ((7, 33, 65), (1.0, 1.0, 1.0)), ((16, 96, 96), (3.0, 1.0, 1.0)),
              ((2, 40, 40), (5.0, 1.0, 1.0))]
    for label in POLICIES:
        for shp, sp_zyx in shapes:
            sp_xyz = tuple(sp_zyx[::-1])
            rng = np.random.default_rng(0)
            base = torch.from_numpy(rng.random(shp).astype(np.float32)).reshape(1, 1, *shp)
            for level in (1, 2, 4, 8):
                rec = dict(policy=label, shape=list(shp), spacing_zyx=list(sp_zyx), level=level)
                try:
                    res = _build(label, base.to(dev_main), sp_xyz, level)
                    t, f, off = res[:3]
                    rec.update(out_shape=list(t.shape[2:]), f=[round(float(x), 4) for x in f], ok=True,
                               finite=bool(torch.isfinite(t).all().item()), min_axis=int(min(t.shape[2:])))
                    # constant image at sigma 0 -> no NaN, value preserved
                    c = _build(label, torch.full((1, 1, *shp), 0.5, device=dev_main), sp_xyz, level)[0]
                    rec["const_preserved"] = bool(torch.isfinite(c).all().item() and float((c - 0.5).abs().max()) < 1e-4)
                    # physical-coordinate correctness: ramp along each axis maps back to j*f+off (interior only)
                    errs = []
                    for ax_t in range(3):
                        ramp = _ramp(shp, dev_main, ax_t)
                        rt, rf, ro = _build(label, ramp, sp_xyz, level)[:3]
                        ax_xyz = 2 - ax_t
                        m = rt.shape[2 + ax_t]
                        j = torch.arange(m, dtype=torch.float32, device=dev_main)
                        expect = j * float(rf[ax_xyz]) + float(ro[ax_xyz])
                        # the line through the centre of the other axes (a cross-section mean would mix in the edge
                        # voxels, where a zero-padded filter darkens the image), interior voxels only: a ramp of native
                        # indices must come back as j*f+off (error in WORKING voxels: native error / f); axes of < 8 working voxels are
                        # entirely inside the filter support and are not scored
                        sel = [slice(None)] * 5
                        for o in range(3):
                            if o != ax_t:
                                sel[2 + o] = rt.shape[2 + o] // 2
                        got = rt[tuple(sel)].reshape(-1)
                        if m >= 8:
                            errs.append(float((got[3:m - 3] - expect[3:m - 3]).abs().max()) / max(float(rf[ax_xyz]), 1e-9))   # in working voxels
                    rec["ramp_err_vox"] = max(errs) if errs else 0.0
                    # determinism (two builds, bitwise) and CPU/MPS parity
                    rec["deterministic"] = bool(torch.equal(_build(label, base.to(dev_main), sp_xyz, level)[0], t))
                    tc = _build(label, base, sp_xyz, level)[0]
                    rec["cpu_vs_dev_maxabs"] = float((tc.to("cpu") - t.to("cpu")).abs().max())
                except Exception as e:
                    rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:160]}")
                _append(out, rec)
    # time and memory at a head-sized volume
    big = torch.from_numpy(np.random.default_rng(1).random((176, 224, 192)).astype(np.float32)).reshape(1, 1, 176, 224, 192)
    for label in POLICIES:
        for lev in (4, 2):
            try:
                x = big.to(dev_main)
                _build(label, x, (1.0, 1.0, 1.0), lev)
                ts = []
                for _ in range(5):
                    t0 = time.time(); r = _build(label, x, (1.0, 1.0, 1.0), lev)[0]
                    if dev_main == "mps":
                        torch.mps.synchronize()
                    ts.append(time.time() - t0)
                mem = _alloc_cpu_mb(label, big, lev)
                rec = dict(policy=label, perf_shape=[176, 224, 192], level=lev, device=dev_main, ok=True,
                           ms_median=1000 * float(np.median(ts)), out_voxels=int(np.prod(r.shape[2:])), cpu_alloc_total_mb=mem)
            except Exception as e:
                rec = dict(policy=label, perf_shape=[176, 224, 192], level=lev, ok=False, error=f"{type(e).__name__}: {str(e)[:160]}")
            _append(out, rec)
    # raw behaviour of the shipping pyramids on a thin axis, without any solver clamp
    for fn_name in ("build_image_pyramid", "build_anti_aliased_pyramid"):
        pm = _legacy()
        for shp in ((3, 64, 64), (11, 128, 128)):
            for level in (4, 8):
                x = torch.rand(1, 1, *shp)
                try:
                    if fn_name == "build_image_pyramid":
                        o = pm.build_image_pyramid(x, (1.0, 1.0, 1.0), [level])[0]
                    else:
                        o = pm.build_anti_aliased_pyramid(x, [level])[0]
                    rec = dict(policy=fn_name, shape=list(shp), level=level, ok=True, out_shape=list(o.shape[2:]),
                               finite=bool(torch.isfinite(o).all()))
                except Exception as e:
                    rec = dict(policy=fn_name, shape=list(shp), level=level, ok=False, error=f"{type(e).__name__}: {str(e)[:160]}")
                _append(out, rec)
    print("properties written to", out)


# ---------------------------------------------------------------------------------------------- deformable characterisation
def cmd_deformable(a):
    """The deformable models (syn, syngs, tvf: build_image_pyramid; greedy: its own shape-floor-16 code) build their pyramid
    inside the model and tie the warp-field grids to its shapes and align-corners convention, so a policy cannot be swapped in
    without rewriting the models (see docs/PYRAMID_EVIDENCE.md). Instead the pyramids themselves are characterised on the
    isotropic volumes those models receive: output shapes, intensity statistics, time, and agreement between policies after
    mapping every level back to the finest grid through its (f, off)."""
    import ants
    out = os.path.join(a.out, "deformable.jsonl")
    vols = {"phantom_96x104x88": _phantom().numpy().astype(np.float32)}
    if a.t1:
        t1 = ants.image_read(a.t1)
        vols["real_T1w_" + "x".join(str(s) for s in t1.shape)] = t1.numpy().astype(np.float32)
    for vname, arr in vols.items():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 1, 0))).reshape(1, 1, *arr.shape[::-1]).to(a.device)
        for level in (8, 4, 2):
            recs = {}
            for label in POLICIES:
                try:
                    t0 = time.time(); res = _build(label, x, (1.0, 1.0, 1.0), level); lv, f, off = res[:3]
                    if a.device == "mps":
                        torch.mps.synchronize()
                    dt = time.time() - t0
                    # map back to the full grid: for each full-res index, linear-interpolate the level at (i-off)/f
                    ups = []
                    for ax_t in range(3):
                        ax = 2 - ax_t
                        ups.append(((torch.arange(x.shape[2 + ax_t], dtype=torch.float32, device=x.device) - float(off[ax])) / float(f[ax])))
                    nn_ = lv.shape[2:]
                    grid = torch.stack(torch.meshgrid(*[(2 * u / (n - 1) - 1) for u, n in zip(ups, nn_)], indexing="ij")[::-1], dim=-1).unsqueeze(0)
                    back = F.grid_sample(lv, grid, mode="bilinear", padding_mode="border", align_corners=True)
                    recs[label] = (back, lv)
                    rec = dict(volume=vname, level=level, policy=label, out_shape=list(lv.shape[2:]), f=[round(float(v), 4) for v in f],
                               mean=float(lv.mean()), std=float(lv.std()), vmin=float(lv.min()), vmax=float(lv.max()),
                               ms=1000 * dt, ok=True)
                except Exception as e:
                    rec = dict(volume=vname, level=level, policy=label, ok=False, error=f"{type(e).__name__}: {str(e)[:160]}")
                _append(out, rec)
            for l1 in recs:
                for l2 in recs:
                    if l1 < l2:
                        d = (recs[l1][0] - recs[l2][0])
                        _append(out, dict(volume=vname, level=level, pair=[l1, l2],
                                          rmse_after_mapping_to_full_grid=float(d.pow(2).mean().sqrt())))
    print("deformable characterisation written to", out)


def cmd_probe_truth(a):
    """Is a synthetic failure an objective problem (the truth scores worse than the solved pose) or a search problem?"""
    import tempfile
    import ants
    out = os.path.join(a.out, "probe_truth.jsonl")
    for ratio, pose in ((4, "A"), (6, "A"), (8, "E"), (10, "A"), (4, "F")):
        fixed, mov, tx = slab_pair(ratio, pose)
        tp = os.path.join(tempfile.mkdtemp(), "truth.mat")
        ants.write_transform(tx, tp)
        for lab in a.policies or ["DEFAULT", "P4", "P5+m"]:
            reg, dt, err = run_robust(fixed, mov, lab, a.device, 42)
            l_truth, _ = loss_at_pose(fixed, mov, tp, lab, a.device, 42)
            rec = dict(ratio=ratio, pose=pose, policy=lab, tre=tre(reg, tx), loss_solved_coarse=float(reg["self_check"]["final_loss_coarse"]),
                       loss_truth_coarse=l_truth, mi_native_solved=native_mi(fixed, mov, reg["fwdtransforms"][0]),
                       mi_native_truth=native_mi(fixed, mov, tp))
            _append(out, rec)
            print(json.dumps(rec), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    def common(p):
        p.add_argument("--out", required=True)
        p.add_argument("--device", default="mps")
        p.add_argument("--policies", nargs="*", default=None)
    p = sub.add_parser("synthetic"); common(p)
    p.add_argument("--ratios", nargs="+", type=int, default=[1, 2, 4, 6, 8, 10])
    p.add_argument("--poses", nargs="+", default=["A", "D", "E", "F"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    p.set_defaults(fn=cmd_synthetic)
    p = sub.add_parser("real"); common(p)
    p.add_argument("--subjects-dir", default="/Users/stnava/data/nm_registration_debug/subjects")
    p.add_argument("--subjects", nargs="+", default=[f"sub-{i:02d}" for i in range(1, 11)])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.set_defaults(fn=cmd_real)
    p = sub.add_parser("isotropic"); common(p)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.set_defaults(fn=cmd_isotropic)
    p = sub.add_parser("motion"); common(p)
    p.add_argument("--repeats", type=int, default=2)
    p.set_defaults(fn=cmd_motion)
    p = sub.add_parser("properties"); common(p); p.set_defaults(fn=cmd_properties)
    p = sub.add_parser("probe_truth"); common(p); p.set_defaults(fn=cmd_probe_truth)
    p = sub.add_parser("deformable"); common(p)
    p.add_argument("--t1", default="/Users/stnava/data/repos/antsxfunctional/data_cache/ds001242/sub-01/anat/T1w.nii.gz")
    p.set_defaults(fn=cmd_deformable)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    a.fn(a)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    main()
