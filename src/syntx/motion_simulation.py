"""Ground-truth simulations for motion correction under encoding-locked systematic noise.

Every modality that cycles an encoding (ASL tag/control, slow-flow ``[b0, +X, +Y, +Z]``, DWI gradient directions) or
breathes (BOLD) adds pose that is not head motion. It comes in two physically different kinds, and they must be
handled differently:

* **Registration bias** (contrast-driven pseudomotion). The anatomy did not move; the frame's contrast differs from the
  reference (perfusion in ASL, attenuated CSF in flow-encoded and diffusion-weighted frames, direction-dependent white
  matter in DWI), and the metric optimum is displaced. It must be REMOVED from the written transforms -- applying it
  would move anatomy that never moved.
* **Real encoding-locked displacement**. The image really is displaced: eddy currents of the encoding gradient shift /
  shear / scale the EPI image along phase encode (DWI, slow flow); the respiratory field offset shifts it along phase
  encode (BOLD). It must be APPLIED to the data (the voxels must be put back) but is not head motion, so it must be
  excluded from FD and from the reported motion parameters.

The simulators below generate a 3-D head phantom (gray matter shell, white matter with a fibre orientation field,
ventricles, subarachnoid CSF, smooth texture so every rigid degree of freedom is constrained), move it with a known
rigid head trajectory, render each frame with its modality-specific contrast (bias of the first kind arises from the
contrast alone, it is never injected), optionally displace it with a known encoding-locked affine along phase encode
(second kind), and add noise. ``evaluate_motion_estimate`` scores a motion-correction result against the truth.

Convention: the true pose ``H_t`` maps reference (anatomy) points to frame points, ``frame_t(y) = content(H_t^-1 y)``,
which is what ``motion_correction`` writes as ``fwdtransforms`` (reference -> frame). Poses are compared in tangent
coordinates ``[d, w]``: displacement (mm) of the phantom centre of mass ``p0`` and rotation vector (rad).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import ants
import numpy as np
from scipy.spatial.transform import Rotation as _Rot

from .motion_batched import is_eddy_transform

__all__ = [
    "HeadPhantom",
    "make_head_phantom",
    "simulate_head_motion",
    "simulate_asl_series",
    "simulate_slowflow_series",
    "simulate_dwi_series",
    "simulate_bold_series",
    "evaluate_motion_estimate",
    "pose_to_matrix",
    "matrix_to_pose",
]


@dataclass
class HeadPhantom:
    """Tissue fractions on one grid (numpy, x-y-z), the grid itself and the fibre field."""

    reference: ants.ANTsImage
    gm: np.ndarray
    wm: np.ndarray
    csf: np.ndarray
    texture: np.ndarray
    fibre: np.ndarray  # (X, Y, Z, 3) unit white-matter fibre direction
    p0: np.ndarray  # physical centre of mass of the brain (mm)


@dataclass
class SimulatedSeries:
    image: ants.ANTsImage  # 4-D series
    head_pose: np.ndarray  # (T, 6) true head motion [d_p0 (mm), rotvec (rad)]
    phantom: HeadPhantom
    encoding_covariates: Optional[np.ndarray] = None  # (T,) or (T, K): what the caller would pass to motion correction
    state_labels: Optional[np.ndarray] = None
    pe_axis: Optional[np.ndarray] = None  # physical unit vector
    eddy: Optional[np.ndarray] = None  # (T, 4) true encoding-locked PE affine a = [a0 (mm), shear/scale (3)]
    extra: dict = field(default_factory=dict)


def make_head_phantom(shape=(36, 40, 32), spacing=(2.5, 2.5, 2.5), seed: int = 0, direction=None) -> HeadPhantom:
    """Ellipsoidal head phantom: GM shell, WM core with fibre field, lateral + third ventricles, subarachnoid CSF and
    smooth asymmetric texture (so rotations about every axis are constrained). ``direction`` (3x3) makes it oblique."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = shape
    gx, gy, gz = np.meshgrid(
        (np.arange(nx) - (nx - 1) / 2.0) / (nx / 2.0),
        (np.arange(ny) - (ny - 1) / 2.0) / (ny / 2.0),
        (np.arange(nz) - (nz - 1) / 2.0) / (nz / 2.0),
        indexing="ij",
    )

    def soft(r, edge, width=0.06):
        return 1.0 / (1.0 + np.exp((r - edge) / width))

    r_head = np.sqrt((gx / 0.80) ** 2 + (gy / 0.86) ** 2 + (gz / 0.78) ** 2)
    brain = soft(r_head, 0.92)
    outer_csf = soft(r_head, 0.99) - brain
    wm_core = soft(r_head, 0.70)
    lv_l = soft(np.sqrt(((gx + 0.17) / 0.10) ** 2 + ((gy - 0.05) / 0.32) ** 2 + ((gz - 0.08) / 0.14) ** 2), 1.0, 0.12)
    lv_r = soft(np.sqrt(((gx - 0.15) / 0.09) ** 2 + ((gy - 0.08) / 0.30) ** 2 + ((gz - 0.10) / 0.13) ** 2), 1.0, 0.12)
    v3 = soft(np.sqrt((gx / 0.04) ** 2 + ((gy + 0.05) / 0.16) ** 2 + ((gz + 0.12) / 0.10) ** 2), 1.0, 0.15)
    vent = np.clip(lv_l + lv_r + v3, 0.0, 1.0)
    wm = np.clip(wm_core - vent, 0.0, 1.0) * brain
    csf = np.clip(outer_csf + vent * brain, 0.0, 1.0)
    gm = np.clip(brain - wm - vent * brain, 0.0, 1.0)

    # smooth random texture (low-pass noise), asymmetric so the phantom has no symmetry axis
    from scipy.ndimage import gaussian_filter

    tex = gaussian_filter(rng.normal(size=shape), sigma=2.5)
    tex = 0.12 * tex / max(float(np.abs(tex).max()), 1e-9) + 0.06 * gx - 0.04 * gy * gz

    # fibre field: left-right in a callosal slab above the ventricles, superior-inferior laterally, anterior-posterior
    # elsewhere (direction-dependent DWI contrast)
    fib = np.zeros(shape + (3,))
    callosal = (np.abs(gz - 0.28) < 0.12) & (np.abs(gx) < 0.45)
    lateral = (np.abs(gx) > 0.35) & ~callosal
    fib[..., 1] = 1.0
    fib[lateral] = [0.0, 0.2, 1.0]
    fib[callosal] = [1.0, 0.0, 0.1]
    fib /= np.linalg.norm(fib, axis=-1, keepdims=True)

    ref = ants.from_numpy(np.zeros(shape, dtype=np.float32), origin=(0.0, 0.0, 0.0), spacing=tuple(float(s) for s in spacing))
    if direction is not None:
        ref.set_direction(np.asarray(direction, dtype=np.float64))
    brain_img = ants.from_numpy((gm + wm).astype(np.float32))
    brain_img = ants.copy_image_info(ref, brain_img)
    p0 = np.asarray(ants.get_center_of_mass(brain_img), dtype=np.float64)
    return HeadPhantom(reference=ref, gm=gm, wm=wm, csf=csf, texture=tex, fibre=fib, p0=p0)


def pose_to_matrix(pose: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """4x4 of ``y = R (x - p0) + p0 + d`` for ``pose = [d, rotvec]``."""
    R = _Rot.from_rotvec(np.asarray(pose[3:6], dtype=np.float64)).as_matrix()
    A = np.eye(4)
    A[:3, :3] = R
    A[:3, 3] = p0 + np.asarray(pose[:3], dtype=np.float64) - R @ p0
    return A


def matrix_to_pose(A: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """Inverse of ``pose_to_matrix`` for the rigid part of ``A`` (displacement of ``p0`` and rotation vector)."""
    R = A[:3, :3]
    d = R @ p0 + A[:3, 3] - p0
    U, _, Vt = np.linalg.svd(R)
    return np.r_[d, _Rot.from_matrix(U @ Vt).as_rotvec()]


def _tx_from_matrix(A: np.ndarray, center: np.ndarray) -> ants.ANTsTransform:
    M = A[:3, :3]
    t = A[:3, 3] - center + M @ center
    tx = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3)
    tx.set_parameters(np.concatenate([M.ravel(), t]))
    tx.set_fixed_parameters(np.asarray(center, dtype=np.float64))
    return tx


def _matrix_from_tx(tx: ants.ANTsTransform) -> np.ndarray:
    p = np.asarray(tx.parameters, dtype=np.float64)
    c = np.asarray(tx.fixed_parameters, dtype=np.float64)[:3]
    M = p[:9].reshape(3, 3)
    A = np.eye(4)
    A[:3, :3] = M
    A[:3, 3] = p[9:12] + c - M @ c
    return A


def simulate_head_motion(
    n_frames: int,
    *,
    seed: int = 0,
    drift_mm: float = 0.6,
    drift_deg: float = 0.5,
    jumps: tuple = ((0.55, 1.2, 0.8),),
    jitter_mm: float = 0.04,
    nod_mm: float = 0.0,
    nod_deg: float = 0.0,
    nod_period_frames: float = 7.0,
    alt_mm: float = 0.0,
) -> np.ndarray:
    """(T, 6) head trajectory ``[d (mm), rotvec (rad)]``: linear drift, step jumps ``(fraction_of_series, mm, deg)``
    in a random direction, small random jitter, optional sagittal nodding (z translation + pitch about x) and an optional
    REAL frame-to-frame alternation ``alt_mm * (-1)^t`` along a random direction (period-2 head jitter)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_frames)
    u = rng.normal(size=3)
    u /= np.linalg.norm(u)
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    pose = np.zeros((n_frames, 6))
    pose[:, :3] = drift_mm * t[:, None] * u
    pose[:, 3:] = np.radians(drift_deg) * t[:, None] * v
    for frac, mm, deg in jumps:
        k = int(round(frac * (n_frames - 1)))
        a = rng.normal(size=3)
        a /= np.linalg.norm(a)
        b = rng.normal(size=3)
        b /= np.linalg.norm(b)
        pose[k:, :3] += mm * a
        pose[k:, 3:] += np.radians(deg) * b
    pose[:, :3] += jitter_mm * rng.normal(size=(n_frames, 3))
    pose[:, 3:] += (jitter_mm / 50.0) * rng.normal(size=(n_frames, 3))  # same displacement at 50 mm
    if nod_mm or nod_deg:
        ph = 2.0 * np.pi * np.arange(n_frames) / float(nod_period_frames)
        pose[:, 2] += nod_mm * np.sin(ph)
        pose[:, 3] += np.radians(nod_deg) * np.sin(ph)
    if alt_mm:
        w = np.random.default_rng(seed + 77).normal(size=3)
        pose[:, :3] += float(alt_mm) * ((-1.0) ** np.arange(n_frames))[:, None] * (w / np.linalg.norm(w))[None, :]
    return pose


def _render(ph: HeadPhantom, content: np.ndarray, pose: np.ndarray, eddy: Optional[np.ndarray], pe_axis: Optional[np.ndarray],
            noise: float, rng) -> np.ndarray:
    """frame(y) = content(F^-1 y) with F = E o H (head pose, then the encoding-locked PE affine), plus Gaussian noise."""
    content_img = ants.copy_image_info(ph.reference, ants.from_numpy(content.astype(np.float32)))
    A = pose_to_matrix(pose, ph.p0)
    if eddy is not None and np.any(eddy != 0):
        p = np.asarray(pe_axis, dtype=np.float64)
        E = np.eye(4)
        E[:3, :3] = np.eye(3) + np.outer(p, eddy[1:4])
        E[:3, 3] = p * (eddy[0] - float(eddy[1:4] @ ph.p0))
        A = E @ A
    inv = np.linalg.inv(A)
    tx = _tx_from_matrix(inv, ph.p0)
    out = tx.apply_to_image(content_img, reference=ph.reference, interpolation="linear").numpy()
    if noise > 0:
        out = out + noise * rng.normal(size=out.shape)
    return np.clip(out, 0.0, None)


def _stack(ph: HeadPhantom, frames: list, tr: float = 1.0) -> ants.ANTsImage:
    arr = np.stack(frames, axis=-1).astype(np.float32)
    img = ants.from_numpy(arr, origin=(*ph.reference.origin, 0.0), spacing=(*ph.reference.spacing, float(tr)))
    D = np.eye(4)
    D[:3, :3] = ph.reference.direction
    img.set_direction(D)
    return img


def _pe_vector(ph: HeadPhantom, pe_voxel_axis: int) -> np.ndarray:
    return np.asarray(ph.reference.direction, dtype=np.float64)[:, int(pe_voxel_axis)].copy()


def simulate_asl_series(n_frames: int = 24, *, seed: int = 0, perfusion: float = 0.10, noise: float = 0.01,
                        phantom: Optional[HeadPhantom] = None, **motion) -> SimulatedSeries:
    """Alternating control / label. Label = control - dM with dM a perfusion map on gray matter that is stronger
    posteriorly and on the left (asymmetric, so the label-control contrast pulls the metric optimum: tag-control
    registration bias arises from the contrast only)."""
    ph = phantom or make_head_phantom(seed=seed)
    rng = np.random.default_rng(seed + 101)
    gy = np.linspace(-1, 1, ph.gm.shape[1])[None, :, None]
    gx = np.linspace(-1, 1, ph.gm.shape[0])[:, None, None]
    control = (1.0 + ph.texture) * (1.0 * ph.gm + 0.75 * ph.wm) + 0.6 * ph.csf
    dM = perfusion * ph.gm * (1.0 - 0.6 * gy) * (1.0 - 0.4 * gx)
    s = np.array([1.0 if t % 2 == 0 else -1.0 for t in range(n_frames)])
    pose = simulate_head_motion(n_frames, seed=seed, **motion)
    frames = [_render(ph, control if s[t] > 0 else control - dM, pose[t], None, None, noise, rng) for t in range(n_frames)]
    return SimulatedSeries(image=_stack(ph, frames), head_pose=pose, phantom=ph, encoding_covariates=s, state_labels=(s < 0).astype(int))


def _dwi_signal(ph: HeadPhantom, b: float, g: np.ndarray, s0_scale: float = 1.0) -> np.ndarray:
    """b0 contrast CSF 2.0 / GM 1.0 / WM 0.8; diffusion: CSF 3.0e-3, GM 0.8e-3, WM tensor 1.7e-3 / 0.3e-3 along fibre."""
    s0_gm, s0_wm, s0_csf = 1.0, 0.8, 2.0
    if b <= 0 or np.linalg.norm(g) == 0:
        att_wm = att_gm = att_csf = 1.0
    else:
        gu = np.asarray(g, dtype=np.float64) / np.linalg.norm(g)
        cos2 = np.einsum("xyzk,k->xyz", ph.fibre, gu) ** 2
        att_wm = np.exp(-b * (0.3e-3 + (1.7e-3 - 0.3e-3) * cos2))
        att_gm = np.exp(-b * 0.8e-3)
        att_csf = np.exp(-b * 3.0e-3)
    return s0_scale * (1.0 + ph.texture) * (s0_gm * ph.gm * att_gm + s0_wm * ph.wm * att_wm) + s0_csf * ph.csf * att_csf


def simulate_slowflow_series(n_cycles: int = 6, *, seed: int = 0, b_flow: float = 500.0, noise: float = 0.01,
                             eddy_mm: float = 0.0, pe_voxel_axis: int = 1, phantom: Optional[HeadPhantom] = None,
                             **motion) -> SimulatedSeries:
    """Cyclic ``[b0, +X, +Y, +Z]`` velocity-encoded magnitude series. The flow-encoding gradients diffusion-weight the
    encoded states (CSF attenuated, white matter direction-dependent): contrast-driven registration bias. With
    ``eddy_mm > 0`` each encoded state is also REALLY shifted along phase encode by a state-locked eddy translation."""
    ph = phantom or make_head_phantom(seed=seed)
    rng = np.random.default_rng(seed + 202)
    dirs = [np.zeros(3), *np.eye(3)]  # b0 and the three flow-encoding axes
    states = np.tile(np.arange(4), n_cycles)
    T = len(states)
    pose = simulate_head_motion(T, seed=seed, **motion)
    p = _pe_vector(ph, pe_voxel_axis)
    eddy_by_state = np.array([[0, 0, 0, 0], [eddy_mm, 0.002 * eddy_mm, 0, 0], [-0.7 * eddy_mm, 0, 0.002 * eddy_mm, 0],
                              [0.4 * eddy_mm, 0, 0, -0.002 * eddy_mm]], dtype=np.float64)
    eddy = eddy_by_state[states]
    contents = [_dwi_signal(ph, 0.0 if k == 0 else b_flow, dirs[k]) for k in range(4)]
    frames = [_render(ph, contents[states[t]], pose[t], eddy[t], p, noise, rng) for t in range(T)]
    bvals = np.where(states == 0, 0.0, b_flow)
    bvecs = np.array([dirs[k] for k in states])
    return SimulatedSeries(image=_stack(ph, frames), head_pose=pose, phantom=ph, encoding_covariates=states.astype(float),
                           state_labels=states, pe_axis=p, eddy=eddy, extra={"bvals": bvals, "bvecs": bvecs})


def simulate_dwi_series(n_dirs: int = 24, *, seed: int = 0, b: float = 1000.0, b0_every: int = 6, noise: float = 0.01,
                        eddy_mm_per_unit: float = 0.0, pe_voxel_axis: int = 1, phantom: Optional[HeadPhantom] = None,
                        **motion) -> SimulatedSeries:
    """Single-shell DWI with interleaved b0s. Contrast-driven bias comes from the direction-dependent white matter and
    the attenuated CSF; with ``eddy_mm_per_unit > 0`` each diffusion frame is REALLY displaced along phase encode by an
    eddy affine linear in the gradient vector ``a = C g`` (translation ``eddy_mm_per_unit * g``, small shear/scale)."""
    ph = phantom or make_head_phantom(seed=seed)
    rng = np.random.default_rng(seed + 303)
    g = rng.normal(size=(n_dirs, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    bvals, bvecs = [], []
    k = 0
    while k < n_dirs:
        if len(bvals) % b0_every == 0:
            bvals.append(0.0)
            bvecs.append(np.zeros(3))
        else:
            bvals.append(b)
            bvecs.append(g[k])
            k += 1
    bvals, bvecs = np.asarray(bvals), np.asarray(bvecs)
    T = len(bvals)
    pose = simulate_head_motion(T, seed=seed, **motion)
    p = _pe_vector(ph, pe_voxel_axis)
    C = np.array([[eddy_mm_per_unit, -0.6 * eddy_mm_per_unit, 0.3 * eddy_mm_per_unit],
                  [0.002 * eddy_mm_per_unit, 0, 0], [0, 0.002 * eddy_mm_per_unit, 0], [0, 0, -0.002 * eddy_mm_per_unit]])
    eddy = (bvecs * (bvals[:, None] / max(b, 1.0))) @ C.T
    frames = [_render(ph, _dwi_signal(ph, bvals[t], bvecs[t]), pose[t], eddy[t], p, noise, rng) for t in range(T)]
    return SimulatedSeries(image=_stack(ph, frames), head_pose=pose, phantom=ph, pe_axis=p, eddy=eddy,
                           extra={"bvals": bvals, "bvecs": bvecs, "C": C})


def simulate_multishell_dwi_series(n_dirs_per_shell: int = 12, *, seed: int = 0, shells=(1500.0, 3000.0),
                                   b0_every: int = 16, noise: float = 0.01, eddy_mm_per_unit: float = 0.0,
                                   pe_voxel_axis: int = 1, phantom: Optional[HeadPhantom] = None,
                                   **motion) -> SimulatedSeries:
    """Multi-shell DWI laid out like the FPA protocol: the shells alternate frame by frame (one direction each) with a b0
    every ``b0_every`` frames. With ``eddy_mm_per_unit > 0`` each diffusion frame is REALLY displaced along phase encode
    by an eddy affine linear in the gradient vector scaled by the gradient amplitude, ``a = C g sqrt(b / b_max)``."""
    ph = phantom or make_head_phantom(seed=seed)
    rng = np.random.default_rng(seed + 505)
    shells = tuple(float(b) for b in shells)
    dirs = {b: rng.normal(size=(n_dirs_per_shell, 3)) for b in shells}
    for b in shells:
        dirs[b] /= np.linalg.norm(dirs[b], axis=1, keepdims=True)
    bvals, bvecs, used = [], [], {b: 0 for b in shells}
    i = 0
    while any(used[b] < n_dirs_per_shell for b in shells):
        if len(bvals) % b0_every == 0:
            bvals.append(0.0)
            bvecs.append(np.zeros(3))
            continue
        b = shells[i % len(shells)]
        i += 1
        if used[b] >= n_dirs_per_shell:
            continue
        bvals.append(b)
        bvecs.append(dirs[b][used[b]])
        used[b] += 1
    bvals, bvecs = np.asarray(bvals), np.asarray(bvecs)
    T = len(bvals)
    pose = simulate_head_motion(T, seed=seed, **motion)
    p = _pe_vector(ph, pe_voxel_axis)
    C = np.array([[eddy_mm_per_unit, -0.6 * eddy_mm_per_unit, 0.3 * eddy_mm_per_unit],
                  [0.002 * eddy_mm_per_unit, 0, 0], [0, 0.002 * eddy_mm_per_unit, 0], [0, 0, -0.002 * eddy_mm_per_unit]])
    eddy = (bvecs * np.sqrt(bvals / max(shells))[:, None]) @ C.T
    frames = [_render(ph, _dwi_signal(ph, bvals[t], bvecs[t]), pose[t], eddy[t], p, noise, rng) for t in range(T)]
    return SimulatedSeries(image=_stack(ph, frames), head_pose=pose, phantom=ph, pe_axis=p, eddy=eddy,
                           extra={"bvals": bvals, "bvecs": bvecs, "C": C})


def simulate_bold_series(n_frames: int = 48, *, seed: int = 0, resp_mm: float = 0.4, tr: float = 2.0, resp_hz: float = 0.28,
                         noise: float = 0.01, pe_voxel_axis: int = 1, phantom: Optional[HeadPhantom] = None,
                         **motion) -> SimulatedSeries:
    """BOLD EPI with a respiratory B0 shift along phase encode (REAL image displacement, not head motion) sampled at TR
    (aliased), plus head motion (pass ``nod_mm`` / ``nod_deg`` for real sagittal nodding)."""
    ph = phantom or make_head_phantom(seed=seed)
    rng = np.random.default_rng(seed + 404)
    content = (1.0 + ph.texture) * (1.0 * ph.gm + 0.8 * ph.wm) + 1.3 * ph.csf
    pose = simulate_head_motion(n_frames, seed=seed, **motion)
    p = _pe_vector(ph, pe_voxel_axis)
    shift = resp_mm * np.sin(2.0 * np.pi * resp_hz * tr * np.arange(n_frames) + 0.3)
    eddy = np.zeros((n_frames, 4))
    eddy[:, 0] = shift
    frames = [_render(ph, content, pose[t], eddy[t], p, noise, rng) for t in range(n_frames)]
    return SimulatedSeries(image=_stack(ph, frames, tr=tr), head_pose=pose, phantom=ph, pe_axis=p, eddy=eddy,
                           extra={"resp_shift_mm": shift})


def evaluate_motion_estimate(fwdtransforms, sim: SimulatedSeries, fd_radius: float = 50.0) -> dict:
    """Head-motion error of a motion-correction result against the simulated truth.

    The rigid part of each frame's chain (all elements except ``*_eddy.mat``) is composed and converted to ``[d_p0, rotvec]``; both trajectories
    are demeaned over time (the reference sits at an unknown mean pose -- a gauge, not an error). Reports RMS
    translation (mm) and rotation (deg) error, the mean point-displacement error on a ``fd_radius`` sphere (mm), and
    the FD error. Jumps: recovered step size for every true jump > 0.5 mm."""
    p0 = sim.phantom.p0

    def _rigid_pose(chain):
        # every rigid element of the chain (grouped results: corrections, crosses, the frame's own rigid), first element
        # acting first on the point; the eddy PE affine (``*_eddy.mat``) is a real displacement, not head motion
        h = np.eye(4)
        for f in chain:
            if not is_eddy_transform(f):
                h = _matrix_from_tx(ants.read_transform(f)) @ h
        return matrix_to_pose(h, p0)

    est = np.array([_rigid_pose(chain) for chain in fwdtransforms])
    tru = np.asarray(sim.head_pose, dtype=np.float64)
    e = (est - est.mean(axis=0)) - (tru - tru.mean(axis=0))
    disp = np.sqrt((e[:, :3] ** 2).sum(axis=1) + (fd_radius ** 2) * (e[:, 3:] ** 2).sum(axis=1))

    def fd(y):
        d = np.diff(y, axis=0)
        return np.r_[0.0, np.abs(d[:, :3]).sum(axis=1) + fd_radius * np.abs(d[:, 3:]).sum(axis=1)]

    fd_t, fd_e = fd(tru), fd(est)
    jumps = np.where(fd_t > 0.5 + 3.0 * np.median(fd_t))[0]
    return {
        "translation_rms_mm": float(np.sqrt((e[:, :3] ** 2).sum(axis=1).mean())),
        "rotation_rms_deg": float(np.degrees(np.sqrt((e[:, 3:] ** 2).sum(axis=1).mean()))),
        "pose_error_mm": float(disp.mean()),
        "fd_error_mm": float(np.abs(fd_e - fd_t)[1:].mean()),
        "fd_true_mean": float(fd_t[1:].mean()),
        "fd_est_mean": float(fd_e[1:].mean()),
        "jump_recovery": [float(fd_e[k] / fd_t[k]) for k in jumps],
        "estimated_pose": est,
    }
