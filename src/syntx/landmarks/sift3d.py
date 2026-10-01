"""
3-D SIFT-like keypoints and descriptors in PyTorch.

Volumetric generalisation of Lowe's SIFT:

1. DoG keypoints at **physical** scales (sigmas in mm, per-axis voxel sigmas derived from the
   spacing), restricted to foreground: spatial 3x3x3 maxima of |DoG| that are also larger
   than the two neighbouring scales at the same voxel.
2. Physical-space gradient: finite differences along the array axes are
   divided by the spacing and rotated by the direction cosine matrix, so the
   gradient components are expressed in LPS scanner axes.
3. Descriptor sampled on a **millimetre grid** around each keypoint (radius
   ``n_cells * sigma_mm``), trilinearly interpolated from the gradient volume:
   ``n_cells^3`` spatial cells × ``n_bins`` gradient directions (nearest of a
   fixed set of unit vectors on the sphere), Gaussian window weighting,
   L2-normalise / clamp 0.2 / re-normalise (Lowe 2004).

Because both the sampling lattice and the orientation binning live in physical
LPS space, descriptors of the same anatomy stored in different native frames
(axis flips, permutations, anisotropic spacing) are directly comparable.  The
default descriptor is not rotation invariant beyond that, so it suits images that differ
by small rotations.  For larger rotations pass ``rotation_invariant=True``: each
descriptor is expressed in a local frame from the gradient structure tensor
(eigenvectors, signs fixed by a reference vector, see ``_build_descriptor``), trading
discriminative power for invariance to rigid rotation (up to frame ambiguities when the
structure tensor is near-isotropic).  Alternatively ``frame_rotation`` builds the aligned
descriptor in a given globally rotated frame (used by ``orient``'s rotation search).

Functions
---------
detect_sift3d      : keypoint coordinates [N, 4] + descriptors [N, n_cells^3 * n_bins]
sift3d_keypoints   : detection stage only, returns a reusable state dict
sift3d_descriptors : descriptor stage for a state dict
sym3x3_eigh        : batched closed-form 3x3 symmetric eigen-decomposition
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .blob import (
    FOREGROUND_LEVEL,
    _foreground_mask,
    _shift_pad,
    _get_device,
    _normalize_intensity,
    _separable_gaussian3d,
    _to_tensor,
    _voxel_sigmas,
)
from .spatial import get_image_affine, physical_to_vox, vox_to_physical

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gradient computation (physical space)
# ---------------------------------------------------------------------------

def _gradient_axes(vol: torch.Tensor) -> torch.Tensor:
    """Central differences along array axes (half one-sided differences at the borders).
    vol [1,1,nx,ny,nz] -> [1,3,nx,ny,nz]
    ordered (d/dix, d/diy, d/diz) in intensity per index (replicate borders).
    Implemented with slicing (``_shift_pad``), not ``F.pad`` — MPS-safe on
    full-size volumes (see the note in ``blob._shift_pad``)."""
    grads = [0.5 * (_shift_pad(vol, dim, +1, "replicate") - _shift_pad(vol, dim, -1, "replicate"))
             for dim in (2, 3, 4)]
    return torch.cat(grads, dim=1)


def _physical_gradient(
    vol: torch.Tensor,
    spacing,
    direction,
    local_normalize: bool = False,
    norm_sigma_mm: float = 2.0,
) -> torch.Tensor:
    """[1,3,nx,ny,nz] gradient in LPS physical axes (intensity per mm).

    When local_normalize=True, scales gradient vectors by local window variance:
        g_norm = g / sqrt(G_sigma * ||g||^2 + eps)
    where G_sigma is a Gaussian of ``norm_sigma_mm`` (mm) applied to ||g||^2, eps = 1e-6.
    This reduces the dominance of high-contrast boundaries over weaker edges.
    ``spacing`` [3] and ``direction`` [3, 3] are the image geometry.
    """
    g = _gradient_axes(vol)
    sp = torch.tensor(np.asarray(spacing, dtype=np.float32), device=vol.device).view(1, 3, 1, 1, 1)
    g = g / sp
    D = torch.tensor(np.asarray(direction, dtype=np.float32).reshape(3, 3), device=vol.device)
    # g_phys[k] = sum_a D[k, a] * g_axes[a]
    g_phys = torch.einsum("ka,bahwd->bkhwd", D, g)
    if local_normalize:
        norm_sq = torch.sum(g_phys ** 2, dim=1, keepdim=True)
        sig_vox = _voxel_sigmas(norm_sigma_mm, spacing)
        local_var = _separable_gaussian3d(norm_sq, sig_vox)
        denom = torch.sqrt(local_var + 1e-6)
        g_phys = g_phys / denom
    return g_phys


# ---------------------------------------------------------------------------
# Batched closed-form 3x3 symmetric eigensolver (device-agnostic, MPS-safe)
# ---------------------------------------------------------------------------

def sym3x3_eigh(T: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Eigen-decomposition of a batch of symmetric 3x3 matrices ``T`` [K, 3, 3] using
    the trigonometric (Cardano) closed form — elementwise tensor math only, so it
    runs on MPS/CUDA/CPU without ``torch.linalg.eigh``.

    Returns ``(evals [K, 3] descending, evecs [K, 3, 3])`` with eigenvectors in the
    columns (``evecs[:, :, i]`` belongs to ``evals[:, i]``), orthonormalised and
    right-handed (det = +1).  Computed in float64; both outputs are float64.

    Eigenvalues: with q = tr(T)/3, p = sqrt(tr((T - qI)^2) / 6), B = (T - qI)/p,
    phi = acos(clamp(det(B)/2, -1, 1)) / 3:  q + 2p cos(phi), q + 2p cos(phi ± 2π/3).
    Eigenvectors: e1 from the cross product of two rows of (T - λ1 I) (the row pair
    with the largest cross product is chosen), e3 likewise for λ3, e2 = e3 × e1.
    Degenerate spectra fall back to an arbitrary (deterministic) orthonormal frame.
    """
    T = T.double()
    K = T.shape[0]
    I = torch.eye(3, dtype=T.dtype, device=T.device).expand(K, 3, 3)
    q = torch.einsum("kii->k", T) / 3.0
    A = T - q[:, None, None] * I
    p = torch.sqrt(torch.einsum("kij,kij->k", A, A) / 6.0).clamp_min(eps)
    B = A / p[:, None, None]
    r = (torch.linalg.det(B) / 2.0).clamp(-1.0, 1.0)
    phi = torch.acos(r) / 3.0
    two_pi_3 = 2.0 * math.pi / 3.0
    l1 = q + 2.0 * p * torch.cos(phi)
    l3 = q + 2.0 * p * torch.cos(phi + two_pi_3)
    l2 = 3.0 * q - l1 - l3
    evals = torch.stack([l1, l2, l3], dim=1)                       # descending

    def _null_vector(lam: torch.Tensor) -> torch.Tensor:
        M = T - lam[:, None, None] * I                             # rank-2 (generically)
        c01 = torch.cross(M[:, 0], M[:, 1], dim=1)
        c12 = torch.cross(M[:, 1], M[:, 2], dim=1)
        c20 = torch.cross(M[:, 2], M[:, 0], dim=1)
        C = torch.stack([c01, c12, c20], dim=1)                    # [K, 3, 3]
        n = C.norm(dim=2)                                          # [K, 3]
        best = n.argmax(dim=1)
        v = C[torch.arange(K, device=T.device), best]              # [K, 3]
        vn = v.norm(dim=1, keepdim=True)
        # degenerate (repeated eigenvalue): fall back to a fixed axis
        fallback = torch.zeros_like(v); fallback[:, 0] = 1.0
        v = torch.where(vn > eps, v / vn.clamp_min(eps), fallback)
        return v

    e1 = _null_vector(l1)
    e3 = _null_vector(l3)
    # make e3 orthogonal to e1 (guards degenerate cases), then complete right-handed frame
    e3 = e3 - (e3 * e1).sum(dim=1, keepdim=True) * e1
    n3 = e3.norm(dim=1, keepdim=True)
    alt = torch.cross(e1, torch.tensor([0.0, 1.0, 0.0], dtype=T.dtype, device=T.device).expand(K, 3), dim=1)
    alt2 = torch.cross(e1, torch.tensor([0.0, 0.0, 1.0], dtype=T.dtype, device=T.device).expand(K, 3), dim=1)
    alt = torch.where(alt.norm(dim=1, keepdim=True) > 1e-6, alt, alt2)
    e3 = torch.where(n3 > 1e-6, e3 / n3.clamp_min(eps), alt / alt.norm(dim=1, keepdim=True).clamp_min(eps))
    e2 = torch.cross(e3, e1, dim=1)
    evecs = torch.stack([e1, e2, e3], dim=2)                       # columns
    return evals, evecs


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------

def _sphere_directions(n_bins: int) -> np.ndarray:
    """Fixed set of ``n_bins`` roughly uniform unit vectors.  n_bins=8 uses the
    cube corners (±1,±1,±1)/√3; n_bins=6 the axis directions; otherwise a
    Fibonacci sphere.  Deterministic, so descriptors are comparable across runs."""
    if n_bins == 8:
        d = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
    elif n_bins == 6:
        d = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64)
    else:
        i = np.arange(n_bins) + 0.5
        phi = np.arccos(1 - 2 * i / n_bins)
        theta = np.pi * (1 + 5 ** 0.5) * i
        d = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _build_descriptor(
    grad_phys: torch.Tensor,        # [1,3,nx,ny,nz] physical gradient
    image_affine,                   # (origin, spacing, direction)
    kp_mm: np.ndarray,              # [N,3] physical keypoint positions
    kp_sigma_mm: np.ndarray,        # [N]  keypoint scales (mm)
    n_cells: int = 4,
    n_bins: int = 8,
    samples_per_cell: int = 3,
    chunk: int = 64,
    rotation_invariant: bool = False,
    frame_scale: float = 2.0,
    sign_mode: str = "centroid",
    frame_rotation: Optional[np.ndarray] = None,
    return_frames: bool = False,
):
    """
    Physical-space 3D SIFT descriptor.  For each keypoint a cube of half-width
    ``n_cells * sigma`` mm centred on the keypoint is sampled on a regular lattice of
    ``(n_cells * samples_per_cell)^3`` points (cell-centred, spacing scales with sigma).
    Gradient vectors are trilinearly interpolated there (zero outside the image), each
    assigned to its nearest of ``n_bins`` fixed directions (``_sphere_directions``; hard
    assignment), weighted by |g| times a Gaussian window (sigma = half-width), and
    accumulated into ``n_cells^3 * n_bins`` bins; then L2-normalise, clamp at 0.2,
    re-normalise.  Keypoints are processed in chunks of ``chunk``.

    ``rotation_invariant=False``: lattice and direction bins are aligned with the
    LPS scanner axes (frame-independent, but not invariant to head rotation).
    ``rotation_invariant=True``: a local orthonormal frame is estimated per
    keypoint from the Gaussian-weighted gradient structure tensor
    T = sum_i w_i g_i g_i^T (eigenvectors ordered by
    eigenvalue, signs fixed by the weighted mean gradient, right-handed); the
    lattice is then sampled in that frame and gradients are expressed in it,
    making the descriptor invariant to rigid rotation of the anatomy.

    Frame options (rotation_invariant only):
      frame_scale    : the structure tensor is accumulated over a window
                       ``frame_scale`` times the descriptor half-width, so the
                       surrounding anatomy (not just the symmetric blob core)
                       determines the axes.
      sign_mode      : 'centroid' fixes each axis sign by the gradient-magnitude
                       centroid offset along that axis (robust at blob centres
                       where the mean gradient vanishes); 'mean_grad' uses the
                       weighted mean gradient (Rister et al.). Other values raise
                       ValueError.
      The third axis is e1 x e2 (its sign is not tested).
    frame_rotation : optional 3x3 rotation applied to the *axis-aligned* descriptor
    frame (lattice offsets -> R @ offsets, gradients -> R^T g).  Building the
    moving image's descriptors with the global fixed->moving rotation makes them
    directly comparable with the fixed image's axis-aligned descriptors.
    Giving it together with ``rotation_invariant=True`` raises ValueError.
    Returns ``[N, n_cells^3 * n_bins]`` float32, or with ``return_frames`` a tuple
    (descriptors, per-keypoint frames ``[N, 3, 3]``; columns = frame axes, the rotation for
    the aligned descriptor) -- also for N = 0.
    """
    if sign_mode not in ("centroid", "mean_grad"):
        raise ValueError(f"sign_mode must be 'centroid' or 'mean_grad', got {sign_mode!r}")
    if rotation_invariant and frame_rotation is not None:
        raise ValueError("frame_rotation is for the axis-aligned descriptor; it cannot be "
                         "combined with rotation_invariant=True")
    N = kp_mm.shape[0]
    desc_dim = n_cells ** 3 * n_bins
    if N == 0:
        empty = np.zeros((0, desc_dim), dtype=np.float32)
        return (empty, np.zeros((0, 3, 3), dtype=np.float32)) if return_frames else empty

    dev = grad_phys.device
    n_side = n_cells * samples_per_cell
    # unit lattice in [-1, 1) cell-centred, same for every keypoint
    lin = (np.arange(n_side) + 0.5) / n_side * 2.0 - 1.0
    ox, oy, oz = np.meshgrid(lin, lin, lin, indexing="ij")
    unit_off = np.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=1)        # [S, 3]
    S = unit_off.shape[0]
    cell_of_sample = (
        (np.arange(n_side) // samples_per_cell)[:, None, None] * n_cells * n_cells
        + (np.arange(n_side) // samples_per_cell)[None, :, None] * n_cells
        + (np.arange(n_side) // samples_per_cell)[None, None, :]
    ).ravel()                                                                # [S]
    gauss_w = np.exp(-0.5 * (np.sum(unit_off ** 2, axis=1)))                # window sigma = half-width
    unit_off_t = torch.from_numpy(unit_off.astype(np.float32)).to(dev)
    cell_t = torch.from_numpy(cell_of_sample.astype(np.int64)).to(dev)
    gauss_t = torch.from_numpy(gauss_w.astype(np.float32)).to(dev)
    dirs_t = torch.from_numpy(_sphere_directions(n_bins).astype(np.float32)).to(dev)   # [B, 3]

    n_vox = np.array(grad_phys.shape[2:5], dtype=np.float64)
    denom = np.maximum(n_vox - 1.0, 1.0)
    out = torch.zeros((N, desc_dim), dtype=torch.float32, device=dev)
    frames = torch.eye(3, device=dev).expand(N, 3, 3).clone()
    R_glob = None
    if frame_rotation is not None:
        R_glob = torch.from_numpy(np.asarray(frame_rotation, dtype=np.float32).reshape(3, 3)).to(dev)

    def _sample(pos_mm: np.ndarray, K: int) -> torch.Tensor:
        """Trilinearly interpolate the physical gradient at [K*S, 3] mm positions (zeros
        outside the image) -> [K, S, 3]."""
        idx = physical_to_vox(image_affine, pos_mm)                          # (ix, iy, iz)
        norm = idx / denom * 2.0 - 1.0
        grid = torch.from_numpy(norm[:, ::-1].copy().astype(np.float32)).to(dev).view(1, 1, 1, -1, 3)
        g = F.grid_sample(grad_phys, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return g.reshape(3, K, S).permute(1, 2, 0)

    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        kp = kp_mm[start:end].astype(np.float64)                             # [K, 3]
        half = (n_cells * kp_sigma_mm[start:end]).astype(np.float64)         # [K]
        K = kp.shape[0]
        # axis-aligned physical sample positions [K, S, 3]
        pos = kp[:, None, :] + unit_off[None, :, :] * half[:, None, None]
        g = _sample(pos.reshape(-1, 3), K)                                   # [K, S, 3]

        if R_glob is not None and not rotation_invariant:
            # descriptor in a globally rotated frame: resample lattice along R columns, express g in it
            off_rot = (unit_off_t @ R_glob.T)                                # [S, 3]  = R @ off
            pos_r = torch.from_numpy(kp.astype(np.float32)).to(dev)[:, None, :] + \
                    off_rot[None, :, :] * torch.from_numpy(half.astype(np.float32)).to(dev)[:, None, None]
            g = _sample(pos_r.reshape(-1, 3).cpu().numpy().astype(np.float64), K)
            g = g @ R_glob                                                   # components along R columns
            frames[start:end] = R_glob

        if rotation_invariant:
            # local frame from the weighted structure tensor over a (larger) frame window
            if abs(frame_scale - 1.0) > 1e-6:
                pos_f = kp[:, None, :] + unit_off[None, :, :] * (half * frame_scale)[:, None, None]
                gf = _sample(pos_f.reshape(-1, 3), K)
            else:
                gf = g
            wgf = gf * gauss_t[None, :, None]
            T = torch.einsum("ksa,ksb->kab", wgf, gf)                        # [K, 3, 3]
            # closed-form batched 3x3 eigensolve on the device (no torch.linalg.eigh → MPS-safe)
            evals, E = sym3x3_eigh(T)
            E = E.float()                                                    # columns: e1 (largest), e2, e3
            ev = evals.float().clamp_min(1e-12)
            if sign_mode == "mean_grad":
                ref = wgf.sum(dim=1)                                         # [K, 3] weighted mean gradient
            else:
                # gradient-magnitude centroid offset (unit lattice coordinates)
                magf = gf.norm(dim=2) * gauss_t[None, :]                     # [K, S]
                ref = torch.einsum("ks,sc->kc", magf, unit_off_t)            # [K, 3]
            s1 = torch.sign(torch.einsum("kc,kc->k", ref, E[:, :, 0])); s1[s1 == 0] = 1
            s2 = torch.sign(torch.einsum("kc,kc->k", ref, E[:, :, 1])); s2[s2 == 0] = 1
            e1 = E[:, :, 0] * s1[:, None]
            e2 = E[:, :, 1] * s2[:, None]
            e3 = torch.cross(e1, e2, dim=1)                                  # right-handed
            R = torch.stack([e1, e2, e3], dim=2)                             # [K, 3, 3], columns = frame axes
            frames[start:end] = R
            # resample the lattice in the local frame and express gradients in it
            off_local = torch.einsum("kab,sb->ksa", R, unit_off_t)           # [K, S, 3] (physical)
            pos_r = torch.from_numpy(kp.astype(np.float32)).to(dev)[:, None, :] + \
                    off_local * torch.from_numpy(half.astype(np.float32)).to(dev)[:, None, None]
            g = _sample(pos_r.reshape(-1, 3).cpu().numpy().astype(np.float64), K)
            g = torch.einsum("kab,ksa->ksb", R, g)                           # components along e1, e2, e3

        mag = g.norm(dim=2)                                                  # [K, S]
        # nearest direction bin
        proj = torch.einsum("ksc,bc->ksb", g, dirs_t)                        # [K, S, B]
        b = proj.argmax(dim=2)                                               # [K, S]
        w = mag * gauss_t[None, :]                                           # [K, S]
        flat_idx = (torch.arange(K, device=dev)[:, None] * desc_dim
                    + cell_t[None, :] * n_bins + b)                          # [K, S]
        hist = torch.zeros(K * desc_dim, dtype=torch.float32, device=dev)
        hist.index_add_(0, flat_idx.reshape(-1), w.reshape(-1))
        hist = hist.view(K, desc_dim)
        hist = hist / (hist.norm(dim=1, keepdim=True) + 1e-8)
        hist = hist.clamp(max=0.2)
        hist = hist / (hist.norm(dim=1, keepdim=True) + 1e-8)
        out[start:end] = hist

    descs = out.cpu().numpy().astype(np.float32)
    if return_frames:
        return descs, frames.cpu().numpy()
    return descs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_sift3d(
    image,
    sigma_min: float = 1.5,
    sigma_max: float = 6.0,
    n_scales: int = 5,
    threshold: float = 0.005,
    min_distance_mm: float = 4.0,
    max_keypoints: int = 256,
    n_cells: int = 4,
    n_bins: int = 8,
    device: Optional[str] = None,
    preprocess: bool = True,
    use_n4: bool = False,
    use_denoise: bool = True,
    samples_per_cell: int = 3,
    rotation_invariant: bool = False,
    frame_rotation: Optional[np.ndarray] = None,
    return_frames: bool = False,
    spatial_bucketing: bool = True,
    grid_bins: int = 4,
    local_normalize: bool = False,
):
    """
    Full 3D SIFT: DoG keypoint detection + physical-space 3D gradient histogram
    descriptors (``sift3d_keypoints`` then ``sift3d_descriptors``).

    Parameters
    ----------
    image : ants.ANTsImage | np.ndarray | torch.Tensor
        3-D volume. ndarray / tensor input has identity geometry and is not preprocessed.
    sigma_min, sigma_max : float, default 1.5, 6.0
        Gaussian scale range in **mm**.
    n_scales : int, default 5
        Number of Gaussian levels (n_scales-1 DoG maps; the first and last DoG map never
        produce keypoints, so at least 4 are needed).
    threshold : float, default 0.005
        Relative response threshold: a candidate must exceed ``threshold`` times
        the maximum over the foreground of that level's |σ²-normalised DoG| (or of its
        locally normalised version with ``local_normalize``).  The blob detectors instead
        threshold the raw response.
    min_distance_mm : float, default 4.0
        Greedy 3-D NMS suppression radius in mm.
    max_keypoints : int, default 256
        Upper bound on returned keypoints.
    n_cells : int, default 4
        Descriptor spatial grid cells per axis (n_cells^3 total cells).
    n_bins : int, default 8
        Gradient direction bins per spatial cell (8 = cube-corner directions, 6 = axes,
        otherwise a Fibonacci sphere).
    device : str | None
        Torch device; auto-selected (mps > cuda > cpu) if None.
    preprocess, use_n4, use_denoise : bool, default True, False, True
        ``preprocess_for_landmarks`` options (ANTsImage input only).
    samples_per_cell : int, default 3
        Lattice samples per descriptor cell per axis (3 -> 12^3 = 1728 samples).
    rotation_invariant : bool, default False
        Express each descriptor in a local structure-tensor frame instead of the
        scanner axes (see ``_build_descriptor``).  Less discriminative than the
        aligned descriptor; mainly useful to *estimate* a global rotation via
        ``syntx.landmarks.orient.estimate_rotation_from_frames``.
    frame_rotation : 3x3 array | None
        Build axis-aligned descriptors in a globally rotated frame (see
        ``_build_descriptor``); used by the rotation-search matching pipeline.
        Raises ValueError together with ``rotation_invariant=True``.
    return_frames : bool, default False
        Also return the per-keypoint frames ``[N, 3, 3]``.
    spatial_bucketing, grid_bins : bool, int, default True, 4
        NMS bucketing (see ``blob._greedy_nms``; same algorithm).
    local_normalize : bool, default False
        If True, the DoG used for the spatial-max / threshold test is divided by its
        local RMS, and the descriptor gradients are divided by their local RMS magnitude
        (``_physical_gradient``).  The scale test and the stored response use the
        un-normalised DoG.

    Returns
    -------
    coords : np.ndarray, shape [N, 4], float32
        (x_mm, y_mm, z_mm, sigma_mm) in physical LPS space; sigma_mm is the DoG scale
        sqrt(sigma_i * sigma_{i+1}).
    descriptors : np.ndarray, shape [N, n_cells**3 * n_bins], float32
        L2-normalised 3-D SIFT descriptors (default 512-D).
    frames : np.ndarray, shape [N, 3, 3], float32
        Only if ``return_frames``.
    """
    state = sift3d_keypoints(image, sigma_min=sigma_min, sigma_max=sigma_max, n_scales=n_scales,
                             threshold=threshold, min_distance_mm=min_distance_mm, max_keypoints=max_keypoints,
                             device=device, preprocess=preprocess, use_n4=use_n4, use_denoise=use_denoise,
                             spatial_bucketing=spatial_bucketing, grid_bins=grid_bins,
                             local_normalize=local_normalize)
    pts = state["pts"]
    empty_ret = (np.zeros((0, 4), dtype=np.float32), np.zeros((0, n_cells ** 3 * n_bins), dtype=np.float32)) + \
                ((np.zeros((0, 3, 3), dtype=np.float32),) if return_frames else ())
    if pts.shape[0] == 0:
        return empty_ret
    descs, frames = sift3d_descriptors(state, n_cells=n_cells, n_bins=n_bins, samples_per_cell=samples_per_cell,
                                       rotation_invariant=rotation_invariant, frame_rotation=frame_rotation,
                                       return_frames=True)
    if return_frames:
        return pts, descs, frames
    return pts, descs


def sift3d_keypoints(
    image,
    sigma_min: float = 1.5,
    sigma_max: float = 6.0,
    n_scales: int = 5,
    threshold: float = 0.005,
    min_distance_mm: float = 4.0,
    max_keypoints: int = 256,
    device: Optional[str] = None,
    preprocess: bool = True,
    use_n4: bool = False,
    use_denoise: bool = True,
    spatial_bucketing: bool = True,
    grid_bins: int = 4,
    local_normalize: bool = False,
) -> dict:
    """
    Stage 1 of ``detect_sift3d``: preprocessing, DoG keypoints, NMS and the physical-space
    gradient volume.  Parameters as in ``detect_sift3d``.

    A voxel of DoG level s (interior levels only) is a candidate if it is in the foreground
    (normalised intensity > 0.05), is a 3x3x3 spatial maximum of the (optionally locally
    normalised) |σ²-normalised DoG| above ``threshold`` times that level's foreground
    maximum, and its |DoG| is larger than at levels s-1 and s+1 at the same voxel.
    Candidates are thinned by greedy NMS in decreasing |DoG| (with the same spatial
    bucketing as ``blob._greedy_nms``).

    Returns a state dict
    ``{'pts': [N,4], 'response': [N] |σ²-normalised DoG|, 'grad': [1,3,nx,ny,nz] tensor,
    'affine': (o, s, D), 'vol': [1,1,nx,ny,nz] normalised tensor}``
    that ``sift3d_descriptors`` can consume repeatedly (e.g. once per candidate
    frame rotation) without redoing detection.  With no keypoints, ``pts`` / ``response``
    are empty and ``grad`` is None.  Tensors stay on the compute device.
    """
    if preprocess and hasattr(image, "numpy") and hasattr(image, "new_image_like"):
        from syntx.landmarks.preprocess import preprocess_for_landmarks
        image = preprocess_for_landmarks(image, use_n4=use_n4, use_denoise=use_denoise)

    dev = _get_device(device)
    vol = _to_tensor(image, dev)                    # [1,1,nx,ny,nz]
    vol = _normalize_intensity(vol)
    origin, spacing, direction = get_image_affine(image)
    affine = (origin, spacing, direction)
    fg = _foreground_mask(vol, FOREGROUND_LEVEL)

    sigmas = np.geomspace(sigma_min, sigma_max, n_scales).tolist()      # mm
    gaussians = [_separable_gaussian3d(vol, _voxel_sigmas(s, spacing)) for s in sigmas]

    dog_images, dog_sigmas = [], []
    for i in range(len(sigmas) - 1):
        dog = gaussians[i + 1] - gaussians[i]
        dog_images.append(dog * (sigmas[i] ** 2))
        dog_sigmas.append(math.sqrt(sigmas[i] * sigmas[i + 1]))

    empty_state = dict(pts=np.zeros((0, 4), dtype=np.float32), response=np.zeros(0, dtype=np.float32), grad=None, affine=affine, vol=vol)
    if len(dog_images) < 3:
        return empty_state

    all_pts: list[np.ndarray] = []
    for s in range(1, len(dog_images) - 1):
        curr, prev, nxt = dog_images[s], dog_images[s - 1], dog_images[s + 1]
        curr_abs = curr.abs()
        curr_eval = curr_abs
        if local_normalize:
            sig_vox = _voxel_sigmas(dog_sigmas[s], spacing)
            local_rms = torch.sqrt(_separable_gaussian3d(curr ** 2, sig_vox) + 1e-6)
            curr_eval = curr_abs / (local_rms + 1e-3)

        scale_max = float((curr_eval * fg).max())
        if scale_max < 1e-12:
            continue
        max_pool = F.max_pool3d(curr_eval, 3, 1, 1)
        is_local = (curr_eval == max_pool) & (curr_eval > threshold * scale_max)
        is_scale = (curr.abs() > prev.abs()) & (curr.abs() > nxt.abs())
        mask_np = (is_local & is_scale & fg).squeeze().cpu().numpy()
        idxs = np.argwhere(mask_np)                                   # [K, 3] (ix, iy, iz)
        if idxs.shape[0] == 0:
            continue
        phys = vox_to_physical(affine, idxs)
        resp = curr.squeeze().cpu().numpy()[tuple(idxs.T)]
        all_pts.append(np.column_stack([
            phys,
            np.full(len(idxs), dog_sigmas[s], dtype=np.float32),
            np.abs(resp).astype(np.float32),
        ]))

    if not all_pts:
        return empty_state

    cand = np.vstack(all_pts)                                        # [M, 5] x y z sigma |resp|
    # Greedy NMS ordered by response strength (strongest first)
    order = np.argsort(-cand[:, 4])
    cand = cand[order]
    kept_coords: list[np.ndarray] = []
    kept_rows: list[int] = []

    if spatial_bucketing and cand.shape[0] > max_keypoints:
        coords = cand[:, :3]
        p_min = coords.min(axis=0)
        p_max = coords.max(axis=0)
        span = np.maximum(p_max - p_min, 1e-3)
        bin_idx = np.clip(((coords - p_min) / span * grid_bins).astype(int), 0, grid_bins - 1)
        cell_id = bin_idx[:, 0] * (grid_bins * grid_bins) + bin_idx[:, 1] * grid_bins + bin_idx[:, 2]
        unique_cells = np.unique(cell_id)
        quota = max(1, int(np.ceil(max_keypoints / len(unique_cells))))

        cell_counts = {c: 0 for c in unique_cells}
        for i in range(cand.shape[0]):
            c = cell_id[i]
            if cell_counts[c] < quota:
                if kept_coords:
                    d = np.linalg.norm(np.stack(kept_coords) - cand[i, :3], axis=1)
                    if d.min() < min_distance_mm:
                        continue
                kept_rows.append(i)
                kept_coords.append(cand[i, :3])
                cell_counts[c] += 1
                if len(kept_rows) >= max_keypoints:
                    break

        if len(kept_rows) < max_keypoints:
            kept_set = set(kept_rows)
            for i in range(cand.shape[0]):
                if i in kept_set:
                    continue
                if kept_coords:
                    d = np.linalg.norm(np.stack(kept_coords) - cand[i, :3], axis=1)
                    if d.min() < min_distance_mm:
                        continue
                kept_rows.append(i)
                kept_coords.append(cand[i, :3])
                if len(kept_rows) >= max_keypoints:
                    break
    else:
        for i in range(cand.shape[0]):
            if len(kept_rows) >= max_keypoints:
                break
            if kept_coords:
                d = np.linalg.norm(np.stack(kept_coords) - cand[i, :3], axis=1)
                if d.min() < min_distance_mm:
                    continue
            kept_rows.append(i)
            kept_coords.append(cand[i, :3])

    if not kept_rows:
        return empty_state

    pts = cand[kept_rows, :4].astype(np.float32)
    grad_phys = _physical_gradient(vol, spacing, direction, local_normalize=local_normalize)
    return dict(pts=pts, response=cand[kept_rows, 4].astype(np.float32), grad=grad_phys, affine=affine, vol=vol)


def sift3d_descriptors(
    state: dict,
    n_cells: int = 4,
    n_bins: int = 8,
    samples_per_cell: int = 3,
    rotation_invariant: bool = False,
    frame_rotation: Optional[np.ndarray] = None,
    return_frames: bool = False,
):
    """Stage 2 of ``detect_sift3d``: descriptors for ``state['pts']`` (see ``_build_descriptor``).

    Returns the ``[N, n_cells^3 * n_bins]`` float32 descriptors, plus the ``[N, 3, 3]``
    frames if ``return_frames``; empty arrays when the state has no keypoints.
    """
    pts = state["pts"]
    if pts.shape[0] == 0 or state["grad"] is None:
        d = np.zeros((0, n_cells ** 3 * n_bins), dtype=np.float32)
        return (d, np.zeros((0, 3, 3), dtype=np.float32)) if return_frames else d
    descs, frames = _build_descriptor(state["grad"], state["affine"], pts[:, :3], pts[:, 3],
                                      n_cells=n_cells, n_bins=n_bins, samples_per_cell=samples_per_cell,
                                      rotation_invariant=rotation_invariant, frame_rotation=frame_rotation,
                                      return_frames=True)
    return (descs, frames) if return_frames else descs
