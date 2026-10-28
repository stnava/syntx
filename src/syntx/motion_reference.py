"""syntx.motion_reference -- building a sharp, modality-agnostic reference image for
motion correction without (a) a naive whole-series temporal mean, which is itself
motion-blurred whenever real inter-frame motion exists, or (b) an expensive iterative
template-build over the whole series.

Two real, confirmed findings motivate this module (2026-10-04 session):

1. A naive ``reference="mean"`` across a CONTRAST-HETEROGENEOUS series (e.g. DWI's b0 +
   multiple diffusion-weighted shells) biases the reference toward whichever contrast is
   the numerical majority, producing the WORST registration fits for the minority class
   (confirmed on real SOCOM DWI data: b0 volumes, only 7/99 frames, showed the highest
   apparent FD of any group when registered against the full 99-volume blended mean --
   nearly 3x higher than when b0 volumes are registered among themselves instead). An
   initial fix (register every frame, both groups, to one anchor-group-only reference)
   was ALSO wrong -- it still forces every non-anchor frame through a per-frame
   CROSS-contrast registration, which the fast narrow-seed backend handles worse, not
   better (b0 FD rose to 7.28mm under that design, worse than the original 4.66mm).
   ``motion_correct_grouped`` instead builds one mean per group, aligns the two means to
   each other exactly ONCE (the only cross-contrast registration anywhere in the
   pipeline, so it can afford to be careful/robust), and keeps every per-frame
   registration same-contrast (where the fast backend is actually validated to work).

2. Even within a single-contrast series, the plain temporal mean of ALL frames is sharper
   only when real motion is near-zero; with real motion present, averaging many
   differently-positioned frames smears anatomy, which can itself inflate apparent
   registration error. ``build_low_motion_reference`` addresses this by building the
   reference from only a small, cheaply-selected low-motion subset.
"""

from __future__ import annotations

import time as _time
from typing import List, Optional, Sequence

import ants
import warnings

import numpy as np

from .motion_batched import batched_rigid_register_pass, estimate_coarse_fd


def _mean_image(imgs: Sequence[ants.ANTsImage]) -> ants.ANTsImage:
    """Plain unweighted voxelwise mean of a list of same-grid images."""
    arr = np.mean(np.stack([im.numpy() for im in imgs], axis=0), axis=0)
    return imgs[0].new_image_like(arr.astype(np.float32))


def _invert_affine_file(path: str, out_path: str) -> str:
    """Write the inverse of an ANTs AffineTransform file (same centre), return ``out_path``."""
    tx = ants.read_transform(path)
    inv = tx.invert()
    ants.write_transform(inv, out_path)
    return out_path


def _register_group_backward(img4d_g, mean_img, kw, backend, verbose, ref_intensity, smoothing_mm, interp):
    """Register the group MEAN onto every frame (each frame is the fixed image, the mean the moving one) and return the
    result in the forward convention (``fwdtransforms`` map mean-space points into the frame, ``motion_corrected`` holds
    the frames resampled into the mean's space).

    Why backward: with the frame as the fixed image the metric's samples and Parzen marginal come from the frame's own
    contrast and the smooth mean is the image being interpolated. On the sub-BLAST079 still phantom (real contrast,
    measured noise) a b3000 volume registered forward to its shell mean is off by 0.253 mm RMS; backward 0.148 mm;
    with 2 mm smoothing of both images for the metric 0.144 (b1500 0.097), and with the tied even-l=2 bias term removed
    0.113 / 0.089 mm. The twin floor (same contrast) is 0.03-0.08.

    ``smoothing_mm`` smooths the images used by the metric only; the data are resampled from the originals.
    """
    from .motion import _extract_rigid_parameters, motion_correction
    from .transforms import apply_transforms as _apply_transforms

    n = img4d_g.shape[-1]
    frames = [ants.slice_image(img4d_g, axis=3, idx=i) for i in range(n)]
    refs = [ants.smooth_image(f, smoothing_mm) if smoothing_mm else f for f in frames]
    mean_m = ants.smooth_image(mean_img, smoothing_mm) if smoothing_mm else mean_img
    mov4d = ants.list_to_ndimage(img4d_g, [mean_m] * n)
    res_b = motion_correction(mov4d, reference=refs, backend=backend, verbose=verbose,
                              dvars_reference_intensity=ref_intensity, **kw)
    # backward chain per frame: [eddy_b, rigid_b] maps frame points -> mean points (rigid applied first). The forward
    # map is R_b^-1 o E_b^-1 = E' o R_b^-1 with E' = R_b^-1 E_b^-1 R_b, kept in the pipeline's [eddy, rigid] order.
    fwd, params, homogs, jac_f = [], [], [], []
    pe_b = res_b.get("pe_affine")
    for li in range(n):
        chain_b = list(res_b.fwdtransforms[li])
        rigid_b = chain_b[-1]
        rigid_f = _invert_affine_file(rigid_b, rigid_b.replace(".mat", "_fwd.mat"))
        chain_f = [rigid_f]
        if len(chain_b) == 2:  # an eddy affine precedes the rigid
            tb = ants.read_transform(rigid_b)
            eb = ants.read_transform(chain_b[0])
            Rb = _extract_rigid_parameters(tb, 3)[2]
            Mb = np.array(eb.parameters[:9], dtype=np.float64).reshape(3, 3)
            tbv = np.array(eb.parameters[9:12], dtype=np.float64)
            c = np.array(eb.fixed_parameters, dtype=np.float64)
            Hb_e = np.eye(4); Hb_e[:3, :3] = Mb; Hb_e[:3, 3] = tbv + c - Mb @ c  # y = M (x - c) + c + t
            Hp = np.linalg.inv(Rb) @ np.linalg.inv(Hb_e) @ Rb  # E' (affine)
            te = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3)
            te.set_parameters(np.concatenate([Hp[:3, :3].flatten(), Hp[:3, 3] - c + Hp[:3, :3] @ c]))
            te.set_fixed_parameters(c)
            e_path = chain_b[0].replace(".mat", "_fwd.mat")
            ants.write_transform(te, e_path)
            chain_f = [e_path, rigid_f]
        fwd.append(chain_f)
        tr_, rot_, T_h = _extract_rigid_parameters(ants.read_transform(rigid_f), 3)
        params.append(np.concatenate([tr_, rot_]))
        homogs.append(T_h)
        jac_f.append(1.0 / float(pe_b["jacobian"][li]) if pe_b is not None else 1.0)
    corrected = []
    for li in range(n):
        w = _apply_transforms(fixed=mean_img, moving=frames[li], transformlist=fwd[li], interpolator=interp)
        corrected.append(w * jac_f[li] if jac_f[li] != 1.0 else w)
    out = dict(res_b)
    out["motion_corrected"] = ants.list_to_ndimage(img4d_g, corrected)
    out["fwdtransforms"] = fwd
    out["invtransforms"] = [list(c) for c in fwd]
    mp = res_b.motion_parameters
    out["motion_parameters"] = type(mp)(np.array(params), columns=list(getattr(mp, "_columns", []) or []) or None,
                                        rad_to_deg_cols=list(getattr(mp, "_rad_cols", []) or []) or None, spatial_dim=3)
    if pe_b is not None:
        pe_f = dict(pe_b)
        pe_f["a"] = -np.asarray(pe_b["a"])  # the forward eddy coefficients are the negated backward ones (to first order)
        pe_f["jacobian"] = np.asarray(jac_f)
        pe_f["coefficients"] = [None if c is None else -np.asarray(c) for c in pe_b["coefficients"]] if isinstance(pe_b.get("coefficients"), list) else pe_b.get("coefficients")
        out["pe_affine"] = pe_f
    out["registration_direction"] = "backward"
    from .motion import MotionCorrectionResult

    return MotionCorrectionResult(out)


def build_low_motion_reference(
    images: List[ants.ANTsImage],
    n_low_motion: int = 10,
    coarse_resolution_cap_mm: Optional[float] = 8.0,
    fd_radius: float = 50.0,
    device: str = "auto",
    num_bins: int = 18,
    verbose: bool = False,
) -> ants.ANTsImage:
    """Build a sharp reference from the ``n_low_motion`` frames with the smallest cheap,
    registration-derived FD relative to the series' own crude (unweighted) mean --
    avoids both a motion-blurred whole-series mean and an expensive full-series
    iterative template build.

    Steps: (1) one cheap coarse registration pass (``estimate_coarse_fd`` -- the same
    primitive behind ``batched_rigid_register_pass_adaptive``) ranks every frame; (2) the
    ``n_low_motion`` lowest-FD frames are registered to EACH OTHER (small N, so this
    narrow+fast pass is cheap even though it's a "real" registration, not the coarse
    ranking pass); (3) the corrected low-motion subset is averaged into the final
    reference.

    If the series has ``n_low_motion`` frames or fewer, every frame is used (no subset
    selection needed) and this reduces to "register everything to the crude mean, then
    average the corrected result" -- a single-pass sharpened mean.

    Parameters
    ----------
    images : list of ants.ANTsImage
        Candidate frames (same grid). For a grouped series (e.g. DWI b0 volumes), pass
        only that group's own frames -- this function has no modality-specific logic.
    n_low_motion : int, default 10
        Number of lowest-FD frames to build the reference from.
    coarse_resolution_cap_mm, fd_radius, device, num_bins
        See ``estimate_coarse_fd``.
    verbose : bool, default False

    Returns
    -------
    ants.ANTsImage
        The sharpened reference, on the same grid as ``images[0]``.
    """
    n = len(images)
    if n == 0:
        raise ValueError("build_low_motion_reference requires at least 1 image.")
    if n == 1:
        return images[0].clone()

    t0 = _time.time()
    crude_ref = _mean_image(images)

    if n <= n_low_motion:
        selected = images
        if verbose:
            print(f"[syntx.motion_reference] {n} frames <= n_low_motion={n_low_motion} -- "
                  f"using all of them (no subset selection).")
    else:
        coarse_fd, _, _ = estimate_coarse_fd(
            crude_ref, images, resolution_cap_mm=coarse_resolution_cap_mm, fd_radius=fd_radius,
            device=device, num_bins=num_bins,
        )
        lowest_idx = np.argsort(coarse_fd)[:n_low_motion]
        selected = [images[i] for i in lowest_idx]
        if verbose:
            print(f"[syntx.motion_reference] selected {n_low_motion}/{n} lowest-FD frames "
                  f"(coarse FD range of selection: {coarse_fd[lowest_idx].min():.3f}-"
                  f"{coarse_fd[lowest_idx].max():.3f}mm, full-series range: "
                  f"{coarse_fd.min():.3f}-{coarse_fd.max():.3f}mm).")

    selected_mean = _mean_image(selected)
    fwd, _inv, _elapsed = batched_rigid_register_pass(
        selected_mean, selected, device=device, num_bins=num_bins, verbose=False,
        seed_mode="narrow", schedule_mode="fast",
    )
    from .transforms import apply_transforms as _apply_transforms
    corrected = [
        _apply_transforms(fixed=selected_mean, moving=im, transformlist=f, interpolator="linear")
        for im, f in zip(selected, fwd)
    ]
    final_ref = _mean_image(corrected)

    if verbose:
        print(f"[syntx.motion_reference] build_low_motion_reference: {_time.time() - t0:.2f}s "
              f"total ({len(selected)} frames registered+averaged).")
    return final_ref


def _make_4d_template_from_3d(ref3d: ants.ANTsImage, n_frames: int, tr: float = 1.0) -> ants.ANTsImage:
    """Build a real-geometry 4D template for ``ants.list_to_ndimage``, instead of
    ``ants.from_numpy(np.zeros(...))`` (default header: origin (0,0,0), identity
    direction, unit spacing). ``list_to_ndimage`` copies ITS template's own header onto
    the assembled output, so a zeros-array template silently discards the real subject
    geometry from every frame stacked into it -- same convention as
    ``antsxslowflow/motion/api.py`` L326-338 and ``ANTsPyMM/antspymm/mm.py`` L2143-2154:
    3x3 spatial block from the reference 3D image, a fresh 4x4 identity otherwise (never
    copied wholesale from a stale 4D image).
    """
    spc3 = list(ants.get_spacing(ref3d))
    org3 = list(ants.get_origin(ref3d))
    dir3 = np.asarray(ants.get_direction(ref3d), dtype=float)
    mydir4d = np.eye(4, dtype=float)
    mydir4d[:3, :3] = dir3[:3, :3]
    shape4d = list(ref3d.shape) + [n_frames]
    return ants.make_image(shape4d, 0, spacing=spc3 + [tr], origin=org3 + [0.0], direction=mydir4d)


def shell_groups(bvals: np.ndarray, b0_threshold: float = 50.0, shell_tol: float = 150.0) -> np.ndarray:
    """Group label per volume for ``motion_correct_grouped``: 0 for b0 (b <= ``b0_threshold``), then 1..k for the diffusion
    shells in ascending b (b-values within ``shell_tol`` of the previous sorted b-value share a shell)."""
    b = np.asarray(bvals, dtype=float)
    labels = np.zeros(len(b), dtype=int)
    dw = np.where(b > b0_threshold)[0]
    order = dw[np.argsort(b[dw])]
    g = 1
    for k, i in enumerate(order):
        if k > 0 and b[i] - b[order[k - 1]] > shell_tol:
            g += 1
        labels[i] = g
    return labels


def build_bilateral_window_references(
    frames: Sequence[ants.ANTsImage] | ants.ANTsImage,
    encoding_covariates: Optional[np.ndarray] = None,
    window_size: Optional[int] = None,
    sigma_enc: float = 0.5,
    exclude_self: bool = False,
) -> List[ants.ANTsImage]:
    """Build per-frame contrast-matched references with a kernel over encoding state (and optionally time):

    ``I_ref,t(x) = sum_tau K_time(t, tau) K_enc(e_t, e_tau) I_tau(x) / sum_tau K_time(t, tau) K_enc(e_t, e_tau)``

    ``K_enc(e_t, e_tau) = exp(-||e_t - e_tau||^2 / (2 K sigma_enc^2))`` weights frames of matching encoding state; for
    discrete states (slow-flow ``[b0, +X, +Y, +Z]``, ASL tag/control) the reference of frame t is the mean of all frames
    in t's state. ``window_size=None`` (default, what motion correction uses) sets ``K_time = 1``: every reference is a
    mean over the WHOLE series, so all references share one head position and the registered poses are in one common
    space. A finite ``window_size`` (``K_time = max(0, 1 - |t - tau| / w)``) gives local references; poses against
    them are relative to a moving average of the series, i.e. slow drift is NOT measured -- use it only to build
    templates, never as a motion-correction target.

    ``exclude_self``: leave-one-out, frame t gets no weight in its own reference. Off by default: for discrete states
    the inclusive state mean is ONE reference shared by every frame of the state, which is what puts their poses in a
    common space; leave-one-out gives every frame a slightly different reference (simulation, 3 seeds: slow-flow preset
    head error 0.031 -> 0.115 mm, ASL state-matched 0.017 -> 0.060 mm). With continuous encodings (one frame per DWI
    direction) the frame's own weight dominates the inclusive kernel; see ``encoding_reference_predictor``.
    """
    from .motion_batched import build_encoding_covariates

    if isinstance(frames, ants.ANTsImage):
        if frames.dimension != 4:
            raise ValueError(f"Expected 4D ANTsImage or sequence of 3D images, got {frames.dimension}D")
        frame_list = [ants.slice_image(frames, axis=3, idx=t) for t in range(frames.shape[3])]
    else:
        frame_list = list(frames)
    T = len(frame_list)
    if T == 0:
        return []
    if T == 1:
        return [frame_list[0].clone()]

    stack = np.stack([f.numpy().astype(np.float32) for f in frame_list], axis=0)  # (T, X, Y, Z)
    if window_size is None:
        k_time = np.ones((T, T), dtype=np.float64)
    else:
        w = max(1.0, float(window_size))
        t_idx = np.arange(T, dtype=np.float64)
        dt = np.abs(t_idx[:, None] - t_idx[None, :])
        k_time = np.maximum(0.0, 1.0 - dt / w) + 1e-3 * np.exp(-0.5 * (dt / w) ** 2)

    if encoding_covariates is not None:
        enc_basis = build_encoding_covariates(
            encoding_covariates=encoding_covariates,
            num_frames=T,
        )
        E = enc_basis["E"]
        K_eff = int(enc_basis["K_eff"])
        if K_eff > 0:
            diff_e = E[:, None, :] - E[None, :, :]
            dist2 = np.sum(diff_e * diff_e, axis=-1) / float(K_eff)
            sig2 = max(float(sigma_enc) ** 2, 1e-4)
            k_enc = np.exp(-0.5 * dist2 / sig2)
        else:
            k_enc = np.ones((T, T), dtype=np.float64)
    else:
        k_enc = np.ones((T, T), dtype=np.float64)

    W = k_time * k_enc
    if exclude_self:
        np.fill_diagonal(W, 0.0)
    W = W / np.maximum(np.sum(W, axis=1, keepdims=True), 1e-12)
    W = W.astype(np.float32)
    ref_dtype = frame_list[0].dtype
    # one reference at a time: peak memory is the input stack plus the returned images, not a second 4-D stack
    return [ants.new_image_like(frame_list[0], np.tensordot(W[t], stack, axes=(0, 0)).astype(ref_dtype)) for t in range(T)]


def level_adjacency(levels: np.ndarray) -> np.ndarray:
    """Symmetric level adjacency matrix A where A[u, v] counts adjacent transitions between levels u and v."""
    lev = np.asarray(levels, dtype=int).reshape(-1)
    if lev.size < 2:
        n = int(np.max(lev) + 1) if lev.size == 1 else 0
        return np.zeros((n, n), dtype=int)
    n = int(np.max(lev) + 1)
    adj = np.zeros((n, n), dtype=int)
    u_seq = lev[:-1]
    v_seq = lev[1:]
    diff = u_seq != v_seq
    u_diff = u_seq[diff]
    v_diff = v_seq[diff]
    for u, v in zip(u_diff, v_diff):
        adj[u, v] += 1
        adj[v, u] += 1
    return adj


def offset_identifiable(levels: np.ndarray, min_mixed: int = 4) -> tuple[dict, list]:
    """Analyze level adjacency graph connected components and determine offset identifiability.

    A connected component C requires at least 8 total frames.
    Within each component C with root r(C) = min(C), a non-root level g in C has its offset
    identifiable relative to r(C) iff it has at least 4 active surround rows on C's sub-timeline.

    Returns:
        info: dict[int, dict] mapping each level k to:
            - 'identifiable': bool
            - 'root': int (r(C))
            - 'component': list[int]
            - 'n': int (frame count of level k)
            - 'active': int (active surround rows on C's sub-timeline)
            - 'mixed': int (mixed transitions to other levels)
            - 'interleaved': bool
        components: list[list[int]] connected components sorted by root
    """
    lev = np.asarray(levels, dtype=int).reshape(-1)
    T = lev.size
    unique_levels = sorted(set(lev.tolist()))
    if not unique_levels:
        return {}, []

    adj = level_adjacency(lev)

    # Build adjacency graph with threshold min_mixed
    adj_graph = {k: set() for k in unique_levels}
    for u in unique_levels:
        for v in unique_levels:
            if u < v and adj[u, v] >= min_mixed:
                adj_graph[u].add(v)
                adj_graph[v].add(u)

    # Find connected components via BFS
    visited = set()
    components = []
    for k in unique_levels:
        if k not in visited:
            comp = []
            queue = [k]
            visited.add(k)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj_graph[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(sorted(comp))
    components.sort(key=lambda c: c[0])

    info = {}
    for comp in components:
        r = comp[0]  # root is min(comp)
        comp_mask = np.isin(lev, comp)
        n_comp_frames = int(np.sum(comp_mask))
        comp_levels = lev[comp_mask]

        for k in comp:
            k_mask = (lev == k)
            n_k = int(np.sum(k_mask))
            idx_k = np.flatnonzero(k_mask)
            mixed_k = int(sum(1 for t in idx_k if (t > 0 and lev[t - 1] != k) or (t < T - 1 and lev[t + 1] != k)))

            # Active surround rows on comp's sub-timeline
            if len(comp) > 1 and n_comp_frames >= 8:
                ind_k = (comp_levels == k).astype(np.float64)
                if ind_k.size >= 3:
                    surr_k = ind_k[1:-1] - 0.5 * (ind_k[:-2] + ind_k[2:])
                    active_k = int(np.sum(np.abs(surr_k) > 0.1))
                else:
                    active_k = 0
            else:
                active_k = 0

            is_identifiable = bool(k != r and len(comp) > 1 and n_comp_frames >= 8 and active_k >= 4)
            is_interleaved = bool(n_k >= 2 and mixed_k >= max(2, (n_k + 1) // 2) and (len(comp) > 1 and n_comp_frames >= 8))

            info[k] = {
                "identifiable": is_identifiable,
                "root": r,
                "component": comp,
                "n": n_k,
                "active": active_k,
                "mixed": mixed_k,
                "interleaved": is_interleaved,
            }

    return info, components


def _group_offsets(H, groups, idx, com):
    """Per-group constant pose offset of composed rigid poses ``H`` (n, 4, 4) relative to each component root,
    from the surround Tukey projection on the group indicators evaluated on each connected component's sub-timeline.
    Returns ``(Y, offsets)`` with ``Y`` the (n, 6) poses ``[d_com, rotvec]`` and ``offsets`` {group: (6,)} for
    the identifiable non-root groups, or ``None`` when no group is identifiable."""
    from scipy.spatial.transform import Rotation as _Rot

    from .motion_batched import _project_encoding_surround_tukey_np, build_encoding_covariates

    groups = np.asarray(groups)
    n_frames, n_groups = len(H), len(idx)
    if n_frames < 8 or n_groups < 2:
        return None

    info, components = offset_identifiable(groups)
    est_groups = [g for g, v in info.items() if v["identifiable"]]
    if not est_groups:
        return None

    H = np.asarray(H)
    Y = np.c_[np.einsum("nij,j->ni", H[:, :3, :3], com) + H[:, :3, 3] - com, _Rot.from_matrix(H[:, :3, :3]).as_rotvec()]

    per_group = {}
    for comp in components:
        r = comp[0]
        est_C = [g for g in comp if g != r and info[g]["identifiable"]]
        if not est_C:
            continue
        keep = np.flatnonzero(np.isin(groups, comp))
        if keep.size < 8:
            continue
        labels = np.searchsorted(np.array([r] + est_C), groups[keep])
        basis = build_encoding_covariates(state_labels=labels, num_frames=keep.size, window_size=keep.size)
        _, dY, _ = _project_encoding_surround_tukey_np(Y[keep], basis["E"], basis["psi"])
        off = dY - np.median(dY[labels == 0], axis=0)  # relative to the component root (which stays put)
        for j, g in enumerate(est_C):
            per_group[g] = np.median(off[labels == j + 1], axis=0)

    if not per_group:
        return None

    return Y, per_group


def _cross_link(fixed, moving, link, *, verbose, device, num_bins, convergence_tol_vox, max_extend_factor):
    """Link of an adjacent pair of group means: ``"registration"`` (robust rigid registration of the corrected means) or
    ``"identity"`` (groups interleaved in time: their raw means share the series' mean head position, so a cross-contrast
    registration would only add its own contrast bias; the remaining offset is the cross-group debias' job)."""
    import os

    if link == "registration":
        from .robust_affine import robust_affine

        return robust_affine(fixed=fixed, moving=moving, dof="rigid", mode="auto", verbose=verbose, device=device,
                             num_bins=num_bins, convergence_tol_vox=convergence_tol_vox, max_extend_factor=max_extend_factor)
    if link != "identity":
        raise ValueError(f"cross_link entries must be 'registration' or 'identity', got {link!r}")
    import tempfile

    tx = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3)
    path = os.path.join(tempfile.mkdtemp(prefix="syntx_identity_link_"), "identity.mat")
    ants.write_transform(tx, path)
    return {"fwdtransforms": [path], "convergence": [], "link": "identity"}


def _pose_matrix(y, com):
    """4x4 rigid matrix of the pose ``y = [d_com, rotvec]`` about ``com``."""
    from scipy.spatial.transform import Rotation as _Rot

    R = _Rot.from_rotvec(y[3:]).as_matrix()
    h = np.eye(4)
    h[:3, :3] = R
    h[:3, 3] = com + y[:3] - R @ com
    return h


def _debias_cross_group_offsets(chains, groups, idx, results, cross, com, outprefix):
    """Per-frame rigid corrections removing the contrast-driven offset of each non-anchor group (see
    ``motion_correct_grouped(cross_group_debias=True)``). Returns ``None`` when no non-anchor group is interleaved
    enough to separate an offset from head motion (fewer than 4 active surround rows); a group that is not keeps its
    linked pose and is listed in ``not_identifiable_groups``."""
    import os
    import tempfile

    from .motion import _extract_rigid_parameters

    n_frames = len(chains)
    n_cross = [len(c["fwdtransforms"]) for c in cross]
    H = []
    for i in range(n_frames):
        g = int(groups[i])
        li = int(np.where(idx[g] == i)[0][0])
        n_own = len(results[g].fwdtransforms[li])
        rigid_chain = chains[i][:-n_own] + [chains[i][-1]]   # crosses + the frame's own rigid (no eddy)
        h = np.eye(4)
        for f in rigid_chain:
            h = _extract_rigid_parameters(ants.read_transform(f), 3)[2] @ h
        H.append(h)
    H = np.asarray(H)
    est = _group_offsets(H, groups, idx, com)
    if est is None:
        return None
    Y, per_group = est
    work = os.path.dirname(f"{outprefix}x") if outprefix else tempfile.mkdtemp(prefix="syntx_xgroup_")
    os.makedirs(work, exist_ok=True)
    paths = {}
    for i in range(n_frames):
        g = int(groups[i])
        if g not in per_group:          # the anchor, and groups whose offset is not identifiable (left as linked)
            continue
        Hc = _pose_matrix(Y[i] - per_group[g], com)
        corr = np.linalg.inv(H[i]) @ Hc                  # H_new = H @ corr: corr acts on the anchor-space point first
        tx = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3)
        tx.set_parameters(np.concatenate([corr[:3, :3].ravel(), corr[:3, 3]]))
        tx.set_fixed_parameters(np.zeros(3))
        path = os.path.join(work, f"{os.path.basename(outprefix) if outprefix else ''}xgroup_corr_{i:04d}.mat")
        ants.write_transform(tx, path)
        paths[i] = path
    return {
        "method": "surround_tukey_projection_on_group_indicators",
        "offset_mm_rad_by_group": {int(g): v.tolist() for g, v in per_group.items()},
        "offset_mm_by_group": {int(g): float(np.linalg.norm(v[:3])) for g, v in per_group.items()},
        "offset_deg_by_group": {int(g): float(np.degrees(np.linalg.norm(v[3:]))) for g, v in per_group.items()},
        "not_identifiable_groups": [g for g in range(1, len(idx)) if g not in per_group],
        "corrected_frames": sorted(paths),
        "correction_paths": paths,
    }


def encoding_reference_predictor(encoding_covariates: np.ndarray, sigma_enc: float = 0.5, exclude_self: bool = True):
    """``reference_predictor`` for ``motion_correct_grouped``: each frame's reference is the kernel-weighted mean of the
    group's OTHER corrected frames with a similar encoding (``build_bilateral_window_references``, whole series, no time
    window), i.e. a contrast-matched reference for continuous encodings. DWI: pass the even (l = 2) gradient terms --
    the image contrast is even in g. A group with no encoding variation (b0) keeps its mean (``None``).

    Measured on the simulated DWI series (b1000, 20 directions, 3 seeds, ``benchmarks/motion_encoding_bias_simulation.py``):
    leave-one-out gives no reliable gain over the shell mean (rotation error 0.56 -> 0.51 deg without eddy, 0.63 ->
    0.58 with). Including the frame (``exclude_self=False``) lowered it to 0.40 deg, but with 20 distinct directions the
    frame's own weight dominates that kernel (the frame is largely registered to its own previous resampling) and some
    frames stopped converging on antsxdwi's convergence test, so it is not used by default anywhere."""
    cov = np.asarray(encoding_covariates, dtype=np.float64)
    cov = cov[:, None] if cov.ndim == 1 else cov

    def predictor(frames, group_index=None, frame_indices=None):
        c = cov[np.asarray(frame_indices, dtype=int)]
        if not np.any(np.std(c, axis=0) > 1e-8):
            return None
        return build_bilateral_window_references(list(frames), encoding_covariates=c, sigma_enc=sigma_enc, exclude_self=exclude_self)

    return predictor


class GroupedMotionCorrectionResult(dict):
    """Dict subclass (attribute access) returned by ``motion_correct_grouped``.

    Keys: ``motion_corrected`` (4D, all frames in group 0's mean space), ``fd`` (series-level Jenkinson FD in acquisition
    order, every frame's total transform in group 0's space), ``fwdtransforms`` (per frame, input order: the full chain
    ``[cross_1, ..., cross_g, own]`` ready for ``ants.apply_transforms``), ``group_results`` (per-group
    ``MotionCorrectionResult``), ``group_indices`` (frame indices of each group), ``group_means`` (the corrected group means
    the cross registrations were computed on), ``cross_registrations`` (``robust_affine`` results; entry ``g - 1`` registers
    group ``g``'s mean onto group ``g - 1``'s), ``preset``.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'GroupedMotionCorrectionResult' has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value


def motion_correct_grouped(
    image: ants.ANTsImage,
    groups: np.ndarray,
    n_low_motion: int = 10,
    backend: str = "pytorch_batched_adaptive",
    verbose: bool = False,
    preset: Optional[str] = None,
    cross_num_bins: int = 32,
    nuisance_covariates: Optional[np.ndarray] = None,
    nuisance_axis: Optional[np.ndarray] = None,
    pe_affine_covariates: Optional[np.ndarray] = None,
    pe_axis: Optional[np.ndarray] = None,
    reference_predictor=None,
    reference_iterations: int = 1,
    sample_mask_fn=None,
    bias_covariates: Optional[np.ndarray] = None,
    registration_direction: str = "forward",
    metric_smoothing_mm: float = 0.0,
    encoding_covariates: Optional[np.ndarray] = None,
    window_size: Optional[int] = None,
    cross_group_debias: bool = False,
    debias_pe_pseudomotion: bool = False,
    slice_axis=None,
    sidecar=None,
    cross_link="registration",
    **kwargs,
) -> GroupedMotionCorrectionResult:
    """Motion-correct a contrast-heterogeneous series (e.g. DWI: b0 and one group per b-shell) with same-contrast per-frame
    registrations and one careful registration per pair of adjacent contrasts.

    (1) Each group gets its own sharp mean; (2) every frame is registered to ITS OWN group's mean (same contrast: no
    contrast-driven bias); (3) the group means are recomputed from all corrected frames and registered to each other in
    a chain, group ``g`` onto group ``g - 1``, so every registration is between the two most similar contrasts (still
    travelling-subject DWI, true bias 0: b3000 mean onto b0 directly 0.55 mm, through the b1000 -> b2000 -> b3000 chain 0.14 mm);
    (4) every frame of a later group is resampled ONCE from the original through the composed chain into group 0's space.
    This is the ``[b0_mean, dwi_mean, ...]`` multi-channel representation: all channels in one space.

    Parameters
    ----------
    image : ants.ANTsImage
        4D time series (time is the last axis).
    groups : np.ndarray, shape (n_frames,)
        Group label per frame, 0 = anchor (e.g. b0), then 1..k-1 in order of increasing contrast distance from the anchor
        (``shell_groups`` for DWI). A boolean array (True = anchor) is accepted for two groups.
    n_low_motion : int, default 10
        Forwarded to ``build_low_motion_reference`` for each group's own mean build.
    backend : str, default 'pytorch_batched_adaptive'
        Forwarded to ``syntx.motion_correction`` for each group; overridden by ``preset``.
    preset : str, optional
        ``"dwi"`` (``syntx.motion_presets``): full-precision batched backend at ``resolution_cap_mm``, b-spline final
        resampling, ``cross_num_bins`` histogram bins in the cross registrations (on a still scan the b2000 -> b1000 bias fell
        from 0.16 mm at 32 bins to 0.04 mm at 128).
    cross_num_bins : int, default 32
        MI histogram bins of the cross registrations when no preset is given.
    nuisance_covariates : np.ndarray, shape (n_frames, k), optional
        Deprecated; use ``pe_affine_covariates``. Per-frame covariates of a translation that is NOT head motion but is
        absorbed by the rigid fit, with ``nuisance_axis`` the physical unit vector it acts along.
    nuisance_axis : Optional[np.ndarray], optional
        Deprecated; use ``pe_axis``.
    pe_affine_covariates : np.ndarray, shape (n_frames, k), optional
        With ``pe_axis``: the eddy-current term estimated INSIDE each group's registration (``pytorch_batched`` backend,
        see ``motion_batched._batched_rigid_register_pass_core``): per frame a translation, two shears and a scale along
        the phase-encode axis, tied across the group's frames as a linear function of the covariates (DWI: the gradient
        vectors; b0 frames have zero covariates and get no term). The eddy term is applied to the data (single resampling
        through ``[cross..., eddy, rigid]``, Jacobian-modulated); ``fd`` is the Jenkinson FD of the rigid part only;
        ``fd_raw`` includes the eddy term; ``pe_affine`` holds per-group coefficient matrices and per-frame ``a``,
        ``jacobian``. This supersedes the post-hoc ``nuisance_covariates`` regression (which only models the translation).
    reference_predictor : callable, optional
        ``predictor(corrected_frames, group_index=g, frame_indices=idx) -> list of per-frame reference images on the
        group's grid, or None``. A group's frames are first registered to the group mean (same contrast on average, but
        each frame's own contrast can differ from it in a way that biases the pose -- a diffusion volume's attenuation
        pattern depends on its gradient direction); when a predictor is given, it is called with the corrected frames
        and the group is re-registered, frame by frame, to its predicted reference (``motion_correction(reference=
        <list>)``, same eddy term), ``reference_iterations`` times, re-predicting from the newly corrected frames each
        time. ``None`` from the predictor keeps the mean (the anchor/b0 group has no direction dependence). The
        predictor carries the modality knowledge (antsxdwi: a spherical-harmonic prediction at each gradient).
        ``reference_mode`` per group ('mean' | 'predicted') and the iterations are reported.
    sample_mask_fn : callable, optional
        ``fn(anchor_mean) -> 3-D mask image in the anchor group's space`` (DWI: gray matter + CSF from the T1, projected
        onto the corrected b0 mean). After the first pass and the cross registrations, every group g >= 1 is registered
        AGAIN with the mask mapped into its own mean space (nearest neighbour through the inverse cross chain) as the
        solver's ``sample_mask``: the metric then sees only voxels whose signal does not depend on the frame's
        contrast. The mask must be right relative to the reference only; the moving frames' motion is the unknown and
        the sampled points are warped by it. Cross registrations are recomputed from the second pass.
    bias_covariates : np.ndarray, shape (n_frames, k), optional
        Per-frame covariates of a deterministic registration bias (DWI: even spherical-harmonic terms, l = 2, of the
        gradient direction); forwarded per group to the solver, which regresses the 6 rigid parameters on them and
        removes the explained part from the written poses (see ``motion_batched``). Zero rows (b0) are skipped.
    registration_direction : {'forward', 'backward'}
        'forward' registers every frame (moving) to its group mean (fixed). 'backward' registers the mean onto each frame
        (the frame is the fixed image) and returns the result in the forward convention -- see
        ``_register_group_backward`` for the measurement that motivates it (0.253 -> 0.148 mm RMS pose error per b3000
        volume on the still phantom). Applies to every group; the b0 group is unaffected in practice (same contrast).
    metric_smoothing_mm : float
        Gaussian smoothing (mm) of both images for the metric only, applied with 'backward' (2 mm: 0.148 -> 0.144 mm,
        and no loss under 1 mm / 1 deg motion; 3 mm hurt capture). The data are resampled from the originals.
    encoding_covariates : np.ndarray, shape (n_frames,) or (n_frames, k), optional
        Per-frame encoding covariates forwarded per group to ``motion_correction(encoding_covariates=...)`` (in-optimizer
        overlapping-window Tukey projection of encoding-locked registration bias out of the written poses). Use terms
        that describe the CONTRAST (DWI: even, l = 2, gradient terms), not the eddy displacement (odd in g, modelled by
        ``pe_affine_covariates`` and applied).
    debias_pe_pseudomotion : bool, default False
        Opt-in (needs ``pe_axis`` and >= 12 frames; otherwise a warning and ``{"method": "skipped"}``): shrink the
        high-frequency residual of the REPORTED rigid trajectory (``fd``, ``translations_mm``, ``rotations_deg``) along
        phase encode (and, gated on a negative step autocorrelation, the other degrees of freedom) as non-linear eddy /
        respiratory B0 pseudomotion. The transforms and the corrected series are unchanged; ``fd_before_pe_debias`` keeps
        the previous FD. Independent real jitter between volumes also alternates and is shrunk (audit: nodding jitter
        FD 0.94 -> 0.15 mm), so FD-based censoring after it under-counts real motion.
    cross_link : str or sequence of str, default 'registration'
        How each adjacent pair of group means is linked (one entry per pair, or one for all): ``'registration'`` (robust rigid
        registration of the corrected means) or ``'identity'`` (groups interleaved in time share the series' mean head
        position; pair it with ``cross_group_debias`` for the remaining contrast offset).
    slice_axis : int or str, optional
        Acquired slice axis for the ``debias_pe_pseudomotion`` frame (voxel index or BIDS ``i/j/k``); default: measured
        from the series with the group levels removed (``syntx.acquisition_axes.infer_slice_axis``); undecidable raises.
    sidecar : dict or Path or str, optional
        Optional BIDS sidecar dict or path for metadata-guided axis inference.
    cross_group_debias : bool, default False
        Opt-in: remove a contrast-driven bias of the cross-group (group mean -> group mean) registrations: when the groups
        are interleaved in time, the composed rigid trajectory is projected on the group indicators with the surround
        Tukey estimator (``motion_batched._project_encoding_surround_tukey_np``; smooth head drift and isolated jumps
        are in its nullspace / downweighted) and the per-frame offset of every non-anchor group relative to the anchor
        is removed by a correction transform at the START of its chain (applied before the single resampling).
        On the simulated DWI series (``benchmarks/motion_encoding_bias_simulation.py``, 3 seeds) the cross chain was
        already unbiased (head translation error 0.111 -> 0.115 mm with it), hence opt-in. Reported as
        ``cross_group_offset``; ``None`` when off or when the groups are not interleaved.
    **kwargs
        Forwarded to ``syntx.motion_correction`` (e.g. ``fd_radius``).
    """
    from .motion import _extract_rigid_parameters, motion_correction
    from .robust_affine import robust_affine
    from .transforms import apply_transforms as _apply_transforms

    if nuisance_covariates is not None or nuisance_axis is not None:
        warnings.warn(
            "nuisance_covariates and nuisance_axis in motion_correct_grouped are deprecated and will be removed "
            "in a future release; use pe_affine_covariates and pe_axis instead.",
            FutureWarning,
            stacklevel=2,
        )

    groups = np.asarray(groups)
    if groups.dtype == bool:
        groups = (~groups).astype(int)
    n_frames = image.shape[-1]
    if groups.shape != (n_frames,):
        raise ValueError(f"groups must have one entry per frame ({n_frames}), got shape {groups.shape}")
    labels = np.unique(groups)
    if len(labels) < 2 or not np.array_equal(labels, np.arange(len(labels))):
        raise ValueError(f"groups must use the labels 0..k-1 (k >= 2) with every label present; got {labels.tolist()}")
    n_groups = len(labels)
    cfg = None
    if preset is not None:
        from .motion_presets import get_motion_preset

        cfg = get_motion_preset(preset)
        if preset != "dwi":
            raise ValueError(f"motion_correct_grouped takes the 'dwi' preset; {preset!r} is used through motion_correction")
        backend, cross_num_bins = cfg["backend"], cfg["cross_num_bins"]
        kwargs.setdefault("resolution_cap_mm", cfg["resolution_cap_mm"])
        for key in ("seed_mode", "schedule_mode", "sample_fraction", "convergence_tol_vox", "max_extend_factor"):
            if key in cfg:
                kwargs.setdefault(key, cfg[key])
    interp = cfg["interpolator"] if cfg is not None else "linear"
    idx = [np.where(groups == g)[0] for g in range(n_groups)]

    # Each group's own 4D container ONCE, every frame and mean sliced from it: all share EXACTLY the same grid (independently
    # built ants images drift in origin/direction and are rejected by the batched passes' grid check).
    ref3d = ants.slice_image(image, axis=3, idx=0)
    tr = float(image.spacing[3]) if image.dimension == 4 else 1.0
    img4d, imgs, means = [], [], []
    for g in range(n_groups):
        c4 = ants.list_to_ndimage(_make_4d_template_from_3d(ref3d, len(idx[g]), tr=tr),
                                  [ants.slice_image(image, axis=3, idx=int(i)) for i in idx[g]])
        img4d.append(c4)
        imgs.append([ants.slice_image(c4, axis=3, idx=i) for i in range(len(idx[g]))])
        means.append(build_low_motion_reference(imgs[g], n_low_motion=n_low_motion, verbose=verbose))
    if verbose:
        print(f"[syntx.motion_reference] motion_correct_grouped: group sizes {[len(i) for i in idx]} of {n_frames} frames.")

    # DVARS of every group is normalised against the anchor group's mean intensity (comparable across groups and subjects).
    anchor = means[0].numpy()
    anchor_fg = anchor[anchor > 0]
    ref_intensity = float(np.mean(anchor_fg)) if anchor_fg.size > 0 else None
    group_ref_intensities: list[float | None] = [ref_intensity] * n_groups
    results = []
    reference_modes: list = []
    group_kw: list = []
    for g in range(n_groups):
        ref_int_g = group_ref_intensities[g]
        # group 0's corrected frames are final outputs; later groups' only feed their mean (resampled once from the originals below)
        kw = {**kwargs, "interpolator": interp if g == 0 else "linear"} if cfg is not None else dict(kwargs)
        # Every group writes its own transform files. Frame t of each group would otherwise be written to the same
        # f"{outprefix}batch_{t:04d}.mat", the last group overwriting the others; the chains below are read back from
        # these paths after all groups have run, so group g's frames would be resampled with group g+1's transforms and
        # the series FD would see one group's poses twice (found on real 3-shell DWI, sub-BLAST079, 2026-10-20).
        if kw.get("outprefix") is not None:
            kw["outprefix"] = f"{kw['outprefix']}g{g}_"
        if pe_affine_covariates is not None:
            cov_g = np.asarray(pe_affine_covariates, dtype=np.float64).reshape(n_frames, -1)[np.asarray(idx[g])]
            if np.any(cov_g != 0):
                kw["pe_affine_covariates"] = cov_g
                kw["pe_axis"] = pe_axis
        if bias_covariates is not None:
            bc_g = np.asarray(bias_covariates, dtype=np.float64).reshape(n_frames, -1)[np.asarray(idx[g])]
            if np.any(bc_g != 0):
                kw["bias_covariates"] = bc_g
        if encoding_covariates is not None:
            ec_g = np.asarray(encoding_covariates, dtype=np.float64).reshape(n_frames, -1)[np.asarray(idx[g])]
            if np.any(np.std(ec_g, axis=0) > 1e-6):
                kw["encoding_covariates"] = ec_g
        if window_size is not None and "encoding_covariates" in kw:
            kw["window_size"] = int(window_size)
        group_kw.append(kw)
        if registration_direction == "backward":
            res_g = _register_group_backward(img4d[g], means[g], kw, backend, verbose, ref_int_g, metric_smoothing_mm, interp if g == 0 else "linear")
        elif registration_direction == "forward":
            res_g = motion_correction(img4d[g], reference=means[g], backend=backend, verbose=verbose,
                                      dvars_reference_intensity=ref_int_g, **kw)
        else:
            raise ValueError(f"registration_direction must be 'forward' or 'backward', got {registration_direction!r}")
        mode_g = "mean"
        if reference_predictor is not None:
            for it in range(int(reference_iterations)):
                frames_g = [ants.slice_image(res_g.motion_corrected, axis=3, idx=i) for i in range(len(idx[g]))]
                refs = reference_predictor(frames_g, group_index=g, frame_indices=[int(i) for i in idx[g]])
                if refs is None:
                    break
                if isinstance(refs, (list, tuple)):
                    if len(refs) != len(idx[g]):
                        raise ValueError(f"reference_predictor returned {len(refs)} references for {len(idx[g])} frames of group {g}")
                    ref_input = list(refs)
                else:
                    ref_input = refs
                kw_it = dict(kw)
                if kw_it.get("outprefix") is not None:
                    kw_it["outprefix"] = f"{kw_it['outprefix']}it{it + 1}_"
                res_g = motion_correction(img4d[g], reference=ref_input, backend=backend, verbose=verbose,
                                          dvars_reference_intensity=ref_int_g, **kw_it)
                mode_g = f"predicted (iteration {it + 1})"
        reference_modes.append(mode_g)
        results.append(res_g)

    # Corrected group means, then one link per adjacent pair of contrasts (registration, or identity for interleaved groups).
    links = [cross_link] * (n_groups - 1) if isinstance(cross_link, str) else list(cross_link)
    if len(links) != n_groups - 1:
        raise ValueError(f"cross_link has {len(links)} entries for {n_groups - 1} adjacent group pairs")
    cmeans = [_mean_image([ants.slice_image(results[g].motion_corrected, axis=3, idx=i) for i in range(len(idx[g]))])
              for g in range(n_groups)]
    cross = [_cross_link(cmeans[g - 1], cmeans[g], links[g - 1], verbose=verbose, device=kwargs.get("device"),
                         num_bins=cross_num_bins, convergence_tol_vox=kwargs.get("convergence_tol_vox", 0.05),
                         max_extend_factor=kwargs.get("max_extend_factor", 4)) for g in range(1, n_groups)]

    info, components = offset_identifiable(groups)
    root_prefix: dict[int, list] = {0: []}
    for comp in components:
        r = comp[0]
        if r == 0:
            continue
        prev_root = info.get(r - 1, {}).get("root", 0)
        root_prefix[r] = list(root_prefix.get(prev_root, [])) + list(cross[r - 1]["fwdtransforms"])

    # Second pass on the direction-dependent groups with the metric restricted to the caller's mask (anchor space ->
    # each group's mean space through the inverse cross chain); the cross registrations are then recomputed.
    mask_pass = False
    if sample_mask_fn is not None and n_groups > 1:
        mask_anchor = sample_mask_fn(cmeans[0])
        for g in range(1, n_groups):
            chain_paths = list(root_prefix.get(info[g]["root"], []))
            if not chain_paths:
                mask_g = mask_anchor
            else:
                mask_g = ants.apply_transforms(fixed=cmeans[g], moving=mask_anchor, transformlist=chain_paths,
                                               whichtoinvert=[True] * len(chain_paths), interpolator="nearestNeighbor")
            kw2 = dict(group_kw[g])
            kw2["sample_mask"] = mask_g
            if kw2.get("outprefix") is not None:
                kw2["outprefix"] = f"{kw2['outprefix']}m_"
            results[g] = motion_correction(img4d[g], reference=means[g], backend=backend, verbose=verbose,
                                           dvars_reference_intensity=group_ref_intensities[g], **kw2)
            reference_modes[g] = reference_modes[g] + " + masked metric"
        mask_pass = True
        cmeans = [_mean_image([ants.slice_image(results[g].motion_corrected, axis=3, idx=i) for i in range(len(idx[g]))])
                  for g in range(n_groups)]
        cross = [_cross_link(cmeans[g - 1], cmeans[g], links[g - 1], verbose=verbose, device=kwargs.get("device"),
                             num_bins=cross_num_bins, convergence_tol_vox=kwargs.get("convergence_tol_vox", 0.05),
                             max_extend_factor=kwargs.get("max_extend_factor", 4)) for g in range(1, n_groups)]
        root_prefix = {0: []}
        for comp in components:
            r = comp[0]
            if r == 0:
                continue
            prev_root = info.get(r - 1, {}).get("root", 0)
            root_prefix[r] = list(root_prefix.get(prev_root, [])) + list(cross[r - 1]["fwdtransforms"])

    chains: list = [None] * n_frames
    for g in range(n_groups):
        pre = list(root_prefix.get(info[g]["root"], []))
        for li, gi in enumerate(idx[g]):
            chains[gi] = pre + list(results[g].fwdtransforms[li])

    # Series-level Jenkinson FD in acquisition order: every frame's total rigid transform in group 0's space
    # (a point maps through cross_1 ... cross_g, then the frame's own transform: H = H_own @ H_cross_g @ ... @ H_cross_1).
    from .motion import jenkinson_framewise_displacement, attach_band_limited_fd

    com = np.asarray(means[0].get_center_of_mass(), dtype=np.float64)

    def _compose(per_frame_mats):
        homog = []
        for ms in per_frame_mats:
            h = np.eye(4)
            for m in ms:
                h = m @ h
            homog.append(h)
        return homog

    cross_group_offset = None
    if cross_group_debias and n_groups > 1:
        cross_group_offset = _debias_cross_group_offsets(chains, groups, idx, results, cross, com, kwargs.get("outprefix"))
        if cross_group_offset is not None:
            for gi, corr_path in cross_group_offset.pop("correction_paths").items():
                chains[gi] = [corr_path] + chains[gi]   # first in the chain: acts on the anchor-space point first

    frames_out: list = [None] * n_frames
    for g in range(n_groups):
        for li, gi in enumerate(idx[g]):
            if g == 0 and len(chains[gi]) == len(results[0].fwdtransforms[li]):
                frames_out[gi] = ants.slice_image(results[0].motion_corrected, axis=3, idx=li)
            else:  # ONE resampling from the original through the composed chain: every interpolation blurs
                frames_out[gi] = _apply_transforms(fixed=means[0], moving=imgs[g][li], transformlist=chains[gi],
                                                   interpolator=interp)
                pe_g = results[g].get("pe_affine")
                if pe_g is not None and pe_g["jacobian"][li] != 1.0:
                    frames_out[gi] = frames_out[gi] * float(pe_g["jacobian"][li])
    motion_corrected = ants.list_to_ndimage(_make_4d_template_from_3d(frames_out[0], n_frames, tr=tr), frames_out)

    mats = [[_extract_rigid_parameters(ants.read_transform(f), 3)[2] for f in chain] for chain in chains]

    composed_all = _compose(mats)
    fd_raw = jenkinson_framewise_displacement(composed_all, kwargs.get("fd_radius", 50.0), com)
    nuisance = None
    pe_affine = None
    fd = fd_raw
    composed_rigid = composed_all
    if any(r.get("pe_affine") is not None for r in results):
        # the eddy PE-affine is in the chain (and in fd_raw); FD counts the rigid part only: crosses + the frame's rigid
        mats_rigid = []
        for i, (ms, g) in enumerate(zip(mats, [int(groups[i]) for i in range(n_frames)])):
            li = int(np.where(idx[g] == i)[0][0])
            n_own = len(results[g].fwdtransforms[li])
            mats_rigid.append(ms[:-n_own] + [ms[-1]])
        composed_rigid = _compose(mats_rigid)
        fd = jenkinson_framewise_displacement(composed_rigid, kwargs.get("fd_radius", 50.0), com)
        a_all = np.zeros((n_frames, 4)); jac_all = np.ones(n_frames); coef = []
        for g in range(n_groups):
            pe_g = results[g].get("pe_affine")
            cg = None if pe_g is None else pe_g["coefficients"]
            coef.append(cg[0] if isinstance(cg, list) and len(cg) == 1 else cg)  # one matrix per batch; usually one batch
            if pe_g is not None:
                a_all[np.asarray(idx[g])] = pe_g["a"]; jac_all[np.asarray(idx[g])] = pe_g["jacobian"]
        p_unit = np.asarray(pe_axis, dtype=np.float64) / np.linalg.norm(pe_axis)
        pe_affine = {"axis": [float(x) for x in p_unit], "coefficients_by_group": coef,
                     "a": a_all, "jacobian": jac_all, "shift_mm": a_all[:, 0].copy()}
    if nuisance_covariates is not None:
        if nuisance_axis is None:
            raise ValueError("nuisance_covariates needs nuisance_axis (physical unit vector the nuisance translation acts along)")
        cov = np.asarray(nuisance_covariates, dtype=np.float64).reshape(n_frames, -1)
        axis = np.asarray(nuisance_axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        shift = np.zeros(n_frames)
        coefficients, r2 = [], []
        for g in range(n_groups):
            ig = np.asarray(idx[g])
            own = np.array([mats[i][-1] for i in ig])  # the frame's own transform (last in its chain)
            d = np.einsum("nij,j->ni", own[:, :3, :3] - np.eye(3), com) + own[:, :3, 3]
            d = d @ axis  # displacement of the reference centre of mass along the axis, per frame
            X = np.c_[np.ones(len(ig)), cov[ig]]
            if not np.any(cov[ig]) or len(ig) <= X.shape[1]:
                coefficients.append(None)
                r2.append(None)
                continue
            beta, *_ = np.linalg.lstsq(X, d, rcond=None)
            fit = X[:, 1:] @ beta[1:]
            shift[ig] = fit
            ss = float(np.sum((d - d.mean()) ** 2))
            r2.append(float(1.0 - np.sum((d - X @ beta) ** 2) / ss) if ss > 0 else None)
            coefficients.append([float(b) for b in beta[1:]])
        mats_motion = []
        for i, ms in enumerate(mats):
            own = ms[-1].copy()
            own[:3, 3] = own[:3, 3] - shift[i] * axis
            mats_motion.append(ms[:-1] + [own])
        composed_rigid = _compose(mats_motion)
        fd = jenkinson_framewise_displacement(composed_rigid, kwargs.get("fd_radius", 50.0), com)
        nuisance = {"axis": [float(a) for a in axis], "shift_mm": shift, "coefficients": coefficients, "r2": r2}

    # Interleaved groups (multi-shell DWI switches shell almost every volume): a constant per-group pose offset left by the
    # cross-group registration is contrast-driven, not head motion, yet it would be a step at every group switch of the
    # reported motion and FD. Report motion and FD without it; the transforms are unchanged (cross_group_debias=True also
    # removes it from them). ``fd_with_group_offsets`` keeps the previous FD.
    group_offset_fd = None
    if cross_group_offset is None and n_groups > 1:
        est = _group_offsets(composed_rigid, groups, idx, com)
        if est is not None:
            Y_rigid, per_group = est
            composed_rigid = [h if int(groups[i]) not in per_group else _pose_matrix(Y_rigid[i] - per_group[int(groups[i])], com)
                              for i, h in enumerate(composed_rigid)]
            group_offset_fd = {
                "method": "surround_tukey_projection_on_group_indicators",
                "offset_mm_by_group": {int(g): float(np.linalg.norm(v[:3])) for g, v in per_group.items()},
                "offset_deg_by_group": {int(g): float(np.degrees(np.linalg.norm(v[3:]))) for g, v in per_group.items()},
                "fd_with_group_offsets": np.asarray(fd, dtype=np.float64),
            }
            fd = jenkinson_framewise_displacement(composed_rigid, kwargs.get("fd_radius", 50.0), com)

    # Phase-encode B0 / non-linear eddy pseudomotion debiasing on the reported rigid trajectory (composed_rigid / fd):
    # After linear-in-g eddy removal and cross-group offset removal, high-b DWI volumes retain non-linear (g x g, long-time-constant)
    # eddy and respiratory Delta-B0 jitter strictly along the low-bandwidth phase-encode axis p_PE, whereas readout translation
    # and PE/slice rotations remain at the true rigid step floor (~0.05 mm). Shrink excess high-frequency PE residual toward
    # the smooth postural trajectory while gating to zero on frames where the B0-immune DOFs or PE steps indicate a real jump.
    # OPT-IN: an independent audit showed this shrinkage cannot tell independent real head jitter (step lag-1
    # autocorrelation ~ -0.5) from pseudomotion: real nodding jitter FD 0.94 -> 0.15 mm, PE jitter 0.53 -> 0.07 mm on
    # synthetic trajectories; only isolated jumps are protected by the gate. Requested but impossible: announced.
    pe_pseudomotion_debias = None
    if debias_pe_pseudomotion and (pe_axis is None or n_frames < 12):
        reason = "no pe_axis" if pe_axis is None else f"{n_frames} frames < 12"
        warnings.warn(f"motion_correct_grouped: debias_pe_pseudomotion requested but skipped ({reason})", RuntimeWarning, stacklevel=2)
        pe_pseudomotion_debias = {"method": "skipped", "reason": reason}
    if debias_pe_pseudomotion and pe_axis is not None and n_frames >= 12:
        from .motion_batched import _shrink_pe_pseudomotion_core
        from .acquisition_axes import encoding_frame, infer_slice_axis

        fd_rad = float(kwargs.get("fd_radius", 50.0))
        sl_inf = infer_slice_axis(image, slice_axis=slice_axis, sidecar=sidecar, levels=groups)
        p_pe, p_sl, p_ro = encoding_frame(getattr(means[0], "direction", np.eye(3)), pe_axis, sl_inf.voxel_axis)

        R_rig = np.array([H[:3, :3] for H in composed_rigid], dtype=np.float64)
        t_rig = np.array([H[:3, 3] + (H[:3, :3] - np.eye(3)) @ com for H in composed_rigid], dtype=np.float64)

        R_clean, t_clean, deltas, pe_meta = _shrink_pe_pseudomotion_core(
            R_rig, t_rig, com, p_pe, p_sl, p_ro, fd_radius=fd_rad, mode="grouped"
        )
        pe_pseudomotion_debias = pe_meta
        if pe_meta is not None and pe_meta.get("applied"):
            composed_clean = []
            for i in range(n_frames):
                H_c = np.eye(4, dtype=np.float64)
                H_c[:3, :3] = R_clean[i]
                H_c[:3, 3] = com + t_clean[i] - R_clean[i] @ com
                composed_clean.append(H_c)
            fd_before_pe = np.asarray(fd, dtype=np.float64).copy()
            composed_rigid = composed_clean
            fd = jenkinson_framewise_displacement(composed_rigid, fd_rad, com)
            pe_pseudomotion_debias.update({
                "slice_axis": {"voxel_axis": int(sl_inf.voxel_axis), "source": sl_inf.source},
                "fd_before_pe_debias_mean_mm": float(np.mean(fd_before_pe[1:])) if n_frames > 1 else 0.0,
                "fd_before_pe_debias_max_mm": float(np.max(fd_before_pe[1:])) if n_frames > 1 else 0.0,
                "fd_before_pe_debias": fd_before_pe,
            })

    # Extract composed 6-DOF rigid parameters in anchor space (centered at com), matching composed_rigid / fd
    translations_mm = np.zeros((n_frames, 3), dtype=np.float64)
    rotations_deg = np.zeros((n_frames, 3), dtype=np.float64)
    for i, H in enumerate(composed_rigid):
        R_i = H[:3, :3]
        translations_mm[i] = H[:3, 3] + (R_i - np.eye(3)) @ com
        sy = -R_i[2, 0]
        if np.abs(sy) < 0.999999:
            ry = np.arcsin(np.clip(sy, -1.0, 1.0))
            rx = np.arctan2(R_i[2, 1], R_i[2, 2])
            rz = np.arctan2(R_i[1, 0], R_i[0, 0])
        else:
            ry = (np.pi / 2.0) if sy > 0 else (-np.pi / 2.0)
            rx = np.arctan2(-R_i[1, 2], R_i[1, 1])
            rz = 0.0
        rotations_deg[i] = np.degrees([rx, ry, rz])

    out = GroupedMotionCorrectionResult()
    out["motion_corrected"] = motion_corrected
    out["fd"] = fd
    out["fd_raw"] = fd_raw
    out["translations_mm"] = translations_mm
    out["rotations_deg"] = rotations_deg
    out["nuisance"] = nuisance
    out["pe_affine"] = pe_affine
    out["reference_modes"] = reference_modes
    out["masked_metric"] = mask_pass
    out["bias"] = [r.get("bias") for r in results]
    out["encoding_pose_bias_mm_rad"] = [r.get("encoding_pose_bias_mm_rad") for r in results]
    out["cross_group_offset"] = cross_group_offset
    out["group_offset_fd"] = group_offset_fd
    out["pe_pseudomotion_debias"] = pe_pseudomotion_debias
    attach_band_limited_fd(out, image, translations_mm, rotations_deg, tr_s=kwargs.get("tr"), radius=float(kwargs.get("fd_radius", 50.0)), fd_method="jenkinson", center_of_mass=com)
    # convergence of the per-frame registrations (every stage of every frame ended on the pose-step test) and of each cross-shell
    # registration (the winning path's stages); False entries are frames/steps that were CAPPED, never silently final.
    frame_converged = np.ones(int(sum(len(i) for i in idx)), dtype=bool)
    for g, r in enumerate(results):
        conv = r.get("convergence")
        if conv:
            for local_i, t in enumerate(idx[g]):
                frame_converged[t] = bool(conv["frame_converged"][local_i])
    out["frame_converged"] = frame_converged
    out["cross_converged"] = [bool(all(rec["converged"] for rec in c.get("convergence", []) if rec.get("path") == c.get("init_candidate") and rec.get("level", 1) == 1)) for c in cross]
    out["cross_link"] = list(links)
    out["fwdtransforms"] = chains
    out["group_results"] = results
    out["group_indices"] = idx
    out["group_means"] = cmeans
    out["cross_registrations"] = cross
    out["preset"] = preset
    return out
