"""syntx.frames — Canonical vector and tensor coordinate frame conventions (F1–F6).

Storage convention (F1):
    Every saved vector or tensor pixel (velocity, V1, fixel peaks, FOD SH) has its
    components along the image axes of the image holding it. Physical direction is:
        v_lps = D @ v_img
        v_img = D.T @ v_lps
    Displacement fields and warps remain in physical LPS mm (ITK exception).

Gradient tables (F2):
    On disk, bvecs follow FSL/BIDS (radiological voxel frame).
    On read, convert to image frame: g_img = F(D) @ g_bvec
    On write, convert to bvec:       g_bvec = F(D_out) @ g_img
    where F(D) = diag(-1, 1, 1) if det(D) > 0 else I.

Reorientation (F3):
    Vectors, tensors, and gradient tables move between spaces using:
        g_fix = D_fix.T @ polar_+(M^-1) @ D_mov @ g_mov
    where polar_+(M^-1) is the proper rotation (det = +1) in physical LPS.
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np

VECTOR_FRAME: str = "image_axes"


def _as_direction(direction: Any) -> np.ndarray:
    if hasattr(direction, "direction"):
        d = direction.direction
    else:
        d = direction
    arr = np.asarray(d, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 3:
        return arr[:3, :3]
    raise ValueError(f"Expected 2-D direction matrix of at least (3, 3), got shape {arr.shape}")


def _proper_polar(mat: np.ndarray) -> np.ndarray:
    """Exact orthogonal polar factor R in SO(3) with det(R) = +1 in float64."""
    from .spatial import polar_rotation_field

    return polar_rotation_field(mat.astype(np.float64))


def fsl_handedness(direction: Any) -> np.ndarray:
    """Return FSL/BIDS handedness involution matrix F(D).

    Returns diag(-1, 1, 1) if det(D[:3, :3]) > 0 (neurological / RAS / LPI),
    or eye(3) if det(D[:3, :3]) < 0 (radiological / LAS / RPI).
    """
    D = _as_direction(direction)
    det = float(np.linalg.det(D))
    if abs(det) < 1e-6:
        raise ValueError(f"Singular direction matrix with det {det}")
    if det > 0.0:
        return np.diag([-1.0, 1.0, 1.0])
    return np.eye(3, dtype=np.float64)


def bvecs_to_image_frame(
    bvecs: np.ndarray,
    direction: Any,
    normalize: bool = True,
) -> np.ndarray:
    """Convert FSL/BIDS bvecs (N, 3) or (3,) to image-frame unit vectors via F(D) @ bvec.

    b0 rows (zero norm) remain exactly zero [0.0, 0.0, 0.0].
    """
    arr = np.asarray(bvecs, dtype=np.float64)
    orig_ndim = arr.ndim
    if orig_ndim == 1 and arr.size == 3:
        arr = arr.reshape(1, 3)
    elif arr.shape[-1] != 3:
        raise ValueError(f"bvecs last dimension must be 3, got shape {bvecs.shape}")

    F = fsl_handedness(direction)
    out = arr @ F
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    if normalize:
        nz = norms > 0.0
        out = np.where(nz, out / np.maximum(norms, 1e-12), 0.0)
    out[norms.squeeze(-1) == 0.0] = 0.0
    return out.squeeze(0) if orig_ndim == 1 else out


def image_frame_to_bvecs(
    g: np.ndarray,
    direction: Any,
    normalize: bool = True,
) -> np.ndarray:
    """Convert image-frame gradient vectors (N, 3) or (3,) to FSL/BIDS bvecs via F(D) @ g.

    b0 rows remain exactly zero [0.0, 0.0, 0.0].
    """
    return bvecs_to_image_frame(g, direction, normalize=normalize)


def image_to_lps(
    v: Any,
    direction: Any | None = None,
    kind: str = "vector",
) -> Any:
    """Transform image-frame vectors or tensors to physical LPS.

    For vectors: v_lps = D @ v_img
    For 3x3 tensors: T_lps = D @ T_img @ D.T
    For sym6 tensors: unpacks sym6, applies D @ T @ D.T, repacks to sym6.
    """
    if kind not in ("vector", "tensor"):
        raise ValueError(f"kind must be 'vector' or 'tensor', got {kind}")

    import ants

    is_ants = hasattr(ants, "is_image") and ants.is_image(v)
    ref_img = v if is_ants else None
    arr = v.numpy() if is_ants else np.asarray(v)

    if direction is None:
        if is_ants:
            D = _as_direction(ref_img.direction)
        else:
            raise ValueError("direction must be provided when v is a numpy array")
    else:
        D = _as_direction(direction)

    if kind == "vector":
        if is_ants and getattr(ref_img, "has_components", False):
            if getattr(ref_img, "components", None) != 3:
                raise ValueError(
                    f"Expected ANTsImage with 3 components for kind='vector', got {ref_img.components}"
                )
        if arr.shape[-1] != 3:
            raise ValueError(f"Expected vector with last dimension 3, got shape {arr.shape}")
        out = (arr[..., None, :] @ D.T).squeeze(-2)
    elif kind == "tensor":
        if arr.ndim >= 2 and arr.shape[-2:] == (3, 3):
            out = np.einsum("ij,...jk,lk->...il", D, arr, D)
        elif arr.shape[-1] == 6:
            # sym6 order: [xx, xy, xz, yy, yz, zz]
            idx = np.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])
            t33 = arr[..., idx]
            t_lps = np.einsum("ij,...jk,lk->...il", D, t33, D)
            out = np.stack(
                [
                    t_lps[..., 0, 0],
                    t_lps[..., 0, 1],
                    t_lps[..., 0, 2],
                    t_lps[..., 1, 1],
                    t_lps[..., 1, 2],
                    t_lps[..., 2, 2],
                ],
                axis=-1,
            )
        else:
            raise ValueError(
                f"Expected tensor ending with (3, 3) or 6 for kind='tensor', got shape {arr.shape}"
            )

    if is_ants:
        return ants.from_numpy(
            out.astype(np.float32),
            origin=tuple(float(x) for x in ref_img.origin[:3]),
            spacing=tuple(float(x) for x in ref_img.spacing[:3]),
            direction=np.asarray(ref_img.direction)[:3, :3],
            has_components=getattr(ref_img, "has_components", True),
        )
    return out


def lps_to_image(
    v: Any,
    direction: Any | None = None,
    kind: str = "vector",
) -> Any:
    """Transform physical LPS vectors or tensors to image frame.

    For vectors: v_img = D.T @ v_lps
    For 3x3 tensors: T_img = D.T @ T_lps @ D
    For sym6 tensors: unpacks sym6, applies D.T @ T @ D, repacks to sym6.
    """
    if kind not in ("vector", "tensor"):
        raise ValueError(f"kind must be 'vector' or 'tensor', got {kind}")

    import ants

    is_ants = hasattr(ants, "is_image") and ants.is_image(v)
    ref_img = v if is_ants else None
    arr = v.numpy() if is_ants else np.asarray(v)

    if direction is None:
        if is_ants:
            D = _as_direction(ref_img.direction)
        else:
            raise ValueError("direction must be provided when v is a numpy array")
    else:
        D = _as_direction(direction)

    if kind == "vector":
        if is_ants and getattr(ref_img, "has_components", False):
            if getattr(ref_img, "components", None) != 3:
                raise ValueError(
                    f"Expected ANTsImage with 3 components for kind='vector', got {ref_img.components}"
                )
        if arr.shape[-1] != 3:
            raise ValueError(f"Expected vector with last dimension 3, got shape {arr.shape}")
        out = (arr[..., None, :] @ D).squeeze(-2)
    elif kind == "tensor":
        if arr.ndim >= 2 and arr.shape[-2:] == (3, 3):
            out = np.einsum("ij,...jk,lk->...il", D.T, arr, D.T)
        elif arr.shape[-1] == 6:
            idx = np.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])
            t33 = arr[..., idx]
            t_img = np.einsum("ij,...jk,lk->...il", D.T, t33, D.T)
            out = np.stack(
                [
                    t_img[..., 0, 0],
                    t_img[..., 0, 1],
                    t_img[..., 0, 2],
                    t_img[..., 1, 1],
                    t_img[..., 1, 2],
                    t_img[..., 2, 2],
                ],
                axis=-1,
            )
        else:
            raise ValueError(
                f"Expected tensor ending with (3, 3) or 6 for kind='tensor', got shape {arr.shape}"
            )

    if is_ants:
        return ants.from_numpy(
            out.astype(np.float32),
            origin=tuple(float(x) for x in ref_img.origin[:3]),
            spacing=tuple(float(x) for x in ref_img.spacing[:3]),
            direction=np.asarray(ref_img.direction)[:3, :3],
            has_components=getattr(ref_img, "has_components", True),
        )
    return out


def reorient_gradient_table(
    g_img: np.ndarray,
    chains: Any,
    moving_direction: Any,
    fixed_direction: Any,
    normalize: bool = True,
) -> np.ndarray:
    """Reorient image-frame gradients using g_fix = D_fix.T @ polar_+(M^-1) @ D_mov @ g_mov.

    Parameters
    ----------
    g_img : (N, 3) ndarray
        Image-frame gradients.
    chains : TransformChain | Sequence[TransformChain | None] | None
        Motion/registration chains. If None, identity rotation is used.
    moving_direction : Any
        Direction matrix of moving series.
    fixed_direction : Any
        Direction matrix of reference/fixed frame.
    normalize : bool
        If True, non-zero output gradients are normalized to unit length.
    """
    g = np.asarray(g_img, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != 3:
        raise ValueError(f"g_img must be (N, 3), got {g.shape}")

    d_mov = _as_direction(moving_direction)
    d_fix = _as_direction(fixed_direction)

    if chains is None or hasattr(chains, "linear_part"):
        # Single chain for all volumes
        if chains is None:
            r_lps = np.eye(3, dtype=np.float64)
        else:
            m = chains.linear_part()
            r_lps = _proper_polar(np.linalg.inv(m))
        r_img = d_fix.T @ r_lps @ d_mov
        out = g @ r_img.T
    else:
        chain_seq = list(chains)
        if len(chain_seq) != g.shape[0]:
            raise ValueError(f"len(chains)={len(chain_seq)} != number of bvecs {g.shape[0]}")
        out = np.zeros_like(g)
        for k, ch in enumerate(chain_seq):
            if ch is None:
                r_lps = np.eye(3, dtype=np.float64)
            else:
                m = ch.linear_part()
                r_lps = _proper_polar(np.linalg.inv(m))
            r_img = d_fix.T @ r_lps @ d_mov
            out[k] = r_img @ g[k]

    if normalize:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        in_norms = np.linalg.norm(g, axis=1, keepdims=True)
        nz = (in_norms > 0.0) & (norms > 0.0)
        out = np.where(nz, out / np.maximum(norms, 1e-12), 0.0)
    in_norms = np.linalg.norm(g, axis=1, keepdims=True)
    out[in_norms.squeeze(1) == 0.0] = 0.0
    return out


def dec_rgb(
    vector_image: Any,
    weight: Any | None = None,
    frame: str = "image",
    mask: Any | None = None,
    vmax: float | None = None,
    direction: Any | None = None,
    origin: Sequence[float] | None = None,
    spacing: Sequence[float] | None = None,
) -> Any:
    """Compute direction-encoded color RGB image (multichannel ANTsImage).

    When frame="image": |v_img| * weight (saved dtirgb standard).
    When frame="anatomical": |D @ v_img| * weight (native grid; Red = Left-Right,
    Green = Anterior-Posterior, Blue = Inferior-Superior).
    """
    import ants

    is_ants = hasattr(ants, "is_image") and ants.is_image(vector_image)
    ref_img = vector_image if is_ants else None
    v_arr = vector_image.numpy() if is_ants else np.asarray(vector_image, dtype=np.float32)

    if v_arr.shape[-1] != 3:
        raise ValueError(f"vector_image last dimension must be 3, got shape {v_arr.shape}")

    if frame == "image":
        rgb_raw = np.abs(v_arr)
    elif frame == "anatomical":
        if direction is not None:
            D = _as_direction(direction)
        elif is_ants:
            D = _as_direction(ref_img.direction)
        elif hasattr(vector_image, "direction") and vector_image.direction is not None:
            D = _as_direction(vector_image.direction)
        else:
            raise ValueError(
                "dec_rgb(frame='anatomical') requires an image direction matrix; none was provided"
            )
        v_lps = (v_arr[..., None, :] @ D.T).squeeze(-2)
        rgb_raw = np.abs(v_lps)
    else:
        raise ValueError(f"Unknown frame {frame!r}. Expected 'image' or 'anatomical'.")

    if weight is not None:
        w_arr = weight.numpy() if (hasattr(ants, "is_image") and ants.is_image(weight)) else np.asarray(weight, dtype=np.float32)
        rgb_raw = rgb_raw * w_arr[..., None]

    if mask is not None:
        m_arr = mask.numpy() if (hasattr(ants, "is_image") and ants.is_image(mask)) else np.asarray(mask)
        rgb_raw = np.where((m_arr > 0)[..., None], rgb_raw, 0.0)

    if vmax is not None and vmax > 0.0:
        rgb_raw = np.clip(rgb_raw / vmax, 0.0, 1.0)

    out_arr = rgb_raw.astype(np.float32)
    if is_ants:
        out_origin = tuple(float(x) for x in ref_img.origin[:3]) if origin is None else tuple(float(x) for x in origin[:3])
        out_spacing = tuple(float(x) for x in ref_img.spacing[:3]) if spacing is None else tuple(float(x) for x in spacing[:3])
        out_direction = np.asarray(ref_img.direction)[:3, :3] if direction is None else _as_direction(direction)
    else:
        out_origin = tuple(float(x) for x in origin[:3]) if origin is not None else None
        out_spacing = tuple(float(x) for x in spacing[:3]) if spacing is not None else None
        out_direction = _as_direction(direction) if direction is not None else None

    kwargs: dict[str, Any] = {"has_components": True}
    if out_origin is not None:
        kwargs["origin"] = out_origin
    if out_spacing is not None:
        kwargs["spacing"] = out_spacing
    if out_direction is not None:
        kwargs["direction"] = out_direction

    return ants.from_numpy(out_arr, **kwargs)


def reference_directions() -> dict[str, np.ndarray]:
    """Canonical test direction matrices.

    Returns dict mapping names to (3, 3) float64 direction cosine matrices:
    - "LAS" / "RPI": det = -1.0 (radiological standard)
    - "RAS" / "LPI": det = +1.0 (neurological standard)
    - "PIR": sagittal acquisition permutation (det = +1.0)
    - "oblique_pos": 30 deg oblique rotation with det = +1.0
    - "oblique_neg": 30 deg oblique rotation with det = -1.0
    """
    th = np.deg2rad(30.0)
    c, s = np.cos(th), np.sin(th)
    r_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    las = np.diag([1.0, -1.0, 1.0])
    ras = np.diag([-1.0, -1.0, 1.0])
    pir = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)

    oblique_pos = (r_z @ np.diag([-1.0, -1.0, 1.0])).astype(np.float64)
    oblique_neg = (r_z @ np.diag([1.0, -1.0, 1.0])).astype(np.float64)

    return {
        "LAS": las.copy(),
        "RPI": las.copy(),
        "RAS": ras.copy(),
        "LPI": ras.copy(),
        "PIR": pir.copy(),
        "oblique_pos": oblique_pos.copy(),
        "oblique_neg": oblique_neg.copy(),
    }
