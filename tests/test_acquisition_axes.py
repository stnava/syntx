"""syntx.acquisition_axes: slice and phase-encode axes from metadata or from the image (interleave, Nyquist ghost)."""
import ants
import numpy as np
import pytest

from syntx.acquisition_axes import (AcquisitionAxisUndecidable, encoding_frame, infer_phase_encode_axis,
                                    infer_phase_encode_axis_detail, infer_slice_axis)
from syntx.motion_simulation import make_head_phantom

_c, _s = np.cos(np.radians(10.0)), np.sin(np.radians(10.0))
_OBLIQUE = np.array([[_c, -_s, 0.0], [_s, _c, 0.0], [0.0, 0.0, 1.0]])


def _series(slice_axis, pe_axis, T=60, spacing=(2.0, 2.0, 2.0), ghost=0.06, seed=0):
    """Head phantom series in a padded field of view: interleaved slice acquisition along ``slice_axis`` (odd and even
    slices see different temporal fluctuations), a respiratory B0 shift (0.3 voxel, sub-voxel) and an N/2 Nyquist ghost
    along ``pe_axis``."""
    from scipy.ndimage import shift as nd_shift

    ph = make_head_phantom(shape=(36, 40, 32), spacing=(2.5, 2.5, 2.5), seed=seed)
    content = (1.0 + ph.texture) * (1.0 * ph.gm + 0.8 * ph.wm) + 1.3 * ph.csf  # as simulate_bold_series
    base = np.pad(content.astype(np.float32), 10)
    rng = np.random.default_rng(seed)
    f_odd = np.cumsum(rng.normal(size=T)) * 0.004
    f_even = np.cumsum(rng.normal(size=T)) * 0.004
    shp = base.shape
    par = (np.arange(shp[slice_axis]) % 2).astype(bool)
    bshape = [1, 1, 1]
    bshape[slice_axis] = -1
    par = par.reshape(bshape)
    frames = []
    for t in range(T):
        fr = base * (1.0 + np.where(par, f_odd[t], f_even[t]))
        sh = [0.0, 0.0, 0.0]
        sh[pe_axis] = 0.3 * np.sin(2.0 * np.pi * 0.23 * t)
        fr = nd_shift(fr, sh, order=1)
        fr = fr + ghost * np.roll(fr, shp[pe_axis] // 2, axis=pe_axis)
        fr = fr + 0.01 * np.abs(rng.normal(size=shp))
        frames.append(fr)
    a = np.stack(frames, -1).astype(np.float32)
    img = ants.from_numpy(a, origin=(0.0, 0.0, 0.0, 0.0), spacing=tuple(spacing) + (1.0,))
    D = np.eye(4)
    D[:3, :3] = _OBLIQUE
    img.set_direction(D)
    return img


@pytest.mark.parametrize("slice_axis,pe_axis", [(2, 1), (0, 2), (1, 0)])
def test_image_cues_find_slice_and_phase_encode_axes(slice_axis, pe_axis):
    img = _series(slice_axis, pe_axis)
    s = infer_slice_axis(img)
    assert (s.voxel_axis, s.source) == (slice_axis, "interleave")
    p = infer_phase_encode_axis_detail(img, slice_axis=s.voxel_axis)
    assert p.voxel_axis == pe_axis and p.scores["ghost_ratio"] >= 1.5


def test_metadata_wins_in_order_explicit_sidecar_slice_timing():
    img = _series(2, 1, T=24)
    assert infer_slice_axis(img, slice_axis="i").source == "explicit"
    s = infer_slice_axis(img, sidecar={"SliceEncodingDirection": "j-"})
    assert (s.voxel_axis, s.source) == (1, "sidecar")
    n_k = img.shape[2]
    s = infer_slice_axis(img, sidecar={"SliceTiming": list(np.linspace(0, 1, n_k))})
    assert (s.voxel_axis, s.source) == (2, "slice_timing")


def test_thick_axis_is_the_slice_axis_for_short_series():
    img = _series(0, 1, T=12, spacing=(4.0, 1.9, 1.9))
    s = infer_slice_axis(img)
    assert (s.voxel_axis, s.source) == (0, "voxel_thickness")


def test_undecidable_without_metadata_short_isotropic_series():
    img = _series(2, 1, T=10)
    with pytest.raises(AcquisitionAxisUndecidable):
        infer_slice_axis(img)


def test_no_ghost_is_undecidable():
    img = _series(2, 1, ghost=0.0)
    with pytest.raises(AcquisitionAxisUndecidable):
        infer_phase_encode_axis_detail(img, slice_axis=2)


def test_resolve_phase_encode_hook_returns_the_physical_axis():
    from syntx.phase_encode import resolve_phase_encode

    img = _series(2, 1)
    pe = resolve_phase_encode(img, allow_inference=True, inference=infer_phase_encode_axis)
    assert pe.voxel_axis == 1 and pe.sign is None and pe.source == "inferred"


def test_encoding_frame_is_orthonormal_and_follows_the_header():
    p_pe, p_sl, p_ro = encoding_frame(_OBLIQUE, 1, 2)
    M = np.stack([p_pe, p_sl, p_ro])
    np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(p_pe, _OBLIQUE[:, 1], atol=1e-10)
    np.testing.assert_allclose(p_sl, _OBLIQUE[:, 2], atol=1e-10)
    with pytest.raises(ValueError):
        encoding_frame(_OBLIQUE, 1, 1)
