"""``resample_series`` applies the motion chain's affine Jacobian (the eddy term's ``1 + a . p``), as the solver does for
``motion_corrected``; a rigid-only chain is unchanged."""
import os

import ants
import numpy as np

from syntx.transforms import TransformChain, chain_linear_jacobian, resample_series


def _write_affine(M, t, c, path):
    tx = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3)
    tx.set_parameters(np.concatenate([M.flatten(), t])); tx.set_fixed_parameters(c)
    ants.write_transform(tx, path)
    return path


def test_resample_series_applies_the_eddy_jacobian(tmp_path):
    N = 24
    ref = ants.from_numpy(np.full((N, N, N), 10.0, np.float32), spacing=(2.0, 2.0, 2.0), origin=(-24.0, -24.0, -24.0))
    frames = [ref.clone(), ref.clone()]
    img4d = ants.list_to_ndimage(ants.from_numpy(np.zeros((N, N, N, 2), np.float32), spacing=(2.0, 2.0, 2.0, 1.0), origin=(-24.0, -24.0, -24.0, 0.0)), frames)
    zero_field = ants.from_numpy(np.zeros((N, N, N, 3), np.float32), spacing=(2.0, 2.0, 2.0), origin=(-24.0, -24.0, -24.0), has_components=True)
    field_path = str(tmp_path / "zero_warp.nii.gz"); ants.image_write(zero_field, field_path)
    c = np.zeros(3)
    p = np.array([0.0, 1.0, 0.0]); a = np.array([0.0, 0.08, 0.0])            # scale 8 % along phase encode
    rigid = _write_affine(np.eye(3), np.zeros(3), c, str(tmp_path / "0001.mat"))
    eddy = _write_affine(np.eye(3) + np.outer(p, a), np.zeros(3), c, str(tmp_path / "0001_eddy.mat"))
    chains = [TransformChain.from_paths([rigid]), TransformChain.from_paths([rigid, eddy])]
    assert abs(chain_linear_jacobian(chains[0]) - 1.0) < 1e-6
    assert abs(chain_linear_jacobian(chains[1]) - (1.0 + float(a @ p))) < 1e-5
    out = resample_series(img4d, chains, field_path, ref, jacobian=True, device="cpu")
    arr = out.numpy()
    core = (slice(4, -4),) * 3
    assert np.allclose(arr[core + (0,)], 10.0, atol=1e-3)
    assert np.allclose(arr[core + (1,)], 10.0 * (1.0 + float(a @ p)), atol=2e-2), float(arr[core + (1,)].mean())
    out_nojac = resample_series(img4d, chains, field_path, ref, jacobian=False, device="cpu")
    assert np.allclose(out_nojac.numpy()[core + (1,)], 10.0, atol=1e-3)
