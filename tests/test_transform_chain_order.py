"""``TransformChain.linear_part`` composes in the order the chain is applied: first-listed acts first on the point."""
import os

import ants
import numpy as np

from syntx.transforms import TransformChain, apply_transforms


def _write(R, t, path):
    tx = ants.create_ants_transform(transform_type="AffineTransform", precision="float", dimension=3)
    tx.set_parameters(np.concatenate([R.flatten(), t])); tx.set_fixed_parameters([0.0, 0.0, 0.0])
    ants.write_transform(tx, path)
    return path


def test_linear_part_follows_application_order(tmp_path):
    Rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    a = _write(Rz, np.zeros(3), str(tmp_path / "A.mat")); b = _write(Rx, np.zeros(3), str(tmp_path / "B.mat"))
    chain = TransformChain.from_paths([a, b])
    assert np.allclose(chain.linear_part(), Rx @ Rz)                       # A acts first, then B
    # the same order as the resampling: a point source at m lands where chain(x) = m
    img = ants.from_numpy(np.zeros((40, 40, 40), np.float32), spacing=(1.0, 1.0, 1.0), origin=(-20.0, -20.0, -20.0))
    arr = img.numpy(); arr[30, 20, 20] = 1.0; mov = img.new_image_like(arr)   # m = (10, 0, 0)
    out = apply_transforms(fixed=img, moving=mov, transformlist=[a, b], interpolator="nearestNeighbor").numpy()
    x = np.array(np.unravel_index(int(np.argmax(out)), out.shape), dtype=float) + np.asarray(img.origin)
    assert np.allclose(chain.linear_part() @ x, [10.0, 0.0, 0.0]), x
