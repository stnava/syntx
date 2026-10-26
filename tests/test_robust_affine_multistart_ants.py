"""robust_affine(mode='ants', multi_start=True): the rotation candidates are rotations about the fixed centre of mass
(no double centring) and the low-resolution MI ranks them (images normalised to the (0, 1) histogram range).

Known-transform head phantom: the moving image is the fixed one rotated about its centre of mass by a single-axis angle
that is on the candidate grid, so the matching candidate must win and the ANTs affine started from it must recover the
transform (from the CoM candidate the local optimiser stays 5-9 mm off).
"""

import contextlib
import io
import re

import ants
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from syntx import robust_affine
from syntx.motion_simulation import make_head_phantom
from syntx.robust_affine import _eval_low_res_mi
from syntx.sampling import normalize_robust


def _pair(rot_xyz_deg, origin=(0.0, 0.0, 0.0), seed=0):
    ph = make_head_phantom(shape=(56, 64, 48), spacing=(3.0, 3.0, 3.0), seed=seed)
    arr = (0.6 * ph.gm + 1.0 * ph.wm + 0.25 * ph.csf + ph.texture * (ph.gm + ph.wm)).astype(np.float32) * 800
    fixed = ants.from_numpy(arr, origin=origin, spacing=(3.0, 3.0, 3.0))
    p0 = np.asarray(ants.get_center_of_mass(fixed))
    R = Rotation.from_euler("xyz", rot_xyz_deg, degrees=True).as_matrix()
    t = np.array([3.0, -2.0, 1.5])
    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=3)
    tx.set_parameters(np.concatenate([R.ravel(), t]))
    tx.set_fixed_parameters(p0)
    moving = tx.invert().apply_to_image(fixed, reference=fixed, interpolation="linear")   # moving(T x) = fixed(x)
    pts = np.array([ants.transform_index_to_physical_point(fixed, [int(i) for i in v])
                    for v in np.argwhere((ph.gm + ph.wm) > 0.5)[::40]])
    return fixed, moving, pts, (R @ (pts - p0).T).T + p0 + t


@pytest.mark.parametrize("rot,expected", [((8, 0, 0), "Rotation_pitch_+8deg"), ((0, 0, 12), "Rotation_yaw_+12deg")])
@pytest.mark.parametrize("origin", [(0.0, 0.0, 0.0), (-90.0, -110.0, -70.0)])
def test_rotation_candidate_selected_and_recovered(rot, expected, origin):
    fixed, moving, pts, true = _pair(rot, origin)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = robust_affine(fixed, moving, mode="ants", multi_start=True, verbose=True, seed=0)
    selected = re.search(r"Selected winning candidate '([^']+)'", buf.getvalue()).group(1)
    assert selected == expected
    est = ants.apply_transforms_to_points(3, pd.DataFrame(pts, columns=list("xyz")), res["fwdtransforms"])[list("xyz")].values
    assert np.sqrt(np.mean(np.sum((est - true) ** 2, axis=1))) < 2.5


def test_low_res_mi_ranks_alignment_on_normalised_images():
    fixed, moving, _, _ = _pair((0, 0, 12))
    fi, mi = normalize_robust(fixed), normalize_robust(moving)
    aligned = _eval_low_res_mi(fi, normalize_robust(fixed))
    misaligned = _eval_low_res_mi(fi, mi)
    assert aligned < misaligned - 0.05   # -MI: lower is better
