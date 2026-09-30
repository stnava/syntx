"""
syntx.liouville_determinant: the per-method Jacobian determinant used as the folding measure.
Where the map is smooth the finite-difference Jacobian of the exported warp is reliable, so the
two must agree there; greedy is defined as that finite difference.
"""

import numpy as np
import pytest
import ants

import syntx


@pytest.fixture(scope="module")
def pair():
    fi = ants.image_read(ants.get_ants_data("r16")).resample_image((2, 2), 0, 0)
    mi = ants.image_read(ants.get_ants_data("r64")).resample_image((2, 2), 0, 0)
    return fi, mi


def _run(method, fi, mi):
    fn = getattr(syntx, method)
    kw = dict(fixed=fi, moving=mi, device="cpu", verbose=False)
    kw["initial_transform"] = False if method == "greedy" else "identity"
    kw["reg_iterations"] = [20, 10, 5, 0] if method == "syngs" else [20, 10, 5]
    return fn(**kw)


@pytest.mark.parametrize("method", ["tvf", "syngs", "syn", "greedy"])
def test_determinant_agrees_with_finite_differences_where_smooth(method, pair):
    fi, mi = pair
    r = _run(method, fi, mi)
    det, info = syntx.liouville_determinant(r, fi, return_details=True)
    assert info["method"] == method
    warp = next(x for x in r["fwdtransforms"] if x.endswith(".nii.gz"))
    fd = ants.create_jacobian_determinant_image(fi, warp).numpy()
    d = det.numpy()
    assert d.shape == fd.shape
    if method == "greedy":
        assert np.allclose(d, fd)
        return
    ok = (fd > 0.5) & (fd < 2.0) & (ants.get_mask(fi).numpy() > 0)
    assert ok.sum() > 100
    assert abs(np.median(d[ok] / fd[ok]) - 1.0) < 0.02, (method, np.median(d[ok] / fd[ok]))


def test_method_is_detected_and_validated(pair):
    fi, mi = pair
    r = _run("tvf", fi, mi)
    assert syntx.liouville_determinant(r, fi, return_details=True)[1]["method"] == "tvf"
    with pytest.raises(ValueError, match="method must be one of"):
        syntx.liouville_determinant(r, fi, method="ants")
    s = syntx.determinant_summary(syntx.liouville_determinant(r, fi), fi)
    assert set(s) == {"min", "max", "mean", "std", "folding_pct"} and s["min"] > 0
