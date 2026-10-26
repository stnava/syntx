"""syntx.motion_design: the acquisition-design menu and the single motion-correction entry point."""
import numpy as np
import pytest

from syntx.motion_design import DESIGNS, _real_sh_l2, make_design, motion_correct_design


def test_menu_entries_and_unknown_name():
    assert set(DESIGNS) == {"none", "tag_control", "shells", "encoding_states", "custom"}
    with pytest.raises(ValueError, match="unknown acquisition design"):
        make_design("sense")


def test_tag_control_levels_and_column():
    d = make_design("tag_control", volume_types=["m0scan"] + ["control", "label"] * 4)
    assert d.levels.tolist() == [1] + [0] * 8 and d.n_levels == 2
    assert d.encoding.tolist() == [0.0] + [1.0, -1.0] * 4
    short = make_design("tag_control", volume_types=["control", "label"] * 2)
    assert short.encoding is None  # fewer than 6 control/label volumes: no tag-control column


def test_shells_levels_and_direction_terms():
    rng = np.random.default_rng(0)
    bvals = np.array([0, 1000, 2000, 1000, 2000, 0, 1000, 2000])
    g = rng.normal(size=(8, 3))
    g[bvals == 0] = 0.0
    d = make_design("shells", bvals=bvals, bvecs=g, pe_axis=(0.0, 1.0, 0.0))
    assert d.levels.tolist() == [0, 1, 2, 1, 2, 0, 1, 2]
    assert d.bias_covariates.shape == (8, 5) and np.all(d.bias_covariates[bvals == 0] == 0.0)
    assert d.pe_affine_covariates.shape == (8, 3)


def test_l2_harmonics_are_the_traceless_quadratic_forms():
    """The five l = 2 real harmonics span {x^2 - y^2, 2z^2 - x^2 - y^2, xy, xz, yz} (unit vectors)."""
    rng = np.random.default_rng(1)
    g = rng.normal(size=(40, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    Y = _real_sh_l2(g, np.ones(40, bool))
    x, y, z = g.T
    Q = np.c_[x * x - y * y, 2 * z * z - x * x - y * y, x * y, x * z, y * z]
    coef = np.linalg.lstsq(Q, Y, rcond=None)[0]
    np.testing.assert_allclose(Q @ coef, Y, atol=1e-10)
    assert np.linalg.matrix_rank(Y) == 5


def test_frame_count_must_match():
    import ants

    img = ants.from_numpy(np.ones((8, 8, 8, 5), np.float32))
    with pytest.raises(ValueError, match="frames"):
        motion_correct_design(img, make_design("none", n_frames=6))


def test_dispatch_reproduces_the_modality_call_bold_and_asl():
    """``engine='dispatch'`` is today's per-modality call: identical transforms on simulated BOLD and ASL."""
    import ants

    from syntx.motion import motion_correction
    from syntx.motion_simulation import simulate_asl_series, simulate_bold_series

    def params(chains):
        return np.array([ants.read_transform(c[-1] if isinstance(c, (list, tuple)) else c).parameters for c in chains])

    bold = simulate_bold_series(n_frames=16, seed=3)
    a = motion_correction(bold.image, reference="mean", type_of_transform="Rigid", preset="bold", device="cpu")
    b = motion_correct_design(bold.image, make_design("none", n_frames=16), device="cpu")
    np.testing.assert_allclose(params(a["fwdtransforms"]), params(b["fwdtransforms"]), atol=1e-6)
    assert b["design"]["call"] == "motion_correction(preset='bold')"

    asl = simulate_asl_series(n_frames=12, seed=3)
    vt = ["control" if s > 0 else "label" for s in asl.encoding_covariates]
    a = motion_correction(asl.image, reference="mean", type_of_transform="Rigid", preset="asl",
                          encoding_covariates=asl.encoding_covariates, device="cpu")
    b = motion_correct_design(asl.image, make_design("tag_control", volume_types=vt), device="cpu")
    np.testing.assert_allclose(params(a["fwdtransforms"]), params(b["fwdtransforms"]), atol=1e-6)
