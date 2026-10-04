"""Regression tests for the docstring-audit behaviour issues in the registration-method
modules (syn, syngs, tvf, greedy, template, motion, deformation_metrics, image_utils).
CPU only, tiny inputs, spies instead of full registrations where possible."""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _no_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


class _Stop(Exception):
    pass


# ---------------------------------------------------------------------------------------
# motion.py
# ---------------------------------------------------------------------------------------

def test_motion_pytorch_backend_calls_robust_affine_without_backend_kw(monkeypatch):
    import ants
    from syntx.motion import motion_correction
    ra = sys.modules["syntx.robust_affine"]            # syntx.robust_affine is the function
    seen = {}

    def spy(**kw):
        seen.update(kw)
        raise _Stop
    monkeypatch.setattr(ra, "robust_affine", spy)
    arr = np.random.default_rng(0).random((12, 12, 12, 3)).astype(np.float32)
    with pytest.raises(_Stop):
        motion_correction(ants.from_numpy(arr), backend="pytorch", type_of_transform="Rigid")
    assert "backend" not in seen and seen["mode"] == "auto"


def test_motion_rejects_ignored_options():
    import ants
    from syntx.motion import motion_correction
    img = ants.from_numpy(np.random.default_rng(0).random((12, 12, 12, 3)).astype(np.float32))
    with pytest.raises(ValueError, match="aff_metric"):
        motion_correction(img, backend="pytorch", aff_metric="mattes")
    with pytest.raises(TypeError, match="num_bins"):
        motion_correction(img, backend="pytorch_batched", sampling_percentage=0.3)


# ---------------------------------------------------------------------------------------
# template.py
# ---------------------------------------------------------------------------------------

def test_build_template_synonly_iterations_start_from_each_images_affine(monkeypatch, tmp_path):
    import ants
    from syntx.template import build_template
    calls = []
    real = ants.registration

    def spy(fixed, moving, type_of_transform="SyN", **kw):
        calls.append((type_of_transform, kw.get("initial_transform")))
        kw.setdefault("reg_iterations", (3, 0))
        out = real(fixed, moving, type_of_transform=type_of_transform, **kw)
        calls[-1] += (out["fwdtransforms"][-1],)
        return out
    monkeypatch.setattr(ants, "registration", spy)
    imgs = [ants.resample_image(ants.image_read(ants.get_data(k)), (32, 32), use_voxels=True)
            for k in ("r16", "r64")]
    build_template(image_list=imgs, iterations=2, type_of_transform="SyN", backend="ants",
                   output_dir=str(tmp_path), verbose=False)
    first, second = calls[:2], calls[2:]
    assert [c[0] for c in second] == ["SyNOnly", "SyNOnly"]
    for (t0, i0, aff0), (t1, i1, _) in zip(first, second):
        assert i0 is None and i1 == aff0                       # iteration-0 affine passed on


# ---------------------------------------------------------------------------------------
# syngs.py
# ---------------------------------------------------------------------------------------

def test_syngs_level_spacing_pairs_axes_correctly():
    from syntx.syngs import GeodesicShootingModel
    # tensor shape (z, y, x) = (10, 20, 40); ITK spacing (x, y, z) = (1, 2, 4)
    m = GeodesicShootingModel(dim=3, image_shape=(10, 20, 40), spacing=[1.0, 2.0, 4.0])
    disp = m.get_forward_warp(image_shape=(5, 10, 20))         # half resolution
    # the physical grid behind it must span the same box: x spacing (40-1)/(20-1) * 1, etc.
    from syntx.syngs import _level_spacing
    got = _level_spacing([1.0, 2.0, 4.0], (10, 20, 40), (5, 10, 20))
    assert np.allclose(got, [1.0 * 39 / 19, 2.0 * 19 / 9, 4.0 * 9 / 4])
    assert disp.shape == (1, 5, 10, 20, 3)


def test_integrate_momentum_trajectory_endpoint_matches_default():
    import ants
    from syntx.syngs import integrate_momentum
    rng = np.random.default_rng(0)
    ref = ants.from_numpy(np.zeros((20, 18), np.float32), spacing=(1.0, 1.5))
    v0 = ants.from_numpy((rng.standard_normal((20, 18, 2)) * 0.8).astype(np.float32),
                         spacing=(1.0, 1.5), has_components=True)
    end = integrate_momentum(v0, ref, n_steps=6, device="cpu").numpy()
    traj = integrate_momentum(v0, ref, n_steps=6, return_trajectory=True, device="cpu")
    assert len(traj) == 7
    assert np.allclose(traj[-1].numpy(), end, atol=1e-5)
    half = integrate_momentum(v0, ref, n_steps=6, t_end=0.5, device="cpu").numpy()
    assert np.abs(half).max() < np.abs(end).max()


def test_syngs_rejects_parameters_it_does_not_have():
    import ants
    import syntx
    img = ants.from_numpy(np.zeros((16, 16), np.float32))
    for kw in ({"total_sigma": 0.1}, {"multipoint_loss": [0.5]}, {"interpolator": "linear"},
               {"sobolev_alpha": 1.0}, {"type_of_transform": "SyNGS"}, {"not_a_param": 1}):
        with pytest.raises(TypeError):
            syntx.syngs(img, img, initial_transform="identity", device="cpu", **kw)


# ---------------------------------------------------------------------------------------
# syn.py
# ---------------------------------------------------------------------------------------

def _tiny_pair(n=24):
    import ants
    f = ants.resample_image(ants.image_read(ants.get_data("r16")), (n, n), use_voxels=True)
    m = ants.resample_image(ants.image_read(ants.get_data("r64")), (n, n), use_voxels=True)
    return f, m


def test_syn_rejects_tvf_syngs_only_parameters():
    import syntx
    f, m = _tiny_pair()
    for kw in ({"cfl_momentum": 0.9}, {"multipoint_loss": [0.5]}, {"n_time_steps": 4}, {"n_steps": 6}):
        with pytest.raises(TypeError, match=list(kw)[0]):
            syntx.syn(f, m, initial_transform="identity", device="cpu", reg_iterations=[1], **kw)


def test_syn_flow_sigma_warning_only_when_it_is_a_gate():
    import warnings
    import syntx
    f, m = _tiny_pair()
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # fast_smooth=False: flow_sigma is the post-filter
        syntx.syn(f, m, initial_transform="identity", device="cpu", reg_iterations=[1], flow_sigma=1.0)
    with pytest.warns(UserWarning, match="fast_smooth=True"):
        syntx.syn(f, m, initial_transform="identity", device="cpu", reg_iterations=[1], flow_sigma=1.0,
                  fast_smooth=True)


def test_auto_reg_metrics_use_true_determinant_and_real_inverse_errors():
    import syntx
    f, m = _tiny_pair(32)
    res = syntx.auto_reg(f, m, type_of_transform="SyN", diagnose=False, device="cpu",
                         reg_iterations=[2], initial_transform="identity")
    mt = res["metrics"]
    assert np.isfinite(mt["inverse_identity_mean_error"]) and np.isfinite(mt["inverse_identity_max_error"])
    assert mt["jac_measure"] == "finite difference of each half field"     # liouville for syn
    for k in ("jac_min", "jac_mean", "folding_pct", "fd_jac_min", "fd_folding_pct"):
        assert np.isfinite(mt[k]), k


def test_syn_jacobian_hinge_penalty_gets_itk_order_spacing(monkeypatch):
    import ants
    import syntx
    ssyn = sys.modules["syntx.syn"]
    seen = []
    real = ssyn.compute_jacobian_hinge_penalty

    def spy(w, physical_spacing=None, epsilon=0.05):
        seen.append(tuple(physical_spacing))
        return real(w, physical_spacing=physical_spacing, epsilon=epsilon)
    monkeypatch.setattr(ssyn, "compute_jacobian_hinge_penalty", spy)
    f = ants.from_numpy(np.random.default_rng(0).random((20, 12)).astype(np.float32), spacing=(1.0, 3.0))
    syntx.syn(f, f * 0.9, initial_transform="identity", device="cpu", reg_iterations=[1],
              jacobian_penalty_weight=0.1)
    assert len(seen) == 2 and all(np.allclose(sp, (1.0, 3.0)) for sp in seen)   # ITK order, fixed grid


def test_syn_unknown_keywords_raise_and_sigma_mode_is_gone():
    import inspect
    import syntx
    f, m = _tiny_pair()
    for kw in ({"syn_regularizer": "sobolev"}, {"epochs": [3]}, {"affine_epochs": [2]},
               {"sigma_mode": "physical"}, {"write_composite_transform": True}):
        with pytest.raises(TypeError):
            syntx.syn(f, m, initial_transform="identity", device="cpu", reg_iterations=[1], **kw)
    src = inspect.getsource(sys.modules["syntx.syn"].auto_reg)
    assert "'sigma_mode'" not in src                          # it was never read by syntx.syn


# ---------------------------------------------------------------------------------------
# tvf.py
# ---------------------------------------------------------------------------------------

def test_tvf_inverse_identity_weight_is_a_named_option():
    import syntx
    from syntx.tvf import TVF_ADVANCED_OPTIONS
    assert "inverse_identity_weight" in TVF_ADVANCED_OPTIONS
    f, m = _tiny_pair()
    r = syntx.tvf(f, m, initial_transform="identity", device="cpu", reg_iterations=[1],
                  multipoint_loss=[0.0, 1.0], inverse_identity_weight=0.2)
    assert r["model"].inverse_identity_weight == 0.2
    r = syntx.tvf(f, m, initial_transform="identity", device="cpu", reg_iterations=[1])
    assert r["model"].inverse_identity_weight == 0.05                 # default unchanged


# ---------------------------------------------------------------------------------------
# greedy.py
# ---------------------------------------------------------------------------------------

def test_greedy_unknown_metric_raises():
    import syntx
    f, m = _tiny_pair()
    with pytest.raises(ValueError, match="similarity_metric"):
        syntx.greedy(f, m, similarity_metric="invalid_metric", initial_transform=False, reg_iterations=[1], device="cpu")


# ---------------------------------------------------------------------------------------
# deformation_metrics.py
# ---------------------------------------------------------------------------------------

def test_bidirectional_dice_does_not_modify_input_labels(tmp_path):
    import ants
    from syntx.deformation_metrics import compute_bidirectional_dice
    arr = np.zeros((16, 16), np.float32); arr[4:12, 4:12] = 1
    fi = ants.from_numpy(arr, spacing=(2.0, 2.0), origin=(5.0, 5.0))
    fl = ants.from_numpy(arr.copy())                           # default geometry
    ml = ants.from_numpy(arr.copy())
    tx = ants.create_ants_transform(transform_type="AffineTransform", dimension=2)
    f = str(tmp_path / "id.mat"); ants.write_transform(tx, f)
    d = compute_bidirectional_dice(fl, ml, fi, fi, [f], [f])
    assert fl.spacing == (1.0, 1.0) and fl.origin == (0.0, 0.0)
    assert np.allclose(d, (1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------------------
# image_utils.py
# ---------------------------------------------------------------------------------------

def test_reflect_image_axis_is_physical_whatever_the_storage_order():
    import ants
    from syntx.image_utils import reflect_image
    arr = np.zeros((21, 21, 21), np.float32)
    arr[5:16, 5:16, 5:16] = 1.0                               # symmetric bulk fixes the centre
    arr[15, 10, 10] = 30.0                                    # marker off-centre along array axis 0
    D = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)   # array axis 0 = physical z (SI)
    img = ants.from_numpy(arr, direction=D)
    peak = lambda a: tuple(int(v) for v in np.unravel_index(np.argmax(a.numpy()), arr.shape))
    assert peak(reflect_image(img, "SI")) == (5, 10, 10)     # moved: SI is physical z
    assert peak(reflect_image(img, "LR")) == (15, 10, 10)    # not moved


def test_syn_optimizer_names_validated():
    """registration(optimizer_type=...) was silently overridden by optimizer=; an optimizer the
    PyTorch SyN has no update for (e.g. 'lbfgs') left the warps at zero."""
    import ants
    import numpy as np
    import syntx
    from syntx.syn import SyNTo
    f = ants.from_numpy(np.random.default_rng(0).random((16, 16)).astype('float32'))
    with pytest.raises(TypeError, match="optimizer="):
        syntx.syn(f, f, optimizer_type='adam', reg_iterations=[1])
    import torch
    m = SyNTo(dim=2, grid_shape=(16, 16))
    with pytest.raises(ValueError, match="optimizer"):
        m.fit(torch.rand(1, 1, 16, 16), torch.rand(1, 1, 16, 16), levels=[1], epochs_per_level=[1],
              optimizer_type='lbfgs')
    with pytest.raises(ValueError, match="optimizer"):
        syntx.syn(f, f, optimizer='lbfgs', reg_iterations=[1])
