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
