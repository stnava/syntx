"""Regression tests for the docstring-audit behaviour issues in the misc modules
(cli, diagnose, classifier / resnet, features, policy, generators, perf_tracking,
provenance, tabulate, surface, landmarks). All CPU, tiny inputs, no downloads."""

import os
import sys
import json
import math

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _no_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


# ---------------------------------------------------------------------------------------
# cli.py
# ---------------------------------------------------------------------------------------

def _reg_args(*extra):
    from syntx import cli
    return cli.build_parser().parse_args(["register", "-f", "a.nii.gz", "-m", "b.nii.gz", *extra])


def test_cli_report_flag_toggles():
    assert _reg_args().report is True
    assert _reg_args("--no-report").report is False
    assert _reg_args("--no-report", "--report").report is True


def test_cli_register_uses_library_defaults():
    from syntx import cli
    args = _reg_args("--model", "syn")
    assert cli._model_kwargs(args) == {}
    args = _reg_args("--model", "syn", "--grad-step", "0.3", "--optimizer", "adam",
                     "--similarity-metric", "mattes_mi", "--regularizer", "bspline")
    assert cli._model_kwargs(args) == {"grad_step": 0.3, "optimizer": "adam",
                                       "syn_metric": "mattes_mi", "regularizer": "bspline"}


def test_cli_register_tvf_rejects_syn_only_options():
    from syntx import cli
    args = _reg_args("--model", "tvf", "--bootstrap-mode", "forward")
    with pytest.raises(ValueError, match="--bootstrap-mode"):
        cli._model_kwargs(args)
    args = _reg_args("--model", "affine", "--grad-step", "0.3")
    with pytest.raises(ValueError, match="--grad-step"):
        cli._model_kwargs(args)
    assert cli._model_kwargs(_reg_args("--model", "tvf", "--optimizer", "adam")) == {"optimizer": "adam"}


def test_cli_info_reports_package_version(capsys):
    import syntx
    from syntx import cli
    cli.cmd_info(None)
    out = capsys.readouterr().out
    assert f"Syntx Version   : {syntx.__version__}" in out
    assert "4.0.2" not in out


# ---------------------------------------------------------------------------------------
# perf_tracking.py
# ---------------------------------------------------------------------------------------

def test_perf_record_run_writes_standard_json_for_nonfinite(tmp_path):
    from syntx.perf_tracking import record_run
    log = tmp_path / "new" / "log.jsonl"                      # directory does not exist yet
    rec = record_run(str(log), "rsfmri", "ds", {"a": 1.0, "b": float("nan"), "c": float("inf")})
    line = log.read_text().strip()
    json.loads(line, parse_constant=lambda c: pytest.fail(f"non-standard JSON token {c}"))
    assert rec["metrics"] == {"a": 1.0, "b": None, "c": None}
    assert rec["nonfinite_metrics"] == ["b", "c"]


def test_perf_git_lookup_after_directory_creation(tmp_path, monkeypatch):
    import syntx.perf_tracking as pt
    seen = {}
    monkeypatch.setattr(pt, "_best_effort_git_commit", lambda p: seen.setdefault("exists", os.path.isdir(os.path.dirname(p))))
    pt.record_run(str(tmp_path / "d" / "log.jsonl"), "m", "d", {"a": 1.0})
    assert seen["exists"] is True


def test_perf_detect_regressions_nan_zero_baseline_and_strings(tmp_path):
    from syntx.perf_tracking import record_run, detect_regressions
    log = str(tmp_path / "log.jsonl")
    for v in (0.0, 0.0, 0.0):
        record_run(log, "m", "d", {"zero": v, "dice": 0.8, "s": 2.0})
    flags = {f.metric: f for f in detect_regressions(log, "m", "d", {"zero": 0.5, "dice": float("nan"), "s": "2.0"})}
    assert "zero" in flags                                    # baseline 0: absolute difference
    assert "dice" in flags and flags["dice"].direction == "nan"
    assert "s" not in flags                                   # "2.0" converted, unchanged


# ---------------------------------------------------------------------------------------
# tabulate.py
# ---------------------------------------------------------------------------------------

def test_correlation_matrix_constant_column_is_nan_like_numpy():
    from syntx.tabulate import correlation_matrix
    ts = np.stack([np.arange(6.0), np.ones(6), np.arange(6.0) ** 2], axis=1)
    c = correlation_matrix(ts)
    ref = np.corrcoef(ts, rowvar=False)
    assert np.isnan(c[1]).all() and np.isnan(c[:, 1]).all()
    ok = np.isfinite(ref)
    assert np.allclose(c[ok], ref[ok])


def test_correlation_matrix_mps_device_uses_float32(monkeypatch):
    from syntx import tabulate
    calls = {}
    real = torch.as_tensor

    def spy(x, dtype=None, device=None):
        calls["dtype"] = dtype
        return real(x, dtype=dtype)                          # run on CPU in the test
    monkeypatch.setattr(tabulate.torch, "as_tensor", spy)
    tabulate.correlation_matrix(np.random.default_rng(0).random((5, 3)), device="mps")
    assert calls["dtype"] == torch.float32


def _ts_and_labels(label_arr, spacing=(1.0, 1.0, 1.0)):
    import ants
    ts = ants.from_numpy(np.random.default_rng(0).random((4, 4, 4, 5)).astype(np.float32))
    lab = ants.from_numpy(label_arr.astype(np.float32), spacing=spacing)
    return ts, lab


def test_roi_mean_timeseries_rejects_fractional_labels_and_grid_mismatch():
    from syntx.tabulate import roi_mean_timeseries
    lab = np.zeros((4, 4, 4)); lab[:2] = 1.5
    with pytest.raises(ValueError, match="integer"):
        roi_mean_timeseries(*_ts_and_labels(lab))
    lab = np.zeros((4, 4, 4)); lab[:2] = 1
    with pytest.raises(ValueError, match="grid"):
        roi_mean_timeseries(*_ts_and_labels(lab, spacing=(2.0, 1.0, 1.0)))
    with pytest.raises(ValueError, match="grid"):
        roi_mean_timeseries(*_ts_and_labels(np.zeros((4, 4, 3))))
    m, ids = roi_mean_timeseries(*_ts_and_labels(lab))
    assert ids == [1] and m.shape == (5, 1)


# ---------------------------------------------------------------------------------------
# provenance.py
# ---------------------------------------------------------------------------------------

def test_capture_records_internal_and_nested_calls(monkeypatch):
    import sys as _sys
    import syntx.provenance as prov
    ra_mod = _sys.modules["syntx.robust_affine"]

    def fake_affine(fixed, moving, **kw):                   # stand-in, so no solver runs
        return {"fwdtransforms": []}
    monkeypatch.setattr(ra_mod, "robust_affine", fake_affine)
    monkeypatch.setattr(prov, "_ENTRY_POINTS", [("syntx.robust_affine", "robust_affine")])
    monkeypatch.setattr(prov, "_MODELS", [])

    def internal_caller():                                   # like syn.py's function-local import
        from syntx.robust_affine import robust_affine
        return robust_affine(1, 2, dof="rigid")
    with prov.capture_registration_calls() as outer:
        with prov.capture_registration_calls() as inner:
            internal_caller()
    assert [c["function"] for c in outer.calls] == ["syntx.robust_affine.robust_affine"]
    assert [c["function"] for c in inner.calls] == ["syntx.robust_affine.robust_affine"]
    assert inner.calls[0]["kwargs"] == {"dof": "rigid"}
    assert ra_mod.robust_affine is fake_affine                # restored


def test_capture_records_positional_fit_arguments(monkeypatch):
    import syntx.provenance as prov

    class Model:
        def fit(self, fixed, moving, levels=(4, 2, 1), epochs=3):
            return None

    class Mod:
        pass
    mod = Mod()
    mod.Model = Model

    def entry():
        Model().fit("F", "M", [2, 1], epochs=5)
    mod.entry = entry
    monkeypatch.setitem(__import__("sys").modules, "fake_reg_mod", mod)
    monkeypatch.setattr(prov, "_ENTRY_POINTS", [("fake_reg_mod", "entry")])
    monkeypatch.setattr(prov, "_MODELS", [("fake_reg_mod", "Model")])
    with prov.capture_registration_calls() as cap:
        mod.entry()
    res = cap.calls[0]["resolved"][0]
    assert res["fit_kwargs"] == {"epochs": 5}
    assert res["fit_args"] == {"fixed": "F", "moving": "M", "levels": [2, 1]}


def test_assert_manifest_complete_requires_diff_text_for_dirty_checkout():
    from syntx.provenance import assert_manifest_complete, REQUIRED_KEYS
    man = {k: None for k in REQUIRED_KEYS}
    man.update(calls=[], code={"git": {"commit": "a" * 40, "dirty": True, "diff_sha256": "x", "diff": None}})
    with pytest.raises(ValueError, match="no diff"):
        assert_manifest_complete(man, require_calls=False)
    man["code"]["git"]["diff"] = "diff --git ..."
    assert_manifest_complete(man, require_calls=False)


def test_environment_does_not_import_heavy_packages():
    import subprocess
    code = ("import sys, syntx.provenance as p; [sys.modules.pop(m, None) for m in ('jax', 'antstorch')];"
            "p.environment(); print('jax' in sys.modules, 'antstorch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out[-2:] == ["False", "False"]


def test_code_state_handles_untracked_paths_with_spaces(tmp_path):
    import subprocess
    from syntx.provenance import code_state
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "my file.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=a@b", "-c", "user.name=a",
                    "commit", "-q", "--allow-empty", "-m", "i"], check=True)
    st = code_state(root=str(tmp_path), pkg_dir=str(pkg))
    assert "pkg/my file.py" in st["git"]["untracked"]
    assert "pkg/my file.py" in st["git"]["untracked_package_files"]


# ---------------------------------------------------------------------------------------
# generators.py
# ---------------------------------------------------------------------------------------

def test_generator_ants_base_in_tensor_order_and_roundtrip():
    import ants
    from syntx.generators import CrossProductGenerator
    arr = np.zeros((12, 7), np.float32); arr[2, 5] = 1.0          # ANTs (x, y)
    img = ants.from_numpy(arr, spacing=(2.0, 0.5))
    g = CrossProductGenerator(base_image=img)
    assert g.base_tensor.shape[-2:] == (7, 12)                     # (H = y, W = x)
    back = g.to_ants_image(g.base_tensor)
    assert back.shape == (12, 7) and np.allclose(back.numpy(), arr)


def test_generator_rotation_is_rigid_on_anisotropic_grid():
    from syntx.generators import CrossProductGenerator
    g = CrossProductGenerator(base_image=np.zeros((9, 21), np.float32), spacing=(1.0, 3.0))
    _, _, u, _ = g.generate(None, "rotation", seed=3)
    H, W = 9, 21
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    def phys(nx, ny):                                              # normalised -> mm
        return torch.stack([nx * (W - 1) / 2 * 1.0, ny * (H - 1) / 2 * 3.0], dim=-1)
    p0 = phys(xs, ys)
    p1 = phys(xs + u[0, ..., 0], ys + u[0, ..., 1])
    d0 = torch.cdist(p0.reshape(-1, 2)[::17], p0.reshape(-1, 2)[::17])
    d1 = torch.cdist(p1.reshape(-1, 2)[::17], p1.reshape(-1, 2)[::17])
    assert torch.allclose(d0, d1, atol=1e-3)                       # distances preserved


def test_generator_unknown_magnitude_level_raises():
    from syntx.generators import CrossProductGenerator
    with pytest.raises(ValueError, match="magnitude_level"):
        CrossProductGenerator().generate(None, "translation", seed=1, magnitude_level="huge")


def test_temp_seed_restores_accelerator_rng(monkeypatch):
    from syntx import generators
    calls = []
    monkeypatch.setattr(generators.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(generators.torch.cuda, "get_rng_state_all", lambda: ["cuda-state"])
    monkeypatch.setattr(generators.torch.cuda, "set_rng_state_all", lambda s: calls.append(("cuda", s)))
    monkeypatch.setattr(generators.torch.cuda, "manual_seed_all", lambda s: None)
    with generators.temp_seed(5):
        pass
    assert ("cuda", ["cuda-state"]) in calls


# ---------------------------------------------------------------------------------------
# diagnose.py / classifier.py / resnet.py
# ---------------------------------------------------------------------------------------

def test_diagnose_records_why_tier2_did_not_run(monkeypatch):
    from syntx.diagnose import diagnose_image
    import ants
    vol = ants.from_numpy(np.random.default_rng(0).random((48, 48, 48)).astype(np.float32) * 100,
                          spacing=(2.0, 2.0, 2.0))                 # 96 mm: classifier-eligible
    d = diagnose_image(vol, fast=False)
    assert d.details["tier2"] == "weights_missing"
    assert diagnose_image(vol, fast=True).details["tier2"] == "skipped (fast=True)"


def test_diagnose_tier2_error_is_reported_not_swallowed(monkeypatch, tmp_path):
    import syntx.diagnose as dg
    import syntx.classifier as cl
    monkeypatch.setattr(dg.os.path, "exists", lambda p: True)
    monkeypatch.setattr(dg.torch, "load", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("corrupt")))
    import ants
    vol = ants.from_numpy(np.random.default_rng(0).random((48, 48, 48)).astype(np.float32) * 100,
                          spacing=(2.0, 2.0, 2.0))
    with pytest.warns(UserWarning, match="corrupt"):
        d = dg.diagnose_image(vol, fast=False)
    assert d.details["tier2"].startswith("error: RuntimeError")


def test_diagnose_signed_data_and_no_inert_field():
    import dataclasses
    from syntx.diagnose import diagnose_image, ImageDiagnosis
    z = np.random.default_rng(0).standard_normal((20, 20, 20)).astype(np.float32)   # z-scored MRI
    assert diagnose_image(z, fast=True).intensity_domain == "SIGNED_FLOAT"
    assert "is_contrast_enhanced" not in {f.name for f in dataclasses.fields(ImageDiagnosis)}


def test_diagnose_pair_default_matches_diagnose_image():
    import inspect
    from syntx.diagnose import diagnose_image, diagnose_pair
    assert inspect.signature(diagnose_pair).parameters["fast"].default == \
        inspect.signature(diagnose_image).parameters["fast"].default


def test_classifier_cpu_default_and_2d_error():
    import inspect
    from syntx.classifier import predict_diagnosis_deep, preprocess_volume_for_classifier
    assert inspect.signature(predict_diagnosis_deep).parameters["device"].default == "cpu"
    with pytest.raises(ValueError, match="3-D"):
        preprocess_volume_for_classifier(np.zeros((16, 16), np.float32))


def test_resnet10_has_no_inert_num_classes():
    import inspect
    from syntx.resnet import ResNet10
    assert "num_classes" not in inspect.signature(ResNet10).parameters


# ---------------------------------------------------------------------------------------
# features.py: weight loading
# ---------------------------------------------------------------------------------------

def test_medicalnet_style_state_dict_loads_into_resnet10(monkeypatch):
    from syntx.features import _load_state_dict_checked
    from syntx.resnet import resnet10_3d
    src = resnet10_3d()
    sd = {}
    for k, v in src.state_dict().items():                      # MedicalNet naming
        sd["module." + k.replace("shortcut", "downsample")] = v.clone() + 1.0
    dst = resnet10_3d()
    report = _load_state_dict_checked(dst, sd, rename={"downsample": "shortcut"})
    assert report["missing"] == [] and report["loaded"] == len(src.state_dict())
    k = "layer2.0.shortcut.0.weight"
    assert torch.equal(dst.state_dict()[k], src.state_dict()[k] + 1.0)


def test_state_dict_with_no_matching_keys_raises():
    from syntx.features import _load_state_dict_checked
    from syntx.resnet import resnet10_2d
    with pytest.raises(RuntimeError, match="no parameter"):
        _load_state_dict_checked(resnet10_2d(), {"foo.weight": torch.zeros(1)})


def test_swin_extractor_has_no_inert_img_size():
    import inspect
    from syntx.features import SwinUNETRExtractor
    assert "img_size" not in inspect.signature(SwinUNETRExtractor).parameters


# ---------------------------------------------------------------------------------------
# features.py: DINOv2 / FeatureSpaceLoss
# ---------------------------------------------------------------------------------------

class _FakeDino(torch.nn.Module):
    """Stand-in for a hub DINOv2: tokens = [cls, reg x n, patches]; blocks are identities."""
    def __init__(self, n_reg):
        super().__init__()
        self.num_register_tokens = n_reg
        self.blocks = torch.nn.ModuleList([torch.nn.Identity()])

    def prepare_tokens_with_masks(self, x):
        B, C, H, W = x.shape
        patches = x[:, :1, ::14, ::14].flatten(2).transpose(1, 2).repeat(1, 1, 4)  # (B, N, 4)
        extra = torch.full((B, 1 + self.num_register_tokens, 4), -7.0)
        return torch.cat([extra, patches], dim=1)


def _fake_dino_extractor(n_reg):
    from syntx.features import DINOv2Extractor
    ext = DINOv2Extractor.__new__(DINOv2Extractor)
    torch.nn.Module.__init__(ext)
    ext.model, ext.patch_size, ext.feature_layers = _FakeDino(n_reg), 14, [0]
    ext.register_buffer("mean", torch.zeros(1, 3, 1, 1))
    ext.register_buffer("std", torch.ones(1, 3, 1, 1))
    return ext


@pytest.mark.parametrize("n_reg", [0, 4])
def test_dinov2_drops_register_tokens(n_reg):
    ext = _fake_dino_extractor(n_reg)
    feat = ext.extract(torch.rand(1, 3, 28, 42))[0]
    assert feat.shape == (1, 4, 2, 3)
    assert (feat != -7.0).all()                                # no cls / register token leaked


def test_feature_space_loss_rejects_unknown_mode():
    from syntx.features import FeatureSpaceLoss
    with pytest.raises(ValueError, match="mode"):
        FeatureSpaceLoss(extractor=torch.nn.Identity(), mode="lncc3d")


class _Tiny2D:
    """Minimal 2-D extractor: returns [x, 2x] as two 'layers' (1 channel)."""
    is_3d, in_channels = False, 1

    def normalize(self, x):
        return x

    def extract(self, x):
        return [x, 2 * x]


def test_lncc_3d_uses_window_and_all_layers(monkeypatch):
    ssyn = sys.modules["syntx.syn"]                          # syntx.syn (attribute) is the function
    from syntx.features import FeatureSpaceLoss
    seen = []
    monkeypatch.setattr(ssyn, "local_ncc_loss_nd", lambda a, b, window_size: seen.append(window_size) or torch.tensor(0.0))
    loss = FeatureSpaceLoss(_Tiny2D(), mode="lncc_3d", lncc_window=7)
    loss(torch.rand(1, 1, 6, 6, 6), torch.rand(1, 1, 6, 6, 6))
    assert seen == [7] * 6                                     # 3 axes x 2 layers, window 7


def test_triplanar_3channel_edge_slices_not_empty():
    from syntx.features import FeatureSpaceLoss

    class Tiny3c(_Tiny2D):
        in_channels = 3
    loss = FeatureSpaceLoss(Tiny3c(), mode="triplanar", num_slices=4)
    out = loss(torch.rand(1, 1, 3, 9, 9), torch.rand(1, 1, 3, 9, 9))   # D // 4 == 0
    assert torch.isfinite(out)


def test_policy_has_no_unreachable_roi_branch():
    import inspect
    import syntx.policy as pol
    assert "is_roi_crop" not in inspect.getsource(pol)


def test_surface_full_grouping_keeps_all_eight_codes(monkeypatch):
    import ants
    import syntx.surface as surf
    codes = np.arange(10, dtype=np.float32).reshape(10, 1, 1).repeat(2, 1).repeat(2, 2)
    monkeypatch.setattr(surf.antstorch, "weingarten_image_curvature",
                        lambda *a, **k: ants.from_numpy(codes))
    img = ants.from_numpy(np.ones((10, 2, 2), np.float32))
    out = surf.compute_surface_classes(img, grouping="full", mask=img).numpy()
    assert sorted(np.unique(out).tolist()) == list(range(9))


# ---------------------------------------------------------------------------------------
# landmarks/preprocess.py
# ---------------------------------------------------------------------------------------

def test_is_ct_image_uses_hu_range_not_negative_fraction():
    from syntx.landmarks.preprocess import is_ct_image
    z = np.random.default_rng(0).standard_normal((10, 10, 10)).astype(np.float32)  # z-scored MRI
    assert not is_ct_image(z)
    ct = np.full((10, 10, 10), -1000.0, np.float32); ct[3:7] = 40.0; ct[0] = 700.0
    assert is_ct_image(ct)


def test_ct_without_window_keeps_negative_hu_contrast():
    import ants
    from syntx.landmarks.preprocess import preprocess_for_landmarks
    arr = np.full((12, 12, 12), -1000.0, np.float32)
    arr[2:10, 2:10, 2:10] = -100.0                            # fat
    arr[4:8, 4:8, 4:8] = 50.0                                  # soft tissue
    arr[0, 0, :] = 800.0                                       # bone
    out = preprocess_for_landmarks(ants.from_numpy(arr), use_denoise=False, is_ct=True).numpy()
    assert out[0, 5, 5] < out[3, 3, 3] < out[5, 5, 5]          # air < fat < soft tissue


def test_unknown_ct_window_raises():
    import ants
    from syntx.landmarks.preprocess import preprocess_for_landmarks
    with pytest.raises(ValueError, match="ct_window"):
        preprocess_for_landmarks(ants.from_numpy(np.zeros((8, 8, 8), np.float32)), is_ct=True,
                                 ct_window="brain", use_denoise=False)


# ---------------------------------------------------------------------------------------
# landmarks/sift2d.py
# ---------------------------------------------------------------------------------------

def test_normalize_slice_uint8_does_not_wrap():
    from syntx.landmarks.sift2d import _normalize_slice_uint8
    out = _normalize_slice_uint8(np.full((4, 4), 3.0, np.float32))      # no usable range, > 1
    assert out.max() == 255 and out.dtype == np.uint8


def _blobs_volume():
    import ants
    rng = np.random.default_rng(0)
    arr = np.zeros((40, 40, 40), np.float32)
    zz, yy, xx = np.mgrid[:40, :40, :40]
    for _ in range(25):
        c = rng.integers(6, 34, 3)
        arr += np.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2 + (zz - c[2]) ** 2) / (2 * 2.0 ** 2))
    return ants.from_numpy(arr / arr.max())


class _FakeKP:
    def __init__(self, pt, size, response):
        self.pt, self.size, self.response = pt, size, response


class _FakeSIFT:
    """One keypoint per slice at the slice centre; response = slice mean (so it varies)."""
    def detectAndCompute(self, img8, mask):
        h, w = img8.shape
        r = float(img8.mean()) + 1.0
        return [_FakeKP((w / 2.0, h / 2.0), 4.0, r)], np.full((1, 128), r, np.float32)


def _install_fake_cv2(monkeypatch):
    import types
    fake = types.ModuleType("cv2")
    fake.SIFT_create = lambda **kw: _FakeSIFT()
    monkeypatch.setitem(sys.modules, "cv2", fake)


def test_sift2d_interior_slices_planes_and_response_ranking(monkeypatch):
    _install_fake_cv2(monkeypatch)
    import syntx.landmarks.sift2d as s2
    coords, descs, planes = s2.detect_sift2d(_blobs_volume(), n_slices_per_axis=4, preprocess=False,
                                             min_distance_mm=0.0, return_planes=True)
    assert len(coords) == len(descs) == len(planes) > 0
    assert set(np.unique(planes)) <= {0, 1, 2}
    # slice positions are interior: no keypoint lies on the first / last slice of its axis
    for p, axis in ((0, 2), (1, 1), (2, 0)):
        fixed = coords[planes == p][:, axis]
        assert ((fixed > 0.5) & (fixed < 38.5)).all()
    # ranked by response: only sagittal slices (shape (DZ, DY) = (20, 30)) respond strongly
    import ants

    class Sag(_FakeSIFT):
        def detectAndCompute(self, img8, mask):
            kps, d = super().detectAndCompute(img8, mask)
            kps[0].response = 100.0 if img8.shape == (20, 30) else 1.0
            return kps, d
    sys.modules["cv2"].SIFT_create = lambda **kw: Sag()
    vol = ants.from_numpy(np.random.default_rng(1).random((40, 30, 20)).astype(np.float32))
    _, _, pl = s2.detect_sift2d(vol, n_slices_per_axis=3, preprocess=False, max_keypoints=1,
                                return_planes=True)
    assert pl.tolist() == [2]
    # default call keeps the two-output signature
    out = s2.detect_sift2d(_blobs_volume(), n_slices_per_axis=2, preprocess=False)
    assert len(out) == 2


def test_sift2d_run_slice_has_no_unused_flags():
    import inspect
    import syntx.landmarks.sift2d as s2
    assert "u_maps_ix" not in inspect.getsource(s2.detect_sift2d)


# ---------------------------------------------------------------------------------------
# landmarks/sift3d.py
# ---------------------------------------------------------------------------------------

def _grad_and_affine():
    g = torch.randn(1, 3, 12, 12, 12)
    return g, (np.zeros(3), np.ones(3), np.eye(3))


def test_sift3d_descriptor_empty_respects_return_frames():
    from syntx.landmarks.sift3d import _build_descriptor
    g, aff = _grad_and_affine()
    out = _build_descriptor(g, aff, np.zeros((0, 3)), np.zeros(0), return_frames=True)
    assert isinstance(out, tuple) and out[0].shape == (0, 512) and out[1].shape == (0, 3, 3)


def test_sift3d_descriptor_rejects_conflicting_and_unknown_options():
    from syntx.landmarks.sift3d import _build_descriptor
    g, aff = _grad_and_affine()
    kp, sg = np.array([[6.0, 6.0, 6.0]]), np.array([1.0])
    with pytest.raises(ValueError, match="frame_rotation"):
        _build_descriptor(g, aff, kp, sg, rotation_invariant=True, frame_rotation=np.eye(3))
    with pytest.raises(ValueError, match="sign_mode"):
        _build_descriptor(g, aff, kp, sg, rotation_invariant=True, sign_mode="mean")
    import inspect
    assert "min_anisotropy" not in inspect.signature(_build_descriptor).parameters
    d = _build_descriptor(g, aff, kp, sg)
    assert d.shape == (1, 512) and np.isfinite(d).all()
