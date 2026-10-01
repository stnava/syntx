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


# ---------------------------------------------------------------------------------------
# landmarks/mind.py
# ---------------------------------------------------------------------------------------

def test_mind_validates_patch_size_offsets_and_reused_volume():
    import ants
    from syntx.landmarks.mind import compute_mind, extract_mind_at_points
    img = ants.from_numpy(np.random.default_rng(0).random((10, 10, 10)).astype(np.float32))
    with pytest.raises(ValueError, match="odd"):
        compute_mind(img, patch_size=4, device="cpu")
    with pytest.raises(ValueError, match="1 .. 26"):
        compute_mind(img, n_offsets=30, device="cpu")
    vol = compute_mind(img, n_offsets=6, device="cpu")
    assert vol.shape == (1, 6, 10, 10, 10)
    pts = np.array([[5.0, 5.0, 5.0]])
    with pytest.raises(ValueError, match="channels"):
        extract_mind_at_points(img, pts, n_offsets=12, mind_vol=vol)
    other = ants.from_numpy(np.zeros((8, 10, 10), np.float32))
    with pytest.raises(ValueError, match="grid"):
        extract_mind_at_points(other, pts, n_offsets=6, mind_vol=vol)
    assert extract_mind_at_points(img, pts, n_offsets=6, mind_vol=vol).shape == (1, 6)


# ---------------------------------------------------------------------------------------
# landmarks/matcher.py
# ---------------------------------------------------------------------------------------

def test_match_landmarks_validation_and_single_candidate():
    from syntx.landmarks.matcher import match_landmarks
    k = np.zeros((3, 3)); d = np.eye(3, 8)
    with pytest.raises(ValueError, match="row-aligned"):
        match_landmarks(k[:2], k, d, d, device="cpu")
    with pytest.raises(ValueError, match="metric"):
        match_landmarks(k, k, d, d, metric="hamming", device="cpu")
    out = match_landmarks(k, k[:1], d, d[:1], device="cpu")           # one destination: no ratio test
    assert out.shape == (0, 2)


def test_ransac_returns_nothing_unverified():
    from syntx.landmarks.matcher import ransac_filter
    rng = np.random.default_rng(0)
    src = rng.random((3, 3)) * 50
    m = np.array([[0, 0], [1, 1], [2, 2]])
    inl, M = ransac_filter(src, src + 1, m, min_inliers=4)          # too few matches
    assert inl.shape == (0, 2) and np.allclose(M, np.eye(4))
    with pytest.raises(ValueError, match="model"):
        ransac_filter(src, src, m, model="similarity")
    # random correspondences: no 6-point consensus
    a, b = rng.random((10, 3)) * 100, rng.random((10, 3)) * 100
    mm = np.stack([np.arange(10), np.arange(10)], 1)
    inl, M = ransac_filter(a, b, mm, model="rigid", inlier_thresh_mm=0.5, min_inliers=6)
    assert inl.shape == (0, 2)
    # a true rigid shift is verified
    inl, M = ransac_filter(a, a + [3.0, -2.0, 1.0], mm, model="rigid", inlier_thresh_mm=0.5)
    assert len(inl) == 10 and np.allclose(M[:3, 3], [3, -2, 1], atol=1e-3)


def test_compute_tre_checks_counts():
    from syntx.landmarks.matcher import compute_tre
    with pytest.raises(ValueError):
        compute_tre(np.zeros((3, 3)), np.zeros((2, 3)))


# ---------------------------------------------------------------------------------------
# landmarks/orient.py
# ---------------------------------------------------------------------------------------

def test_rotation_search_rejects_frame_options_and_early_exit_has_winner():
    import ants
    from syntx.landmarks.orient import match_sift3d_with_rotation_search
    flat = ants.from_numpy(np.zeros((12, 12, 12), np.float32))
    with pytest.raises(ValueError, match="rotation_invariant"):
        match_sift3d_with_rotation_search(flat, flat, rotation_invariant=True, device="cpu")
    res = match_sift3d_with_rotation_search(flat, flat, device="cpu")     # no keypoints
    assert res["winner"] is None and res["matches"].shape == (0, 2)


def test_refine_returns_descriptors_in_returned_frame(monkeypatch):
    import syntx.landmarks.orient as orient
    built = []

    def fake_desc(state, frame_rotation=None, **kw):
        built.append(None if frame_rotation is None else np.asarray(frame_rotation).copy())
        return np.full((len(state["pts"]), 4), len(built), np.float32)
    rng = np.random.default_rng(0)
    pts = np.c_[rng.random((20, 3)) * 50, np.ones(20)]
    th = np.deg2rad(30)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    monkeypatch.setattr(orient, "sift3d_descriptors", fake_desc)
    monkeypatch.setattr(orient, "match_landmarks", lambda *a, **k: np.stack([np.arange(20)] * 2, 1).astype(np.int32))
    monkeypatch.setattr(orient, "ransac_filter", lambda cf, cm, m, **k: (m, np.r_[np.c_[Rz, np.zeros(3)], [[0, 0, 0, 1]]].astype(np.float32)))
    sf = {"pts": pts, "response": np.ones(20)}
    out = orient.refine_rotation_iteratively(sf, dict(sf), np.zeros((20, 4)), None, n_iter=1,
                                             coarse_to_fine=(1.0,))
    assert np.allclose(out["R"], Rz, atol=1e-5)
    assert built[-1] is not None and np.allclose(built[-1], Rz, atol=1e-5)    # rebuilt in final R
    assert out["descs_moving"][0, 0] == len(built)


# ---------------------------------------------------------------------------------------
# landmarks/optimal_transport.py
# ---------------------------------------------------------------------------------------

def test_weighted_procrustes_scale_with_reflection_sign():
    from syntx.landmarks.optimal_transport import weighted_procrustes
    rng = np.random.default_rng(0)
    src_np = rng.standard_normal((40, 3))
    dst_np = 2.0 * src_np * np.array([1.0, 1.0, -1.0]) + 0.01 * rng.standard_normal((40, 3))  # mirrored
    R, t, scale, A = weighted_procrustes(torch.tensor(src_np), torch.tensor(dst_np), allow_scaling=True)
    # reference Umeyama (1991): scale = trace(D S) / (n var_src), D = diag(1, 1, sign det(U V^T))
    sc, dc = src_np - src_np.mean(0), dst_np - dst_np.mean(0)
    U, S, Vt = np.linalg.svd(sc.T @ dc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    ref = (S[0] + S[1] + d * S[2]) / (sc ** 2).sum()
    assert np.linalg.det(R.numpy()) > 0
    assert abs(scale - ref) < 1e-6


def _ot_images(fg_voxels=20):
    import ants
    a = np.zeros((10, 10, 10), np.float32)
    a.reshape(-1)[:fg_voxels] = 100.0                          # raw (unnormalised) intensities
    return ants.from_numpy(a), ants.from_numpy(a.copy())


def test_ot_small_foreground_does_not_raise_and_keys_are_consistent():
    from syntx.landmarks.optimal_transport import sampled_optimal_transport_affine
    fi, mi = _ot_images(fg_voxels=60)
    out = sampled_optimal_transport_affine(fi, mi, min_samples=500, device="cpu", return_dict=True,
                                           n_sinkhorn_iters=3)
    assert out["status"] == "OK" and out["n_samples_fixed"] == 60
    empty = sampled_optimal_transport_affine(*_ot_images(fg_voxels=0), device="cpu", return_dict=True)
    assert empty["status"] == "EMPTY_FG" and set(empty) == set(out)
    import os
    os.remove(out["transform_path"]); os.remove(empty["transform_path"])


def test_score_rotation_intensity_is_rotation_sensitive_and_validates_type():
    import ants
    from syntx.landmarks.optimal_transport import score_rotation_candidates_sampled
    rng = np.random.default_rng(0)
    arr = np.zeros((16, 16, 16), np.float32)
    arr[3:13, 3:13, 3:13] = rng.random((10, 10, 10)) + 0.5
    arr[3:8, 3:13, 3:13] += 2.0                                 # asymmetric
    img = ants.from_numpy(arr)
    I = np.eye(3); R180 = np.diag([-1.0, -1.0, 1.0])
    with pytest.raises(ValueError, match="feature_type"):
        score_rotation_candidates_sampled(img, img, [I], feature_type="hog", device="cpu")
    res = score_rotation_candidates_sampled(img, img, [R180, I], feature_type="intensity", device="cpu")
    assert res[0]["index"] == 1 and res[0]["score"] > res[1]["score"] + 0.2


def test_landmarks_spatial_geometry_errors():
    from syntx.landmarks.spatial import get_image_affine, ortho_view_spec
    assert np.allclose(get_image_affine(np.zeros((4, 4, 4)))[1], 1.0)
    with pytest.raises(TypeError):
        get_image_affine("image.nii.gz")
    geom = (np.zeros(3), np.ones(3), np.eye(3))
    with pytest.raises(ValueError, match="center_mm"):
        ortho_view_spec(geom, "axial")
    spec = ortho_view_spec(geom, "axial", center_mm=np.zeros(3))
    assert isinstance(spec, dict)


def test_batched_motion_passes_reject_frames_off_the_reference_grid():
    import ants
    import numpy as np
    from syntx.motion_batched import batched_rigid_register_pass, batched_group_bias_register_pass
    ref = ants.from_numpy(np.random.default_rng(0).random((12, 12, 12)).astype('float32'))
    off = ants.from_numpy(ref.numpy(), spacing=(1.0, 1.0, 2.0))
    with pytest.raises(ValueError, match="reference grid"):
        batched_rigid_register_pass(ref, [ref, off], device='cpu')
    with pytest.raises(ValueError, match="reference grid"):
        batched_group_bias_register_pass(ref, [ref, off], [False, True], device='cpu')


def _write_tar(path, members):
    import io
    import tarfile
    with tarfile.open(path, "w") as tar:
        for name, data in members:
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))


def test_msd_download_extract_is_safe(tmp_path, monkeypatch):
    from syntx.data import msd
    task = msd.get_msd_task_info("Task09")
    tgt = tmp_path / "data"
    tgt.mkdir()
    _write_tar(tgt / task.archive_name, [("../escape.txt", b"x")])
    with pytest.raises(Exception):                 # tarfile 'data' filter or the explicit check
        msd.download_msd_task("Task09", str(tgt), progress_bar=False)
    assert not (tmp_path / "escape.txt").exists()
    (tgt / task.archive_name).unlink()
    _write_tar(tgt / task.archive_name, [(f"{task.name}/readme.txt", b"no dataset.json")])
    with pytest.raises(FileNotFoundError, match="dataset.json"):
        msd.download_msd_task("Task09", str(tgt), progress_bar=False)


def test_msd_dataset_labels_transform_and_4d(tmp_path):
    import json
    import ants
    import numpy as np
    from syntx.data.msd import MSDDataset
    td = tmp_path / "Task01_BrainTumour"
    (td / "imagesTr").mkdir(parents=True)
    (td / "labelsTr").mkdir()
    img = np.random.default_rng(0).random((10, 12, 8, 4)).astype('float32')
    ants.image_write(ants.from_numpy(img, has_components=False), str(td / "imagesTr" / "a.nii.gz"))
    lab = np.zeros((10, 12, 8), 'float32')
    lab[3:6, 4:8, 2:5] = 2
    ants.image_write(ants.from_numpy(lab), str(td / "labelsTr" / "a.nii.gz"))
    json.dump({"training": [{"image": "./imagesTr/a.nii.gz", "label": "./labelsTr/a.nii.gz"}]},
              open(td / "dataset.json", "w"))
    ds = MSDDataset([str(td)], target_shape=(6, 6, 6), transform=lambda s: {**s, "seen": True})
    s = ds[0]
    assert s["seen"] and s["image"].shape == (1, 6, 6, 6) and s["label"].shape == (1, 6, 6, 6)
    assert set(np.unique(s["label"].numpy())) <= {0.0, 2.0}                 # NN for labels


def test_scattered_result_methods_read_xyz_components_in_3d():
    """The solver's fields have (x, y, z) components; the result's helpers read them that way
    (warp_scattered_coordinates' 3-D default assumes (z, y, x) and flipped them)."""
    import torch
    from syntx.scattered.solver import ScatteredRegistrationResult
    from syntx.scattered.mapping import warp_scattered_coordinates
    g = (6, 7, 8)
    u = torch.zeros(1, *g, 3)
    u[..., 0] = 0.2                                  # pure x shift in [-1, 1] units
    z = torch.zeros_like(u)
    res = ScatteredRegistrationResult(disp_fwd=u, disp_inv=u, warp_l2r=z, warp_r2l=z, warp_l2r_inv=z,
                                      warp_r2l_inv=z, fixed_grid=z, moving_grid=z,
                                      warped_moving_grid=z, warped_fixed_grid=z, grid_shape=g)
    pts = torch.tensor([[0.1, -0.2, 0.3], [-0.4, 0.0, 0.2]])
    out = res.warp_points(pts)
    np.testing.assert_allclose(out.numpy(), (pts + torch.tensor([0.2, 0.0, 0.0])).numpy(), atol=1e-5)
    ref = warp_scattered_coordinates(pts, u, direction='forward', vector_convention='xyz')
    np.testing.assert_allclose(out.numpy(), ref.numpy(), atol=1e-6)
    feats = torch.tensor([[1.0], [2.0]])
    moved = res.transport_features(pts, feats, pts + torch.tensor([0.2, 0.0, 0.0]))
    np.testing.assert_allclose(moved.reshape(feats.shape).numpy(), feats.numpy(), atol=0.05)


def test_scattered_config_fails_loudly_and_inverse_method_is_used(monkeypatch):
    import torch
    from syntx.scattered.solver import ScatteredRegistrationConfig, SyNScattered
    import syntx.scattered.solver as sol
    for bad in (dict(regularizer='tv'), dict(optimizer_type='lbfgs'), dict(similarity_metric='mi'),
                dict(inverse_method='hybrid'), dict(regularizer='bspline', mesh_size=(4, 4))):
        with pytest.raises(ValueError):
            ScatteredRegistrationConfig(**bad)
    with pytest.raises(TypeError):
        ScatteredRegistrationConfig(w_distortion=0.1)
    calls = []
    monkeypatch.setattr(sol, "update_inverse_field_nd", lambda *a, **k: calls.append(k.get('method')) or a[1] if a[1] is not None else -a[0])
    cfg = ScatteredRegistrationConfig(dim=2, grid_res=16, iterations=2, inverse_method='fixed_point',
                                      inverse_steps=3, initial_transform=False)
    pts = torch.rand(40, 2) * 1.6 - 0.8
    feats = torch.ones(40, 1)
    SyNScattered(config=cfg).fit(pts, feats, pts + 0.02, feats)
    assert calls and set(calls) == {'fixed_point'}


def test_scattered_cfl_spacing_and_coordinate_voxel_size():
    """The CFL voxel normalisation pairs the (x, y, z) components with the reversed shape, and
    the projection floor uses the box in coordinate units."""
    from syntx.scattered.solver import ScatteredRegistrationConfig, SyNScattered
    m = SyNScattered(config=ScatteredRegistrationConfig(dim=2, domain_bounds=((0.0, 0.0), (10.0, 4.0))))
    # x spans 10 units over 21 nodes (0.5), y spans 4 over 5 (1.0): tensor shape (5, 21)
    assert m._coordinate_voxel_size((5, 21), []) == 1.0
    m2 = SyNScattered(config=ScatteredRegistrationConfig(dim=2))
    assert abs(m2._coordinate_voxel_size((32, 32), []) - 2.0 / 31) < 1e-12


def test_syn_scattered_wrapper_and_energies():
    import torch
    from syntx.scattered.solver import ScatteredRegistrationConfig, syn_scattered, _deformation_energies
    pts = torch.rand(40, 2) * 1.6 - 0.8
    feats = torch.ones(40, 1)
    cfg = ScatteredRegistrationConfig(dim=2, grid_res=16, iterations=1, initial_transform=False)
    syn_scattered(pts, feats, pts + 0.01, feats, config=cfg, iterations=2)
    assert cfg.iterations == 1                       # not mutated
    with pytest.raises(TypeError):
        syn_scattered(pts, feats, pts, feats, config=cfg, epochs=3)
    with pytest.raises(ValueError, match="dim"):
        syn_scattered(fixed_grid=torch.rand(8, 8, 8), moving_grid=torch.rand(8, 8, 8))
    # harmonic energy of u = (a x, 0) on [-1, 1]^2 is a^2 (one non-zero derivative)
    n = 17
    xs = torch.linspace(-1, 1, n)
    u = torch.zeros(1, n, n, 2)
    u[..., 0] = 0.3 * xs.view(1, 1, n)
    e = _deformation_energies(u)
    assert abs(e['harmonic'] - 0.09) < 1e-5 and abs(e['l2'] - float(torch.mean(u ** 2))) < 1e-7


def test_adaptive_sigma_deterministic_and_scale_covariant():
    import torch
    from syntx.scattered.projection import compute_adaptive_sigma
    g = torch.Generator().manual_seed(0)
    pts = torch.rand(2600, 3, generator=g) * 2 - 1
    assert compute_adaptive_sigma(pts) == compute_adaptive_sigma(pts)         # was randperm
    # mm coordinates: sigma scales with the cloud instead of pinning at the 0.2 cap
    a = compute_adaptive_sigma(pts[:500])
    b = compute_adaptive_sigma(pts[:500] * 100.0)
    assert abs(b / a - 100.0) < 1e-3


def test_scattered_projector_static_3d_and_options():
    import torch
    from syntx.scattered.projection import ScatteredProjector, ProjectionConfig, project_scattered_to_grid
    pts = torch.rand(30, 3) * 1.6 - 0.8
    vals = torch.rand(30, 1)
    proj = ScatteredProjector(grid_shape=12, domain_bounds=(-1.0, 1.0), sigma=0.2)
    out = proj(pts, vals)                                   # 3-D points with an int grid_shape
    ref = project_scattered_to_grid(pts, vals, grid_shape=12, domain_bounds=(-1.0, 1.0), sigma=0.2)
    assert out.shape == ref.shape == (1, 1, 12, 12, 12)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="sigma"):
        ScatteredProjector(grid_shape=12, domain_bounds=(-1.0, 1.0), sigma=0.0)
    # sigma_scale reaches the non-static path (sigma='auto' with one point)
    cfg_a = ProjectionConfig(grid_shape=16, sigma='auto', sigma_scale=1.5)
    cfg_b = ProjectionConfig(grid_shape=16, sigma='auto', sigma_scale=4.0)
    one = torch.tensor([[0.1, -0.2]])
    a = ScatteredProjector(config=cfg_a)(one, torch.ones(1, 1))
    b = ScatteredProjector(config=cfg_b)(one, torch.ones(1, 1))
    assert not torch.allclose(a, b)


def test_bspline_projection_zyx_on_a_non_cubic_grid():
    """'zyx' points (component 0 along tensor axis 0) on a non-cubic grid give the same grid as
    the same points in 'xyz' order (the ITK size was not reversed for 'zyx')."""
    import torch
    from syntx.scattered.projection import project_scattered_to_grid, has_antstorch
    if not has_antstorch():
        pytest.skip("antstorch")
    g = torch.Generator().manual_seed(0)
    pts_xyz = torch.rand(80, 2, generator=g) * 1.6 - 0.8
    vals = (pts_xyz[:, :1] * 2 + pts_xyz[:, 1:] ** 2)
    kw = dict(grid_shape=(10, 18), domain_bounds=(-1.0, 1.0), method='bspline', mesh_size=2)
    a = project_scattered_to_grid(pts_xyz, vals, coord_convention='xyz', **kw)
    b = project_scattered_to_grid(pts_xyz.flip(-1), vals, coord_convention='zyx', **kw)
    assert a.shape == b.shape == (1, 1, 10, 18)
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


def test_scattered_auto_bounds_shared_by_fixed_moving_and_warps():
    """'auto' bounds are one box for the whole registration (each projection / warp derived its
    own from its own point set)."""
    import torch
    from syntx.scattered.solver import ScatteredRegistrationConfig, SyNScattered
    g = torch.Generator().manual_seed(0)
    pts_f = torch.rand(50, 2, generator=g) * 4.0 + 10.0          # [10, 14]^2
    pts_m = pts_f + torch.tensor([3.0, 0.0])                     # shifted out of the fixed box
    feats = torch.ones(50, 1)
    m = SyNScattered(ScatteredRegistrationConfig(dim=2, grid_res=16, domain_bounds='auto', iterations=1,
                                                 initial_transform=False))
    res = m.fit(pts_f, feats, pts_m, feats)
    lo, hi = res.domain_bounds
    both = torch.cat([pts_f, pts_m])
    assert abs(lo[0] - (float(both[:, 0].min()) - 0.09)) < 1e-4       # both clouds, 3 sigma margin
    assert abs(hi[0] - (float(both[:, 0].max()) + 0.09)) < 1e-4
    # zero warp: moving the points with the result keeps them in place (same box everywhere)
    z = torch.zeros_like(res.disp_inv)
    res.disp_inv = z
    torch.testing.assert_close(res.warp_points(pts_m), pts_m, atol=1e-4, rtol=0)


def test_warp_scattered_auto_invert_physical_and_options():
    import torch
    import ants
    from syntx.scattered.mapping import warp_scattered_coordinates
    n = 41
    xs = torch.linspace(0.0, 20.0, n)
    u = torch.zeros(1, n, n, 2)
    u[..., 1] = 0.2 * (xs.view(1, 1, n) - 10.0)        # x displacement (tensor-order comps: (y, x))
    u.is_physical = True
    u.vector_convention = 'zyx'
    pts = torch.tensor([[6.0, 9.0], [12.0, 4.0], [10.5, 15.0]])
    bounds = ((0.0, 0.0), (20.0, 20.0))
    fwd = warp_scattered_coordinates(pts, u, domain_bounds=bounds)
    back = warp_scattered_coordinates(fwd, u, direction='inverse', auto_invert=True, inversion_steps=60,
                                      domain_bounds=bounds)
    torch.testing.assert_close(back, pts, atol=0.05, rtol=0)
    with pytest.raises(ValueError, match="auto"):
        warp_scattered_coordinates(pts, u, domain_bounds='auto')
    with pytest.raises(ValueError, match="contradicts"):
        warp_scattered_coordinates(pts, u, domain_bounds=bounds, scale_displacement=True, is_physical=True)
    # ANTs image: the box comes from its geometry
    arr = np.zeros((n, n, 2), dtype='float32')
    arr[..., 0] = 1.5
    img = ants.from_numpy(arr, origin=(0.0, 0.0), spacing=(0.5, 0.5), has_components=True)
    out = warp_scattered_coordinates(pts, img)
    torch.testing.assert_close(out, pts + torch.tensor([1.5, 0.0]), atol=1e-4, rtol=0)


def test_scattered_warper_caches_its_inverse(monkeypatch):
    import torch
    import syntx.scattered.mapping as mp
    calls = []
    real = mp._invert_displacement
    monkeypatch.setattr(mp, "_invert_displacement", lambda *a, **k: calls.append(1) or real(*a, **k))
    u = torch.zeros(1, 9, 9, 2)
    u[..., 0] = 0.1
    w = mp.ScatteredWarper(u, domain_bounds=(-1.0, 1.0), vector_convention='xyz')
    pts = torch.tensor([[0.0, 0.0], [0.2, -0.3]])
    a = w.inverse(pts)
    b = w.inverse(pts)
    assert len(calls) == 1 and torch.equal(a, b)
    torch.testing.assert_close(a, pts - torch.tensor([0.1, 0.0]), atol=1e-3, rtol=0)
    w.displacement_field.add_(0.0)            # in-place change -> recomputed
    w.inverse(pts)
    assert len(calls) == 2


def test_transport_backward_grid_bridge_box_and_density():
    import torch
    from syntx.scattered.transport import transport_scattered_to_scattered, pushforward_scattered_to_grid
    n = 17
    u = torch.zeros(1, n, n, 2)
    u[..., 0] = 0.2                                              # +0.2 in x, [-1, 1] units
    src = torch.tensor([[0.0, 0.0], [0.3, -0.2], [-0.4, 0.5]])
    feats = torch.tensor([[1.0], [2.0], [3.0]])
    kw = dict(domain_bounds=(-1.0, 1.0), vector_convention='xyz', sigma=0.02)
    fwd = transport_scattered_to_scattered(src, feats, src + torch.tensor([0.2, 0.0]), u, **kw)
    bwd = transport_scattered_to_scattered(src, feats, src - torch.tensor([0.2, 0.0]), u, direction='backward', **kw)
    torch.testing.assert_close(fwd.reshape(-1), feats.reshape(-1), atol=0.05, rtol=0)
    torch.testing.assert_close(bwd.reshape(-1), feats.reshape(-1), atol=0.05, rtol=0)  # was = forward
    # grid bridge in mm-like coordinates: one box for both halves; density available
    big = src * 50 + 100
    out, den = transport_scattered_to_scattered(big, feats, big, None, sigma=2.0, method='grid_bridge',
                                                grid_shape=64, return_density=True)
    assert den is not None and torch.isfinite(out).all()
    torch.testing.assert_close(out.reshape(-1), feats.reshape(-1), atol=0.35, rtol=0)
    g = pushforward_scattered_to_grid(src.numpy(), feats.numpy(), grid_shape=8)          # NumPy input
    assert g.shape == (1, 1, 8, 8)


def test_scattered_levels_ascending_factors_rejected():
    import torch
    from syntx.scattered.solver import ScatteredRegistrationConfig, SyNScattered
    pts = torch.rand(30, 2) * 1.6 - 0.8
    with pytest.raises(ValueError, match="non-increasing"):
        SyNScattered(ScatteredRegistrationConfig(dim=2, grid_res=16, levels=[1, 2], iterations=1,
                                                 initial_transform=False)).fit(pts, torch.ones(30, 1), pts, torch.ones(30, 1))


def test_registration_report_scores_and_missing_values(tmp_path):
    import ants
    from syntx.viz.reports import create_registration_report
    a = ants.from_numpy(np.random.default_rng(0).random((24, 24)).astype('float32') + 0.1)
    rep = create_registration_report(a, a, warped=a, output_html=str(tmp_path / "r.html"))
    m = rep["metrics"]
    assert abs(m["SSIM"] - 1.0) < 1e-5 and abs(m["NCC"] - 1.0) < 1e-5 and abs(m["LNCC (w=9)"] - 1.0) < 1e-5
    assert np.isnan(rep["inverse_error"]["max"]) and np.isnan(rep["jacobian"]["folding_pct"])   # not 0
    html_txt = open(rep["html_path"]).read()
    assert "Verified Provenance" not in html_txt and "No run provenance" in html_txt and "n/a" in html_txt
    with pytest.raises(TypeError):
        create_registration_report(a, a, output_html=str(tmp_path / "r2.html"), bogus=1)
    rep2 = create_registration_report(a, a, output_html=str(tmp_path / "r3.html"), dice_overlap=0.7)
    assert rep2["dice"] == 0.7


def test_report_jacobian_stats_tensor_layout_and_provenance():
    import ants
    import torch
    import syntx
    from syntx.viz.reports import _compute_jacobian_stats, build_engine_provenance
    from syntx.spatial import get_physical_grid_torch
    fixed = ants.from_numpy(np.zeros((14, 10), dtype='float32'), spacing=(1.5, 0.75))     # ANTs (x, y)
    X = get_physical_grid_torch((10, 14), (1.5, 0.75), (0.0, 0.0), np.eye(2))           # (1, y, x, 2) tensor order
    u = torch.zeros_like(X)
    u[..., 1] = 0.2 * X[..., 1]                    # x component (tensor order (y, x)): u_x = 0.2 x
    det, st = _compute_jacobian_stats(u, fixed)
    assert det.shape == (14, 10)
    assert abs(st["mean"] - 1.2) < 1e-3 and st["folding_pct"] == 0.0
    prov = build_engine_provenance()
    assert prov["syntx_version"] == syntx.__version__ and prov["antisymmetric"] == "N/A"


def test_benchmark_report_means_paired_and_missing(tmp_path):
    from syntx.viz.reports import create_benchmark_report
    syn = {0: {'dice_sym': 0.8}, 1: {'dice_sym': 0.6}, 2: {'dice_sym': 0.9}, 3: {}}   # 3: missing
    ants_r = {0: {'dice_sym': 0.7}, 1: {'dice_sym': 0.5}}
    out = create_benchmark_report(syn, ants_r, 4, output_html=str(tmp_path / "sub" / "b.html"))
    txt = open(out).read()
    assert "0.7000" in txt          # syntx mean over the paired pairs 0, 1 (not 0.575 with a 0.0)
    assert "0.6000" in txt          # ANTs mean


def test_display_displacement_units_and_axes():
    """+2 mm along physical x on a 2 mm-x-spacing grid is one display pixel along the axis that
    shows x, zero along the other, in the axial view; the coronal / sagittal views agree."""
    import ants
    from syntx.viz.figures import _display_displacement
    shape = (12, 10, 8)
    fixed = ants.from_numpy(np.random.default_rng(0).random(shape).astype('float32'), spacing=(2.0, 1.0, 1.5))
    u = np.zeros(shape + (3,), dtype='float32')
    u[..., 0] = 2.0
    warp = ants.from_numpy(u, spacing=fixed.spacing, origin=fixed.origin, direction=fixed.direction, has_components=True)
    dcol, drow, bg, _, mag = _display_displacement(warp, fixed, 2, None, True)
    inner = (slice(2, -2), slice(2, -2))
    moved = np.stack([dcol[inner], drow[inner]], -1)
    assert np.allclose(np.abs(moved).max(axis=(0, 1)).max(), 1.0, atol=1e-4)      # one pixel
    assert np.allclose(np.sort(np.abs(moved).mean(axis=(0, 1))), [0.0, 1.0], atol=1e-4)
    assert np.allclose(mag, 2.0) and bg.shape == dcol.shape
    # sagittal (x is out of plane): no in-plane motion
    dcol_s, drow_s, *_ = _display_displacement(warp, fixed, 0, None, True)
    assert np.abs(dcol_s).max() < 1e-4 and np.abs(drow_s).max() < 1e-4


def test_input_pair_figure_titles_and_shared_row_range():
    import ants
    import matplotlib.pyplot as plt
    from syntx.viz.figures import render_input_pair_figure
    rng = np.random.default_rng(0)
    f = ants.from_numpy(rng.random((12, 14, 10)).astype('float32') + 0.1)
    fig = render_input_pair_figure(f, f, slice_indices=(3, 5, 7))
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert any("Moving: Sagittal (X=3)" in t for t in titles)          # printed the z index
    rows = [ax.images[0].get_clim() for ax in fig.axes if ax.images][:3]
    assert len(set(rows)) == 1                                        # one range per row
    plt.close(fig)


def test_deformation_tensor_rgb_numpy_fallback_and_geometry():
    """u = (0.5 x, 0, 0): the main stretch is along x -> red; the numpy fallback F is I + du/dx."""
    import ants
    from syntx.viz.figures import compute_deformation_tensor_rgb
    n = 10
    xs = np.arange(n, dtype=np.float32)
    u = np.zeros((n, n, n, 3), dtype=np.float32)             # tensor layout, comps (z, y, x)
    u[..., 2] = 0.5 * xs[None, None, :]
    rgb = compute_deformation_tensor_rgb(u).numpy()
    c = rgb[5, 5, 5]
    assert c[0] > 0.5 and c[1] < 1e-3 and c[2] < 1e-3          # x stretch shows as red
    img = ants.from_numpy(np.ascontiguousarray(u.transpose(2, 1, 0, 3)[..., ::-1]), origin=(1.0, 2.0, 3.0),
                          spacing=(1.0, 1.0, 1.0), has_components=True)
    out = compute_deformation_tensor_rgb(img)
    assert tuple(out.origin) == (1.0, 2.0, 3.0)


def test_velocity_grid_arrows_in_display_pixels():
    import ants
    import matplotlib.pyplot as plt
    from matplotlib.quiver import Quiver
    from syntx.viz.figures import plot_time_varying_velocity_grid
    fixed = ants.from_numpy(np.random.default_rng(0).random((16, 12)).astype('float32') + 0.1, spacing=(2.0, 1.0))
    v = np.zeros((2, 12, 16, 2), dtype=np.float32)          # tensor order (Y, X), comps (v_y, v_x)
    v[..., 1] = 1.0                                         # 1 mm / unit time along x
    fig = plot_time_varying_velocity_grid(v, fixed_image=fixed, mode='quiver', subsample_step=4)
    qs = [c for ax in fig.axes for c in ax.collections if isinstance(c, Quiver)]
    assert qs
    U, V = np.asarray(qs[0].U), np.asarray(qs[0].V)
    assert np.isclose(np.sort([np.abs(U).mean(), np.abs(V).mean()]), [0.0, 0.5], atol=1e-3).all()
    plt.close(fig)


def test_label_alignment_figure_uses_the_common_orientation():
    import ants
    import matplotlib.pyplot as plt
    from syntx.viz.figures import render_label_alignment_figure, extract_oriented_slice
    rng = np.random.default_rng(0)
    lab = ants.from_numpy(rng.integers(0, 30, (14, 12, 10)).astype('float32'))
    fig = render_label_alignment_figure(lab, lab, slice_indices=(4, 5, 6), crop_background=False,
                                        colormap_type='continuous', show_colorbar=False)
    shown = fig.axes[2].images[-1].get_array()                  # sagittal, fixed row
    ref, _ = extract_oriented_slice(lab, slice_axis=0, slice_idx=4)
    np.testing.assert_array_equal(np.ma.filled(shown, 0), ref)
    plt.close(fig)


def test_extract_slice_array_layout_same_with_or_without_ref():
    import ants
    from syntx.viz.core import AnatomicalVisualizer as AV
    a = np.random.default_rng(0).random((6, 7, 8)).astype('float32')   # tensor order (z, y, x)
    ref = ants.from_numpy(np.ascontiguousarray(a.T))
    for plane in ('axial', 'coronal', 'sagittal'):
        want = AV.extract_slice(ref, plane=plane, slice_idx=2).data
        np.testing.assert_array_equal(AV.extract_slice(a, plane=plane, slice_idx=2).data, want)
        np.testing.assert_array_equal(AV.extract_slice(a, plane=plane, slice_idx=2, ref_image=ref).data, want)
        np.testing.assert_array_equal(AV.extract_slice(torch.from_numpy(a), plane=plane, slice_idx=2).data, want)


def test_extract_slice_rejects_bad_plane_unreadable_file_and_off_grid_array(tmp_path):
    import ants
    from syntx.viz.core import AnatomicalVisualizer as AV
    img = ants.from_numpy(np.ones((6, 7, 8), 'float32'))
    for plane in ('axail', 5, -1):
        with pytest.raises(ValueError):
            AV.extract_slice(img, plane=plane)
    with pytest.raises(Exception):
        AV.extract_slice(str(tmp_path / 'missing.nii.gz'))
    with pytest.raises(ValueError):
        AV.extract_slice('affine0GenericAffine.mat')
    with pytest.raises(ValueError):
        AV.extract_slice(np.ones((5, 5, 5), 'float32'), ref_image=img)


def test_extract_slice_vector_layout_matches_scalar_components():
    import ants
    from syntx.viz.core import AnatomicalVisualizer as AV
    u = np.random.default_rng(1).random((6, 7, 8, 3)).astype('float32')   # ANTs order
    w = ants.from_numpy(u, has_components=True)
    comps = ants.split_channels(w)
    for plane in ('axial', 'coronal', 'sagittal'):
        got = AV.extract_slice(w, plane=plane, slice_idx=3).data
        want = np.stack([AV.extract_slice(c, plane=plane, slice_idx=3).data for c in comps], -1)
        np.testing.assert_array_equal(got, want)


def test_corner_watermark_corner_and_seed():
    from syntx.viz.core import corner_watermark
    a = np.zeros((20, 30), 'float32')
    a[5, 5] = 100.0
    br = corner_watermark(a, patch_size=4, corner='bottom_right')
    assert br[-4:, -4:].min() >= 85.0 and br[:4, :4].max() == 0.0
    tr = corner_watermark(a, patch_size=4, corner='top_right')
    assert tr[:4, -4:].min() >= 85.0 and tr[-4:, -4:].max() == 0.0
    np.testing.assert_array_equal(corner_watermark(a, 4), corner_watermark(a, 4))
    with pytest.raises(ValueError):
        corner_watermark(a, 4, corner='middle')


def test_deformation_tensor_rgb_array_in_tensor_layout():
    from syntx.viz.figures import compute_deformation_tensor_rgb
    D, H, W = 6, 7, 8
    u = np.zeros((D, H, W, 3), 'float32')                      # tensor layout, comps (z, y, x)
    u[..., 2] = 0.3 * np.arange(W)[None, None, :]              # stretch along x
    rgb = compute_deformation_tensor_rgb(u).numpy()             # ANTs (x, y, z, rgb)
    m = rgb[2:-2, 2:-2, 2:-2].reshape(-1, 3).mean(0)
    assert m[0] > 5 * max(m[1], m[2]), m                       # red = x


def test_label_colours_agree_between_palette_and_colormap():
    from syntx.viz.colormaps import build_dkt_label_palette, get_dkt_colormap
    cmap = get_dkt_colormap(60)
    color_map, lut = build_dkt_label_palette([7, 40, '12.0', 3.0, 'Brain-Stem'])
    for lab in (3, 7, 12, 40):
        np.testing.assert_allclose(color_map[lab], cmap(lab), atol=1e-6)
        np.testing.assert_allclose(lut[lab], cmap(lab), atol=1e-6)
    assert 'Brain-Stem' in color_map and '12.0' not in color_map
    with pytest.raises(ValueError):
        build_dkt_label_palette([2.5])


def test_label_overlap_stats_region_lists_and_single_box():
    from syntx.viz.stats import plot_label_overlap_stats
    fig = plot_label_overlap_stats({1: [0.7, 0.8], 2: [0.9, 0.85], 3: 0.6})
    assert len(fig.axes[0].patches) == 1                       # one box, not three copies
    assert len(fig.axes[1].patches) == 3                       # three region bars
    fig = plot_label_overlap_stats({"fixed_dice": [0.7, 0.8], "moving_dice": [0.75, 0.8]})
    assert len(fig.axes[0].patches) == 3


def test_jacobian_distribution_mask_and_status(tmp_path):
    from syntx.viz.stats import plot_jacobian_distribution
    d = np.ones((10, 10), 'float32')
    d[0, 0] = -1.0                                              # fold outside the mask
    m = np.zeros_like(d)
    m[2:, 2:] = 1
    fig = plot_jacobian_distribution(torch.from_numpy(d).requires_grad_(), mask=m)
    t = fig.axes[0].get_title()
    assert "0.00% Folding" in t and "Diffeomorphic" not in t
    assert "1.000% Grid Folding" in plot_jacobian_distribution(d).axes[0].get_title()


def test_loss_convergence_labels_export_and_dirs(tmp_path):
    import syntx.viz as viz
    out = tmp_path / "a" / "b" / "loss.png"
    fig = viz.plot_loss_convergence([3.0, 2.0, 1.0], output_path=str(out), label="MSE")
    ax = fig.axes[0]
    assert out.exists() and ax.get_xlabel() == "Iteration"
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["MSE"]
