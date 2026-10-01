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
