"""
Regression tests for behaviour bugs found by the 2026-10-01 docstring audit
(docs/DOCSTRING_AUDIT.md). CPU only, each test well under a second.
"""
import numpy as np
import pytest
import torch


# 1. core/jacobian.py 'bspline' derivative kernel ------------------------------------------
@pytest.mark.parametrize("dim", [2, 3])
def test_bspline_derivative_of_linear_field_is_exact(dim):
    from syntx.core.jacobian import _spatial_jacobian_nd
    shape = (9,) * dim
    axes = torch.meshgrid(*[torch.arange(n, dtype=torch.float64) for n in shape], indexing="ij")
    field = torch.stack([0.3 * a for a in reversed(axes)], dim=-1).unsqueeze(0)   # u_k = 0.3 x_k
    J = _spatial_jacobian_nd(field, physical_spacing=[1.0] * dim, method="bspline")
    interior = (0,) + (slice(2, -2),) * dim
    expected = 0.3 * torch.eye(dim, dtype=torch.float64)
    assert torch.allclose(J[interior], expected.expand_as(J[interior]), atol=1e-10)
    # agrees with the central-difference path
    Jc = _spatial_jacobian_nd(field, physical_spacing=[1.0] * dim, method="central")
    assert torch.allclose(J[interior], Jc[interior], atol=1e-10)


# 2. viz/reports.py: no hard-coded benchmark numbers ----------------------------------------
def _rec(idx, dice, t=10.0, aff_t=1.25, cfg=None):
    r = {"status": "SUCCESS", "pair_idx": idx, "syntx_dice_sym": dice, "syntx_time": t,
         "syntx_fold": 0.0, "syntx_affine_dice_sym": 0.5, "syntx_affine_time": aff_t,
         "ants_baseline": {"dice_sym": 0.55, "runtime_seconds": 50.0}}
    if cfg:
        r["config"] = cfg
    return r


def test_affine_report_uses_recorded_times_not_constants(tmp_path, monkeypatch):
    from syntx.viz.reports import create_affine_benchmark_report
    monkeypatch.chdir(tmp_path)                       # no results/ files -> ANTs values unknown
    out = create_affine_benchmark_report({"gaussian_results": {"0": _rec(0, 0.6), "1": _rec(1, 0.7)}},
                                         output_html=str(tmp_path / "a.html"))
    html = open(out).read()
    assert "1.2s" in html or "1.3s" in html           # recorded syntx_affine_time (1.25 s)
    for fabricated in ("2.8s", "28.5s", "0.3472", "10.2&times;", "90 / 90", "0.5303"):
        assert fabricated not in html, fabricated
    assert "n/a" in html                              # unknown ANTs time / speedup


def test_population_report_counts_from_data(tmp_path):
    from syntx.viz.reports import create_population_benchmark_report
    tvf = {str(i): _rec(i, 0.65, cfg={"grad_step": 2.0}) for i in range(3)}
    sob = {str(i): _rec(i, 0.60 if i < 2 else 0.70) for i in range(3)}
    import json
    src = tmp_path / "summary.json"
    src.write_text(json.dumps({"tvf_results": tvf, "sobolev_results": sob}))
    out = create_population_benchmark_report(str(src), output_html=str(tmp_path / "p.html"))
    html = open(out).read()
    assert "88/90" not in html and "/ 90" not in html and "TVF 0.6466" not in html
    assert "TVF beats Sobolev in 2/3" in html
    assert "grad_step" in html                        # recorded config, not a static table
    assert "lr=1.2" not in html


# 3. generators.benchmark_data('mbhard') never synthesises Mindboggle data -------------------
def test_mbhard_missing_data_raises_instead_of_synthesising(tmp_path, monkeypatch):
    import syntx.benchmark.data as bdata
    from syntx.generators import benchmark_data
    monkeypatch.setattr(bdata, "resolve_data_dir", lambda *a, **k: str(tmp_path / "nowhere"))
    with pytest.raises(FileNotFoundError, match="Mindboggle"):
        benchmark_data("mbhard", data_dir=str(tmp_path))
    assert not list((tmp_path / "mbhard").glob("*.nii.gz"))     # nothing written


def test_mbhard_rejects_old_synthetic_placeholder(tmp_path, monkeypatch):
    import ants
    import syntx.benchmark.data as bdata
    from syntx.generators import benchmark_data
    monkeypatch.setattr(bdata, "resolve_data_dir", lambda *a, **k: str(tmp_path / "nowhere"))
    d = tmp_path / "mbhard"
    d.mkdir()
    img = ants.from_numpy(np.zeros((64, 64, 64), np.float32))
    for n in ("NKI-TRT-20-2_t1brain", "NKI-TRT-20-2_dkt31", "MMRR-21-2_t1brain", "MMRR-21-2_dkt31"):
        ants.image_write(img, str(d / f"{n}.nii.gz"))
    with pytest.raises(RuntimeError, match="synthetic placeholder"):
        benchmark_data("mbhard", data_dir=str(tmp_path))


# 4. benchmark/tune.py: the dataset is part of the cache key ---------------------------------
def test_tune_cache_is_separated_by_dataset(tmp_path):
    from tests.test_tune import _spec, synthetic_evaluator
    from syntx.benchmark.tune import Tuner
    kw = dict(pairs=[0], out_dir=str(tmp_path / "shared"), check_canonical=False,
              code_fingerprint={"commit": "0" * 40, "diff_sha256": "x"}, log=lambda s: None)
    calls_3d, calls_2d = [], []
    t3 = Tuner(_spec(), evaluator=synthetic_evaluator(calls_3d), dataset="mindboggle", **kw)
    t3.evaluate({}, stage="t")
    t2 = Tuner(_spec(), evaluator=synthetic_evaluator(calls_2d), dataset="2d", **kw)
    t2.evaluate({}, stage="t")
    assert len(calls_3d) == 1 and len(calls_2d) == 1          # 2-D did not reuse the 3-D row
    assert t3._cache_key(0, {}, 0) != t2._cache_key(0, {}, 0)
    t2b = Tuner(_spec(), evaluator=synthetic_evaluator(calls_2d), dataset="2d", **kw)
    t2b.evaluate({}, stage="t")
    assert len(calls_2d) == 1                                 # same dataset: cached


def test_tune_default_out_dir_names_dataset(tmp_path, monkeypatch):
    from tests.test_tune import _spec, synthetic_evaluator
    from syntx.benchmark.tune import Tuner
    monkeypatch.chdir(tmp_path)
    t = Tuner(_spec(), pairs=[0], evaluator=synthetic_evaluator([]), dataset="2d", check_canonical=False,
              code_fingerprint={"commit": "0" * 40, "diff_sha256": "x"}, log=lambda s: None)
    assert "_2d_" in t.out_dir


# 5. policy.py / auto_reg: every policy field is applied; TVF policies run ------------------
def test_policy_has_only_applied_fields():
    import dataclasses
    from syntx.policy import RegistrationPolicy
    names = {f.name for f in dataclasses.fields(RegistrationPolicy)}
    assert names == {"transform_type", "similarity_metric", "guided", "cohort_type",
                     "robust_affine", "denoise", "ct_window", "explanation"}


def test_auto_reg_thorax_ct_policy_runs_tvf(monkeypatch):
    import ants
    import syntx
    import sys
    import syntx.diagnose as sdiag
    stvf = sys.modules["syntx.tvf"]                 # syntx.tvf (the attribute) is the function
    from syntx.diagnose import ImageDiagnosis, PairDiagnosis
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)    # keep it on the CPU
    ct = ImageDiagnosis(modality="CT", body_part="THORAX", intensity_domain="HOUNSFIELD")
    pair = PairDiagnosis(fixed=ct, moving=ct, relationship="MONO_MODAL_INTRA",
                         is_same_anatomy=True, is_same_modality=True)
    monkeypatch.setattr(sdiag, "diagnose_pair", lambda *a, **k: pair)
    seen = {}
    real = stvf.tvf_registration

    def spy(**kw):
        seen.update(kw)
        return real(**kw)
    monkeypatch.setattr(stvf, "tvf_registration", spy)
    fi = ants.resample_image(ants.image_read(ants.get_data("r16")), (64, 64), use_voxels=True)
    mi = ants.resample_image(ants.image_read(ants.get_data("r64")), (64, 64), use_voxels=True)
    res = syntx.auto_reg(fi, mi, device="cpu", reg_iterations=[2, 0, 0])
    assert seen["syn_metric"] == "cc2" and "similarity_metric" not in seen
    assert res["metrics"]["type_of_transform_used"] == "TVF"


# 6. JAX backends: run, no hidden affine re-optimisation, sigmas in mm --------------------
class _StopFit(Exception):
    pass


def test_tvf_jax_wrapper_passes_mm_sigmas_and_no_affine_stage(monkeypatch):
    pytest.importorskip("jax")
    import ants
    import syntx
    from syntx.tvf_jax import TVFModelJAX
    seen = {}

    def spy(self, *a, **kw):
        seen.update(kw)
        raise _StopFit
    monkeypatch.setattr(TVFModelJAX, "fit", spy)
    fi = ants.resample_image(ants.image_read(ants.get_data("r16")), (48, 48), use_voxels=True)
    fi.set_spacing((2.0, 2.0))
    with pytest.raises(_StopFit):
        syntx.tvf(fi, fi, backend="jax", regularizer="gaussian", flow_sigma=3.0,
                  reg_iterations=[2, 2], device="cpu", initial_transform="identity")
    assert seen["affine_epochs"] == 0
    assert seen["levels"] == [2, 1]
    assert np.allclose(seen["fluid_sigmas"], [(3.0 / 4.0) ** 2, (3.0 / 2.0) ** 2])   # mm -> voxels^2
    assert "alpha" not in seen


def test_tvf_jax_alpha_resolution_ignores_none():
    pytest.importorskip("jax")
    from syntx.tvf_jax import _resolve_alpha
    assert _resolve_alpha({"alpha": None}, 0.5) == 0.5            # syntx.tvf passes alpha=None
    assert _resolve_alpha({"sobolev_alpha": None, "alpha": 2.0}, 0.5) == 2.0
    assert _resolve_alpha({"sobolev_alpha": 1.0, "alpha": 2.0}, 0.5) == 1.0


@pytest.mark.slow
def test_tvf_jax_fit_accepts_alpha_none():
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from syntx.tvf_jax import TVFModelJAX
    m = TVFModelJAX(dim=2, image_shape=(8, 8), velocity_shape=(8, 8), n_time_steps=1,
                    spacing=[1.0, 1.0], origin=[0.0, 0.0], direction=np.eye(2).tolist(), fluid_sigma=1.0)
    rng = np.random.default_rng(0)
    img = jnp.asarray(rng.random((1, 1, 8, 8), dtype=np.float32))
    img2 = jnp.asarray(rng.random((1, 1, 8, 8), dtype=np.float32))   # identical pairs exit early
    m.fit(img, img2, levels=[1], epochs_per_level=[1], affine_epochs=0, alpha=None,
          regularizer="gaussian", fixed_spacing=[1.0, 1.0], fixed_origin=[0.0, 0.0],
          fixed_direction=np.eye(2))
    assert np.isfinite(np.asarray(m.velocity)).all()


def _spy_syn_jax_fit(monkeypatch):
    from syntx.syn_jax import SyNJAX
    seen = {}

    def spy(self, *a, **kw):
        seen.update(kw)
        raise _StopFit
    monkeypatch.setattr(SyNJAX, "fit", spy)
    return seen


def test_syn_jax_does_not_reoptimise_given_affine(monkeypatch):
    pytest.importorskip("jax")
    import ants
    import syntx
    seen = _spy_syn_jax_fit(monkeypatch)
    fi = ants.resample_image(ants.image_read(ants.get_data("r16")), (32, 32), use_voxels=True)
    with pytest.raises(_StopFit):
        syntx.syn(fi, fi, backend="jax", reg_iterations=[1], device="cpu", initial_transform="identity")
    assert seen["affine_epochs"] == 0
    assert seen["use_analytical_gradients"] is False            # passed through ...


def test_syn_jax_fit_honours_use_analytical_gradients():
    pytest.importorskip("jax")
    import ast
    import inspect
    import textwrap
    from syntx.syn_jax import SyNJAX
    tree = ast.parse(textwrap.dedent(inspect.getsource(SyNJAX.fit)))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "get"
             and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "use_analytical_gradients"]
    assert not calls                                            # ... and no longer overridden


def test_warp_images_jax_accepts_string_interpolator():
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from syntx.syn_jax import warp_images_jax
    z, img, e, v = jnp.zeros((1, 4, 4, 2)), jnp.zeros((1, 1, 4, 4)), jnp.eye(2), jnp.ones(2)
    sh = jnp.array([4.0, 4.0])
    out = warp_images_jax(z, z, z, z, img, img, z, sh, v, 0 * v, e, sh, v, 0 * v, e, e, 0 * v, None,
                          "linear")                             # a str: must be a static argument
    assert len(out) == 2


# 7. scattered/solver.py: inverse errors use the fields' own convention ([-1, 1], x-y-z) ------
def _scaled_x_pair(shape=(9, 10, 11), s=1.1):
    """u(x) = (s - 1) x_x and its exact inverse v(x) = (1/s - 1) x_x, components (x, y, z), on a
    normalised grid in tensor order; linear, so bilinear sampling is exact."""
    zs, ys, xs = torch.meshgrid(*[torch.linspace(-1, 1, n, dtype=torch.float64) for n in shape], indexing="ij")
    zero = torch.zeros_like(xs)
    fwd = torch.stack([(s - 1) * xs, zero, zero], dim=-1).unsqueeze(0)
    inv = torch.stack([(1 / s - 1) * xs, zero, zero], dim=-1).unsqueeze(0)
    return fwd, inv


def test_scattered_inverse_error_is_zero_for_exact_inverse_3d():
    from syntx.scattered.solver import SyNScattered
    fwd, inv = _scaled_x_pair()
    model = SyNScattered.__new__(SyNScattered)            # only the two fields are needed
    torch.nn.Module.__init__(model)
    model.disp_fwd, model.disp_inv = fwd, inv
    err = model.compute_inverse_error()
    assert err["max_error"] < 1e-6 and err["mean_error"] < 1e-6


def test_scattered_inverse_error_detects_wrong_inverse_3d():
    from syntx.scattered.solver import _inverse_consistency_error
    fwd, inv = _scaled_x_pair()
    e_ok = _inverse_consistency_error(fwd, inv)
    e_bad = _inverse_consistency_error(fwd, -fwd)             # -u is not the inverse of x -> 1.1 x
    assert e_ok["max_error"] < 1e-6
    assert e_bad["max_error"] > 1e-3
