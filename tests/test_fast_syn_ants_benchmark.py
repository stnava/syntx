"""Contract & smoke tests for ``benchmarks/fast_syn_ants_benchmark.py``."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import ants
import numpy as np

from syntx.provenance import assert_manifest_complete


def _load_benchmark_module():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "benchmarks" / "fast_syn_ants_benchmark.py"
    spec = importlib.util.spec_from_file_location("fast_syn_ants_benchmark", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_synthetic_mindboggle_fixture(root: Path) -> tuple[Path, Path]:
    """Create a tiny 2-subject 3-D Mindboggle directory and pairs.csv with non-trivial geometry."""
    data_dir = root / "volumes"
    cohort_dir = data_dir / "OASIS-TRT-20_volumes"
    grid = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    xx, yy, zz = np.meshgrid(grid, grid, grid, indexing="ij")

    for idx, shift in ((1, -0.08), (2, 0.08)):
        subj_dir = cohort_dir / f"OASIS-TRT-20-{idx}"
        subj_dir.mkdir(parents=True, exist_ok=True)
        r2 = (xx - shift) ** 2 + (yy + 0.5 * shift) ** 2 + zz**2
        brain = np.clip(np.exp(-3.0 * r2) * (1.0 + 0.25 * np.cos(5.0 * xx)), 0.0, None)
        brain[r2 > 0.68] = 0.0

        labels = np.zeros_like(brain, dtype=np.uint32)
        labels[(r2 <= 0.55) & (xx - shift < 0.0)] = 1002
        labels[(r2 <= 0.55) & (xx - shift >= 0.0)] = 2002

        img = ants.from_numpy(
            brain.astype(np.float32),
            origin=(-24.0, -24.0, -24.0),
            spacing=(2.0, 2.0, 2.0),
            direction=np.eye(3),
        )
        lbl = ants.from_numpy(
            labels,
            origin=(-24.0, -24.0, -24.0),
            spacing=(2.0, 2.0, 2.0),
            direction=np.eye(3),
        )
        ants.image_write(img, str(subj_dir / "t1weighted_brain.nii.gz"))
        ants.image_write(lbl, str(subj_dir / "labels.DKT31.manual.nii.gz"))

    pairs_csv = root / "pairs.csv"
    pairs_csv.write_text(
        "type,cohort1,subject1,cohort2,subject2\n"
        "inter,OASIS-TRT-20,OASIS-TRT-20-1,OASIS-TRT-20,OASIS-TRT-20-2\n",
        encoding="utf-8",
    )
    return data_dir, pairs_csv


def test_fast_syn_ants_benchmark_smoke_and_contracts(tmp_path: Path):
    mod = _load_benchmark_module()
    assert mod.DEFAULT_REG_ITERATIONS == (100, 100, 20)

    data_dir, pairs_csv = _make_synthetic_mindboggle_fixture(tmp_path)
    out_dir = tmp_path / "results"
    report_stem = tmp_path / "reports" / "fast_bench"

    res = mod.main(
        [
            "--pairs",
            "0",
            "--repro-pair",
            "0",
            "--reg-iterations",
            "2",
            "1",
            "--regularizers",
            "sobolev",
            "hyperelastic",
            "--pairs-csv",
            str(pairs_csv),
            "--data-dir",
            str(data_dir),
            "--no-n4",
            "--device",
            "cpu",
            "--ants-aff-iterations",
            "3",
            "2",
            "--syntx-aff-iterations",
            "3",
            "2",
            "--skip-figures",
            "--out-dir",
            str(out_dir),
            "--report-stem",
            str(report_stem),
        ]
    )

    assert Path(res["html_path"]).exists()
    assert Path(res["md_path"]).exists()
    assert Path(res["json_path"]).exists()

    summary = json.loads(Path(res["json_path"]).read_text(encoding="utf-8"))
    # Default must NOT run rep=1 reproducibility duplicates
    assert summary["track4_reproducibility"] == []
    assert len(summary["track1_affine_summary"]) == 4
    assert len(summary["track2_deformable_summary"]) == 4  # 2 regularizers + 2 ANTs arms
    assert len(summary["track3_e2e_summary"]) == 2

    # Shared canonical affine SHA-256 must match across every Track 2 deformable run
    run_files = sorted((out_dir / "runs").glob("*.json"))
    assert len(run_files) == 10  # 4 affine + 4 deformable + 2 e2e (zero duplicate reruns)
    deform_shas = set()
    valid_qc = {"pass", "warn", "fail", "not_assessable"}
    for rf in run_files:
        rec = json.loads(rf.read_text(encoding="utf-8"))
        assert rec["status"] == "SUCCESS"
        assert rec["rep"] == 0
        assert 0.0 <= rec["dice_sym"] <= 1.0
        assert rec["qc"]["status"] in valid_qc
        assert_manifest_complete(rec["provenance"], require_calls=True)
        if rec["track"] == "deformable":
            assert rec["shared_affine_sha256"] is not None
            deform_shas.add(rec["shared_affine_sha256"])

    assert len(deform_shas) == 1


def test_fast_syn_ants_benchmark_cli_smoke_flag(tmp_path: Path):
    mod = _load_benchmark_module()
    smoke_out = tmp_path / "smoke_results"
    smoke_stem = tmp_path / "smoke_reports" / "smoke"
    res = mod.main(["--smoke", "--out-dir", str(smoke_out), "--report-stem", str(smoke_stem)])
    assert res["summary"]["smoke"] is True
    assert res["summary"]["pairs"] == [0, 1]
    assert res["summary"]["reg_iterations"] == [2, 2]
    assert Path(res["html_path"]).exists()
    assert Path(res["md_path"]).exists()
    assert Path(res["json_path"]).exists()
    assert len(res["summary"]["figures"]) >= 4


def test_pair_by_pair_shuffle_best_models_mode(tmp_path: Path):
    mod = _load_benchmark_module()
    data_dir, pairs_csv = _make_synthetic_mindboggle_fixture(tmp_path)
    # Add a second intra-site pair so both intra and inter subgroups are exercised
    subj3 = data_dir / "OASIS-TRT-20_volumes" / "OASIS-TRT-20-3"
    subj3.mkdir(parents=True, exist_ok=True)
    src2 = data_dir / "OASIS-TRT-20_volumes" / "OASIS-TRT-20-2"
    for fn in ("t1weighted_brain.nii.gz", "labels.DKT31.manual.nii.gz"):
        (subj3 / fn).write_bytes((src2 / fn).read_bytes())
    pairs_csv.write_text(
        "type,cohort1,subject1,cohort2,subject2\n"
        "inter,OASIS-TRT-20,OASIS-TRT-20-1,OASIS-TRT-20,OASIS-TRT-20-2\n"
        "intra,OASIS-TRT-20,OASIS-TRT-20-1,OASIS-TRT-20,OASIS-TRT-20-3\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "pbp_results"
    report_stem = tmp_path / "pbp_reports" / "cohort_bench"
    res = mod.main(
        [
            "--all-pairs",
            "--shuffle-pairs",
            "--pair-by-pair",
            "--best-models-only",
            "--include-syngs-tvf",
            "--include-syngs-beltrami",
            "--regularizers",
            "beltrami",
            "beltrami_proj",
            "sobolev",
            "--reg-iterations",
            "2",
            "1",
            "--seed",
            "20261112",
            "--pairs-csv",
            str(pairs_csv),
            "--data-dir",
            str(data_dir),
            "--no-n4",
            "--device",
            "cpu",
            "--ants-aff-iterations",
            "3",
            "2",
            "--syntx-aff-iterations",
            "3",
            "2",
            "--out-dir",
            str(out_dir),
            "--report-stem",
            str(report_stem),
        ]
    )

    assert (out_dir / "pair_order.json").exists()
    p_order = json.loads((out_dir / "pair_order.json").read_text(encoding="utf-8"))
    assert p_order["seed"] == 20261112
    assert sorted(p_order["shuffled_pairs"]) == [0, 1]

    summary = res["summary"]
    assert summary["in_progress"] is False
    assert summary["completed_pairs"] == 2
    assert summary["total_pairs"] == 2
    assert len(summary["track1_affine_summary"]) == 2
    assert len(summary["track2_deformable_summary"]) == 7
    assert summary["track3_e2e_summary"] == []

    arms_def = {r["arm"] for r in summary["track2_deformable_summary"]}
    assert arms_def == {
        "syn_beltrami",
        "syn_beltrami_proj",
        "syn_sobolev",
        "syngs_beltrami",
        "syngs_best",
        "tvf_best",
        "ants_syn_matched",
    }
    for r in summary["track2_deformable_summary"]:
        assert r["n_intra"] == 1
        assert r["n_inter"] == 1
        assert "mean_fd_folding_pct" in r

    figs = summary["figures"]
    for req_fig in (
        "pareto",
        "distribution",
        "4panel_tvf",
        "4panel_syngs_beltrami",
        "4panel_syn_beltrami",
        "4panel_syn_beltrami_proj",
    ):
        assert req_fig in figs
        assert Path(figs[req_fig]).exists()

    html_txt = Path(res["html_path"]).read_text(encoding="utf-8")
    assert "Intra-Cohort vs. Cross-Cohort" in html_txt
    assert "syngs_beltrami" in html_txt


def test_beltrami_and_beltrami_proj_config_separation():
    from syntx.benchmark.config import get_model_config, syn_config_to_syn_kwargs

    mod = _load_benchmark_module()
    assert mod.DEFORMABLE_SYN_CONFIG_KEY["beltrami"] == "syn_beltrami"
    assert mod.DEFORMABLE_SYN_CONFIG_KEY["beltrami_proj"] == "syn_beltrami_proj"
    assert mod.DEFORMABLE_SYN_CONFIG_KEY["matched"] == "syn_matched"

    kw_belt = syn_config_to_syn_kwargs(get_model_config("syn_beltrami"))
    assert kw_belt["regularizer"] == "beltrami"
    assert kw_belt["grad_step"] == 0.7
    assert kw_belt["dilatation_weight"] == 0.1
    assert kw_belt["sobolev_alpha"] == 2.25
    assert kw_belt.get("cfl_project_inverse", False) is False

    kw_proj = syn_config_to_syn_kwargs(get_model_config("syn_beltrami_proj"))
    assert kw_proj["regularizer"] == "beltrami"
    assert kw_proj["grad_step"] == 0.6
    assert kw_proj["dilatation_weight"] == 0.1
    assert kw_proj["sobolev_alpha"] == 2.25
    assert kw_proj["cfl_project_inverse"] is True
    assert kw_proj["projection_frequency"] == 1
    assert kw_proj["cfl_project_tol"] == 1.0
    assert kw_proj["cfl_project_jac_floor"] == 0.0

    kw_matched = syn_config_to_syn_kwargs(get_model_config("syn_matched"))
    assert kw_matched["regularizer"] == "gaussian"
    assert kw_matched["grad_step"] == 0.30
    assert kw_matched["flow_sigma"] == 3.6
    assert kw_matched["cfl_project_inverse"] is True
    assert kw_matched["projection_frequency"] == 1
    assert kw_matched["cfl_project_tol"] == 0.20
    assert kw_matched["cfl_project_jac_floor"] == 0.0
    assert kw_matched["inv_refresh_every"] == [1, 1, 1]
    assert kw_matched["lncc_single_pass"] is True

