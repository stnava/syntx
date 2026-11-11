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
