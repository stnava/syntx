"""
Published results (docs/provenance/best_parameters.json) must carry provenance derived
from the runs' own manifests (syntx.provenance.record_result).

The records listed in LEGACY_RECORDS predate syntx.provenance (2026-09-28) and are
exempt; the list is closed -- new records cannot be added to it. The 2026-09-20 5-arm
cohort (mindboggle_90pair_5arm_canonical_seeded_benchmark_2026_09_20) is the case that
motivated this: its parameters had to be reconstructed after the fact.
"""

import copy
import json
import os

import numpy as np
import pytest
import ants

import syntx
from syntx.provenance import (
    build_manifest,
    capture_registration_calls,
    check_record_provenance,
    record_result,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BEST = os.path.join(ROOT, "docs", "provenance", "best_parameters.json")

LEGACY_RECORDS = frozenset({
    "syntx.syn/90pair_population_benchmark_zero_folding_parity_dsti_shield_mps",
    "syntx.syn/90pair_population_benchmark_gaussian_mps",
    "syntx.syn/90pair_population_benchmark_sobolev_mps",
    "syntx.syn/3D_peak_sobolev",
    "syntx.syn/3D_dsti1_evaluation",
    "syntx.syn/3D_peak_autograd",
    "syntx.syn/peak_relaxed_gaussian_syn_step05_mbhard",
    "syntx.syn/peak_regadam_gaussian_syn_mbhard",
    "syntx.syn/peak_regadam_dsti1_syn_mbhard",
    "syntx.syn/peak_regadam_sobolev_syn_mbhard",
    "syntx.syn/2D_peak_sobolev",
    "syntx.robust_affine/quick_search_initialization",
    "syntx.tvf/peak_accuracy_reg_adam_gaussian",
    "syntx.tvf/peak_speed_topology_reg_adam_sobolev",
    "syntx.tvf/90pair_population_benchmark_dirichlet_shield_mps",
    "syntx.tvf/peak_dirichlet_reg_adam_dsti",
    "syntx.tvf/legacy_sobolev_adam_cfl035",
    "syntx.syngs/peak_antithetic_regadam_sobolev_mps",
    "syntx.syngs/peak_diffeomorphic_sobolev_regadam_mps",
    "syntx.syngs/strict_diffeomorphic_zero_folding_mps",
    "syntx.syngs/peak_gaussian_velocity_transport_mps",
    "syntx.syngs/balanced_gaussian_velocity_transport_mps",
    "syntx.syngs/strict_gaussian_velocity_transport_zero_folding_mps",
    "syntx.syngs/peak_sobolev_velocity_transport_mps",
    "syntx.syngs/strict_dsti1_zero_folding_mps",
    "affine_pt2",
    "mindboggle_90pair_5arm_canonical_seeded_benchmark_2026_09_20",
})


def _records(data):
    for k, v in data.items():
        if k.startswith("syntx.") and isinstance(v, dict):
            for k2, v2 in v.items():
                yield f"{k}/{k2}", v2
        else:
            yield k, v


def test_every_new_published_record_has_provenance():
    with open(BEST) as f:
        data = json.load(f)
    names = dict(_records(data))
    assert LEGACY_RECORDS <= set(names), f"legacy records vanished: {LEGACY_RECORDS - set(names)}"
    bad = {}
    for name, rec in names.items():
        if name in LEGACY_RECORDS:
            continue
        try:
            check_record_provenance(rec)
        except ValueError as e:
            bad[name] = str(e)
    assert not bad, ("records without run-derived provenance (add them with "
                     f"syntx.provenance.record_result): {bad}")


@pytest.fixture(scope="module")
def manifests():
    arr = np.random.default_rng(0).random((16, 16, 16)).astype(np.float32)
    f, m = ants.from_numpy(arr), ants.from_numpy(np.roll(arr, 1, 0))
    with capture_registration_calls() as cap:
        syntx.syn(fixed=f, moving=m, initial_transform="identity", reg_iterations=[1, 1, 1],
                  device="cpu")
    man = build_manifest(calls=cap.calls, run={}, include_diff=False)
    return [man, copy.deepcopy(man)]


def test_record_result_single_and_multi_arm(tmp_path, manifests):
    path = str(tmp_path / "best.json")
    with open(path, "w") as f:
        json.dump({"syntx.syn": {}}, f)

    rec = record_result(path, "demo_single", {"mean_dice": 0.6}, manifests, method_key="syntx.syn")
    check_record_provenance(rec)
    assert rec["provenance"]["n_runs"] == 2

    multi = record_result(path, "demo_multi", {"arms": {"syn": {"mean_dice": 0.6}}},
                          {"syn": manifests})
    check_record_provenance(multi)

    with pytest.raises(ValueError, match="exists"):
        record_result(path, "demo_single", {}, manifests, method_key="syntx.syn")
    with pytest.raises(ValueError, match="without manifests"):
        record_result(path, "demo_bad", {"arms": {"syn": {}, "tvf": {}}}, {"syn": manifests})


def test_check_record_provenance_rejects_hand_written_parameters():
    with pytest.raises(ValueError, match="no provenance"):
        check_record_provenance({"mean_dice": 0.6, "parameters": {"grad_step": 0.25}})
    with pytest.raises(ValueError, match="per arm"):
        check_record_provenance({"arms": {"syn": {}}, "provenance": {"commit": "a" * 40}})
