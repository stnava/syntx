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
