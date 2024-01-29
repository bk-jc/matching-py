import os
from pathlib import Path

import yaml

from jarvis2.main import main, parse_args
from tests.constants import SHARED_ARGS


# TODO e2e msm test


def test_train(tmpdir):
    main(parse_args([
        "--save_path", str(tmpdir),
        "--n_splits", str(0),
    ]))
    out_dir = os.path.join(tmpdir, parse_args([]).exp_name)
    assert len(os.listdir(out_dir)) == 1  # Only one output experiment
    out_time = os.listdir(out_dir)[0]
    out_dir = os.path.join(out_dir, out_time)

    # onnx_files = [f for f in os.listdir(out_dir) if f.endswith(".onnx")]
    # assert len(onnx_files) == 1  TODO fix onnx for MSM model


def test_train_cross_val(tmpdir):
    n_splits = 2
    main(parse_args([
        "--save_path", str(tmpdir),
        "--n_splits", str(n_splits),
    ]))
    out_dir = os.path.join(tmpdir, parse_args([]).exp_name)
    assert len(os.listdir(out_dir)) == 1  # Only one output experiment
    out_time = os.listdir(out_dir)[0]
    out_dir = os.path.join(out_dir, out_time)

    fold_dirs = [f for f in os.listdir(out_dir) if f.startswith("fold")]
    assert len(fold_dirs) == n_splits

    kfold_files = [f for f in os.listdir(out_dir) if f.startswith("kfold")]
    assert len(kfold_files) == 1


def test_optuna(tmpdir):
    optuna_config_path = Path(__file__).parent.parent / 'config' / 'optuna_config.yaml'
    main(parse_args([
        "--save_path", str(tmpdir),
        "--optuna_path", str(optuna_config_path),
        *SHARED_ARGS
    ]))
    out_dir = os.path.join(tmpdir, parse_args([]).exp_name)
    assert len(os.listdir(out_dir)) == 1  # Only one output experiment
    out_time = os.listdir(out_dir)[0]
    out_dir = os.path.join(out_dir, out_time)

    assert len([d for d in os.listdir(out_dir) if d.startswith("run")]) == yaml.safe_load(open(optuna_config_path))[
        "n_runs"]
    assert "optuna.csv" in os.listdir(out_dir) and "coordinates.png" in os.listdir(out_dir)
