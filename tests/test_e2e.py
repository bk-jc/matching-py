import os
from pathlib import Path

import yaml

from src.train import main, parse_args

SHARED_ARGS = ["--train_steps", "2", "--val_steps", "1", "--train_batch_size", "2", "--val_batch_size", "2"]


def test_train(tmpdir):
    main(parse_args([
        "--save_path", str(tmpdir),
        "--n_splits", str(0),
    ]))
    out_dir = os.path.join(tmpdir, parse_args([]).exp_name)
    assert len(os.listdir(out_dir)) == 1  # Only one output experiment
    out_time = os.listdir(out_dir)[0]
    out_dir = os.path.join(out_dir, out_time)

    onnx_files = [f for f in os.listdir(out_dir) if f.endswith(".onnx")]
    assert len(onnx_files) == 1


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
