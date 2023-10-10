import copy
import logging
import os
import sys

import numpy as np
import optuna
import yaml
from sklearn import model_selection

from src.data import get_data
from src.utils.onnx import export_to_onnx
from src.utils.training import compute_kfold_scores, train_pipeline, get_csv_score
from src.utils.utils import parse_args

logging.getLogger().setLevel(logging.DEBUG)


def run_experiment(a):
    logging.info("Getting data")
    train_data = get_data(a, a.raw_train_path)
    test_data = get_data(a, a.raw_test_path)

    if a.n_splits > 1:
        logging.info(f"Splitting in {a.n_splits} train-test splits")
        kfold = model_selection.KFold(n_splits=a.n_splits)

        for fold, (train_ids, val_ids) in enumerate(kfold.split(train_data)):
            logging.info(f"Fold {fold + 1} out of {a.n_splits}")
            train_split = np.array(train_data)[train_ids].tolist()
            test_split = np.array(train_data)[val_ids].tolist()

            train_pipeline(a, test_split, train_split, fold=fold)

        return compute_kfold_scores(a, a.version)

    else:
        model, test_ds = train_pipeline(a, test_data, train_data)

        logging.info("Exporting model artefact")
        export_to_onnx(a, model=model, test_ds=test_ds)

        return get_csv_score(a, csv_path=os.path.join(a.save_path, a.exp_name, a.version, "metrics.csv"))


def grid_search(a):
    with open(a.optuna_path, 'r') as f:
        config = yaml.safe_load(f)

    a.run_idx = 0

    def optuna_main_wrapper(trial):

        trial_a = copy.deepcopy(a)
        trial_a.version = f"{a.version}/run{a.run_idx}"
        a.run_idx += 1

        for key, values in config.items():
            if key == "n_runs":
                continue
            if hasattr(trial_a, key):
                v = trial.suggest_categorical(key, values)
                setattr(trial_a, key, v)
            else:
                logging.warning(f"Found key {key} but this is not an argument for the training script.")

        return run_experiment(trial_a)

    study = optuna.create_study(
        direction="minimize" if a.lower_is_better else "maximize",
        study_name=f"{a.exp_name}_{a.version}"
    )
    study.optimize(optuna_main_wrapper, n_trials=config["n_runs"])

    return study.best_params


def main(args):
    if args.optuna_path:
        grid_search(args)
    else:
        run_experiment(args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
