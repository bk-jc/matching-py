import copy
import logging
import time
from pathlib import Path

import optuna
import yaml

from training.main import run_experiment


def grid_search(a):
    with open(a.optuna_path, 'r') as f:
        config = yaml.safe_load(f)

    a.run_idx = 0

    def optuna_main_wrapper(trial, valid_keys=["low", "high", "log", "int"]):

        trial_a = copy.deepcopy(a)
        trial_a.version = f"{a.version}/run{a.run_idx}"
        a.run_idx += 1

        for key, val in config.items():
            if key == "n_runs":
                continue
            if hasattr(trial_a, key):
                if isinstance(val, list):
                    v = trial.suggest_categorical(key, val)
                elif isinstance(val, dict):
                    for config_key in val.keys():
                        if config_key not in valid_keys:
                            logging.warning(f"Found Optuna key {config_key} but this is not a valid key "
                                            f"(e.g. {', '.join(valid_keys)})")
                    kwargs = {
                        "name": key,
                        "low": val["low"],
                        "high": val["high"],
                        "log": val.get("log", False)
                    }
                    v = trial.suggest_int(**kwargs) if val.get("int", False) else trial.suggest_float(**kwargs)
                else:
                    v = val
                setattr(trial_a, key, v)
            else:
                logging.warning(f"Found key {key} but this is not an argument for the training script.")

        return run_experiment(trial_a)

    study = optuna.create_study(
        direction="minimize" if a.lower_is_better else "maximize",
        study_name=f"{a.exp_name}_{a.version}"
    )
    start_time = time.time()
    study.optimize(optuna_main_wrapper, n_trials=config["n_runs"])
    duration = time.time() - start_time
    logging.info(f"Grid search duration: {duration} seconds.")
    logging.info(f"Average run duration: {duration / config['n_runs']} seconds.")

    # Persist data about grid search
    base_path = Path(a.save_path) / config.get("exp_name", a.exp_name) / a.version
    study.trials_dataframe().to_csv(base_path / "optuna.csv")
    optuna.visualization.plot_parallel_coordinate(
        study,
        params=[k for k, v in config.items() if isinstance(v, list) or isinstance(v, dict)],
        target_name=config.get("score_metric", a.score_metric)
    ).write_image(base_path / "coordinates.png")
    return study.best_params