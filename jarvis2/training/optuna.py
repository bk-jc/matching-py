import copy
import logging
import os
import time
from pathlib import Path

import optuna
import yaml

from jarvis2.training.main import run_experiment


def grid_search(a):
    with open(a.optuna_path, 'r') as f:
        config = yaml.safe_load(f)

    a.run_idx = 0

    base_path = Path(a.save_path) / config.get("exp_name", a.exp_name) / a.version
    os.makedirs(base_path, exist_ok=True)
    yaml.dump(config, open(base_path / "grid.yaml", "w"))

    def optuna_main_wrapper(trial):

        valid_keys = ["low", "high", "log", "int"]
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

    def save_results_callback(_study: optuna.Study, trial: optuna.trial.FrozenTrial):

        _study.trials_dataframe().to_csv(base_path / "optuna.csv")

        # Save the parallel coordinate plot only for the best trial so far
        optuna.visualization.plot_parallel_coordinate(
            _study,
            params=[
                k for k, v in config.items() if isinstance(v, list) or isinstance(v, dict)
            ],
            target_name=config.get("score_metric", a.score_metric)
        ).write_image(base_path / "coordinates.png")

    study = optuna.create_study(
        direction="minimize" if a.lower_is_better else "maximize",
        study_name=f"{a.exp_name}_{a.version}",
    )

    start_time = time.time()
    study.optimize(optuna_main_wrapper, n_trials=config["n_runs"], callbacks=[save_results_callback])
    duration = time.time() - start_time
    logging.info(f"Grid search duration: {duration} seconds.")
    logging.info(f"Average run duration: {duration / config['n_runs']} seconds.")
    
    return study.best_params
