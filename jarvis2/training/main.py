import logging
import os

import numpy as np

from jarvis2.data.getters import get_data
from jarvis2.eval.google_sheet import compute_google_sheet
from jarvis2.utils.data import get_kfold_and_groups
from jarvis2.utils.training import train_pipeline, compute_kfold_scores, get_csv_score
from jarvis2.utils.utils import persist_args, seed_everything


def run_experiment(a):
    seed_everything(a)
    persist_args(a)

    logging.info("Getting data")
    train_data = get_data(a, a.raw_train_path)

    if a.n_splits > 1:
        logging.info(f"Splitting in {a.n_splits} train-test splits")
        kfold, groups = get_kfold_and_groups(a, data=train_data)

        for fold, (train_ids, val_ids) in enumerate(kfold.split(train_data, groups=groups)):
            logging.info(f"Fold {fold + 1} out of {a.n_splits}")
            train_split = np.array(train_data)[train_ids].tolist()
            test_split = np.array(train_data)[val_ids].tolist()

            train_pipeline(a, test_split, train_split, fold=fold)

        return compute_kfold_scores(a, a.version)

    else:
        test_data = get_data(a, a.raw_test_path)
        model, test_ds = train_pipeline(a, test_data, train_data)

        logging.info("Computing Google Sheet")
        compute_google_sheet(a, model, test_ds.dataset)

        # TODO this currently seems broken and should be fixed
        # logging.info("Exporting model artefact")
        # export_to_onnx(a, model=model, test_ds=test_ds)

        return get_csv_score(a, csv_path=os.path.join(a.save_path, a.exp_name, a.version, "metrics.csv"))
