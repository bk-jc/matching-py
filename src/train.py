import logging
import sys
from datetime import datetime

import numpy as np
from sklearn import model_selection

from src.data import get_data
from src.utils.onnx import export_to_onnx
from src.utils.training import compute_kfold_scores, train_pipeline
from src.utils.utils import parse_args

logging.getLogger().setLevel(logging.DEBUG)


def main(a):
    logging.info("Getting data")
    train_data = get_data(a, a.raw_train_path)
    test_data = get_data(a, a.raw_test_path)

    version = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')  # TODO move this to a
    if a.n_splits > 1:
        logging.info(f"Splitting in {a.n_splits} train-test splits")
        kfold = model_selection.KFold(n_splits=a.n_splits)

        for fold, (train_ids, val_ids) in enumerate(kfold.split(train_data)):
            logging.info(f"Fold {fold + 1} out of {a.n_splits}")
            train_split = np.array(train_data)[train_ids].tolist()
            test_split = np.array(train_data)[val_ids].tolist()

            train_pipeline(a, test_split, train_split, version=version, fold=fold)

        compute_kfold_scores(a, version)

    else:
        model, test_ds = train_pipeline(a, test_data, train_data, version=version)

        logging.info("Exporting model artefact")
        export_to_onnx(a, model=model, test_ds=test_ds, version=version)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
