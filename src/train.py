import logging
import os
import sys
from datetime import datetime

import torch
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from src.data import get_data
from src.data import preprocess_dataset
from src.model import get_model
from src.utils.onnx import export_to_onnx
from src.utils.training import get_callbacks, save_conf_matrix
from src.utils.utils import parse_args


def main(a):
    logging.info("Getting data")
    train_ds = get_data(a, a.raw_train_path)
    test_ds = get_data(a, a.raw_test_path)

    logging.info("Preprocessing data")
    train_ds = preprocess_dataset(train_ds, a, train=True)
    test_ds = preprocess_dataset(test_ds, a, train=False)

    logging.info("Loading model")
    pl_model = get_model(a, train_ds, test_ds)

    logging.info("Loading trainer")
    version = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    trainer = Trainer(
        accelerator="auto" if (torch.cuda.is_available() and a.use_gpu) else "cpu",
        max_steps=a.train_steps,
        val_check_interval=a.val_steps,
        callbacks=get_callbacks(a, version),
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        precision=16 if a.fp16 else 32,
        logger=(
            CSVLogger(name=a.save_path, save_dir=os.path.join(a.save_path, a.exp_name), version=version),
            TensorBoardLogger(name=a.save_path, save_dir=os.path.join(a.save_path, a.exp_name), version=version),
        ),
    )

    logging.info("Starting training")
    trainer.fit(
        model=pl_model,
    )

    save_conf_matrix(
        confusion_matrix=pl_model.val_metrics["all"]['confusion_matrix'].compute(),
        csv_logger=trainer.loggers[0]
    )

    logging.info("Exporting model artefact")
    export_to_onnx(a, model=pl_model.model, test_ds=test_ds, version=version)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
