import csv
import logging
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer
from transformers import TrainerCallback

from src.data import preprocess_data
from src.model import get_model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    idx = np.where(labels != 0)
    return {
        "mae": np.sum(np.abs(logits[idx] - labels[idx])) / len(logits),
        "accuracy": np.mean(np.argmax(labels, axis=-1) == np.argmax(logits, axis=-1))
    }


class CustomCallback(TrainerCallback):

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        control_copy = deepcopy(control)
        self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
        return control_copy


def get_callbacks(a, version):
    return [
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_on_train_epoch_end=False,
            dirpath=os.path.join(a.save_path, a.exp_name, version),
            every_n_epochs=1
        ),
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=a.es_delta,
            patience=a.es_patience,
            verbose=True,
            mode="min",
            check_on_train_epoch_end=False,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step")
    ]


def save_conf_matrix(confusion_matrix, csv_logger):
    labels = ['Negative', 'Positive']  # Labels for the two classes
    plt.figure(figsize=(4, 4))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)

    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = torch.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    # Save to png
    save_folder = csv_logger.log_dir
    save_path = os.path.join(save_folder, 'confusion_matrix.png')
    plt.savefig(save_path)


def compute_kfold_scores(a, version):
    base_path = os.path.join(a.save_path, a.exp_name, version)
    cvs_paths = [os.path.join(base_path, f"fold{fold}", "metrics.csv") for fold in range(1, a.n_splits + 1)]
    fold_scores = []
    for csv_path in cvs_paths:
        fold_score = get_csv_score(a, csv_path)
        fold_scores.append(fold_score)

    kfold_score = np.mean(fold_scores)

    with open(os.path.join(base_path, 'kfold_score.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["kfold_metric",
             f"kfold_score (metric={a.score_metric},"
             f" lower_is_better={a.lower_is_better})"
             ])
        writer.writerow([a.score_metric, kfold_score])

    return kfold_score


def get_csv_score(a, csv_path):
    csv_file = pd.read_csv(csv_path)
    fold_score = csv_file[a.score_metric].min() if a.lower_is_better else csv_file[a.score_metric].max()
    return fold_score


def train_pipeline(a, test_data, train_data, fold=None):
    logging.info("Preprocessing data")
    train_ds = preprocess_data(train_data, a, train=True)
    test_ds = preprocess_data(test_data, a, train=False)

    logging.info("Loading model")
    pl_model = get_model(a, train_ds, test_ds)

    logging.info("Loading trainer")
    version = a.version
    if fold is not None:
        version += f"/fold{fold + 1}"
    trainer = Trainer(
        accelerator="auto" if (torch.cuda.is_available() and a.use_gpu) else "cpu",
        max_steps=a.train_steps,
        val_check_interval=a.val_steps,
        callbacks=get_callbacks(a, version),
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        precision=16 if a.fp16 else 32,
        logger=(
            CSVLogger(name=a.exp_name, save_dir=a.save_path, version=version),
            TensorBoardLogger(name=a.exp_name, save_dir=a.save_path, version=version),
        ),  # type: ignore
        check_val_every_n_epoch=None,
    )

    logging.info("Starting training")
    trainer.fit(
        model=pl_model,
    )
    save_conf_matrix(
        confusion_matrix=pl_model.val_metrics["all"]['confusion_matrix'].compute(),
        csv_logger=trainer.loggers[0]
    )

    return pl_model.model, test_ds
