import csv
import logging
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer
from transformers import TrainerCallback

from jarvis2.data.data import preprocess
from jarvis2.modeling.model import get_model
from training import msm
from utils.utils import get_callbacks


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


def save_conf_matrix(confusion_matrix, csv_logger):
    if confusion_matrix is None:
        return
    labels = ['Negative', 'Positive']  # Labels for the two classes
    plt.figure(figsize=(6, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap='viridis')

    plt.title(f"Confusion Matrix (n={torch.sum(confusion_matrix).numpy()})")
    plt.colorbar()

    # Add sample counts to each box
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(confusion_matrix[i, j].numpy()), horizontalalignment='center',
                     verticalalignment='center', color='white')

    # Calculate row and column sums
    row_sums = torch.sum(confusion_matrix, axis=1).numpy()
    col_sums = torch.sum(confusion_matrix, axis=0).numpy()

    # Add row and column n_sample sums to the axis ticks
    row_labels = [f"{label} ({sum_})" for label, sum_ in zip(labels, row_sums)]
    col_labels = [f"{label} ({sum_})" for label, sum_ in zip(labels, col_sums)]

    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, col_labels)
    plt.yticks(tick_marks, row_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    # Save to png
    save_folder = csv_logger.log_dir
    save_path = os.path.join(save_folder, 'confusion_matrix.png')
    plt.savefig(save_path)


def compute_kfold_scores(a, version):
    base_path = os.path.join(a.save_path, a.exp_name, version)
    if a.score_metric.startswith("msm_"):
        csv_paths = [os.path.join(base_path, f"fold{fold}_msm", "metrics.csv") for fold in range(1, a.n_splits + 1)]
    else:
        csv_paths = [os.path.join(base_path, f"fold{fold}", "metrics.csv") for fold in range(1, a.n_splits + 1)]
    get_score_fn = get_msm_csv_score if a.score_metric.startswith("msm_") else get_csv_score

    kfold_score = np.mean([get_score_fn(a, csv_path) for csv_path in csv_paths])

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


def get_msm_csv_score(a, csv_path):
    csv_file = pd.read_csv(csv_path)
    try:
        fold_score = csv_file[a.score_metric].max()
    except:
        logging.warning(f"Could not use given metric {a.score_metric} - using default MSM metric of 'val_acc_epoch'")
        fold_score = csv_file["val_acc_epoch"].max()
    return fold_score


def train_pipeline(a, test_data, train_data, fold=None):
    encoder = None
    if a.do_msm:
        logging.info("Performing MSM")
        encoder = msm.main(a, test_data, train_data, fold=fold)

    logging.info("Preprocessing data")
    train_ds = preprocess(train_data, a, train=True)
    test_ds = preprocess(test_data, a, train=False)

    logging.info("Loading model")
    pl_model = get_model(a, train_ds, test_ds, encoder)

    logging.info("Loading trainer")
    version = a.version
    if fold is not None:
        version += f"/fold{fold + 1}"
    trainer = Trainer(
        accelerator="auto" if (torch.cuda.is_available() and a.use_gpu) else "cpu",
        enable_checkpointing=a.n_splits <= 1,
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

    try:
        pl_model.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"])
    except Exception as e:
        logging.info(f"Failed to load best model: {e}")

    save_conf_matrix(
        confusion_matrix=pl_model.best_conf_matrix,
        csv_logger=trainer.loggers[0]
    )

    return pl_model.model, test_ds
