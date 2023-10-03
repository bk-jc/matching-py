import os
from copy import deepcopy

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from transformers import TrainerCallback


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
            dirpath=os.path.join(a.exp_name, a.save_path, "output", version),
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
