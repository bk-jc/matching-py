from copy import deepcopy

import numpy as np
import pytorch_lightning as pl
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


def get_callbacks(a):
    return [
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_on_train_epoch_end=False,
            dirpath=a.save_path,
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
