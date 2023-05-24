from copy import deepcopy

import numpy as np
from transformers import TrainerCallback, Trainer


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


class CustomTrainer(Trainer):

    def create_optimizer(self):
        return Trainer.create_optimizer(self)
