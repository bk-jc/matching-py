import pytorch_lightning
import torch
import torchmetrics
from sklearn.metrics import f1_score
from torch import nn as nn
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from src.jarvis import Jarvis


def get_model_fn(a):
    def get_model():
        return Jarvis(a=a, model_name=a.model_name, emb_size=300, tokenizer=get_tokenizer(a))

    return get_model


class ModuleJarvis(pytorch_lightning.LightningModule):
    """PyTorch Lightning wrapper for Jarvis model, such that it is compatible with Lightning trainer"""

    def __init__(
            self,
            base_model,
            lr,
            weight_decay,
            n_steps,
            n_thresholds,
            train_ds=None,
            val_ds=None,
    ):
        super().__init__()
        self.model = base_model
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_steps = n_steps
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.n_thresholds = n_thresholds

        self.threshold = 0.5

        # Freeze transformer model
        for p in self.model.base_model.parameters():
            p.requires_grad = False

        # Metrics
        self.metric_splits = {
            "all": lambda x: [True] * len(x["label"]),
            "min_6_skills": lambda x: [
                min(
                    len(x["cv"][i]["skills"]),
                    len(x["job"][i]["skills"])
                ) >= 6 for i in range(len(x["label"]))]
        }
        self.train_metrics = {}
        self.val_metrics = {}
        for split_name in self.metric_splits.keys():
            self.train_metrics[split_name] = self.get_metric_dict()
            self.val_metrics[split_name] = self.get_metric_dict()
        self.train_step_loss = []
        self.val_step_loss = []

        # For getting threshold
        self.val_sims = []
        self.val_labels = []
        self.val_batches = []
        self.val_outputs = []

    @staticmethod
    def get_metric_dict():
        return {
            'precision': torchmetrics.Precision(task="binary", num_classes=2),
            'recall': torchmetrics.Recall(task="binary", num_classes=2),
            'f1': torchmetrics.F1Score(task="binary", num_classes=2),
            'confusion_matrix': torchmetrics.ConfusionMatrix(task="binary", num_classes=2),
        }

    def update_metrics(self, batch, outputs, metrics):
        target, preds = batch['label'], outputs['sim']
        preds = (preds > self.threshold).to(torch.int64)  # threshold the predictions
        for split_name, filter_fn in self.metric_splits.items():
            filtered_idx = filter_fn(batch)
            if True in filtered_idx:
                for metric in metrics[split_name].values():
                    metric(preds=preds[filtered_idx], target=target[filtered_idx])

    def log_metrics(self, metrics, prefix):
        for split_name in self.metric_splits.keys():
            for metric_key, metric in metrics[split_name].items():
                if metric_key != "confusion_matrix":
                    self.log(f"{prefix}_{split_name}_{metric_key}", metric.compute())
                else:
                    conf_matrix = metric.confmat

                    TP = conf_matrix[1, 1]
                    TN = conf_matrix[0, 0]
                    FP = conf_matrix[0, 1]
                    FN = conf_matrix[1, 0]

                    # Compute FPR and FNR
                    FPR = FP / (FP + TN)
                    FNR = FN / (FN + TP)

                    self.log(f"{prefix}_{split_name}_FPR", FPR)
                    self.log(f"{prefix}_{split_name}_FNR", FNR)

    def clear_metrics(self, metrics):
        for split_name in self.metric_splits.keys():
            for metric_key, metric in metrics[split_name].items():
                metric.reset()

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        self.update_metrics(batch, outputs, self.train_metrics)
        self.train_step_loss.append(outputs['loss'])
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        self.val_step_loss.append(outputs['loss'])
        self.val_sims += outputs['sim'].tolist()
        self.val_labels += batch['label'].tolist()
        self.val_batches.append(batch)
        self.val_outputs.append(outputs)
        return outputs

    def on_train_epoch_end(self) -> None:
        self.log_metrics(self.train_metrics, prefix="train")
        self.log("train_loss", torch.stack(self.train_step_loss).mean())
        self.train_step_loss.clear()

    def on_validation_epoch_end(self) -> None:
        self.update_threshold()
        for batch, outputs in list(zip(self.val_batches, self.val_outputs)):
            self.update_metrics(batch, outputs, self.val_metrics)
        self.log_metrics(self.val_metrics, prefix="val")
        self.log("val_loss", torch.stack(self.val_step_loss).mean())
        self.val_step_loss.clear()

        # self.clear_metrics(self.val_metrics)
        self.val_sims = []
        self.val_labels = []
        self.val_outputs = []
        self.val_labels = []

    def update_threshold(self):

        best_threshold = 0
        best_f1 = 0

        for value in range(0, self.n_thresholds):
            threshold = value / self.n_thresholds

            f1 = f1_score(self.val_labels, [1 if p >= threshold else 0 for p in self.val_sims])

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        self.threshold = best_threshold

    def train_dataloader(self):
        return self.train_ds

    def val_dataloader(self):
        return self.val_ds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=[p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=self.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_training_steps=self.n_steps,
            num_warmup_steps=int(0.1 * self.n_steps)
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]


def get_model(a, train_ds, val_ds):
    base_model = Jarvis(a=a, model_name=a.model_name, emb_size=300, tokenizer=get_tokenizer(a))
    return ModuleJarvis(base_model=base_model, lr=a.learning_rate, weight_decay=a.weight_decay, n_steps=a.train_steps,
                        n_thresholds=a.n_thresholds, train_ds=train_ds, val_ds=val_ds)


def get_tokenizer(a):
    return AutoTokenizer.from_pretrained(a.model_name)
