import logging

import pytorch_lightning
import torch
import torchmetrics
from sklearn.metrics import f1_score
from transformers import get_linear_schedule_with_warmup


class LightningWrapper(pytorch_lightning.LightningModule):
    """PyTorch Lightning wrapper for Jarvis model, such that it is compatible with Lightning trainer"""

    def __init__(
            self,
            a,
            base_model,
            lr,
            weight_decay,
            n_steps,
            n_thresholds,
            train_ds=None,
            val_ds=None,
            finetune_lr=None,
    ):
        super().__init__()
        self.model = base_model
        self.lr = lr
        self.finetune_lr = finetune_lr if finetune_lr is not None else lr
        self.weight_decay = weight_decay
        self.n_steps = n_steps
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.n_thresholds = n_thresholds
        self.lower_is_better = a.lower_is_better
        self.score_metric = a.score_metric
        self.best_score = float('inf') if self.lower_is_better else 0

        self.model.threshold = 0.5  # Initial threshold
        self.best_conf_matrix = None

        # Metrics
        self.metric_splits = {
            "all": lambda x: [True] * len(x["label"]),
            "min_2_skills": lambda x: [
                min(
                    len(x["cv"][i]["skills"]),
                    len(x["job"][i]["skills"])
                ) >= 2 for i in range(len(x["label"]))],
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
        preds = (preds > self.model.threshold).to(torch.int64)  # threshold the predictions
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
                    conf_matrix = metric.compute()

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
        if hasattr(self.model, "cache_path") and self.model.cache:
            torch.save(self.model.cache, self.model.cache_path)
        self.clear_metrics(self.train_metrics)

    def on_validation_epoch_end(self) -> None:
        self.update_threshold()
        for batch, outputs in list(zip(self.val_batches, self.val_outputs)):
            self.update_metrics(batch, outputs, self.val_metrics)
        score = float(self.val_metrics["all"]["f1"].compute().numpy())
        if score > self.best_score:
            self.best_score = score
            self.best_conf_matrix = self.val_metrics["all"]["confusion_matrix"].compute()
        self.log_metrics(self.val_metrics, prefix="val")
        self.log("val_loss", torch.stack(self.val_step_loss).mean())
        self.val_step_loss.clear()

        self.val_sims = []
        self.val_labels = []
        self.val_outputs = []
        self.val_labels = []
        self.clear_metrics(self.val_metrics)

    def update_threshold(self):
        best_threshold = 0.5
        current_best_score = float('inf') if self.lower_is_better else 0

        for value in range(0, self.n_thresholds):
            threshold = value / self.n_thresholds

            if self.score_metric == "val_all_f1":
                score = f1_score(self.val_labels, [1 if p >= threshold else 0 for p in self.val_sims])
            else:
                logging.warning(
                    f"Thresholding is not implemented for score_metric {self.score_metric}. Setting threshold to 0.5")
                break
                
            if (self.lower_is_better and score < current_best_score) or (
                    (not self.lower_is_better) and score > current_best_score):
                current_best_score = score
                best_threshold = threshold

        if (self.lower_is_better and current_best_score < self.best_score) or (
                (not self.lower_is_better) and current_best_score > self.best_score):
            self.best_score = current_best_score
            self.model.threshold = best_threshold

    def train_dataloader(self):
        return self.train_ds

    def val_dataloader(self):
        return self.val_ds

    def configure_optimizers(self):
        training_params = [(n, p) for (n, p) in self.model.named_parameters() if p.requires_grad]
        param_groups = [
            {'params': [p for (n, p) in training_params if "ffn_readout" not in n], 'lr': self.finetune_lr},
            {'params': [p for (n, p) in training_params if "ffn_readout" in n], 'lr': self.lr},
        ]
        optimizer = torch.optim.AdamW(
            params=param_groups,
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
