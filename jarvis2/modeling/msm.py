import pytorch_lightning
import torch
from torch import nn
from torchmetrics import Accuracy
from transformers import get_linear_schedule_with_warmup

from data.skills import get_mapping
from modeling.encoder import Encoder


class LightningWrapper(pytorch_lightning.LightningModule):

    def __init__(self, model, a, train_ds=None, val_ds=None):
        super(LightningWrapper, self).__init__()
        self.model = model
        self.lr = a.msm_lr
        self.n_steps = a.msm_train_steps
        self.weight_decay = a.msm_weight_decay
        self.train_ds = train_ds
        self.val_ds = val_ds

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=model.readout.out_features)
        self.val_acc = Accuracy(task="multiclass", num_classes=model.readout.out_features)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs['loss']
        preds = torch.argmax(outputs['logits'], dim=1)
        self.train_acc(preds, batch['y'])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs['loss']
        preds = torch.argmax(outputs['logits'], dim=1)
        self.val_acc(preds, batch['y'])
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

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


class MaskedSkillEncoder(nn.Module):

    def __init__(self, a):
        super(MaskedSkillEncoder, self).__init__()

        self.encoder = Encoder(a)
        self.readout = nn.Linear(self.encoder.skill_attention_layer.output.dense.out_features,
                                 len(get_mapping(a).keys()))
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y, mask_idx):
        x = self.encoder(x, return_pre_pooled=True)
        x = self.readout(x)
        # TODO do this without looping
        x = torch.stack([x[idx[0], idx[1]] for idx in mask_idx])

        if y is None:
            return torch.argmax(x, dim=1)
        else:
            return {
                "loss": self.loss_fn(input=x, target=y),
                "logits": x
            }
