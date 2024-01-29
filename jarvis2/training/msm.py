import logging

import torch
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning import Trainer

from jarvis2.data.msm import get_dataloader
from jarvis2.modeling.msm import LightningWrapper, MaskedSkillEncoder
from jarvis2.utils.utils import get_callbacks


def main(a, test_data, train_data, fold):
    train_dataloader = get_dataloader(a, train_data, train=True, batch_size=a.msm_train_batch_size)
    val_dataloader = get_dataloader(a, test_data, train=False, batch_size=a.msm_val_batch_size)

    version = a.version
    if fold is not None:
        version += f"/fold{fold + 1}"
    version += "_msm"

    model = MaskedSkillEncoder(a)

    lightning_model = LightningWrapper(
        model=model,
        a=a,
        train_ds=train_dataloader,
        val_ds=val_dataloader,
    )

    trainer = Trainer(
        accelerator="auto" if (torch.cuda.is_available() and a.use_gpu) else "cpu",
        enable_checkpointing=a.n_splits <= 1,
        max_steps=a.msm_train_steps,
        val_check_interval=a.msm_val_steps,
        callbacks=get_callbacks(a, version),
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        precision=16 if a.fp16 else 32,
        logger=(
            CSVLogger(name=a.exp_name, save_dir=a.save_path, version=version),
            TensorBoardLogger(name=a.exp_name, save_dir=a.save_path, version=version),
        ),  # type: ignore
        check_val_every_n_epoch=None, )

    trainer.fit(
        model=lightning_model
    )

    try:
        lightning_model.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"])
        torch.save(lightning_model.model.encoder.state_dict(),
                   f"{trainer.checkpoint_callback.dirpath}/encoder_weights.pth")
    except Exception as e:
        logging.info(f"Failed to load best model: {e}")

    return lightning_model.model.encoder
