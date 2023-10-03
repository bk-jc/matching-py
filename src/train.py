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
from src.utils.training import get_callbacks, save_conf_matrix
from src.utils.utils import parse_args


def main(a):
    train_ds = get_data(a, a.raw_train_path)
    test_ds = get_data(a, a.raw_test_path)

    train_ds = preprocess_dataset(train_ds, a, train=True)
    test_ds = preprocess_dataset(test_ds, a, train=False)

    pl_model = get_model(a, train_ds, test_ds)

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

    trainer.fit(
        model=pl_model,
    )

    def get_example_input(ds):
        return {
            "cv": [ds.dataset[0]['cv']],
            "job": [ds.dataset[0]['job']]
        }

    model = pl_model.model

    save_conf_matrix(
        confusion_matrix=pl_model.val_metrics["all"]['confusion_matrix'].compute(),
        csv_logger=trainer.loggers[0]
    )

    _ = model(**get_example_input(test_ds))

    # Export the model to ONNX
    example_input = get_example_input(test_ds)
    onnx_path = os.path.join(a.save_path, a.exp_name, "output", version, "jarvis_v2.onnx")
    os.makedirs(os.path.join(a.save_path, a.exp_name, "output", version), exist_ok=True)
    torch.onnx.export(model, tuple(example_input.values()), onnx_path,
                      input_names=list(example_input.keys()), output_names=["similarity"])


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
