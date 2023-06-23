import os
import sys

import torch
from pytorch_lightning import Trainer

from src.data import get_data
from src.data import preprocess_dataset
from src.model import get_model
from src.utils.training import get_callbacks
from src.utils.utils import parse_args


def main(a):
    train_ds = get_data(a, a.raw_train_path)
    test_ds = get_data(a, a.raw_test_path)

    train_ds = preprocess_dataset(train_ds, a.train_batch_size)
    test_ds = preprocess_dataset(test_ds, a.val_batch_size)

    pl_model = get_model(a, train_ds, test_ds)

    trainer = Trainer(
        accelerator="auto" if (torch.cuda.is_available() and a.use_gpu) else "cpu",
        max_steps=a.train_steps,
        val_check_interval=a.val_steps,
        callbacks=get_callbacks(a),
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        precision=16 if a.fp16 else 32,
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

    _ = model(**get_example_input(test_ds))

    # Export the model to ONNX
    example_input = get_example_input(test_ds)
    onnx_path = os.path.join(a.save_path, "jarvis_v2.onnx")
    os.makedirs(a.save_path, exist_ok=True)
    torch.onnx.export(model, tuple(example_input.values()), onnx_path,
                      input_names=list(example_input.keys()), output_names=["similarity"])


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
