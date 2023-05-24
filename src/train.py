import torch
from transformers import TrainingArguments

from src.data import get_data
from src.data import preprocess_dataset, collate_fn
from src.model import get_model_fn, get_tokenizer
from src.utils.training import compute_metrics, CustomTrainer
from src.utils.utils import parse_args


def main(a):
    train_ds = get_data(a.raw_train_path)
    test_ds = get_data(a.raw_test_path)

    tokenizer = get_tokenizer(a)

    train_ds = preprocess_dataset(train_ds, tokenizer, a)
    test_ds = preprocess_dataset(test_ds, tokenizer, a)

    training_args = TrainingArguments(
        output_dir="test_trainer",
        do_eval=True,
        evaluation_strategy="steps",
        logging_strategy="steps",
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size,
        learning_rate=a.learning_rate,
        logging_steps=1,
        log_on_each_node=False,
        num_train_epochs=a.num_epochs,
        overwrite_output_dir=True,
        run_name="dev",
        warmup_ratio=a.warmup_ratio,
        eval_steps=10,
        fp16=False,
        prediction_loss_only=True,
        weight_decay=a.weight_decay,
    )

    trainer = CustomTrainer(
        model_init=get_model_fn(a),
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    trainer.train()

    def get_example_input(ds):
        return {
            "cv": [ds[0]['cv']],
            "job": [ds[0]['job']]
        }

    _ = trainer.model(**get_example_input(test_ds))

    # Export the model to ONNX
    example_input = get_example_input(test_ds)
    onnx_path = "jarvis_v2.onnx"
    torch.onnx.export(trainer.model, tuple(example_input.values()), onnx_path,
                      input_names=list(example_input.keys()), output_names=["similarity"])


if __name__ == '__main__':
    args = parse_args()
    main(args)
