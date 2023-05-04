from utils.utils import parse_args
from data.training_data_reader import get_data
from model.model import get_model_fn, get_tokenizer
from transformers import TrainingArguments
from utils.training import compute_metrics, CustomCallback, CustomTrainer
import torch


def main(a):
    train_ds = get_data(a.raw_train_path)
    test_ds = get_data(a.raw_test_path)

    tokenizer = get_tokenizer()

    train_ds = preprocess_dataset(train_ds, tokenizer, a)
    test_ds = preprocess_dataset(test_ds, tokenizer, a)

    training_args = TrainingArguments(
        output_dir="test_trainer",
        do_eval=True,
        # report_to="all",
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
        # weight_decay=0.01
    )

    trainer = CustomTrainer(
        model_init=get_model_fn(a),
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        # tokenizer=tokenizer
    )

    trainer.train()
    print("finished")


def preprocess_dataset(ds, tokenizer, a):

    def tokenize_fn(x):

        for doctype in ["cv", "job"]:

            x.data[doctype]["skills"] = tokenizer(x.data[doctype]["skills"], max_length=a.max_len).data

        return x.data

    ds = ignore_empty_skill_docs(ds)
    # ds = ds.map(lambda x: tokenize_fn(x))

    return ds


def ignore_empty_skill_docs(ds):
    ignore_idx = [i for i, doc in enumerate(ds) if not doc["cv"]["skills"] or not doc["job"]["skills"]]
    ds = ds.select(
        (
            i for i in range(len(ds))
            if i not in set(ignore_idx)
        )
    )
    return ds


def tokenize_ds(tokenize_fn, ds):

    doctypes = ["cv", "job"]
    for doctype in doctypes:

        for i, doc in enumerate(ds[doctype]):
            ds[i][doctype]["skills"] = tokenize_fn(doc["skills"]).data

    return ds


def collate_fn(inputs):

    cv = [i["cv"] for i in inputs]
    job = [i["job"] for i in inputs]
    label = torch.tensor([i["label"] for i in inputs])

    return {"cv": cv, "job": job, "label": label}


if __name__ == '__main__':
    args = parse_args()
    main(args)
