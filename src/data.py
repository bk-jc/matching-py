import json

import torch
from datasets import Dataset


def get_data(data_filepath):
    data = []

    if 'sample' in data_filepath:
        data.append(json.load(open(data_filepath)))
        data = data * 500
    else:
        with open(data_filepath, 'r') as file:
            for line in file:
                data.append(json.loads(line))

    return Dataset.from_list(data)


def preprocess_dataset(ds, tokenizer, a):
    def tokenize_fn(x):
        for doctype in ["cv", "job"]:
            x[doctype]["skills"] = tokenizer(
                x[doctype]["skills"],
                max_length=a.max_len,
                return_tensors='pt',
                padding="max_length"
            ).data

        return x

    ds = ignore_empty_skill_docs(ds)

    # TODO cleanup; tokenization is done inside the model
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
