import json
from copy import deepcopy

import torch
from datasets import Dataset
from torch.utils.data import DataLoader


def get_data(a, data_filepath):
    data = []

    if 'sample' in data_filepath:
        sample = json.load(open(data_filepath))
        data.append(sample)
        data = data * a.val_steps * a.train_batch_size

        # Synthetically create various numbers of skills
        skill_freqs = {i: a.train_batch_size for i in range(5, 8)}
        for n_skills, freq in skill_freqs.items():
            n_skills_sample = deepcopy(sample)
            n_skills_sample["cv"]["skills"] = ["Python"] * n_skills
            n_skills_sample["job"]["skills"] = ["Python"] * n_skills
            data += [n_skills_sample] * freq

    else:
        with open(data_filepath, 'r') as file:
            for line in file:
                data.append(json.loads(line))

    return Dataset.from_list(data)


def preprocess_dataset(ds, batch_size):
    ds = ignore_empty_skill_docs(ds)
    dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
    return dataloader


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
