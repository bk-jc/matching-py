import logging

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader


def preprocess(data, a, train):
    if train:
        batch_size = a.train_batch_size
    else:
        batch_size = a.val_batch_size

    logging.info("Removing documents without skills")
    data = remove_zero_skill_docs(data)

    # For cross-validation, we want to have negative samples for the test dataset too
    logging.info("Creating negative samples")
    if (train or a.n_splits > 0) and a.negative_sampling:
        data = insert_negative_samples(data, a)

    return DataLoader(
        dataset=Dataset.from_list(data),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=train,
        **({
               "num_workers": a.n_workers,
               "multiprocessing_context": "fork",
               "persistent_workers": True,
           } if a.n_workers else {})
    )


def insert_negative_samples(data, a):
    n_samples = int(len(data) * a.negative_ratio)

    samples = []
    for _ in range(n_samples):
        i = np.random.choice(len(data))
        j = (i + np.random.choice(len(data) - 2) + 1) % len(data)  # Make sure that j != i
        samples.append({
            "cv": data[i]["cv"],
            "job": data[j]["job"],
            "label": 0.,
        })

    return samples + data


def remove_zero_skill_docs(data):
    ignore_idx = [i for i, doc in enumerate(data) if not doc["cv"]["skills"] or not doc["job"]["skills"]]
    select_idx = [i for i in range(len(data)) if i not in set(ignore_idx)]
    return np.array(data)[select_idx].tolist()


def collate_fn(inputs):
    cv = [i["cv"] for i in inputs]
    job = [i["job"] for i in inputs]
    label = torch.tensor([i["label"] for i in inputs])

    return {"cv": cv, "job": job, "label": label}
