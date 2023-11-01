import json
import logging
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader


def get_data(a, data_filepath):
    data = []

    if data_filepath.endswith(".json"):

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

    elif data_filepath.endswith(".csv"):
        csv_data = pd.read_csv(data_filepath, header=None)

        data = []
        for _, entry in csv_data.iterrows():

            if entry[3] == 0.5:
                if not a.allow_half_label:
                    continue
                # Assume B-bucket is A-bucket
                entry[3] = 1

            if type(entry[0]) != str or type(entry[1]) != str:
                continue  # TODO should probly be converted to empty list

            sample = {
                "cv": {
                    "skills": entry[0].split(",") if entry[0] else [],
                    "jobtitle": ""
                },
                "job": {
                    "skills": entry[1].split(",") if entry[1] else [],
                    "jobtitle": entry[2] if type(entry[2]) == str else ""
                },
                "label": float(entry[3])
            }
            data.append(sample)

    else:
        raise NotImplementedError

    return data


def preprocess_data(data, a, train):
    if train:
        batch_size = a.train_batch_size
    else:
        batch_size = a.val_batch_size

    logging.info("Removing documents without skills")
    data = ignore_empty_skill_docs(data)

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


def ignore_empty_skill_docs(data):
    ignore_idx = [i for i, doc in enumerate(data) if not doc["cv"]["skills"] or not doc["job"]["skills"]]
    select_idx = [i for i in range(len(data)) if i not in set(ignore_idx)]
    return np.array(data)[select_idx].tolist()


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
