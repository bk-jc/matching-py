import logging
import re

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from jarvis2.data.skills import IGNORED, DUPLICATES, MAPPING


def preprocess(data, a, train):
    if train:
        batch_size = a.train_batch_size
    else:
        batch_size = a.val_batch_size

    logging.info("Preprocessing skills")
    data = preprocess_skills(data, a)

    logging.info("Preprocessing jobtitles")
    data = preprocess_jobtitles(data, a)

    logging.info("Removing documents without skills")
    data = remove_zero_skill_docs(data)

    # For cross-validation, we want to have negative samples for the test dataset too
    logging.info("Creating negative samples")
    if (train or a.n_splits > 0) and a.negative_sampling:
        data = insert_negative_samples(data, a, train)

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


def insert_negative_samples(data, a, train):
    n_samples = int(len(data) * (a.negative_ratio if train else 1))

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


def preprocess_jobtitles(data, a):
    if a.preprocess_jobtitle:
        for docs in tqdm(data, desc="Preprocessing skills"):
            job = docs["job"]
            job["jobtitle"] = preprocess_jobtitle(job["jobtitle"])
    return data


def preprocess_jobtitle(jobtitle):
    jobtitle = jobtitle.replace('&amp;', '&')
    jobtitle = jobtitle.replace('&ndash;', '-')
    jobtitle = jobtitle.replace('&#39;', "'")
    jobtitle = jobtitle.replace('&#8211;', "-")
    jobtitle = jobtitle.replace('&lt;', "<")
    jobtitle = jobtitle.replace('&gt;', ">")
    jobtitle = jobtitle.replace('&#226;', "â")

    # Remove data inbetween parentheses
    jobtitle = re.sub(r" ? \([^)]+\)", "", jobtitle)
    jobtitle = re.sub(r" ?<[^>]+>", "", jobtitle)
    jobtitle = re.sub(r" ?\[[^>]+\]", "", jobtitle)

    # Remove percentages
    jobtitle = re.sub(r'(\d+\s*%?\s*-?\s*)+', "", jobtitle)

    # Remove gender specifications
    # TODO french/ita variants
    jobtitle = re.sub(r'[mwfdMWFD][/][mwfdMWFD]([/][mwfdMWFD])*', '', jobtitle)

    # Standardize uppercasing
    if jobtitle.isupper():
        jobtitle = jobtitle.capitalize()

    # Remove leading/trailing whitespace
    jobtitle = jobtitle.strip()

    # Standardise whitespace
    jobtitle = jobtitle.replace('\r', ' ').replace('\n', ' ')
    jobtitle = re.sub(' +', ' ', jobtitle)

    # Remove German gender indication
    jobtitle = jobtitle.replace("*in", "").replace("/in", "").replace("/-in", "").replace(":in", "").replace(
        "(in)", "")

    # Trailing dashes and space
    jobtitle = re.sub(r'[ \-]+$', '', jobtitle)

    return jobtitle


def preprocess_skills(data, a):
    synonym_mapping = {alias: category[0] for category in DUPLICATES for alias in category[1:]}
    for docs in tqdm(data, desc="Preprocessing skills"):
        for d in (docs["cv"], docs["job"]):
            for i, skill in reversed(list(enumerate(d["skills"]))):
                if a.remove_synonym_skills:
                    d["skills"][i] = synonym_mapping.get(skill, skill)
                if a.remove_strange_skills and skill in IGNORED:
                    del d["skills"][i]
                    continue
                if a.rename_skills:
                    new_skill = MAPPING.get(skill, skill)
                    if new_skill:
                        d["skills"][i] = new_skill

            # Remove duplicates
            d["skills"] = list(set(d["skills"]))

    return data
