import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from data.skills import get_skill_to_idx, MAPPING
from utils.msm import MASK_TOKEN


# TODO deduplicate?
def get_msm_pairs(a, data):  # TODO this mask token should probably be custom
    skill_to_idx = get_skill_to_idx(a)
    msm_pairs = []
    # TODO weight documents inversely with number of skills to avoid skewed dataset
    for d in data:
        if a.ignore_old_skills:
            d["skills"] = [s for s in d["skills"] if s in MAPPING.keys()]
        if len(d["skills"]) <= 1:
            continue
        if len(d["skills"]) > a.max_skills + 1:
            continue  # TODO this could be more clever by e.g. sampling subsets of max_len skills
        for i, s in enumerate(d["skills"]):
            doc = {k: v for k, v in d.items() if k != "skills"}
            doc["skills"] = [*d["skills"][:i], MASK_TOKEN, *d["skills"][i + 1:]]
            msm_pairs.append({
                "x": doc,
                "y": skill_to_idx[s],
                "mask_idx": i,
            })
    return msm_pairs


def get_dataloader(a, data, train: bool, batch_size: int):
    cv_data = [d["cv"] for d in data]
    job_data = [d["job"] for d in data]

    msm_pairs = []
    msm_pairs += get_msm_pairs(a, cv_data)
    msm_pairs += get_msm_pairs(a, job_data)

    ds = Dataset.from_list(msm_pairs)

    def collate_fn(inputs):
        return {
            "x": [i["x"] for i in inputs],
            "y": torch.tensor([i["y"] for i in inputs]),
            "mask_idx": torch.tensor([[idx, i["mask_idx"]] for idx, i in enumerate(inputs)])
        }

    return DataLoader(
        dataset=ds,
        collate_fn=collate_fn,
        shuffle=train,
        batch_size=batch_size,
        num_workers=0
    )
