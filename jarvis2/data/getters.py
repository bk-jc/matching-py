import json
from copy import deepcopy

import pandas as pd


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
