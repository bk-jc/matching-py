import json

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
