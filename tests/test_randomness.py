from pathlib import Path

import torch

from src.data import preprocess_data
from src.train import get_data
from src.utils.utils import parse_args, init_logger_and_seed
from tests.constants import SHARED_ARGS


def test_data_sampling(tmpdir):
    n_batches = 10

    a_seed1 = get_seed_args(tmpdir, seed=1)
    a_seed2 = get_seed_args(tmpdir, seed=2)

    batches1 = get_batches(a_seed1, n_batches)
    batches2 = get_batches(a_seed1, n_batches)
    batches3 = get_batches(a_seed2, n_batches)

    # Sampling with same random seed results in same batches
    assert all(same_batch(b1, b2) for b1, b2 in list(zip(batches1, batches2)))

    # Sampling with different random seed results in different batches
    assert not all(same_batch(b1, b2) for b1, b2 in list(zip(batches1, batches3)))


def same_batch(b1, b2):
    return b1["cv"] == b2["cv"] and b1["job"] == b2["job"] and torch.equal(b1["label"], b2["label"])


def get_batches(a, n_batches):
    init_logger_and_seed(a)
    ds1 = get_dataset(a)
    batches1 = [ds1.__iter__().__next__() for _ in range(n_batches)]
    return batches1


def get_seed_args(tmpdir, seed):
    return parse_args([
        "--save_path", str(tmpdir),
        "--seed", str(seed),
        "--negative_sampling", True
                               * SHARED_ARGS
    ])


def get_dataset(a):
    data = get_data(a, str(Path(__file__).parent.parent / 'data' / 'mock' / 'sample.json'))
    dataset = preprocess_data(data, a, train=True)
    return dataset
