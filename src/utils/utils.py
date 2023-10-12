import argparse
import logging
from datetime import datetime
from pathlib import Path

import lightning

log_levels = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the log level')

    # Data options
    parser.add_argument("--raw_train_path", help='Path of the raw train data', type=str, required=False,
                        default=str(Path(__file__).parent.parent.parent / 'data' / 'mock' / 'sample.json'))
    parser.add_argument("--raw_test_path", help='Path of the raw test data', type=str, required=False,
                        default=str(Path(__file__).parent.parent.parent / 'data' / 'mock' / 'sample.json'))
    parser.add_argument("--seed", type=int, required=False, default=1997)
    parser.add_argument("--use_skill_weights", type=bool, required=False, default=False)
    parser.add_argument("--allow_half_label", type=bool, required=False, default=False)
    parser.add_argument("--negative_sampling", type=bool, required=False, default=True)
    parser.add_argument("--negative_ratio", type=float, required=False, default=1,
                        help="How many negatives to sample for each positive sample. Defaults to 1, "
                             "which means the dataset is balanced between positive and negatives samples.")

    # Model config
    parser.add_argument("--model_name", type=str, required=False, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--pooling_mode", type=str, required=False, default="cls", choices=["cls", "max"])
    parser.add_argument("--untrained", type=bool, required=False, default=False)
    parser.add_argument("--num_heads", type=int, required=False, default=4)
    parser.add_argument("--max_skills", type=int, required=False, default=20)
    parser.add_argument("--max_len", type=int, required=False, default=64,
                        help="Max number of tokens for an input (job title or skill)")
    parser.add_argument("--cache_embeddings", type=bool, required=False, default=True)

    # Training options
    parser.add_argument("--train_batch_size", type=int, required=False, default=4)
    parser.add_argument("--val_batch_size", type=int, required=False, default=4)
    parser.add_argument("--learning_rate", type=float, required=False, default=1e-2)
    parser.add_argument("--train_steps", type=int, required=False, default=10)
    parser.add_argument("--val_steps", type=int, required=False, default=5)
    parser.add_argument("--warmup_ratio", type=float, required=False, default=0.1)
    parser.add_argument("--dropout_rate", type=float, required=False, default=0.1)
    parser.add_argument("--weight_decay", type=float, required=False, default=0.01)
    parser.add_argument("--es_delta", type=float, required=False, default=0.01)
    parser.add_argument("--es_patience", type=float, required=False, default=5)
    parser.add_argument("--fp16", type=bool, required=False, default=False)

    # Cross-validation & grid search
    parser.add_argument("--n_splits", type=int, required=False, default=0,
                        help="Number of CV splits. Setting to 0 means no cross-validation.")
    parser.add_argument("--score_metric", type=str, required=False, default="val_all_f1")
    parser.add_argument("--lower_is_better", type=bool, required=False, default=False)
    parser.add_argument("--optuna_path", type=str, required=False, default="")

    # Artefact options
    parser.add_argument("--save_path", type=str, required=False, default="output")
    parser.add_argument("--exp_name", type=str, required=False, default="develop")

    parser.add_argument("--version", type=str, required=False, default=datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

    return parser.parse_args(args)


def init_logger_and_seed(args):
    logging.info("Setting up")
    lightning.seed_everything(args.seed)
    logging.getLogger().setLevel(log_levels[args.log_level])
