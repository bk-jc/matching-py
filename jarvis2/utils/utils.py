import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import git
import lightning
import pytorch_lightning as pl
import transformers
import yaml

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
    parser.add_argument("--allow_half_label", type=bool, required=False, default=False)
    parser.add_argument("--negative_sampling", type=bool, required=False, default=True)
    parser.add_argument("--negative_ratio", type=float, required=False, default=1,
                        help="How many negatives to sample for each positive sample. Defaults to 1, "
                             "which means the dataset is balanced between positive and negatives samples.")

    # Data augmentation
    parser.add_argument("--remove_synonym_skills", type=bool, required=False, default=False)
    parser.add_argument("--remove_strange_skills", type=bool, required=False, default=False)
    parser.add_argument("--rename_skills", type=bool, required=False, default=False)
    parser.add_argument("--p_add_skill", type=float, required=False, default=0.)
    parser.add_argument("--p_remove_skill", type=float, required=False, default=0.)
    parser.add_argument("--p_change_skill", type=float, required=False, default=0.)
    parser.add_argument("--preprocess_jobtitle", type=bool, required=False, default=False)

    # Model config
    parser.add_argument("--model_name", type=str, required=False, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--pretrained_path", type=str, required=False, default="")  # TODO implement
    parser.add_argument("--pooling_mode", type=str, required=False, default="cls", choices=["cls", "max", "mean"])
    parser.add_argument("--num_heads", type=int, required=False, default=4)
    parser.add_argument("--max_skills", type=int, required=False, default=20)
    parser.add_argument("--max_len", type=int, required=False, default=64,
                        help="Max number of tokens for an input (job title or skill)")
    parser.add_argument("--cache_embeddings", type=bool, required=False, default=True)
    parser.add_argument("--hidden_dim", type=int, required=False, default=300)
    parser.add_argument("--readout_dim", type=int, required=False, default=300)
    parser.add_argument("--n_ffn_blocks_emb", type=int, required=False, default=1)
    parser.add_argument("--n_ffn_blocks_readout", type=int, required=False, default=1)
    parser.add_argument("--skill_prefix", type=str, required=False, default="")
    parser.add_argument("--use_jobtitle", type=bool, required=False, default=False)
    parser.add_argument("--alpha", type=float, required=False, default=0)
    parser.add_argument("--siamese", type=bool, required=False, default=True,
                        help="When enabled, share weights for job and CV embeddings")

    # Training options
    parser.add_argument("--train_batch_size", type=int, required=False, default=4)
    parser.add_argument("--val_batch_size", type=int, required=False, default=4)
    parser.add_argument("--learning_rate", type=float, required=False, default=1e-2)
    parser.add_argument("--train_steps", type=int, required=False, default=10)
    parser.add_argument("--val_steps", type=int, required=False, default=5)
    parser.add_argument("--warmup_ratio", type=float, required=False, default=0.1)
    parser.add_argument("--dropout_rate", type=float, required=False, default=0.1)
    parser.add_argument("--weight_decay", type=float, required=False, default=1e-4)
    parser.add_argument("--es_delta", type=float, required=False, default=0.01)
    parser.add_argument("--es_patience", type=float, required=False, default=5)
    parser.add_argument("--fp16", type=bool, required=False, default=False)
    parser.add_argument("--n_workers", type=int, required=False, default=0)
    parser.add_argument("--loss_fn", type=str, required=False, default="contrastive", choices=["contrastive", "cosine"])
    parser.add_argument("--pos_label_bias", type=float, required=False, default=0.)
    parser.add_argument("--neg_label_bias", type=float, required=False, default=0.)
    parser.add_argument("--n_thresholds", type=int, required=False, default=100)

    # Masked skill modelling
    parser.add_argument("--do_msm", type=bool, required=False, default=False)
    parser.add_argument("--msm_lr", type=float, required=False, default=1e-4)
    parser.add_argument("--msm_weight_decay", type=float, required=False, default=1e-8)
    parser.add_argument("--msm_train_batch_size", type=int, required=False, default=128)
    parser.add_argument("--msm_val_batch_size", type=int, required=False, default=512)
    parser.add_argument("--msm_train_steps", type=int, required=False, default=1000)
    parser.add_argument("--msm_val_steps", type=int, required=False, default=100)
    parser.add_argument("--ignore_old_skills", type=bool, required=False, default=True)
    parser.add_argument("--finetune_lr", type=float, required=False, default=1e-7)

    # Cross-validation & grid search
    parser.add_argument("--n_splits", type=int, required=False, default=0,
                        help="Number of CV splits. Setting to 0 means no cross-validation.")
    parser.add_argument("--group_hashed", type=bool, required=False, default=True)
    parser.add_argument("--group_name", type=str, choices=["cv", "job"], required=False, default="cv")
    parser.add_argument("--score_metric", type=str, required=False, default="val_all_f1")
    parser.add_argument("--lower_is_better", type=bool, required=False, default=False)
    parser.add_argument("--optuna_path", type=str, required=False, default="")
    parser.add_argument("--random_sampler", type=bool, required=False, default=False)

    # Artefact options
    parser.add_argument("--save_path", type=str, required=False, default=Path(__file__).parent.parent.parent / "output")
    parser.add_argument("--exp_name", type=str, required=False, default="develop")

    parser.add_argument("--version", type=str, required=False, default=datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

    a = parser.parse_args(args)
    validate_args(a)

    return a


def init_logger_and_seed(args):
    logging.info("Setting up")
    base_path = Path(args.save_path) / args.exp_name / args.version
    os.makedirs(base_path, exist_ok=True)

    args.commit_hash = get_current_git_commit_hash()
    if args.n_workers:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
    seed_everything(args)

    # Set up the logger
    logger = logging.getLogger()
    logger.setLevel(log_levels[args.log_level])

    file_handler = logging.FileHandler(base_path / "out.log")
    file_handler.setLevel(log_levels[args.log_level])
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_levels[args.log_level])
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return args


def seed_everything(args):
    lightning.seed_everything(args.seed)
    transformers.set_seed(args.seed)


def get_current_git_commit_hash():
    return git.Repo(search_parent_directories=True).head.object.hexsha


def persist_args(a):
    os.makedirs(Path(a.save_path) / a.exp_name / a.version, exist_ok=True)
    with open(Path(a.save_path) / a.exp_name / a.version / "args.yaml", 'w') as yaml_file:
        yaml.dump(a, yaml_file)


def validate_args(a):
    if a.pooling_mode == "cls" and a.alpha > 0:
        raise ValueError(
            "CLS pooling with a positive alpha does not make sense, because you compute the same thing twice."
        )
    if a.do_msm and a.pooling_mode != "cls":
        raise ValueError(
            f"Masked skill modelling without CLS pooling does not make sense. Configured pooling method:"
            f" {a.pooling_mode}"
        )
    if a.do_msm and not a.siamese:
        raise NotImplementedError(
            "Masked skill modelling for non-siamese networks is currently not implemented."
        )
    if a.do_msm and a.pretrained_path:
        raise ValueError(
            "You cannot use a pretrained model for doing MSM. Either disable do_msm or set pretrained_path to an empty"
            " string"
        )


def get_callbacks(a, version):
    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=a.es_delta,
            patience=a.es_patience,
            verbose=True,
            mode="min",
            check_on_train_epoch_end=False,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step")
    ]

    if a.n_splits <= 1:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                save_on_train_epoch_end=False,
                dirpath=os.path.join(a.save_path, a.exp_name, version),
                every_n_epochs=1
            ),
        )

    return callbacks
