import sys

from jarvis2.utils.utils import parse_args, init_logger_and_seed
from training.main import run_experiment
from training.optuna import grid_search


def main(args):
    args = init_logger_and_seed(args)

    if args.optuna_path:
        grid_search(args)
    else:
        run_experiment(args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
