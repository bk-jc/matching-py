import sys

from jarvis2.training.main import run_experiment
from jarvis2.training.optuna import grid_search
from jarvis2.utils.utils import parse_args, init_logger_and_seed


def main(args):
    args = init_logger_and_seed(args)

    if args.optuna_path:
        grid_search(args)
    else:
        run_experiment(args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
