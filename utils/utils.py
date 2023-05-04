import argparse


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Data paths
    parser.add_argument("--raw_train_path", help='Path of the raw train data', type=str, required=False,
                        default='/Users/bas/PycharmProjects/jarvis2/data/sample.json')
    parser.add_argument("--raw_test_path", help='Path of the raw test data', type=str, required=False,
                        default='/Users/bas/PycharmProjects/jarvis2/data/sample.json')
    parser.add_argument("--use_skill_weights", type=bool, required=False, default=False)

    # Model config
    parser.add_argument("--num_heads", type=int, required=False, default=4)
    parser.add_argument("--max_skills", type=int, required=False, default=20)
    parser.add_argument("--max_len", type=int, required=False, default=64,
                        help="Max number of tokens for an input (job title or skill)")

    # Training options
    parser.add_argument("--batch_size", type=int, required=False, default=32)
    parser.add_argument("--learning_rate", type=float, required=False, default=1e-2)
    parser.add_argument("--num_epochs", type=int, required=False, default=5)
    parser.add_argument("--warmup_ratio", type=float, required=False, default=0.1)
    parser.add_argument("--dropout_rate", type=float, required=False, default=0.1)

    return parser.parse_args()
