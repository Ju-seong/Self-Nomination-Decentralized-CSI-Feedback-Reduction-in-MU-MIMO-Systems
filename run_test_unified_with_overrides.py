import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run selfnomination_project/test_unified.py after overriding selected config values."
    )
    parser.add_argument(
        "--num_users_override",
        type=int,
        required=True,
        help="Value assigned to config.num_users before importing test_unified.",
    )
    args, remaining = parser.parse_known_args()
    return args, remaining


def main():
    args, remaining = parse_args()

    import config

    config.num_users = args.num_users_override

    import test_unified

    sys.argv = ["test_unified.py", *remaining]
    test_unified.main()


if __name__ == "__main__":
    main()
