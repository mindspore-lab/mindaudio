import argparse

import yaml
from easydict import EasyDict as ed


def create_parser():
    parser_config = argparse.ArgumentParser(description="Training", add_help=False)
    parser_config.add_argument(
        "-c", "--config", type=str, default="", help="YAML config file"
    )
    parser = argparse.ArgumentParser(description="Training", parents=[parser_config])

    group = parser.add_argument_group("System parameters")
    group.add_argument(
        "--pre_trained_model_path",
        type=str,
        default="",
        help="Pretrained checkpoint path",
    )
    group.add_argument(
        "--is_distributed",
        action="store_true",
        default=False,
        help="Distributed training",
    )
    group.add_argument(
        "--bidirectional",
        action="store_false",
        default=True,
        help="Use bidirectional RNN",
    )
    group.add_argument(
        "--device_target",
        type=str,
        default="Ascend",
        choices=("GPU", "CPU", "Ascend"),
        help="Device target, support GPU and CPU, Default: GPU",
    )
    group.add_argument(
        "--device_id",
        default=0,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )

    return parser_config, parser


def parse_args():
    parser_config, parser = create_parser()

    args_config, remaining = parser_config.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            cfg = ed(cfg)
            parser.set_defaults(**cfg)
            parser.set_defaults(config=args_config.config)
    args = parser.parse_args(remaining)
    return args
