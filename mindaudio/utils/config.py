"""Config dict for configure parse module."""

import argparse
import ast
import os

import yaml

BASE_CONFIG = "base_config"


class Config(dict):
    """A Config class is inherit from dict.

    Config class can parse arguments from a config file of yaml or a dict.

    Args:
        args (list) : config file_names
        kwargs (dict) : config dictionary list

    Example:
        test.yaml:
            a:1
        >>> cfg = Config('./test.yaml')
        >>> cfg.a
        1
        >>> cfg = Config(**dict(a=1, b=dict(c=[0,1])))
        >>> cfg.b
        {'c': [0, 1]}
    """

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__()
        cfg_dict = {}

        # load from file
        for arg in args:
            if isinstance(arg, str):
                if arg.endswith("yaml") or arg.endswith("yml"):
                    raw_dict = Config._file2dict(arg)
                    cfg_dict.update(raw_dict)

        # load dictionary configs
        if kwargs:
            cfg_dict.update(kwargs)
        Config._dict2config(self, cfg_dict)

    def __getattr__(self, key):
        """Get a object attr by `key`.

        Args:
            key(str): the name of object attr.

        Returns:
            Attr of object that name is `key`.
        """
        if key not in self:
            return None
        return self[key]

    def __setattr__(self, key, value):
        """Set a object value `key` with `value`.

        Args:
            key(str): The name of object attr.
            value: the `value` need to set to the target object attr.
        """
        self[key] = value

    def __delattr__(self, key):
        """Delete a object attr by its `key`.

        Args:
            key(str): The name of object attr.
        """
        del self[key]

    def merge_from_dict(self, options):
        """Merge options into config file.

        Args:
            options(dict): dict of configs to merge from.

        Examples:
            >>> options = {'model.backbone.depth': 101, 'model.rpn_head.in_channels': 512}
            >>> cfg = Config(**dict(model=dict(backbone=dict(type='ResNet'))))
            >>> cfg.merge_from_dict(options)
        """
        option_cfg_dict = {}
        for full_key, value in options.items():
            d = option_cfg_dict
            key_list = full_key.split(".")
            for sub_key in key_list[:-1]:
                d.setdefault(sub_key, Config())
                d = d[sub_key]
            sub_key = key_list[-1]
            d[sub_key] = value
        merge_dict = Config._merge_into(option_cfg_dict, self)
        Config._dict2config(self, merge_dict)

    @staticmethod
    def _merge_into(a, b):
        """Merge dict ``a`` into dict ``b``, values in ``a`` will overwrite
        ``b``.

        Args:
            a(dict): The source dict to be merged into b.
            b(dict): The origin dict to be fetch keys from ``a``.

        Returns:
            dict: The modified dict of ``b`` using ``a``.
        """
        b = b.copy()
        for k, v in a.items():
            if isinstance(v, dict) and k in b:
                b[k] = Config._merge_into(v, b[k])
            else:
                b[k] = v
        return b

    @staticmethod
    def _file2dict(file_name=None):
        """Convert config file to dictionary.

        Args:
            file_name(str): Config file.
        """
        if not file_name:
            raise NameError(f"The {file_name} cannot be empty.")

        with open(os.path.realpath(file_name)) as f:
            cfg_dict = yaml.load(f, Loader=yaml.FullLoader)

        # Load base config file.
        if BASE_CONFIG in cfg_dict:
            cfg_dir = os.path.dirname(file_name)
            base_file_names = cfg_dict.pop(BASE_CONFIG)
            base_file_names = (
                base_file_names
                if isinstance(base_file_names, list)
                else [base_file_names]
            )

            cfg_dict_list = list()
            for base_file_name in base_file_names:
                cfg_dict_item = Config._file2dict(os.path.join(cfg_dir, base_file_name))
                cfg_dict_list.append(cfg_dict_item)
            base_cfg_dict = dict()
            for cfg in cfg_dict_list:
                base_cfg_dict.update(cfg)

            # Merge config
            base_cfg_dict = Config._merge_into(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict
        return cfg_dict

    @staticmethod
    def _dict2config(config, dic):
        """Convert dictionary to config.

        Args:
            config: Config object.
            dic(dict): dictionary.
        """
        if isinstance(dic, dict):
            for key, value in dic.items():
                if isinstance(value, dict):
                    sub_config = Config()
                    dict.__setitem__(config, key, sub_config)
                    Config._dict2config(sub_config, value)
                else:
                    config[key] = dic[key]

    @staticmethod
    def _save_yaml(config, path):
        with open(path, "w") as f:
            yaml.dump(config, f)


def parse_cli_to_yaml(
    parser, cfg, helper=None, choices=None, cfg_path="asr_config.yaml"
):
    """Parse command line arguments to the configuration according to the
    default yaml.

    Args:
        parser: Parent parser.
        cfg: Base configuration.
        helper: Helper description.
        cfg_path: Path to the default yaml config.
    """
    parser = argparse.ArgumentParser(
        description="[REPLACE THIS at config.py]", parents=[parser]
    )
    helper = {} if helper is None else helper
    choices = {} if choices is None else choices
    for item in cfg:
        if not isinstance(cfg[item], list) and not isinstance(cfg[item], dict):
            help_description = (
                helper[item]
                if item in helper
                else "Please reference to {}".format(cfg_path)
            )
            choice = choices[item] if item in choices else None
            if isinstance(cfg[item], bool):
                parser.add_argument(
                    "--" + item,
                    type=ast.literal_eval,
                    default=cfg[item],
                    choices=choice,
                    help=help_description,
                )
            else:
                parser.add_argument(
                    "--" + item,
                    type=type(cfg[item]),
                    default=cfg[item],
                    choices=choice,
                    help=help_description,
                )
    args = parser.parse_args()
    return args


def merge(args, cfg):
    """Merge the base config from yaml file and command line arguments.

    Args:
        args: Command line arguments.
        cfg: Base configuration.
    """
    args_var = vars(args)
    for item in args_var:
        cfg[item] = args_var[item]
    return cfg


def get_config(config=""):
    """Get Config according to the yaml file and cli arguments."""
    if config == "":
        raise ValueError(
            "config name should be choose in ['asr_config', 'asr_conformer', "
            "'asr_conformer_bidecoder.yaml', asr_transformer]"
        )
    yaml_file = "config/" + config + ".yaml"
    parser = argparse.ArgumentParser(description="default name", add_help=False)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(current_dir, yaml_file),
        help="Config file path",
    )
    path_args, _ = parser.parse_known_args()
    cfg_path = path_args.config_path
    cfg = Config(cfg_path)
    args = parse_cli_to_yaml(parser=parser, cfg=cfg, cfg_path=path_args.config_path)
    # args.train_url = os.path.join(
    #     args.train_url, str(int(datetime.datetime.now().timestamp()))
    # )

    merge(args, cfg)
    return cfg
