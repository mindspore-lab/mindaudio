# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Log."""
import logging
import logging.config
import logging.handlers
import os
import sys
from logging import StreamHandler  # pylint: disable=C0412
from logging.handlers import RotatingFileHandler  # pylint: disable=C0412
from typing import List, Tuple, Union

from flyspeech.adapter.moxing_adapter import check_in_modelarts
from flyspeech.adapter.parallel_info import get_device_id

logger_list = []
stream_handler_list = {}
file_handler_list = {}

_level = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_modelarts_log_file_dir = "/cache/log"
_local_default_log_file_dir = "~/.cache/flyspeech"
_default_filehandler_format = (
    "[%(levelname)s] %(asctime)s [%(pathname)s:%(lineno)d] %(funcName)s: %(message)s"
)
_default_stdout_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def validate_std_input_format(to_std, stdout_devices, stdout_level):
    """Validate the input about stdout of the get_logger function."""

    if not isinstance(to_std, bool):
        raise TypeError("The format of the to_std must be of type bool.")

    if not isinstance(stdout_devices, (list, tuple)):
        raise TypeError(
            "The value of stdout_devices should be a value of type tuple, list"
        )
    for node in stdout_devices:
        if not isinstance(node, int):
            raise TypeError(
                "The type of the elements inside stdout_devices must be int."
            )

    if not isinstance(stdout_level, str):
        raise TypeError("The type of the stdout_level must be str.")
    if stdout_level not in _level:
        raise ValueError("stdout_level needs to be in {}".format(_level))


def validate_file_input_format(file_level):
    """Validate the input about file of the get_logger function."""

    if not isinstance(file_level, str):
        raise TypeError("The type of the file_level must be str.")
    if file_level not in _level:
        raise ValueError("file_level needs to be in {}".format(_level))


def create_dirs(path: str, mode=0o750):
    """Recursively create folders."""
    if not os.path.exists(path):
        os.makedirs(path, mode)


def judge_stdout(is_output: bool, stdout_devices: Union[List, Tuple]) -> bool:
    """Determines if logs will be output.

    Args:
        is_output (bool): If set to true, logs or others will be output.
        stdout_devices (list or tuple): Device list. The devices in the
            list or output the log to stdout.

    Returns:
        is_output (bool): If true, logs or others will be output.
    """
    device_id = get_device_id()
    if not stdout_devices:
        # A node has a machine number of (0, 1, 2, 3, 4, 5, 6, 7).
        # No matter how the machines are assigned for training,
        # the numbering will not leave these ranges.
        stdout_devices = (0, 1, 2, 3, 4, 5, 6, 7)

    if is_output and device_id not in stdout_devices:
        is_output = False

    return is_output


def _convert_level(level: str) -> int:
    """Convert the format of the log to logging level.

    Args:
        level (str): User log level.

    Returns:
        level (int): Logging level.
    """
    level_convert = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    logging_level = level_convert.get(level, logging.INFO)

    return logging_level


def get_stream_handler(stdout_format: str, stdout_level: str) -> StreamHandler:
    """Set stream handler of logger."""
    if not stdout_format:
        stdout_format = _default_stdout_format

    handler_name = "{}.{}".format(stdout_format, stdout_level)
    if handler_name in stream_handler_list:
        return stream_handler_list[handler_name]

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(_convert_level(stdout_level))
    stream_formatter = logging.Formatter(stdout_format)
    stream_handler.setFormatter(stream_formatter)

    stream_handler_list[handler_name] = stream_handler

    return stream_handler


def get_file_path(file_save_dir: str, file_name: str) -> str:
    """Gets the list of files where the logs are saved."""
    if not file_save_dir:
        file_save_dir = os.path.expanduser(_local_default_log_file_dir)
    if check_in_modelarts():
        file_save_dir = _modelarts_log_file_dir

    device_id = get_device_id()
    file_save_dir = os.path.join(file_save_dir, "device_{}".format(device_id))
    file_save_dir = os.path.realpath(file_save_dir)
    create_dirs(file_save_dir)
    file_path = os.path.join(file_save_dir, file_name)

    return file_path


def get_file_handler_list(
    file_level: str,
    file_save_dir: str,
    file_name: str,
    max_file_size: int,
    max_num_of_files: int,
) -> RotatingFileHandler:
    """get file handler of logger."""
    file_level = _convert_level(file_level)
    file_path = get_file_path(file_save_dir, file_name)

    max_file_size = max_file_size * 1024 * 1024
    file_formatter = logging.Formatter(_default_filehandler_format)
    handler_name = "{}.{}.{}.{}".format(
        file_path, max_file_size, max_num_of_files, file_level
    )
    if handler_name in file_handler_list:
        return file_handler_list[handler_name]

    file_handler = logging.handlers.RotatingFileHandler(
        filename=file_path, maxBytes=max_file_size, backupCount=max_num_of_files
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    return file_handler


def get_logger(
    logger_name: str = "Flyspeech",
    to_std: bool = True,
    stdout_devices: Union[List, Tuple] = (),
    stdout_level: str = "INFO",
    stdout_format: str = "",
    file_level: str = "INFO",
    file_save_dir: str = "",
    file_name: str = "flyspeech.log",
    max_file_size: int = 50,
    max_num_of_files: int = 5,
) -> logging.Logger:
    """Get the logger. Both computing centers and bare metal servers are
    available.

    Args:
        logger_name (str): Logger name.
        to_std (bool): If set to True, output the log to stdout.
        stdout_devices (list[int] or tuple[int]):
            The computation devices that will output the log to stdout.
            default: (0,), indicates that devices 0 will output logs to stdout.
            eg: [0, 1, 2, 3] or (0, 1, 2, 3): indicates that devices 0, 1, 2,
                and 3 all output logs to stdout.
        stdout_level (str): The level of the log output to stdout. Optional DEBUG,
            INFO, WARNING, ERROR, CRITICAL.
        stdout_format (str): Log format. Default: '', indicates that use default
            log format '%(asctime)s - %(name)s - %(levelname)s - %(message)s'.
        file_level (str): The level of the log output to file.
            Default: 'INFO' indicates that the logger will output info log.
        file_save_dir (str): The folder where the log files are stored.
            Default: '', indicate that save log file to ~/.cache/flyspeech/
        file_name (str): The name of log file.
            Default: 'flyspeech', indicate that save log to flyspeech.log.
        max_file_size (int): The maximum size of a single log file. Unit: MB.
            Default: 50.
        max_num_of_files (int): The maximum number of files to save. Default: 5

    Returns:
        logger (logging.Logger): Logger.
    """
    logger = logging.getLogger(logger_name)
    if logger_name in logger_list:
        return logger

    validate_std_input_format(to_std, stdout_devices, stdout_level)
    validate_file_input_format(file_level)

    to_std = judge_stdout(to_std, stdout_devices)
    if to_std:
        stream_handler = get_stream_handler(stdout_format, stdout_level)
        logger.addHandler(stream_handler)

    file_handler = get_file_handler_list(
        file_level, file_save_dir, file_name, max_file_size, max_num_of_files
    )
    logger.addHandler(file_handler)

    logger.propagate = False
    logger.setLevel(_convert_level("DEBUG"))

    logger_list.append(logger_name)

    return logger


def print_log(*msg, output_obj="Flyspeech", level="INFO"):
    """Output logs by different ways.

    Args:
        *msg (tuple): Log content.
        output_obj (str or logging.Logger or optional): The object used to
            output the log.
            Default: 'Flyspeech', use the logger named Flyspeech.
            if output_obj is set to None, it means use python's print
                function to output *msg.
            if output_obj is set to 'silent', it means that nothing will
                be output.
            if output_obj is set to a text other than 'silent', it means
                that the logger with that name is used.
            if output_obj is set to an instance of logging.Logger, the logger
                is used directly.
            Throws TypeError if output_obj is set to others.
        level (str): Log level. DEBUG, INFO, WARNING, ERROR and CRITICAL
            are available.

    Returns:
        None
    """
    level = _convert_level(level)

    if output_obj is None:
        print(*msg)
    elif output_obj == "silent":
        pass
    elif isinstance(output_obj, str):
        logger = get_logger(output_obj)
        logger.log(level, *msg)
    elif isinstance(output_obj, logging.Logger):
        output_obj.log(level, *msg)
    else:
        raise TypeError(
            "The output_obj parameter can only be None, a logging.Logger object, "
            "or a variable of type str."
        )
