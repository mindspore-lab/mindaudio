# Copyright 2022 Huawei Technologies Co., Ltd
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
"""cloud_callback function."""

import os
import re
from multiprocessing import Process

import moxing as mox
from flyspeech.adapter.log import get_logger
from mindspore.communication.management import get_group_size, get_rank
from mindspore.train.callback import Callback

logger = get_logger()
os.environ.pop("CREDENTIAL_PROFILES_FILE", None)
os.environ.pop("AWS_SHARED_CREDENTIALS_FILE", None)


def _sort_obs_ckpt(obs_path):
    """Sorts checkpoint files by name."""
    file_list = mox.file.list_directory(obs_path)
    ckpt_list = [x for x in file_list if x.endswith(".ckpt")]
    if not ckpt_list:
        return None

    # sort the ckpt_file_list according to the ckpt name.
    fake_ckpt_list = []
    for ckpt in ckpt_list:
        if ckpt.count("_") == 2:
            fake_ckpt_list.append(ckpt)
        else:
            prefix, suffix = ckpt.split("-")
            new_ckpt = prefix + "_0" + "-" + suffix
            fake_ckpt_list.append(new_ckpt)

    fake_ckpt_list.sort(
        key=lambda x: (
            -int(re.split(r"[_|\-|.]", x)[1]),
            -int(re.split(r"[_|\-|.]", x)[2]),
            -int(re.split(r"[_|\-|.]", x)[3]),
        )
    )
    sorted_ckpt_list = [x.replace("_0", "") for x in fake_ckpt_list]
    return sorted_ckpt_list


class OBSUpdate(Callback):
    """Update the checkpoint from local to OBS in training.

    Args:

    Raises:
        ValueError: If data_size is not positive int.
    """

    def __init__(self, local_file_dir, obs_file_dir, steps_size, config):
        super(OBSUpdate, self).__init__()
        self.steps_size = steps_size
        self.local_file_dir = local_file_dir
        self.obs_file_dir = obs_file_dir
        self.config = config

    def step_begin(self, run_context):
        pass

    def step_end(self, run_context):
        """step end function."""
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num
        if cur_step % self.steps_size == 0:
            if os.path.exists(self.local_file_dir):
                files = os.listdir(self.local_file_dir)
                if files:
                    files.sort(
                        key=lambda fn: os.path.getatime(self.local_file_dir + "/" + fn)
                    )
                    name_ext = os.path.splitext(files[-1])
                    if name_ext[-1] != ".ckpt":
                        raise ValueError(
                            "Invalid file, checkpoint file should be .ckpt file"
                        )
                    newest_ckpt_file = os.path.join(self.local_file_dir, files[-1])
            else:
                logger.warning("this path not exist")
            obs_ckpt_dir = os.path.join(self.obs_file_dir, "model")
            if not mox.file.exists(obs_ckpt_dir):
                mox.file.mk_dir(obs_ckpt_dir)
            file_list = mox.file.list_directory(obs_ckpt_dir)
            ckpt_file = [x for x in file_list if x.endswith(".ckpt")]
            if len(ckpt_file) >= self.config.keep_checkpoint_max:
                oldest_ckpt = _sort_obs_ckpt(obs_ckpt_dir)[-1]
                mox.file.remove(obs_ckpt_dir, oldest_ckpt)

            obs_ckpt_file = os.path.join(obs_ckpt_dir, newest_ckpt_file.split("/")[-1])
            mox.file.copy(newest_ckpt_file, obs_ckpt_file)


class CloudSummaryCallback(Callback):
    """Update the checkpoint from local to OBS in training.

    Args:

    Raises:
        ValueError: If data_size is not positive int.
    """

    def __init__(self, local_file_dir, obs_file_dir, steps_size, config):
        super(CloudSummaryCallback, self).__init__()
        self.steps_size = steps_size
        self.local_file_dir = local_file_dir
        self.obs_file_dir = obs_file_dir
        self.config = config

    def step_end(self, run_context):
        """step end function."""
        cb_params = run_context.original_args()
        cur_step_num = cb_params.cur_step_num
        if cur_step_num % self.steps_size == 0:
            obs_summary_dir = os.path.join(self.obs_file_dir, "summary")
            if not mox.file.exists(obs_summary_dir):
                mox.file.mk_dir(obs_summary_dir)
            mox.file.copy_parallel(self.local_file_dir, obs_summary_dir)


class BaseSyncCallback(Callback):
    """Base class for classes that synchronise files or folders to OBS."""

    def __init__(self, local_path: str, obs_path: str) -> None:
        """Init method of BaseSyncCallback.

        Args:
            local_path (str): Local file path.
            obs_path (str): OBS path.
            steps_size (int): If the current number of steps is divisible by steps_zie, start data synchronization.
        """
        super(BaseSyncCallback, self).__init__()
        self.obs_path = obs_path
        self.local_path = local_path

        try:
            # Each server contains 8 devices as most.
            self.is_sync = get_rank() % min(get_group_size(), 8) == 0
        except (ValueError, RuntimeError):
            self.is_sync = True


class BaseSyncDirCallback(BaseSyncCallback):
    """Parent class of SyncDirCallback."""

    def __init__(
        self, local_path: str, obs_path: str, delete_old: bool = False
    ) -> None:
        super(BaseSyncDirCallback, self).__init__(local_path, obs_path)
        self.delete_old = delete_old

        if not mox.file.exists(self.obs_path):
            mox.file.mk_dir(self.obs_path)

    def sync_dir(self):
        if self.delete_old:
            mox.file.remove(self.obs_path, recursive=True)

        mox.file.copy_parallel(self.local_path, self.obs_path)

        log_message = "[SyncDirCallback] Finish synchronising {} to {}.".format(
            self.local_path, self.obs_path
        )
        logger.info(log_message)


class BaseSyncFileCallback(BaseSyncCallback):
    def sync_file(self):
        mox.file.copy(self.local_path, self.obs_path)

        log_message = "[SyncFileCallback] Finish synchronising {} to {}.".format(
            self.local_path, self.obs_path
        )
        logger.info(log_message)


class SyncDirCallback(BaseSyncDirCallback):
    """Class for synchronising folders to OBS."""

    def __init__(
        self, local_path: str, obs_path: str, steps_size: int, delete_old: bool = False
    ) -> None:
        """Init method of SyncDirCallback.

        Args:
            local_path (str): Local folder path.
            obs_path (str): OBS path.
            steps_size (int): If the current number of steps is divisible by steps_zie, start data synchronization.
            delete_old (bool): Whether delete old file. Default: False.
        """
        super(SyncDirCallback, self).__init__(local_path, obs_path, delete_old)
        self.steps_size = steps_size

    def step_end(self, run_context):
        """At the end of each step, determine if the data is synchronised.

        If so, synchronise the data asynchronously.
        """
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num

        if cur_step % self.steps_size != 0:
            return
        if not self.is_sync:
            return

        if os.path.exists(self.local_path) and self.is_sync:
            sync_process = Process(target=self.sync_dir)
            sync_process.start()
        else:
            logger.info(
                "[SyncDirCallback] Local directory %s does not exist!", self.local_path
            )


class SyncDirEndCallback(BaseSyncDirCallback):
    def end(self, run_context):  # pylint: disable=W0613
        if not self.is_sync:
            return

        if os.path.exists(self.local_path) and self.is_sync:
            self.sync_dir()
        else:
            logger.info(
                "[SyncDirEndCallback] Local directory %s does not exist!",
                self.local_path,
            )


class SyncFileCallback(BaseSyncFileCallback):
    """Class for synchronising file to OBS."""

    def __init__(self, local_path: str, obs_path: str, steps_size: int) -> None:
        super(SyncFileCallback, self).__init__(local_path, obs_path)
        self.steps_size = steps_size

    def step_end(self, run_context):
        """At the end of each step, determine if the data is synchronised.

        If so, synchronise the data asynchronously.
        """
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num

        if cur_step % self.steps_size != 0:
            return
        if not self.is_sync:
            return

        if os.path.exists(self.local_path) and self.is_sync:
            sync_process = Process(target=self.sync_file)
            sync_process.start()
        else:
            logger.info(
                "[SyncFileCallback] Local file %s does not exist!", self.local_path
            )


class SyncFileEndCallback(BaseSyncFileCallback):
    """Synchronise the files at the end of the training."""

    def end(self, run_context):
        super(SyncFileEndCallback, self).end(run_context)
        if not self.is_sync:
            return

        if os.path.exists(self.local_path) and self.is_sync:
            self.sync_file()
        else:
            logger.info(
                "[SyncFileEndCallback] Local file %s does not exist!", self.local_path
            )
