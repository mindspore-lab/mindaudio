"""Moxing adapter for ModelArts."""

import functools
import os
import time

from mindspore import context
from mindspore.profiler import Profiler

_global_sync_count = 0


def get_job_id():
    job_id = os.getenv("JOB_ID")
    job_id = job_id if job_id != "" else "default"
    return job_id


def check_in_modelarts():
    """Check if the training is on modelarts.

    Returns:
        (bool): If it is True, it means ModelArts environment.
    """
    return (
        "KUBERNETES_PORT" in os.environ
        or "MA_LOG_DIR" in os.environ
        or "MA_JOB_DIR" in os.environ
        or "MA_LOCAL_LOG_PATH" in os.environ
    )


def sync_data(from_path, to_path):
    """Download data from remote obs to local directory if the first url is
    remote url and the second one is local path Upload data from local
    directory to remote obs in contrast."""
    import moxing as mox
    from adapter.log import get_logger
    from adapter.parallel_info import get_device_id, get_device_num

    logger = get_logger()

    global _global_sync_count
    os.environ.pop("CREDENTIAL_PROFILES_FILE", None)
    os.environ.pop("AWS_SHARED_CREDENTIALS_FILE", None)

    sync_lock = "/tmp/copy_sync.lock" + str(_global_sync_count)
    _global_sync_count += 1

    # Each server contains 8 devices as most.
    if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(
        sync_lock
    ):
        logger.info("Start sync data from %s to %s.", from_path, to_path)
        mox.file.copy_parallel(from_path, to_path)
        try:
            os.mknod(sync_lock)
        except IOError:
            pass

    while True:
        if os.path.exists(sync_lock):
            break
        time.sleep(1)

    logger.info("Finish sync data from %s to %s.", from_path, to_path)


def modelarts_pre_process(config):
    """modelarts pre process function."""
    from flyspeech.adapter.log import get_logger
    from flyspeech.adapter.parallel_info import get_device_id, get_device_num

    logger = get_logger()

    def unzip(zip_file, save_dir):
        import zipfile

        s_time = time.time()
        if not os.path.exists(
            os.path.join(save_dir, config.modelarts_dataset_unzip_name)
        ):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, "r")
                data_num = len(fz.namelist())
                logger.info("Extract Start...")
                logger.info("unzip file num: %d", data_num)
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        log_message = "unzip percent: {}%".format(
                            int(i * 100 / data_num)
                        )
                        logger.info(log_message)
                    i += 1
                    fz.extract(file, save_dir)
                log_message = "cost time: {}min:{}s.".format(
                    int((time.time() - s_time) / 60),
                    int(int(time.time() - s_time) % 60),
                )
                logger.info(log_message)
                logger.info("Extract Done.")
            else:
                logger.info("This is not zip.")
        else:
            logger.info("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(
            config.data_path, config.modelarts_dataset_unzip_name + ".zip"
        )
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(
            sync_lock
        ):
            logger.info("Zip file path: %s", zip_file_1)
            logger.info("Unzip file save dir: %s", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            logger.info("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        logger.info(
            "Device: %d, Finish sync unzip data from %s to %s.",
            get_device_id(),
            zip_file_1,
            save_dir_1,
        )

    config.ckpt_path = os.path.join(config.output_path, config.ckpt_path)


def moxing_wrapper(config, pre_process=None, post_process=None):
    """Moxing wrapper to download dataset and upload outputs."""
    from adapter.log import get_logger
    from adapter.parallel_info import get_device_id, get_device_num, get_rank_id

    logger = get_logger()

    def wrapper(run_func):
        @functools.wraps(run_func)
        def wrapped_func(*args, **kwargs):
            # Download data from data_url
            if config.enable_modelarts:
                # config ak sk
                import moxing as mox

                mox.file.set_auth(
                    ak=config.ak, sk=config.sk, server=config.server,
                )
                if config.compile_url and "MS_COMPILER_CACHE_PATH" in os.environ:
                    local_cache_dir = os.path.dirname(
                        os.environ["MS_COMPILER_CACHE_PATH"]
                    )
                    sync_data(config.compile_url, local_cache_dir)
                    basename = os.path.basename(config.compile_url.rstrip("/"))
                    local_path = os.path.realpath(
                        os.path.join(local_cache_dir, basename)
                    )
                    compile_cache_path = os.path.realpath(
                        os.environ["MS_COMPILER_CACHE_PATH"]
                    )
                    if local_path != compile_cache_path:
                        os.rename(local_path, compile_cache_path)
                if config.data_url and not config.mnt_enable:
                    sync_data(config.data_url, config.data_path)
                    log_message = "Dataset downloaded: {}".format(
                        os.listdir(config.data_path)
                    )
                    logger.info(log_message)
                if config.checkpoint_url:
                    sync_data(config.checkpoint_url, config.load_path)
                    log_message = "Preload downloaded: {}".format(
                        os.listdir(config.load_path)
                    )
                    logger.info(log_message)
                if config.train_url:
                    mkdir_if_not_exist(config.train_url)
                    sync_data(config.train_url, config.output_path)
                    log_message = "Workspace downloaded: {}".format(
                        os.listdir(config.output_path)
                    )
                    logger.info(log_message)

                context.set_context(
                    save_graphs_path=os.path.join(
                        config.output_path, str(get_rank_id())
                    )
                )
                config.device_num = get_device_num()
                config.device_id = get_device_id()
                if not os.path.exists(config.output_path):
                    os.makedirs(config.output_path)

                if pre_process:
                    pre_process()

            if config.enable_profiling:
                output_path = config.exp_name + "/" + "summary" + "/profiler"
                profiler = Profiler(output_path=output_path)

            run_func(*args, **kwargs)

            if config.enable_profiling:
                profiler.analyse()
            # # Upload profiler data to train_url
            if config.enable_modelarts:
                if post_process:
                    post_process()
                if config.enable_profiling:
                    sync_data(output_path, config.train_url)

        return wrapped_func

    return wrapper


def mkdir_if_not_exist(path):
    import moxing as mox

    if not mox.file.exists(path):
        mox.file.make_dirs(path)
    else:
        path = path + "_"
        mkdir_if_not_exist(path)
