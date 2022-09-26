import sys
import os
import os.path
import time
import numpy as np
import mindspore

from mindspore.context import ParallelMode
from mindspore.communication import management as MultiDevice
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore import context
from mindspore import Model
from mindspore import Tensor
from mindspore.nn.optim import Adam

sys.path.append('.')
from examples.tacotron2.dataset import create_dataset
from examples.tacotron2.hparams import hparams as hps
from mindaudio.nn.lr_generator import get_tacotron2_lr
from mindaudio.utils.callback import LossCallBack
from mindaudio.models.tacotron2 import Tacotron2, Tacotron2Loss, NetWithLossClass, TrainStepWrap

from mindaudio.models.tacotron2 import config
from mindaudio.adapter.local_adapter import get_device_id, get_device_num

def get_ms_timestamp():
    '''get timestamp'''
    t = time.time()
    return int(round(t * 1000))

np.random.seed(0)
mindspore.common.set_seed(1024)
time_stamp_init = False
time_stamp_first = 0

context.set_context(mode=context.PYNATIVE_MODE, save_graphs=False, device_target=config.device_target, device_id=get_device_id())

def prepare_dataloaders(manifest_path, rank_id, group_size):
    '''prepare dataloaders'''
    vocab_size, dataset = create_dataset(
        manifest_path,
        hps,
        batch_size=hps.batch_size,
        rank=rank_id,
        group_size=group_size
    )
    return dataset

def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))

    config.save_ckpt_dir = os.path.join(config.output_path, config.save_ckpt_dir)

def _build_training_pipeline(pre_dataset, run_distribute=False):
    ''' training '''

    epoch_num = config.epoch_num

    steps_per_epoch = pre_dataset.get_dataset_size()

    learning_rate = get_tacotron2_lr(config.lr, epoch_num, steps_per_epoch, steps_per_epoch * config.warmup_epochs)
    learning_rate = Tensor(learning_rate)

    scale_update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2**12,
                                                   scale_factor=2,
                                                   scale_window=1000)
    net = Tacotron2()
    loss_fn = Tacotron2Loss()
    loss_net = NetWithLossClass(net, loss_fn)


    resume_epoch = None
    if config.pretrain_ckpt:
        resume_epoch = int(config.pretrain_ckpt.split('-')[-1].split('_')[0])
        learning_rate = learning_rate[resume_epoch * steps_per_epoch:]
        param_dict = load_checkpoint(config.pretrain_ckpt)
        load_param_into_net(net, param_dict)
        print(
            'Successfully loading the pretrained model {}'.format(
                config.pretrain_ckpt))

    optimizer = Adam(params=net.trainable_params(), learning_rate=learning_rate)

    train_net = TrainStepWrap(loss_net, optimizer, scale_update_cell)
    train_net.set_train()

    model = Model(train_net)

    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch,
                                   keep_checkpoint_max=config.keep_ckpt_max)


    callbacks = [LossCallBack(steps_per_epoch),
                 TimeMonitor(data_size=steps_per_epoch)]

    ckpt_callback = ModelCheckpoint(prefix='tacotron2',
                                    directory=os.path.join(config.save_ckpt_dir,
                                                           'ckpt_{}'.format(os.getenv("DEVICE_ID"))),
                                    config=ckpt_config)

    callbacks.append(ckpt_callback)


    print("Prepare to Training....")
    if resume_epoch is not None:
        epoch_num = epoch_num - resume_epoch

    print("Epoch size ", epoch_num)
    if run_distribute:
        print(f" | Rank {MultiDevice.get_rank()} Call model train.")

    model.train(
        epoch_num,
        pre_dataset,
        callbacks=callbacks,
        dataset_sink_mode=True)


def set_parallel_env():
    '''set parallel context'''
    context.reset_auto_parallel_context()
    MultiDevice.init()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                      device_num=MultiDevice.get_group_size(),
                                      gradients_mean=True)

def train_parallel(input_file_path):
    """
    Train model on multi device
    Args:
        input_file_path: preprocessed dataset path
    """
    set_parallel_env()
    print("Starting traning on multiple devices. |~ _ ~| |~ _ ~| |~ _ ~| |~ _ ~|")
    hps.batch_size = config.batch_size

    preprocessed_data = prepare_dataloaders(input_file_path,
                                            MultiDevice.get_rank(),
                                            MultiDevice.get_group_size())

    _build_training_pipeline(preprocessed_data, True)

def train_single(input_file_path):
    """
    Train model on single device
    Args:
        input_file_path: preprocessed dataset path
    """
    print("Staring training on single device.")
    hps.batch_size = config.batch_size

    preprocessed_data = prepare_dataloaders(input_file_path,
                                            rank_id=config.rank_id,
                                            group_size=1)

    _build_training_pipeline(preprocessed_data)

def run_train():
    '''run train.'''
    if config.device_target in {"Ascend"}:
        config.rank_id = get_device_id()
    else:
        raise ValueError("Not support device target: {}".format(config.device_target))

    device_num = get_device_num()
    print('DEVICE:', device_num)
    if device_num > 1:
        train_parallel(config.manifest_path)
    else:
        train_single(config.manifest_path)


if __name__ == '__main__':
    run_train()
