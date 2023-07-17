""" Train """
import argparse
import json
import os

import mindspore
import mindspore.dataset as ds
from data import DatasetGenerator
from mindspore import context, load_checkpoint, load_param_into_net, nn, set_seed
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.context import ParallelMode
from mindspore.train import Model
from mindspore.train.callback import (
    CheckpointConfig,
    LossMonitor,
    ModelCheckpoint,
    TimeMonitor,
)
from train_wrapper import TrainingWrapper

import mindaudio.data.io as io
from mindaudio.loss.separation_loss import NetWithLoss, Separation_Loss
from mindaudio.models.tasnet import TasNet
from mindaudio.utils.hparams import parse_args

set_seed(1)


def preprocess_one_dir(in_dir, out_dir, out_filename, sample_rate=8000):
    """
    sample_rate: 8000
    Read the wav file and save the path and len to the json file
    """
    file_infos = []
    in_dir = os.path.abspath(in_dir)
    wav_list = os.listdir(in_dir)
    for wav_file in wav_list:
        if not wav_file.endswith(".wav"):
            continue
        wav_path = os.path.join(in_dir, wav_file)
        samples, _ = io.read(wav_path)
        file_infos.append((wav_path, len(samples)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + ".json"), "w") as f:
        json.dump(file_infos, f, indent=4)


def preprocess(arg):
    """ Process all files """
    print("Begin preprocess")
    for data_type in ["train-100"]:
        for speaker in ["mix_clean", "s1", "s2"]:
            preprocess_one_dir(
                os.path.join(arg.in_dir, data_type, speaker),
                os.path.join(arg.out_dir, data_type),
                speaker,
                sample_rate=arg.sample_rate,
            )
    print("Preprocess done")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    if args.device_target == "Ascend":
        context.set_context(device_id=args.device_id)

    device_num = int(os.environ.get("RANK_SIZE", 1))
    if device_num == 1:
        is_distributed = "False"
    elif device_num > 1:
        is_distributed = "True"

    if is_distributed == "True":
        print("parallel init", flush=True)
        init()
        rank_id = get_rank()
        context.reset_auto_parallel_context()
        parallel_mode = ParallelMode.DATA_PARALLEL
        rank_size = get_group_size()
        context.set_auto_parallel_context(
            parallel_mode=parallel_mode, gradients_mean=True, device_num=rank_size
        )
        context.set_auto_parallel_context(parameter_broadcast=True)
        args.save_folder = os.path.join(
            args.save_folder, "ckpt_" + str(get_rank()) + "/"
        )
        print("Starting traning on multiple devices.")

    print("Preparing Data")
    tr_dataset = DatasetGenerator(
        args.train_dir, args.batch_size, sample_rate=args.sample_rate, L=args.L
    )
    if is_distributed == "True":
        tr_loader = ds.GeneratorDataset(
            tr_dataset,
            ["mixture", "lens", "sources"],
            shuffle=True,
            num_shards=rank_size,
            shard_id=rank_id,
        )
    else:
        tr_loader = ds.GeneratorDataset(
            tr_dataset, ["mixture", "lens", "sources"], shuffle=True
        )
    tr_loader = tr_loader.batch(batch_size=args.batch_size)
    print("Prepare Data done")

    # model
    net = TasNet(
        args.L,
        args.N,
        args.hidden_size,
        args.num_layers,
        bidirectional=bool(args.bidirectional),
        nspk=args.nspk,
    ).to_float(mindspore.float16)
    if args.continue_train == 1:
        home = os.path.dirname(os.path.realpath(__file__))
        ckpt = os.path.join(home, args.model_path)
        print("=====> load params into generator")
        params = load_checkpoint(ckpt)
        load_param_into_net(net, params)
        print("=====> finish load generator")

    print(net)
    num_steps = tr_loader.get_dataset_size()

    milestone = [10 * num_steps, 40 * num_steps, 50 * num_steps]
    learning_rates = [1e-3, 3e-4, 1e-4]
    lr = nn.piecewise_constant_lr(milestone, learning_rates)
    optimizer = nn.Adam(
        net.get_parameters(), learning_rate=lr, weight_decay=args.l2, loss_scale=0.01
    )
    my_loss = Separation_Loss()
    loss_cb = LossMonitor()
    time_cb = TimeMonitor()
    net_with_loss = NetWithLoss(net, my_loss)
    net_with_clip_norm = TrainingWrapper(net_with_loss, optimizer)
    net_with_clip_norm.set_train()

    config_ck = CheckpointConfig(save_checkpoint_steps=num_steps, keep_checkpoint_max=1)

    ckpt_cb = ModelCheckpoint(
        prefix="TasNet_train", directory=args.save_folder, config=config_ck
    )
    cb = [time_cb, loss_cb, ckpt_cb]
    model = Model(net_with_clip_norm)

    print("Training......", flush=True)
    model.train(
        epoch=args.epochs, train_dataset=tr_loader, callbacks=cb, dataset_sink_mode=True
    )
