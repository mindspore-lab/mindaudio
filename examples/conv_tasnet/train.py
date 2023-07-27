import os

import mindspore.dataset as ds
from data import DatasetGenerator
from mindspore import Model, context, load_checkpoint, load_param_into_net, nn
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.context import ParallelMode
from mindspore.train.callback import (
    CheckpointConfig,
    LossMonitor,
    ModelCheckpoint,
    TimeMonitor,
)
from preprocess import preprocess

from mindaudio.loss.separation_loss import NetWithLoss, Separation_Loss
from mindaudio.models.conv_tasnet import ConvTasNet
from mindaudio.utils.hparams import parse_args


def main(args):
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
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
            parallel_mode=parallel_mode, gradients_mean=True, device_num=args.device_num
        )
        context.set_auto_parallel_context(parameter_broadcast=True)
        print("Starting traning on multiple devices...")
    else:
        context.set_context(device_id=args.device_id)

    print("Start datasetgenerator")
    tr_dataset = DatasetGenerator(
        args.train_dir,
        args.batch_size,
        sample_rate=args.sample_rate,
        segment=args.segment,
    )

    print("start Generatordataset")
    if is_distributed == "True":
        tr_loader = ds.GeneratorDataset(
            tr_dataset,
            ["mixture", "lens", "sources"],
            shuffle=False,
            num_shards=rank_size,
            shard_id=rank_id,
        )
    else:
        tr_loader = ds.GeneratorDataset(
            tr_dataset, ["mixture", "lens", "sources"], shuffle=False
        )
    tr_loader = tr_loader.batch(1)
    num_steps = tr_loader.get_dataset_size()

    print("data loading done")

    net = ConvTasNet(
        args.N,
        args.L,
        args.B,
        args.H,
        args.P,
        args.X,
        args.R,
        args.C,
        norm_type=args.norm_type,
        causal=args.causal,
        mask_nonlinear=args.mask_nonlinear,
    )

    if args.continue_train:
        params = load_checkpoint(args.ckpt_path)
        load_param_into_net(net, params)
    print(net)
    net.set_train()
    milestone = [35 * num_steps, 55 * num_steps, 75 * num_steps, 100 * num_steps]
    learning_rates = [1e-3, 1e-4, 5e-5, 1e-5]
    lr = nn.piecewise_constant_lr(milestone, learning_rates)
    optimizer = nn.SGD(
        net.trainable_params(),
        learning_rate=lr,
        weight_decay=args.l2,
        momentum=args.momentum,
    )

    my_loss = Separation_Loss()
    net_with_loss = NetWithLoss(net, my_loss)
    model = Model(net_with_loss, optimizer=optimizer)

    time_cb = TimeMonitor()
    loss_cb = LossMonitor(1)
    cb = [time_cb, loss_cb]
    config_ck = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=1)

    ckpt_cb = ModelCheckpoint(
        prefix="Conv-TasNet", directory=args.save_folder, config=config_ck
    )
    cb += [ckpt_cb]
    model.train(
        epoch=args.epochs, train_dataset=tr_loader, callbacks=cb, dataset_sink_mode=True
    )


if __name__ == "__main__":
    arg = parse_args()
    main(arg)
