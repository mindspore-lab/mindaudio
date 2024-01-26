"""ASR training process.

python train.py --config_path <CONFIG_FILE>
"""

import os

from asr_model import creadte_asr_model, create_asr_eval_net
from dataset import create_dataset
from mindspore import ParameterTuple, context, set_seed
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.nn.optim import Adam
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, SummaryCollector
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindaudio.scheduler.scheduler_factory import ASRWarmupLR
from mindaudio.utils.callback import (
    CalRunTimeCallback,
    EvalCallback,
    MemoryStartTimeCallback,
    ResumeCallback,
    TimeMonitor,
)
from mindaudio.utils.common import get_parameter_numel
from mindaudio.utils.config import get_config
from mindaudio.utils.log import get_logger
from mindaudio.utils.parallel_info import get_device_id, get_device_num, get_rank_id
from mindaudio.utils.train_one_step import TrainOneStepWithLossScaleCell

logger = get_logger()
config = get_config("conformer")


columns_list = [
    "xs_pad",
    "ys_pad",
    "ys_in_pad",
    "ys_out_pad",
    "r_ys_in_pad",
    "r_ys_out_pad",
    "xs_masks",
    "ys_masks",
    "ys_sub_masks",
    "ys_lengths",
    "xs_chunk_masks",
]


def train():
    """main function for asr_train."""
    # Set random seed
    set_seed(777)
    exp_dir = config.exp_name
    model_dir = os.path.join(exp_dir, "model")
    graph_dir = os.path.join(exp_dir, "graph")
    summary_dir = os.path.join(exp_dir, "summary")
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target="Ascend",
        device_id=get_device_id(),
        save_graphs=config.save_graphs,
        save_graphs_path=graph_dir,
    )

    device_num = get_device_num()
    rank = get_rank_id()
    # configurations for distributed training
    if config.is_distributed:
        init()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num,
        )

    logger.info("Initializing training dataset.")
    vocab_size, train_dataset = create_dataset(
        config.train_data,
        config.dict,
        collate_conf=config.collate_conf,
        dataset_conf=config.dataset_conf,
        rank=rank,
        group_size=device_num,
        number_workers=1,
    )
    eval_dataset = None
    if config.training_with_eval:
        logger.info("Initializing evaluation dataset.")
        collate_conf = config.collate_conf
        collate_conf["use_speed_perturb"] = False
        collate_conf["use_spec_aug"] = False
        _, eval_dataset = create_dataset(
            config.eval_data,
            config.dict,
            collate_conf=collate_conf,
            dataset_conf=config.test_dataset_conf,
            rank=rank,
            group_size=device_num,
            number_workers=1,
        )

    input_dim = config.collate_conf.feature_extraction_conf.mel_bins
    steps_size = train_dataset.get_dataset_size()
    logger.info("Training dataset has %d steps in each epoch.", steps_size)

    # define network
    net_with_loss = creadte_asr_model(config, input_dim, vocab_size)
    weights = ParameterTuple(net_with_loss.trainable_params())
    logger.info("Total parameter of ASR model: %s.", get_parameter_numel(net_with_loss))

    start_epoch_num = 0
    if config.resume_ckpt != "":
        param_dict = load_checkpoint(
            config.resume_ckpt, filter_prefix=["learning_rate", "global_step"]
        )
        start_epoch_num = int(param_dict.get("epoch_num", 0).asnumpy().item())
        load_param_into_net(net_with_loss, param_dict)
        logger.info("Successfully loading the pre-trained model")

    if config.scheduler == "none":
        optimizer = Adam(weights, learning_rate=config.optim_conf.lr)
    elif config.scheduler == "warmuplr":
        lr_schedule = ASRWarmupLR(
            learninig_rate=config.optim_conf.lr,
            warmup_steps=config.scheduler_conf.warmup_steps,
            start_steps=start_epoch_num * steps_size,
        )
        optimizer = Adam(weights, learning_rate=lr_schedule)
    else:
        raise ValueError("Only 'none', and 'warmuplr' are supported.")

    update_cell = DynamicLossScaleUpdateCell(
        loss_scale_value=1024, scale_factor=2, scale_window=1000
    )
    train_net = TrainOneStepWithLossScaleCell(net_with_loss, optimizer, update_cell)

    callback_list = [
        MemoryStartTimeCallback(),
        TimeMonitor(config.max_epoch, steps_size),
        ResumeCallback(start_epoch_num=start_epoch_num),
    ]

    if config.training_with_eval:
        eval_net = create_asr_eval_net(net_with_loss, device_num)
        callback_list.append(
            EvalCallback(
                eval_net,
                eval_dataset,
                columns_list,
                steps_size * config.save_checkpoint_epochs,
                model_dir,
                net_with_loss,
            )
        )
    elif config.save_checkpoint and rank == 0:
        ckpt_append_info = [{"epoch_num": 0}]
        config_ck = CheckpointConfig(
            save_checkpoint_steps=steps_size * config.save_checkpoint_epochs,
            keep_checkpoint_max=config.keep_checkpoint_max,
            append_info=ckpt_append_info,
        )
        callback_list.append(ModelCheckpoint(directory=model_dir, config=config_ck))

    callback_list.append(CalRunTimeCallback())

    model = Model(train_net)
    logger.info("Training start.")
    model.train(
        config.max_epoch - start_epoch_num,
        train_dataset,
        callbacks=callback_list,
        dataset_sink_mode=False,
    )


if __name__ == "__main__":
    train()
