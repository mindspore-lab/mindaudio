import os
import sys

from mindspore import ParameterTuple, context, set_seed
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.nn.optim import Adam
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, SummaryCollector
from mindspore.train.serialization import load_checkpoint, load_param_into_net

sys.path.append('.')
from examples.conformer.dataset import create_dataset
from mindaudio.adapter.config import get_config
from mindaudio.adapter.log import get_logger
from mindaudio.adapter.parallel_info import get_device_id, get_device_num, get_rank_id
from mindaudio.models.conformer.src.model.asr_model import init_asr_model
from mindaudio.utils.callback import CalRunTimeCallback, EvalCallBack, MemoryStartTimeCallback, TimeMonitor
from mindaudio.utils.common import get_parameter_numel
from mindaudio.utils.scheduler import ASRWarmupLR
from mindaudio.utils.train_one_step import TrainOneStepWithLossScaleCell


logger = get_logger()
config = get_config('asr_config')


def train():
    """main function for asr_train."""
    # Set random seed
    set_seed(777)
    exp_dir = config.exp_name
    model_dir = os.path.join(exp_dir, 'model')
    graph_dir = os.path.join(exp_dir, 'graph')
    summary_dir = os.path.join(exp_dir, 'summary')
    context.set_context(mode=context.PYNATIVE_MODE,
                        device_target=config.device_target,
                        device_id=get_device_id(),
                        save_graphs=config.save_graphs,
                        save_graphs_path=graph_dir)

    device_num = get_device_num()
    rank = get_rank_id()
    # configurations for distributed training
    if config.is_distributed:
        init()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True,
                                          device_num=device_num)

    logger.info('Initializing training dataset.')
    vocab_size, train_dataset = create_dataset(
        config.train_data,
        config.train_manifest,
        collate_conf=config.collate_conf,
        dataset_conf=config.dataset_conf,
        rank=rank,
        group_size=device_num,
    )

    if config.training_with_eval:
        logger.info('Initializing evaluation dataset.')
        collate_conf = config.collate_conf
        collate_conf['use_speed_perturb'] = False
        collate_conf['use_spec_aug'] = False
        _, eval_dataset = create_dataset(
            config.eval_data,
            config.eval_manifest,
            collate_conf=collate_conf,
            dataset_conf=config.dataset_conf,
            rank=rank,
            group_size=device_num,
        )

    input_dim = config.collate_conf.feature_extraction_conf.mel_bins
    steps_size = train_dataset.get_dataset_size()
    logger.info('Training dataset has %d steps in each epoch.', steps_size)

    # define network
    net_with_loss = init_asr_model(config, input_dim, vocab_size)
    weights = ParameterTuple(net_with_loss.trainable_params())
    logger.info('Total parameter of ASR model: %s.', get_parameter_numel(net_with_loss))

    if config.scheduler == 'none':
        optimizer = Adam(weights, learning_rate=config.optim_conf.lr)
    elif config.scheduler == 'warmuplr':
        lr_schedule = ASRWarmupLR(learninig_rate=config.optim_conf.lr, warmup_steps=config.scheduler_conf.warmup_steps)
        optimizer = Adam(weights, learning_rate=lr_schedule)
    else:
        raise ValueError("Only 'none', and 'warmuplr' are supported.")

    if config.ckpt_file != '':
        param_dict = load_checkpoint(config.ckpt_file)
        load_param_into_net(net_with_loss, param_dict)
        logger.info('Successfully loading the pre-trained model')

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=1024, scale_factor=2, scale_window=1000)
    train_net = TrainOneStepWithLossScaleCell(net_with_loss, optimizer, update_cell)

    callback_list = [MemoryStartTimeCallback(), TimeMonitor(steps_size)]
    if config.training_with_eval and rank == 0:
        logger.info('Opening training_with_eval.')
        callback_list.append(
            EvalCallBack(net_with_loss, eval_dataset, steps_size * config.save_checkpoint_epochs, 'asr',
                         config.keep_checkpoint_max, model_dir))
    elif config.save_checkpoint and rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=steps_size * config.save_checkpoint_epochs,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(directory=model_dir, config=config_ck)
        callback_list.append(ckpt_cb)

    # mindinsight summary
    if config.enable_summary and rank == 0:
        summary_collector = SummaryCollector(summary_dir=summary_dir, collect_freq=config.log_interval)
        callback_list.append(summary_collector)

    callback_list.append(CalRunTimeCallback())

    model = Model(train_net)
    logger.info('Training start.')

    model.train(config.max_epoch, train_dataset, callbacks=callback_list, dataset_sink_mode=False)


if __name__ == '__main__':
    train()
