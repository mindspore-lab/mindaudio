"""train_criteo."""

import os

from mindspore import context, Tensor, ParameterTuple
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.nn import TrainOneStepCell
from mindspore.nn.optim import Adam
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore.common.dtype as mstype
from mindspore import nn
import mindspore.ops as ops

from mindaudio.models.deepspeech2 import DeepSpeechModel
from mindaudio.scheduler.lr_generator import get_lr
from hparams.hparams import parse_args
from dataset import create_base_dataset, train_data_pipeline


class NetWithLossClass(nn.Cell):
    """
    NetWithLossClass definition
    """

    def __init__(self, network, ascend=False):
        super(NetWithLossClass, self).__init__(auto_prefix=False)
        if ascend:
            self.loss = ops.CTCLoss(ctc_merge_repeated=True,
                                    ignore_longer_outputs_than_inputs=True)
        else:
            self.loss = ops.CTCLoss(ctc_merge_repeated=True)
        self.network = network
        self.ReduceMean_false = ops.ReduceMean(keep_dims=False)
        self.squeeze_op = ops.Squeeze(0)
        self.cast_op = ops.Cast()

    def construct(self, inputs, input_length, target_indices, label_values):
        predict, output_length = self.network(inputs, input_length)
        predict = self.cast_op(predict, mstype.float32)
        loss = self.loss(predict, target_indices, label_values, self.cast_op(output_length, mstype.int32))
        return self.ReduceMean_false(loss[0])


def train(args):
    ds_train = create_base_dataset(manifest_filepath=args.DataConfig.train_manifest,
                                   labels=args.labels, rank=rank_id, group_size=group_size)
    ds_train = train_data_pipeline(ds_train, batch_size=args.DataConfig.batch_size,
                                   audio_conf=args.DataConfig.SpectConfig)

    steps_size = ds_train.get_dataset_size()
    lr = get_lr(lr_init=args.OptimConfig.learning_rate, total_epochs=args.TrainingConfig.epochs,
                steps_per_epoch=steps_size)
    lr = Tensor(lr)

    deepspeech_net = DeepSpeechModel(batch_size=args.DataConfig.batch_size,
                                     rnn_hidden_size=args.ModelConfig.hidden_size,
                                     nb_layers=args.ModelConfig.hidden_layers,
                                     labels=args.labels,
                                     rnn_type=args.ModelConfig.rnn_type,
                                     audio_conf=args.DataConfig.SpectConfig,
                                     bidirectional=True,
                                     device_target=args.device_target)

    loss_net = NetWithLossClass(deepspeech_net, ascend=(args.device_target == 'Ascend'))
    weights = ParameterTuple(deepspeech_net.trainable_params())

    optimizer = Adam(weights, learning_rate=args.OptimConfig.learning_rate, eps=args.OptimConfig.epsilon,
                     loss_scale=args.OptimConfig.loss_scale)
    train_net = TrainOneStepCell(loss_net, optimizer)
    train_net.set_train(True)
    if args.pre_trained_model_path != '':
        param_dict = load_checkpoint(args.pre_trained_model_path)
        load_param_into_net(train_net, param_dict)
        print('Successfully loading the pre-trained model')

    model = Model(train_net)
    callback_list = [TimeMonitor(steps_size), LossMonitor()]

    if args.is_distributed:
        args.CheckpointConfig.ckpt_path = os.path.join(args.CheckpointConfig.ckpt_path,
                                                         'ckpt_' + str(get_rank()) + '/')

    config_ck = CheckpointConfig(save_checkpoint_steps=5,
                                 keep_checkpoint_max=args.CheckpointConfig.keep_checkpoint_max)
    ckpt_cb = ModelCheckpoint(prefix=args.CheckpointConfig.ckpt_file_name_prefix,
                              directory=args.CheckpointConfig.ckpt_path, config=config_ck)

    callback_list.append(ckpt_cb)
    print(callback_list)
    model.train(args.TrainingConfig.epochs, ds_train, callbacks=callback_list, dataset_sink_mode=data_sink)


if __name__ == '__main__':
    rank_id = 0
    group_size = 1
    args = parse_args()
    data_sink = (args.device_target != "CPU")
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=False)
    if args.device_target == "GPU":
        context.set_context(enable_graph_kernel=True)
    if args.is_distributed:
        init()
        rank_id = get_rank()
        group_size = get_group_size()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        if args.device_target == 'Ascend':
            device_id = int(args.device_id)
            context.set_context(device_id=device_id)
    train(args)

