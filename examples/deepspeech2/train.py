import argparse
import json
import os
import sys

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore import context, Tensor, ParameterTuple
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.nn import TrainOneStepCell
from mindspore.nn.optim import Adam
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

sys.path.append('.')
from examples.deepspeech2.config import train_config
from examples.deepspeech2.dataset import create_dataset
from examples.deepspeech2.eval_callback import SaveCallback
from mindaudio.models.deepspeech2 import DeepSpeechModel
from mindaudio.nn.lr_generator import get_lr


parser = argparse.ArgumentParser(description='DeepSpeech2 training')
parser.add_argument('--pre_trained_model_path', type=str, default='', help='Pretrained checkpoint path')
parser.add_argument('--is_distributed', action="store_true", default=False, help='Distributed training')
parser.add_argument('--bidirectional', action="store_false", default=True, help='Use bidirectional RNN')
parser.add_argument('--device_target', type=str, default="CPU", choices=("GPU", "CPU"),
                    help='Device target, support GPU and CPU, Default: GPU')
args = parser.parse_args()

class NetWithLossClass(nn.Cell):
    """
    NetWithLossClass definition
    """

    def __init__(self, network):
        super(NetWithLossClass, self).__init__(auto_prefix=False)
        self.loss = ops.CTCLoss(ctc_merge_repeated=True)
        self.network = network
        self.ReduceMean_false = ops.ReduceMean(keep_dims=False)
        self.squeeze_op = ops.Squeeze(0)
        self.cast_op = ops.Cast()

    def construct(self, inputs, input_length, target_indices, label_values):
        predict, output_length = self.network(inputs, input_length)
        loss = self.loss(predict, target_indices, label_values, self.cast_op(output_length, mstype.int32))
        return self.ReduceMean_false(loss[0])

if __name__ == '__main__':

    rank_id = 0
    group_size = 1
    config = train_config
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

    with open(config.DataConfig.labels_path) as label_file:
        labels = json.load(label_file)

    _, ds_train = create_dataset(data_path=config.DataConfig.data_path,
                              manifest_path=config.DataConfig.train_manifest,
                              labels=labels, normalize=True, train_mode=True,
                              batch_size=config.DataConfig.batch_size, rank=rank_id, group_size=group_size)
    steps_size = ds_train.get_dataset_size()

    lr = get_lr(lr_init=config.OptimConfig.learning_rate, total_epochs=config.TrainingConfig.epochs,
                steps_per_epoch=steps_size)
    lr = Tensor(lr)
    # lr = nn.exponential_decay_lr(learning_rate=config.OptimConfig.learning_rate, decay_rate=1.1,
    #                              total_step=config.TrainingConfig.epochs, step_per_epoch=steps_size, decay_epoch=1)

    deepspeech_net = DeepSpeechModel(batch_size=config.DataConfig.batch_size,
                                     rnn_hidden_size=config.ModelConfig.hidden_size,
                                     nb_layers=config.ModelConfig.hidden_layers,
                                     labels=labels,
                                     rnn_type=config.ModelConfig.rnn_type,
                                     audio_conf=config.DataConfig.SpectConfig,
                                     bidirectional=True,
                                     device_target=args.device_target)

    loss_net = NetWithLossClass(deepspeech_net)
    weights = ParameterTuple(deepspeech_net.trainable_params())

    optimizer = Adam(weights, learning_rate=config.OptimConfig.learning_rate, eps=config.OptimConfig.epsilon,
                     loss_scale=config.OptimConfig.loss_scale)
    train_net = TrainOneStepCell(loss_net, optimizer)
    train_net.set_train(True)
    if args.pre_trained_model_path != '':
        param_dict = load_checkpoint(args.pre_trained_model_path)
        load_param_into_net(train_net, param_dict)
        print('Successfully loading the pre-trained model')

    model = Model(train_net)
    callback_list = [TimeMonitor(steps_size), LossMonitor()]

    if args.is_distributed:
        config.CheckpointConfig.ckpt_path = os.path.join(config.CheckpointConfig.ckpt_path,
                                                         'ckpt_' + str(get_rank()) + '/')
        if rank_id == 0:

            callback_update = SaveCallback(config.CheckpointConfig.ckpt_path)
            callback_list += [callback_update]
    else:
        config_ck = CheckpointConfig(save_checkpoint_steps=5,
                                     keep_checkpoint_max=config.CheckpointConfig.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix=config.CheckpointConfig.ckpt_file_name_prefix,
                                  directory=config.CheckpointConfig.ckpt_path, config=config_ck)

        callback_list.append(ckpt_cb)
    print(callback_list)
    model.train(config.TrainingConfig.epochs, ds_train, callbacks=callback_list, dataset_sink_mode=data_sink)
