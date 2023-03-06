import os
import numpy as np

import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.communication import init
from mindspore import SummaryCollector

import argparse
import ast

from dataset import create_wavegrad_dataset
import mindaudio


def parse_args():
    parser = argparse.ArgumentParser(description='WaveGrad training')
    parser.add_argument('--is_distributed', type=ast.literal_eval, default=False)
    parser.add_argument('--device_target', type=str, default="CPU", choices=("GPU", "CPU", 'Ascend'))
    parser.add_argument('--device_id', '-i', type=int, default=0)
    parser.add_argument('--context_mode', type=str, default='graph', choices=['py', 'graph'])
    parser.add_argument('--config', '-c', type=str, default='recipes/LJSpeech/tts/wavegrad/wavegrad_base.yaml')
    parser.add_argument('--restore', '-r', type=str, default='')
    parser.add_argument('--data_url', default='')
    parser.add_argument('--train_url', default='')
    args = parser.parse_args()
    return args


_grad_scale = ops.MultitypeFuncGraph("grad_scale")

@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.cast(ops.Reciprocal()(scale), ops.dtype(grad))


class MyTrainOneStepCell(nn.TrainOneStepWithLossScaleCell):
    def __init__(self,
                 network,
                 optimizer,
                 max_grad_norm=1., 
                 scale_sense=ms.Tensor(1.),
                 grad_clip=True
    ):
        super().__init__(network, optimizer, scale_sense)
        self.grad_clip = grad_clip
        self.max_grad_norm = max_grad_norm
        self.slr = ops.ScalarSummary()

    def construct(self, noisy_audio, noise_scale, noise, spectrogram):
        loss = self.network(noisy_audio, noise_scale, noise, spectrogram)
        self.slr('loss', loss)

        status, scaling_sens = self.start_overflow_check(loss, self.scale_sense)
        self.slr('scaling_sens', scaling_sens)
        scaling_sens_filled = ops.ones_like(loss) * ops.cast(scaling_sens, ops.dtype(loss))
        grads = self.grad(self.network, self.weights)(noisy_audio, noise_scale, noise, spectrogram, scaling_sens_filled)
        grads = self.hyper_map(ops.partial(_grad_scale, scaling_sens), grads)
        if self.grad_clip:
            grads = ops.clip_by_global_norm(grads, clip_norm=self.max_grad_norm)
        grads = self.grad_reducer(grads)

        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        if not overflow:
            self.optimizer(grads)

        return loss, cond, scaling_sens

def main():
    args = parse_args()
    hps = mindaudio.load_config(args.config)

    mode = ms.context.PYNATIVE_MODE if args.context_mode == 'py' else ms.context.GRAPH_MODE
    ms.context.set_context(mode=mode, device_target=args.device_target)
    if args.is_distributed:
        init()
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
    rank = int(os.getenv('DEVICE_ID', '0'))
    group = int(os.getenv('RANK_SIZE', '1'))
    print('[info] rank: %d group: %d batch: %d' % (rank, group, hps.batch_size // group))
    ms.context.set_context(device_id=rank if args.is_distributed else args.device_id)

    np.random.seed(0)
    ms.set_seed(0)

    ds = create_wavegrad_dataset(
        hps=hps,
        batch_size=hps.batch_size // group,
        is_train=True,
        rank=rank,
        group_size=group
    )
    lr = nn.exponential_decay_lr(
        hps.learning_rate,
        0.96,
        int(ds.get_dataset_size() * hps.num_epochs),
        int(ds.get_dataset_size()),
        1000,
        is_stair=True
    )
    wavegrad, ckpt = mindaudio.create_model('wavegrad', hps, args.restore, is_train=True)
    optimiser = nn.Adam(wavegrad.trainable_params(), learning_rate=lr)
    global_step = 0
    if 'cur_step' in ckpt:
        global_step = int(ckpt['cur_step'].asnumpy())

    scale_sense = nn.DynamicLossScaleUpdateCell(loss_scale_value=2**12, scale_factor=2, scale_window=1000)
    net = MyTrainOneStepCell(wavegrad, optimiser, max_grad_norm=hps.max_grad_norm, scale_sense=scale_sense)

    slr = ops.ScalarSummary()
    slr('lr', lr)

    num_epochs = hps.num_epochs
    callbacks = []
    if rank == 0:
        callbacks.append(ms.LossMonitor(1))
        callbacks.append(ms.TimeMonitor())
        save = mindaudio.callbacks.SaveCallBack(
            wavegrad,
            save_step=hps.save_step,
            global_step=global_step,
            save_dir=hps.save_dir,
            optimiser=optimiser,
            train_url=args.train_url
        )
        callbacks.append(save)
        specified = {
            "collect_metric": True, 
            "histogram_regular": "^conv1*|^conv2*|^dense.*", 
            "collect_graph": True,
            "collect_dataset_graph": False,
        }
        summary_collector = SummaryCollector(
            summary_dir=os.path.join(args.train_url, 'summary_0'), 
            collect_specified_data=specified,
            collect_freq=200,
            keep_default_action=False, 
            collect_tensor_freq=5000
        )
        callbacks.append(summary_collector)

    model = ms.Model(network=net)
    model.train(num_epochs, ds, dataset_sink_mode=False, callbacks=callbacks, initial_epoch=global_step // ds.get_dataset_size())


if __name__ == '__main__':
    main()
