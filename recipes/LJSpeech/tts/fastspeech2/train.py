import os
import numpy as np

import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.communication import init
from mindspore.amp import all_finite
from mindspore import SummaryCollector

from time import time
import argparse
import ast

import mindaudio
from recipes.LJSpeech.tts.fastspeech2.dataset import create_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='FastSpeech2 training')
    parser.add_argument('--is_distributed', type=ast.literal_eval, default=False)
    parser.add_argument('--device_target', type=str, default="GPU", choices=("GPU", "CPU", 'Ascend'))
    parser.add_argument('--device_id', '-i', type=int, default=0)
    parser.add_argument('--context_mode', type=str, default='py', choices=['py', 'graph'])
    parser.add_argument('--config', '-c', type=str, default='recipes/LJSpeech/tts/fastspeech2/fastspeech2.yaml')
    parser.add_argument('--restore', '-r', type=str, default='')
    parser.add_argument('--data_url', default='')
    parser.add_argument('--train_url', default='')
    args = parser.parse_args()
    return args


_grad_scale = ops.MultitypeFuncGraph("grad_scale")

@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.cast(ops.Reciprocal()(scale), ops.dtype(grad))


class MyTrainOneStepCell(nn.TrainOneStepCell):
    def __init__(self,
                 network,
                 optimizer,
                 sens=ms.Tensor(1.),
                 grad_clip=True,
                 max_grad_norm=1.,
    ):
        super().__init__(network, optimizer, sens)
        self.grad_clip = grad_clip
        self.max_grad_norm = max_grad_norm
        self.t = time()

    def construct(self, *args):
        losses, grads = ops.value_and_grad(self.network, weights=self.weights, has_aux=True)(*args)

        losses = self.network.scale.unscale(losses)
        grads = self.network.scale.unscale(grads)
        if self.grad_clip:
            grads = ops.clip_by_global_norm(grads[1], clip_norm=self.max_grad_norm)
        grads = self.grad_reducer(grads)

        # cond = self.get_overflow_status(status, grads)
        # overflow = self.process_loss_scale(cond)
        overflow = all_finite(grads) and False
        if not overflow:
            self.optimizer(grads)

        t2 = self.t
        self.t = time()
        return losses, overflow, self.t - t2


def main():
    profiler = ms.Profiler(output_path='./')
    args = parse_args()
    hps = mindaudio.load_hparams(args.config)

    model, ckpt = mindaudio.create_model('FastSpeech2', hps, args.restore, is_train=True)

    mode = ms.context.PYNATIVE_MODE if args.context_mode == 'py' else ms.context.GRAPH_MODE
    ms.context.set_context(mode=mode, device_target=args.device_target)
    if args.is_distributed:
        init()
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
    rank = int(os.getenv('DEVICE_ID', '0')) if args.is_distributed else 0
    group = int(os.getenv('RANK_SIZE', '1')) if args.is_distributed else 1
    print('[info] rank: %d group: %d batch: %d' % (rank, group, hps.batch_size // group))
    ms.context.set_context(device_id=rank if args.is_distributed else args.device_id)

    np.random.seed(0)
    ms.set_seed(0)

    ds = create_dataset(
        data_path=hps.data_path,
        manifest_path=hps.manifest_path,
        batch_size=hps.batch_size // group,
        is_train=True,
        rank=rank,
        group_size=group
    )
    print('[info] num batches: %d' % ds.get_dataset_size())
    lr = nn.exponential_decay_lr(
        hps.learning_rate,
        0.96,
        int(ds.get_dataset_size() * hps.num_epochs),
        int(ds.get_dataset_size()),
        1000,
        is_stair=True
    )
    optimiser = nn.Adam(model.trainable_params(), learning_rate=lr, beta1=hps.beta1, beta2=hps.beta2, eps=hps.eps)
    global_step = 0
    if ckpt is not None:
        if 'cur_step' in ckpt:
            global_step = int(ckpt['cur_step'].asnumpy())
    print('[info] global_step:', global_step)

    network = MyTrainOneStepCell(model, optimiser, max_grad_norm=hps.max_grad_norm)

    slr = ops.ScalarSummary()
    slr('lr', lr)

    num_epochs = hps.num_epochs
    callbacks = []
    if rank == args.device_id:
        callbacks.append(ms.TimeMonitor())
        save = mindaudio.callbacks.SaveCallBack(
            model,
            save_step=hps.save_step,
            global_step=global_step,
            save_dir=hps.save_dir,
            optimiser=optimiser,
            train_url=args.train_url
        )
        callbacks.append(save)
        specified = {
            "collect_metric": False, 
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

    model = ms.Model(network=network)
    model.train(num_epochs, ds, dataset_sink_mode=False, callbacks=callbacks, initial_epoch=global_step // ds.get_dataset_size())
    profiler.analyse()

if __name__ == '__main__':
    main()
