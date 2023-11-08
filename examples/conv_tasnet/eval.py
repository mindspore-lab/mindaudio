import mindspore
import mindspore.dataset as ds
import mindspore.ops as ops
import numpy as np
from data import DatasetGenerator
from mindspore import context, load_checkpoint, load_param_into_net
from mir_eval.separation import bss_eval_sources

from mindaudio.loss.separation_loss import Convtasnet_Loss, NetWithLoss
from mindaudio.metric.snr import cal_SDRi, cal_SISNRi
from mindaudio.models.conv_tasnet import ConvTasNet
from mindaudio.utils.hparams import parse_args


def evaluate(args):
    total_SISNRi = 0
    total_SDRi = 0
    total_cnt = 0

    # Load model
    model = ConvTasNet(
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
    model.set_train(mode=False)
    param_dict = load_checkpoint(args.model_path)
    load_param_into_net(model, param_dict)

    # Load data
    tt_dataset = DatasetGenerator(
        args.data_dir,
        args.eval_batch_size,
        sample_rate=args.sample_rate,
        segment=args.segment,
    )
    tt_loader = ds.GeneratorDataset(
        tt_dataset, ["mixture", "lens", "sources"], shuffle=False
    )
    tt_loader = tt_loader.batch(batch_size=8)

    for data in tt_loader.create_dict_iterator():
        padded_mixture = data["mixture"]
        mixture_lengths = data["lens"]
        padded_source = data["sources"]
        padded_mixture = ops.cast(padded_mixture, mindspore.float32)
        padded_source = ops.cast(padded_source, mindspore.float32)
        mixture_lengths_with_list = mixture_lengths.asnumpy().tolist()
        estimate_source = model(padded_mixture)  # [B, C, T]

        my_loss = Convtasnet_Loss()
        _, _, estimate_source, reorder_estimate_source = my_loss(
            padded_source, estimate_source, mixture_lengths
        )
        mixture = remove_pad(padded_mixture, mixture_lengths_with_list)
        source = remove_pad(padded_source, mixture_lengths_with_list)
        # NOTE: use reorder estimate source
        estimate_source = remove_pad(reorder_estimate_source, mixture_lengths_with_list)
        # for each utterance
        for mix, src_ref, src_est in zip(mixture, source, estimate_source):
            print("Utt", total_cnt + 1)
            # Compute SDRi
            if args.cal_sdr:
                avg_SDRi = cal_SDRi(src_ref, src_est, mix)
                total_SDRi += avg_SDRi
                print("\tSDRi={0:.2f}".format(-avg_SDRi))
            # Compute SI-SNRi
            avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
            print("\tSI-SNRi={0:.2f}".format(avg_SISNRi))
            total_SISNRi += avg_SISNRi
            total_cnt += 1
    if args.cal_sdr:
        print("Average SDR improvement: {0:.2f}".format(total_SDRi / total_cnt))
    print("Average SISNR improvement: {0:.2f}".format(total_SISNRi / total_cnt))


def remove_pad(inputs, inputs_lengths):
    """
    Args:
        inputs: torch.Tensor, [B, C, T] or [B, T], B is batch size
        inputs_lengths: torch.Tensor, [B]
    Returns:
        results: a list containing B items, each item is [C, T], T varies
    """
    results = []
    dim = inputs.ndim
    if dim == 3:
        C = inputs.shape[1]
    for i, data in enumerate(inputs):
        if dim == 3:  # [B, C, T]
            results.append(data[:, : inputs_lengths[i]].view(C, -1).asnumpy())
        elif dim == 2:  # [B, T]
            results.append(data[: inputs_lengths[i]].view(-1).asnumpy())
    return results


if __name__ == "__main__":
    arg = parse_args()
    context.set_context(
        mode=context.PYNATIVE_MODE,
        device_target=arg.device_target,
        device_id=arg.device_id,
    )
    print(arg)
    evaluate(arg)
