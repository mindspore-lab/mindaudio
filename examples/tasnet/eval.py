import argparse
import json
import os

import mindspore
import mindspore.dataset as ds
import mindspore.ops as ops
from data import DatasetGenerator
from mindspore import (
    Parameter,
    Tensor,
    context,
    load_checkpoint,
    load_param_into_net,
    set_seed,
)

import mindaudio.data.io as io
from mindaudio.loss.separation_loss import Separation_Loss
from mindaudio.metric.snr import cal_SDRi, cal_SISNRi
from mindaudio.models.tasnet import TasNet
from mindaudio.utils.hparams import parse_args


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
        # if len(samples) > 128000:
        #     continue
        file_infos.append((wav_path, len(samples)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + ".json"), "w") as f:
        json.dump(file_infos, f, indent=4)


def preprocess(args):
    """Process all files"""
    print("Begin preprocess")
    for data_type in ["test"]:
        for speaker in ["mix", "s1", "s2"]:
            preprocess_one_dir(
                os.path.join(args.in_dir, data_type, speaker),
                os.path.join(args.out_dir, data_type),
                speaker,
                sample_rate=args.sample_rate,
            )
    print("Preprocess done")


def evaluate(args):
    total_SISNRi = 0
    total_SDRi = 0
    total_cnt = 0

    set_seed(1)

    # Load model
    model = TasNet(
        args.L,
        args.N,
        args.hidden_size,
        args.num_layers,
        bidirectional=bool(args.bidirectional),
        nspk=args.nspk,
    )
    model.set_train(mode=False)
    home = os.path.dirname(os.path.realpath(__file__))
    ckpt = os.path.join(home, args.model_path)
    print("=====> load params into generator")
    params = load_checkpoint(ckpt)
    load_param_into_net(model, params)
    print("=====> finish load generator")

    # Load data
    tt_dataset = DatasetGenerator(
        args.data_dir, args.eval_batch_size, sample_rate=args.sample_rate, L=args.L
    )
    tt_loader = ds.GeneratorDataset(
        tt_dataset, ["mixture", "lens", "sources"], shuffle=False
    )
    tt_loader = tt_loader.batch(batch_size=args.eval_batch_size)

    for data in tt_loader.create_dict_iterator():
        padded_mixture = data["mixture"]
        mixture_lengths = data["lens"]
        padded_source = data["sources"]
        padded_mixture = ops.Cast()(padded_mixture, mindspore.float32)
        padded_source = ops.Cast()(padded_source, mindspore.float32)
        # mixture_lengths_with_list = get_input_with_list(args.data_dir)
        estimate_source = model(padded_mixture)

        my_loss = Separation_Loss()
        loss, max_snr, estimate_source, reorder_estimate_source = my_loss(
            padded_source, estimate_source, mixture_lengths
        )
        # Remove padding and flat
        # mixture = remove_pad_and_flat(padded_mixture, mixture_lengths_with_list)
        # source = remove_pad_and_flat(padded_source, mixture_lengths_with_list)
        mixture = remove_pad_and_flat(padded_mixture)
        source = remove_pad_and_flat(padded_source)
        # NOTE: use reorder estimate source
        estimate_source = remove_pad_and_flat(reorder_estimate_source)
        # mixture_lengths_with_list)
        for mix, src_ref, src_est in zip(mixture, source, estimate_source):
            print("Utt", total_cnt + 1)
            # Compute SDRi
            if args.cal_sdr:
                avg_SDRi = cal_SDRi(src_ref, src_est, mix)
                total_SDRi += avg_SDRi
                print("\tSDRi={0:.2f}".format(avg_SDRi))
            # Compute SI-SNRi
            avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
            print("\tSI-SNRi={0:.2f}".format(avg_SISNRi))
            total_SISNRi += avg_SISNRi
            total_cnt += 1
    if args.cal_sdr:
        print("Average SDR improvement: {0:.2f}".format(total_SDRi / total_cnt))
    print("Average SISNR improvement: {0:.2f}".format(total_SISNRi / total_cnt))


def remove_pad_and_flat(inputs):
    """
    Args:
        inputs: Tensor, [B, C, K, L] or [B, K, L]
        inputs_lengths: Tensor, [B]
    Returns:
        results: a list containing B items, each item is [C, T], T varies
    """
    results = []
    dim = inputs.ndim
    if dim == 4:
        C = inputs.shape[1]
    for i, input in enumerate(inputs):
        if dim == 4:  # [B, C, K, L]
            results.append(input[:, :3320].view(C, -1).asnumpy())
        elif dim == 3:  # [B, K, L]
            results.append(input[:3320].view(-1).asnumpy())
    return results


if __name__ == "__main__":
    print("*+*+" * 100)
    args = parse_args()
    print(args)
    context.set_context(device_target="CPU")
    evaluate(args)
