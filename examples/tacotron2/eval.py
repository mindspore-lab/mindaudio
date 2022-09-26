import time
import os
import sys
from os.path import join

import matplotlib
import matplotlib.pylab as plt
import numpy as np

import mindspore
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import Tensor
from mindspore.dataset.audio import DBToAmplitude, InverseMelScale, GriffinLim

sys.path.append('.')
from mindaudio.data.io import write
from mindaudio.models.tacotron2.tacotron2 import Tacotron2
from mindaudio.models.tacotron2.hparams import hparams as hps
from examples.tokenisers.utils import text_to_sequence

from mindaudio.models.tacotron2.config import config
from mindaudio.adapter.local_adapter import get_device_id, get_device_num

matplotlib.use('Agg')

context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target=config.device_target)


def load_model(ckpt_pth):
    '''
    load model
    '''
    net = Tacotron2()
    param_dict = load_checkpoint(ckpt_pth)

    load_param_into_net(net, param_dict)
    net.set_train(False)
    net.decoder.prenet.dropout.set_train(True)

    return net.to_float(mindspore.float32)


def infer(text, net):
    '''
    inference
    '''
    sequence = text_to_sequence(text, hps.text_cleaners)
    sequence = Tensor(sequence, mindspore.int32).view(1, -1)
    text_mask = Tensor(np.zeros(sequence.shape).astype('bool'))

    mel_outputs, mel_outputs_postnet, _, alignments = net.inference(
        sequence, text_mask)

    return (mel_outputs, mel_outputs_postnet, alignments)


def plot_data(data, figsize=(16, 4)):
    '''
    plot alignments
    '''
    _, axes = plt.subplots(1, len(data), figsize=figsize)
    for _, i in enumerate(range(len(data))):
        axes[i].imshow(data[i], aspect='auto', origin='lower')


def plot(output, dir_pth, filename):
    '''
    plot alignments
    '''
    mel_outputs, mel_outputs_postnet, alignments = output
    plot_data((mel_outputs.asnumpy()[0],
               mel_outputs_postnet.asnumpy()[0],
               alignments.asnumpy()[0].T))
    plt.savefig(join(dir_pth, filename + '.png'))


def _denormalize(S):
    '''denormalize '''
    return (np.clip(S, 0, 1) * -hps.min_level_db) + hps.min_level_db


def inv_melspectrogram(spec):
    mel = DBToAmplitude()(_denormalize(spec) + hps.ref_level_db)
    S = InverseMelScale(
        hps.num_freq,
        n_mels=hps.num_mels,
        sample_rate=hps.sample_rate,
        f_min=hps.fmin, 
        f_max=hps.fmax,
    )(mel)
    return GriffinLim(
        n_fft=(hps.num_freq - 1) * 2,
        n_iter=hps.gl_iters, 
        win_length=hps.win_length,
        hop_length=hps.hop_length,
    )(S ** hps.power)


def audio(output, dir_pth, filename):
    '''
    save waveform
    '''
    mel_outputs, _, _ = output

    wav = inv_melspectrogram(mel_outputs.asnumpy()[0])
    np.save(join(dir_pth, filename + '-wave.npy'), wav, allow_pickle=False)
    write(join(dir_pth, filename + '.wav'), wav, 22050)


def save_mel(output, dir_pth, filename):
    '''
    save mel spectrogram
    '''
    mel_outputs, _, _ = output
    np.save(
        join(
            dir_pth,
            filename +
            '-feats.npy'),
        mel_outputs.asnumpy()[0].T,
        allow_pickle=False)

def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))

def run_tacotron2_infer():
    ''' run tacotron2 inference '''
    model = load_model(config.model_ckpt)
    print('Successfully loading checkpoint {}'.format(config.model_ckpt))
    print(config.output_path)
    start = time.time()
    outputs = infer(config.text, model)
    end = time.time()
    print('inference elapsed :{}s'.format(end - start))
    plot(outputs, config.output_path, config.audioname)
    audio(outputs, config.output_path, config.audioname)
    save_mel(outputs, config.output_path, config.audioname)

if __name__ == '__main__':
    run_tacotron2_infer()
