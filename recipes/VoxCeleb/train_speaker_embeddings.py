"""
Recipe for training speaker embeddings using the VoxCeleb Dataset.
"""
import math
import os
import pickle
import random
import time
from datetime import datetime
from multiprocessing import Manager, Process

import mindspore as ms
import mindspore.dataset as ds
import mindspore.nn as nn
import numpy as np
import wget
from mindspore import Tensor, context, load_checkpoint, load_param_into_net
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.context import ParallelMode
from mindspore.nn import FixedLossScaleUpdateCell
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, RunContext, _InternalCallbackParam

import mindaudio.data.io as io
from config import config as hparams
from loss_scale import TrainOneStepWithLossScaleCellv2 as TrainOneStepWithLossScaleCell
from mindaudio.data.features import fbank
from mindaudio.data.processing import stereo_to_mono
from mindaudio.models.ecapatdnn import Classifier, EcapaTDNN
from reader import DatasetGeneratorBatch as DatasetGenerator
from sampler import DistributedSampler
from spec_augment import EnvCorrupt, InputNormalization, TimeDomainSpecAugment
from util import AdditiveAngularMargin
from voxceleb_prepare import prepare_voxceleb

spk_id_encoded_dict = {}
spk_id_encoded = -1


def dataio_prep():
    "Creates the datasets and their data processing pipelines."

    snt_len_sample = int(hparams.sample_rate * hparams.sentence_len)

    # Define audio pipeline:
    def audio_pipeline(duration, wav, start, stop):
        if hparams.random_chunk:
            duration_sample = int(float(duration) * hparams.sample_rate)
            start = random.randint(0, duration_sample - snt_len_sample)
            stop = start + snt_len_sample
        else:
            start = int(start)
            stop = int(stop)

        sig, _ = io.read(str(wav))
        if len(sig.shape) > 1:
            sig = stereo_to_mono(sig)

        return sig[start:stop]

    # Define text pipeline:
    def label_pipeline(spk_id):
        spk_id = str(spk_id)
        return spk_id_encoded_dict[spk_id]

    train_datalist = []
    for i in range(hparams.data_process_num):
        train_data = ms.dataset.CSVDataset(
            dataset_files=[hparams.train_annotation],
            num_parallel_workers=hparams.dataloader_options.num_workers,
            shuffle=hparams.dataloader_options.shuffle,
            num_shards=hparams.data_process_num,
            shard_id=i,
        )

        train_data = train_data.map(
            lambda duration, wav, start, stop: audio_pipeline(duration, wav, start, stop),
            input_columns=["duration", "wav", "start", "stop"],
            output_columns=["sig"],
            column_order=["ID", "sig", "spk_id"],
        )

        train_data = train_data.map(
            operations=[label_pipeline],
            input_columns=["spk_id"],
            output_columns=["spk_id_encoded"],
            column_order=["ID", "sig", "spk_id_encoded"],
        )

        train_data = train_data.project(columns=["sig", "spk_id_encoded"])
        train_datalist.append(train_data)

    return train_datalist


def preprocess_raw_new(
    fidx, fea_utt_lst, label_utt_lst, samples_dict_global, labels_dict_global, output_path,
):
    """merge single files into one

    :param samples_per_file: Number of samples per file
    :return: None
    """
    # initialize
    samples = []
    labels = []
    samples_dict = {}
    labels_dict = {}
    offset = 0
    offset_label = 0
    file_ind = fidx
    count = 0
    interval = 500
    for fea_path, label_path in zip(fea_utt_lst, label_utt_lst):
        fea = np.load(fea_path)
        label = np.load(label_path)
        nplabel = None
        if label.shape[0] != fea.shape[0]:
            print("shape not same：", label.shape[0], "!=", fea.shape[0])
            break
        else:
            nplabel = label.squeeze()
        fea_flat = fea.flatten()
        utt = fea_path[fea_path.rfind("/") + 1 :]
        samples_dict[utt[utt.rfind("/") + 1 :]] = (
            file_ind,
            offset,
            fea_flat.shape[0],
        )
        labels_dict[utt[utt.rfind("/") + 1 :]] = (
            file_ind,
            offset_label,
            nplabel.shape[0],
        )
        samples.append(fea_flat)
        labels.append(nplabel)
        offset += fea_flat.shape[0]
        offset_label += nplabel.shape[0]
        count += 1
        if count % interval == 0:
            print("process", fidx, count)

    labels = np.hstack(labels)
    np.save(os.path.join(output_path, f"{file_ind}_label.npy"), labels)
    samples = np.hstack(samples)
    np.save(os.path.join(output_path, f"{file_ind}.npy"), samples)
    print("save to", os.path.join(output_path))
    samples_dict_global.update(samples_dict)
    labels_dict_global.update(labels_dict)
    print("process", fidx, "done")


def data_trans_dp(datasetPath, dataSavePath):
    if not os.path.exists(dataSavePath):
        os.makedirs(dataSavePath)
    fea_lst = os.path.join(datasetPath, "fea.lst")
    label_lst = os.path.join(datasetPath, "label.lst")
    print("fea_lst, label_lst:", fea_lst, label_lst)
    fea_utt_lst = []
    label_utt_lst = []
    with open(os.path.join(datasetPath, "fea.lst"), "r") as fp:
        for line in fp:
            fea_utt_lst.append(os.path.join(datasetPath, line.strip()))
    with open(os.path.join(datasetPath, "label.lst"), "r") as fp:
        for line in fp:
            label_utt_lst.append(os.path.join(datasetPath, line.strip()))

    print("total length of fea, label:", len(fea_utt_lst), len(label_utt_lst))

    fea_utt_lst_new = []
    label_utt_lst_new = []
    epoch_len = 73357
    for idx in range((int)(len(fea_utt_lst))):
        if (idx + 1) % epoch_len == 0:
            continue
        fea_utt_lst_new.append(fea_utt_lst[idx])
        label_utt_lst_new.append(label_utt_lst[idx])

    print(len(fea_utt_lst_new), len(label_utt_lst_new))
    fea_utt_lst = fea_utt_lst_new
    label_utt_lst = label_utt_lst_new

    samples_per_file = 4000
    total_process_num = math.ceil(len(fea_utt_lst) / samples_per_file)
    print(
        "samples_per_file, total_process_num:", samples_per_file, total_process_num,
    )
    samples_dict = Manager().dict()
    labels_dict = Manager().dict()

    thread_num = 4
    print(datetime.now().strftime("%m-%d-%H:%M:%S"))
    batchnum = math.ceil(total_process_num / thread_num)
    print("batch num:", batchnum)
    for batchid in range(batchnum):
        threadlist = []
        for idx in range(thread_num):
            start = (batchid * thread_num + idx) * samples_per_file
            end = (batchid * thread_num + idx + 1) * samples_per_file
            if start >= len(fea_utt_lst):
                break
            if end > len(fea_utt_lst):
                end = len(fea_utt_lst)
            print(batchid * thread_num + idx, "start, end:", start, end)
            p = Process(
                target=preprocess_raw_new,
                args=(
                    batchid * thread_num + idx,
                    fea_utt_lst[start:end],
                    label_utt_lst[start:end],
                    samples_dict,
                    labels_dict,
                    dataSavePath,
                ),
            )
            p.start()
            threadlist.append(p)
        for p in threadlist:
            p.join()
    print(datetime.now().strftime("%m-%d-%H:%M:%S"))
    pickle.dump(
        dict(samples_dict), open(os.path.join(dataSavePath, "ind_sample.p"), "wb"),
    )
    pickle.dump(
        dict(labels_dict), open(os.path.join(dataSavePath, "ind_label.p"), "wb"),
    )


def create_dataset(cfg, data_home, shuffle=False):
    """
    create a train or evaluate cifar10 dataset for resnet50
    Args:
        data_home(string): the path of dataset.
        batch_size(int): the batch size of dataset.
        repeat_num(int): the repeat times of dataset. Default: 1
    Returns:
        dataset
    """

    dataset_generator = DatasetGenerator(data_home)
    distributed_sampler = None
    if cfg.run_distribute:
        distributed_sampler = DistributedSampler(len(dataset_generator), cfg.group_size, cfg.rank, shuffle=True)
    vox2_ds = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=shuffle, sampler=distributed_sampler,)
    cnt = int(len(dataset_generator) / cfg.group_size)
    return vox2_ds, cnt


class CorrectLabelNum(nn.Cell):
    def __init__(self):
        super(CorrectLabelNum, self).__init__()
        self.argmax = ms.ops.Argmax(axis=1)
        self.sum = ms.ops.ReduceSum()

    def construct(self, output, target):
        output = self.argmax(output)
        correct = self.sum((output == target).astype(ms.dtype.float32))
        return correct


class BuildTrainNetwork(nn.Cell):
    """Build train network."""

    def __init__(
        self, my_network, classifier, lossfunc, my_criterion, train_batch_size, class_num_,
    ):
        super(BuildTrainNetwork, self).__init__()
        self.network = my_network
        self.classifier = classifier
        self.criterion = my_criterion
        self.lossfunc = lossfunc
        # Initialize self.output
        self.output = ms.Parameter(Tensor(np.ones((train_batch_size, class_num_)), ms.float32), requires_grad=False,)
        self.onehot = ms.nn.OneHot(depth=class_num_, axis=-1, dtype=ms.float32)

    def construct(self, input_data, label):
        output = self.network(input_data.astype(ms.float32))
        label_onehot = self.onehot(label.astype(ms.int32))
        # Get the network output and assign it to self.output
        logits = self.classifier(output)
        output = self.lossfunc(logits, label_onehot)
        self.output = output
        loss0 = self.criterion(output, label_onehot)
        return loss0


def update_average(loss_, avg_loss, step):
    avg_loss -= avg_loss / step
    avg_loss += loss_ / step
    return avg_loss


def train_net(
    rank, model, epoch_max, data_train, ckpt_cb, steps_per_epoch, train_batch_size,
):
    """define the training method"""
    # Create dict to save internal callback object's parameters
    cb_params = _InternalCallbackParam()
    cb_params.train_network = model
    cb_params.epoch_num = epoch_max
    cb_params.batch_num = steps_per_epoch
    cb_params.cur_epoch_num = 0
    cb_params.cur_step_num = 0
    run_context = RunContext(cb_params)
    ckpt_cb.begin(run_context)
    if rank == 0:
        print("============== Starting Training ==============")
    correct_num = CorrectLabelNum()
    correct_num.set_train(False)

    for epoch in range(epoch_max):
        t_start = time.time()
        train_loss = 0
        avg_loss = 0
        train_loss_cur = 0
        train_correct_cur = 0
        train_correct = 0
        print_dur = 3000
        for idx, (data, gt_classes) in enumerate(data_train):
            if data.shape[0] != train_batch_size:
                continue
            model.set_train()
            batch_loss, _, _, output = model(data, gt_classes)
            correct = correct_num(output, gt_classes)
            train_loss += batch_loss
            train_correct += correct.sum()
            train_loss_cur += batch_loss
            avg_loss = update_average(batch_loss, avg_loss, idx + 1)
            train_correct_cur += correct.sum()
            if rank == 0 and idx % print_dur == 0:
                cur_loss = train_loss_cur.asnumpy()
                acc = correct.sum().asnumpy() / float(train_batch_size)
                total_avg = train_loss.asnumpy() / float(idx + 1)
                if idx > 0:
                    cur_loss = train_loss_cur.asnumpy() / float(print_dur)
                    acc = train_correct_cur.asnumpy() / float(train_batch_size * print_dur)
                print(
                    f"{datetime.now()}, epoch:{epoch + 1}/{epoch_max}, \
                    iter-{idx}/{steps_per_epoch},"
                    f"cur loss:{cur_loss:.4f}, \
                    aver loss:{avg_loss.asnumpy():.4f},"
                    f"total_avg loss:{total_avg:.4f}, acc_aver:{acc:.4f}"
                )
                train_loss_cur = 0
                train_correct_cur = 0
            # Update current step number
            cb_params.cur_step_num += 1
            # Check whether save checkpoint or not
            if rank == 0:
                ckpt_cb.step_end(run_context)

        cb_params.cur_epoch_num += 1
        my_train_loss = train_loss / steps_per_epoch
        my_train_accuracy = 100 * train_correct / (train_batch_size * steps_per_epoch)
        time_used = time.time() - t_start
        fps = train_batch_size * steps_per_epoch / time_used
        if rank == 0:
            print("epoch[{}], {:.2f} imgs/sec".format(epoch, fps))
            print("Train Loss:", my_train_loss)
            print("Train Accuracy:", my_train_accuracy, "%")


def triangular():
    """
    triangular for cyclic LR. https://arxiv.org/abs/1506.01186
    """
    return 1.0


def triangular2(cycle):
    """
    triangular2 for cyclic LR. https://arxiv.org/abs/1506.01186
    """
    return 1.0 / (2.0 ** (cycle - 1))


def learning_rate_clr_triangle_function(step_size, max_lr, base_lr, clr_iterations):
    """
    get learning rate for cyclic LR. https://arxiv.org/abs/1506.01186
    """
    cycle = math.floor(1 + clr_iterations / (2 * step_size))
    x = abs(clr_iterations / step_size - 2 * cycle + 1)
    return base_lr + (max_lr - base_lr) * max(0, (1 - x)) * triangular()


def train():
    # init distributed
    if hparams.run_distribute:
        device_id = int(os.getenv("DEVICE_ID", "0"))
        context.set_context(
            mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id,
        )
        init()
        hparams.rank = get_rank()
        hparams.group_size = get_group_size()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True, device_num=8, parameter_broadcast=True,
        )
    else:
        hparams.rank = 0
        hparams.group_size = 1
        context.set_context(
            mode=context.GRAPH_MODE, device_target="Ascend", device_id=hparams.device_id,
        )
    data_dir = hparams.train_data_path
    in_channels = hparams.in_channels
    channels = hparams.channels
    base_lrate = hparams.base_lrate
    max_lrate = hparams.max_lrate
    weight_decay = hparams.weight_decay
    num_epochs = hparams.num_epochs
    minibatch_size = hparams.minibatch_size
    emb_size = hparams.emb_size
    clc_step_size = hparams.step_size
    class_num = hparams.class_num
    ckpt_save_dir = hparams.ckpt_save_dir
    # Configure operation information

    mymodel = EcapaTDNN(
        in_channels, channels=(channels, channels, channels, channels, channels * 3), lin_neurons=emb_size,
    )
    # Construct model
    ds_train, steps_per_epoch_train = create_dataset(hparams, data_dir)
    print(
        f"group_size:{hparams.group_size}, \
        data total len:{steps_per_epoch_train}"
    )
    # Define the optimizer and model
    my_classifier = Classifier(1, 0, emb_size, class_num)
    aam = AdditiveAngularMargin(0.2, 30)
    lr_list = []
    lr_list_total = steps_per_epoch_train * num_epochs
    for i in range(lr_list_total):
        lr_list.append(learning_rate_clr_triangle_function(clc_step_size, max_lrate, base_lrate, i))

    loss = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=False, reduction="mean")

    scale_mag = FixedLossScaleUpdateCell(loss_scale_value=2 ** 14)
    model_constructed = BuildTrainNetwork(mymodel, my_classifier, aam, loss, minibatch_size, class_num)
    opt = nn.Adam(model_constructed.trainable_params(), learning_rate=lr_list, weight_decay=weight_decay,)
    model_constructed = TrainOneStepWithLossScaleCell(model_constructed, opt, scale_sense=scale_mag)

    if hparams.pre_trained:
        pre_trained_model = os.path.join(ckpt_save_dir, hparams.checkpoint_path)
        param_dict = load_checkpoint(pre_trained_model)
        # load parameter to the network
        load_param_into_net(model_constructed, param_dict)
    # CheckPoint CallBack definition
    save_steps = int(steps_per_epoch_train / 10)
    checkpoint_max = hparams.keep_checkpoint_max
    config_ck = CheckpointConfig(save_checkpoint_steps=save_steps, keep_checkpoint_max=checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="ecapatdnn_vox12", directory=ckpt_save_dir, config=config_ck)

    train_net(
        hparams.rank, model_constructed, num_epochs, ds_train, ckpoint_cb, steps_per_epoch_train, minibatch_size,
    )


def generate_npy(
    iterator, spec_aug, save_dir, label_fp_list_global, fea_fp_list_global, batch_counts, index, file_lock
):
    count = 0
    label_fp_list = []
    fea_fp_list = []
    for batch in iterator:
        wavs = batch["sig"].astype(ms.float32)
        lens = np.ones(wavs.shape[0])
        wavs_aug_tot = []
        wavs_aug_tot.append(wavs)

        for aug in spec_aug:
            wavs_aug = Tensor(aug.construct(wavs.asnumpy(), lens), ms.float32)

            if wavs_aug.shape[1] > wavs.shape[1]:
                wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
            else:
                zeroslike = ms.ops.ZerosLike()
                zero_sig = zeroslike(wavs).astype(ms.float32)
                zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                wavs_aug = zero_sig

            if hparams.concat_augment:
                wavs_aug_tot.append(wavs_aug)
            else:
                wavs = wavs_aug
                wavs_aug_tot[0] = wavs

        wavs = ms.ops.concat(wavs_aug_tot, axis=0)
        n_augment = len(wavs_aug_tot)

        feats = fbank(
            wavs.asnumpy(), deltas=False, n_mels=80, left_frames=0, right_frames=0, n_fft=400, hop_length=160,
        ).transpose(0, 2, 1)

        normal_func = InputNormalization(norm_type="sentence", std_norm=False)
        feats = normal_func.construct(feats)

        ct = datetime.now()
        ts = ct.timestamp()
        id_save_name = str(ts) + "_" + str(index) + "_id.npy"
        fea_save_name = str(ts) + "_" + str(index) + "_fea.npy"
        spkid = batch["spk_id_encoded"].asnumpy()
        out_spkid = []
        for i in range(n_augment):
            out_spkid.append([spkid])

        spkid = np.concatenate(out_spkid).reshape(-1, 1)

        np.save(os.path.join(save_dir, id_save_name), spkid)
        np.save(os.path.join(save_dir, fea_save_name), feats)
        label_fp_list.append(id_save_name)
        fea_fp_list.append(fea_save_name)
        count = count + 1
        percentage = float(count) / batch_counts * 100
        percentage = round(percentage, 2)
        print("Process {} percentage {}%".format(index, percentage))

    file_lock.acquire()
    label_fp_list_global.extend(label_fp_list)
    fea_fp_list_global.extend(fea_fp_list)
    file_lock.release()


def generate_train_data():
    print("Generate train data.")
    context.set_context(device_target="CPU")
    if not os.path.exists(os.path.join(hparams.save_folder)):
        os.makedirs(os.path.join(hparams.save_folder), exist_ok=False)

    # Download verification list (to exlude verification sentences from train)
    veri_file_path = os.path.join(hparams.save_folder, os.path.basename(hparams.verification_file))
    wget.download(hparams.verification_file, veri_file_path)

    # Dataset prep (parsing VoxCeleb and annotation into csv files)
    prepare_voxceleb(
        data_folder_path=hparams.data_folder,
        save_folder_path=hparams.save_folder,
        verification_pairs_file=veri_file_path,
        splits=["train", "dev"],
        split_ratio=[90, 10],
        seg_dur=hparams.sentence_len,
        skip_prep=hparams.skip_prep,
    )

    if not os.path.exists(os.path.join(hparams.feat_folder)):
        os.makedirs(os.path.join(hparams.feat_folder), exist_ok=False)
    save_dir = os.path.join(hparams.feat_folder)

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    label_data = ms.dataset.CSVDataset(
        dataset_files=[hparams.train_annotation], num_parallel_workers=hparams.dataloader_options.num_workers,
    )
    label_iterator = label_data.create_dict_iterator(num_epochs=1)
    for batch in label_iterator:
        global spk_id_encoded
        spk_id = str(batch["spk_id"])
        if spk_id not in spk_id_encoded_dict.keys():
            spk_id_encoded += 1
            spk_id_encoded_dict[spk_id] = spk_id_encoded
    print("spk_id_encoded_dict len = ", len(spk_id_encoded_dict))
    train_datalist = dataio_prep()

    fea_fp = open(os.path.join(save_dir, "fea.lst"), "w")
    label_fp = open(os.path.join(save_dir, "label.lst"), "w")

    spec_aug1 = TimeDomainSpecAugment(sample_rate=16000, speeds=[100])
    spec_aug2 = TimeDomainSpecAugment(sample_rate=16000, speeds=[95, 100, 105])
    spec_aug3 = EnvCorrupt(
        openrir_folder=hparams.data_folder,
        openrir_max_noise_len=3.0,
        reverb_prob=1.0,
        noise_prob=0.0,
        noise_snr_low=0,
        noise_snr_high=15,
    )
    spec_aug4 = EnvCorrupt(
        openrir_folder=hparams.data_folder,
        openrir_max_noise_len=3.0,
        reverb_prob=0.0,
        noise_prob=1.0,
        noise_snr_low=0,
        noise_snr_high=15,
    )
    spec_aug5 = EnvCorrupt(
        openrir_folder=hparams.data_folder,
        openrir_max_noise_len=3.0,
        reverb_prob=1.0,
        noise_prob=1.0,
        noise_snr_low=0,
        noise_snr_high=15,
    )

    spec_aug = [spec_aug1, spec_aug2, spec_aug3, spec_aug4, spec_aug5]
    processlist = []
    index = 0
    manager = Manager()
    label_fp_list = manager.list()
    fea_fp_list = manager.list()
    file_lock = manager.Lock()
    for train_data in train_datalist:
        dataset_size = train_data.get_dataset_size()
        train_data = train_data.batch(batch_size=hparams.dataloader_options.batch_size)
        iterator = train_data.create_dict_iterator(num_epochs=hparams.number_of_epochs)
        print("len of train:", dataset_size)
        batch_counts = dataset_size / hparams.dataloader_options.batch_size
        process = Process(
            target=generate_npy,
            args=(iterator, spec_aug, save_dir, label_fp_list, fea_fp_list, batch_counts, index, file_lock),
        )
        process.start()
        processlist.append(process)
        index += 1

    for process in processlist:
        process.join()

    label_fp_list.sort()
    for id_save_name in label_fp_list:
        label_fp.write(id_save_name + "\n")

    fea_fp_list.sort()
    for fea_save_name in fea_fp_list:
        fea_fp.write(fea_save_name + "\n")

    dataset_path = hparams.feat_folder
    save_path = hparams.feat_folder_merge
    data_trans_dp(dataset_path, save_path)


if __name__ == "__main__":
    if hparams.need_generate_data:
        generate_train_data()
    train()
