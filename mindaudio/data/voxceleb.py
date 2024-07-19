"""
Data preparation, from mindaudio VoxCeleb recipe.
"""

import csv
import glob
import logging
import os
import pickle
import random
import shutil
import time

import numpy as np
from tqdm.contrib import tqdm

from .io import read
from .processing import stereo_to_mono

__all__ = ["prepare_voxceleb"]

logger = logging.getLogger(__name__)
VOX_OPT_FILE = "opt_voxceleb_prepare.pkl"
VOX_TRAIN_CSV = "train.csv"
VOX_DEV_CSV = "dev.csv"
VOX_TEST_CSV = "test.csv"
VOX_ENROL_CSV = "enrol.csv"
SAMPLERATE = 16000

VOX_DEV_WAV = "vox1_dev_wav.zip"
VOX_TEST_WAV = "vox1_test_wav.zip"
META = "meta"


def load_pkl(file):
    """
    Loads a pkl file.

    Args:
        file : str, Path to the input pkl file.

    Returns The loaded object.
    """

    # Deals with the situation where two processes are trying
    # to access the same label dictionary by creating a lock
    count = 100
    while count > 0:
        if os.path.isfile(file + ".lock"):
            time.sleep(1)
            count -= 1
        else:
            break

    try:
        open(file + ".lock", "w").close()
        with open(file, "rb") as f:
            return pickle.load(f)
    finally:
        if os.path.isfile(file + ".lock"):
            os.remove(file + ".lock")


def save_pkl(obj, file):
    """
    Save an object in pkl format.

    Args:
        obj : object, Object to save in pkl format
        file : str, Path to the output file
    """
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def prepare_voxceleb(
    data_folder_path,
    save_folder_path,
    verification_pairs_file,
    splits=("train", "dev", "test"),
    split_ratio=(90, 10),
    seg_dur=3.0,
    skip_prep=False,
    amp_th=5e-04,
    source=None,
    split_speaker=False,
    random_segment=False,
):
    """
    Prepares the csv files for the Voxceleb1 or Voxceleb2 datasets.
    Please follow the instructions in the readme.md file for
    preparing Voxceleb2.
    """

    if skip_prep:
        return
    # Create configuration for easily skipping data_preparation stage
    save_conf = {
        "data_folder": data_folder_path,
        "splits": splits,
        "split_ratio": split_ratio,
        "save_folder": save_folder_path,
        "seg_dur": seg_dur,
        "split_speaker": split_speaker,
    }

    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # Setting output files
    save_option = os.path.join(save_folder_path, VOX_OPT_FILE)
    save_csv_train = os.path.join(save_folder_path, VOX_TRAIN_CSV)
    save_csv_dev = os.path.join(save_folder_path, VOX_DEV_CSV)

    # Create the data folder contains VoxCeleb1 test data from the source
    if source is not None:
        if not os.path.exists(os.path.join(data_folder_path, "wav", "id10270")):
            shutil.unpack_archive(os.path.join(source, VOX_TEST_WAV), data_folder_path)
        if not os.path.exists(os.path.join(data_folder_path, "meta")):
            shutil.copytree(
                os.path.join(source, "meta"),
                os.path.join(data_folder_path, "meta"),
            )

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder_path, save_conf):
        return

    # Additional checks to make sure the data folder contains VoxCeleb data
    if "," in data_folder_path:
        data_folder_path = data_folder_path.replace(" ", "").split(",")
    else:
        data_folder_path = [data_folder_path]

    msg = "\tCreating csv file for the VoxCeleb Dataset.."
    logger.info(msg)

    # Split data into 90% train and 10% validation (verification split)
    wav_lst_train, wav_lst_dev = get_utt_split_lists(
        data_folder_path, split_ratio, verification_pairs_file, split_speaker
    )

    # Creating csv file for training data
    if "train" in splits:
        prepare_csv_file(seg_dur, wav_lst_train, save_csv_train, random_segment, amp_th)

    if "dev" in splits:
        prepare_csv_file(seg_dur, wav_lst_dev, save_csv_dev, random_segment, amp_th)

    # For PLDA verification
    if "test" in splits:
        prepare_csv_enrol_test(
            data_folder_path, save_folder_path, verification_pairs_file
        )

    # Saving options (useful to skip this phase when already done)
    save_pkl(save_conf, save_option)


def skip(splits, save_folder, save_conf):
    """
    Detects if the voxceleb data_preparation has been already done.
    If the preparation has been done, we can skip it.

    """
    # Checking csv files
    skip_prep = True

    split_files = {
        "train": VOX_TRAIN_CSV,
        "dev": VOX_DEV_CSV,
        "test": VOX_TEST_CSV,
        "enrol": VOX_ENROL_CSV,
    }
    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split_files[split])):
            skip_prep = False
    #  Checking saved options
    save_opt = os.path.join(save_folder, VOX_OPT_FILE)
    if skip_prep is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            skip_prep = bool(opts_old == save_conf)
        else:
            skip_prep = False

    return skip_prep


# Used for verification split
def get_utt_split_lists(
    data_folders, split_ratio, verification_pairs_file, split_speaker=False
):
    """
    Splits the audio file list into train and dev.
    This function automatically removes verification test files from the
    training and dev set (if any).
    """
    vox_train_lst = []
    vox_dev_lst = []

    print("Getting file list...")
    for data_folder in data_folders:
        test_list = [
            line.rstrip("\n").split(" ")[1] for line in open(verification_pairs_file)
        ]
        test_list = set(sorted(test_list))

        test_speakers = [snt.split("/")[0] for snt in test_list]

        path = os.path.join(data_folder, "wav", "**", "*.wav")
        if split_speaker:
            # avoid test speakers for train and dev splits
            audio_file_dict = {}
            for f in glob.glob(path, recursive=True):
                spk_id = f.split("/wav/")[1].split("/")[0]
                if spk_id not in test_speakers:
                    audio_file_dict.setdefault(spk_id, []).append(f)

            spk_id_list = list(audio_file_dict.keys())
            random.shuffle(spk_id_list)
            split = int(0.01 * split_ratio[0] * len(spk_id_list))
            for spk_id in spk_id_list[:split]:
                vox_train_lst.extend(audio_file_dict[spk_id])

            for spk_id in spk_id_list[split:]:
                vox_dev_lst.extend(audio_file_dict[spk_id])
        else:
            # avoid test speakers for train and dev splits
            audio_file_list = []
            for f in glob.glob(path, recursive=True):
                try:
                    spk_id = f.split("/wav/")[1].split("/")[0]
                except ValueError:
                    logger.info("Malformed path: %s", f)
                    continue
                if spk_id not in test_speakers:
                    audio_file_list.append(f)

            random.shuffle(audio_file_list)
            split = int(0.01 * split_ratio[0] * len(audio_file_list))
            train_snts = audio_file_list[:split]
            dev_snts = audio_file_list[split:]

            vox_train_lst.extend(train_snts)
            vox_dev_lst.extend(dev_snts)

    return vox_train_lst, vox_dev_lst


def get_chunks(seg_dur, audio_id, audio_duration):
    """
    Returns list of chunks
    """
    num_chunks = int(audio_duration / seg_dur)  # all in milliseconds

    chunk_list = [
        audio_id + "_" + str(i * seg_dur) + "_" + str(i * seg_dur + seg_dur)
        for i in range(num_chunks)
    ]

    return chunk_list


def prepare_csv_file(seg_dur, wav_lst, csv_file, random_segment=False, amp_th=0):
    """
    Creates the csv file given a list of wav files.
    """

    msg = '\t"Creating csv lists in  %s..."' % (csv_file)
    logger.info(msg)

    csv_output_header = [["ID", "duration", "wav", "start", "stop", "spk_id"]]

    # For assigning unique ID to each chunk
    each_sep = "--"
    entry = []
    # Processing all the wav files in the list
    for each_wav_file in tqdm(wav_lst, dynamic_ncols=True):
        # Getting sentence and speaker ids
        try:
            [spk_id, sess_id, utt_id] = each_wav_file.split("/")[-3:]
        except ValueError:
            logger.info("Malformed path: %s", each_wav_file)
            continue
        audio_id = each_sep.join([spk_id, sess_id, utt_id.split(".")[0]])

        # Reading the signal (to retrieve duration in seconds)
        try:
            signal, _ = read(each_wav_file)
        except ValueError:
            continue

        if len(signal.shape) > 1:
            signal = stereo_to_mono(signal)

        if random_segment:
            audio_duration = signal.shape[0] / SAMPLERATE
            start_sample_index = 0
            stop_sample_index = signal.shape[0]

            # Composition of the csv_line
            csv_each_line = [
                audio_id,
                str(audio_duration),
                each_wav_file,
                start_sample_index,
                stop_sample_index,
                spk_id,
            ]
            entry.append(csv_each_line)
        else:
            audio_duration = signal.shape[0] / SAMPLERATE

            uniq_chunks_list = get_chunks(seg_dur, audio_id, audio_duration)
            for chunk in uniq_chunks_list:
                s, e = chunk.split("_")[-2:]
                start_sample_index = int(float(s) * SAMPLERATE)
                end_sample = int(float(e) * SAMPLERATE)

                #  Avoid chunks with very small energy
                mean_sig = np.mean(np.abs(signal[start_sample_index:end_sample]))
                if mean_sig < amp_th:
                    continue

                # Composition of the csv_line
                csv_each_line = [
                    chunk,
                    str(audio_duration),
                    each_wav_file,
                    start_sample_index,
                    end_sample,
                    spk_id,
                ]
                entry.append(csv_each_line)

    csv_output_header = csv_output_header + entry

    # Writing the csv lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_output_header:
            csv_writer.writerow(line)

    # Final prints
    msg_info = "\t%s successfully created!" % (csv_file)
    logger.info(msg_info)


def prepare_csv_enrol_test(data_folders, save_folder, verification_pairs_file):
    """
    Creates the csv file for test data (useful for verification)
    """

    csv_output_head = [["ID", "duration", "wav", "start", "stop", "spk_id"]]

    for each_data_folder in data_folders:
        test_list_file = verification_pairs_file

        vox_enrol_ids, vox_test_ids = [], []

        # Get unique ids (enrol and test utterances)
        for each_line in open(test_list_file):
            e_id = each_line.split(" ")[1].rstrip().split(".")[0].strip()
            t_id = each_line.split(" ")[2].rstrip().split(".")[0].strip()
            vox_enrol_ids.append(e_id)
            vox_test_ids.append(t_id)

        vox_enrol_ids = list(np.unique(np.array(vox_enrol_ids)))
        vox_test_ids = list(np.unique(np.array(vox_test_ids)))

        # Prepare enrol csv
        logger.info("preparing enrol csv")
        enrol_csv = []
        for e_id in vox_enrol_ids:
            wav = each_data_folder + "/wav/" + e_id + ".wav"

            # Reading the signal (to retrieve duration in seconds)
            try:
                signal, _ = read(wav)
            except ValueError:
                continue
            if len(signal.shape) > 1:
                signal = stereo_to_mono(signal)

            audio_duration = signal.shape[0] / SAMPLERATE
            start_sample_index = 0
            stop_sample_index = signal.shape[0]
            [spk_id, _, _] = wav.split("/")[-3:]

            csv_line = [
                e_id,
                audio_duration,
                wav,
                start_sample_index,
                stop_sample_index,
                spk_id,
            ]

            enrol_csv.append(csv_line)

        csv_output = csv_output_head + enrol_csv
        csv_file_path = os.path.join(save_folder, VOX_ENROL_CSV)

        # Writing the csv lines
        with open(csv_file_path, mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for each_line in csv_output:
                csv_writer.writerow(each_line)

        # Prepare test csv
        logger.info("preparing test csv")
        test_csv = []
        for t_id in vox_test_ids:
            wav = each_data_folder + "/wav/" + t_id + ".wav"

            # Reading the signal (to retrieve duration in seconds)
            try:
                signal, _ = read(wav)
            except ValueError:
                continue
            if len(signal.shape) > 1:
                signal = stereo_to_mono(signal)

            audio_duration = signal.shape[0] / SAMPLERATE
            start_sample_index = 0
            stop_sample_index = signal.shape[0]
            [spk_id, _, _] = wav.split("/")[-3:]

            csv_line = [
                t_id,
                audio_duration,
                wav,
                start_sample_index,
                stop_sample_index,
                spk_id,
            ]

            test_csv.append(csv_line)

        csv_output = csv_output_head + test_csv
        csv_file_path = os.path.join(save_folder, VOX_TEST_CSV)

        # Writing the csv lines
        with open(csv_file_path, mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for each_line in csv_output:
                csv_writer.writerow(each_line)
