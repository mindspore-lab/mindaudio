#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# general configuration

get_real_path(){
if [ "${1:0:1}" == "/" ]; then
echo "$1"
else
echo "$(realpath -m $PWD/$1)"
fi
}

. ./path.sh || exit 1;

stage=-1
stop_stage=2
nj=16
# data related
data=/home/corpus/asr-data
data_url=www.openslr.org/resources/33
dict=data/dict/lang_char.txt

train_set=train
# Optional train_config
# 1. config/asr_transformer.yaml
# 2. config/asr_conformer.yaml
train_config=config/asr_conformer.yaml
training_with_eval=True
# ckpt file will be saved at $exp/${net_name}/${exp_name}/model
exp=exp
net_name=$(basename ${train_config} .yaml)
exp_name=default

# distribute train
is_distribute=False
device_num=4
rank_table_file=
# Create separate folders named ${distribute_dir_name}{device_id} for each card for distributed training.
distribute_dir_name=asr_train_parallel

# decode option
exp_predict=exp_predict
average_num=30
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"

# export option
export_ckpt_path=''

# Ascend310 infer
ctc_greedy_search_path="ctc_greedy_search.mindir"
ctc_prefix_beam_search_path="ctc_prefix_beam_search.mindir"
attention_encoder_path="attention_encoder.mindir"
attention_predict_path="attention_predict.mindir"
attention_rescoring_ctc_path="attention_rescoring_ctc.mindir"
attention_rescoring_rescore_path="attention_rescoring_rescore.mindir"

infer_data_path=/home/aishell/data_aishell/wav/test/
label_file=data/test/text


if [[ "${training_with_eval}" != "True" && "${training_with_eval}" != "False"  ]]; then
  echo "training_with_eval: expected True or False, but get ${training_with_eval}"
  exit 1;
fi

if [[ "${is_distribute}" != "True" && "${is_distribute}" != "False"  ]]; then
  echo "is_distribute: expected True or False, but get ${is_distribute}"
  exit 1;
fi

if [[ "${is_distribute}" == "True" && "${rank_table_file}" == ""  ]]; then
  echo "rank_table_file is not configure."
  exit 1;
fi

. tools/parse_options.sh || exit 1;

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data download started @ `date`"
    local/download_and_untar.sh ${data} ${data_url} data_aishell
    local/download_and_untar.sh ${data} ${data_url} resource_aishell
    echo "stage -1: Done @ `date`"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data prepare started @ `date`"
    # Task dependent. You should make the following preparation part by yourself.
    # For most corpora, you can follow the recipes of WeNet or ESPnet toolkits.
    local/aishell_data_prep.sh \
        ${data}/data_aishell/wav \
        ${data}/data_aishell/transcript

    # remove the space between the text labels for Mandarin dataset
    for x in train dev test; do
        cp data/${x}/text data/${x}/text.org
        paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
            > data/${x}/text
        rm data/${x}/text.org
    done

    echo "Start computing cmvn..."
    python tools/compute_cmvn_stats.py --num_workers 16 --train_config $train_config \
      --in_scp data/${train_set}/wav.scp \
      --out_cmvn data/$train_set/global_cmvn
    echo "stage 0: Done @ `date`"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Dict prepare started @ `date`"
    # use characters for Mandarin corpus and BPE for English corpus
    mkdir -p "$(dirname $dict)"
    echo "<blank> 0" > ${dict} # 0 will be used for "blank" in CTC
    echo "<unk> 1" >> ${dict} # <unk> must be 1
    tools/text2token.py -s 1 -n 1 data/train/text | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    num_token=$(cat $dict | wc -l)
    echo "<sos/eos> $num_token" >> $dict # <eos>
    echo "stage 1: Done @ `date`"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Format file prepare started @ `date`"
    # generate format.data file and remove too long or
    # short utterances for train and dev sets
    for x in train dev; do
        tools/format_data.sh --nj ${nj} \
            --feat-type wav --feat data/$x/wav.scp \
            data/$x ${dict} > data/$x/format.data.tmp

        tools/remove_longshortdata.py \
            --min_input_len 0.5 \
            --max_input_len 20 \
            --max_output_len 400 \
            --max_output_input_ratio 10.0 \
            --data_file data/$x/format.data.tmp \
            --output_data_file data/$x/format.data
    done

    # generate format.data file for test sets
    tools/format_data.sh --nj ${nj} \
        --feat-type wav --feat data/test/wav.scp \
        data/test ${dict} > data/test/format.data
    echo "stage 2: Done @ `date`"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 3: start training."
  cmvn_file=$(get_real_path data/${train_set}/global_cmvn)
  train_data=$(get_real_path data/${train_set}/format.data)
  eval_data=$(get_real_path data/dev/format.data)
  if [ ${is_distribute} == False ]; then
    ./one_stop_standalone_asr_train.sh ${exp} ${net_name} ${exp_name} ${train_config} \
                                                      ${training_with_eval} ${cmvn_file} ${train_data} \
                                                      ${eval_data} || exit 1
  else
    ./one_stop_distribute_asr_train.sh ${rank_table_file} ${distribute_dir_name} ${exp_name} \
                                                      ${train_config} ${training_with_eval} ${cmvn_file} \
                                                      ${train_data} ${eval_data} ${device_num} || exit 1
  fi
  echo "stage 3: Done @ `date`"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "stage 4: Test model."
  rm -rf ./${exp_predict}/${net_name}
  mkdir -p ./${exp_predict}/${net_name}
  cp  ../../predict.py ./${exp_predict}/${net_name}
  cp -r config ./${exp_predict}/${net_name}
  cp -r ../../flyspeech ./${exp_predict}/${net_name}
  cmvn_file=$(get_real_path data/${train_set}/global_cmvn)
  test_data=$(get_real_path data/test/format.data)
  dict=$(get_real_path data/dict/lang_char.txt)
  decode_ckpt=${exp}/${net_name}/${exp_name}/model/Flyspeech_avg_${average_num}.ckpt
  decode_ckpt_dir=${exp}/${net_name}/${exp_name}/model
  if [ ${is_distribute} == True ]; then
    decode_ckpt=${distribute_dir_name}0/${exp_name}/model/Flyspeech_avg_${average_num}.ckpt
    decode_ckpt_dir=${distribute_dir_name}0/${exp_name}/model
  fi
  if [ ${training_with_eval} == False ]; then
    python tools/average_model.py --src_path ${decode_ckpt_dir} --num ${average_num} || exit 1
    decode_ckpt=${decode_ckpt_dir}/avg_${average_num}.ckpt
  fi
  decode_ckpt=$(get_real_path ${decode_ckpt})
  export_ckpt_path=${decode_ckpt}
  device_id=0
  for mode in ${decode_modes}; do
  {
    ./one_stop_asr_predict.sh ${exp_predict} ${net_name} ${exp_name} ${train_config} ${cmvn_file} \
                                            ${test_data} ${dict} ${decode_ckpt} ${mode} ${device_id} || exit 1
    result_dir=${exp_predict}/${net_name}/${exp_name}/test_${mode}
    python tools/compute-cer.py --char=1 --v=1 data/test/text ${result_dir}/result.txt > ${result_dir}/cer.txt || exit 1
  } &
  ((device_id=device_id+1))
  done
  wait
  echo "stage 4: Done @ `date`"
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "stage 5: Export model for Ascend310 infer."
  cmvn_file=$(get_real_path data/${train_set}/global_cmvn)
  dict=$(get_real_path data/dict/lang_char.txt)
  for mode in ${decode_modes}; do
  {
    python tools/export.py --decode_mode ${mode} --decode_ckpt ${export_ckpt_path} --config_path ${train_config} \
                           --cmvn_file ${cmvn_file} --dict ${dict}
  }
  done
  echo "stage 5: Done @ `date`"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "stage 6: Ascend310 infer."

  # switch to the project directory of qinling-flyspeech
  cd ../../
  export FLYSPEECH_DIR=$PWD
  export PYTHONPATH=FLYSPEECH_DIR:$PYTHONPATH
  config_path=examples/aishell/${train_config}
  for decode_mode in ${decode_modes}; do
  {
    if [ x$decode_mode == x"ctc_greedy_search" ]; then
      echo "Using decoding strategy: ctc_greedy_search"
      ctc_greedy_search_path=$(get_real_path $ctc_greedy_search_path)
      # converter .mindir file into .ms file for Ascend310 infer
      python tools/mslite_converter.py --file_path $ctc_greedy_search_path
      # Ascend310 infer
      python infer/infer_ascend_python/infer_ascend.py \
        --decode_mode $decode_mode \
        --infer_model_path_1 "$(dirname $ctc_greedy_search_path)"/ctc_greedy_search.mindir.ms \
        --infer_data_path $infer_data_path \
        --config_path $config_path
      # calculation CER
      python tools/compute-cer.py --char=1 --v=1 ${label_file} ${exp_name}/${decode_mode}/result.txt > ${exp_name}/${decode_mode}/cer.txt || exit 1

    elif [ x$decode_mode == x"ctc_prefix_beam_search" ]; then
      echo "Using decoding strategy: ctc_prefix_beam_search"
      ctc_prefix_beam_search_path=$(get_real_path $ctc_prefix_beam_search_path)
      # converter .mindir file into .ms file for Ascend310 infer
      python tools/mslite_converter.py --file_path $ctc_prefix_beam_search_path
      # Ascend310 infer
      python infer/infer_ascend_python/infer_ascend.py \
        --decode_mode $decode_mode \
        --infer_model_path_1 "$(dirname $ctc_prefix_beam_search_path)"/ctc_prefix_beam_search.mindir.ms \
        --infer_data_path $infer_data_path \
        --config_path $config_path
      # calculation CER
      python tools/compute-cer.py --char=1 --v=1 ${label_file} ${exp_name}/${decode_mode}/result.txt > ${exp_name}/${decode_mode}/cer.txt || exit 1

    elif [ x$decode_mode == x"attention" ]; then
      echo "Using decoding strategy: attention"
      encoder_model=$(get_real_path $attention_encoder_path)
      predict_model=$(get_real_path $attention_predict_path)
      # converter .mindir file into .ms file for Ascend310 infer
      python tools/mslite_converter.py --file_path $encoder_model
      python tools/mslite_converter.py --file_path $predict_model
      # Ascend310 infer
      python infer/infer_ascend_python/infer_ascend.py \
        --decode_mode $decode_mode \
        --infer_model_path_1 "$(dirname $encoder_model)"/attention_encoder.mindir.ms \
        --infer_model_path_2 "$(dirname $predict_model)"/attention_predict.mindir.ms \
        --infer_data_path $infer_data_path --config_path $config_path

      # calculation CER
      python tools/compute-cer.py --char=1 --v=1 ${label_file} ${exp_name}/${decode_mode}/result.txt > ${exp_name}/${decode_mode}/cer.txt || exit 1

    elif [ x$decode_mode == x"attention_rescoring" ]; then
      echo "Using decoding strategy: attention_rescoring"
      ctc_model=$(get_real_path $attention_rescoring_ctc_path)
      rescore_model=$(get_real_path $attention_rescoring_rescore_path)
      # converter .mindir file into .ms file for Ascend310 infer
      python tools/mslite_converter.py --file_path $ctc_model
      python tools/mslite_converter.py --file_path $rescore_model
      # Ascend310 infer
      python infer/infer_ascend_python/infer_ascend.py \
        --decode_mode $decode_mode \
        --infer_model_path_1 "$(dirname $ctc_model)"/attention_rescoring_ctc.mindir.ms \
        --infer_model_path_2 "$(dirname $rescore_model)"/attention_rescoring_rescore.mindir.ms \
        --infer_data_path $infer_data_path --config_path $config_path

      # calculation CER
      python tools/compute-cer.py --char=1 --v=1 ${label_file} ${exp_name}/${decode_mode}/result.txt > ${exp_name}/${decode_mode}/cer.txt || exit 1
    else
      echo "only support mode ['ctc_greedy_search','ctc_prefix_beam_search','attention', 'attention_rescoring']"
  exit 1
fi
  }
  done
fi
