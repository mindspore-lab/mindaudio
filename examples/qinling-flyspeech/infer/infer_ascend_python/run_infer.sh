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

echo "example for ctc_greedy_search:"
echo "bash run_infer.sh ctc_greedy_search ctc_greedy_search.mindir.ms wav/test/ asr_conformer.yaml 0"

echo "example for ctc_prefix_beam_search:"
echo "bash run_infer.sh ctc_prefix_beam_search ctc_prefix_beam_search.mindir.ms wav/test/ asr_conformer.yaml 0"

echo "example for attention:"
echo "bash run_infer.sh attention attention_encoder.mindir.ms+attention_predict.mindir.ms wav/test/ asr_conformer.yaml 0"

echo "example for attention_rescoring:"
echo "bash run_infer.sh attention_rescoring attention_rescoring_ctc.mindir.ms+attention_rescoring_rescore.mindir.ms wav/test/ asr_conformer.yaml 0"

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "Usage: sh run_infer.sh [DECODE_MODE] [MINDIR_PATH] [DATA_PATH] [CONFIG_FILE] [DEVICE_ID]
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

decode_mode=$1
model=$(get_real_path $2)
data_path=$(get_real_path $3)
config_yaml_file=$(get_real_path $4)

if [ $# == 5 ]; then
    device_id=$5
elif [ $# == 4 ]; then
    if [ -z $device_id ]; then
        device_id=0
    else
        device_id=$device_id
    fi
fi

filename=$(basename ${config_yaml_file} .yaml)
export PYTHONPATH=$(pwd)/../..:$PYTHONPATH
echo "decode_mode = ${decode_mode}"
echo $model
echo $data_path
echo $config_yaml_file
echo $device_id

if [ x$decode_mode == x"ctc_greedy_search" ]; then
  echo "Using decoding strategy: ctc_greedy_search"
  python infer_ascend.py --decode_mode $decode_mode --infer_model_path_1 $model --infer_data_path $data_path \
                            --config_name $filename &> ${decode_mode}_infer.log

elif [ x$decode_mode == x"ctc_prefix_beam_search" ]; then
  echo "Using decoding strategy: ctc_prefix_beam_search"
  python infer_ascend.py --decode_mode $decode_mode --infer_model_path_1 $model --infer_data_path $data_path \
                            --config_name $filename &> ${decode_mode}_infer.log

elif [ x$decode_mode == x"attention" ]; then
  echo "Using decoding strategy: attention"
  encoder_model=$(get_real_path "$(basename ${model%+*})")
  predict_model=$(get_real_path "$(basename ${model#*+})")
  python infer_ascend.py --decode_mode $decode_mode --infer_model_path_1 $encoder_model \
                            --infer_model_path_2 $predict_model \
                            --infer_data_path $data_path --config_name $filename &> ${decode_mode}_infer.log

elif [ x$decode_mode == x"attention_rescoring" ]; then
  echo "Using decoding strategy: attention_rescoring"
  ctc_model=$(get_real_path "$(basename ${model%+*})")
  rescore_model=$(get_real_path "$(basename ${model#*+})")
  python infer_ascend.py --decode_mode $decode_mode --infer_model_path_1 $ctc_model \
                            --infer_model_path_2 $rescore_model \
                            --infer_data_path $data_path --config_name $filename &> ${decode_mode}_infer.log
else
  echo "only support mode ['ctc_greedy_search','ctc_prefix_beam_search','attention','attention_rescoring']"
  exit 1
fi
