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

if [ $# != 10 ]; then
  echo "Usage: bash ./one_stop_asr_predict.sh [EXP] [NAME] [EXP_NAME] [CONDITIONS] [CMVN_FILE] [TEST_DATA] [DICT] [DECODE_CKPT] [DECODE_MODE] [DEVICE_ID]"
  exit 1
fi

# env setting
export GLOG_v=3
export DEVICE_ID=${10}

exp=$1
name=$2
exp_name=$3
config_path=$4
cmvn_file=$5
test_data=$6
dict=$7
decode_ckpt=$8
decode_mode=$9
cd ./${exp}/${name} || exit
echo "start decoding using $decode_mode on device $DEVICE_ID"
python predict.py \
  --exp_name $exp_name \
  --config_path $config_path \
  --cmvn_file $cmvn_file \
  --test_data $test_data \
  --dict $dict \
  --decode_ckpt $decode_ckpt \
  --decode_mode $decode_mode 2>&1 | tee decode${10}.log &
wait
cd ../../
