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

if [ $# != 8 ]; then
  echo "Usage: bash ./one_stop_standalone_asr_train.sh [EXP] [NAME] [EXP_NAME] [CONFIG_FILE] [TRAINING_WITH_EVAL] [CMVN_FILE] [TRAIN_DATA] [EVAL_DATA]"
  exit 1
fi


export DEVICE_ID=0
export GLOG_v=3

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

exp=$1
name=$2
exp_name=$3
config_path=$4
training_with_eval=$5
cmvn_file=$6
train_data=$7
eval_data=$8

rm -rf ./${exp}/${name}
mkdir -p ./${exp}/${name}
cp -r config ./${exp}/${name}
cp -r ../../flyspeech ./${exp}/${name}
cp -r ../../train.py ./${exp}/${name}

cd ./${exp}/${name} || exit

echo "start training for device $DEVICE_ID"
env > env.log
python train.py \
  --config_path $config_path \
  --training_with_eval $training_with_eval \
  --cmvn_file $cmvn_file \
  --exp_name $exp_name \
  --train_data $train_data \
  --eval_data $eval_data 2>&1 | tee log.txt &
wait
cd ../../
