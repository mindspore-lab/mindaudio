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

if [ $# != 9 ]; then
  echo "Usage: bash ./one_stop_asr_predict.sh [RANK_TABLE_FILE] [DISTRIBUTE_DIR_NAME] [EXP_NAME] [CONFIG_FILE] [TRAINING_WITH_EVAL] [CMVN_FILE] [TRAIN_DATA] [EVAL_DATA] [DEVICE_NUM]"
  exit 1
fi

# env setting

RANK_TABLE_FILE=$1
echo $RANK_TABLE_FILE

export DEVICE_NUM=$9
export RANK_SIZE=$9
export RANK_TABLE_FILE=$RANK_TABLE_FILE

export HCCL_EXEC_TIMEOUT=3000
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GLOG_v=3

distribute_dir_name=$2
exp_name=$3
config_path=$4
training_with_eval=$5
cmvn_file=$6
train_data=$7
eval_data=$8


for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i
    rm -rf ./${distribute_dir_name}$i
    mkdir ./${distribute_dir_name}$i
    cp -r ../../flyspeech ./${distribute_dir_name}$i
    cp -r config ./${distribute_dir_name}$i
    cp -r ../../train.py ./${distribute_dir_name}$i
    cd ./${distribute_dir_name}$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    python train.py \
      --exp_name $exp_name \
      --config_path $config_path \
      --training_with_eval $training_with_eval \
      --is_distributed True \
      --cmvn_file $cmvn_file \
      --train_data $train_data \
      --eval_data $eval_data 2>&1 | tee log.txt &
    cd ..
done
wait
