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
"""
##############converter .mindir file into .ms file for ascend310 infer #################
python mslite_converter.py
"""
import argparse
import mindspore_lite as mslite

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="converter model")
    parser.add_argument("--file_path", required=True, help="the file that needs to be converted")
    args = parser.parse_args()
    converter = mslite.Converter(fmk_type=mslite.FmkType.MINDIR, model_file=args.file_path, output_file=args.file_path)
    converter.converter()
