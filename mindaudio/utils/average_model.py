import argparse
import glob
import os

import numpy as np
from mindspore import Parameter, context
from mindspore.train.serialization import load_checkpoint, save_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="average model")
    parser.add_argument("--src_path", required=True, help="src model path for average")
    parser.add_argument("--num", default=5, type=int, help="nums for averaged model")
    args = parser.parse_args()

    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=7)

    path_list = glob.glob("{}/[!AVG]*.ckpt".format(args.src_path))
    path_list = sorted(path_list, key=os.path.getmtime)
    path_list = path_list[-args.num :]
    print(path_list)

    # read all params
    model_list = []
    for path in path_list:
        print("Processing {}".format(path))
        param_dict = load_checkpoint(path)
        cur_model = []
        for k in param_dict.keys():
            cur_param = {}
            if not k.startswith("moment"):
                cur_param["name"] = k
                cur_param["data"] = param_dict[k].data.asnumpy()
                cur_model.append(cur_param)
        model_list.append(cur_model)

    # average all params
    avg_model = []
    num_param = len(model_list[0])
    for i in range(num_param):
        avg_param = {}
        avg_param["name"] = model_list[0][i]["name"]
        avg_param["data"] = model_list[0][i]["data"]
        for j in range(1, args.num):
            avg_param["data"] = avg_param["data"] + model_list[j][i]["data"]
        avg_param["data"] = np.true_divide(avg_param["data"], args.num)
        avg_param["data"] = Parameter(avg_param["data"], name=avg_param["name"])

        avg_model.append(avg_param)

    dst_model = os.path.join(args.src_path, "avg_" + str(args.num) + ".ckpt")

    print("Saving to {}".format(dst_model))
    save_checkpoint(avg_model, dst_model)
