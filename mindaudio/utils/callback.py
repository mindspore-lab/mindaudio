"""callback definition."""

import time
import os
import logging
from mindspore.train.serialization import save_checkpoint
from mindspore.train.callback._callback import Callback


class TimeMonitor(Callback):
    """
    Monitor the time in training.

    Args:
        step_size (int): How many steps are the intervals between print information each time.
            if the program get `batch_num` during training, `data_size` will be set to `batch_num`,
            otherwise `data_size` will be used. Default: None.

    Raises:
        ValueError: If data_size is not positive int.
    """
    def __init__(self, steps_size):
        super(TimeMonitor, self).__init__()
        self.step = 0
        self.steps_size = steps_size

    def step_begin(self, run_context):
        """step begin function"""
        self.step_time = time.time()

    def step_end(self, run_context):
        """step end function"""
        step_seconds = (time.time() - self.step_time) * 1000
        cb_params = run_context.original_args()
        # TrainOneStepWithLossScaleCell returns tuple while TrainOneStepCell returns loss directly
        loss = cb_params.net_outputs[0].asnumpy()
        overflow = cb_params.net_outputs[3]
        scale = cb_params.net_outputs[2]
        if hasattr(cb_params, "batch_num"):
            batch_num = cb_params.batch_num
            if isinstance(batch_num, int) and batch_num > 0:
                step_size = cb_params.batch_num

        if not isinstance(step_size, int) or step_size < 1:
            raise ValueError("data_size must be positive int.")

        step_seconds = step_seconds / 1000

        if overflow:
            logging.warning(
                "Epoch: %d, Step: %d, Step Time: %s sec, Total Loss: %s, Overflow: %s, Scale: %s.",
                int(self.step / self.steps_size), self.step % self.steps_size,
                str(step_seconds)[:5], str(loss), str(overflow), str(scale))
        else:
            logging.warning(
                "Epoch: %d, Step: %d, Step Time: %s sec, Total Loss: %s, Scale: %s.",
                int(self.step / self.steps_size), self.step % self.steps_size,
                str(step_seconds)[:5], str(loss), str(scale))
        self.step += 1

class EvalCallBack(Callback):
    """Evaluation callback"""
    def __init__(self, network, dataset, save_ckpt_step, columns_type, keep_checkpoint_max=5, directory=""):
        super(EvalCallBack, self).__init__()
        self.network = network
        self.model_dict = {}
        self.keep_checkpoint_max = keep_checkpoint_max
        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.dataset = dataset
        self.save_ckpt_step = save_ckpt_step
        self.columns_list = None
        if columns_type == "asr":
            self.columns_list = [
                "xs_pad", "ys_pad", "ys_in_pad", "ys_out_pad", "xs_masks",
                "ys_masks", "ys_sub_masks", "ys_lengths", "xs_chunk_masks"
            ]
        else:
            self.columns_list = [
                'x', 'xs_len', 'wav2vec_mask', 'mask_valid_index',
                'encoder_mask'
            ]

    def step_end(self, run_context):
        """step end and save ckpt"""
        cb_params = run_context.original_args()
        if cb_params.cur_step_num % self.save_ckpt_step != 0:
            return
        total_num = 0
        rec_num = 0
        total_acc = 0
        for data in self.dataset.create_dict_iterator(num_epochs=1):
            input_data = []
            for i in self.columns_list:
                input_data.append(data[i])
            self.network.acc_net.set_train(False)
            _, acc_tensor = self.network.acc_net(*input_data)
            one_acc = acc_tensor.asnumpy()
            total_num += len(input_data[0])
            rec_num += 1
            total_acc += one_acc
        acc = total_acc / rec_num
        with open("./callback_eval.log", "a+") as f:
            f.write("total_num {}, accuracy{:.6f}".format(total_num, acc))
            f.write('\n')

        self.network.acc_net.set_train(True)
        logging.warning("total_num {}, accuracy{:.6f}".format(total_num, acc))

        ckpt_file_name = "best_model_{}.ckpt".format(cb_params.cur_step_num)
        ckpt_file_name = os.path.join(self.directory, ckpt_file_name)
        if len(self.model_dict) < self.keep_checkpoint_max:
            save_checkpoint(self.network, ckpt_file_name)
            self.model_dict[ckpt_file_name] = acc
        else:
            min_acc = min(self.model_dict.values())
            if acc > min_acc:
                min_ckpt_name = list(self.model_dict.keys())[list(self.model_dict.values()).index(min_acc)]
                if os.path.exists(min_ckpt_name):
                    os.remove(min_ckpt_name)
                self.model_dict.pop(min_ckpt_name)
                save_checkpoint(self.network, ckpt_file_name)
                self.model_dict[ckpt_file_name] = acc
