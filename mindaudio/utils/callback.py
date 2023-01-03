"""callback definition."""

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

# pylint: disable=C0411,C0412
import math
import logging
import numpy as np
import yaml
import mindspore
from mindspore import Parameter, load_checkpoint, nn
from mindspore.communication.management import get_group_size, get_rank
from mindspore.train.serialization import save_checkpoint
from mindspore.train.callback import RunContext
from mindspore.train.callback._callback import Callback
from mindaudio.adapter.log import get_logger
logger = get_logger()


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
        super().__init__()
        self.step = 0
        self.steps_size = steps_size
        self.step_time = None

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


class TimeMonitorTacotron2(Callback):
    """
    Time monitor for calculating cost of each epoch.

    Args:
        data_size (int): step size of an epoch.
    """

    def __init__(self, data_size):
        super().__init__()
        self.data_size = data_size
        self.epoch_time = None
        self.step_time = None

    def epoch_begin(self, run_context):
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        epoch_seconds = (time.time() - self.epoch_time)
        per_step_seconds = epoch_seconds / self.data_size
        print(f"epoch time: {epoch_seconds}, per step time: {per_step_seconds}", flush=True)

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        step_seconds = (time.time() - self.step_time)
        print(f'step time {step_seconds}', flush=True)


class EvalCallBack(Callback):
    """Evaluation callback"""
    def __init__(self, network, dataset, save_ckpt_step, columns_type, keep_checkpoint_max=5, directory=""):
        super().__init__()
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
        with open("./callback_eval.log", "a+", encoding='utf-8') as file1:
            file1.write(f"total_num {total_num}, accuracy {acc}")
            file1.write('\n')

        self.network.acc_net.set_train(True)
        logging.warning("total_num %.4f, accuracy %.4f" % (total_num, acc))

        ckpt_file_name = f"best_model_{cb_params.cur_step_num}.ckpt"
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


class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    Note:
        if per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """
    def __init__(self, dataset_size=-1):
        super().__init__()
        self._dataset_size = dataset_size

    def step_end(self, run_context):
        """
        Print loss after each step
        """
        cb_params = run_context.original_args()
        if self._dataset_size > 0:
            percent, epoch_num = math.modf(cb_params.cur_step_num / self._dataset_size)
            if percent == 0:
                percent = 1
                epoch_num -= 1
            print(f"epoch: {int(epoch_num)}, current epoch percent: {percent}, "
                  "step: {cb_params.cur_step_num}, outputs are {str(cb_params.net_outputs)}", flush=True)
        else:
            print(f"epoch: {cb_params.cur_epoch_num}, step: {cb_params.cur_step_num}, "
                  "outputs are {str(cb_params.net_outputs)}", flush=True)


class BaseCallback(Callback):
    """Base class for implementing the rank 0 runtime program and other rank
    wait functions.
    """
    _call_num = 0
    _call_lock = '/tmp/call.lock.'

    def __init__(self, only_device_0: bool = True):
        super().__init__()
        self.only_device_0 = only_device_0

        try:
            self.is_device_0 = get_rank() % min(get_group_size(), 8) == 0
        except (ValueError, RuntimeError):
            self.is_device_0 = True

        self.device_0_run(self.clean_lock_file)

    def device_0_run(self, run_func: Callable) -> None:
        """Only rank 0 will run the input function.

        Args:
            run_func (Callable): A function that implements a certain function.
                Rank 0 will run this function, the other ranks wait for the
                function to finish executing.

        Returns:
            None
        """
        eval_lock = self._call_lock + str(self._call_num)
        self._call_num += 1

        if self.is_device_0 and not os.path.exists(eval_lock):
            run_func()

            try:
                os.mknod(eval_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(eval_lock):
                break
            time.sleep(1)

    def clean_lock_file(self) -> None:
        """Delete lock file."""
        lock_dir = os.path.dirname(self._call_lock)
        lock_prefix = os.path.dirname(self._call_lock)
        for file in os.listdir(lock_dir):
            if file.startswith(lock_prefix):
                os.remove(os.path.join(lock_dir, file))

    def real_run(self, run_func: Callable) -> None:
        """running start interface"""
        if self.only_device_0:
            self.device_0_run(run_func)
        else:
            run_func()


class MemoryStartTimeCallback(Callback):
    """ MemoryStartTimeCallback
    """
    def begin(self, run_context: RunContext) -> None:
        cb_params = run_context.original_args()
        cb_params.begin_start_time = time.time()

    def epoch_begin(self, run_context: RunContext) -> None:
        cb_params = run_context.original_args()
        cb_params.epoch_begin_start_time = time.time()


class CalRunTimeCallback(BaseCallback):
    """The callback used to calculate the elapsed time per epoch and the
    elapsed time for training."""
    _call_num = 0
    _call_lock = '/tmp/run_time.lock.'

    def end(self, run_context: RunContext) -> None:
        """Called at the end of the training."""
        cb_params = run_context.original_args()

        def run():
            if 'begin_start_time' in cb_params:
                end_time = time.time()
                start_time = cb_params.begin_start_time
                run_time = end_time - start_time

                logger.info('[CalRunTimeCallback] Total Run Time: %dh %dm %ds.', int(run_time // 3600),
                            int(run_time % 3600 // 60), int(run_time % 3600 % 60))

        self.real_run(run)

    @staticmethod
    def cal_stop_time(cur_epoch_num: int, epoch_num: int, avg_spend_time: float) -> str:
        """Cal time."""
        remain_epoch = epoch_num - cur_epoch_num
        remain_time = avg_spend_time * remain_epoch

        cur_time = datetime.now().timestamp()
        stop_time = cur_time + remain_time
        stop_time = datetime.fromtimestamp(stop_time).astimezone(timezone(timedelta(hours=8)))
        return stop_time.strftime('%Y-%m-%d %H:%M:%S')

    def epoch_end(self, run_context: RunContext) -> None:
        """Called at the end of the epoch."""
        cb_params = run_context.original_args()
        cur_epoch_num = cb_params.cur_epoch_num
        epoch_num = cb_params.epoch_num

        def run():
            if 'epoch_begin_start_time' in cb_params:
                end_time = time.time()
                start_time = cb_params.epoch_begin_start_time
                run_time = end_time - start_time

                logger.info('[CalRunTimeCallback] Epoch: %d, Run Time: %dm %ds.', cur_epoch_num, int(run_time // 60),
                            int(run_time % 60))

                if cur_epoch_num > 1:
                    if 'total_run_time' not in cb_params or 'total_epoch' not in cb_params:
                        cb_params.total_run_time = run_time
                        cb_params.total_epoch = 1
                    else:
                        cb_params.total_run_time += run_time
                        cb_params.total_epoch += 1
                    avg_spend_time = cb_params.total_run_time / cb_params.total_epoch
                    stop_time = self.cal_stop_time(cur_epoch_num, epoch_num, avg_spend_time)

                    logger.info('[CalRunTimeCallback] Training will end in approximately %s.', stop_time)

        self.real_run(run)


class EvalCallback(BaseCallback):
    """Validation of model accuracy.

    Save the checkpoint file. Average the model if necessary.
    """
    _call_num = 0
    _call_lock = '/tmp/eval_sync.lock.'

    def __init__(self,
                 network: nn.Cell,
                 dataset: mindspore.dataset,
                 column_list: List,
                 run_interval: int,
                 save_ckpt_path: str,
                 save_ckpt_network: Optional[nn.Cell] = None,
                 ckpt_prefix: str = 'Flyspeech',
                 eval_log_interval: int = 10,
                 average_model_flag: bool = True,
                 num_best_ckpt: int = 30,
                 only_device_0: bool = True) -> None:
        super().__init__(only_device_0)

        self.network = network
        self.dataset = dataset
        self.column_list = column_list
        self.run_interval = run_interval
        self.save_ckpt_path = save_ckpt_path
        self.save_ckpt_network = network if save_ckpt_network is None else save_ckpt_network
        self.ckpt_prefix = ckpt_prefix
        self.num_best_ckpt = num_best_ckpt
        self.eval_log_interval = eval_log_interval
        self.average_model_flag = average_model_flag

        if not os.path.exists(save_ckpt_path):
            os.makedirs(save_ckpt_path)
        self.total_eval_time = 0.0
        self.loss_ckpt_record = []

    def eval(self) -> Tuple[float, float]:
        """Validation of model accuracy."""
        total_loss = 0.0
        total_utts = 0
        total_step = self.dataset.get_dataset_size()

        self.network.set_train(False)
        start_time = time.time()
        for i, data in enumerate(self.dataset.create_dict_iterator(num_epochs=1)):
            input_data = [data[column_name] for column_name in self.column_list]
            loss = self.network(*input_data).asnumpy()
            total_loss += loss * input_data[0].shape[0]
            total_utts += input_data[0].shape[0]

            if i % self.eval_log_interval == 0:

                def run():
                    logger.info('[EvalCallback] Step: %d/%d, Eval Loss: %.4f.', i, total_step, loss)  # pylint: disable=W0640

                self.real_run(run)
        end_time = time.time()
        self.network.set_train(True)

        avg_loss = total_loss / total_utts
        eval_time = end_time - start_time
        self.total_eval_time += eval_time

        return avg_loss, eval_time

    def save_ckpt(self, prefix: str, ckpt_infos: Dict) -> str:
        """Save the checkpoint file."""
        ckpt_name = f'{prefix}.ckpt'
        ckpt_path = os.path.join(self.save_ckpt_path, ckpt_name)
        save_checkpoint(self.save_ckpt_network, ckpt_path)

        if ckpt_infos:
            info_file_name = f'{prefix}.yaml'
            info_path = os.path.join(self.save_ckpt_path, info_file_name)
            with open(info_path, 'w', encoding='utf-8') as f_out:
                f_out.write(yaml.dump(ckpt_infos))

        logger.info('[EvalCallback] Successfully save %s to %s.', ckpt_name, self.save_ckpt_path)

        return ckpt_path

    def begin(self, run_context: RunContext) -> None:
        """Called at the start of training."""
        cb_params = run_context.original_args()
        _ = cb_params.network

        def begin_run() -> None:
            ckpt_prefix = f'{self.ckpt_prefix}_init'
            self.save_ckpt(ckpt_prefix, {})

        self.real_run(begin_run)

    def step_end(self, run_context: RunContext) -> None:
        """Called at the end of each step."""
        cb_params = run_context.original_args()
        cur_step_num = cb_params.cur_step_num
        cur_epoch_num = cb_params.cur_epoch_num
        epoch_num = cb_params.epoch_num

        if cur_step_num % self.run_interval != 0:
            return

        avg_loss, eval_time = self.eval()
        avg_loss = float(avg_loss)

        def epoch_end_run():
            logger.info('[EvalCallback] Epoch %d/%d, Average Eval Loss: %.4f, Eval Spend Time: %dm %ds.', cur_epoch_num,
                        epoch_num, avg_loss, int(eval_time // 60), int(eval_time % 60))

            ckpt_prefix = f'{self.ckpt_prefix}_{cur_epoch_num}_{cur_step_num}'
            ckpt_infos = {'loss': avg_loss, 'time': eval_time}
            ckpt_path = self.save_ckpt(ckpt_prefix, ckpt_infos)

            self.loss_ckpt_record.append({'loss': avg_loss, 'ckpt_path': ckpt_path})

        self.real_run(epoch_end_run)

    def average_model(self) -> None:
        """Average model."""
        if not self.average_model_flag:
            return

        self.loss_ckpt_record = sorted(self.loss_ckpt_record, key=lambda i: i['loss'])
        model_params = {}
        for loss_ckpt in self.loss_ckpt_record[:self.num_best_ckpt]:
            ckpt_path = loss_ckpt['ckpt_path']
            param_dict = load_checkpoint(ckpt_path)
            for param_key, _ in param_dict.items():
                if not param_key.startswith('moment'):
                    if param_key not in model_params:
                        model_params[param_key] = []
                    model_params[param_key].append(param_dict[param_key].data.asnumpy())

        # average params
        avg_model = []
        for key, val in model_params.items():
            avg_param = {}
            avg_param['name'] = key
            avg_param['data'] = Parameter(np.mean(np.array(val), axis=0), name=key)
            avg_model.append(avg_param)

        avg_ckpt_name = f'{self.ckpt_prefix}_avg_{self.num_best_ckpt}.ckpt'
        avg_ckpt_path = os.path.join(self.save_ckpt_path, avg_ckpt_name)
        save_checkpoint(avg_model, avg_ckpt_path)

        logger.info('[EvalCallback] Successfully save %s to %s.', avg_ckpt_name, self.save_ckpt_path)

    def end(self, run_context) -> None:  # pylint: disable=W0613
        """Called at the end of training."""

        def run():
            logger.info('[EvalCallback] [After training] Total Eval Time: %dh %dm %ds.',
                        int(self.total_eval_time // 3600), int(self.total_eval_time % 3600 // 60),
                        int(self.total_eval_time % 3600 % 60))

        self.device_0_run(self.average_model)
        self.real_run(run)
