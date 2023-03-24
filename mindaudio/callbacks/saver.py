import os

import mindspore as ms
from mindspore.train.callback import Callback


class SaveCallBack(Callback):
    def __init__(
        self,
        model,
        save_step,
        save_dir,
        global_step=None,
        optimiser=None,
        checkpoint_path=None,
        model_save_name="model",
        optimiser_save_name="optimiser",
    ):
        super().__init__()
        self.save_step = save_step
        self.checkpoint_path = checkpoint_path
        self.model = model
        self.optimiser = optimiser
        self.save_dir = save_dir
        self.global_step = global_step
        self.model_save_name = model_save_name
        self.optimiser_save_name = optimiser_save_name
        os.makedirs(save_dir, exist_ok=True)

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num + self.global_step
        if cur_step % self.save_step != 0:
            return
        for module, name in zip(
            [self.model, self.optimiser],
            [self.model_save_name, self.optimiser_save_name],
        ):
            name = os.path.join(self.save_dir, name)
            ms.save_checkpoint(
                module, name + "_%d.ckpt" % cur_step, append_dict={"cur_step": cur_step}
            )
