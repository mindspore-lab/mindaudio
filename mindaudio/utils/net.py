"""Define net-related methods."""
import mindspore
import numpy as np


def get_activation(act):
    """Return activation function."""
    # Lazy load to avoid unused import
    from flyspeech.layers.swish import Swish

    activation_funcs = {
        "tanh": mindspore.nn.Tanh,
        "relu": mindspore.nn.ReLU,
        "swish": Swish,
        "gelu": mindspore.nn.GELU,
    }

    return activation_funcs[act]()


def get_parameter_numel(net: mindspore.nn.Cell):
    num = (
        np.array([np.prod(item.shape) for item in net.get_parameters()]).sum()
        / 1024
        / 1024
    )
    return str(num)[:5] + "M"
