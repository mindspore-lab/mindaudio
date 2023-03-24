import mindspore
import numpy as np
from mindspore import Tensor, nn, ops


def pad_right_to(array, target_shape, mode="CONSTANT"):
    """
    This function takes a numpy array of arbitrary shape and pads it to target
    shape by appending values on the right.

    Args
    ----------
    array : np.ndarray
        Input array whose dimension we need to pad.
    target_shape : (list, tuple)
        Target shape we want for the target array its len must be equal to array.ndim
    mode : str
        Specifies padding mode. The optional values are "CONSTANT", "REFLECT", "SYMMETRIC".
        Default: "CONSTANT".

    Returns
    -------
    array : np.ndarray
        Padded array.
    valid_values : list
        List containing proportion for each dimension of original, non-padded values.
    """
    assert len(target_shape) == array.ndim
    pads = []  # this contains the abs length of the padding for each dimension.
    valid_values = []  # this contains the relative lengths for each dimension.
    for i in range(len(target_shape)):
        assert (
            target_shape[i] >= array.shape[i]
        ), "Target shape must be >= original shape for every dim"
        pads.append((0, target_shape[i] - array.shape[i]))
        valid_values.append(array.shape[i] / target_shape[i])
    array = np.array(array, dtype=np.float32)
    array = nn.Pad(paddings=tuple(pads), mode=mode)(
        Tensor(array, dtype=mindspore.float32)
    )
    array = array.asnumpy()
    return array, valid_values


def batch_pad_right(arrays, mode="CONSTANT"):
    """Given a list of numpy arrays it batches them together by padding to the right
    on each dimension in order to get same length for all.

    Args
    ----------
    arrays : list
        List of arrays we wish to pad together.
    mode : str
        Specifies padding mode. The optional values are "CONSTANT", "REFLECT", "SYMMETRIC".
        Default: "CONSTANT".

    Returns
    -------
    arrays : np.ndarray
        Padded array.
    valid_values : list
        List containing proportion for each dimension of original, non-padded values.

    """

    if not len(arrays):
        raise IndexError("Arrays list must not be empty")

    if len(arrays) == 1:
        # if there is only one array in the batch we simply unsqueeze it.
        return ops.ExpandDims()(arrays[0], 0), np.array([1.0])

    if not (any([arrays[i].ndim == arrays[0].ndim for i in range(1, len(arrays))])):
        raise IndexError("All arrays must have same number of dimensions")

    # multichannel
    max_shape = []
    for dim in range(arrays[0].ndim):
        max_shape.append(max([x.shape[dim] for x in arrays]))

    batched = []
    valid = []
    for t in arrays:
        # for each array we apply pad_right_to
        padded, valid_percent = pad_right_to(t, max_shape, mode=mode)
        batched.append(padded)
        valid.append(valid_percent[-1])
    batched = np.stack(batched)

    return batched, np.array(valid)
