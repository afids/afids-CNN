from __future__ import annotations

import os

import numpy as np
from numpy.typing import NDArray


def gen_training_array(
    num_channels: int,
    dims: NDArray,
    patches_path: os.PathLike[str] | str,
) -> NDArray:
    """Generate a training array containing patches of raw image and AFID location."""
    bps = 4 * num_channels * np.prod(dims)
    file_size = os.path.getsize(patches_path)
    num_samples = np.floor_divide(file_size, bps)

    arr_shape_train = (int(num_samples), dims[0], dims[1], dims[2], num_channels)

    arr_train = np.memmap(patches_path, "float32", "r", shape=arr_shape_train)
    return np.swapaxes(arr_train, 1, 3)
