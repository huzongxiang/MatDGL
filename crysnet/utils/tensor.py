# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 13:26:10 2021

@author: huzongxiang
"""

from typing import Union, Sequence

import numpy as np
import tensorflow as tf

Integer = Union[int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                np.uint32, np.uint64]
Float = Union[float, np.float16, np.float32, np.float64]
TensorLike = Union[Integer, Float, Sequence, np.ndarray, tf.Tensor, tf.Variable]