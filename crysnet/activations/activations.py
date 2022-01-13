# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 11:52:10 2022

@author: huzongxiang
"""


import tensorflow as tf


def swish(x):
    """
    Parameters
    ----------
    x : tf.Tensor
        DESCRIPTION.

    Returns
    -------
    tf.Tensor
        DESCRIPTION.

    """
    return x*tf.sigmoid(x)


def shifted_softplus(x):
    """
    Parameters
    ----------
    x : tf.Tensor
        DESCRIPTION.

    Returns
    -------
    tf.Tensor
        DESCRIPTION.

    """
    
    return tf.nn.softplus(x) - tf.math.log(2.0)