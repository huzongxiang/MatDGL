# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 13:26:59 2021

@author: huzongxiang
source code from tensorflow_graphics
"""

import numpy as np
import tensorflow as tf
from typing import Tuple
from .tensor import TensorLike


def _double_factorial_loop_body(n, result, two):
  result = tf.where(tf.greater_equal(n, two), result * n, result)
  return n - two, result, two


def _double_factorial_loop_condition(n, result, two):
  return tf.cast(tf.math.count_nonzero(tf.greater_equal(n, two)), tf.bool)


def double_factorial(n: TensorLike) -> TensorLike:
  n = tf.convert_to_tensor(value=n)

  two = tf.ones_like(n) * 2
  result = tf.ones_like(n)
  _, result, _ = tf.while_loop(
      cond=_double_factorial_loop_condition,
      body=_double_factorial_loop_body,
      loop_vars=[n, result, two])
  return result


def factorial(n: TensorLike) -> TensorLike:
  n = tf.convert_to_tensor(value=n)

  return tf.exp(tf.math.lgamma(n + 1))


def generate_l_m_permutations(
    max_band: int,
    name: str = "spherical_harmonics_generate_l_m_permutations") -> Tuple[TensorLike, TensorLike]:
  with tf.name_scope(name):
    degree_l = []
    order_m = []
    for degree in range(0, max_band + 1):
      for order in range(-degree, degree + 1):
        degree_l.append(degree)
        order_m.append(order)
    return (tf.convert_to_tensor(value=degree_l),
            tf.convert_to_tensor(value=order_m))


def _evaluate_legendre_polynomial_pmm_eval(m, x):
  pmm = tf.pow(1.0 - tf.pow(x, 2.0), tf.cast(m, dtype=x.dtype) / 2.0)
  ones = tf.ones_like(m)
  pmm *= tf.cast(
      tf.pow(-ones, m) * double_factorial(2 * m - 1),
      dtype=pmm.dtype)
  return pmm


def _evaluate_legendre_polynomial_loop_cond(x, n, l, m, pmm, pmm1):
  return tf.cast(tf.math.count_nonzero(n <= l), tf.bool)


def _evaluate_legendre_polynomial_loop_body(x, n, l, m, pmm, pmm1):
  n_float = tf.cast(n, dtype=x.dtype)
  m_float = tf.cast(m, dtype=x.dtype)
  pmn = (x * (2.0 * n_float - 1.0) * pmm1 - (n_float + m_float - 1) * pmm) / (
      n_float - m_float)
  pmm = tf.where(tf.less_equal(n, l), pmm1, pmm)
  pmm1 = tf.where(tf.less_equal(n, l), pmn, pmm1)
  n += 1
  return x, n, l, m, pmm, pmm1


def _evaluate_legendre_polynomial_loop(x, m, l, pmm, pmm1):
  n = m + 2
  x, n, l, m, pmm, pmm1 = tf.while_loop(
      cond=_evaluate_legendre_polynomial_loop_cond,
      body=_evaluate_legendre_polynomial_loop_body,
      loop_vars=[x, n, l, m, pmm, pmm1])
  return pmm1


def _evaluate_legendre_polynomial_branch(l, m, x, pmm):
  pmm1 = x * (2.0 * tf.cast(m, dtype=x.dtype) + 1.0) * pmm
  # if, l == m + 1 return pmm1, otherwise lift to the next band.
  res = tf.where(
      tf.equal(l, m + 1), pmm1,
      _evaluate_legendre_polynomial_loop(x, m, l, pmm, pmm1))
  return res


def evaluate_legendre_polynomial(degree_l: TensorLike,
                                 order_m: TensorLike,
                                 x: TensorLike) -> TensorLike:
  degree_l = tf.convert_to_tensor(value=degree_l)
  order_m = tf.convert_to_tensor(value=order_m)
  x = tf.convert_to_tensor(value=x)

  pmm = _evaluate_legendre_polynomial_pmm_eval(order_m, x)
  return tf.where(
      tf.equal(degree_l, order_m), pmm,
      _evaluate_legendre_polynomial_branch(degree_l, order_m, x, pmm))


def _spherical_harmonics_normalization(l, m, var_type=tf.float64):
  l = tf.cast(l, dtype=var_type)
  m = tf.cast(m, dtype=var_type)
  numerator = (2.0 * l + 1.0) * factorial(l - tf.abs(m))
  denominator = 4.0 * np.pi * factorial(l + tf.abs(m))
  return tf.sqrt(numerator / denominator)


def _evaluate_spherical_harmonics_branch(degree,
                                         order,
                                         theta,
                                         phi,
                                         sign_order,
                                         var_type=tf.float64):
  sqrt_2 = tf.constant(1.41421356237, dtype=var_type)
  order_float = tf.cast(order, dtype=var_type)
  tmp = sqrt_2 * _spherical_harmonics_normalization(
      degree, order, var_type) * evaluate_legendre_polynomial(
          degree, order, tf.cos(theta))
  positive = tmp * tf.cos(order_float * phi)
  negative = tmp * tf.sin(order_float * phi)
  return tf.where(tf.greater(sign_order, 0), positive, negative)


def evaluate_spherical_harmonics(
    degree_l: TensorLike,
    order_m: TensorLike,
    theta: TensorLike,
    phi: TensorLike,
    name: str = "spherical_harmonics_evaluate_spherical_harmonics") -> TensorLike:    # pylint: disable=line-too-long

  with tf.name_scope(name):
    degree_l = tf.convert_to_tensor(value=degree_l)
    order_m = tf.convert_to_tensor(value=order_m)
    theta = tf.convert_to_tensor(value=theta)
    phi = tf.convert_to_tensor(value=phi)

    var_type = theta.dtype
    sign_m = tf.math.sign(order_m)
    order_m = tf.abs(order_m)
    zeros = tf.zeros_like(order_m)
    result_m_zero = _spherical_harmonics_normalization(
        degree_l, zeros, var_type) * evaluate_legendre_polynomial(
            degree_l, zeros, tf.cos(theta))
    result_branch = _evaluate_spherical_harmonics_branch(
        degree_l, order_m, theta, phi, sign_m, var_type)
    return tf.where(tf.equal(order_m, zeros), result_m_zero, result_branch)
