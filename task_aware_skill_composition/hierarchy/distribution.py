"""Parametric softmax distribution in JAX."""

import abc
import jax
import jax.numpy as jnp

from brax.training.distribution import ParametricDistribution
from distrax import Softmax


# https://github.com/tensorflow/probability/blob/v0.23.0/tensorflow_probability/python/bijectors/identity.py
class IdentityBijector:
  """Identity Bijector."""

  def forward(self, x):
    return x

  def inverse(self, y):
    return y

  def forward_log_det_jacobian(self, x):
    return 0 # tf.constant(0., dtype=x.dtype)


class SoftmaxDistribution(ParametricDistribution):
  """Softmax distribution"""

  def __init__(self, event_size, min_std=0.001, var_scale=1):
    """Initialize the distribution.

    Args:
      event_size: the size of events (i.e. actions).
      min_std: minimum std for the gaussian.
      var_scale: adjust the gaussian's scale parameter.
    """
    super().__init__(
        param_size=event_size,
        postprocessor=IdentityBijector(),
        event_ndims=0,
        reparametrizable=True
    )
    self._temperature = 1

  def create_dist(self, parameters):
    return Softmax(logits=parameters, temperature=self._temperature)
