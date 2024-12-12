"""Network definitions."""

import dataclasses
from typing import Any, Callable, Sequence, Tuple
import warnings

from brax.training import types
from brax.training.spectral_norm import SNDense
from flax import linen
import jax
import jax.numpy as jnp

from brax.training.networks import ActivationFn, FeedForwardNetwork, MLP


def make_option_value_network(
    obs_size: int,
    num_options: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types
    .identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu) -> FeedForwardNetwork:
  """Creates a policy network."""
  value_module = MLP(
      layer_sizes=list(hidden_layer_sizes) + [1],
      activation=activation,
      kernel_init=jax.nn.initializers.lecun_uniform())

  def apply(processor_params, policy_params, obs, option):
    obs = preprocess_observations_fn(obs, processor_params)
    option_enc = jax.nn.one_hot(option, num_classes=num_options)
    inp = jnp.concat((obs, option_enc), axis=-1)
    return jnp.squeeze(value_module.apply(policy_params, inp), axis=-1)

  dummy_obs = jnp.zeros((1, obs_size+num_options))
  return FeedForwardNetwork(
      init=lambda key: value_module.init(key, dummy_obs), apply=apply)


def make_option_q_network(
    obs_size: int,
    num_options: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    activation_final: ActivationFn = None,
    n_critics: int = 2) -> FeedForwardNetwork:
  """Creates a value network."""

  # Q function for discrete space of options.
  class DiscreteQModule(linen.Module):
    """Q Module."""
    n_critics: int

    @linen.compact
    def __call__(self, obs: jnp.ndarray):
      hidden = jnp.concatenate([obs], axis=-1)
      res = []
      for _ in range(self.n_critics):
        critic_option_qs = MLP(
          layer_sizes=list(hidden_layer_sizes) + [num_options],
          activation=activation,
          activate_final=activation_final,
          kernel_init=jax.nn.initializers.lecun_uniform()
        )(hidden)
        res.append(critic_option_qs)

      return jnp.stack(res, axis=-1)

  q_module = DiscreteQModule(n_critics=n_critics)

  def apply(processor_params, q_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    return q_module.apply(q_params, obs)

  dummy_obs = jnp.zeros((1, obs_size))
  return FeedForwardNetwork(
      init=lambda key: q_module.init(key, dummy_obs), apply=apply)
