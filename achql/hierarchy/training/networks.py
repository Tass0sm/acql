"""Network definitions."""

import dataclasses
from collections import namedtuple
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
    final_activation: ActivationFn = lambda x: x,
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
          kernel_init=jax.nn.initializers.lecun_uniform()
        )(hidden)
        critic_option_qs = final_activation(critic_option_qs)
        res.append(critic_option_qs)

      return jnp.stack(res, axis=-1)

  q_module = DiscreteQModule(n_critics=n_critics)

  def apply(processor_params, q_params, obs):
    obs = preprocess_observations_fn(obs, processor_params)
    return q_module.apply(q_params, obs)

  dummy_obs = jnp.zeros((1, obs_size))
  return FeedForwardNetwork(
      init=lambda key: q_module.init(key, dummy_obs), apply=apply)


def make_multi_headed_option_q_network(
    obs_size: int,
    num_heads: int,
    num_options: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    preprocess_cond_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    shared_hidden_layer_sizes: Sequence[int] = (256,),
    head_hidden_layer_sizes: Sequence[int] = (256,),
    activation: ActivationFn = linen.relu,
    final_activation: ActivationFn = lambda x: x,
    num_critics: int = 2) -> FeedForwardNetwork:
  """Creates a option Q-network which takes an OBS_SIZE observation, outputs Q
  value predictions for the NUM_OPTIONS options through one of NUM_HEADS heads."""

  # multi-headed Q function for discrete space of options.
  class MultiHeadedDiscreteQModule(linen.Module):
    """Multi-Headed Q Module."""
    n_heads: int

    def setup(self):
      self.shared = MLP(
        layer_sizes=list(shared_hidden_layer_sizes),
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform()
      )

      self.heads = [MLP(
        layer_sizes=list(head_hidden_layer_sizes) + [num_options],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform()
      ) for _ in range(self.n_heads)]

    def __call__(self, obs: jnp.ndarray, head: int):
      hidden = jnp.concatenate([obs], axis=-1)
      hidden = self.shared(hidden)

      def head_fn(j):
        return lambda mdl, x: mdl.heads[j](x)
      branches = [head_fn(j) for j in range(self.n_heads)]

      # run all branches on init
      if self.is_mutable_collection('params'):
        for branch in branches:
          _ = branch(self, hidden)

      def single_head_forward(hidden_h, h):
        return linen.switch(h, branches, self, hidden_h)

      critic_option_qs = single_head_forward(hidden, head)
      critic_option_qs = final_activation(critic_option_qs)

      return critic_option_qs

  DoubleMultiHeadedDiscreteQModule = linen.vmap(
    MultiHeadedDiscreteQModule,
    in_axes=None, out_axes=-1,
    variable_axes={'params': 0},
    split_rngs={'params': True},
    axis_size=num_critics
  )

  q_module = DoubleMultiHeadedDiscreteQModule(num_heads)

  def apply(processor_params, q_params, obs: jnp.ndarray, aut_state: int):
    obs = preprocess_observations_fn(obs, processor_params)
    head = preprocess_cond_fn(aut_state)
    return q_module.apply(q_params, obs, head)

  dummy_obs = jnp.zeros((1, obs_size))
  dummy_head = jnp.int32(0)
  return FeedForwardNetwork(init=lambda key: q_module.init(key, dummy_obs, dummy_head), apply=apply)
