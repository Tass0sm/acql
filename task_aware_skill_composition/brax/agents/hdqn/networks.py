from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from brax.training import distribution
from brax.training import networks

from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen

from task_aware_skill_composition.hierarchy.training import networks as h_networks
from task_aware_skill_composition.hierarchy.state import OptionState
from task_aware_skill_composition.hierarchy.option import Option


@flax.struct.dataclass
class HDQNetworks:
  option_q_network: networks.FeedForwardNetwork
  options: Sequence[Option]
  # parametric_option_distribution: distribution.ParametricDistribution


def make_q_fn(hdq_networks: HDQNetworks):

  def make_q(
      params: types.PolicyParams,
  ) -> types.Policy:

    def q_fn(observation: types.Observation,) -> jnp.ndarray:
      qs = hdq_networks.option_q_network.apply(*params, observation)
      min_q = jnp.min(qs, axis=-1)
      return min_q

    return q_fn

  return make_q


def make_option_inference_fn(hdq_networks: HDQNetworks):

  def make_policy(
      params: types.PolicyParams,
      deterministic: bool = False,
  ) -> types.Policy:

    options = hdq_networks.options
    n_options = len(options)

    def random_option_policy(observation: types.Observation,
                             option_key: PRNGKey) -> Tuple[types.Action, types.Extra]:
      option = jax.random.randint(option_key, (observation.shape[0],), 0, n_options)
      return option, {}

    def greedy_option_policy(observation: types.Observation,
                             option_key: PRNGKey) -> Tuple[types.Action, types.Extra]:
      qs = hdq_networks.option_q_network.apply(*params, observation)
      min_q = jnp.min(qs, axis=-1)
      option = min_q.argmax(axis=-1)
      return option, {}

    epsilon = jnp.float32(0.1)

    def eps_greedy_option_policy(observation: types.Observation,
                                 key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
      key_sample, key_coin = jax.random.split(key_sample)
      coin_flip = jax.random.bernoulli(key_coin, 1 - epsilon)
      return jax.lax.cond(coin_flip, greedy_option_policy, random_option_policy, observation, key_sample)

    if deterministic:
      return greedy_option_policy
    else:
      return eps_greedy_option_policy

  return make_policy


def make_inference_fn(hdq_networks: HDQNetworks):

  def make_policy(
      params: types.PolicyParams,
      deterministic: bool = False
  ) -> types.Policy:

    options = hdq_networks.options
    n_options = len(options)

    def random_option_policy(observation: types.Observation,
                             option_key: PRNGKey) -> Tuple[types.Action, types.Extra]:
      option = jax.random.randint(option_key, (), 0, n_options)
      return option

    def greedy_option_policy(observation: types.Observation,
                             option_key: PRNGKey) -> Tuple[types.Action, types.Extra]:
      qs = hdq_networks.option_q_network.apply(*params, observation)
      min_q = jnp.min(qs, axis=-1)
      option = min_q.argmax(axis=-1)
      return option

    epsilon = jnp.float32(0.1)

    def eps_greedy_option_policy(observation: types.Observation,
                                 key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
      key_sample, key_coin = jax.random.split(key_sample)
      coin_flip = jax.random.bernoulli(key_coin, 1 - epsilon)
      return jax.lax.cond(coin_flip, greedy_option_policy, random_option_policy, observation, key_sample)

    def policy(
        observations: types.Observation,
        option_state: OptionState,
        key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:

      option_key, inference_key = jax.random.split(key_sample, 2)

      def get_new_option(s_t, new_option_key):
        if deterministic:
          option_idx = greedy_option_policy(s_t, new_option_key)
        else:
          option_idx = eps_greedy_option_policy(s_t, new_option_key)
        return option_idx

      # calculate o_t from s_t, b_t, o_t-1
      def get_option(s_t, b_t, o_t_minus_1, option_key):
        return jax.lax.cond((b_t == 1),
                            lambda: get_new_option(s_t, option_key),
                            lambda: o_t_minus_1)

      option_keys = jax.random.split(option_key, observations.shape[0])
      option = jax.vmap(get_option)(observations, option_state.option_beta, option_state.option, option_keys)

      def low_level_inference(s_t, o_t, inf_key):
        # ignore info for now
        actions, _ = jax.lax.switch(o_t, [o.inference for o in options], s_t, inf_key)
        return actions, {}

      inf_keys = jax.random.split(inference_key, observations.shape[0])
      actions, info = jax.vmap(low_level_inference)(observations, option, inf_keys)
      info.update(option=option)

      return actions, info

    return policy

  return make_policy


def make_hdq_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    options: Sequence[Option] = [],
) -> HDQNetworks:

  assert len(options) > 0, "Must pass at least one option"

  # parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
  # policy_network = networks.make_policy_network(
  #     parametric_action_distribution.param_size,
  #     observation_size,
  #     preprocess_observations_fn=preprocess_observations_fn,
  #     hidden_layer_sizes=hidden_layer_sizes,
  #     activation=activation)
  option_q_network = h_networks.make_option_q_network(
      observation_size,
      len(options),
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation
  )

  return HDQNetworks(
      # policy_network=policy_network,
      option_q_network=option_q_network,
      options=options,
      # parametric_action_distribution=parametric_action_distribution
  )
