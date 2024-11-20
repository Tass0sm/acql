"""Hierarchical Actor"""

from typing import Sequence, Tuple

import jax
from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen

from task_aware_skill_composition.hierarchy.distribution import SoftmaxDistribution
from task_aware_skill_composition.hierarchy.training import networks as h_networks
from task_aware_skill_composition.hierarchy.state import OptionState
from task_aware_skill_composition.hierarchy.option import Option


@flax.struct.dataclass
class OptionCriticNetworks:
  hi_policy_network: networks.FeedForwardNetwork
  options: Sequence[Option]
  option_value_network: networks.FeedForwardNetwork
  parametric_option_distribution: distribution.ParametricDistribution


def make_inference_fn(option_critic_networks: OptionCriticNetworks):
  """Creates params and inference function for the PPO agent."""

  def make_policy(params: types.PolicyParams,
                  deterministic: bool = False) -> types.Policy:
    hi_policy_network = option_critic_networks.hi_policy_network
    parametric_option_distribution = option_critic_networks.parametric_option_distribution
    options = option_critic_networks.options

    def policy(
        observations: types.Observation,
        option_state: OptionState,
        key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:

      option_key, inference_key = jax.random.split(key_sample, 2)

      def get_new_option(s_t, new_option_key):
        logits = hi_policy_network.apply(*params, s_t)

        if deterministic:
          option_idx = parametric_option_distribution.mode(logits)
        else:
          option_idx = parametric_option_distribution.sample_no_postprocessing(logits, new_option_key)

        log_prob = parametric_option_distribution.log_prob(logits, option_idx)

        return option_idx, log_prob

      # calculate o_t from s_t, b_t, o_t-1
      def get_option(s_t, b_t, o_t_minus_1, option_key):
        log_prob_of_1 = 0.0
        return jax.lax.cond((b_t == 1),
                            lambda: get_new_option(s_t, option_key),
                            lambda: (o_t_minus_1, log_prob_of_1))

      option_keys = jax.random.split(option_key, observations.shape[0])
      o_t, log_prob = jax.vmap(get_option)(observations, option_state.option_beta, option_state.option, option_keys)

      # calculate a_t from s_t, o_t

      def low_level_inference(s_t, o_t, inf_key):
        actions, info = jax.lax.switch(o_t, [o.inference for o in options], s_t, inf_key)
        log_prob_given_option = info["log_prob"]

        return actions, {
          'action_log_prob_given_option': log_prob_given_option,
        }

      inf_keys = jax.random.split(inference_key, observations.shape[0])
      actions, info = jax.vmap(low_level_inference)(observations, o_t, inf_keys)
      info.update(option=o_t, option_log_prob=log_prob)

      return actions, info

    return policy

  return make_policy


def make_option_critic_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: networks.ActivationFn = linen.swish,
    options: Sequence[Option] = [],
) -> OptionCriticNetworks:
  """Make Option Critic networks with preprocessor."""
  parametric_option_distribution = SoftmaxDistribution(event_size=len(options))
  hi_policy_network = networks.make_policy_network(
      parametric_option_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation)
  option_value_network = h_networks.make_option_value_network(
      observation_size,
      len(options),
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=value_hidden_layer_sizes,
      activation=activation)

  return OptionCriticNetworks(
      hi_policy_network=hi_policy_network,
      options=options,
      option_value_network=option_value_network,
      parametric_option_distribution=parametric_option_distribution)
