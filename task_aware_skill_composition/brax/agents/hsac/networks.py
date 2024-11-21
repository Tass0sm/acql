"""SOAC networks."""

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
class SOACNetworks:
  hi_policy_network: networks.FeedForwardNetwork
  options: Sequence[Option]
  option_q_network: networks.FeedForwardNetwork
  parametric_option_distribution: distribution.ParametricDistribution


def make_inference_fn(soac_networks: SOACNetworks):
  """Creates params and inference function for the SAC agent."""

  def make_policy(
          params: types.PolicyParams,
          deterministic: bool = False
  ) -> types.Policy:

    hi_policy_network = soac_networks.hi_policy_network
    parametric_option_distribution = soac_networks.parametric_option_distribution
    options = soac_networks.options

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


def make_soac_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    options: Sequence[Option] = [],
) -> SOACNetworks:
  """Make SAC networks."""

  assert len(options) > 0, "Must pass at least one option"

  parametric_option_distribution = SoftmaxDistribution(event_size=len(options))
  hi_policy_network = networks.make_policy_network(
      parametric_option_distribution.param_size,
      observation_size,
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation)
  option_q_network = h_networks.make_option_q_network(
      observation_size,
      len(options),
      preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation)
  return SOACNetworks(
      hi_policy_network=hi_policy_network,
      options=options,
      option_q_network=option_q_network,
      parametric_option_distribution=parametric_option_distribution)
