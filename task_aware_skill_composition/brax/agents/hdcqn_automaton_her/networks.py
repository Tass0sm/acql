from typing import Sequence, Tuple

import spot

import jax
import jax.numpy as jnp
from brax import envs
from brax.training import distribution
from brax.training import networks

from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen

from task_aware_skill_composition.hierarchy.training import networks as h_networks
from task_aware_skill_composition.hierarchy.state import OptionState
from task_aware_skill_composition.hierarchy.option import Option
from task_aware_skill_composition.brax.agents.hdqn_automaton_her.networks import get_compiled_q_function_branches

from corallab_stl.utils import fold_spot_formula


@flax.struct.dataclass
class HDCQNetworks:
  option_q_network: networks.FeedForwardNetwork
  cost_q_network: networks.FeedForwardNetwork
  options: Sequence[Option]
  # parametric_option_distribution: distribution.ParametricDistribution




def make_option_inference_fn(
    hdcq_networks: HDCQNetworks,
    env: envs.Env,
    cost_budget: float
):

  q_func_branches = get_compiled_q_function_branches(hdcq_networks, env)

  def make_policy(
      params: types.PolicyParams,
      deterministic: bool = False,
  ) -> types.Policy:

    normalizer_params, option_q_params, cost_q_params = params
    options = hdcq_networks.options
    n_options = len(options)

    def random_option_policy(observation: types.Observation,
                             option_key: PRNGKey) -> Tuple[types.Action, types.Extra]:
      option = jax.random.randint(option_key, (observation.shape[0],), 0, n_options)
      return option, {}

    def greedy_safe_option_policy(observation: types.Observation,
                                  unused_key: PRNGKey) -> Tuple[types.Action, types.Extra]:
      aut_state = env.aut_state_from_obs(observation)
      double_qs = jax.vmap(lambda a_s, obs: jax.lax.switch(a_s, q_func_branches, (normalizer_params, option_q_params), obs))(jnp.atleast_1d(aut_state), jnp.atleast_2d(observation))
      qs = jnp.min(double_qs, axis=-1)

      cost_obs = env.cost_obs(observation)
      # goalless_obs = env.goalless_obs(observation)
      # trimmed_normalizer_params = jax.tree.map(lambda x: x[..., :env.goalless_observation_size] if x.ndim >= 1 else x, normalizer_params)
      double_cqs = hdcq_networks.cost_q_network.apply(None, cost_q_params, cost_obs)
      cqs = jnp.max(double_cqs, axis=-1)

      masked_q = jnp.where(cqs < cost_budget, qs, -jnp.inf)

      option = masked_q.argmax(axis=-1)
      return option, {}

    epsilon = jnp.float32(0.1)

    def eps_greedy_safe_option_policy(observation: types.Observation,
                                      key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
      key_sample, key_coin = jax.random.split(key_sample)
      coin_flip = jax.random.bernoulli(key_coin, 1 - epsilon)
      return jax.lax.cond(coin_flip, greedy_safe_option_policy, random_option_policy, observation, key_sample)

    if deterministic:
      return greedy_safe_option_policy
    else:
      return eps_greedy_safe_option_policy

  return make_policy


def make_inference_fn(
    hdcq_networks: HDCQNetworks,
    env: envs.Env,
    cost_budget: float
):

  q_func_branches = get_compiled_q_function_branches(hdcq_networks, env)

  def make_policy(
      params: types.PolicyParams,
      deterministic: bool = False
  ) -> types.Policy:

    normalizer_params, option_q_params, cost_q_params = params
    options = hdcq_networks.options
    n_options = len(options)

    # TODO: Random option policy could still respect Cost Q
    def random_option_policy(observation: types.Observation,
                             option_key: PRNGKey) -> Tuple[types.Action, types.Extra]:
      option = jax.random.randint(option_key, (), 0, n_options)
      return option

    def greedy_safe_option_policy(observation: types.Observation,
                                  option_key: PRNGKey) -> Tuple[types.Action, types.Extra]:
      aut_state = env.aut_state_from_obs(observation)
      # double_qs = jax.vmap(lambda a_s, obs: jax.lax.switch(a_s, q_func_branches, (normalizer_params, option_q_params), obs))(jnp.atleast_1d(aut_state), jnp.atleast_2d(observation))
      double_qs = jax.lax.switch(aut_state, q_func_branches, (normalizer_params, option_q_params), observation)
      qs = jnp.min(double_qs, axis=-1)

      cost_obs = env.cost_obs(observation)
      # goalless_obs = env.goalless_obs(observation)
      # trimmed_normalizer_params = jax.tree.map(lambda x: x[..., :env.goalless_observation_size] if x.ndim >= 1 else x, normalizer_params)
      double_cqs = hdcq_networks.cost_q_network.apply(None, cost_q_params, cost_obs)
      cqs = jnp.max(double_cqs, axis=-1)

      masked_q = jnp.where(cqs < cost_budget, qs, -jnp.inf)

      option = masked_q.argmax(axis=-1)
      return option

    epsilon = jnp.float32(0.1)

    def eps_greedy_safe_option_policy(observation: types.Observation,
                                 key_sample: PRNGKey) -> Tuple[types.Action, types.Extra]:
      key_sample, key_coin = jax.random.split(key_sample)
      coin_flip = jax.random.bernoulli(key_coin, 1 - epsilon)
      return jax.lax.cond(coin_flip, greedy_safe_option_policy, random_option_policy, observation, key_sample)

    def policy(
        observations: types.Observation,
        option_state: OptionState,
        key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:

      option_key, inference_key = jax.random.split(key_sample, 2)

      def get_new_option(s_t, new_option_key):
        if deterministic:
          option_idx = greedy_safe_option_policy(s_t, new_option_key)
        else:
          option_idx = eps_greedy_safe_option_policy(s_t, new_option_key)
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


def make_hdcq_networks(
    observation_size: int,
    cost_observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: networks.ActivationFn = linen.relu,
    options: Sequence[Option] = [],
) -> HDCQNetworks:

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
  cost_q_network = h_networks.make_option_q_network(
      cost_observation_size,
      len(options),
      preprocess_observations_fn=types.identity_observation_preprocessor,
      # preprocess_observations_fn=preprocess_observations_fn,
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation
  )

  return HDCQNetworks(
      # policy_network=policy_network,
      option_q_network=option_q_network,
      cost_q_network=cost_q_network,
      options=options,
      # parametric_action_distribution=parametric_action_distribution
  )
