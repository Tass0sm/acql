from typing import Sequence, Tuple

import spot

import jax
import jax.numpy as jnp
from brax import envs
from brax.training import distribution
from brax.training import networks
from brax.training.networks import ActivationFn, Initializer, FeedForwardNetwork, MLP

from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen

from task_aware_skill_composition.brax.training.acme import running_statistics
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



def make_option_q_fn(
    hdcq_networks: HDCQNetworks,
    env: envs.Env,
    safety_minimum: float
):

  q_func_branches = get_compiled_q_function_branches(hdcq_networks, env)

  def make_q(
      params: types.PolicyParams,
  ) -> types.Policy:

    normalizer_params, option_q_params, cost_q_params = params
    options = hdcq_networks.options
    n_options = len(options)

    def q(observation: types.Observation) -> jnp.ndarray:
      aut_state = env.aut_state_from_obs(observation)
      double_qs = jax.vmap(lambda a_s, obs: jax.lax.switch(a_s, q_func_branches, (normalizer_params, option_q_params), obs))(jnp.atleast_1d(aut_state), jnp.atleast_2d(observation))
      qs = jnp.min(double_qs, axis=-1)
      return qs

    return q

  return make_q


def make_option_inference_fn(
    hdcq_networks: HDCQNetworks,
    env: envs.Env,
    safety_minimum: float
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
      trimmed_normalizer_params = jax.tree.map(lambda x: x[..., :env.cost_observation_size] if x.ndim >= 1 else x, normalizer_params)
      double_cqs = hdcq_networks.cost_q_network.apply(trimmed_normalizer_params, cost_q_params, cost_obs)
      cqs = jnp.max(double_cqs, axis=-1)

      masked_q = jnp.where(cqs > safety_minimum, qs, -jnp.inf)

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
    safety_minimum: float
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
      double_qs = jax.vmap(lambda a_s, obs: jax.lax.switch(a_s, q_func_branches, (normalizer_params, option_q_params), obs))(jnp.atleast_1d(aut_state), jnp.atleast_2d(observation)).squeeze()
      # double_qs = jax.lax.switch(aut_state, q_func_branches, (normalizer_params, option_q_params), observation)
      qs = jnp.min(double_qs, axis=-1)

      cost_obs = env.cost_obs(observation)
      # goalless_obs = env.goalless_obs(observation)
      trimmed_normalizer_params = jax.tree.map(lambda x: x[..., :env.cost_observation_size] if x.ndim >= 1 else x, normalizer_params)
      double_cqs = hdcq_networks.cost_q_network.apply(trimmed_normalizer_params, cost_q_params, cost_obs)
      cqs = jnp.max(double_cqs, axis=-1)

      masked_q = jnp.where(cqs > safety_minimum, qs, -jnp.inf)

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


class FiLM(linen.Module):
  feature_dim: int
  condition_dim: int

  @linen.compact
  def __call__(self, features: jnp.ndarray, condition: jnp.ndarray) -> jnp.ndarray:
    """
    Args:
        features: A tensor of shape (batch_size, feature_dim).
        condition: A tensor of shape (batch_size, condition_dim).
    
    Returns:
        A tensor of shape (batch_size, feature_dim) with modulated features.
    """
    # Modulation network: outputs gamma and beta from the condition
    gamma_beta = linen.Dense(2 * self.feature_dim)(condition)  # (batch_size, 2 * feature_dim)
    gamma, beta = jnp.split(gamma_beta, 2, axis=-1)  # Split into gamma and beta

    # Apply feature-wise linear modulation
    return gamma * features + beta


class FiLMedMLP(linen.Module):
  """FiLMed MLP module."""
  condition_dim: int
  layer_sizes: Sequence[int]
  activation: ActivationFn = linen.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  bias: bool = True
  layer_norm: bool = False
  final_activation: ActivationFn = linen.tanh

  @linen.compact
  def __call__(self, data: jnp.ndarray, conditioning_data: jnp.ndarray):
    hidden = data
    for i, hidden_size in enumerate(self.layer_sizes):
      hidden = linen.Dense(
          hidden_size,
          name=f'hidden_{i}',
          kernel_init=self.kernel_init,
          use_bias=self.bias)(
              hidden)

      if i != len(self.layer_sizes) - 1:
        hidden = self.activation(hidden)
        if self.layer_norm:
          hidden = linen.LayerNorm()(hidden)

        # hidden = FiLM(
        #   feature_dim=hidden_size,
        #   condition_dim=self.condition_dim
        # )(hidden, conditioning_data)

        # hidden = self.activation(hidden)

    hidden = self.final_activation(hidden)

    return hidden



def make_option_cost_q_network(
    obs_size: int,
    aut_states: int,
    num_options: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    n_critics: int = 2) -> FeedForwardNetwork:
  """Creates a value network."""

  plain_obs_size = obs_size - aut_states

  # Q function for discrete space of options.
  class DiscreteCostQModule(linen.Module):
    """Q Module."""
    n_critics: int

    @linen.compact
    def __call__(self, obs: jnp.ndarray, aut_obs: jnp.ndarray):
      hidden = jnp.concatenate([obs], axis=-1)
      res = []
      for _ in range(self.n_critics):
        critic_option_qs = FiLMedMLP(
          condition_dim=aut_states,
          layer_sizes=list(hidden_layer_sizes) + [num_options],
          activation=activation,
          # final_activation=linen.tanh,
          # layer_norm=True,
          kernel_init=jax.nn.initializers.lecun_uniform()
        )(hidden, aut_obs)
        res.append(critic_option_qs)

      return jnp.stack(res, axis=-1)

  q_module = DiscreteCostQModule(n_critics=n_critics)

  def apply(processor_params, q_params, obs):
    trimmed_proc_params = jax.tree.map(lambda x: x[..., :plain_obs_size] if x.ndim >= 1 else x, processor_params)
    state_obs, aut_state_obs = obs[..., :plain_obs_size], obs[..., plain_obs_size:]
    norm_state_obs = preprocess_observations_fn(state_obs, trimmed_proc_params)
    return q_module.apply(q_params, norm_state_obs, aut_state_obs)

  dummy_obs = jnp.zeros((1, plain_obs_size))
  dummy_aut_obs = jnp.zeros((1, aut_states))
  return FeedForwardNetwork(
      init=lambda key: q_module.init(key, dummy_obs, dummy_aut_obs), apply=apply)


def make_hdcq_networks(
    observation_size: int,
    cost_observation_size: int,
    num_aut_states: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    preprocess_cost_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
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
  cost_q_network = make_option_cost_q_network(
      cost_observation_size,
      num_aut_states,
      len(options),
      preprocess_observations_fn=preprocess_cost_observations_fn,
      hidden_layer_sizes=(64, 64, 64, 32),
      activation=activation
  )

  return HDCQNetworks(
      # policy_network=policy_network,
      option_q_network=option_q_network,
      cost_q_network=cost_q_network,
      options=options,
      # parametric_action_distribution=parametric_action_distribution
  )
