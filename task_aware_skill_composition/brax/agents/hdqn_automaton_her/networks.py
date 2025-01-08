import functools
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

from corallab_stl.utils import fold_tree, fold_spot_formula


@flax.struct.dataclass
class HDQNetworks:
    option_q_network: networks.FeedForwardNetwork
    options: Sequence[Option]
    # parametric_option_distribution: distribution.ParametricDistribution


def get_compiled_q_function_branches(
    hdq_networks: HDQNetworks,
    env: envs.Env,
):
    out_conditions = env.out_conditions
    bdd_dict = env.automaton.bdd_dict
    state_and_ap_to_goal_idx_dict = env.state_and_ap_to_goal_idx_dict

    preds = []

    def to_q_func_helper(root, children, aut_state=0):
        nonlocal state_and_ap_to_goal_idx_dict

        children = list(children)

        k = root.kind()

        if k == spot.op_ff:
            return lambda params, obs: -987
        elif k == spot.op_tt:
            return lambda params, obs: 987
        elif k == spot.op_ap:
            ap_id = int(root.ap_name()[3:])
            goal_idx = state_and_ap_to_goal_idx_dict[(aut_state, ap_id)]

            def get_ap_q(params, obs):
                nn_input = jnp.concatenate((env.goalless_obs(obs),
                                            env.ith_goal(obs, goal_idx)), axis=-1)
                qs = hdq_networks.option_q_network.apply(*params, nn_input)
                return qs

            return get_ap_q
        elif k == spot.op_Not:
            return lambda params, obs: -children[0](params, obs)
        elif k == spot.op_And:
            return lambda params, obs: jnp.min(jnp.stack(
              (children[0](params, obs), children[1](params, obs)), axis=-1
            ), axis=-1)
        elif k == spot.op_Or:
            return lambda params, obs: jnp.max(jnp.stack(
              (children[0](params, obs), children[1](params, obs)), axis=-1
            ), axis=-1)
        else:
            raise NotImplementedError(f"Formula {root} with kind = {k}")

    for k, cond_bdd in sorted(out_conditions.items(), key=lambda x: x[0]):
        if k == 0 and not env.use_incoming_conditions_for_final_state:
            def default_q(params, obs):
                nn_input = jnp.concatenate((env.goalless_obs(obs),
                                            env.ith_goal(obs, 0)), axis=-1)
                qs = hdq_networks.option_q_network.apply(*params, nn_input)
                return qs
            preds.append(jax.jit(default_q))
        else:
            f_k = spot.bdd_to_formula(cond_bdd, bdd_dict)
            q_func_k = fold_spot_formula(functools.partial(to_q_func_helper, aut_state=k),
                                         f_k)
            preds.append(jax.jit(q_func_k))

    return preds

# def get_compiled_q_function_branches(
#     hdq_networks: HDQNetworks,
#     env: envs.Env,
# ):
#     # out_conditions = env.out_conditions
#     # bdd_dict = env.automaton.bdd_dict
#     # ap_to_goal_idx_dict = env.ap_to_goal_idx_dict

#     preds = []

#     def q_zero(params, obs):
#         nn_input = jnp.concatenate((env.goalless_obs(obs),
#                                     env.ith_goal(obs, 0)), axis=-1)
#         qs = hdq_networks.option_q_network.apply(*params, nn_input)
#         return qs

#     def q_one(params, obs):
#         nn_input = jnp.concatenate((env.goalless_obs(obs),
#                                     env.ith_goal(obs, 0)), axis=-1)
#         qs = hdq_networks.option_q_network.apply(*params, nn_input)
#         return qs

#     preds.append(q_zero)
#     preds.append(q_one)

#     return preds


def make_option_inference_fn(
    hdq_networks: HDQNetworks,
    env: envs.Env,
):

  q_func_branches = get_compiled_q_function_branches(hdq_networks, env)

  def make_policy(
      params: types.PolicyParams,
      deterministic: bool = False,
  ) -> types.Policy:

    options = hdq_networks.options
    n_options = len(options)

    # jaxpr0 = jax.make_jaxpr(q_func_branches[0])(params, jnp.array([0., 0., 0., 0., 0., 0., 0., 0., 0.]))
    # jaxpr1 = jax.make_jaxpr(q_func_branches[1])(params, jnp.array([0., 0., 0., 0., 0., 0., 0., 0., 0.]))
    # breakpoint()
    # print(jaxpr)

    def random_option_policy(observation: types.Observation,
                             option_key: PRNGKey) -> Tuple[types.Action, types.Extra]:
      option = jax.random.randint(option_key, (observation.shape[0],), 0, n_options)
      return option, {}


    def greedy_option_policy(observation: types.Observation,
                             option_key: PRNGKey) -> Tuple[types.Action, types.Extra]:
      aut_state = env.aut_state_from_obs(observation)
      double_qs = jax.vmap(lambda a_s, obs: jax.lax.switch(a_s, q_func_branches, params, obs))(jnp.atleast_1d(aut_state), jnp.atleast_2d(observation))
      qs = double_qs.min(axis=-1)
      option = qs.argmax(axis=-1)
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


def make_inference_fn(
    hdq_networks: HDQNetworks,
    env: envs.Env,
):

  q_func_branches = get_compiled_q_function_branches(hdq_networks, env)

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

      aut_state = env.aut_state_from_obs(observation)
      # double_qs = jax.vmap(lambda a_s, obs: jax.lax.switch(a_s, q_func_branches, params, obs))(jnp.atleast_1d(aut_state), jnp.atleast_2d(observation))
      double_qs = jax.lax.switch(aut_state, q_func_branches, params, observation)
      qs = double_qs.min(axis=-1)
      option = qs.argmax(axis=-1)
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
