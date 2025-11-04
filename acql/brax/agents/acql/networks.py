import functools
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

from acql.stl import fold_spot_formula
from acql.brax.training.acme import running_statistics
from acql.hierarchy.training import networks as h_networks
from acql.hierarchy.state import OptionState
from acql.hierarchy.option import Option
# from acql.brax.agents.hdqn_automaton_her.networks import get_compiled_q_function_branches
from acql.brax.agents.acql.argmaxes import argmax_with_safest_tiebreak, argmax_with_random_tiebreak




@flax.struct.dataclass
class ACQLNetworks:
    option_q_network: networks.FeedForwardNetwork
    cost_q_network: networks.FeedForwardNetwork
    options: Sequence[Option]


def make_option_qr_fn(
    acql_networks: ACQLNetworks,
    safety_threshold: float,
    use_sum_cost_critic: bool = False,
):

    def make_qr(
            params: types.PolicyParams,
    ) -> types.Policy:

        normalizer_params, option_q_params, cost_q_params = params

        def qr(observation: types.Observation) -> jnp.ndarray:
            double_qs = acql_networks.option_q_network.apply(normalizer_params, option_q_params, observation)
            qs = jnp.min(double_qs, axis=-1)
            return qs

        return qr

    return make_qr


def make_option_qc_fn(
    acql_networks: ACQLNetworks,
    env: envs.Env,
    use_sum_cost_critic: bool = False,
):

    def make_qc(
            params: types.PolicyParams,
    ) -> types.Policy:

        normalizer_params, option_q_params, cost_q_params = params

        def qc(observation: types.Observation) -> jnp.ndarray:
            double_cqs = acql_networks.cost_q_network.apply(normalizer_params, cost_q_params, observation)

            if use_sum_cost_critic:
                cqs = jnp.max(double_cqs, axis=-1)
            else:
                cqs = jnp.min(double_cqs, axis=-1)

            return cqs

        return qc

    return make_qc


def make_option_inference_fn(
    acql_networks: ACQLNetworks,
    env: envs.Env,
    safety_threshold: float,
    use_sum_cost_critic : bool = False,
    actor_type : str = "argmax"
):

    def make_policy(
            params: types.PolicyParams,
            deterministic: bool = False,
    ) -> types.Policy:

        normalizer_params, option_q_params, cost_q_params = params
        options = acql_networks.options
        n_options = len(options)

        def random_option_policy(observation: types.Observation,
                                                          option_key: PRNGKey) -> Tuple[types.Action, types.Extra]:
            option = jax.random.randint(option_key, (observation.shape[0],), 0, n_options)
            return option, {}

        def greedy_safe_option_policy(observation: types.Observation,
                                      option_key: PRNGKey) -> Tuple[types.Action, types.Extra]:
            state_obs, goals, aut_state = env.split_obs(observation)
            _, aut_state_obs = env.split_aut_obs(observation)

            # aut_state = env.automaton.one_hot_decode(aut_state_obs)

            qr_input = jnp.atleast_2d(observation)
            double_qs = acql_networks.option_q_network.apply(normalizer_params, option_q_params, qr_input)
            qs = jnp.min(double_qs, axis=-1)

            qc_input = jnp.atleast_2d(observation)
            double_cqs = acql_networks.cost_q_network.apply(normalizer_params, cost_q_params, qc_input)

            if use_sum_cost_critic:
                cqs = jnp.max(double_cqs, axis=-1)
                masked_q = jnp.where(cqs < safety_threshold, qs, -999.)
            else:
                cqs = jnp.min(double_cqs, axis=-1)
                masked_q = jnp.where(cqs > safety_threshold, qs, -999.)

            if actor_type == "argmax_random":
                option = argmax_with_random_tiebreak(masked_q, option_key, axis=-1)
            elif actor_type == "argmax_safest":
                option = argmax_with_safest_tiebreak(masked_q, cqs, axis=-1, use_sum_cost_critic=use_sum_cost_critic)
            elif actor_type == "argmax":
                option = masked_q.argmax(axis=-1)
            else:
                raise NotImplementedError()
            
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
    acql_networks: ACQLNetworks,
    env: envs.Env,
    safety_threshold: float,
    use_sum_cost_critic : bool = False,
    actor_type : str = "argmax"
):

    make_option_policy = make_option_inference_fn(acql_networks,
                                                  env,
                                                  safety_threshold,
                                                  use_sum_cost_critic=use_sum_cost_critic,
                                                  actor_type=actor_type)

    def make_policy(
            params: types.PolicyParams,
            deterministic: bool = False
    ) -> types.Policy:

        option_policy = make_option_policy(params, deterministic=deterministic)
        options = acql_networks.options

        def policy(
                observations: types.Observation,
                option_state: OptionState,
                key_sample: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:

            option_key, inference_key = jax.random.split(key_sample, 2)

            # calculate o_t from s_t, b_t, o_t-1
            def get_option(s_t, b_t, o_t_minus_1, option_key):
                return jax.lax.cond((b_t == 1),
                                    lambda: option_policy(s_t, option_key).squeeze(),
                                    lambda: o_t_minus_1)

            new_option, _ = option_policy(observations, option_key)
            option = jnp.where(option_state.option_beta == 1, new_option, option_state.option)

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


# class FiLM(linen.Module):
#     feature_dim: int
#     condition_dim: int

#     @linen.compact
#     def __call__(self, features: jnp.ndarray, condition: jnp.ndarray) -> jnp.ndarray:
#         """
#         Args:
#                 features: A tensor of shape (batch_size, feature_dim).
#                 condition: A tensor of shape (batch_size, condition_dim).

#         Returns:
#                 A tensor of shape (batch_size, feature_dim) with modulated features.
#         """
#         # Modulation network: outputs gamma and beta from the condition
#         gamma_beta = linen.Dense(2 * self.feature_dim)(condition)  # (batch_size, 2 * feature_dim)
#         gamma, beta = jnp.split(gamma_beta, 2, axis=-1)  # Split into gamma and beta

#         # Apply feature-wise linear modulation
#         return gamma * features + beta


# class FiLMedMLP(linen.Module):
#     """FiLMed MLP module."""
#     condition_dim: int
#     layer_sizes: Sequence[int]
#     activation: ActivationFn = linen.relu
#     kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
#     bias: bool = True
#     layer_norm: bool = False
#     final_activation: ActivationFn = linen.tanh

#     @linen.compact
#     def __call__(self, data: jnp.ndarray, conditioning_data: jnp.ndarray):
#         hidden = data
#         for i, hidden_size in enumerate(self.layer_sizes):
#             hidden = linen.Dense(
#                     hidden_size,
#                     name=f'hidden_{i}',
#                     kernel_init=self.kernel_init,
#                     use_bias=self.bias)(
#                             hidden)

#             if i != len(self.layer_sizes) - 1:
#                 hidden = self.activation(hidden)
#                 if self.layer_norm:
#                     hidden = linen.LayerNorm()(hidden)

#                 # hidden = FiLM(
#                 #   feature_dim=hidden_size,
#                 #   condition_dim=self.condition_dim
#                 # )(hidden, conditioning_data)

#                 # hidden = self.activation(hidden)

#         hidden = self.final_activation(hidden)

#         return hidden


# def make_option_cost_q_network(
#     obs_size: int,
#     aut_states: int,
#     num_options: int,
#     preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
#     hidden_layer_sizes: Sequence[int] = (256, 256),
#     activation: ActivationFn = linen.relu,
#     n_critics: int = 2,
#     use_sum_cost_critic: bool = False,
# ) -> FeedForwardNetwork:
#     """Creates a value network."""

#     plain_obs_size = obs_size - aut_states

#     # Q function for discrete space of options.
#     class DiscreteCostQModule(linen.Module):
#         """Q Module."""
#         n_critics: int

#         @linen.compact
#         def __call__(self, obs: jnp.ndarray, aut_obs: jnp.ndarray):
#             hidden = jnp.concatenate([obs], axis=-1)
#             res = []
#             for _ in range(self.n_critics):
#                 critic_option_qs = FiLMedMLP(
#                     condition_dim=aut_states,
#                     layer_sizes=list(hidden_layer_sizes) + [num_options],
#                     activation=activation,
#                     final_activation=(lambda x: x) if use_sum_cost_critic else linen.tanh,
#                     # layer_norm=True,
#                     kernel_init=jax.nn.initializers.lecun_uniform()
#                 )(hidden, aut_obs)
#                 res.append(critic_option_qs)

#             return jnp.stack(res, axis=-1)

#     q_module = DiscreteCostQModule(n_critics=n_critics)

#     def apply(processor_params, q_params, obs):
#         trimmed_proc_params = jax.tree.map(lambda x: x[..., :plain_obs_size] if x.ndim >= 1 else x, processor_params)
#         state_obs, aut_state_obs = obs[..., :plain_obs_size], obs[..., plain_obs_size:]
#         norm_state_obs = preprocess_observations_fn(state_obs, trimmed_proc_params)
#         return q_module.apply(q_params, norm_state_obs, aut_state_obs)

#     dummy_obs = jnp.zeros((1, plain_obs_size))
#     dummy_aut_obs = jnp.zeros((1, aut_states))
#     return FeedForwardNetwork(
#             init=lambda key: q_module.init(key, dummy_obs, dummy_aut_obs), apply=apply)



def get_compiled_option_q_function_branches_for_active_goal_obs(
    option_q_network: FeedForwardNetwork,
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
                goalless_obs = obs[..., :env.no_goal_obs_dim]
                goal = env.ith_goal(obs, goal_idx)
                aut_state_obs = obs[..., -env.automaton.n_states:]
                nn_input = jnp.concatenate((goalless_obs, goal, aut_state_obs), axis=-1)
                # qs = option_q_network.apply(*params, nn_input)

                if nn_input.ndim < 2:
                    nn_input = jnp.atleast_2d(nn_input)
                    qs = option_q_network.apply(*params, nn_input).squeeze()
                else:
                    qs = option_q_network.apply(*params, nn_input)

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
        if k == env.accepting_state and not env.use_incoming_conditions_for_final_state:
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



# Utils for the algebraic Q network - for multi-goal obs
def get_compiled_option_q_function_branches_for_multi_goal_obs(
    option_q_network: FeedForwardNetwork,
    env: envs.Env,
):
    out_conditions = env.out_conditions
    bdd_dict = env.automaton.bdd_dict
    ap_id_to_goal_idx_dict = env.ap_id_to_goal_idx_dict

    preds = []

    def to_q_func_helper(root, children, aut_state=0):
        nonlocal ap_id_to_goal_idx_dict

        children = list(children)

        k = root.kind()

        if k == spot.op_ff:
            return lambda params, obs: -987
        elif k == spot.op_tt:
            return lambda params, obs: 987
        elif k == spot.op_ap:
            ap_id = int(root.ap_name()[3:])
            goal_idx = ap_id_to_goal_idx_dict[ap_id]

            # this function is usually vmapped so it should just take and return vectors
            def get_ap_q(params, obs):
                state_obs = obs[..., :env.no_goal_obs_dim]
                goals = obs[..., env.no_goal_obs_dim:env.no_goal_obs_dim+env.all_goals_dim]
                goal = goals[..., goal_idx*env.goal_dim:(goal_idx+1)*env.goal_dim]

                aut_state_obs = obs[..., -env.automaton.n_states:]

                nn_input = jnp.concatenate((state_obs, goal, aut_state_obs), axis=-1)
                # nn_input = jnp.concatenate((state_obs, goal), axis=-1)

                if nn_input.ndim < 2:
                    nn_input = jnp.atleast_2d(nn_input)
                    qs = option_q_network.apply(*params, nn_input).squeeze()
                else:
                    qs = option_q_network.apply(*params, nn_input)

                # qs = option_q_network.apply(*params, nn_input)

                # TODO, consider selecting individual goal if passing all goals
                # is impossible.
                # nn_input = jnp.concatenate((o, gs), axis=-1)
                # qs = option_q_network.apply(*params, obs)
                # qs = option_q_network.apply(*params, nn_input).squeeze()
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
        if k == env.accepting_state and not env.use_incoming_conditions_for_final_state:
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


def make_algebraic_option_q_network_for_multi_goal_obs(
        subnetwork: FeedForwardNetwork,
        env,
) -> FeedForwardNetwork:

    q_func_branches = get_compiled_option_q_function_branches_for_multi_goal_obs(subnetwork, env)

    def apply(processor_params, q_params, obs):
        _, _, aut_state = env.split_obs(obs)
        # this algebraic q network still passes the full observation to the
        # underlying network incase it still wants to do anything with the
        # automaton state information
        batched_q = jax.vmap(lambda a_s, o: jax.lax.switch(a_s, q_func_branches, (processor_params, q_params), o))
        return batched_q(aut_state, obs)

    return FeedForwardNetwork(init=subnetwork.init, apply=apply)


def make_algebraic_option_q_network_for_active_goal_obs(
        subnetwork: FeedForwardNetwork,
        env,
) -> FeedForwardNetwork:

    q_func_branches = get_compiled_option_q_function_branches_for_active_goal_obs(subnetwork, env)

    def apply(processor_params, q_params, obs):
        _, _, aut_state = env.split_obs(obs)
        # this algebraic q network still passes the full observation to the
        # underlying network incase it still wants to do anything with the
        # automaton state information
        batched_q = jax.vmap(lambda a_s, o: jax.lax.switch(a_s, q_func_branches, (processor_params, q_params), o))
        return batched_q(aut_state, obs)

    return FeedForwardNetwork(init=subnetwork.init, apply=apply)


def get_compiled_cost_q_function_branches_for_active_goal_obs(
    cost_q_network: FeedForwardNetwork,
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
                goalless_obs = obs[..., :env.no_goal_obs_dim]
                goal = env.ith_goal(obs, goal_idx)
                aut_state_obs = obs[..., -env.automaton.n_states:]
                nn_input = jnp.concatenate((goalless_obs, goal, aut_state_obs), axis=-1)

                if nn_input.ndim < 2:
                    nn_input = jnp.atleast_2d(nn_input)
                    qs = cost_q_network.apply(*params, nn_input).squeeze()
                else:
                    qs = cost_q_network.apply(*params, nn_input)

                return qs

            return get_ap_q
        elif k == spot.op_Not:
            return lambda params, obs: -children[0](params, obs)
        elif k == spot.op_And:
            return lambda params, obs: jnp.min(jnp.stack(
              (children[0](params, obs), children[1](params, obs)), axis=-1
            ), axis=-1)
        elif k == spot.op_Or:
            return lambda params, obs: jnp.min(jnp.stack(
              (children[0](params, obs), children[1](params, obs)), axis=-1
            ), axis=-1)
        else:
            raise NotImplementedError(f"Formula {root} with kind = {k}")

    for k, cond_bdd in sorted(out_conditions.items(), key=lambda x: x[0]):
        if k == env.accepting_state and not env.use_incoming_conditions_for_final_state:
            def default_q(params, obs):
                nn_input = jnp.concatenate((env.goalless_obs(obs),
                                            env.ith_goal(obs, 0)), axis=-1)
                qs = hdq_networks.cost_q_network.apply(*params, nn_input)
                return qs
            preds.append(jax.jit(default_q))
        else:
            f_k = spot.bdd_to_formula(cond_bdd, bdd_dict)
            q_func_k = fold_spot_formula(functools.partial(to_q_func_helper, aut_state=k),
                                         f_k)
            preds.append(jax.jit(q_func_k))

    return preds


def make_algebraic_cost_q_network_for_active_goal_obs(
        subnetwork: FeedForwardNetwork,
        env,
) -> FeedForwardNetwork:

    q_func_branches = get_compiled_cost_q_function_branches_for_active_goal_obs(subnetwork, env)

    def apply(processor_params, q_params, obs):
        _, _, aut_state = env.split_obs(obs)
        # this algebraic q network still passes the full observation to the
        # underlying network incase it still wants to do anything with the
        # automaton state information
        batched_q = jax.vmap(lambda a_s, o: jax.lax.switch(a_s, q_func_branches, (processor_params, q_params), o))
        return batched_q(aut_state, obs)

    return FeedForwardNetwork(init=subnetwork.init, apply=apply)


def make_residual_option_q_network(
        obs_size,
        main_network: FeedForwardNetwork,
        residual_network: FeedForwardNetwork,
) -> FeedForwardNetwork:

    # multi-headed Q function for discrete space of options.
    class ResidualQModule(linen.Module):
        """Multi-Headed Q Module."""
        q_network: linen.Module
        residual_q_network: linen.Module

        @linen.compact
        def __call__(self, obs: jnp.ndarray):
            # Learnable scalar weight (initialized to 0.0)
            # residual_weight = self.param('residual_weight', nn.initializers.ones, (1,))

            qs = self.q_network(obs)
            residual_qs = self.residual_q_network(obs)
            qs = qs + residual_qs
            return qs

    q_module = ResidualQModule(main_network, residual_network)

    def apply(processor_params, q_params, obs):
        return q_module(processor_params, q_params, obs)

    dummy_obs = jnp.zeros((obs_size,))
    return FeedForwardNetwork(init=lambda key: q_module.init(key, dummy_obs), apply=apply)


def make_acql_networks(
        observation_size: int,
        cost_observation_size: int,
        num_aut_states: int,
        num_unique_safety_conditions: int,
        action_size: int,
        env,
        network_type: str,
        preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
        preprocess_cost_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
        hidden_layer_sizes: Sequence[int] = (256, 256),
        hidden_cost_layer_sizes=(64, 64, 64, 32),
        activation: networks.ActivationFn = linen.relu,
        options: Sequence[Option] = [],
        use_sum_cost_critic: bool = False,
) -> ACQLNetworks:

    assert len(options) > 0, "Must pass at least one option"
    # assert num_unique_safety_conditions == 1, "Only supporting 1 safety condition"

    # ALGEBRAIC VERSION

    # def qr_preprocess_obs_fn(obs: jax.Array, processor_params) -> int:
    #     # obs, aut_obs = obs[..., :-env.automaton.n_states], obs[..., -env.automaton.n_states:]
    #     # obs = preprocess_observations_fn(obs, processor_params)
    #     # obs = jnp.concatenate((obs, aut_obs), axis=-1)
    #     obs, aut_obs = obs[..., :-env.automaton.n_states], obs[..., -env.automaton.n_states:]
    #     obs = preprocess_observations_fn(obs, processor_params)
    #     return obs

    # def qr_preprocess_cond_fn(obs: jax.Array) -> int:
    #     aut_obs = obs[-env.automaton.n_states:]
    #     aut_state = env.automaton.one_hot_decode(aut_obs)
    #     return env.state_to_unique_safety_cond_idx_arr[aut_state]

    # single_gc_network = h_networks.make_option_q_network(
    #     observation_size,
    #     len(options),
    #     preprocess_observations_fn=qr_preprocess_obs_fn,
    #     hidden_layer_sizes=hidden_layer_sizes,
    #     activation=activation,
    # )

    # option_q_network = make_algebraic_option_q_network(single_gc_network, env)

    # def qc_preprocess_obs_fn(state_and_aut_obs: jax.Array, trimmed_processor_params) -> int:
    #     state_obs = state_and_aut_obs[..., :-env.automaton.n_states]
    #     return preprocess_cost_observations_fn(state_obs, trimmed_processor_params)

    # cost_q_network = h_networks.make_option_q_network(
    #     cost_observation_size,
    #     len(options),
    #     preprocess_observations_fn=qc_preprocess_obs_fn,
    #     hidden_layer_sizes=hidden_cost_layer_sizes,
    #     activation=activation,
    #     final_activation=(lambda x: x) if use_sum_cost_critic else linen.tanh,
    # )

    # RESIDUAL VERSION

    # def qr_preprocess_obs_fn(obs: jax.Array, processor_params) -> int:

    #     obs = obs[..., :-env.automaton.n_states]
    #     return preprocess_observations_fn(obs, processor_params)

    # def qc_preprocess_obs_fn(state_and_aut_obs: jax.Array, trimmed_processor_params) -> int:
    #     state_obs = state_and_aut_obs[..., :-env.automaton.n_states]
    #     return preprocess_cost_observations_fn(state_obs, trimmed_processor_params)

    # single_gc_network = h_networks.make_option_q_network(
    #     env.no_goal_obs_dim + env.goal_width * env.goal_dim,
    #     len(options),
    #     preprocess_observations_fn=qr_preprocess_obs_fn,
    #     hidden_layer_sizes=hidden_layer_sizes,
    #     activation=activation,
    # )

    # ALGEBRAIC WITH RESIDUAL FOR GOALS VERSION

    # def qr_preprocess_obs_fn(state_and_goal: jax.Array, processor_params) -> int:
    #     # obs = obs[..., :-env.automaton.n_states]
    #     obs = preprocess_observations_fn(state_and_goal, processor_params)
    #     return obs

    # single_gc_network = h_networks.make_option_q_network(
    #     observation_size,
    #     len(options),
    #     preprocess_observations_fn=qr_preprocess_obs_fn,
    #     hidden_layer_sizes=hidden_layer_sizes,
    #     activation=activation,
    # )

    # algebraic_gc_network = make_algebraic_option_q_network(single_gc_network, env)

    # def residual_preprocess_obs_fn(obs: jax.Array, processor_params) -> int:
    #     goal_processor_params = jax.tree.map(lambda x: x[..., -env.goal_dim:] if x.ndim >= 1 else x, normalizer_params)
    #     goals_processor_params = jax.tree.map(lambda x: jnp.tile(x, (1, env.n_goal_aps - 1)) if x.ndim >= 1 else x, goal_processor_params)
    #     full_processor_params = jax.tree.map(lambda x: jnp.concatenate((x, goals_processor_params), axis=-1) if x.ndim >= 1 else x, normalizer_params)
    #     rest, aut_state_obs = env.split_aut_obs(obs)
    #     rest = preprocess_observations_fn(rest, full_processor_params)
    #     obs = jnp.concatenate((rest, aut_state_obs), axis=-1)
    #     return obs

    # residual_network = h_networks.make_option_q_network(
    #     env.no_goal_obs_dim + env.all_goals_dim + env.automaton.n_states,
    #     len(options),
    #     preprocess_observations_fn=residual_preprocess_obs_fn,
    #     hidden_layer_sizes=hidden_layer_sizes,
    #     activation=activation,
    # )

    # option_q_network = make_residual_option_q_network(
    #     env.no_goal_obs_dim + env.all_goals_dim + env.automaton.n_states,
    #     algebraic_gc_network,
    #     residual_network,
    # )


    # def qc_preprocess_obs_fn(state_and_aut_obs: jax.Array, trimmed_processor_params) -> int:
    #     state_obs = state_and_aut_obs[..., :-env.automaton.n_states]
    #     return preprocess_cost_observations_fn(state_obs, trimmed_processor_params)

    # cost_q_network = h_networks.make_option_q_network(
    #     cost_observation_size,
    #     len(options),
    #     preprocess_observations_fn=qc_preprocess_obs_fn,
    #     hidden_layer_sizes=hidden_cost_layer_sizes,
    #     activation=activation,
    #     final_activation=(lambda x: x) if use_sum_cost_critic else linen.tanh,
    # )



    # # ALGEBRAIC VERSION IGNORING FUTURE GOALS

    if network_type == "old_default":
        def qr_preprocess_obs_fn(state_and_goal_and_aut_obs: jax.Array, processor_params) -> jax.Array:
            state_and_goal = state_and_goal_and_aut_obs[..., :-env.automaton.n_states]
            obs = preprocess_observations_fn(state_and_goal, processor_params)
            return obs

        single_gc_network = h_networks.make_option_q_network(
            observation_size,
            len(options),
            preprocess_observations_fn=qr_preprocess_obs_fn,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation
        )

        option_q_network = make_algebraic_option_q_network_for_active_goal_obs(single_gc_network, env)

        def qc_preprocess_obs_fn(state_and_goal_and_aut_obs: jax.Array, processor_params) -> jax.Array:
            state_and_goal = state_and_goal_and_aut_obs[..., :-env.automaton.n_states]
            obs = preprocess_cost_observations_fn(state_and_goal, processor_params)
            return obs

        single_gc_cost_q_network = h_networks.make_option_q_network(
                cost_observation_size,
                len(options),
                preprocess_observations_fn=qc_preprocess_obs_fn,
                hidden_layer_sizes=hidden_cost_layer_sizes,
                activation=activation,
                final_activation=(lambda x: x) if use_sum_cost_critic else linen.tanh,
        )

        cost_q_network = make_algebraic_cost_q_network_for_active_goal_obs(single_gc_cost_q_network, env)

    elif network_type == "old_multihead":
        def qr_preprocess_obs_fn(state_and_goal_and_aut_state: jax.Array, processor_params) -> int:
            state_and_goal = state_and_goal_and_aut_state[..., :-env.automaton.n_states]
            obs = preprocess_observations_fn(state_and_goal, processor_params)
            return obs

        def qr_preprocess_cond_fn(state_and_goal_and_aut_state: jax.Array) -> int:
            aut_obs = state_and_goal_and_aut_state[..., -env.automaton.n_states:]
            aut_state = env.automaton.one_hot_decode(aut_obs)
            return env.state_to_unique_safety_cond_idx_arr[aut_state]

        single_gc_network = h_networks.make_multi_headed_option_q_network(
                observation_size,
                num_unique_safety_conditions,
                len(options),
                env,
                preprocess_observations_fn=qr_preprocess_obs_fn,
                preprocess_cond_fn=qr_preprocess_cond_fn,
                shared_hidden_layer_sizes=hidden_layer_sizes[:1],
                head_hidden_layer_sizes=hidden_layer_sizes[1:],
                activation=activation
        )

        option_q_network = make_algebraic_option_q_network_for_active_goal_obs(single_gc_network, env)

        def qc_preprocess_obs_fn(state_and_goal_and_aut_obs: jax.Array, processor_params) -> jax.Array:
            state_and_goal = state_and_goal_and_aut_obs[..., :-env.automaton.n_states]
            obs = preprocess_cost_observations_fn(state_and_goal, processor_params)
            return obs

        def qc_preprocess_cond_fn(state_and_goal_aut_obs: jax.Array) -> int:
            aut_obs = state_and_goal_aut_obs[..., -env.automaton.n_states:]
            aut_state = env.automaton.one_hot_decode(aut_obs)
            return env.state_to_unique_safety_cond_idx_arr[aut_state]

        single_gc_cost_q_network = h_networks.make_multi_headed_option_q_network(
                cost_observation_size,
                num_unique_safety_conditions,
                len(options),
                env,
                preprocess_observations_fn=qc_preprocess_obs_fn,
                preprocess_cond_fn=qc_preprocess_cond_fn,
                shared_hidden_layer_sizes=hidden_cost_layer_sizes[:2],
                head_hidden_layer_sizes=hidden_cost_layer_sizes[2:],
                activation=activation,
                final_activation=(lambda x: x) if use_sum_cost_critic else linen.tanh,
        )

        cost_q_network = make_algebraic_cost_q_network_for_active_goal_obs(single_gc_cost_q_network, env)

    elif network_type == "old_multihead_no_goal_qc":
        def qr_preprocess_obs_fn(state_and_goal_and_aut_state: jax.Array, processor_params) -> int:
            state_and_goal = state_and_goal_and_aut_state[..., :-env.automaton.n_states]
            obs = preprocess_observations_fn(state_and_goal, processor_params)
            return obs

        def qr_preprocess_cond_fn(state_and_goal_and_aut_state: jax.Array) -> int:
            aut_obs = state_and_goal_and_aut_state[..., -env.automaton.n_states:]
            aut_state = env.automaton.one_hot_decode(aut_obs)
            return env.state_to_unique_safety_cond_idx_arr[aut_state]

        single_gc_network = h_networks.make_multi_headed_option_q_network(
                observation_size,
                num_unique_safety_conditions,
                len(options),
                env,
                preprocess_observations_fn=qr_preprocess_obs_fn,
                preprocess_cond_fn=qr_preprocess_cond_fn,
                shared_hidden_layer_sizes=hidden_layer_sizes[:1],
                head_hidden_layer_sizes=hidden_layer_sizes[1:],
                activation=activation
        )

        option_q_network = make_algebraic_option_q_network_for_active_goal_obs(single_gc_network, env)

        def qc_preprocess_obs_fn(state_and_goals_and_aut_obs: jax.Array, processor_params) -> jax.Array:
            # assumes first no_goal_obs_dim features are the state without goal
            state_obs = state_and_goals_and_aut_obs[..., :env.no_goal_obs_dim]
            obs = preprocess_cost_observations_fn(state_obs, processor_params)
            return obs

        def qc_preprocess_cond_fn(state_and_goals_and_aut_obs: jax.Array) -> int:
            aut_obs = state_and_goals_and_aut_obs[..., -env.automaton.n_states:]
            aut_state = env.automaton.one_hot_decode(aut_obs)
            return env.state_to_unique_safety_cond_idx_arr[aut_state]

        cost_observation_size = env.no_goal_obs_dim

        cost_q_network = h_networks.make_multi_headed_option_q_network(
                cost_observation_size,
                num_unique_safety_conditions,
                len(options),
                env,
                preprocess_observations_fn=qc_preprocess_obs_fn,
                preprocess_cond_fn=qc_preprocess_cond_fn,
                shared_hidden_layer_sizes=hidden_cost_layer_sizes[:2],
                head_hidden_layer_sizes=hidden_cost_layer_sizes[2:],
                activation=activation,
                final_activation=(lambda x: x) if use_sum_cost_critic else linen.tanh,
        )

    elif network_type == "default":
        raise NotImplementedError()
        # def qr_preprocess_obs_fn(state_and_goal_and_aut_state: jax.Array, processor_params) -> jax.Array:
        #     state_and_goal = state_and_goal_and_aut_state[..., :-env.automaton.n_states]
        #     obs = preprocess_observations_fn(state_and_goal, processor_params)
        #     return obs

        # single_gc_network = h_networks.make_option_q_network(
        #     observation_size,
        #     len(options),
        #     preprocess_observations_fn=qr_preprocess_obs_fn,
        #     hidden_layer_sizes=hidden_layer_sizes,
        #     activation=activation
        # )

        # option_q_network = make_algebraic_option_q_network_for_multi_goal_obs(single_gc_network, env)

        # def qc_preprocess_obs_fn(state_and_aut_obs: jax.Array, trimmed_processor_params) -> jax.Array:
        #     state_obs = state_and_aut_obs[..., :-env.automaton.n_states]
        #     return preprocess_cost_observations_fn(state_obs, trimmed_processor_params)

        # cost_q_network = h_networks.make_option_q_network(
        #         cost_observation_size,
        #         len(options),
        #         preprocess_observations_fn=qc_preprocess_obs_fn,
        #         hidden_layer_sizes=hidden_cost_layer_sizes,
        #         activation=activation,
        #         final_activation=(lambda x: x) if use_sum_cost_critic else linen.tanh,
        # )
    elif network_type == "multihead_qc":
        raise NotImplementedError()
        # # ALGEBRAIC VERSION IGNORING FUTURE GOALS AND WITH RESPECT FOR SAFETY CONDITIONS ONLY IN Q^C
        # def qr_preprocess_obs_fn(state_and_goal_and_aut_state: jax.Array, processor_params) -> jax.Array:
        #     state_and_goal = state_and_goal_and_aut_state[..., :-env.automaton.n_states]
        #     obs = preprocess_observations_fn(state_and_goal, processor_params)
        #     return obs

        # single_gc_network = h_networks.make_option_q_network(
        #     observation_size,
        #     len(options),
        #     preprocess_observations_fn=qr_preprocess_obs_fn,
        #     hidden_layer_sizes=hidden_layer_sizes,
        #     activation=activation
        # )

        # option_q_network = make_algebraic_option_q_network_for_multi_goal_obs(single_gc_network, env)

        # def qc_preprocess_obs_fn(state_and_aut_obs: jax.Array, trimmed_processor_params) -> jax.Array:
        #     state_obs = state_and_aut_obs[..., :-env.automaton.n_states]
        #     return preprocess_cost_observations_fn(state_obs, trimmed_processor_params)

        # def qc_preprocess_cond_fn(state_and_aut_obs: jax.Array) -> jax.Array:
        #     aut_obs = state_and_aut_obs[..., -env.automaton.n_states:]
        #     aut_state = env.automaton.one_hot_decode(aut_obs)
        #     return env.state_to_unique_safety_cond_idx_arr[aut_state]

        # cost_q_network = h_networks.make_multi_headed_option_q_network(
        #         cost_observation_size,
        #         num_unique_safety_conditions,
        #         len(options),
        #         env,
        #         preprocess_observations_fn=qc_preprocess_obs_fn,
        #         preprocess_cond_fn=qc_preprocess_cond_fn,
        #         shared_hidden_layer_sizes=hidden_cost_layer_sizes[:2],
        #         head_hidden_layer_sizes=hidden_cost_layer_sizes[2:],
        #         activation=activation,
        #         final_activation=(lambda x: x) if use_sum_cost_critic else linen.tanh,
        # )
    elif network_type == "multihead":
        raise NotImplementedError()

        # ALGEBRAIC VERSION IGNORING FUTURE GOALS AND WITH RESPECT FOR SAFETY CONDITIONS
    
        # def qr_preprocess_obs_fn(state_and_goal_and_aut_state: jax.Array, processor_params) -> int:
        #     state_and_goal = state_and_goal_and_aut_state[..., :-env.automaton.n_states]
        #     obs = preprocess_observations_fn(state_and_goal, processor_params)
        #     return obs

        # def qr_preprocess_cond_fn(state_and_goal_and_aut_state: jax.Array) -> int:
        #     aut_obs = state_and_goal_and_aut_state[..., -env.automaton.n_states:]
        #     aut_state = env.automaton.one_hot_decode(aut_obs)
        #     return env.state_to_unique_safety_cond_idx_arr[aut_state]

        # single_gc_network = h_networks.make_multi_headed_option_q_network(
        #         observation_size,
        #         num_unique_safety_conditions,
        #         len(options),
        #         env,
        #         preprocess_observations_fn=qr_preprocess_obs_fn,
        #         preprocess_cond_fn=qr_preprocess_cond_fn,
        #         shared_hidden_layer_sizes=hidden_layer_sizes[:2],
        #         head_hidden_layer_sizes=hidden_layer_sizes[2:],
        #         activation=activation
        # )

        # option_q_network = make_algebraic_option_q_network(single_gc_network, env)

        # def qc_preprocess_obs_fn(state_and_aut_obs: jax.Array, trimmed_processor_params) -> int:
        #     state_obs = state_and_aut_obs[..., :-env.automaton.n_states]
        #     return preprocess_cost_observations_fn(state_obs, trimmed_processor_params)

        # def qc_preprocess_cond_fn(state_and_aut_obs: jax.Array) -> int:
        #     aut_obs = state_and_aut_obs[..., -env.automaton.n_states:]
        #     aut_state = env.automaton.one_hot_decode(aut_obs)
        #     return env.state_to_unique_safety_cond_idx_arr[aut_state]

        # cost_q_network = h_networks.make_multi_headed_option_q_network(
        #         cost_observation_size,
        #         num_unique_safety_conditions,
        #         len(options),
        #         env,
        #         preprocess_observations_fn=qc_preprocess_obs_fn,
        #         preprocess_cond_fn=qc_preprocess_cond_fn,
        #         shared_hidden_layer_sizes=hidden_cost_layer_sizes[:2],
        #         head_hidden_layer_sizes=hidden_cost_layer_sizes[2:],
        #         activation=activation,
        #         final_activation=(lambda x: x) if use_sum_cost_critic else linen.tanh,
        # )
    else:
        raise NotImplementedError()

    # ALGEBRAIC VERSION (IGNORING FUTURE GOALS)

    # def qr_preprocess_obs_fn(state_and_goal: jax.Array, processor_params) -> int:
    #     # obs = obs[..., :-env.automaton.n_states]
    #     obs = preprocess_observations_fn(state_and_goal, processor_params)
    #     return obs

    # single_gc_network = h_networks.make_option_q_network(
    #     observation_size,
    #     len(options),
    #     preprocess_observations_fn=qr_preprocess_obs_fn,
    #     hidden_layer_sizes=hidden_layer_sizes,
    #     activation=activation,
    # )

    # option_q_network = make_algebraic_option_q_network(single_gc_network, env)

    # def qc_preprocess_obs_fn(state_and_aut_obs: jax.Array, trimmed_processor_params) -> int:
    #     state_obs = state_and_aut_obs[..., :-env.automaton.n_states]
    #     return preprocess_cost_observations_fn(state_obs, trimmed_processor_params)

    # cost_q_network = h_networks.make_option_q_network(
    #     cost_observation_size,
    #     len(options),
    #     preprocess_observations_fn=qc_preprocess_obs_fn,
    #     hidden_layer_sizes=hidden_cost_layer_sizes,
    #     activation=activation,
    #     final_activation=(lambda x: x) if use_sum_cost_critic else linen.tanh,
    # )

    # MLP VERSION (IGNORING AUTOMATON STATE)

    # def qr_preprocess_obs_fn(obs: jax.Array, processor_params) -> int:
    #     obs = obs[..., :-env.automaton.n_states]
    #     return preprocess_observations_fn(obs, processor_params)

    # def qc_preprocess_obs_fn(state_and_aut_obs: jax.Array, trimmed_processor_params) -> int:
    #     state_obs = state_and_aut_obs[..., :-env.automaton.n_states]
    #     return preprocess_cost_observations_fn(state_obs, trimmed_processor_params)

    # option_q_network = h_networks.make_option_q_network(
    #     observation_size,
    #     len(options),
    #     preprocess_observations_fn=qr_preprocess_obs_fn,
    #     hidden_layer_sizes=hidden_layer_sizes,
    #     activation=activation
    # )
    # cost_q_network = h_networks.make_option_q_network(
    #     cost_observation_size,
    #     len(options),
    #     preprocess_observations_fn=qc_preprocess_obs_fn,
    #     hidden_layer_sizes=hidden_cost_layer_sizes,
    #     activation=activation,
    #     final_activation=(lambda x: x) if use_sum_cost_critic else linen.tanh,
    # )

    # MULTI HEAD VERSION

    # def qr_preprocess_obs_fn(obs: jax.Array, processor_params) -> int:
    #     obs = obs[..., :-env.automaton.n_states]
    #     obs = preprocess_observations_fn(obs, processor_params)
    #     return obs

    # def qr_preprocess_cond_fn(obs: jax.Array) -> int:
    #     aut_obs = obs[..., -env.automaton.n_states:]
    #     aut_state = env.automaton.one_hot_decode(aut_obs)
    #     return aut_state

    # option_q_network = h_networks.make_multi_headed_option_q_network(
    #         observation_size,
    #         env.automaton.n_states,
    #         len(options),
    #         env,
    #         preprocess_observations_fn=qr_preprocess_obs_fn,
    #         preprocess_cond_fn=qr_preprocess_cond_fn,
    #         shared_hidden_layer_sizes=hidden_layer_sizes[:1],
    #         head_hidden_layer_sizes=hidden_layer_sizes[1:],
    #         activation=activation
    # )

    # def qc_preprocess_obs_fn(state_and_aut_obs: jax.Array, trimmed_processor_params) -> int:
    #     state_obs = state_and_aut_obs[..., :-env.automaton.n_states]
    #     return preprocess_cost_observations_fn(state_obs, trimmed_processor_params)

    # def qc_preprocess_cond_fn(state_and_aut_obs: jax.Array) -> int:
    #     aut_obs = state_and_aut_obs[..., -env.automaton.n_states:]
    #     aut_state = env.automaton.one_hot_decode(aut_obs)
    #     return env.state_to_unique_safety_cond_idx_arr[aut_state]

    # cost_q_network = h_networks.make_multi_headed_option_q_network(
    #         cost_observation_size,
    #         num_unique_safety_conditions,
    #         len(options),
    #         env,
    #         preprocess_observations_fn=qc_preprocess_obs_fn,
    #         preprocess_cond_fn=qc_preprocess_cond_fn,
    #         shared_hidden_layer_sizes=hidden_cost_layer_sizes[:2],
    #         head_hidden_layer_sizes=hidden_cost_layer_sizes[2:],
    #         activation=activation,
    #         final_activation=(lambda x: x) if use_sum_cost_critic else linen.tanh,
    # )

    return ACQLNetworks(
            # policy_network=policy_network,
            option_q_network=option_q_network,
            cost_q_network=cost_q_network,
            options=options,
            # parametric_action_distribution=parametric_action_distribution
    )
