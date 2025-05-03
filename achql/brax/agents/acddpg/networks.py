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

from achql.stl import fold_spot_formula
from achql.brax.training.acme import running_statistics


@flax.struct.dataclass
class ACDDPGNetworks:
    policy_network: networks.FeedForwardNetwork
    q_network: networks.FeedForwardNetwork
    cost_q_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_qr_fn(
    acddpg_networks: ACDDPGNetworks,
):

    def make_qr(
            params: types.PolicyParams,
    ) -> types.Policy:

        normalizer_params, q_params, _, _ = params

        def qr(observation: types.Observation, action: jax.Array) -> jnp.ndarray:
            double_qs = acddpg_networks.q_network.apply(normalizer_params, q_params, observation, action)
            qs = jnp.min(double_qs, axis=-1)
            return qs

        return qr

    return make_qr


def make_qc_fn(
    acddpg_networks: ACDDPGNetworks,
    env: envs.Env,
    use_sum_cost_critic: bool = False,
):

    def make_qc(
            params: types.PolicyParams,
    ) -> types.Policy:

        normalizer_params, _, cost_q_params, _ = params

        def qc(observation: types.Observation, action: jax.Array) -> jnp.ndarray:
            state_obs, _, aut_state = env.split_obs(observation)
            _, aut_state_obs = env.split_aut_obs(observation)

            qc_input = jnp.concatenate((state_obs, aut_state_obs), axis=-1)
            cost_observation_size = state_obs.shape[-1]
            trimmed_normalizer_params = jax.tree.map(lambda x: x[..., :cost_observation_size] if x.ndim >= 1 else x, normalizer_params)
            double_cqs = acddpg_networks.cost_q_network.apply(trimmed_normalizer_params, cost_q_params, qc_input, action)

            if use_sum_cost_critic:
                cqs = jnp.max(double_cqs, axis=-1)
            else:
                cqs = jnp.min(double_cqs, axis=-1)

            return cqs

        return qc

    return make_qc


def make_inference_fn(acddpg_networks: ACDDPGNetworks):

    def make_policy(params: types.PolicyParams, deterministic: bool = False) -> types.Policy:

        def policy(
                observations: types.Observation,
                key_sample: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:
            logits = acddpg_networks.policy_network.apply(*params, observations)
            if deterministic:
                return acddpg_networks.parametric_action_distribution.mode(logits), {}
            origin_action = acddpg_networks.parametric_action_distribution.sample_no_postprocessing(logits, key_sample)
            return acddpg_networks.parametric_action_distribution.postprocess(origin_action), {}

        return policy

    return make_policy


def get_compiled_q_function_branches_for_active_goal_obs(
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

            def get_ap_q(params, obs, action):
                goalless_obs = obs[..., :env.no_goal_obs_dim]
                goal = env.ith_goal(obs, goal_idx)
                aut_state_obs = obs[..., -env.automaton.n_states:]
                nn_input = jnp.concatenate((goalless_obs, goal, aut_state_obs), axis=-1)
                # qs = option_q_network.apply(*params, nn_input)

                if nn_input.ndim < 2:
                    nn_input = jnp.atleast_2d(nn_input)
                    action = jnp.atleast_2d(action)
                    qs = option_q_network.apply(*params, nn_input, action).squeeze()
                else:
                    qs = option_q_network.apply(*params, nn_input, action)

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
            def default_q(params, obs, action):
                nn_input = jnp.concatenate((env.goalless_obs(obs),
                                            env.ith_goal(obs, 0)), axis=-1)
                qs = hdq_networks.option_q_network.apply(*params, nn_input, action)
                return qs
            preds.append(jax.jit(default_q))
        else:
            f_k = spot.bdd_to_formula(cond_bdd, bdd_dict)
            q_func_k = fold_spot_formula(functools.partial(to_q_func_helper, aut_state=k),
                                         f_k)
            preds.append(jax.jit(q_func_k))

    return preds



# Utils for the algebraic Q network - for multi-goal obs
def get_compiled_q_function_branches_for_multi_goal_obs(
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


def make_algebraic_q_network_for_multi_goal_obs(
        subnetwork: FeedForwardNetwork,
        env,
) -> FeedForwardNetwork:

    q_func_branches = get_compiled_q_function_branches_for_multi_goal_obs(subnetwork, env)

    def apply(processor_params, q_params, obs, action):
        _, _, aut_state = env.split_obs(obs)
        # this algebraic q network still passes the full observation to the
        # underlying network incase it still wants to do anything with the
        # automaton state information
        batched_q = jax.vmap(lambda a_s, o, a: jax.lax.switch(a_s, q_func_branches, (processor_params, q_params), o, a))
        return batched_q(aut_state, obs, action)

    return FeedForwardNetwork(init=subnetwork.init, apply=apply)


def make_algebraic_q_network_for_active_goal_obs(
        subnetwork: FeedForwardNetwork,
        env,
) -> FeedForwardNetwork:

    q_func_branches = get_compiled_q_function_branches_for_active_goal_obs(subnetwork, env)

    def apply(processor_params, q_params, obs, action):
        _, _, aut_state = env.split_obs(obs)
        # this algebraic q network still passes the full observation to the
        # underlying network incase it still wants to do anything with the
        # automaton state information
        batched_q = jax.vmap(lambda a_s, o, a: jax.lax.switch(a_s, q_func_branches, (processor_params, q_params), o, a))
        return batched_q(aut_state, obs, action)

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


def make_acddpg_networks(
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
        layer_norm: bool = False,
        use_sum_cost_critic: bool = False,
) -> ACDDPGNetworks:

    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)

    # ALGEBRAIC VERSION IGNORING FUTURE GOALS

    if network_type == "old_default":
        def policy_preprocess_obs_fn(state_and_goal_and_aut_state: jax.Array, processor_params) -> jax.Array:
            state_and_goal = state_and_goal_and_aut_state[..., :-env.automaton.n_states]
            obs = preprocess_observations_fn(state_and_goal, processor_params)
            return obs

        policy_network = networks.make_policy_network(
            parametric_action_distribution.param_size,
            observation_size,
            preprocess_observations_fn=policy_preprocess_obs_fn,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            layer_norm=layer_norm,
        )

        def qr_preprocess_obs_fn(state_and_goal_and_aut_state: jax.Array, processor_params) -> jax.Array:
            state_and_goal = state_and_goal_and_aut_state[..., :-env.automaton.n_states]
            obs = preprocess_observations_fn(state_and_goal, processor_params)
            return obs

        single_gc_network = networks.make_q_network(
            observation_size,
            action_size,
            preprocess_observations_fn=qr_preprocess_obs_fn,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            layer_norm=layer_norm,
        )

        q_network = make_algebraic_q_network_for_active_goal_obs(single_gc_network, env)

        def qc_preprocess_obs_fn(state_and_goal_and_aut_state: jax.Array, processor_params) -> jax.Array:
            state_and_goal = state_and_goal_and_aut_state[..., :-env.automaton.n_states]
            obs = preprocess_cost_observations_fn(state_and_goal, processor_params)
            return obs

        cost_q_network = networks.make_q_network(
                cost_observation_size,
                action_size,
                preprocess_observations_fn=qc_preprocess_obs_fn,
                hidden_layer_sizes=hidden_cost_layer_sizes,
                activation=activation,
                final_activation=(lambda x: x) if use_sum_cost_critic else linen.tanh,
                layer_norm=layer_norm,
        )
    elif network_type == "old_multihead":
        raise NotImplementedError()
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
        #         shared_hidden_layer_sizes=hidden_layer_sizes[:1],
        #         head_hidden_layer_sizes=hidden_layer_sizes[1:],
        #         activation=activation
        # )

        # option_q_network = make_algebraic_q_network_for_active_goal_obs(single_gc_network, env)

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

        # option_q_network = make_algebraic_q_network_for_multi_goal_obs(single_gc_network, env)

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

        # option_q_network = make_algebraic_q_network_for_multi_goal_obs(single_gc_network, env)

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
        # # ALGEBRAIC VERSION IGNORING FUTURE GOALS AND WITH RESPECT FOR SAFETY CONDITIONS

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

        # option_q_network = make_algebraic_q_network(single_gc_network, env)

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

    # option_q_network = make_algebraic_q_network(single_gc_network, env)

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

    return ACDDPGNetworks(
            policy_network=policy_network,
            q_network=q_network,
            cost_q_network=cost_q_network,
            parametric_action_distribution=parametric_action_distribution
    )
