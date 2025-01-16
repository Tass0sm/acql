import functools
import time
from typing import Any, Callable, Optional, Tuple, Union, Sequence

import mlflow

from absl import logging
from brax import base
from brax import envs
from brax.envs import wrappers
from brax.io import model
from brax.training import acting
from brax.training import gradients
from brax.training import pmap
from brax.training import replay_buffers
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import optax
import numpy as np

from task_aware_skill_composition.hierarchy.training.evaluator import HierarchicalEvaluatorWithSpecification
from task_aware_skill_composition.hierarchy.option import Option
from task_aware_skill_composition.hierarchy.envs.options_wrapper import OptionsWrapper

from task_aware_skill_composition.brax.agents.hdqn import networks as hdq_networks
from task_aware_skill_composition.brax.envs.wrappers.automaton_wrapper import JaxAutomaton
from task_aware_skill_composition.baselines.logical_options_framework.lof_wrapper import LOFWrapper

from corallab_stl.automata import get_spot_formula_and_aps, make_just_liveness_automaton, get_outgoing_conditions
from corallab_stl import expression_jax2 as stl

Metrics = types.Metrics


def partition(pred, d):
    fs = {}
    ts = {}
    for q, item in d.items():
        if not pred(q, item):
            fs[q] = (item)
        else:
            ts[q] = (item)
    return fs, ts


class PredicateTerminationPolicy:
    def __init__(self, ap, state_var):
        self.ap = ap
        self.state_var = state_var

    def __call__(self, o_t, key):
        rho = self.ap({ self.state_var.idx: jnp.atleast_2d(o_t) }).squeeze()
        return jnp.where(rho > 0, 1, 0).astype(jnp.int32)


# Algorithm 1 from https://arxiv.org/pdf/2102.12571
# With Caching for Logical Options
def train(
    environment: envs.Env,
    specification: Callable[..., jnp.ndarray],
    state_var,
    task_name,
    # num_timesteps,
    # episode_length: int = 1000,
    # wrap_env: bool = True,
    # action_repeat: int = 1,
    # num_envs: int = 128,
    # # num_sampling_per_update: int = 50,
    # num_eval_envs: int = 16,
    # learning_rate: float = 1e-4,
    # discounting: float = 0.9,
    seed: int = 0,
    # batch_size: int = 256,
    # num_evals: int = 1,
    # normalize_observations: bool = True,
    # max_devices_per_host: Optional[int] = None,
    # reward_scaling: float = 1.,
    # tau: float = 0.005,
    # min_replay_size: int = 0,
    # max_replay_size: Optional[int] = 10_0000,
    # grad_updates_per_step: int = 1,
    # deterministic_eval: bool = True,
    # tensorboard_flag = True,
    # logdir = './logs',
    # network_factory: types.NetworkFactory[hdq_networks.HDQNetworks] = hdq_networks.make_hdq_networks,
    logical_option_run_ids: Optional[list] = None,
    option_train_fn: Callable = None,
    option_train_fn_hps: dict = {},
    options=[],
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    # checkpoint_logdir: Optional[str] = None,
    randomization_fn: Optional[
      Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
):

    mlflow.set_experiment("proj2-lof")

    # Common useful things

    full_automaton = JaxAutomaton(specification, state_var)
    liveness_automaton = make_just_liveness_automaton(full_automaton.automaton)
    plain_out_conditions = get_outgoing_conditions(liveness_automaton)

    no_goal_aps, goal_aps = partition(lambda _, v: "goal" in v.info, full_automaton.aps)

    max_param_dim = max([ap.info["goal"].size for ap in goal_aps.values()])
    automaton_ap_params = jnp.zeros((full_automaton.n_aps, max_param_dim))
    for k, ap in full_automaton.aps.items():
        if k in goal_aps:
            dim = ap.info["goal"].size
            automaton_ap_params = automaton_ap_params.at[k, :dim].set(ap.info["goal"])
        elif ap.default_params is not None:
            dim = ap.default_params.size
            automaton_ap_params = automaton_ap_params.at[k, :dim].set(ap.default_params)

    # Train or load options

    if logical_option_run_ids is None:
        logical_options = train_logical_option_policies(
            environment,
            no_goal_aps,
            goal_aps,
            full_automaton,
            automaton_ap_params,
            state_var,
            task_name,
            seed=seed,
            option_train_fn=option_train_fn,
            option_train_fn_hps=option_train_fn_hps,
            options=options,
            progress_fn=progress_fn,
            randomization_fn=randomization_fn,
        )
    else:
        logical_options = load_logical_option_policies(
            logical_option_run_ids,
            environment,
            no_goal_aps,
            goal_aps,
            full_automaton,
            automaton_ap_params,
            state_var,
            options=options,
        )

    # F
    live_states = [k for k in plain_out_conditions]

    # S
    start_states = (
        environment.possible_starts.tolist() +
        [ap.info["goal"].tolist() for ap in goal_aps.values()]
    )

    # O
    # logical_options

    mu = train_logical_option_metapolicy(
        environment,
        no_goal_aps,
        goal_aps,
        full_automaton,
        automaton_ap_params,
        live_states,
        start_states,
        logical_options,
        specification,
        state_var,
        options=options,
    )

    return None, None, None


def train_logical_option_policies(
    environment: envs.Env,
    no_goal_aps,
    goal_aps,
    full_automaton,
    automaton_ap_params,
    state_var,
    task_name,
    # num_timesteps,
    # episode_length: int = 1000,
    # wrap_env: bool = True,
    # action_repeat: int = 1,
    # num_envs: int = 128,
    # # num_sampling_per_update: int = 50,
    # num_eval_envs: int = 16,
    # learning_rate: float = 1e-4,
    # discounting: float = 0.9,
    seed: int = 0,
    # batch_size: int = 256,
    # num_evals: int = 1,
    # normalize_observations: bool = True,
    # max_devices_per_host: Optional[int] = None,
    # reward_scaling: float = 1.,
    # tau: float = 0.005,
    # min_replay_size: int = 0,
    # max_replay_size: Optional[int] = 10_0000,
    # grad_updates_per_step: int = 1,
    # deterministic_eval: bool = True,
    # tensorboard_flag = True,
    # logdir = './logs',
    # network_factory: types.NetworkFactory[hdq_networks.HDQNetworks] = hdq_networks.make_hdq_networks,
    option_train_fn: Callable = None,
    option_train_fn_hps: dict = {},
    options=[],
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    # checkpoint_logdir: Optional[str] = None,
    randomization_fn: Optional[
      Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
):
    """
    Propositions P partitioned into
    1. subgoals PG
    2. safety propositions PS
    3. event propositions PE

    Logical FSA T = (F, P_G × P_E , TF , RF , f0, fg ) derived from φ_liveness
    Low-level MDP_E = (S, A, R_E, T_E, γ), where RE (s, a) = RE (s, a) + RS (T_P_S (s)) combines the environment and safety rewards

    Proposition labeling functions
    1. T_P_G : S → 2^P_G
    2. T_P_S : S → 2^P_S
    3. T_P_E : 2^P_E → {0, 1}
    """

    ####################################
    # CREATE SAFETY REWARD FUNCTION
    ####################################

    reward_safety_violation = -1000
    reward_safety_matrix = jnp.zeros((2**full_automaton.n_aps,))
    for k in no_goal_aps:
        for i in range(2**full_automaton.n_aps):
            if jnp.bitwise_and(i, k) > 0:
                reward_safety_matrix = reward_safety_matrix.at[i].set(reward_safety_violation)

    def safety_reward_function(action, nobs):
        labels = full_automaton.eval_aps(action, nobs, automaton_ap_params)
        return reward_safety_matrix[labels]

    ####################################
    # TRAIN LOGICAL OPTION POLICIES
    ####################################

    logical_options = []
    for _, ap in goal_aps.items():
        with mlflow.start_run(tags={"task_name": task_name, "ap_name": ap.info["name"]}) as run:

            mlflow.log_params({
                **option_train_fn_hps,
                "seed": seed,
            })

            option_env = LOFWrapper(environment, state_var, ap, safety_reward_function, distance_cost=True)
            make_flat_policy, params, _ = option_train_fn(
                option_env,
                stl.STLUntimedEventually(ap),
                state_var,
                options=options,
                progress_fn=progress_fn,
                randomization_fn=randomization_fn,
                seed=seed,
                **option_train_fn_hps,
            )
    
            run_id = run.info.run_id
            with mlflow.MlflowClient()._log_artifact_helper(run_id, f'policy_params') as tmp_path:
                model.save_params(tmp_path, params)

        if option_train_fn_hps["normalize_observations"] == "True":
            normalize_fn = running_statistics.normalize
        else:
            normalize_fn = lambda x, y: x

        hdq_network = hdq_networks.make_hdq_networks(
              option_env.observation_size,
              option_env.action_size,
              options=options,
              preprocess_observations_fn=normalize_fn,
        )
        inference_fn = make_flat_policy(params, deterministic=True)

        make_q = hdq_networks.make_q_fn(hdq_network)
        reward_model = lambda obs: make_q(params)(obs).max(axis=-1)

        termination_policy = PredicateTerminationPolicy(ap, state_var)
        logical_option = Option(
            ap.info["name"], hdq_network, params, inference_fn,
            termination_policy=termination_policy,
            reward_model=reward_model,
            adapter=lambda x: x[:4]
        )

        logical_options.append(logical_option)

    return logical_options


def load_logical_option_policies(
    logical_option_run_ids,
    environment: envs.Env,
    no_goal_aps,
    goal_aps,
    full_automaton,
    automaton_ap_params,
    state_var,
    # task_name,
    options=[],
):
    """
    Propositions P partitioned into
    1. subgoals PG
    2. safety propositions PS
    3. event propositions PE

    Logical FSA T = (F, P_G × P_E , TF , RF , f0, fg ) derived from φ_liveness
    Low-level MDP_E = (S, A, R_E, T_E, γ), where RE (s, a) = RE (s, a) + RS (T_P_S (s)) combines the environment and safety rewards

    Proposition labeling functions
    1. T_P_G : S → 2^P_G
    2. T_P_S : S → 2^P_S
    3. T_P_E : 2^P_E → {0, 1}
    """

    ####################################
    # CREATE SAFETY REWARD FUNCTION
    ####################################

    reward_safety_violation = -1000
    reward_safety_matrix = jnp.zeros((2**full_automaton.n_aps,))
    for k in no_goal_aps:
        for i in range(2**full_automaton.n_aps):
            if jnp.bitwise_and(i, k) > 0:
                reward_safety_matrix = reward_safety_matrix.at[i].set(reward_safety_violation)

    def safety_reward_function(action, nobs):
        labels = full_automaton.eval_aps(action, nobs, automaton_ap_params)
        return reward_safety_matrix[labels]

    ####################################
    # TRAIN LOGICAL OPTION POLICIES
    ####################################

    logical_options = []
    for (_, ap), run_id in zip(goal_aps.items(), logical_option_run_ids):

        option_env = LOFWrapper(environment, state_var, ap, safety_reward_function)

        training_run_id = run_id
        logged_model_path = f'runs:/{training_run_id}/policy_params'
        real_path = mlflow.artifacts.download_artifacts(logged_model_path)
        params = model.load_params(real_path)

        run = mlflow.get_run(run_id=training_run_id)
        if run.data.params["normalize_observations"] == "True":
            normalize_fn = running_statistics.normalize
        else:
            normalize_fn = lambda x, y: x

        hdq_network = hdq_networks.make_hdq_networks(
              option_env.observation_size,
              option_env.action_size,
              options=options,
              preprocess_observations_fn=normalize_fn,
        )
        make_policy = hdq_networks.make_option_inference_fn(hdq_network)
        inference_fn = make_policy(params, deterministic=True)

        make_q = hdq_networks.make_q_fn(hdq_network)
        reward_model = lambda params, obs: make_q(params)(obs).max(axis=-1)

        termination_policy = PredicateTerminationPolicy(ap, state_var)

        logical_option = Option(
            ap.info["name"], hdq_network, params, inference_fn,
            termination_policy=termination_policy,
            reward_model=reward_model,
            adapter=lambda x: x[:4]
        )

        logical_options.append(logical_option)

    return logical_options



def train_logical_option_metapolicy(
    environment: envs.Env,
    no_goal_aps,
    goal_aps,
    full_automaton,
    automaton_ap_params,
    live_states,
    start_states,
    logical_options,
    specification: Callable[..., jnp.ndarray],
    state_var,
    # num_timesteps,
    episode_length: int = 1000,
    # wrap_env: bool = True,
    action_repeat: int = 1,
    # num_envs: int = 128,
    # # num_sampling_per_update: int = 50,
    # num_eval_envs: int = 16,
    # learning_rate: float = 1e-4,
    discounting: float = 0.9,
    seed: int = 0,
    # batch_size: int = 256,
    # num_evals: int = 1,
    # normalize_observations: bool = True,
    # max_devices_per_host: Optional[int] = None,
    # reward_scaling: float = 1.,
    # tau: float = 0.005,
    # min_replay_size: int = 0,
    # max_replay_size: Optional[int] = 10_0000,
    # grad_updates_per_step: int = 1,
    deterministic_eval: bool = True,
    # tensorboard_flag = True,
    # logdir = './logs',
    # network_factory: types.NetworkFactory[hdq_networks.HDQNetworks] = hdq_networks.make_hdq_networks,
    options=[],
):
    """
    Propositions P partitioned into
    1. subgoals PG
    2. safety propositions PS
    3. event propositions PE

    Logical FSA T = (F, P_G × P_E , TF , RF , f0, fg ) derived from φ_liveness
    Low-level MDP_E = (S, A, R_E, T_E, γ), where RE (s, a) = RE (s, a) + RS (T_P_S (s)) combines the environment and safety rewards

    Proposition labeling functions
    1. T_P_G : S → 2^P_G
    2. T_P_S : S → 2^P_S
    3. T_P_E : 2^P_E → {0, 1}
    """

    # whole start states (assumes goal space subset of state space is first n features.
    n = environment.pos_indices.size
    rem = environment.state_dim - n
    whole_start_states = jnp.hstack((jnp.array(start_states), jnp.zeros((len(start_states), rem))))

    # R_F
    R_F = lambda f: 1.0

    # R_o
    # comes from logical_options[i].reward_model

    ####################################
    # LEARN THE META POLICY WITH VALUE ITERATION
    ####################################

    # get the automaton transition functions
    def T_F(f, sigma, f_prime) -> bool:
        return jnp.where(full_automaton.delta_u[f, sigma] == f_prime, 1, 0).astype(jnp.int32)

    def T_P(s):
        return full_automaton.eval_aps(None, s, automaton_ap_params)

    def T_o(o, s, s_prime):
        if jnp.bitwise_and(2**o, T_P(s_prime)):
            return 1
        else:
            return 0

    # initialize the value function
    Q = jnp.zeros((len(live_states), len(start_states), len(logical_options)))
    V = jnp.zeros((len(live_states), len(start_states)))

    # Q function for state u and symbol sigma
    def get_Q(f, s, o):
        lo = logical_options[o]

        # get the reward for the transition.
        r = R_F(f) * lo.reward_model(lo.params, whole_start_states[s])

        # next state value
        next_v = 0
        for f_prime in range(len(live_states)):
            for s_prime in range(len(start_states)):
                whole_s_prime = whole_start_states[s_prime]
                sigma = T_P(whole_s_prime)
                next_v += T_F(f, sigma, f_prime) * T_o(o, s, whole_s_prime) * V[f_prime, s_prime]

        return r + next_v

    for k in range(15):
        jax.debug.breakpoint()
        for f in range(len(live_states)):
            for s in range(len(start_states)):
                Q_os = []
                for o in range(len(logical_options)):
                    Q_o = get_Q(f, s, o)
                    Q_os.append(Q_o)
                    Q = Q.at[f, s, o].set(Q_o)

                V = V.at[f, s].set(max(Q_os))
                
    make_policy = make_option_inference_fn(Q, logical_options)
    make_flat_policy = make_inference_fn(Q, logical_options)

    first_env = OptionsWrapper(
        environment,
        options,
        discounting=discounting,
        takes_key=False
    )

    second_env = OptionsWrapper(
        first_env,
        logical_options,
        discounting=1.0,
    )

    wrap_for_training = envs.training.wrap

    rng = jax.random.PRNGKey(seed)
    rng, key = jax.random.split(rng)
    v_randomization_fn = None
    env = wrap_for_training(
        second_env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )

    evaluator = HierarchicalEvaluatorWithSpecification(
        env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        logical_options,
        specification=specification,
        state_var=state_var,
        num_eval_envs=16,
        episode_length=10,
        action_repeat=action_repeat,
        key=key
    )

    metrics = evaluator.run_evaluation(None, training_metrics={})
    print(metrics)

    return make_flat_policy, None, None




def make_option_inference_fn(Q, logical_options):

  def make_policy(
      params: types.PolicyParams,
      deterministic: bool = False,
  ) -> types.Policy:

    n_options = len(logical_options)

    def random_option_policy(observation: types.Observation,
                             option_key: PRNGKey) -> Tuple[types.Action, types.Extra]:
      # suboption_keys = jax.random.split(option_key, observation.shape[0])
      option = jax.random.randint(option_key, (observation.shape[0],), 0, n_options)
      return option, {}

    def greedy_option_policy(observation: types.Observation,
                             option_key: PRNGKey) -> Tuple[types.Action, types.Extra]:
      # TODO: Fix this..  
      # suboption_keys = jax.random.split(option_key, observation.shape[0])
      return jnp.zeros((observation.shape[0],), dtype=jnp.int32), {}

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


def make_inference_fn(Q: jnp.ndarray, logical_options):

  def make_policy(
      params: types.PolicyParams,
      deterministic: bool = False
  ) -> types.Policy:

    n_options = len(logical_options)

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
        actions, _ = jax.lax.switch(o_t, [o.inference for o in logical_options], s_t, inf_key)
        return actions, {}

      inf_keys = jax.random.split(inference_key, observations.shape[0])
      actions, info = jax.vmap(low_level_inference)(observations, option, inf_keys)
      info.update(option=option)

      return actions, info

    return policy

  return make_policy
