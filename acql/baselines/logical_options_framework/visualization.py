from brax.training import types
from brax.training.acme import running_statistics
import jax.numpy as jnp

from acql.brax.agents.hdqn import networks as hdq_networks
from acql.brax.envs.wrappers.automaton_wrapper import JaxAutomaton, AutomatonWrapper
from acql.hierarchy.envs.options_wrapper import OptionsWrapper
from acql.baselines.logical_options_framework.lof_wrapper import LOFWrapper
from acql.baselines.logical_options_framework.train import (
    partition, load_logical_options, make_inference_fn1, make_option_inference_fn
)
from acql.baselines.logical_options_framework.utils import get_logical_option_run_ids


Metrics = types.Metrics




def get_lof_option_mdp_network_policy_and_params(task, run, params):
    options = task.get_options()

    if run.data.params["normalize_observations"] == "True":
        normalize_fn = running_statistics.normalize
    else:
        normalize_fn = lambda x, y: x

    full_automaton = JaxAutomaton(task.lo_spec, task.obs_var)

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

    reward_safety_violation = -1000
    reward_safety_matrix = jnp.zeros((2**full_automaton.n_aps,))
    for k in no_goal_aps:
        for i in range(2**full_automaton.n_aps):
            if jnp.bitwise_and(i, k) > 0:
                reward_safety_matrix = reward_safety_matrix.at[i].set(reward_safety_violation)

    def safety_reward_function(action, nobs):
        labels = full_automaton.eval_aps(action, nobs, automaton_ap_params)
        return reward_safety_matrix[labels]

    ap_for_option = next(filter(lambda x: x.info["name"] == run.data.tags["ap_name"], goal_aps.values()))

    mdp = LOFWrapper(task.env,
                     task.obs_var,
                     ap_for_option,
                     safety_reward_function,
                     distance_cost=True)

    hdq_network = hdq_networks.make_hdq_networks(
          mdp.observation_size,
          mdp.action_size,
          options=options,
          preprocess_observations_fn=normalize_fn,
    )
    make_option_policy = hdq_networks.make_option_inference_fn(hdq_network)
    make_policy = hdq_networks.make_inference_fn(hdq_network)

    return mdp, options, hdq_network, make_option_policy, make_policy, params


def get_lof_mdp_network_policy_and_params(task, run, params):
    # first level
    options = task.get_options()

    # second level
    spec = run.data.tags["spec"]
    seed = int(run.data.params["seed"])

    logical_option_run_ids = get_logical_option_run_ids(spec, seed)

    full_automaton = JaxAutomaton(task.lo_spec, task.obs_var)
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

    logical_options = load_logical_options(
        logical_option_run_ids,
        task.env,
        no_goal_aps,
        goal_aps,
        full_automaton,
        automaton_ap_params,
        task.obs_var,
        options=options,
    )

    # making the MDP

    prod_mdp = AutomatonWrapper(
        task.env,
        task.lo_spec,
        task.obs_var,
        strip_goal_obs = True,
        overwrite_reward = True,
        use_incoming_conditions_for_final_state = True,
    )

    mdp = OptionsWrapper(
        prod_mdp,
        options,
        discounting=1.0,
        takes_key=False
    )

    start_states = (
        task.env.possible_starts.tolist() +
        [ap.info["goal"].tolist() for ap in goal_aps.values()]
    )
    start_states_array = jnp.array(start_states)


    make_option_policy = make_option_inference_fn(prod_mdp, logical_options, start_states_array)
    make_policy = make_inference_fn1(prod_mdp, logical_options, options, start_states_array)

    return mdp, logical_options, options, None, make_option_policy, make_policy, params
