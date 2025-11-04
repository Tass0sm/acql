import mlflow

from brax.io import model
from brax.training.acme import running_statistics

from acql.brax.envs.wrappers.automaton_wrapper import AutomatonWrapper
from acql.brax.envs.wrappers.automaton_goal_wrapper import AutomatonGoalWrapper

from acql.brax.utils import make_reward_machine_mdp
# from acql.scripts.new_acql_testing import make_aut_goal_cmdp
from acql.brax.utils import make_aut_goal_cmdp

from brax.training.agents.ppo import networks as ppo_networks
from acql.brax.agents.hdqn import networks as hdq_networks
from acql.brax.agents.hdcqn import networks as hdcq_networks
from acql.brax.agents.hdqn_automaton_her import networks as hdq_aut_networks
from acql.brax.agents.acql import networks as acql_networks
from acql.brax.agents.acddpg import networks as acddpg_networks
from acql.brax.agents.sac_her import networks as sac_networks
from acql.brax.agents.ddpg import networks as ddpg_networks

from acql.baselines.logical_options_framework.visualization import (
    get_lof_option_mdp_network_policy_and_params,
    get_lof_mdp_network_policy_and_params
)

from acql.scripts.multihead_exp_train import make_network_factory as make_rm_multihead_network_factory

from acql.tasks.utils import get_task_by_name


def get_acql_mdp_network_policy_and_params(task, run, params, use_sum_cost_critic=False):
    options = task.get_options()

    if run.data.params["normalize_observations"] == "True":
        normalize_fn = running_statistics.normalize
        cost_normalize_fn = running_statistics.normalize
    else:
        normalize_fn = lambda x, y: x
        cost_normalize_fn = lambda x, y: x

    h_dim = int(run.data.params.get("h_dim", 256))
    n_hidden = int(run.data.params.get("n_hidden", 2))

    aut_goal_cmdp = make_aut_goal_cmdp(task, randomize_goals=False, relu_cost=use_sum_cost_critic)
    acql_network = acql_networks.make_acql_networks(
        aut_goal_cmdp.qr_nn_input_size,
        aut_goal_cmdp.qc_nn_input_size,
        aut_goal_cmdp.automaton.n_states,
        aut_goal_cmdp.n_unique_safety_conds,
        aut_goal_cmdp.action_size,
        aut_goal_cmdp,
        network_type=run.data.params["network_type"],
        options=options,
        preprocess_observations_fn=normalize_fn,
        preprocess_cost_observations_fn=cost_normalize_fn,
        hidden_layer_sizes=[h_dim] * n_hidden,
        # hidden_cost_layer_sizes=(128, 128, 128, 64),
        use_sum_cost_critic=use_sum_cost_critic,
    )
    make_option_policy = acql_networks.make_option_inference_fn(acql_network, 
                                                                 aut_goal_cmdp,
                                                                 task.hdcqn_her_hps["safety_threshold"],
                                                                 actor_type=run.data.params.get("actor_type", "safest"),
                                                                 use_sum_cost_critic=use_sum_cost_critic)
    make_policy = acql_networks.make_inference_fn(acql_network, 
                                                   aut_goal_cmdp,
                                                   task.hdcqn_her_hps["safety_threshold"],
                                                   actor_type=run.data.params.get("actor_type", "safest"),
                                                   use_sum_cost_critic=use_sum_cost_critic)

    return aut_goal_cmdp, options, acql_network, make_option_policy, make_policy, params


def get_acddpg_mdp_network_policy_and_params(task, run, params):
    if run.data.params["normalize_observations"] == "True":
        normalize_fn = running_statistics.normalize
        cost_normalize_fn = running_statistics.normalize
    else:
        normalize_fn = lambda x, y: x
        cost_normalize_fn = lambda x, y: x

    layer_norm = run.data.params.get("use_ln", True)
    h_dim = int(run.data.params.get("h_dim", 256))
    n_hidden = int(run.data.params.get("n_hidden", 2))

    aut_goal_cmdp = make_aut_goal_cmdp(task, randomize_goals=False)
    acddpg_network = acddpg_networks.make_acddpg_networks(
        aut_goal_cmdp.qr_nn_input_size,
        aut_goal_cmdp.qc_nn_input_size,
        aut_goal_cmdp.automaton.n_states,
        aut_goal_cmdp.n_unique_safety_conds,
        aut_goal_cmdp.action_size,
        aut_goal_cmdp,
        network_type=run.data.params.get("network_type", "old_default"),
        preprocess_observations_fn=normalize_fn,
        preprocess_cost_observations_fn=cost_normalize_fn,
        hidden_layer_sizes=[h_dim] * n_hidden,
        layer_norm=layer_norm,
        # hidden_layer_sizes=(256, 256),
        # hidden_cost_layer_sizes=(128, 128, 128, 64),
        use_sum_cost_critic=(run.data.tags["alg"] == "ABLATION_FOUR"),
    )
    make_policy = acddpg_networks.make_inference_fn(acddpg_network)

    return aut_goal_cmdp, None, acddpg_network, None, make_policy, params


def get_hdqn_her_for_aut_mdp_network_policy_and_params(task, run, params):
    options = task.get_options()

    if run.data.params["normalize_observations"] == "True":
        normalize_fn = running_statistics.normalize
    else:
        normalize_fn = lambda x, y: x

    mdp = AutomatonWrapper(
        task.env,
        task.lo_spec,
        task.obs_var,
        augment_obs = True,
        strip_goal_obs = False,
    )

    mdp = AutomatonGoalWrapper(mdp)

    hdq_network = hdq_networks.make_hdq_networks(
          mdp.observation_size,
          mdp.action_size,
          options=options,
          preprocess_observations_fn=normalize_fn,
    )
    make_option_policy = hdq_networks.make_option_inference_fn(hdq_network)
    make_policy = hdq_networks.make_inference_fn(hdq_network)

    return mdp, options, hdq_network, make_option_policy, make_policy, params




def get_ddpg_mdp_network_policy_and_params(task, run, params):

    if run.data.params["normalize_observations"] == "True":
        normalize_fn = running_statistics.normalize
    else:
        normalize_fn = lambda x, y: x

    layer_norm = run.data.params.get("use_ln", True)
    h_dim = int(run.data.params.get("h_dim", 256))
    n_hidden = int(run.data.params.get("n_hidden", 2))

    mdp = task.env
    ddpg_network = ddpg_networks.make_ddpg_networks(
        mdp.observation_size,
        mdp.action_size,
        preprocess_observations_fn=normalize_fn,
        layer_norm=layer_norm,
        hidden_layer_sizes=[h_dim] * n_hidden,
    )
    make_policy = ddpg_networks.make_inference_fn(ddpg_network)

    return mdp, None, ddpg_network, None, make_policy, params


def get_ppo_mdp_network_policy_and_params(task, run, params):

    if run.data.params["normalize_observations"] == "True":
        normalize_fn = running_statistics.normalize
    else:
        normalize_fn = lambda x, y: x

    mdp = task.env
    ddpg_network = ppo_networks.make_ppo_networks(
        mdp.observation_size,
        mdp.action_size,
        preprocess_observations_fn=normalize_fn,
    )
    make_policy = ddpg_networks.make_inference_fn(ddpg_network)

    return mdp, None, ddpg_network, None, make_policy, params


def get_rm_mdp_network_policy_and_params(task, run, params):
    options = task.get_options()

    if run.data.params["normalize_observations"] == "True":
        normalize_fn = running_statistics.normalize
    else:
        normalize_fn = lambda x, y: x

    mdp = make_reward_machine_mdp(task, reward_shaping=run.data.params.get("reward_shaping", False))
    rm_network = hdq_networks.make_hdq_networks(
        mdp.observation_size,
        mdp.action_size,
        options=options,
        preprocess_observations_fn=normalize_fn,
    )
    make_option_policy = hdq_networks.make_option_inference_fn(rm_network)
    make_policy = hdq_networks.make_inference_fn(rm_network)

    return mdp, options, rm_network, make_option_policy, make_policy, params


def get_rm_with_multihead_mdp_network_policy_and_params(task, run, params):
    options = task.get_options()

    if run.data.params["normalize_observations"] == "True":
        normalize_fn = running_statistics.normalize
    else:
        normalize_fn = lambda x, y: x

    mdp = make_reward_machine_mdp(task, reward_shaping=run.data.params.get("reward_shaping", False))
    network_factory = make_rm_multihead_network_factory(mdp)
    rm_network = network_factory(
        mdp.observation_size,
        mdp.action_size,
        options=options,
        preprocess_observations_fn=normalize_fn,
    )
    make_option_policy = hdq_networks.make_option_inference_fn(rm_network)
    make_policy = hdq_networks.make_inference_fn(rm_network)

    return mdp, options, rm_network, make_option_policy, make_policy, params


def get_sac_mdp_network_policy_and_params(task, run, params):
    if run.data.params["normalize_observations"] == "True":
        normalize_fn = running_statistics.normalize
    else:
        normalize_fn = lambda x, y: x

    layer_norm = run.data.params.get("use_ln", True)
    h_dim = int(run.data.params.get("h_dim", 256))
    n_hidden = int(run.data.params.get("n_hidden", 2))

    mdp = task.env
    sac_network = sac_networks.make_sac_networks(
        mdp.observation_size,
        mdp.action_size,
        normalize_fn,
        layer_norm=layer_norm,
        hidden_layer_sizes=[h_dim] * n_hidden,
    )
    make_policy = sac_networks.make_inference_fn(sac_network)

    return mdp, None, sac_network, None, make_policy, params


def get_mdp_network_policy_and_params(training_run_id):
    logged_model_path = f'runs:/{training_run_id}/policy_params'
    real_path = mlflow.artifacts.download_artifacts(logged_model_path)
    params = model.load_params(real_path)

    # %%
    run = mlflow.get_run(run_id=training_run_id)

    if (task_name := run.data.tags.get("task_name", None)) is not None:
        task = get_task_by_name(task_name)
        return get_lof_option_mdp_network_policy_and_params(task, run, params)

    alg_name = run.data.tags["alg"]
    task_name = run.data.tags["spec"]
    task = get_task_by_name(task_name)

    match alg_name:
            case "ACQL":
                return get_acql_mdp_network_policy_and_params(task, run, params)
            case "ACQL_ABLATION_TWO":
                return get_acql_mdp_network_policy_and_params(task, run, params, use_sum_cost_critic=True)
            case "ACDDPG":
                return get_acddpg_mdp_network_policy_and_params(task, run, params)
            case "HDCQN_AUTOMATON_HER":
                return get_acql_mdp_network_policy_and_params(task, run, params)
            case "HDQN_HER_FOR_AUT":
                return get_hdqn_her_for_aut_mdp_network_policy_and_params(task, run, params)
            case "SAC_HER":
                return get_sac_mdp_network_policy_and_params(task, run, params)
            case "CRM":
                return get_rm_mdp_network_policy_and_params(task, run, params)
            case "CRM_RS":
                return get_rm_mdp_network_policy_and_params(task, run, params)
            case "CRM_WITH_MULTIHEAD":
                return get_rm_with_multihead_mdp_network_policy_and_params(task, run, params)
            case "QRM":
                return get_rm_mdp_network_policy_and_params(task, run, params)
            case "QRM_WITH_MULTIHEAD":
                return get_rm_with_multihead_mdp_network_policy_and_params(task, run, params)
            case "DDPG":
                return get_ddpg_mdp_network_policy_and_params(task, run, params)
            case "DDPG_HER":
                return get_ddpg_mdp_network_policy_and_params(task, run, params)
            case "PPO":
                return get_ppo_mdp_network_policy_and_params(task, run, params)
            case "LOF":
                return get_lof_mdp_network_policy_and_params(task, run, params)
            case _:
                raise NotImplementedError(f"{alg_name} not supported")

    # aut_goal_cmdp = make_aut_goal_cmdp(task)
    # acql_network = acql_networks.make_hdcq_networks(
    #       aut_goal_cmdp.observation_size,
    #       aut_goal_cmdp.cost_observation_size,
    #       aut_goal_cmdp.automaton.n_states,
    #       aut_goal_cmdp.action_size,
    #       options=options,
    #       preprocess_observations_fn=normalize,
    #       preprocess_cost_observations_fn=cost_normalize_fn,
    # )
    # make_option_policy = acql_networks.make_option_inference_fn(acql_network, aut_goal_cmdp, task.hdcqn_her_hps["safety_minimum"])
