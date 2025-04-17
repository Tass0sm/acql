import mlflow

from brax.io import model
from brax.training.acme import running_statistics

from achql.brax.envs.wrappers.automaton_wrapper import AutomatonWrapper
from achql.brax.envs.wrappers.automaton_goal_wrapper import AutomatonGoalWrapper

from achql.brax.utils import make_reward_machine_mdp
from achql.scripts.new_achql_testing import make_aut_goal_cmdp

from achql.brax.agents.hdqn import networks as hdq_networks
from achql.brax.agents.hdcqn import networks as hdcq_networks
from achql.brax.agents.hdqn_automaton_her import networks as hdq_aut_networks
from achql.brax.agents.achql import networks as achql_networks
from achql.brax.agents.sac_her import networks as sac_networks
from achql.brax.agents.ddpg import networks as ddpg_networks

from achql.scripts.multihead_exp_train import make_network_factory as make_rm_multihead_network_factory

from achql.tasks.utils import get_task_by_name


def get_achql_mdp_network_policy_and_params(task, run, params):
    options = task.get_options()

    if run.data.params["normalize_observations"] == "True":
        normalize_fn = running_statistics.normalize
        cost_normalize_fn = running_statistics.normalize
    else:
        normalize_fn = lambda x, y: x
        cost_normalize_fn = lambda x, y: x

    aut_goal_cmdp = make_aut_goal_cmdp(task, randomize_goals=True)
    achql_network = achql_networks.make_achql_networks(
        aut_goal_cmdp.qr_nn_input_size,
        aut_goal_cmdp.qc_nn_input_size,
        aut_goal_cmdp.automaton.n_states,
        aut_goal_cmdp.n_unique_safety_conds,
        aut_goal_cmdp.action_size,
        aut_goal_cmdp,
        options=options,
        preprocess_observations_fn=normalize_fn,
        preprocess_cost_observations_fn=cost_normalize_fn,
        # hidden_layer_sizes=(256, 256),
        # hidden_cost_layer_sizes=(128, 128, 128, 64),
        use_sum_cost_critic=(run.data.tags["alg"] == "ABLATION_FOUR"),
    )
    make_option_policy = achql_networks.make_option_inference_fn(achql_network, aut_goal_cmdp, task.hdcqn_her_hps["safety_threshold"], argmax_type=run.data.params.get("argmax_type", "safest"))
    make_policy = achql_networks.make_inference_fn(achql_network, aut_goal_cmdp, task.hdcqn_her_hps["safety_threshold"], argmax_type=run.data.params.get("argmax_type", "safest"))    

    return aut_goal_cmdp, options, achql_network, make_option_policy, make_policy, params


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

    mdp = task.env
    ddpg_network = ddpg_networks.make_ddpg_networks(
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

    mdp = task.env
    sac_network = sac_networks.make_sac_networks(
        mdp.observation_size,
        mdp.action_size,
        normalize_fn
    )
    make_policy = sac_networks.make_inference_fn(sac_network)

    return mdp, None, sac_network, None, make_policy, params


def get_mdp_network_policy_and_params(training_run_id):
    logged_model_path = f'runs:/{training_run_id}/policy_params'
    real_path = mlflow.artifacts.download_artifacts(logged_model_path)
    params = model.load_params(real_path)

    # %%
    run = mlflow.get_run(run_id=training_run_id)

    alg_name = run.data.tags["alg"]
    task_name = run.data.tags["spec"]
    task = get_task_by_name(task_name)

    match alg_name:
            case "ACHQL":
                return get_achql_mdp_network_policy_and_params(task, run, params)
            case "HDCQN_AUTOMATON_HER":
                return get_achql_mdp_network_policy_and_params(task, run, params)
            case "HDQN_HER_FOR_AUT":
                return get_hdqn_her_for_aut_mdp_network_policy_and_params(task, run, params)
            case "SAC_HER":
                return get_sac_mdp_network_policy_and_params(task, run, params)
            case "CRM":
                return get_rm_mdp_network_policy_and_params(task, run, params)
            case "CRM_WITH_MULTIHEAD":
                return get_rm_with_multihead_mdp_network_policy_and_params(task, run, params)
            case "QRM":
                return get_rm_mdp_network_policy_and_params(task, run, params)
            case "QRM_WITH_MULTIHEAD":
                return get_rm_with_multihead_mdp_network_policy_and_params(task, run, params)
            case "DDPG":
                return get_ddpg_mdp_network_policy_and_params(task, run, params)
            case _:
                raise NotImplementedError(f"{alg_name} not supported")

    # aut_goal_cmdp = make_aut_goal_cmdp(task)
    # achql_network = achql_networks.make_hdcq_networks(
    #       aut_goal_cmdp.observation_size,
    #       aut_goal_cmdp.cost_observation_size,
    #       aut_goal_cmdp.automaton.n_states,
    #       aut_goal_cmdp.action_size,
    #       options=options,
    #       preprocess_observations_fn=normalize,
    #       preprocess_cost_observations_fn=cost_normalize_fn,
    # )
    # make_option_policy = achql_networks.make_option_inference_fn(achql_network, aut_goal_cmdp, task.hdcqn_her_hps["safety_minimum"])
