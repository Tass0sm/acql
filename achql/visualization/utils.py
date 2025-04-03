import mlflow

from brax.io import model
from brax.training.acme import running_statistics

from achql.brax.utils import make_aut_goal_cmdp

from achql.brax.agents.hdqn import networks as hdq_networks
from achql.brax.agents.hdcqn import networks as hdcq_networks
from achql.brax.agents.hdqn_automaton_her import networks as hdq_aut_networks
from achql.brax.agents.hdcqn_automaton_her import networks as hdcq_aut_networks

from achql.tasks.utils import get_task_by_name


def get_achql_mdp_network_policy_and_params(task, run, params):
    options = task.get_options()

    if run.data.params["normalize_observations"] == "True":
        normalize_fn = running_statistics.normalize
        cost_normalize_fn = running_statistics.normalize
    else:
        normalize_fn = lambda x, y: x
        cost_normalize_fn = lambda x, y: x

    aut_goal_cmdp = make_aut_goal_cmdp(task, randomize_goals=False)
    hdcq_aut_network = hdcq_aut_networks.make_hdcq_networks(
          aut_goal_cmdp.observation_size,
          aut_goal_cmdp.cost_observation_size,
          aut_goal_cmdp.automaton.n_states,
          aut_goal_cmdp.action_size,
          options=options,
          preprocess_observations_fn=normalize_fn,
          preprocess_cost_observations_fn=cost_normalize_fn,
          # hidden_layer_sizes=(256, 256),
          # hidden_cost_layer_sizes=(128, 128, 128, 64),
          use_sum_cost_critic=(run.data.tags["alg"] == "ABLATION_FOUR"),
    )
    make_option_policy = hdcq_aut_networks.make_option_inference_fn(hdcq_aut_network, aut_goal_cmdp, task.hdcqn_her_hps["safety_threshold"], argmax_type=run.data.params["argmax_type"])
    make_policy = hdcq_aut_networks.make_inference_fn(hdcq_aut_network, aut_goal_cmdp, task.hdcqn_her_hps["safety_threshold"], argmax_type=run.data.params["argmax_type"])    

    return aut_goal_cmdp, hdcq_aut_network, make_option_policy, params


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
            case "HDCQN_AUTOMATON_HER":
                return get_achql_mdp_network_policy_and_params(task, run, params)
            case _:
                raise NotImplementedError()

    # aut_goal_cmdp = make_aut_goal_cmdp(task)
    # hdcq_aut_network = hdcq_aut_networks.make_hdcq_networks(
    #       aut_goal_cmdp.observation_size,
    #       aut_goal_cmdp.cost_observation_size,
    #       aut_goal_cmdp.automaton.n_states,
    #       aut_goal_cmdp.action_size,
    #       options=options,
    #       preprocess_observations_fn=normalize,
    #       preprocess_cost_observations_fn=cost_normalize_fn,
    # )
    # make_option_policy = hdcq_aut_networks.make_option_inference_fn(hdcq_aut_network, aut_goal_cmdp, task.hdcqn_her_hps["safety_minimum"])
