import mlflow

from acql.visualization.utils import get_mdp_network_policy_and_params

from acql.visualization.plots import (
    make_plots_for_achql,
    make_plots_for_3d_achql,
    make_plots_for_acddpg,
    make_plots_for_ddpg,
    make_plots_for_hdqn
)


def make_plots(training_run_id, **kwargs):
    mdp, options, network, make_option_policy, make_policy, params = get_mdp_network_policy_and_params(training_run_id)

    run = mlflow.get_run(run_id=training_run_id)

    if (task_name := run.data.tags.get("task_name", None)) is not None:
        alg_name = "HDQN"
    else:
        alg_name = run.data.tags["alg"]

    match alg_name:
            case "ACHQL":
                env_name = run.data.tags["env"]
                if env_name in ["SimpleMaze", "AntMaze"]:
                    return make_plots_for_achql(mdp, make_option_policy, network, params, **kwargs)
                elif env_name in ["SimpleMaze3D"]:
                    return make_plots_for_3d_achql(mdp, make_option_policy, network, params, **kwargs)
                else:
                    raise NotImplementedError(f"Visualizing ACHQL with {env_name} not supported")
            case "ACHQL_ABLATION_TWO":
                env_name = run.data.tags["env"]
                if env_name in ["SimpleMaze", "AntMaze"]:
                    return make_plots_for_achql(mdp, make_option_policy, network, params, use_sum_cost_critic=True, **kwargs)
                elif env_name in ["SimpleMaze3D"]:
                    return make_plots_for_3d_achql(mdp, make_option_policy, network, params, **kwargs)
                else:
                    raise NotImplementedError(f"Visualizing ACHQL with {env_name} not supported")
            case "ACDDPG":
                return make_plots_for_acddpg(mdp, make_policy, network, params, **kwargs)
            case "HDCQN_AUTOMATON_HER":
                return make_plots_for_achql(mdp, make_option_policy, network, params, **kwargs)
            case "HDQN":
                return make_plots_for_hdqn(mdp, network, params, **kwargs)
            case "HDQN_HER_FOR_AUT":
                return make_plots_for_hdqn(mdp, network, params, **kwargs)
            case "QRM":
                return make_plots_for_hdqn(mdp, network, params, **kwargs)
            case "QRM_WITH_MULTIHEAD":
                return make_plots_for_hdqn(mdp, network, params, **kwargs)
            case "CRM":
                return make_plots_for_hdqn(mdp, network, params, **kwargs)
            case "CRM_RS":
                return make_plots_for_hdqn(mdp, network, params, **kwargs)
            case "CRM_WITH_MULTIHEAD":
                return make_plots_for_hdqn(mdp, network, params, **kwargs)
            case "DDPG":
                return make_plots_for_ddpg(mdp, make_policy, network, params, **kwargs)
            case _:
                raise NotImplementedError(f"{alg_name} not supported")
