import mlflow
import functools

import mediapy
import pandas as pd

import jax
import jax.numpy as jnp
from brax.io import model
from brax import envs

from achql.tasks.utils import get_task_by_name

from achql.brax.training.evaluator_with_specification import EvaluatorWithSpecification

from achql.hierarchy.envs.options_wrapper import OptionsWrapper
from achql.hierarchy.training.evaluator import HierarchicalEvaluatorWithSpecification
from achql.hierarchy.training.lof_evaluator import LOFEvaluatorWithSpecification
from achql.visualization.utils import get_mdp_network_policy_and_params
from achql.visualization import hierarchy, flat


def hierarchical_evaluate(task, mdp, training_run_id, make_option_policy, eval_key):
    eval_environment = OptionsWrapper(
        mdp,
        options,
        discounting=jnp.float32(run.data.params["discounting"]),
        use_sum_cost_critic=False,
    )

    wrap_for_training = envs.training.wrap

    eval_env = wrap_for_training(
            eval_environment,
            episode_length=1000,
            action_repeat=1,
    )

    evaluator = HierarchicalEvaluatorWithSpecification(
            eval_env,
            functools.partial(make_option_policy, deterministic=True),
            options,
            specification=task.lo_spec,
            state_var=task.obs_var,
            num_eval_envs=16,
            episode_length=1000,
            action_repeat=1,
            key=eval_key,
            return_states=True
    )

    return run_final_eval(evaluator, training_run_id, mdp)


def flat_evaluate(task, mdp, training_run_id, make_policy, eval_key):
    eval_environment = mdp
    wrap_for_training = envs.training.wrap
    eval_env = wrap_for_training(
            eval_environment,
            episode_length=100,
            action_repeat=1,
    )

    evaluator = EvaluatorWithSpecification(
            eval_env,
            functools.partial(make_policy, deterministic=True),
            specification=task.lo_spec,
            state_var=task.obs_var,
            num_eval_envs=256,
            episode_length=100,
            action_repeat=1,
            key=eval_key,
            return_states=True
    )

    return run_evals(evaluator, training_run_id, mdp)
    # return run_final_eval(evaluator, training_run_id, mdp)


def lof_evaluate(task, mdp, run, training_run_id, make_option_policy, logical_options, eval_key):
    eval_environment = mdp
    wrap_for_training = envs.training.wrap

    episode_length = task.achql_hps["episode_length"]

    eval_env = wrap_for_training(
            eval_environment,
            episode_length=episode_length,
            action_repeat=1,
    )

    evaluator = LOFEvaluatorWithSpecification(
            eval_env,
            functools.partial(make_option_policy, deterministic=True),
            logical_options,
            specification=task.lo_spec,
            state_var=task.obs_var,
            num_eval_envs=16,
            episode_length=episode_length,
            action_repeat=1,
            key=eval_key,
            return_states=True
    )

    return run_final_eval(evaluator, training_run_id, mdp)


# def run_evals(evaluator, training_run_id):
#     metrics = {}
#     for step in ["000000000",
#                  "000380928",
#                  "000492032",
#                  "000603136",
#                  "000714240",
#                  "000825344",
#                  "000936448",
#                  "001047552",
#                  "001158656",
#                  "001269760",
#                  "001380864",
#                  "001491968",
#                  "001603072",
#                  "001714176",
#                  "001825280",
#                  "001936384",
#                  "002047488",
#                  "002158592",
#                  "002269696",
#                  "002380800",
#                  "002491904",
#                  "002603008",
#                  "002714112",
#                  "002825216",
#                  "002936320",
#                  "003047424",
#                  "003158528",
#                  "003269632",
#                  "003380736",
#                  "003491840",
#                  "003602944",
#                  "003714048",
#                  "003825152",
#                  "003936256",
#                  "004047360",
#                  "004158464",
#                  "004269568",
#                  "004380672",
#                  "004491776",
#                  "004602880",
#                  "004713984",
#                  "004825088",
#                  "004936192",
#                  "005047296",
#                  "005158400",
#                  "005269504",
#                  "005380608",
#                  "005491712",
#                  "005602816",
#                  "005713920"]:
#         logged_model_path = f'runs:/{training_run_id}/policy_params_{step}'
#         real_path = mlflow.artifacts.download_artifacts(logged_model_path)
#         params = model.load_params(real_path)
#         metrics, eval_state, data = evaluator.run_evaluation(params, training_metrics={}, return_data = True)
#         print(step, metrics["eval/episode_reward"])


def run_final_eval(evaluator, training_run_id, mdp):
    metrics = {}
    logged_model_path = f'runs:/{training_run_id}/policy_params'
    real_path = mlflow.artifacts.download_artifacts(logged_model_path)
    params = model.load_params(real_path)
    metrics = evaluator.run_evaluation(params, training_metrics={}, return_data = False)
    print(metrics["eval/episode_reward"])
    return metrics

def run_evals(evaluator, training_run_id, mdp):
    metrics = {}
    for step in ["000000000",
                 "000153600",
                 "000268800",
                 "000384000",
                 "000499200",
                 "000614400",
                 "000729600",
                 "000844800",
                 "000960000",
                 "001075200",
                 "001190400",
                 "001305600",
                 "001420800",
                 "001536000",
                 "001651200",
                 "001766400",
                 "001881600",
                 "001996800",
                 "002112000",
                 "002227200",
                 "002342400",
                 "002457600",
                 "002572800",
                 "002688000",
                 "002803200",
                 "002918400",
                 "003033600",
                 "003148800",
                 "003264000",
                 "003379200",
                 "003494400",
                 "003609600",
                 "003724800",
                 "003840000",
                 "003955200",
                 "004070400",
                 "004185600",
                 "004300800",
                 "004416000",
                 "004531200",
                 "004646400",
                 "004761600",
                 "004876800",
                 "004992000",
                 "005107200",
                 "005222400",
                 "005337600",
                 "005452800",
                 "005568000",
                 "005683200"]:
        logged_model_path = f'runs:/{training_run_id}/policy_params_{step}'
        real_path = mlflow.artifacts.download_artifacts(logged_model_path)
        params = model.load_params(real_path)
        params = (params[0], params[3])
        metrics = evaluator.run_evaluation(params, training_metrics={}, return_data = False)
        print(step, metrics["eval/episode_reward"])

    # results = []
    # for step in ["003047424",
    #              "003158528",
    #              "003269632"]:
    #     logged_model_path = f'runs:/{training_run_id}/policy_params_{step}'
    #     real_path = mlflow.artifacts.download_artifacts(logged_model_path)
    #     params = model.load_params(real_path)

    #     metrics, eval_state, all_data = evaluator.run_evaluation(params, training_metrics={}, return_data = True)

    #     print(step, metrics["eval/episode_reward"])

    #     state, _ = all_data
    #     # put the batch dim first
    #     state = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), state)

    #     num_rollouts, num_steps = state.obs.shape[:-1]

    #     for i in range(5): # range(num_rollouts):
    #         print(f"visualizing for {step} rollout {i}")
    #         rollout_states = []
    #         for j in range(num_steps):
    #             ps = jax.tree.map(lambda x: x[i, j], state.pipeline_state)
    #             rollout_states.append(ps)

    #         mediapy.write_video(f'./eval-rollout-{step}-{i}.mp4',
    #                             mdp.render(rollout_states, camera='overview'),
    #                             fps=1.0 / mdp.dt)

def eval_for_training_run_id(training_run_id):
    process_id = jax.process_index()
    seed = 0
    global_key, local_key = jax.random.split(jax.random.PRNGKey(seed))
    local_key = jax.random.fold_in(local_key, process_id)
    _, _, _, eval_key = jax.random.split(local_key, 4)

    run = mlflow.get_run(run_id=training_run_id)

    task = get_task_by_name(run.data.tags["spec"])

    if (alg := run.data.tags.get("alg", None)) == "LOF":
        mdp, logical_options, _, network, _, make_policy, params = get_mdp_network_policy_and_params(training_run_id)
        return lof_evaluate(task, mdp, run, training_run_id, make_policy, logical_options, eval_key)
    else:
        mdp, options, network, make_option_policy, make_policy, params = get_mdp_network_policy_and_params(training_run_id)
        # return hierarchical_evaluate(task, mdp, training_run_id, make_option_policy, eval_key)
        return flat_evaluate(task, mdp, training_run_id, make_policy, eval_key)


def main():

    simple_maze_3d = {
        "SimpleMaze3DTwoSubgoals": ["cc278d824b614e95a5ef34a88a9f2842", "04c4dee753ad47649f9f4b472fab22b2", "2e71f57cdc2647768560fb7dfc72b057", "6ddccad9a7d24239bd8c43c60ca28bba", "83352e0a4a4743c5a9de1b6b64511658"],
        "SimpleMaze3DBranching1": ["68d955389175492591f613ebdc318524", "13bef1a566804248a76413717bcbfb4e", "35d0e79f8da84235ae5417412e9b8474", "e6a9b750f9ed426b9baa913acc8a3af2", "dffa0a199e0c4b2083496a55422c35d2"],
        "SimpleMaze3DObligationConstraint2": ["98ecb3ff52274d0d99c69242b672dd47", "7d05a67ad54b4b158b3403322e005d92", "86bbb4f1f6e14e6c96b29b88d3be4518", "11513461a8d44655b4a2ab43609702ea", "ab50ebac5e8b43bfbf4abe913796541c"]
    }

    ant_maze_ids = {
        "AntMazeTwoSubgoals": ['b3af5d20b06146db8e4f0ced184cc1b5', '9c056fe4ce5f4cc5b1b9ce59dedc1d08', '7efbb24d8bf742d281dbe428b177c8f9', 'c1ef77710bdf49029ce20da05623268a', '5258b4f3cbdd4df8b5ad02cdb802939f'],
        "AntMazeBranching1": ['8bb8f95f531348a1b6d2dd98b26b0bb7', '5f2a8c5e1c6247779ae77487c8e553df', 'd45e26d2aefc41c9b71f41789bfd8a82', '04d10a573a3740aa9d9a9abfa49a1245', 'e009c1b3494f432ea57f23e3702ffaf0'],
        "AntMazeObligationConstraint3": ['3a838f3303ee4638ba6a380decb9ecbd', 'ec35adf6cf8746ad8e5c57f76273471a', '819df827e8044fee926fb2c80056c4e7', 'ebb27593f2064450921d45a472236fa3', 'baf955a49dee4a64803f800cd5680e3f'],
        "AntMazeUntil1": ['0f6c37a7bb1144f29282b920dbdcae9a', 'f4838147c07341abb1b736beabee6f7c', 'e763864d78804d5b81a5d239a861586d', 'd8e8e4d53ad048bd9fa13d9f96166537', '498af3bb6fc04035a63d143badbf9039'],
        "AntMazeLoopWithObs": ['b338cfc4065948eba2204f1899ec5e54', 'e318f0049d504984856b1b1ff342cd40', 'f647cff95248445283cb2dd0f6297955', '9dd40e5b8bda4a9fb016a90e6bfbb99a', '8be20d3c606e45958cd1ce9174615762'],
    }

    for spec, ids in ant_maze_ids.items():
        reward_l = []
        success_rate_l = []
        for run_id in ids:
            metrics = eval_for_training_run_id(run_id)
            reward = metrics["eval/episode_reward"].item()
            success_rate = metrics["eval/proportion_robustness_over_zero"].item()

            reward_l.append(reward)
            success_rate_l.append(success_rate)

        print("reward_l", reward_l)
        print("success_rate_l", success_rate_l)

        reward_l = pd.Series(reward_l)
        reward_mean = reward_l.mean().round(decimals=1)
        reward_std = reward_l.std().round(decimals=1)
        print(spec, "success", f"{reward_mean} +- {reward_std}")

        success_rate_l = pd.Series(success_rate_l) * 100.0
        success_rate_mean = success_rate_l.mean().round(decimals=1)
        success_rate_std = success_rate_l.std().round(decimals=1)
        print(spec, "good rho", f"{success_rate_mean} +- {success_rate_std}")
            


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")

    results = main()
