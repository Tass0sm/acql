import mlflow
import functools

import mediapy

import jax
import jax.numpy as jnp
from brax.io import model
from brax import envs

from achql.tasks.utils import get_task_by_name

from achql.brax.training.evaluator_with_specification import EvaluatorWithSpecification

from achql.hierarchy.envs.options_wrapper import OptionsWrapper
from achql.hierarchy.training.evaluator import HierarchicalEvaluatorWithSpecification
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


def main():
    process_id = jax.process_index()
    seed = 0
    global_key, local_key = jax.random.split(jax.random.PRNGKey(seed))
    local_key = jax.random.fold_in(local_key, process_id)
    _, _, _, eval_key = jax.random.split(local_key, 4)

    training_run_id = "1d9eb97f787549ab9bb35690dad6644d"
    run = mlflow.get_run(run_id=training_run_id)

    task = get_task_by_name(run.data.tags["spec"])

    mdp, options, network, make_option_policy, make_policy, params = get_mdp_network_policy_and_params(training_run_id)

    # return hierarchical_evaluate(task, mdp, training_run_id, make_option_policy, eval_key)
    return flat_evaluate(task, mdp, training_run_id, make_policy, eval_key)


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")
    mlflow.set_experiment("proj2-notebook")

    results = main()
