import mlflow
import functools

import mediapy

import jax
import jax.numpy as jnp
from brax.io import model
from brax import envs

import achql.stl as stl

from achql.tasks.utils import get_task_by_name

from achql.brax.envs.wrappers.automaton_goal_conditioned_wrapper import partition
from achql.brax.training.evaluator_with_specification import EvaluatorWithSpecification

from achql.hierarchy.envs.options_wrapper import OptionsWrapper
from achql.hierarchy.training.evaluator import HierarchicalEvaluatorWithSpecification
from achql.hierarchy.training.lof_evaluator import LOFEvaluatorWithSpecification
from achql.visualization.utils import get_mdp_network_policy_and_params
from achql.visualization import hierarchy, flat


def get_group(env, spec, alg):
    runs = mlflow.search_runs(
        filter_string=f"tags.env = '{env}' and tags.spec = '{spec}' and tags.alg = '{alg}'"
    )

    return runs


def main():
    mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")
    
    for env, spec in [("SimpleMaze", "SimpleMazeLoopWithObs"),
                      ("SimpleMaze3D", "SimpleMaze3DLoopWithObs"),
                      # ("AntMaze", "AntMazeLoopWithObs")
                      ]:
        for alg in ["LOF", "CRM_RS", "ACHQL"]:

            if alg == "LOF":
                mlflow.set_experiment("proj2-lof-final-experiments")

                group = get_group(env, spec, alg)

                # use the five most recent (tentative)
                group = group.iloc[:5]

                run_ids = group["run_id"].tolist()

                print(spec, alg)
                print(group)

                loop_eval_for_lof_training_run_ids(run_ids)

            else:
                mlflow.set_experiment("proj2-final-experiments")
                group = get_group(env, spec, alg)
    
                # use the five most recent (tentative)
                group = group.iloc[:5]
    
                run_ids = group["run_id"].tolist()
    
                print(spec, alg)
                print(group)
    
                loop_eval_for_training_run_ids(run_ids)
    

def loop_eval_for_lof_training_run_ids(training_run_ids):
    rs = []
    srs = []
    for seed, training_run_id in enumerate(training_run_ids):
        num_loops, success_rate = lof_evaluate_for_run_and_seed(training_run_id, seed)
        print(f"{seed} - {num_loops} - {success_rate}")

        rs.append(num_loops)
        srs.append(success_rate)

    rs = jnp.array(rs)
    srs = jnp.array(srs)

    print(f"Num Loops: {rs.mean()} +- {rs.std()}")
    print(f"Success Rate: {srs.mean()} +- {srs.std()}")

    return rs, srs


def loop_eval_for_training_run_ids(training_run_ids):

    # # SimpleMaze:
    # training_run_ids = ["60381ad329ce430b89b42c77735ddb58",
    #                     "3283f5d0c5ca4ccaa117e6b392a6eea0",
    #                     "2b7114b0e259452a9cc8c8c20b43061f",
    #                     "837be13827d9485fa3ade7441040db48",
    #                     "2173dddaebf4450ca8daceb9bb745a20"]

    # # SimpleMaze3D:
    # training_run_ids = ["d4b21617668c435fad73dad337419876"]

    # # SimpleMaze3D:
    # training_run_ids = ["d4b21617668c435fad73dad337419876"]

    rs = []
    srs = []
    for seed, training_run_id in enumerate(training_run_ids):
        num_loops, success_rate = evaluate_for_run_and_seed(training_run_id, seed)
        print(f"{seed} - {num_loops} - {success_rate}")

        rs.append(num_loops)
        srs.append(success_rate)

    rs = jnp.array(rs)
    srs = jnp.array(srs)

    print(f"Num Loops: {rs.mean()} +- {rs.std()}")
    print(f"Success Rate: {srs.mean()} +- {srs.std()}")

    return rs, srs


def lof_evaluate_for_run_and_seed(training_run_id, seed):
    # process_id = jax.process_index()
    global_key, local_key = jax.random.split(jax.random.PRNGKey(seed))
    # local_key = jax.random.fold_in(local_key, process_id)
    _, _, _, eval_key = jax.random.split(local_key, 4)

    run = mlflow.get_run(run_id=training_run_id)

    task = get_task_by_name(run.data.tags["spec"])

    assert "Loop" in run.data.tags["spec"], "Expecting a training run for a loop task"

    mdp, logical_options, _, network, _, make_policy, params = get_mdp_network_policy_and_params(training_run_id)

    return lof_evaluate(task, mdp, run, training_run_id, make_policy, logical_options, eval_key)


def evaluate_for_run_and_seed(training_run_id, seed):
    # process_id = jax.process_index()
    global_key, local_key = jax.random.split(jax.random.PRNGKey(seed))
    # local_key = jax.random.fold_in(local_key, process_id)
    _, _, _, eval_key = jax.random.split(local_key, 4)

    run = mlflow.get_run(run_id=training_run_id)

    task = get_task_by_name(run.data.tags["spec"])

    assert "Loop" in run.data.tags["spec"], "Expecting a training run for a loop task"

    mdp, options, network, make_option_policy, make_policy, params = get_mdp_network_policy_and_params(training_run_id)

    return hierarchical_evaluate(task, mdp, run, training_run_id, make_option_policy, options, eval_key)
    # return flat_evaluate(task, mdp, training_run_id, make_policy, eval_key)


def hierarchical_evaluate(task, mdp, run, training_run_id, make_option_policy, options, eval_key):
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

    safety_phi = task.lo_spec.children[1]

    evaluator = HierarchicalEvaluatorWithSpecification(
            eval_env,
            functools.partial(make_option_policy, deterministic=True),
            options,
            specification=safety_phi,
            state_var=task.obs_var,
            num_eval_envs=16,
            episode_length=200,
            action_repeat=1,
            key=eval_key,
            return_states=True
    )

    def loop_counter(transitions):
        transitions = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), transitions)
        obs = transitions.observation[..., :mdp.original_obs_dim]

        # goal_aps, assuming that goal_ap[0] is the first point the robot reaches
        _, goal_aps = partition(lambda _, v: "goal" in v.info, mdp.automaton.aps)
        made_loop_phi = stl.STLAnd(goal_aps[0],
                                   stl.STLNext(stl.STLUntimedEventually(
                                       stl.STLAnd(goal_aps[1],
                                                  stl.STLNext(stl.STLUntimedEventually(
                                                      goal_aps[0]))))))

        ap_param_fields = {}
        if hasattr(mdp, "randomize_goals") and getattr(mdp, "randomize_goals", False):
            for _, ap in mdp.automaton.aps.items():
                i = 32 + ap.info["ap_i"]
                ap_param_fields[i] = eval_state.info["ap_params"][:, 0]

        robustness_traces = jax.vmap(made_loop_phi)({ task.obs_var.idx: obs,
                                                      # "action": data.action
                                                    } | ap_param_fields)

        def count_continuous_subsequences(xs):
            def f(carry, x):
                dist_since_positive, count = carry
        
                def increment_and_reset():
                    return (0, count + 1)
    
                def reset():
                    return (0, count)
    
                def move_on():
                    return (dist_since_positive + 1, count)
    
                return jax.lax.cond(
                    x > 0,
                    lambda: jax.lax.cond(dist_since_positive > 4,
                                         increment_and_reset,
                                         reset),
                    move_on), None
    
            (_, count), _ = jax.lax.scan(f, (10, 0), xs)

            return count

        num_loops = jax.vmap(count_continuous_subsequences)(robustness_traces)

        return num_loops


    return run_final_eval(evaluator, loop_counter, training_run_id, mdp)


def lof_evaluate(task, mdp, run, training_run_id, make_option_policy, logical_options, eval_key):

    wrap_for_training = envs.training.wrap

    eval_env = wrap_for_training(
        mdp,
        episode_length=1000,
        action_repeat=1,
    )

    safety_phi = task.lo_spec.children[1]

    evaluator = LOFEvaluatorWithSpecification(
            eval_env,
            functools.partial(make_option_policy, deterministic=True),
            logical_options,
            specification=safety_phi,
            state_var=task.obs_var,
            num_eval_envs=16,
            episode_length=200,
            action_repeat=1,
            key=eval_key,
            return_states=True
    )

    def loop_counter(transitions):
        transitions = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), transitions)
        obs = transitions.observation[..., :mdp.original_obs_dim]

        # goal_aps, assuming that goal_ap[0] is the first point the robot reaches
        _, goal_aps = partition(lambda _, v: "goal" in v.info, mdp.automaton.aps)
        made_loop_phi = stl.STLAnd(goal_aps[0],
                                   stl.STLNext(stl.STLUntimedEventually(
                                       stl.STLAnd(goal_aps[1],
                                                  stl.STLNext(stl.STLUntimedEventually(
                                                      goal_aps[0]))))))

        ap_param_fields = {}
        if hasattr(mdp, "randomize_goals") and getattr(mdp, "randomize_goals", False):
            for _, ap in mdp.automaton.aps.items():
                i = 32 + ap.info["ap_i"]
                ap_param_fields[i] = eval_state.info["ap_params"][:, 0]

        robustness_traces = jax.vmap(made_loop_phi)({ task.obs_var.idx: obs,
                                                      # "action": data.action
                                                    } | ap_param_fields)

        def count_continuous_subsequences(xs):
            def f(carry, x):
                dist_since_positive, count = carry

                def increment_and_reset():
                    return (0, count + 1)

                def reset():
                    return (0, count)

                def move_on():
                    return (dist_since_positive + 1, count)

                return jax.lax.cond(
                    x > 0,
                    lambda: jax.lax.cond(dist_since_positive > 4,
                                         increment_and_reset,
                                         reset),
                    move_on), None

            (_, count), _ = jax.lax.scan(f, (10, 0), xs)

            return count

        num_loops = jax.vmap(count_continuous_subsequences)(robustness_traces)

        return num_loops


    return run_final_eval(evaluator, loop_counter, training_run_id, mdp)



# def flat_evaluate(task, mdp, training_run_id, make_policy, eval_key):
#     eval_environment = mdp
#     wrap_for_training = envs.training.wrap
#     eval_env = wrap_for_training(
#             eval_environment,
#             episode_length=100,
#             action_repeat=1,
#     )

#     evaluator = EvaluatorWithSpecification(
#             eval_env,
#             functools.partial(make_policy, deterministic=True),
#             specification=task.lo_spec,
#             state_var=task.obs_var,
#             num_eval_envs=256,
#             episode_length=100,
#             action_repeat=1,
#             key=eval_key,
#             return_states=True
#     )

#     return run_final_eval(evaluator, training_run_id, mdp)


def run_final_eval(evaluator, loop_counter, training_run_id, mdp):
    metrics = {}
    logged_model_path = f'runs:/{training_run_id}/policy_params'
    real_path = mlflow.artifacts.download_artifacts(logged_model_path)
    params = model.load_params(real_path)

    metrics, eval_state, all_data = evaluator.run_evaluation(params, training_metrics={}, return_data = True)

    state, transitions = all_data
    num_loops = loop_counter(transitions)

    safety_robustness = eval_state.info["eval_metrics"].episode_metrics["robustness"]

    success_rate = jnp.logical_and(safety_robustness > 0, num_loops > 0).mean()

    # print("Success Rate: ", )

    # put the batch dim first
    # state = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), state)
    # num_rollouts, num_steps = state.obs.shape[:-1]

    # for i in range(16): # range(num_rollouts):
    #     if safety_robustness[i] > 0 and num_loops[i] > 0:
    #         print(f"visualizing for final rollout {i}")
    #         rollout_states = []
    #         for j in range(num_steps):
    #             ps = jax.tree.map(lambda x: x[i, j], state.pipeline_state)
    #             rollout_states.append(ps)

    #         mediapy.write_video(f'./eval-rollout-{i}.mp4',
    #                             mdp.render(rollout_states, camera='overview'),
    #                             fps=1.0 / mdp.dt)

    # return metrics, eval_state, all_data
    return num_loops, success_rate


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")

    results = main()
