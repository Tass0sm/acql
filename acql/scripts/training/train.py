import functools
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt

import mlflow

import jax.numpy as jnp
from brax.io import model

from acql.brax.agents.ppo import train as ppo
from acql.brax.agents.achql import train as achql
from acql.brax.agents.sac import train as sac
from acql.brax.agents.ddpg import train as ddpg
from acql.brax.agents.hdqn import train as hdqn
from acql.brax.agents.hdqn_her import train as hdqn_her
from acql.baselines.reward_machines.qrm import train as qrm
from acql.baselines.reward_machines.qrm_ddpg import train as qrm_ddpg
from acql.baselines.reward_machines.crm import train as crm

from acql.tasks import get_task
from acql.brax.utils import make_aut_goal_cmdp, make_reward_machine_mdp

from acql.visualization.plots import make_plots_for_achql, make_plots_for_3d_achql


def progress_fn(num_steps, metrics, *args, **kwargs):
    print(f"Logging for {num_steps}")
    print(metrics)
    mlflow.log_metrics(metrics, step=num_steps)


def progress_fn_with_figs(num_steps, metrics, *args, **kwargs):
    print(f"Logging for {num_steps}")
    print(metrics)
    mlflow.log_metrics(metrics, step=num_steps)

    env = kwargs["env"]
    make_policy = kwargs["make_policy"]
    network = kwargs["network"]
    params = kwargs["params"]

    plots_1 = make_plots_for_3d_achql(env, make_policy, network, params,
                                      grid_size=50)

    # plots_2 = make_plots_for_achql(env, make_policy, network, params,
    #                                grid_size=50,
    #                                tmp_state_fn=lambda x: x.replace(obs=x.obs.at[-6:].set(jnp.array([4., 4., 0., 0., 1., 0.]))))

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir, f"value_function_state1_{num_steps:09}.png")
        plots_1[0][0].savefig(path)
        mlflow.log_artifact(path)
        path = Path(tmp_dir, f"safety_function_state1_{num_steps:09}.png")
        plots_1[1][0].savefig(path)
        mlflow.log_artifact(path)

    # with tempfile.TemporaryDirectory() as tmp_dir:
    #     path = Path(tmp_dir, f"value_function_state2_{num_steps:09}.png")
    #     plots_2[0][0].savefig(path)
    #     mlflow.log_artifact(path)
    #     path = Path(tmp_dir, f"safety_function_state2_{num_steps:09}.png")
    #     plots_2[1][0].savefig(path)
    #     mlflow.log_artifact(path)

    for plot in plots_1:
        plt.close(plot[0])
        del plot

    # for plot in plots_2:
    #     plt.close(plot[0])
    #     del plot

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir, f"policy_params_{num_steps:09}")
        model.save_params(path, params)
        mlflow.log_artifact(path)


def progress_fn_with_saving(num_steps, metrics, *args, **kwargs):
    print(f"Logging for {num_steps}")
    print(metrics)
    mlflow.log_metrics(metrics, step=num_steps)

    env = kwargs["env"]
    make_policy = kwargs["make_policy"]
    network = kwargs["network"]
    params = kwargs["params"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir, f"policy_params_{num_steps:09}")
        model.save_params(path, params)
        mlflow.log_artifact(path)


def training_run(run_id, env, seed, train_fn, progress_fn=progress_fn, hyperparameters={}, extras={}):
    hyperparameters = {
        **hyperparameters,
        "seed": seed,
    }

    mlflow.log_params(hyperparameters)

    train_fn = functools.partial(train_fn, **hyperparameters)

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=progress_fn,
        seed=seed,
        **extras
    )

    with mlflow.MlflowClient()._log_artifact_helper(run_id, 'policy_params') as tmp_path:
        model.save_params(tmp_path, params)

    return make_inference_fn, params


def train_for_all(envs, tasks, func, alg_tag, seed_range=(0, 3), extra_tasks={}, **kwargs):
    for env_name in envs:
        for task_name in tasks:
            task = get_task(env_name, task_name, extra_tasks=extra_tasks)

            spec = task.lo_spec
            spec_tag = type(task).__name__
            env_tag = type(task.env).__name__

            for seed in range(*seed_range):
                with mlflow.start_run(tags={"env": env_tag, "spec": spec_tag, "alg": alg_tag}) as run:
                    func(run, task, seed, spec, **kwargs)


def ddpg_train(run, task, seed, spec, reward_shaping=False):
    make_inference_fn, params = training_run(
        run.info.run_id,
        task.env,
        seed,
        train_fn=ddpg.train,
        hyperparameters=task.ddpg_hps,
        extras={
            "specification": spec,
            "state_var": task.obs_var,
        }
    )


def ppo_train(run, task, seed, spec):
    make_inference_fn, params = training_run(
        run.info.run_id,
        task.env,
        seed,
        train_fn=ppo.train,
        hyperparameters=task.ppo_hps,
        extras={
            "specification": spec,
            "state_var": task.obs_var,
        }
    )


def hdqn_train(run, task, seed, spec):
    options = task.get_options()

    make_inference_fn, params = training_run(
        run.info.run_id,
        task.env,
        seed,
        train_fn=hdqn.train,
        hyperparameters=task.hdqn_hps,
        extras={
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
        }
    )


def hdqn_her_train(run, task, seed, spec):
    options = task.get_options()

    make_inference_fn, params = training_run(
        run.info.run_id,
        task.env,
        seed,
        train_fn=hdqn_her.train,
        hyperparameters=task.hdqn_her_hps,
        extras={
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
        }
    )


def qrm_train(run, task, seed, spec, reward_shaping=False):
    options = task.get_options()

    make_inference_fn, params = training_run(
        run.info.run_id,
        make_reward_machine_mdp(task, reward_shaping=reward_shaping),
        seed,
        train_fn=qrm.train,
        hyperparameters=task.crm_hps,
        extras={
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
        }
    )


def qrm_ddpg_train(run, task, seed, spec, reward_shaping=False):
    make_inference_fn, params = training_run(
        run.info.run_id,
        make_reward_machine_mdp(task, reward_shaping=reward_shaping),
        seed,
        train_fn=qrm_ddpg.train,
        hyperparameters=task.crm_hps,
        extras={
            "specification": spec,
            "state_var": task.obs_var,
        }
    )


def crm_train(run, task, seed, spec, reward_shaping=False):
    options = task.get_options()

    make_inference_fn, params = training_run(
        run.info.run_id,
        make_reward_machine_mdp(task, reward_shaping=reward_shaping),
        seed,
        train_fn=crm.train,
        hyperparameters=task.crm_hps,
        extras={
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
            "eval_environment": make_reward_machine_mdp(task, reward_shaping=False),
        }
    )


def achql_train(run, task, seed, spec, margin=1.0):
    options = task.get_options()

    return training_run(
        run.info.run_id,
        make_aut_goal_cmdp(task, margin=margin, randomize_goals=task.achql_hps["use_her"]),
        seed,
        train_fn=achql.train,
        # progress_fn=progress_fn_with_figs,
        hyperparameters=task.achql_hps | { "network_type": "old_multihead" },
        extras={
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
            "eval_environment": make_aut_goal_cmdp(task, margin=margin, randomize_goals=False),
        }
    )


def sac_train(run, task, seed, spec, margin=1.0):
    return training_run(
        run.info.run_id,
        task.env,
        seed,
        train_fn=sac.train,
        # progress_fn=progress_fn_with_figs,
        hyperparameters=task.sac_hps,
        extras={
            # "specification": spec,
            # "state_var": task.obs_var,
        }
    )


def all_achql_runs(max_seed=5):
    train_for_all(["SimpleMaze"], ["TwoSubgoals"], achql_train, "ACHQL", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze"], ["Branching1"], achql_train, "ACHQL", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze"], ["ObligationConstraint1"], achql_train, "ACHQL", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze"], ["Until2"], achql_train, "ACHQL", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze"], ["LoopWithObs"], achql_train, "ACHQL", seed_range=(0, max_seed))

    train_for_all(["SimpleMaze3D"], ["TwoSubgoals"], achql_train, "ACHQL", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze3D"], ["Branching1"], achql_train, "ACHQL", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze3D"], ["ObligationConstraint2"], achql_train, "ACHQL", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze3D"], ["Until2"], achql_train, "ACHQL", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze3D"], ["LoopWithObs"], achql_train, "ACHQL", seed_range=(0, max_seed))

    train_for_all(["AntMaze"], ["TwoSubgoals"], achql_train, "ACHQL", seed_range=(0, max_seed))
    train_for_all(["AntMaze"], ["Branching1"], achql_train, "ACHQL", seed_range=(0, max_seed))
    train_for_all(["AntMaze"], ["ObligationConstraint3"], achql_train, "ACHQL", seed_range=(0, max_seed))
    train_for_all(["AntMaze"], ["Until2"], achql_train, "ACHQL", seed_range=(0, max_seed))
    train_for_all(["AntMaze"], ["LoopWithObs"], achql_train, "ACHQL", seed_range=(0, max_seed))


def all_crm_rs_runs(max_seed=5):
    crm_rs_train = functools.partial(crm_train, reward_shaping=True)

    train_for_all(["SimpleMaze"], ["TwoSubgoals"], crm_rs_train, "CRM_RS", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze"], ["Branching1"], crm_rs_train, "CRM_RS", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze"], ["ObligationConstraint1"], crm_rs_train, "CRM_RS", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze"], ["Until2"], crm_rs_train, "CRM_RS", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze"], ["LoopWithObs"], crm_rs_train, "CRM_RS", seed_range=(0, max_seed))

    train_for_all(["SimpleMaze3D"], ["TwoSubgoals"], crm_rs_train, "CRM_RS", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze3D"], ["Branching1"], crm_rs_train, "CRM_RS", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze3D"], ["ObligationConstraint2"], crm_rs_train, "CRM_RS", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze3D"], ["Until2"], crm_rs_train, "CRM_RS", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze3D"], ["LoopWithObs"], crm_rs_train, "CRM_RS", seed_range=(0, max_seed))

    train_for_all(["AntMaze"], ["TwoSubgoals"], crm_rs_train, "CRM_RS", seed_range=(0, max_seed))
    train_for_all(["AntMaze"], ["Branching1"], crm_rs_train, "CRM_RS", seed_range=(0, max_seed))
    train_for_all(["AntMaze"], ["ObligationConstraint3"], crm_rs_train, "CRM_RS", seed_range=(0, max_seed))
    train_for_all(["AntMaze"], ["Until2"], crm_rs_train, "CRM_RS", seed_range=(0, max_seed))
    train_for_all(["AntMaze"], ["LoopWithObs"], crm_rs_train, "CRM_RS", seed_range=(0, max_seed))


def main():
    all_achql_runs(max_seed=5)
    all_crm_rs_runs(max_seed=5)


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")
    mlflow.set_experiment("proj2-final-experiments")

    main()
