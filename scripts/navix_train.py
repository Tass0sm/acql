import functools
import jax
import os
from typing import Optional

from datetime import datetime
from jax import numpy as jnp
import matplotlib.pyplot as plt

import mlflow

from brax.io import model

from achql.navix.agents.tabular_hql import train as tabular_hql
from achql.navix.agents.tabular_achql import train as tabular_achql

from achql.tasks.utils import get_task
from achql.navix.utils import make_cmdp


mlflow.set_tracking_uri(f"file:///home/tassos/.local/share/mlflow")
mlflow.set_experiment("proj2-random-argmax-testing")


def progress_fn(num_steps, metrics, **kwargs):
    print(f"Logging for {num_steps}")
    print(metrics)
    mlflow.log_metrics(metrics, step=num_steps)


def training_run(run_id, env, seed, train_fn=tabular_achql.train, progress_fn=progress_fn, hyperparameters={}, extras={}):
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

    with mlflow.MlflowClient()._log_artifact_helper(run_id, f'policy_params') as tmp_path:
        model.save_params(tmp_path, params)

    print("Q:\n", params)

    return make_inference_fn, params


def train_for_all(envs, tasks, func, alg_tag, seed_range=(0, 3)):
    for env_name in envs:
        for task_name in tasks:
            task = get_task(env_name, task_name)

            spec = task.lo_spec
            spec_tag = type(task).__name__
            env_tag = type(task.env).__name__

            for seed in range(*seed_range):
                with mlflow.start_run(tags={"env": env_tag, "spec": spec_tag, "alg": alg_tag}) as run:
                    func(run, task, seed, spec)


def tabular_hql_train(run, task, seed, spec):
    options = task.get_options()

    make_inference_fn, params = training_run(
        run.info.run_id,
        task.env,
        seed,
        train_fn=tabular_hql.train,
        progress_fn=progress_fn,
        hyperparameters=task.tabular_hql_hps,
        extras={
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
        }
    )


def tabular_achql_train(run, task, seed, spec, margin=0.0):
    options = task.get_options()

    make_inference_fn, params = training_run(
        run.info.run_id,
        make_cmdp(task, margin=margin),
        seed,
        train_fn=tabular_achql.train,
        progress_fn=progress_fn,
        hyperparameters=task.tabular_achql_hps,
        extras={
            # "eval_env": make_cmdp(task, margin=margin),
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
        }
    )


if __name__ == "__main__":
    task = get_task("Grid", "Random8x8")

    spec = task.lo_spec
    spec_tag = type(task).__name__
    env_tag = type(task.env).__name__

    for seed in range(0, 1):
        # with mlflow.start_run(tags={"env": env_tag, "spec": spec_tag, "alg": "TABULAR_ACHQL"}) as run:
        #     tabular_achql_train(run, task, seed, spec)

        with mlflow.start_run(tags={"env": env_tag, "spec": spec_tag, "alg": "TABULAR_HQL"}) as run:
            tabular_hql_train(run, task, seed, spec)
