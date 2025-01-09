import functools
import jax
import os
from typing import Optional

from datetime import datetime
from jax import numpy as jnp
import matplotlib.pyplot as plt

import mlflow

import brax

import flax
from brax import envs
from brax.io import model
from brax.io import json
from brax.io import html
from jaxgcrl.src import train as crl

from task_aware_skill_composition.brax.agents.ppo import train as ppo
from task_aware_skill_composition.brax.agents.tasc import train as tasc
from task_aware_skill_composition.brax.agents.dscrl import train as dscrl
from task_aware_skill_composition.brax.agents.sac_her import train as sac_her
from task_aware_skill_composition.brax.agents.ppo_with_spec_rewards import train as ppo_spec
from task_aware_skill_composition.brax.agents.ppo_option_critic import train as ppo_option_critic
from task_aware_skill_composition.brax.agents.hsac import train as soac
from task_aware_skill_composition.brax.agents.hdqn import train as hdqn
from task_aware_skill_composition.brax.agents.hdqn_her import train as hdqn_her
from task_aware_skill_composition.brax.agents.hdqn_automaton_her import train as hdqn_automaton_her
from task_aware_skill_composition.brax.agents.hdcqn_her import train as hdcqn_her
from task_aware_skill_composition.brax.agents.hdcqn_automaton_her import train as hdcqn_automaton_her
from brax.training.agents.apg import train as apg

from task_aware_skill_composition.baselines.reward_machines.crm import train as crm
# from task_aware_skill_composition.baselines.reward_machines.qrm import train as qrm
from task_aware_skill_composition.baselines.reward_machines.hrm import train as hrm
from task_aware_skill_composition.baselines.reward_machines.reward_shaping import train as rm_reward_shaping

# tasks
from task_aware_skill_composition.brax.tasks import get_task
from task_aware_skill_composition.brax.utils import (
    make_cmdp, make_aut_goal_cmdp, make_shaped_reward_mdp, make_shaped_reward_mdp2,
    make_aut_mdp,
    make_reward_machine_mdp,
    make_aut_goal_mdp,
)

from task_aware_skill_composition.visualization.critic import (
    make_plots_for_hdqn,
    make_plots_for_hdcqn,
    make_plots_for_hdcqn_aut,
)



mlflow.set_tracking_uri(f"file:///home/tassos/.local/share/mlflow")
mlflow.set_experiment("proj2-batch-training")


def progress_fn(num_steps, metrics, **kwargs):
    print(f"Logging for {num_steps}")
    print(metrics)
    mlflow.log_metrics(metrics, step=num_steps)
    
def training_run(run_id, env, seed, train_fn=ppo.train, progress_fn=progress_fn, hyperparameters={}, extras={}):
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

    return make_inference_fn, params


def train_for_all(envs, tasks, func, alg_tag):
    for env_name in envs:
        for task_name in tasks:
            task = get_task(env_name, task_name)

            spec = task.lo_spec
            spec_tag = type(task).__name__
            env_tag = type(task.env).__name__
            options = task.get_options()

            for seed in range(0, 3):
                with mlflow.start_run(tags={"env": env_tag, "spec": spec_tag, "alg": alg_tag}) as run:
                    func(run, task, seed, options, spec)


def hdcqn_automaton_her_train(run, task, seed, options, spec):
    make_inference_fn, params = training_run(
        run.info.run_id,
        make_aut_goal_cmdp(task),
        seed,
        train_fn=hdcqn_automaton_her.train,
        hyperparameters=task.hdcqn_her_hps,
        extras={
            "eval_env": make_aut_goal_cmdp(task, randomize_goals=False),
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
        }
    )                    

def crm_train(run, task, seed, options, spec):
    make_inference_fn, params = training_run(
        run.info.run_id,
        make_reward_machine_mdp(task),
        seed,
        train_fn=crm.train,
        hyperparameters=task.crm_hps,
        extras={
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
        }
    )


train_for_all(["ant_maze", "simple_maze_3d", "simple_maze"], ["two_subgoals", "branching1", "obligation2"], hdcqn_automaton_her_train, "HDCQN_AUTOMATON_HER")
train_for_all(["ant_maze", "simple_maze_3d", "simple_maze"], ["two_subgoals", "branching1", "obligation2"], crm_train, "CRM")
