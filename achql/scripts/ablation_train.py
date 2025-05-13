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

from achql.brax.agents.ppo import train as ppo
from achql.brax.agents.tasc import train as tasc
from achql.brax.agents.dscrl import train as dscrl
from achql.brax.agents.sac_her import train as sac_her
from achql.brax.agents.ppo_with_spec_rewards import train as ppo_spec
from achql.brax.agents.ppo_option_critic import train as ppo_option_critic
from achql.brax.agents.hsac import train as soac
from achql.brax.agents.hdqn import train as hdqn
from achql.brax.agents.hdqn_her import train as hdqn_her
from achql.brax.agents.hdqn_automaton_her import train as hdqn_automaton_her
from achql.brax.agents.hdcqn_her import train as hdcqn_her
from achql.brax.agents.hdcqn_automaton_her import train as hdcqn_automaton_her
from brax.training.agents.apg import train as apg

# tasks
from achql.brax.tasks import get_task
from achql.brax.utils import (
    make_cmdp, make_aut_goal_cmdp, make_shaped_reward_mdp, make_shaped_reward_mdp2,
    make_aut_mdp,
    make_reward_machine_mdp,
    make_aut_goal_mdp,
)

from achql.visualization.critic import (
    make_plots_for_hdqn,
    make_plots_for_hdcqn,
    make_plots_for_hdcqn_aut,
)


def achql_ablation_one_train(run, task, seed, spec, margin=1.0):
    options = task.get_options()

    return training_run(
        run.info.run_id,
        make_aut_goal_cmdp(task, margin=margin, randomize_goals=task.achql_hps["use_her"]),
        seed,
        train_fn=achql.train,
        hyperparameters=task.achql_hps,
        extras={
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
            "eval_environment": make_aut_goal_cmdp(task, margin=margin, randomize_goals=False),
        }
    )


def achql_ablation_two_train(run, task, seed, spec, margin=1.0):
    options = task.get_options()

    return training_run(
        run.info.run_id,
        make_aut_goal_cmdp(task, margin=margin, randomize_goals=False, relu_cost=True),
        seed,
        train_fn=achql.train,
        hyperparameters=task.achql_hps | { "use_her": False },
        extras={
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
            "eval_environment": make_aut_goal_cmdp(task, margin=margin, randomize_goals=False, relu_cost=True),
            "use_sum_cost_critic": True
        }
    )


def all_achql_ablation_one_runs(max_seed=5):
    train_for_all(["SimpleMaze"], ["TwoSubgoals"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze"], ["Branching1"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze"], ["ObligationConstraint1"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze"], ["Until2"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze"], ["LoopWithObs"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))

    train_for_all(["SimpleMaze3D"], ["TwoSubgoals"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze3D"], ["Branching1"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze3D"], ["ObligationConstraint2"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze3D"], ["Until2"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze3D"], ["LoopWithObs"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))

    train_for_all(["AntMaze"], ["TwoSubgoals"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    train_for_all(["AntMaze"], ["Branching1"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    train_for_all(["AntMaze"], ["ObligationConstraint3"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    train_for_all(["AntMaze"], ["Until2"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    train_for_all(["AntMaze"], ["LoopWithObs"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))


def all_achql_ablation_two_runs(max_seed=5):
    train_for_all(["SimpleMaze"], ["ObligationConstraint1"], achql_ablation_two_train, "ACHQL_ABLATION_TWO", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze"], ["Until2"], achql_ablation_two_train, "ACHQL_ABLATION_TWO", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze"], ["LoopWithObs"], achql_ablation_two_train, "ACHQL_ABLATION_TWO", seed_range=(0, max_seed))

    train_for_all(["SimpleMaze3D"], ["ObligationConstraint2"], achql_ablation_two_train, "ACHQL_ABLATION_TWO", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze3D"], ["Until2"], achql_ablation_two_train, "ACHQL_ABLATION_TWO", seed_range=(0, max_seed))
    train_for_all(["SimpleMaze3D"], ["LoopWithObs"], achql_ablation_two_train, "ACHQL_ABLATION_TWO", seed_range=(0, max_seed))

    train_for_all(["AntMaze"], ["ObligationConstraint3"], achql_ablation_two_train, "ACHQL_ABLATION_TWO", seed_range=(0, max_seed))
    train_for_all(["AntMaze"], ["Until2"], achql_ablation_two_train, "ACHQL_ABLATION_TWO", seed_range=(0, max_seed))
    train_for_all(["AntMaze"], ["LoopWithObs"], achql_ablation_two_train, "ACHQL_ABLATION_TWO", seed_range=(0, max_seed))



def main():
    all_achql_runs()


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")
    # mlflow.set_experiment("proj2-batch-training")
    mlflow.set_experiment("proj2-ablation-experiments")

    main()
