import functools
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt

import mlflow

import jax.numpy as jnp
from brax.io import model

from jaxgcrl.utils.config import RunConfig

from acql.brax.agents.ppo import train as ppo
from acql.brax.agents.achql import train as achql
from acql.brax.agents.acddpg import train as acddpg
from acql.brax.agents.sac_her import train as sac_her
from acql.brax.agents.sac import train as sac
from acql.brax.agents.crl import train as crl
from acql.brax.agents.ddpg import train as ddpg
from acql.brax.agents.ddpg_her import train as ddpg_her
from acql.brax.agents.hdqn import train as hdqn
from acql.brax.agents.hdqn_her import train as hdqn_her
from acql.baselines.reward_machines.qrm import train as qrm
from acql.baselines.reward_machines.qrm_ddpg import train as qrm_ddpg
from acql.baselines.reward_machines.crm import train as crm

from acql.tasks import get_task
from acql.brax.utils import make_aut_goal_cmdp, make_reward_machine_mdp
from acql.scripts.train import train_for_all, training_run

from acql.visualization.plots import make_plots_for_achql, make_plots_for_3d_achql



def achql_train(run, task, seed, spec, margin=1.0, 
                option_length=None,
                episode_length=None,
                multiplier_num_sgd_steps=None):
    options = task.get_options(length=option_length)

    return training_run(
        run.info.run_id,
        make_aut_goal_cmdp(task, margin=margin, randomize_goals=task.achql_hps["use_her"]),
        seed,
        train_fn=achql.train,
        # progress_fn=progress_fn_with_figs,
        hyperparameters=task.achql_hps | { "episode_length": episode_length,
                                           "multiplier_num_sgd_steps": multiplier_num_sgd_steps },
        extras={
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
            "eval_environment": make_aut_goal_cmdp(task, margin=margin, randomize_goals=False),
        }
    )

def all_achql_runs(max_seed=5):

    # option_length_l = [5, 10]
    # episode_length_l = [200, 100]
    # multiplier_num_sgd_steps_l = [1, 4]

    option_length_l = [10]
    episode_length_l = [100]
    multiplier_num_sgd_steps_l = [1, 4]

    for option_length, episode_length in zip(option_length_l, episode_length_l):
        for multiplier_num_sgd_steps in multiplier_num_sgd_steps_l:
            train_for_all(["SimpleMaze3D"], ["TwoSubgoals"], achql_train, "ACHQL", seed_range=(0, max_seed),
                          option_length=option_length,
                          episode_length=episode_length,
                          multiplier_num_sgd_steps=multiplier_num_sgd_steps)
            # train_for_all(["SimpleMaze3D"], ["Branching1"], achql_train, "ACHQL", seed_range=(3, 5),
            #               option_length=option_length,
            #               episode_length=episode_length,
            #               multiplier_num_sgd_steps=multiplier_num_sgd_steps)
            # train_for_all(["SimpleMaze3D"], ["ObligationConstraint2"], achql_train, "ACHQL", seed_range=(0, max_seed),
            #               option_length=option_length,
            #               episode_length=episode_length,
            #               multiplier_num_sgd_steps=multiplier_num_sgd_steps)
            # train_for_all(["SimpleMaze3D"], ["Until2"], achql_train, "ACHQL", seed_range=(3, 5),
            #               option_length=option_length,
            #               episode_length=episode_length,
            #               multiplier_num_sgd_steps=multiplier_num_sgd_steps)
            # train_for_all(["SimpleMaze3D"], ["LoopWithObs"], achql_train, "ACHQL", seed_range=(0, max_seed),
            #               option_length=option_length,
            #               episode_length=episode_length,
            #               multiplier_num_sgd_steps=multiplier_num_sgd_steps)

def main():
    all_achql_runs(max_seed=5)
    # all_crm_rs_runs(max_seed=1)
    # all_crm_runs(max_seed=1)


if __name__ == "__main__":
    mlflow.set_tracking_uri("REDACTED")
    # mlflow.set_experiment("proj2-batch-training")
    # mlflow.set_experiment("proj2-final-experiments")
    mlflow.set_experiment("proj2-reproducible-experiments")

    main()
