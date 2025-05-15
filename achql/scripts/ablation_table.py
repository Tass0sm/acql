import functools
import jax
import os
from typing import Optional
import pandas as pd

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

from achql.baselines.reward_machines.crm import train as crm
# from achql.baselines.reward_machines.qrm import train as qrm
from achql.baselines.reward_machines.hrm import train as hrm
from achql.baselines.reward_machines.reward_shaping import train as rm_reward_shaping

# tasks
from achql.brax.tasks import get_task




def get_group(alg):
    runs = mlflow.search_runs(
        filter_string=f"tags.alg = '{alg}'"
    )

    return runs

# def get_ablation4_group(alg):
#     groups = []
#     for spec in ["SimpleMazeObligationConstraint1", "SimpleMaze3DObligationConstraint2", "SimpleMaze3DObligationConstraint3"]:
#         runs = mlflow.search_runs(
#             filter_string=f"tags.alg = '{alg}' and tags.spec = '{spec}'"
#         ).iloc[:3]
#         groups.append(runs)

#     breakpoint()
#     return pd.concat(groups)

# # "ABLATION_ONE", "ABLATION_TWO",

# for alg in ["ABLATION_FOUR"]:

#     if alg == "ABLATION_FOUR":
#         mlflow.set_experiment("proj2-ablation-four")
#         group = get_ablation4_group(alg)
#     else:
#         # group = get_group(alg)
#         group = get_group(alg)

#     # group = group.iloc[:3]
#     # assert group.shape[0] >= 3

#     try:
#         success_mean = group["metrics.eval/episode_automaton_success"].mean().round(decimals=1)
#         success_std = group["metrics.eval/episode_automaton_success"].std().round(decimals=1)
#         print(alg, "success", f"{success_mean} +- {success_std}")
#     except:
#         print(alg, "success ERROR")
#         continue

#     try:
#         good_rho_prop_mean = (group["metrics.eval/proportion_robustness_over_zero"].mean() * 100).round(decimals=1)
#         good_rho_std_mean = (group["metrics.eval/proportion_robustness_over_zero"].std() * 100).round(decimals=1)
#         print(alg, "good_rho", f"{good_rho_prop_mean} +- {good_rho_std_mean}")
#     except:
#         print(alg, "good_rho ERROR")
#         continue

# from itertools import product


# def get_group2(spec, alg):
#     runs = mlflow.search_runs(
#         filter_string=f"tags.spec = '{spec}' and tags.alg = '{alg}'"
#     )

#     return runs

import pandas as pd

# mlflow.set_experiment("proj2-batch-training")

# alg = "HDCQN_AUTOMATON_HER"

# # for all_specs in [(["SimpleMaze", "SimpleMaze3D", "AntMaze"], ["TwoSubgoals", "Branching1", "ObligationConstraint1", "ObligationConstraint2", "ObligationConstraint3"]),
# #                   (["SimpleMaze", "SimpleMaze3D", "AntMaze"], ["ObligationConstraint1", "ObligationConstraint2", "ObligationConstraint3"])]:

# #     groups = []
# #     for env, spec_suffix in product(*all_specs):
# #         full_spec = env + spec_suffix
# #         g_i = get_group2(full_spec, alg)
# #         g_i = g_i.iloc[:3]
# #         groups.append(g_i)

# #     group = pd.concat(groups)

# #     try:
# #         success_mean = group["metrics.eval/episode_automaton_success"].mean().round(decimals=1)
# #         success_std = group["metrics.eval/episode_automaton_success"].std().round(decimals=1)
# #         print(alg, "success", f"{success_mean} +- {success_std}")
# #     except:
# #         print(alg, "success ERROR")
# #         continue

# #     try:
# #         good_rho_prop_mean = (group["metrics.eval/proportion_robustness_over_zero"].mean() * 100).round(decimals=1)
# #         good_rho_std_mean = (group["metrics.eval/proportion_robustness_over_zero"].std() * 100).round(decimals=1)
# #         print(alg, "good_rho", f"{good_rho_prop_mean} +- {good_rho_std_mean}")
# #     except:
# #         print(alg, "good_rho ERROR")
# #         continue


# # FOR ABLATION 4

# all_specs = [("SimpleMaze", "ObligationConstraint1"),
#              ("SimpleMaze3D", "ObligationConstraint2"),
#              ("AntMaze", "ObligationConstraint3")]

# groups = []
# for env, spec_suffix in all_specs:
#     full_spec = env + spec_suffix
#     g_i = get_group2(full_spec, alg)
#     g_i = g_i.iloc[:3]
#     groups.append(g_i)

# breakpoint()


# group = pd.concat(groups)

# try:
#     success_mean = group["metrics.eval/episode_automaton_success"].mean().round(decimals=1)
#     success_std = group["metrics.eval/episode_automaton_success"].std().round(decimals=1)
#     print(alg, "success", f"{success_mean} +- {success_std}")
# except:
#     print(alg, "success ERROR")

# try:
#     good_rho_prop_mean = (group["metrics.eval/proportion_robustness_over_zero"].mean() * 100).round(decimals=1)
#     good_rho_std_mean = (group["metrics.eval/proportion_robustness_over_zero"].std() * 100).round(decimals=1)
#     print(alg, "good_rho", f"{good_rho_prop_mean} +- {good_rho_std_mean}")
# except:
#     print(alg, "good_rho ERROR")


def ablation_table_first_row():
    mlflow.set_experiment("proj2-ablation-experiments")

    for alg in ["ACHQL", "ACHQL_ABLATION_ONE"]:
        groups = []
        for spec in ["SimpleMazeTwoSubgoals",
                     "SimpleMazeBranching1",
                     "SimpleMazeObligationConstraint1",
                     "SimpleMazeUntil2",
                     "SimpleMaze3DTwoSubgoals",
                     "SimpleMaze3DBranching1",
                     "SimpleMaze3DObligationConstraint2",
                     "SimpleMaze3DUntil2",
                     "AntMazeTwoSubgoals",
                     "AntMazeBranching1",
                     "AntMazeObligationConstraint3",
                     "AntMazeUntil1"]:
            runs = mlflow.search_runs(
                filter_string=f"tags.alg = '{alg}' and tags.spec = '{spec}'"
            ).iloc[:5]
            groups.append(runs)
    
        group = pd.concat(groups)
    
        try:
            success_mean = group["metrics.eval/episode_reward"].mean().round(decimals=1)
            success_std = group["metrics.eval/episode_reward"].std().round(decimals=1)
            print(alg, "success", f"{success_mean} +- {success_std}")
        except:
            print(alg, "success ERROR")
    
        try:
            good_rho_prop_mean = (group["metrics.eval/proportion_robustness_over_zero"].mean() * 100).round(decimals=1)
            good_rho_std_mean = (group["metrics.eval/proportion_robustness_over_zero"].std() * 100).round(decimals=1)
            print(alg, "good_rho", f"{good_rho_prop_mean} +- {good_rho_std_mean}")
        except:
            print(alg, "good_rho ERROR")


def ablation_table_second_row():
    pass


def main():
    mlflow.set_tracking_uri(f"file:///home/tassos/.local/share/mlflow")

    ablation_table_first_row()
    ablation_table_second_row()
    

if __name__ == "__main__":
    main()
