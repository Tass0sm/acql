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

# tasks
from acql.tasks.utils import get_task_by_name




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

    # "ACQL"
    # , "ACQL_ABLATION_ONE"

    for alg in ["ACQL_ABLATION_ONE"]:

        groups = []
        for spec in [# "SimpleMazeTwoSubgoals",
                     # "SimpleMazeBranching1",
                     # "SimpleMazeObligationConstraint1",
                     # "SimpleMazeUntil2",
                     "SimpleMaze3DTwoSubgoals",
                     "SimpleMaze3DBranching1",
                     "SimpleMaze3DObligationConstraint2",
                     "SimpleMaze3DUntil2",
                     "AntMazeTwoSubgoals",
                     "AntMazeBranching1",
                     "AntMazeObligationConstraint3",
                     "AntMazeUntil1"
                     ]:

            if alg == "ACQL_ABLATION_ONE":
                mlflow.set_experiment("proj2-ablation-experiments")
            else:
                if "SimpleMaze3D" in spec:
                    task = get_task_by_name(spec)
                    mlflow.set_experiment("proj2-reproducible-experiments")
                    episode_length = task.acql_hps["episode_length"]
                    mult = task.acql_hps["multiplier_num_sgd_steps"]
                    extra_str = f" and params.episode_length = '{episode_length}' and params.multiplier_num_sgd_steps = '{mult}'"
                else:
                    extra_str = ""
                    mlflow.set_experiment("proj2-final-experiments")

            runs = mlflow.search_runs(
                filter_string=f"tags.alg = '{alg}' and tags.spec = '{spec}'" + extra_str
            ).iloc[:5]
            groups.append(runs)
    
        group = pd.concat(groups)
    
        print("names", group["tags.mlflow.runName"].tolist())
        print("ids", group["run_id"].tolist())

        print("reward", group["metrics.eval/episode_reward"])

        try:
            success_mean = group["metrics.eval/episode_reward"].mean().round(decimals=1)
            success_std = group["metrics.eval/episode_reward"].std().round(decimals=1)
            print(alg, "success", f"{success_mean} +- {success_std}")
        except:
            print(alg, "success ERROR")

        print("s.r.", group["metrics.eval/proportion_robustness_over_zero"])

        try:
            good_rho_prop_mean = (group["metrics.eval/proportion_robustness_over_zero"].mean() * 100).round(decimals=1)
            good_rho_std_mean = (group["metrics.eval/proportion_robustness_over_zero"].std() * 100).round(decimals=1)
            print(alg, "good_rho", f"{good_rho_prop_mean} +- {good_rho_std_mean}")
        except:
            print(alg, "good_rho ERROR")


def ablation_table_second_row():

    # "ACQL"

    for alg in ["ACQL_ABLATION_TWO"]:

        groups = []
        for spec in [# "SimpleMazeObligationConstraint1",
                     # "SimpleMazeUntil2",
                     # "SimpleMaze3DObligationConstraint2",
                     # "SimpleMaze3DUntil2",
                     "AntMazeObligationConstraint3",
                     "AntMazeUntil1"
                     ]:

            if alg == "ACQL_ABLATION_TWO":
                if spec == "AntMazeUntil1":
                    mlflow.set_experiment("proj2-ablation-experiments")
                    extra_str = f" and params.safety_threshold = '10'"
                elif spec == "AntMazeObligationConstraint3":
                    mlflow.set_experiment("proj2-ablation-two-sweep")
                    extra_str = f" and params.safety_threshold = '10'"
                else:
                    mlflow.set_experiment("proj2-ablation-experiments")
                    extra_str = f" and params.safety_threshold = '10'"
            else:
                if "SimpleMaze3D" in spec:
                    task = get_task_by_name(spec)
                    mlflow.set_experiment("proj2-reproducible-experiments")
                    episode_length = task.acql_hps["episode_length"]
                    mult = task.acql_hps["multiplier_num_sgd_steps"]
                    extra_str = f" and params.episode_length = '{episode_length}' and params.multiplier_num_sgd_steps = '{mult}'"
                else:
                    extra_str = ""
                    mlflow.set_experiment("proj2-final-experiments")

            runs = mlflow.search_runs(
                filter_string=f"tags.alg = '{alg}' and tags.spec = '{spec}'" + extra_str
            ).iloc[:5]
            groups.append(runs)

        group = pd.concat(groups)

        print("names", group["tags.mlflow.runName"].tolist())
        print("ids", group["run_id"].tolist())
        print("reward", group["metrics.eval/episode_reward"])
        print("s.r.", group["metrics.eval/proportion_robustness_over_zero"])

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


def main():
    mlflow.set_tracking_uri(f"file:///home/tassos/.local/share/mlflow")

    # ablation_table_first_row()
    ablation_table_second_row()
    

if __name__ == "__main__":
    main()
