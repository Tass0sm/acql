import matplotlib.pyplot as plt

import mlflow
import pandas as pd
import seaborn as sns

from scipy.ndimage.filters import gaussian_filter1d

from achql.tasks.utils import get_task_by_name


def get_group(env, spec, alg, extra_str=""):
    runs = mlflow.search_runs(
        filter_string=f"tags.env = '{env}' and tags.spec = '{spec}' and tags.alg = '{alg}'" + extra_str
    )

    return runs


def fetch_runs():
    groups = []

    for alg, env, spec in [# ("ACHQL", "SimpleMaze", "SimpleMazeTwoSubgoals"),
                           # ("CRM_RS", "SimpleMaze", "SimpleMazeTwoSubgoals"),
                           # ("ACHQL", "SimpleMaze", "SimpleMazeBranching1"),
                           # ("CRM_RS", "SimpleMaze", "SimpleMazeBranching1"),
                           # ("ACHQL", "SimpleMaze", "SimpleMazeObligationConstraint2"),
                           # ("CRM_RS", "SimpleMaze", "SimpleMazeObligationConstraint2"),
                           # ("ACHQL", "SimpleMaze", "SimpleMazeUntil2"),
                           # ("CRM_RS", "SimpleMaze", "SimpleMazeUntil2"),
                           # ("ACHQL", "SimpleMaze", "SimpleMazeLoopWithObs"),
                           # ("CRM_RS", "SimpleMaze", "SimpleMazeLoopWithObs"),
                           ("ACHQL", "SimpleMaze3D", "SimpleMaze3DTwoSubgoals"),
                           # ("CRM_RS", "SimpleMaze3D", "SimpleMaze3DTwoSubgoals"),
                           ("ACHQL", "SimpleMaze3D", "SimpleMaze3DBranching1"),
                           # ("CRM_RS", "SimpleMaze3D", "SimpleMaze3DBranching1"),
                           ("ACHQL", "SimpleMaze3D", "SimpleMaze3DObligationConstraint2"),
                           # ("CRM_RS", "SimpleMaze3D", "SimpleMaze3DObligationConstraint2"),
                           ("ACHQL", "SimpleMaze3D", "SimpleMaze3DUntil2"),
                           # ("CRM_RS", "SimpleMaze3D", "SimpleMaze3DUntil2"),
                           ("ACHQL", "SimpleMaze3D", "SimpleMaze3DLoopWithObs"),
                           # ("CRM_RS", "SimpleMaze3D", "SimpleMaze3DLoopWithObs"),
                           ("ACHQL", "AntMaze", "AntMazeTwoSubgoals"),
                           ("CRM_RS", "AntMaze", "AntMazeTwoSubgoals"),
                           ("ACHQL", "AntMaze", "AntMazeBranching1"),
                           ("CRM_RS", "AntMaze", "AntMazeBranching1"),
                           ("ACHQL", "AntMaze", "AntMazeObligationConstraint3"),
                           ("CRM_RS", "AntMaze", "AntMazeObligationConstraint3"),
                           ("ACHQL", "AntMaze", "AntMazeUntil1"),
                           ("CRM_RS", "AntMaze", "AntMazeUntil1"),
                           ("ACHQL", "AntMaze", "AntMazeLoopWithObs"),
                           ("CRM_RS", "AntMaze", "AntMazeLoopWithObs")]:
        task = get_task_by_name(spec)

        if alg == "ACHQL" and env == "SimpleMaze3D" and "LoopWithObs" not in spec:
            mlflow.set_experiment("proj2-reproducible-experiments")
            episode_length = task.achql_hps["episode_length"]
            mult = task.achql_hps["multiplier_num_sgd_steps"]
            extra_kwargs = {
                "extra_str": f" and params.episode_length = '{episode_length}' and params.multiplier_num_sgd_steps = '{mult}'"
            }
        else:
            extra_kwargs = {}

            if alg == "LOF":
                mlflow.set_experiment("proj2-lof-final-experiments")
            else:
                mlflow.set_experiment("proj2-final-experiments")

        group = get_group(env, spec, alg, **extra_kwargs)
        group = group.iloc[:5]
        print(alg, "names", group["tags.mlflow.runName"].tolist())
        groups.append(group)

        # if "ObligationConstraint" in spec:
        #     group = get_lof_group(env, spec, "LOF")
        #     group = group.iloc[:3]
        #     print("names", group["tags.mlflow.runName"].tolist())
        #     groups.append(group)

    all_runs = pd.concat(groups)

    return all_runs


def get_metric_values(run_id, metric_name):
    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run_id, metric_name)
    return [(m.step, m.value) for m in metric_history]


def process_runs(runs):
    data = []


    for _, row in runs.iterrows():
        # if all(tag in row for tag in ['algorithm', 'environment', 'task']):
        alg = row["tags.alg"]
        env = row["tags.env"]
        spec = row["tags.spec"]
        if alg == "ACHQL" and env == "SimpleMaze3D" and "LoopWithObs" not in spec:
            mlflow.set_experiment("proj2-reproducible-experiments")
        else:
            if alg == "LOF":
                mlflow.set_experiment("proj2-lof-final-experiments")
            else:
                mlflow.set_experiment("proj2-final-experiments")

        if "LoopWithObs" in spec:
            sr_values = get_metric_values(row['run_id'], 'eval/loop_success_rate')
        else:
            sr_values = get_metric_values(row['run_id'], 'eval/proportion_robustness_over_zero')

        sr_df = pd.DataFrame(sr_values, columns=['step', 'sr']).set_index("step")
        reward_values = get_metric_values(row['run_id'], 'eval/episode_reward')
        reward_df = pd.DataFrame(reward_values, columns=['step', 'reward']).set_index("step")
        metrics_df = pd.concat([sr_df, reward_df], axis=1).reset_index()

        metrics_df['spec'] = row["tags.spec"]
        metrics_df['alg'] = row["tags.alg"]
        metrics_df['seed'] = row["params.seed"]

        data.append(metrics_df)

    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")

    runs = fetch_runs()
    df = process_runs(runs)
    df.to_csv("./lab-pc-data.csv")
