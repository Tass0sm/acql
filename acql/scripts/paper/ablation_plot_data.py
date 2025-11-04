import matplotlib.pyplot as plt

import mlflow
import pandas as pd
import seaborn as sns

from scipy.ndimage.filters import gaussian_filter1d

from acql.tasks.utils import get_task_by_name


def get_group(env, spec, alg, extra_str=""):
    runs = mlflow.search_runs(
        filter_string=f"tags.env = '{env}' and tags.spec = '{spec}' and tags.alg = '{alg}'" + extra_str
    )

    return runs


def fetch_runs():
    groups = []

    for alg, env, spec in [# POINT MASS
                           # ("ACQL_ABLATION_ONE", "SimpleMaze", "SimpleMazeTwoSubgoals"),
                           # ("ACQL", "SimpleMaze", "SimpleMazeTwoSubgoals"),
                           # ("ACQL_ABLATION_ONE", "SimpleMaze", "SimpleMazeBranching1"),
                           # ("ACQL", "SimpleMaze", "SimpleMazeBranching1"),
                           # ("ACQL_ABLATION_TWO", "SimpleMaze", "SimpleMazeObligationConstraint1"),
                           # ("ACQL_ABLATION_ONE", "SimpleMaze", "SimpleMazeObligationConstraint1"),
                           # ("ACQL", "SimpleMaze", "SimpleMazeObligationConstraint1"),
                           # ("ACQL_ABLATION_TWO", "SimpleMaze", "SimpleMazeUntil2"),
                           # ("ACQL_ABLATION_ONE", "SimpleMaze", "SimpleMazeUntil2"),
                           # ("ACQL", "SimpleMaze", "SimpleMazeUntil2"),
                           # QUADCOPTER
                           # ("ACQL_ABLATION_ONE", "SimpleMaze3D", "SimpleMaze3DTwoSubgoals"),
                           ("ACQL", "SimpleMaze3D", "SimpleMaze3DTwoSubgoals"),
                           # ("ACQL_ABLATION_ONE", "SimpleMaze3D", "SimpleMaze3DBranching1"),
                           ("ACQL", "SimpleMaze3D", "SimpleMaze3DBranching1"),
                           # ("ACQL_ABLATION_TWO", "SimpleMaze3D", "SimpleMaze3DObligationConstraint2"),
                           # ("ACQL_ABLATION_ONE", "SimpleMaze3D", "SimpleMaze3DObligationConstraint2"),
                           ("ACQL", "SimpleMaze3D", "SimpleMaze3DObligationConstraint2"),
                           # ("ACQL_ABLATION_TWO", "SimpleMaze3D", "SimpleMaze3DUntil2"),
                           # ("ACQL_ABLATION_ONE", "SimpleMaze3D", "SimpleMaze3DUntil2"),
                           ("ACQL", "SimpleMaze3D", "SimpleMaze3DUntil2"),
                           # ANT MAZE
                           ("ACQL_ABLATION_ONE", "AntMaze", "AntMazeTwoSubgoals"),
                           ("ACQL", "AntMaze", "AntMazeTwoSubgoals"),
                           ("ACQL_ABLATION_ONE", "AntMaze", "AntMazeBranching1"),
                           ("ACQL", "AntMaze", "AntMazeBranching1"),
                           ("ACQL_ABLATION_TWO", "AntMaze", "AntMazeObligationConstraint3"),
                           ("ACQL_ABLATION_ONE", "AntMaze", "AntMazeObligationConstraint3"),
                           ("ACQL", "AntMaze", "AntMazeObligationConstraint3"),
                           ("ACQL_ABLATION_TWO", "AntMaze", "AntMazeUntil1"),
                           ("ACQL_ABLATION_ONE", "AntMaze", "AntMazeUntil1"),
                           ("ACQL", "AntMaze", "AntMazeUntil1")
    ]:
        task = get_task_by_name(spec)

        if alg == "ACQL_ABLATION_ONE":
            mlflow.set_experiment("proj2-ablation-experiments")
            extra_str = ""
        elif alg == "ACQL_ABLATION_TWO":
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

        group = get_group(env, spec, alg, extra_str=extra_str)
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

        if alg == "ACQL_ABLATION_ONE":
            mlflow.set_experiment("proj2-ablation-experiments")
        elif alg == "ACQL_ABLATION_TWO":
            if spec == "AntMazeUntil1":
                mlflow.set_experiment("proj2-ablation-experiments")
            elif spec == "AntMazeObligationConstraint3":
                mlflow.set_experiment("proj2-ablation-two-sweep")
            else:
                mlflow.set_experiment("proj2-ablation-experiments")
        else:
            if env == "SimpleMaze3D":
                mlflow.set_experiment("proj2-reproducible-experiments")
            else:
                mlflow.set_experiment("proj2-final-experiments")

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
    df.to_csv("./lab-pc-ablation-data.csv")
