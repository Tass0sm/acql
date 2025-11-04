import mlflow
from acql.tasks.utils import get_task_by_name


def get_group(env, spec, alg, extra_str=""):
    runs = mlflow.search_runs(
        filter_string=f"tags.env = '{env}' and tags.spec = '{spec}' and tags.alg = '{alg}'" + extra_str
    )

    return runs


def main():
    mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")
    mlflow.set_experiment("proj2-final-experiments")
    
    for env, spec in [# ("SimpleMaze3D", "SimpleMaze3DTwoSubgoals"),
                      # ("SimpleMaze3D", "SimpleMaze3DBranching1"),
                      # ("SimpleMaze3D", "SimpleMaze3DObligationConstraint2"),
                      # ("SimpleMaze3D", "SimpleMaze3DUntil2"),
                      # ("SimpleMaze3D", "SimpleMaze3DLoopWithObs")
                      ("AntMaze", "AntMazeTwoSubgoals"),
                      ("AntMaze", "AntMazeBranching1"),
                      ("AntMaze", "AntMazeObligationConstraint3"),
                      ("AntMaze", "AntMazeUntil1"),
                      ("AntMaze", "AntMazeLoopWithObs")
                      ]:

        for alg in ["LOF"]:

            task = get_task_by_name(spec)

            if alg == "ACQL" and env == "SimpleMaze3D":
                mlflow.set_experiment("proj2-reproducible-experiments")
                episode_length = task.acql_hps["episode_length"]
                mult = task.acql_hps["multiplier_num_sgd_steps"]
                extra_kwargs = {
                    "extra_str": f" and params.episode_length = '{episode_length}' and params.multiplier_num_sgd_steps = '{mult}'"
                }
            else:
                extra_kwargs = {}

                if alg == "LOF":
                    mlflow.set_experiment("proj2-lof-final-experiments")
                else:
                    mlflow.set_experiment("proj2-final-experiments")
                
            # "ACQL", "LOF", "CRM_RS", "CRM"
    

            group = get_group(env, spec, alg, **extra_kwargs)

            # use the five most recent (tentative)
            group = group.iloc[:5]

            print("names", group["tags.mlflow.runName"].tolist())
            print("ids", group["run_id"].tolist())
    
            # assert group.shape[0] >= 3
    
            try:
                success_mean = group["metrics.eval/episode_reward"].mean().round(decimals=1)
                success_std = group["metrics.eval/episode_reward"].std().round(decimals=1)
                print(env, spec, alg, "success", f"{success_mean} +- {success_std}")
            except:
                print(env, spec, alg, "success ERROR")
                continue
    
            try:
                good_rho_prop_mean = (group["metrics.eval/proportion_robustness_over_zero"].mean() * 100).round(decimals=1)
                good_rho_std_mean = (group["metrics.eval/proportion_robustness_over_zero"].std() * 100).round(decimals=1)
                print(env, spec, alg, "good_rho", f"{good_rho_prop_mean} +- {good_rho_std_mean}")
            except:
                print(env, spec, alg, "good_rho ERROR")
                continue


if __name__ == "__main__":
    main()
