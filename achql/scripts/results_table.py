import mlflow


def get_group(env, spec, alg):
    runs = mlflow.search_runs(
        filter_string=f"tags.env = '{env}' and tags.spec = '{spec}' and tags.alg = '{alg}'"
    )

    return runs


def main():
    mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")
    mlflow.set_experiment("proj2-final-experiments")
    
    for env, spec in [# ("SimpleMaze", "SimpleMazeTwoSubgoals"),
                      # ("SimpleMaze", "SimpleMazeBranching1"),
                      # ("SimpleMaze", "SimpleMazeObligationConstraint1"),
                      # ("SimpleMaze3D", "SimpleMaze3DTwoSubgoals"),
                      # ("SimpleMaze3D", "SimpleMaze3DBranching1"),
                      # ("SimpleMaze3D", "SimpleMaze3DObligationConstraint2"),
                      ("AntMaze", "AntMazeTwoSubgoals"),
                      ("AntMaze", "AntMazeBranching1"),
                      ("AntMaze", "AntMazeObligationConstraint1"),
                      ("AntMaze", "AntMazeUntil1"),
                      ("AntMaze", "AntMazeLoopWithObs")]:
        for alg in ["ACHQL"]:
    
            # , "LOF", "CRM_RS", "CRM"
    
            if alg == "LOF":
                raise NotImplementedError()
                # mlflow.set_experiment("proj2-batch-lof-option-training")
            else:
                mlflow.set_experiment("proj2-final-experiments")
    
            group = get_group(env, spec, alg)
    
            
            # use the five most recent (tentative)
            group = group.iloc[:5]
    
            print("names", group["tags.mlflow.runName"].tolist())
    
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
