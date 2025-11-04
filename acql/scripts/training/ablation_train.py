import mlflow

from acql.brax.agents.acql import train as acql

from acql.scripts.training.train import train_for_all, training_run

# tasks
from acql.brax.utils import (
    make_aut_goal_cmdp,
)



def acql_ablation_one_train(run, task, seed, spec, margin=1.0):
    options = task.get_options()

    return training_run(
        run.info.run_id,
        make_aut_goal_cmdp(task, margin=margin, randomize_goals=False),
        seed,
        train_fn=acql.train,
        hyperparameters=task.acql_hps | { "use_her": False },
        extras={
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
            "eval_environment": make_aut_goal_cmdp(task, margin=margin, randomize_goals=False),
        }
    )


def acql_ablation_two_train(run, task, seed, spec, margin=1.0, safety_threshold=None):
    options = task.get_options()

    return training_run(
        run.info.run_id,
        make_aut_goal_cmdp(task, margin=margin, randomize_goals=task.acql_hps["use_her"], relu_cost=True),
        seed,
        train_fn=acql.train,
        hyperparameters=task.acql_hps | { "safety_threshold": safety_threshold },
        extras={
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
            "eval_environment": make_aut_goal_cmdp(task, margin=margin, randomize_goals=False, relu_cost=True),
            "use_sum_cost_critic": True
        }
    )


def all_acql_ablation_one_runs(max_seed=5):
    # train_for_all(["SimpleMaze"], ["TwoSubgoals"], acql_ablation_one_train, "ACQL_ABLATION_ONE", seed_range=(0, max_seed))
    # train_for_all(["SimpleMaze"], ["Branching1"], acql_ablation_one_train, "ACQL_ABLATION_ONE", seed_range=(0, max_seed))
    # train_for_all(["SimpleMaze"], ["ObligationConstraint1"], acql_ablation_one_train, "ACQL_ABLATION_ONE", seed_range=(0, max_seed))
    # train_for_all(["SimpleMaze"], ["Until2"], acql_ablation_one_train, "ACQL_ABLATION_ONE", seed_range=(0, max_seed))
    # train_for_all(["SimpleMaze"], ["LoopWithObs"], acql_ablation_one_train, "ACQL_ABLATION_ONE", seed_range=(0, max_seed))

    # train_for_all(["SimpleMaze3D"], ["TwoSubgoals"], acql_ablation_one_train, "ACQL_ABLATION_ONE", seed_range=(0, max_seed))
    # train_for_all(["SimpleMaze3D"], ["Branching1"], acql_ablation_one_train, "ACQL_ABLATION_ONE", seed_range=(0, max_seed))
    # train_for_all(["SimpleMaze3D"], ["ObligationConstraint2"], acql_ablation_one_train, "ACQL_ABLATION_ONE", seed_range=(0, max_seed))
    # train_for_all(["SimpleMaze3D"], ["Until2"], acql_ablation_one_train, "ACQL_ABLATION_ONE", seed_range=(0, max_seed))
    # train_for_all(["SimpleMaze3D"], ["LoopWithObs"], acql_ablation_one_train, "ACQL_ABLATION_ONE", seed_range=(0, max_seed))

    # train_for_all(["AntMaze"], ["TwoSubgoals"], acql_ablation_one_train, "ACQL_ABLATION_ONE", seed_range=(0, max_seed))
    # train_for_all(["AntMaze"], ["Branching1"], acql_ablation_one_train, "ACQL_ABLATION_ONE", seed_range=(0, max_seed))
    # train_for_all(["AntMaze"], ["ObligationConstraint3"], acql_ablation_one_train, "ACQL_ABLATION_ONE", seed_range=(0, max_seed))
    train_for_all(["AntMaze"], ["Until1"], acql_ablation_one_train, "ACQL_ABLATION_ONE", seed_range=(0, max_seed))
    # train_for_all(["AntMaze"], ["LoopWithObs"], acql_ablation_one_train, "ACQL_ABLATION_ONE", seed_range=(0, max_seed))


def all_acql_ablation_two_runs(max_seed=5):

    safety_threshold_l = [10]

    for safety_threshold in safety_threshold_l:
        # train_for_all(["SimpleMaze"], ["ObligationConstraint1"], acql_ablation_two_train, "ACQL_ABLATION_TWO", seed_range=(0, max_seed),
        #               safety_threshold=safety_threshold)
        # train_for_all(["SimpleMaze"], ["Until2"], acql_ablation_two_train, "ACQL_ABLATION_TWO", seed_range=(0, max_seed),
        #               safety_threshold=safety_threshold)
        # train_for_all(["SimpleMaze"], ["LoopWithObs"], acql_ablation_two_train, "ACQL_ABLATION_TWO", seed_range=(0, max_seed),
        #               safety_threshold=safety_threshold)

        # train_for_all(["SimpleMaze3D"], ["ObligationConstraint2"], acql_ablation_two_train, "ACQL_ABLATION_TWO", seed_range=(0, max_seed),
        #               safety_threshold=safety_threshold)
        # train_for_all(["SimpleMaze3D"], ["Until2"], acql_ablation_two_train, "ACQL_ABLATION_TWO", seed_range=(0, max_seed),
        #               safety_threshold=safety_threshold)
        # train_for_all(["SimpleMaze3D"], ["LoopWithObs"], acql_ablation_two_train, "ACQL_ABLATION_TWO", seed_range=(0, max_seed),
        #               safety_threshold=safety_threshold)

        # train_for_all(["AntMaze"], ["ObligationConstraint3"], acql_ablation_two_train, "ACQL_ABLATION_TWO", seed_range=(0, max_seed),
        #               safety_threshold=safety_threshold)
        train_for_all(["AntMaze"], ["Until1"], acql_ablation_two_train, "ACQL_ABLATION_TWO", seed_range=(0, max_seed),
                      safety_threshold=safety_threshold)
        # train_for_all(["AntMaze"], ["LoopWithObs"], acql_ablation_two_train, "ACQL_ABLATION_TWO", seed_range=(0, max_seed),
        #               safety_threshold=safety_threshold)


def main():
    all_acql_ablation_one_runs()
    # all_acql_ablation_two_runs()


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")
    mlflow.set_experiment("proj2-ablation-experiments")
    # mlflow.set_experiment("proj2-ablation-two-sweep")

    main()
