import mlflow

from acql.brax.agents.achql import train as achql

from acql.scripts.training.train import train_for_all, training_run

# tasks
from acql.brax.utils import (
    make_aut_goal_cmdp,
)



def achql_ablation_one_train(run, task, seed, spec, margin=1.0):
    options = task.get_options()

    return training_run(
        run.info.run_id,
        make_aut_goal_cmdp(task, margin=margin, randomize_goals=False),
        seed,
        train_fn=achql.train,
        hyperparameters=task.achql_hps | { "use_her": False },
        extras={
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
            "eval_environment": make_aut_goal_cmdp(task, margin=margin, randomize_goals=False),
        }
    )


def achql_ablation_two_train(run, task, seed, spec, margin=1.0, safety_threshold=None):
    options = task.get_options()

    return training_run(
        run.info.run_id,
        make_aut_goal_cmdp(task, margin=margin, randomize_goals=task.achql_hps["use_her"], relu_cost=True),
        seed,
        train_fn=achql.train,
        hyperparameters=task.achql_hps | { "safety_threshold": safety_threshold },
        extras={
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
            "eval_environment": make_aut_goal_cmdp(task, margin=margin, randomize_goals=False, relu_cost=True),
            "use_sum_cost_critic": True
        }
    )


def all_achql_ablation_one_runs(max_seed=5):
    # train_for_all(["SimpleMaze"], ["TwoSubgoals"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    # train_for_all(["SimpleMaze"], ["Branching1"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    # train_for_all(["SimpleMaze"], ["ObligationConstraint1"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    # train_for_all(["SimpleMaze"], ["Until2"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    # train_for_all(["SimpleMaze"], ["LoopWithObs"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))

    # train_for_all(["SimpleMaze3D"], ["TwoSubgoals"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    # train_for_all(["SimpleMaze3D"], ["Branching1"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    # train_for_all(["SimpleMaze3D"], ["ObligationConstraint2"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    # train_for_all(["SimpleMaze3D"], ["Until2"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    # train_for_all(["SimpleMaze3D"], ["LoopWithObs"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))

    # train_for_all(["AntMaze"], ["TwoSubgoals"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    # train_for_all(["AntMaze"], ["Branching1"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    # train_for_all(["AntMaze"], ["ObligationConstraint3"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    train_for_all(["AntMaze"], ["Until1"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))
    # train_for_all(["AntMaze"], ["LoopWithObs"], achql_ablation_one_train, "ACHQL_ABLATION_ONE", seed_range=(0, max_seed))


def all_achql_ablation_two_runs(max_seed=5):

    safety_threshold_l = [10]

    for safety_threshold in safety_threshold_l:
        # train_for_all(["SimpleMaze"], ["ObligationConstraint1"], achql_ablation_two_train, "ACHQL_ABLATION_TWO", seed_range=(0, max_seed),
        #               safety_threshold=safety_threshold)
        # train_for_all(["SimpleMaze"], ["Until2"], achql_ablation_two_train, "ACHQL_ABLATION_TWO", seed_range=(0, max_seed),
        #               safety_threshold=safety_threshold)
        # train_for_all(["SimpleMaze"], ["LoopWithObs"], achql_ablation_two_train, "ACHQL_ABLATION_TWO", seed_range=(0, max_seed),
        #               safety_threshold=safety_threshold)

        # train_for_all(["SimpleMaze3D"], ["ObligationConstraint2"], achql_ablation_two_train, "ACHQL_ABLATION_TWO", seed_range=(0, max_seed),
        #               safety_threshold=safety_threshold)
        # train_for_all(["SimpleMaze3D"], ["Until2"], achql_ablation_two_train, "ACHQL_ABLATION_TWO", seed_range=(0, max_seed),
        #               safety_threshold=safety_threshold)
        # train_for_all(["SimpleMaze3D"], ["LoopWithObs"], achql_ablation_two_train, "ACHQL_ABLATION_TWO", seed_range=(0, max_seed),
        #               safety_threshold=safety_threshold)

        # train_for_all(["AntMaze"], ["ObligationConstraint3"], achql_ablation_two_train, "ACHQL_ABLATION_TWO", seed_range=(0, max_seed),
        #               safety_threshold=safety_threshold)
        train_for_all(["AntMaze"], ["Until1"], achql_ablation_two_train, "ACHQL_ABLATION_TWO", seed_range=(0, max_seed),
                      safety_threshold=safety_threshold)
        # train_for_all(["AntMaze"], ["LoopWithObs"], achql_ablation_two_train, "ACHQL_ABLATION_TWO", seed_range=(0, max_seed),
        #               safety_threshold=safety_threshold)


def main():
    all_achql_ablation_one_runs()
    # all_achql_ablation_two_runs()


if __name__ == "__main__":
    mlflow.set_tracking_uri("REDACTED")
    mlflow.set_experiment("proj2-ablation-experiments")
    # mlflow.set_experiment("proj2-ablation-two-sweep")

    main()
