import functools
import jax

import mlflow

from achql.brax.agents.hdqn import train as hdqn

from achql.baselines.logical_options_framework import train as lof
from achql.baselines.logical_options_framework import get_logical_option_run_ids

from achql.brax.tasks import get_task

from achql.scripts.train import train_for_all, training_run


def lof_train(run, task, seed, spec, reward_shaping=False):
    options = task.get_options()
    spec_tag = type(task).__name__

    return training_run(
        run.info.run_id,
        task.env,
        seed,
        train_fn=lof.train,
        hyperparameters=task.lof_hps,
        extras={
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
            "task_name": spec_tag,
            "task_state_costs": task.lof_task_state_costs,
            # "logical_option_run_ids": logical_option_run_ids,
            "logical_option_run_ids": get_logical_option_run_ids(spec_tag, seed),
            "option_train_fn": hdqn.train,
            "option_train_fn_hps": task.hdqn_hps
        }
    )


def main():
    # train_for_all(["SimpleMaze"], ["Until1"], lof_train, "LOF", seed_range=(0, 5))
    # train_for_all(["SimpleMaze"], ["Until2"], lof_train, "LOF", seed_range=(0, 5))
    # train_for_all(["SimpleMaze"], ["LoopWithObs"], lof_train, "LOF", seed_range=(0, 5))

    # train_for_all(["SimpleMaze3D"], ["Until1"], lof_train, "LOF", seed_range=(0, 5))
    # train_for_all(["SimpleMaze3D"], ["Until2"], lof_train, "LOF", seed_range=(0, 5))
    # train_for_all(["SimpleMaze3D"], ["LoopWithObs"], lof_train, "LOF", seed_range=(0, 5))

    train_for_all(["SimpleMaze"], ["TwoSubgoals"], lof_train, "LOF", seed_range=(0, 5))
    train_for_all(["SimpleMaze"], ["Branching1"], lof_train, "LOF", seed_range=(0, 5))
    train_for_all(["SimpleMaze"], ["ObligationConstraint1"], lof_train, "LOF", seed_range=(0, 5))

    train_for_all(["SimpleMaze3D"], ["TwoSubgoals"], lof_train, "LOF", seed_range=(0, 5))
    train_for_all(["SimpleMaze3D"], ["Branching1"], lof_train, "LOF", seed_range=(0, 5))
    train_for_all(["SimpleMaze3D"], ["ObligationConstraint1"], lof_train, "LOF", seed_range=(0, 5))




if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")
    mlflow.set_experiment("proj2-lof-final-experiments")

    main()
