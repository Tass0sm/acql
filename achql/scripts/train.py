import functools

import mlflow

from brax.io import model

from achql.brax.agents.achql import train as achql
from achql.brax.agents.sac_her import train as sac_her
from achql.brax.agents.crl import train as crl
from achql.brax.agents.ddpg import train as ddpg
from achql.brax.agents.hdqn import train as hdqn
from achql.brax.agents.hdqn_her import train as hdqn_her
from achql.baselines.reward_machines.qrm import train as qrm
from achql.baselines.reward_machines.qrm_ddpg import train as qrm_ddpg
from achql.baselines.reward_machines.crm import train as crm

from achql.tasks import get_task
from achql.brax.utils import make_aut_goal_cmdp, make_reward_machine_mdp

from jaxgcrl.utils.config import RunConfig


def progress_fn(num_steps, metrics, *args, **kwargs):
    print(f"Logging for {num_steps}")
    print(metrics)
    mlflow.log_metrics(metrics, step=num_steps)


def training_run(run_id, env, seed, train_fn, progress_fn=progress_fn, hyperparameters={}, extras={}):
    hyperparameters = {
        **hyperparameters,
        "seed": seed,
    }

    mlflow.log_params(hyperparameters)

    train_fn = functools.partial(train_fn, **hyperparameters)

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=progress_fn,
        seed=seed,
        **extras
    )

    with mlflow.MlflowClient()._log_artifact_helper(run_id, 'policy_params') as tmp_path:
        model.save_params(tmp_path, params)

    return make_inference_fn, params


def train_for_all(envs, tasks, func, alg_tag, seed_range=(0, 3), extra_tasks={}):
    for env_name in envs:
        for task_name in tasks:
            task = get_task(env_name, task_name, extra_tasks=extra_tasks)

            spec = task.lo_spec
            spec_tag = type(task).__name__
            env_tag = type(task.env).__name__

            for seed in range(*seed_range):
                with mlflow.start_run(tags={"env": env_tag, "spec": spec_tag, "alg": alg_tag}) as run:
                    func(run, task, seed, spec)


def ddpg_train(run, task, seed, spec, reward_shaping=False):
    make_inference_fn, params = training_run(
        run.info.run_id,
        task.env,
        seed,
        train_fn=ddpg.train,
        hyperparameters=task.ddpg_hps,
        extras={
            "specification": spec,
            "state_var": task.obs_var,
        }
    )


def hdqn_train(run, task, seed, spec):
    options = task.get_options()

    make_inference_fn, params = training_run(
        run.info.run_id,
        task.env,
        seed,
        train_fn=hdqn.train,
        hyperparameters=task.hdqn_hps,
        extras={
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
        }
    )


def hdqn_her_train(run, task, seed, spec):
    options = task.get_options()

    make_inference_fn, params = training_run(
        run.info.run_id,
        task.env,
        seed,
        train_fn=hdqn_her.train,
        hyperparameters=task.hdqn_her_hps,
        extras={
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
        }
    )


def qrm_train(run, task, seed, spec, reward_shaping=False):
    options = task.get_options()

    make_inference_fn, params = training_run(
        run.info.run_id,
        make_reward_machine_mdp(task, reward_shaping=reward_shaping),
        seed,
        train_fn=qrm.train,
        hyperparameters=task.crm_hps,
        extras={
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
        }
    )


def qrm_ddpg_train(run, task, seed, spec, reward_shaping=False):
    make_inference_fn, params = training_run(
        run.info.run_id,
        make_reward_machine_mdp(task, reward_shaping=reward_shaping),
        seed,
        train_fn=qrm_ddpg.train,
        hyperparameters=task.crm_hps,
        extras={
            "specification": spec,
            "state_var": task.obs_var,
        }
    )


def crm_train(run, task, seed, spec, reward_shaping=False):
    options = task.get_options()

    make_inference_fn, params = training_run(
        run.info.run_id,
        make_reward_machine_mdp(task, reward_shaping=reward_shaping),
        seed,
        train_fn=crm.train,
        hyperparameters=task.crm_hps,
        extras={
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
        }
    )


def achql_train(run, task, seed, spec, margin=1.0):
    options = task.get_options()

    return training_run(
        run.info.run_id,
        make_aut_goal_cmdp(task, margin=margin, randomize_goals=task.achql_hps["use_her"]),
        seed,
        train_fn=achql.train,
        hyperparameters=task.achql_hps,
        extras={
            "options": options,
            "specification": spec,
            "state_var": task.obs_var,
        }
    )


def sac_her_train(run, task, seed, spec):

    def train_fn(environment, progress_fn, seed, **kwargs):
        hyperparameters = { k: v for k, v in kwargs.items() \
                            if k in ["learning_rate", "discounting", "batch_size",
                                     "normalize_observations", "reward_scaling",
                                     "tau", "min_replay_size", "max_replay_size", "deterministic_eval",
                                     "train_step_multiplier", "unroll_length", "h_dim", "n_hidden",
                                     "use_ln", "use_her"] }

        agent = sac_her.SAC(**hyperparameters)

        run_params = { k: v for k, v in kwargs.items() \
                       if k in ["total_env_steps",
                                "episode_length",
                                "num_envs",
                                "num_eval_envs",
                                "num_evals",
                                "action_repeat",
                                "max_devices_per_host"] }

        run_config = RunConfig(
            env=environment,
            seed=seed,
            **run_params
        )
            
        return agent.train_fn(
            config=run_config,
            train_env=environment,
            # eval_env=eval_env,
            progress_fn=progress_fn,
        )
    
    make_inference_fn, params = training_run(
        run.info.run_id,
        task.env,
        seed,
        train_fn=train_fn,
        hyperparameters=task.sac_her_hps,
        extras={
            # "options": options,
            # "specification": spec,
            # "state_var": task.obs_var,
        }
    )


    
def crl_train(run, task, seed, spec):

    def train_fn(environment, progress_fn, seed, **kwargs):
        hyperparameters = { k: v for k, v in kwargs.items() if k in
                            ["critic_lr", "alpha_lr", "batch_size",
                             "discounting", "logsumexp_penalty_coeff",
                             "train_step_multiplier", "disable_entropy_actor",
                             "max_replay_size", "min_replay_size",
                             "unroll_length", "h_dim", "n_hidden",
                             "skip_connections", "use_relu", "repr_dim",
                             "use_ln", "contrastive_loss_fn", "energy_fn",
                             "policy_lr"] }

        agent = crl.CRL(**hyperparameters)

        run_params = { k: v for k, v in kwargs.items() \
                       if k in ["total_env_steps",
                                "episode_length",
                                "num_envs",
                                "num_eval_envs",
                                "num_evals",
                                "action_repeat",
                                "max_devices_per_host"] }

        run_config = RunConfig(
            env=environment,
            seed=seed,
            **run_params
        )

        return agent.train_fn(
            config=run_config,
            train_env=environment,
            # eval_env=eval_env,
            progress_fn=progress_fn,
        )

    make_inference_fn, params = training_run(
        run.info.run_id,
        task.env,
        seed,
        train_fn=train_fn,
        hyperparameters=task.crl_hps,
        extras={
            # "options": options,
            # "specification": spec,
            # "state_var": task.obs_var,
        }
    )


def main():
    # train_for_all(["SimpleMaze"], ["TwoSubgoals"], crm_train, "CRM", seed_range=(0, 1))
    # train_for_all(["SimpleMaze"], ["TwoSubgoals"], qrm_train, "QRM", seed_range=(0, 1))
    # train_for_all(["SimpleMaze"], ["NotUntilAlwaysSubgoal"], qrm_train, "QRM", seed_range=(0, 1))
    # train_for_all(["SimpleMaze"], ["NotUntilAlwaysSubgoal"], qrm_ddpg_train, "QRM_DDPG", seed_range=(0, 1))
    # train_for_all(["SimpleMaze"], ["NotUntilAlwaysSubgoal"], qrm_ddpg_train, "QRM_DDPG", seed_range=(0, 1))
    # train_for_all(["SimpleMaze"], ["NotUntilAlwaysSubgoal"], qrm_ddpg_train, "QRM_DDPG", seed_range=(0, 1))

    train_for_all(["SimpleMaze"], ["Nav"], achql_train, "ACHQL", seed_range=(0, 1))

    # train_for_all(["SimpleMaze"], ["SingleSubgoal"], achql_train, "ACHQL", seed_range=(0, 1))
    # train_for_all(["SimpleMaze"], ["Until1"], achql_train, "ACHQL", seed_range=(0, 1))
    # train_for_all(["ArmEEF"], ["BinpickEasyTask"], sac_her_train, "SAC_HER")
    # train_for_all(["Panda"], ["PushEasyTask"], sac_her_train, "SAC_HER", seed_range=(0, 1))
    # train_for_all(["ArmEEF"], ["BinpickEasyTask"], sac_her_train, "SAC_HER", seed_range=(0, 1))
    # train_for_all(["ArmEEF"], ["BinpickEasyTask"], crl_train, "CRL", seed_range=(0, 1))
    # train_for_all(["ArmEEF"], ["BinpickEasyTask"], crl_train, "CRL")

            
if __name__ == "__main__":
    mlflow.set_tracking_uri("file:///home/tassos/.local/share/mlflow")
    mlflow.set_experiment("proj2-batch-training")

    main()
