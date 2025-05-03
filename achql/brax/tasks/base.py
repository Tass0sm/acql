from achql.tasks.base import TaskBase


class BraxTaskBase(TaskBase):

    @property
    def ppo_hps(self):
        return {
            "num_timesteps": 50_000_000,
            "num_evals": 10,
            "reward_scaling": 10,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "unroll_length": 5,
            "num_minibatches": 32,
            "num_updates_per_batch": 4,
            "discounting": 0.97,
            "learning_rate": 3e-4,
            "entropy_cost": 1e-2,
            "num_envs": 4096,
            "batch_size": 2048,
        }

    @property
    def sac_hps(self):
        return {
            "num_timesteps": 5_000_000,
            "num_evals": 20,
            "reward_scaling": 10,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "grad_updates_per_step": 4,
            "discounting": 0.97,
            "learning_rate": 3e-4,
            "num_envs": 256,
            "batch_size": 256,
        }

    @property
    def sac_her_hps(self):
        return {
            "learning_rate": 1e-4,
            "discounting": 0.97,
            "batch_size": 256,
            "normalize_observations": True,
            "reward_scaling": 10.0,
            # target update rate
            "tau": 0.005,
            "min_replay_size": 100,
            "max_replay_size": 10000,
            "deterministic_eval": False,
            "train_step_multiplier": 64,
            "unroll_length": 150,
            "h_dim": 256,
            "n_hidden": 2,
            # layer norm
            "use_ln": True,
            # hindsight experience replay
            "use_her": True,
            # --------------------
            # run params
            "total_env_steps": 5_000_000,
            "episode_length": 150,
            "num_envs": 256,
            "num_eval_envs": 256,
            "num_evals": 50,
            "action_repeat": 1,
            "max_devices_per_host": 1,
            # --------------------
            # # "learning_rate": 3e-4,
            # "discounting": 0.97,
            # "batch_size": 256,
            # "normalize_observations": True,
            # "reward_scaling": 10,
            # "num_timesteps": 2_000_000,
            # "num_evals": 50,
            # "episode_length": 256,
            # "action_repeat": 1,
            # "unroll_length": 62, # TODO: Reducing this increases time. What else does it affect?
            # "multiplier_num_sgd_steps": 1,
            # "max_devices_per_host": 1,
            # "max_replay_size": 10000,
            # # 8192, the default, causes the error "TypeError: broadcast_in_dim shape must have every element be nonnegative, got (-2, 50)."
            # "min_replay_size": 1000,
            # "use_her": True,
            # "num_envs": 512,
        }

    @property
    def ddpg_her_hps(self):
        return {
            "learning_rate": 1e-4,
            "discounting": 0.97,
            "batch_size": 256,
            "normalize_observations": True,
            "reward_scaling": 10.0,
            # target update rate
            "tau": 0.005,
            "min_replay_size": 100,
            "max_replay_size": 10000,
            "deterministic_eval": True,
            "train_step_multiplier": 64,
            "unroll_length": 150,
            "h_dim": 256,
            "n_hidden": 2,
            # layer norm
            "use_ln": True,
            # hindsight experience replay
            "use_her": True,
            # --------------------
            # run params
            "total_env_steps": 5_000_000,
            "episode_length": 150,
            "num_envs": 256,
            "num_eval_envs": 256,
            "num_evals": 50,
            "action_repeat": 1,
            "max_devices_per_host": 1,
            # --------------------
            # # "learning_rate": 3e-4,
            # "discounting": 0.97,
            # "batch_size": 256,
            # "normalize_observations": True,
            # "reward_scaling": 10,
            # "num_timesteps": 2_000_000,
            # "num_evals": 50,
            # "episode_length": 256,
            # "action_repeat": 1,
            # "unroll_length": 62, # TODO: Reducing this increases time. What else does it affect?
            # "multiplier_num_sgd_steps": 1,
            # "max_devices_per_host": 1,
            # "max_replay_size": 10000,
            # # 8192, the default, causes the error "TypeError: broadcast_in_dim shape must have every element be nonnegative, got (-2, 50)."
            # "min_replay_size": 1000,
            # "use_her": True,
            # "num_envs": 512,
        }

    @property
    def ddpg_hps(self):
        return {
            "num_timesteps": 5_000_000,
            "num_evals": 20,
            "reward_scaling": 10,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "grad_updates_per_step": 4,
            "discounting": 0.97,
            "learning_rate": 3e-4,
            "num_envs": 256,
            "batch_size": 256,
        }

    @property
    def hdqn_hps(self):
        return {
            "num_timesteps": 10_000_000,
            "reward_scaling": 1,
            "num_evals": 50,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "discounting": 0.97,
            "learning_rate": 1e-4,
            "num_envs": 128,
            "batch_size": 256,
            "grad_updates_per_step": 1,
        }

    @property
    def hdcqn_hps(self):
        return {
            **self.hdqn_hps,
            "cost_scaling": 1.0,
            "safety_threshold": 0.0,
        }

    @property
    def hdqn_her_hps(self):
        return {
            "num_timesteps": 5_000_000,
            "reward_scaling": 10.0,
            "num_evals": 50,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "discounting": 0.99,
            "learning_rate": 1e-4,
            "num_envs": 256,
            "batch_size": 256,
            "unroll_length": 62,
            "multiplier_num_sgd_steps": 1,
            "max_devices_per_host": 1,
            "max_replay_size": 10000,
            # 8192, the default, causes the error "TypeError: broadcast_in_dim shape must have every element be nonnegative, got (-2, 50)."
            "min_replay_size": 1000,
            "use_her": True,
            "actor_type": "argmax_safest",
            "network_type": "default",
        }

    @property
    def hdcqn_her_hps(self):
        return {
            **self.hdqn_her_hps,
            "cost_scaling": 1.0,
            "safety_threshold": 0.0,
        }

    @property
    def achql_hps(self):
        return self.hdcqn_her_hps

    @property
    def acddpg_hps(self):
        return {
            "learning_rate": 1e-4,
            "discounting": 0.97,
            "batch_size": 256,
            "normalize_observations": True,
            "reward_scaling": 10.0,
            # target update rate
            "tau": 0.005,
            "min_replay_size": 100,
            "max_replay_size": 10000,
            "deterministic_eval": True,
            "train_step_multiplier": 1,
            "unroll_length": 150,
            "h_dim": 256,
            "n_hidden": 2,
            # layer norm
            "use_ln": True,
            # hindsight experience replay
            "use_her": True,
            "network_type": "old_default",
            # --------------------
            # run params
            "total_env_steps": 5_000_000,
            "episode_length": 150,
            "num_envs": 256,
            "num_eval_envs": 256,
            "num_evals": 50,
            "action_repeat": 1,
            "max_devices_per_host": 1,
        }

    @property
    def qrm_hps(self):
        return {
            "num_timesteps": 10_000_000,
            "reward_scaling": 1,
            "num_evals": 50,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "discounting": 0.99,
            # "learning_rate": 3e-4,
            "num_envs": 256,
            "batch_size": 256,
            "unroll_length": 62,
            "multiplier_num_sgd_steps": 1,
            "max_devices_per_host": 1,
            "max_replay_size": 10000,
            # 8192, the default, causes the error "TypeError: broadcast_in_dim shape must have every element be nonnegative, got (-2, 50)."
            "min_replay_size": 1000,
            "use_her": True,
        }

    @property
    def crm_hps(self):
        return {
            "num_timesteps": 10_000_000,
            "reward_scaling": 1,
            "num_evals": 50,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "discounting": 0.99,
            # "learning_rate": 3e-4,
            "num_envs": 256,
            "batch_size": 256,
            # "unroll_length": 62,
            # "multiplier_num_sgd_steps": 1,
            "max_devices_per_host": 1,
            "max_replay_size": 10000,
            # 8192, the default, causes the error "TypeError: broadcast_in_dim shape must have every element be nonnegative, got (-2, 50)."
            "min_replay_size": 1000,
            # "use_her": True,
        }

    @property
    def hrm_hps(self):
        return {
            "num_timesteps": 10_000_000,
            "reward_scaling": 1,
            "num_evals": 50,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "discounting": 0.99,
            # "learning_rate": 3e-4,
            "num_envs": 256,
            "batch_size": 256,
            # "unroll_length": 62,
            # "multiplier_num_sgd_steps": 1,
            "max_devices_per_host": 1,
            "max_replay_size": 10000,
            # 8192, the default, causes the error "TypeError: broadcast_in_dim shape must have every element be nonnegative, got (-2, 50)."
            "min_replay_size": 1000,
            # "use_her": True,
        }

    @property
    def rm_reward_shaping_hps(self):
        return {
            "num_timesteps": 10_000_000,
            "reward_scaling": 1,
            "num_evals": 50,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "discounting": 0.99,
            # "learning_rate": 3e-4,
            "num_envs": 256,
            "batch_size": 256,
            "unroll_length": 62,
            "multiplier_num_sgd_steps": 1,
            "max_devices_per_host": 1,
            "max_replay_size": 10000,
            # 8192, the default, causes the error "TypeError: broadcast_in_dim shape must have every element be nonnegative, got (-2, 50)."
            "min_replay_size": 1000,
            "use_her": True,
        }

    @property
    def lof_hps(self):
        return {
            # "num_timesteps": 10_000_000,
            # "reward_scaling": 1,
            # "num_evals": 50,
            # "episode_length": 1000,
            # "normalize_observations": True,
            # "action_repeat": 1,
            # "discounting": 0.99,
            # "learning_rate": 3e-4,
            # "num_envs": 256,
            # "batch_size": 256,
            # "unroll_length": 62,
            # "multiplier_num_sgd_steps": 1,
            # "max_devices_per_host": 1,
            # "max_replay_size": 10000,
            # 8192, the default, causes the error "TypeError: broadcast_in_dim shape must have every element be nonnegative, got (-2, 50)."
            # "min_replay_size": 1000,
            # "use_her": True,
        }

    @property
    def crl_hps(self):
        return {
            "num_evals": 50,
            "num_timesteps": 10000000,
            "batch_size": 256,
            "num_envs": 512,
            "discounting": 0.99,
            "action_repeat": 1,
            "episode_length": 1000,
            "unroll_length": 62,
            "min_replay_size": 1000,
            "max_replay_size": 10000,
            "contrastive_loss_fn": "infonce_backward",
            "energy_fn": "l2",
            "multiplier_num_sgd_steps": 1,
        }
