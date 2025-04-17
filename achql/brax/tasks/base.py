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
            "num_timesteps": 10_000_000,
            "reward_scaling": 1,
            "num_evals": 50,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "discounting": 0.99,
            "learning_rate": 5e-5,
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
    def ddpg_hps(self):
        return {
            "num_timesteps": 10_000_000,
            "num_evals": 50,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "discounting": 0.97,
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
