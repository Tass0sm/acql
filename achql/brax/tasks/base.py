from abc import ABC, abstractmethod

# import numpy as np

from achql.brax.envs.base import GoalConditionedEnv
from achql.stl import Expression, Var


class TaskBase(ABC):
    def __init__(self, path_max_len: int, time_limit: int, backend: str):
        self.path_max_len = path_max_len
        self.time_limit = time_limit
        self.env = self._build_env(backend)

        # Stateful and ugly but oh well
        self._create_vars()

        # high level constraints
        self.hi_spec = self._build_hi_spec(self.wp_var)

        # low level constraints
        self.lo_spec = self._build_lo_spec(self.obs_var)

    @abstractmethod
    def _build_env(self) -> GoalConditionedEnv:
        raise NotImplementedError()

    @abstractmethod
    def _create_vars(self):
        raise NotImplementedError()

    @abstractmethod
    def _build_hi_spec(self, wp_var: Var) -> Expression:
        raise NotImplementedError()

    @abstractmethod
    def _build_lo_spec(self, obs_var: Var) -> Expression:
        raise NotImplementedError()

    # @property
    # def rm_config(self) -> dict:
    #     raise NotImplementedError()

    # @abstractmethod
    # def demo(self, n_trajs: int) -> jnp.ndarray:
    #     raise NotImplementedError()

    # def __repr__(self):
    #     return repr(self.spec)

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
            "num_evals": 20,
            "reward_scaling": 1,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "discounting": 0.99,
            "learning_rate": 6e-4,
            "num_envs": 256,
            "batch_size": 512,
            "grad_updates_per_step": 64,
            "max_devices_per_host": 1,
            "max_replay_size": 1048576,
            "min_replay_size": 8192,
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
    def hdcqn_her_hps(self):
        return {
            **self.hdqn_her_hps,
            "cost_scaling": 1.0,
            "safety_threshold": 0.0,
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
