from abc import ABC, abstractmethod

# import numpy as np

from task_aware_skill_composition.brax.envs.base import GoalConditionedEnv
from corallab_stl import Expression, Var


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
