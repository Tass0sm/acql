import random

import numpy as np
import jax.numpy as jnp

from gymnasium.spaces.utils import flatdim

from jaxgcrl.envs.ant_maze import AntMaze
from task_aware_skill_composition.brax.envs.base import GoalConditionedEnv
from task_aware_skill_composition.brax.tasks.base import TaskBase

from corallab_stl import Expression, Var
import corallab_stl.expression_jax2 as stl


class AntMazeTaskBase(TaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=2)
        self.obs_var = Var("obs", idx=0, dim=self.env.state_dim + 2, position=(0, 2))


class AntMazeNav(AntMazeTaskBase):
    def __init__(self, backend="mjx"):
        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = AntMaze(
            terminate_when_unhealthy=False,
            backend=backend
        )
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        true = stl.STLPredicate(obs_var, lambda s: 999, lower_bound=0.0)
        return true

    @property
    def ppo_hps(self):
        return {
            "num_timesteps": 50_000_000,
            "num_evals": 20,
            "reward_scaling": 5,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "unroll_length": 128,
            "num_minibatches": 32,
            "num_updates_per_batch": 4,
            "discounting": 0.95,
            "learning_rate": 3e-4,
            "entropy_cost": 1e-2,
            "num_envs": 2048,
            "batch_size": 256,
        }
