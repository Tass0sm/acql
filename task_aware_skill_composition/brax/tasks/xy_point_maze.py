import random

import numpy as np
import jax.numpy as jnp

from gymnasium.spaces.utils import flatdim

from task_aware_skill_composition.brax.envs.xy_point_maze import XYPointMaze
from task_aware_skill_composition.brax.envs.base import GoalConditionedEnv
from task_aware_skill_composition.brax.tasks.base import TaskBase

from corallab_stl import Expression, Var
import corallab_stl.expression_jax2 as stl


class XYPointMazeTaskBase(TaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=2)
        self.obs_var = Var("obs", idx=0, dim=flatdim(self.env.observation_space),
                           position=(0, 2))

    @property
    def ppo_hps(self):
        return {
            "num_timesteps": 50_000_000,
            "num_evals": 20,
            "reward_scaling": 5,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "unroll_length": 5,
            "num_minibatches": 32,
            "num_updates_per_batch": 4,
            "gae_lambda": 0.5,
            "discounting": 0.99,
            "learning_rate": 3e-4,
            "entropy_cost": 5e-1,
            "num_envs": 4096,
            "batch_size": 2048,
        }

    @property
    def sac_hps(self):
        return {
            "num_timesteps": 6_553_600,
            "num_evals": 20,
            "reward_scaling": 30,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "discounting": 0.997,
            "learning_rate": 6e-4,
            "num_envs": 128,
            "batch_size": 512,
            "grad_updates_per_step": 64,
            "max_devices_per_host": 1,
            "max_replay_size": 1048576,
            "min_replay_size": 8192
        }


class XYPointMazeNav(XYPointMazeTaskBase):
    def __init__(self, backend="mjx"):
        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = XYPointMaze(backend=backend)
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        true = stl.STLPredicate(obs_var, lambda s: 999, lower_bound=0.0)
        return true
