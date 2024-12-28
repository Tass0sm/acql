import random

import numpy as np
import jax.numpy as jnp

from task_aware_skill_composition.brax.envs.drone_maze import DroneMaze
from task_aware_skill_composition.brax.envs.base import GoalConditionedEnv
from task_aware_skill_composition.brax.tasks.base import TaskBase
from task_aware_skill_composition.brax.tasks.templates import sequence, inside_circle, outside_circle, inside_box
from task_aware_skill_composition.hierarchy.drone.load import load_hard_coded_drone_options

from corallab_stl import Expression, Var
import corallab_stl.expression_jax2 as stl


class DroneMazeTaskBase(TaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=3)
        self.obs_var = Var("obs", idx=0, dim=self.env.observation_size, position=(0, 3))

    def get_options(self):
        return self.get_hard_coded_options()

    def get_hard_coded_options(self):
        return load_hard_coded_drone_options()

    @property
    def hdqn_her_hps(self):
        return {
            "num_timesteps": 1_000_000,
            "reward_scaling": 1,
            "num_evals": 50,
            "episode_length": 100,
            "normalize_observations": True,
            "action_repeat": 1,
            "discounting": 0.99,
            # "learning_rate": 3e-4,
            "num_envs": 64,
            "batch_size": 64,
            "unroll_length": 62,
            "multiplier_num_sgd_steps": 1,
            "max_devices_per_host": 1,
            "max_replay_size": 5000,
            # 8192, the default, causes the error "TypeError: broadcast_in_dim shape must have every element be nonnegative, got (-2, 50)."
            "min_replay_size": 1000,
            "use_her": True,
        }


class DroneMazeNav(DroneMazeTaskBase):
    def __init__(self, backend="mjx"):
        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = DroneMaze(
            terminate_when_unhealthy=False,
            backend=backend
        )
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        true = stl.STLPredicate(obs_var, lambda s: 999, lower_bound=0.0)
        return true
