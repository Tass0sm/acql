import random

import numpy as np
import jax.numpy as jnp

from achql.brax.envs.drone_maze import DroneMaze
from achql.brax.envs.base import GoalConditionedEnv
from achql.brax.tasks.base import BraxTaskBase
from achql.brax.tasks.templates import sequence, inside_circle, outside_circle, inside_box
from achql.hierarchy.drone.load import load_hard_coded_drone_options

from achql.stl import Expression, Var
import achql.stl as stl


class DroneMazeTaskBase(BraxTaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=3)
        self.obs_var = Var("obs", idx=0, dim=self.env.observation_size, position=(0, 3))

    def get_options(self):
        return self.get_hard_coded_options()

    def get_hard_coded_options(self):
        return load_hard_coded_drone_options()


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
