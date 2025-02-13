import random

import numpy as np
import jax.numpy as jnp

from gymnasium.spaces.utils import flatdim

from achql.brax.envs.xy_point_maze import XYPointMaze
from achql.brax.envs.base import GoalConditionedEnv
from achql.brax.tasks.base import BraxTaskBase

from achql.stl import Expression, Var
import achql.stl.expression_jax2 as stl


class XYPointMazeTaskBase(BraxTaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=2)
        self.obs_var = Var("obs", idx=0, dim=flatdim(self.env.observation_space),
                           position=(0, 2))


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
