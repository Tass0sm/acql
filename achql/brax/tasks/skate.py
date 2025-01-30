import random

import numpy as np
import jax.numpy as jnp

from gymnasium.spaces.utils import flatdim

from achql.brax.envs.skate import Skate
from achql.brax.envs.base import GoalConditionedEnv
from achql.brax.tasks.base import TaskBase
from achql.brax.tasks.templates import sequence, inside_circle, outside_circle

from achql.stl import Expression, Var
import achql.stl.expression_jax2 as stl


class SkateTaskBase(TaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=2)
        self.obs_var = Var("obs", idx=0, dim=flatdim(self.env.observation_space),
                           position=(0, 2))


class SkateStraightSequence(SkateTaskBase):
    """
    F(region1 ^ X(F(region2)))
    """

    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([2.0, 0.0])
        self.goal2_location = jnp.array([6.0, 0.0])
        self.goal_radius = 1.0

        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = Skate(backend=backend)
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        at_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal_radius)
        at_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal_radius)

        phi = stl.STLUntimedEventually(
            stl.STLAnd(at_goal1, stl.STLNext(stl.STLUntimedEventually(at_goal2)))
        )

        return phi


class SkateTurnSequence(SkateTaskBase):
    """
    F(region1 ^ X(F(region2)))
    """

    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([2.0, 0.0])
        self.goal2_location = jnp.array([2.0, 4.0])
        self.goal_radius = 1.0

        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = Skate(backend=backend)
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        at_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal_radius)
        at_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal_radius)

        phi = stl.STLUntimedEventually(
            stl.STLAnd(at_goal1, stl.STLNext(stl.STLUntimedEventually(at_goal2)))
        )

        return phi
