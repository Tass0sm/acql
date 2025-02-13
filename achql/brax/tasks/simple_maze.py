import random

import numpy as np
import jax.numpy as jnp

from achql.brax.envs.simple_maze import SimpleMaze
from achql.brax.envs.base import GoalConditionedEnv
from achql.brax.tasks.base import BraxTaskBase
from achql.brax.tasks.templates import sequence, inside_circle, outside_circle, inside_box, true_exp
from achql.brax.tasks.mixins import *
from achql.hierarchy.xy_point.load import load_hard_coded_xy_point_options

from achql.stl import Expression, Var
import achql.stl.expression_jax2 as stl


class SimpleMazeTaskBase(BraxTaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = SimpleMaze(
            terminate_when_unhealthy=False,
            maze_layout_name="open_maze",
            backend=backend
        )
        return env

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=2)
        self.obs_var = Var("obs", idx=0, dim=self.env.observation_size, position=(0, 2))

    def get_options(self):
        return self.get_hard_coded_options()

    def get_hard_coded_options(self):
        return load_hard_coded_xy_point_options()


class SimpleMazeNav(SimpleMazeTaskBase):
    def __init__(self, backend="mjx"):
        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = SimpleMaze(
            terminate_when_unhealthy=False,
            backend=backend
        )
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        return true_exp(obs_var)


class SimpleMazeCenterConstraint(CenterConstraintMixin, SimpleMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.obs1_location = jnp.array([8.0, 8.0])
        self.obs_radius = 2.0

        super().__init__(None, 1000, backend=backend)


class SimpleMazeUMazeConstraint(UMazeConstraintMixin, SimpleMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.obs_corners = (jnp.array([6.0, 2.0]), jnp.array([10.0, 10.0]))

        super().__init__(None, 1000, backend=backend)


class SimpleMazeSingleSubgoal(SingleSubgoalMixin, SimpleMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([12.0, 4.0])
        self.goal1_radius = 2.0

        super().__init__(None, 1000, backend=backend)


class SimpleMazeTwoSubgoals(TwoSubgoalsMixin, SimpleMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([12.0, 4.0])
        self.goal1_radius = 2.0
        self.goal2_location = jnp.array([4.0, 12.0])
        self.goal2_radius = 2.0

        super().__init__(None, 1000, backend=backend)


class SimpleMazeBranching1(Branching1Mixin, SimpleMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([12.0, 4.0])
        self.goal1_radius = 2.0
        self.goal2_location = jnp.array([4.0, 12.0])
        self.goal2_radius = 2.0

        super().__init__(None, 1000, backend=backend)


class SimpleMazeBranching2(Branching2Mixin, SimpleMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([12.0, 4.0])
        self.goal1_radius = 2.0
        self.goal2_location = jnp.array([4.0, 12.0])
        self.goal2_radius = 2.0

        super().__init__(None, 1000, backend=backend)


class SimpleMazeObligationConstraint1(ObligationConstraint1Mixin, SimpleMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.obs1_corners = (jnp.array([6.0, 2.0]), jnp.array([10.0, 10.0]))

        self.goal1_location = jnp.array([12.0, 4.0])
        self.goal1_radius = 2.0

        super().__init__(None, 1000, backend=backend)

    @property
    def hdcqn_her_hps(self):
        return {
            **self.hdqn_her_hps,
            "cost_scaling": 1.0,
            "safety_threshold": 0.0,
            "gamma_end_value": 0.93,
        }

class SimpleMazeObligationConstraint2(ObligationConstraint2Mixin, SimpleMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.obs1_location = jnp.array([8.0, 8.0])
        self.obs_radius = 2.0

        self.goal1_location = jnp.array([12.0, 4.0])
        self.goal1_radius = 2.0

        super().__init__(None, 1000, backend=backend)


class SimpleMazeObligationConstraint3(ObligationConstraint3Mixin, SimpleMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.obs1_location = jnp.array([8.0, 8.0])
        self.obs_radius = 2.0

        self.goal1_location = jnp.array([12.0, 12.0])
        self.goal1_radius = 2.0

        super().__init__(None, 1000, backend=backend)


class SimpleMazeObligationConstraint4(SimpleMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.obs1_corners = (jnp.array([6.0, 2.0]), jnp.array([10.0, 10.0]))

        self.goal1_location = jnp.array([12.0, 4.0])
        self.goal1_radius = 2.0
        self.goal2_location = jnp.array([4.0, 12.0])
        self.goal2_radius = 2.0
        self.goal3_location = jnp.array([12.0, 12.0])
        self.goal3_radius = 2.0

        super().__init__(None, 1000, backend=backend)

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        at_obs1 = inside_box(obs_var.position, *self.obs1_corners)
        phi_safety = stl.STLUntimedAlways(stl.STLNegation(at_obs1))

        in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal1_radius, has_goal=True)
        in_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal2_radius, has_goal=True)
        in_goal3 = inside_circle(obs_var.position, self.goal3_location, self.goal3_radius, has_goal=True)
        phi_liveness = stl.STLUntimedEventually(
            stl.STLAnd(in_goal1, stl.STLNext(stl.STLUntimedEventually(in_goal2)))
        )

        phi = stl.STLAnd(phi_liveness, phi_safety)
        return phi

class SimpleMazeObligationConstraint5(ObligationConstraint1Mixin, SimpleMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.obs1_corners = (jnp.array([2.0, 6.0]), jnp.array([10.0, 10.0]))

        self.goal1_location = jnp.array([4.0, 12.0])
        self.goal1_radius = 2.0

        super().__init__(None, 1000, backend=backend)


class SimpleMazeUntil1(SimpleMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.obs1_location = jnp.array([8.0, 8.0])
        self.obs1_radius = 2.0

        self.goal1_location = jnp.array([12.0, 12.0])
        self.goal1_radius = 2.0
        self.goal2_location = jnp.array([4.0, 4.0])
        self.goal2_radius = 2.0

        super().__init__(None, 1000, backend=backend)

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        not_in_obs1 = stl.STLNegation(inside_circle(obs_var.position, self.obs1_location, self.obs1_radius))
        in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal1_radius, has_goal=True)
        in_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal2_radius, has_goal=True)

        phi = stl.STLUntimedUntil(
            not_in_obs1,
            stl.STLAnd(in_goal1,
                       stl.STLNext(stl.STLUntimedEventually(in_goal2)))
        )
        return phi
