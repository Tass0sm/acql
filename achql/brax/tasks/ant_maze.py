import random

import numpy as np
import jax.numpy as jnp

from gymnasium.spaces.utils import flatdim

from achql.brax.envs.ant_maze import AntMaze
from achql.brax.envs.base import GoalConditionedEnv
from achql.brax.tasks.base import BraxTaskBase
from achql.brax.tasks.templates import sequence, inside_circle, outside_circle, inside_box, true_exp
from achql.brax.tasks.mixins import *
from achql.hierarchy.ant.load import load_ant_options
from achql.hierarchy.option import FixedLengthTerminationPolicy

from achql.stl import Expression, Var
import achql.stl.expression_jax2 as stl


class AntMazeTaskBase(BraxTaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = AntMaze(
            terminate_when_unhealthy=False,
            maze_layout_name="open_maze",
            backend=backend
        )
        return env

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=2)
        self.obs_var = Var("obs", idx=0, dim=self.env.observation_size, position=(0, 2))

    def get_options(self):
        return self.get_learned_options()

    def get_learned_options(self):
        adapter = lambda x: x[..., 2:-2]
        return load_ant_options(termination_policy=FixedLengthTerminationPolicy(3),
                                # termination_prob=0.3,
                                adapter=adapter)

    def get_hard_coded_options(self):
        raise NotImplementedError()


class AntMazeNav(AntMazeTaskBase):
    def __init__(self, backend="mjx"):
        super().__init__(None, 1000, backend=backend)

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        return true_exp(obs_var)


class AntMazeUMazeConstraint(UMazeConstraintMixin, AntMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.obs_corners = (jnp.array([6.0, 2.0]), jnp.array([10.0, 10.0]))

        super().__init__(None, 1000, backend=backend)


class AntMazeSingleSubgoal(SingleSubgoalMixin, AntMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([12.0, 4.0])
        self.goal1_radius = 2.0

        super().__init__(None, 1000, backend=backend)


class AntMazeTwoSubgoals(TwoSubgoalsMixin, AntMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([12.0, 4.0])
        self.goal1_radius = 2.0
        self.goal2_location = jnp.array([4.0, 12.0])
        self.goal2_radius = 2.0

        super().__init__(None, 1000, backend=backend)


class AntMazeBranching1(Branching1Mixin, AntMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([12.0, 4.0])
        self.goal1_radius = 2.0
        self.goal2_location = jnp.array([4.0, 12.0])
        self.goal2_radius = 2.0

        super().__init__(None, 1000, backend=backend)


class AntMazeBranching2(Branching2Mixin, AntMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([12.0, 4.0])
        self.goal1_radius = 2.0
        self.goal2_location = jnp.array([4.0, 12.0])
        self.goal2_radius = 2.0

        super().__init__(None, 1000, backend=backend)


class AntMazeObligationConstraint1(ObligationConstraint1Mixin, AntMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.obs1_corners = (jnp.array([6.0, 2.0]), jnp.array([10.0, 10.0]))

        self.goal1_location = jnp.array([12.0, 4.0])
        self.goal1_radius = 2.0

        super().__init__(None, 1000, backend=backend)


class AntMazeObligationConstraint2(ObligationConstraint2Mixin, AntMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.obs1_location = jnp.array([8.0, 8.0])
        self.obs_radius = 2.0

        self.goal1_location = jnp.array([12.0, 4.0])
        self.goal1_radius = 2.0

        super().__init__(None, 1000, backend=backend)


class AntMazeObligationConstraint3(ObligationConstraint3Mixin, AntMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.obs1_location = jnp.array([8.0, 8.0])
        self.obs_radius = 2.0

        self.goal1_location = jnp.array([12.0, 12.0])
        self.goal1_radius = 2.0

        super().__init__(None, 1000, backend=backend)


# class AntMazeObligationConstraint2(ObligationConstraint2Mixin, AntMazeTaskBase):
#     def __init__(self, backend="mjx"):
#         self.obs1_corners = (jnp.array([6.0, 2.0]), jnp.array([10.0, 10.0]))

#         self.goal1_location = jnp.array([12.0, 4.0])
#         self.goal1_radius = 2.0
#         self.goal2_location = jnp.array([4.0, 12.0])
#         self.goal2_radius = 2.0

#         super().__init__(None, 1000, backend=backend)

#     def _build_env(self, backend: str) -> GoalConditionedEnv:
#         env = AntMaze(
#             terminate_when_unhealthy=False,
#             maze_layout_name="open_maze",
#             backend=backend
#         )
#         return env

#     def _build_hi_spec(self, wp_var: Var) -> Expression:
#         pass

#     def _build_lo_spec(self, obs_var: Var) -> Expression:
#         at_obs1 = inside_box(obs_var.position, *self.obs1_corners)
#         phi_safety = stl.STLUntimedAlways(stl.STLNegation(at_obs1))

#         in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal1_radius)
#         in_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal2_radius)
#         phi_liveness = stl.STLUntimedEventually(
#             stl.STLAnd(in_goal1, stl.STLNext(stl.STLUntimedEventually(in_goal2)))
#         )

#         phi = stl.STLAnd(phi_liveness, phi_safety)
#         return phi


# class AntMazeObligationConstraint3(ObligationConstraint3Mixin, AntMazeTaskBase):
#     def __init__(self, backend="mjx"):
#         self.obs1_corners = (jnp.array([6.0, 2.0]), jnp.array([10.0, 10.0]))

#         self.goal1_location = jnp.array([12.0, 4.0])
#         self.goal1_radius = 2.0
#         self.goal2_location = jnp.array([4.0, 12.0])
#         self.goal2_radius = 2.0
#         self.goal3_location = jnp.array([12.0, 12.0])
#         self.goal3_radius = 2.0

#         super().__init__(None, 1000, backend=backend)

#     def _build_env(self, backend: str) -> GoalConditionedEnv:
#         env = AntMaze(
#             terminate_when_unhealthy=False,
#             maze_layout_name="open_maze",
#             backend=backend
#         )
#         return env

#     def _build_hi_spec(self, wp_var: Var) -> Expression:
#         pass

#     def _build_lo_spec(self, obs_var: Var) -> Expression:
#         at_obs1 = inside_box(obs_var.position, *self.obs1_corners)
#         phi_safety = stl.STLUntimedAlways(stl.STLNegation(at_obs1))

#         in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal1_radius)
#         in_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal2_radius)
#         in_goal3 = inside_circle(obs_var.position, self.goal3_location, self.goal3_radius)
#         phi_liveness = stl.STLUntimedEventually(
#             stl.STLAnd(in_goal1, stl.STLNext(stl.STLUntimedEventually(in_goal2)))
#         )

#         phi = stl.STLAnd(phi_liveness, phi_safety)
#         return phi
