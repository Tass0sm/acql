import random

import numpy as np
import jax.numpy as jnp

from jaxgcrl.envs.simple_maze import SimpleMaze
from task_aware_skill_composition.brax.envs.base import GoalConditionedEnv
from task_aware_skill_composition.brax.tasks.base import TaskBase
from task_aware_skill_composition.brax.tasks.templates import sequence, inside_circle, outside_circle, inside_box, true_exp
from task_aware_skill_composition.hierarchy.xy_point.load import load_hard_coded_xy_point_options

from corallab_stl import Expression, Var
import corallab_stl.expression_jax2 as stl


class SimpleMazeTaskBase(TaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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


class SimpleMazeCenterConstraint(SimpleMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.obs1_location = jnp.array([8.0, 8.0])
        self.obs_radius = 2.0

        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = SimpleMaze(
            terminate_when_unhealthy=False,
            maze_layout_name="open_maze",
            backend=backend
        )
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        at_obs1 = inside_circle(obs_var.position, self.obs1_location, self.obs_radius)
        phi = stl.STLUntimedAlways(stl.STLNegation(at_obs1))
        return phi


class SimpleMazeUMazeConstraint(SimpleMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.obs_corners = (jnp.array([6.0, 2.0]), jnp.array([10.0, 10.0]))

        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = SimpleMaze(
            terminate_when_unhealthy=False,
            maze_layout_name="open_maze",
            backend=backend
        )
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        at_obs1 = inside_box(obs_var.position, *self.obs_corners)
        phi = stl.STLUntimedAlways(stl.STLNegation(at_obs1))
        return phi


class SimpleMazeSingleSubgoal(SimpleMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([12.0, 4.0])
        self.goal1_radius = 2.0

        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = SimpleMaze(
            terminate_when_unhealthy=False,
            maze_layout_name="open_maze",
            backend=backend
        )
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal1_radius)
        phi = stl.STLUntimedEventually(in_goal1)
        return phi


class SimpleMazeTwoSubgoals(SimpleMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([12.0, 4.0])
        self.goal1_radius = 2.0
        self.goal2_location = jnp.array([4.0, 12.0])
        self.goal2_radius = 2.0

        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = SimpleMaze(
            terminate_when_unhealthy=False,
            maze_layout_name="open_maze",
            backend=backend
        )
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal1_radius)
        in_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal2_radius)
        phi = stl.STLUntimedEventually(
            stl.STLAnd(in_goal1, stl.STLNext(stl.STLUntimedEventually(in_goal2)))
        )
        return phi


class SimpleMazeBranching1(SimpleMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([12.0, 4.0])
        self.goal1_radius = 2.0
        self.goal2_location = jnp.array([4.0, 12.0])
        self.goal2_radius = 2.0

        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = SimpleMaze(
            terminate_when_unhealthy=False,
            maze_layout_name="open_maze",
            backend=backend
        )
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal1_radius)
        in_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal2_radius)
        phi = stl.STLAnd(stl.STLUntimedEventually(in_goal1),
                         stl.STLUntimedEventually(in_goal2))
        return phi


class SimpleMazeBranching2(SimpleMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([12.0, 4.0])
        self.goal1_radius = 2.0
        self.goal2_location = jnp.array([4.0, 12.0])
        self.goal2_radius = 2.0

        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = SimpleMaze(
            terminate_when_unhealthy=False,
            maze_layout_name="open_maze",
            backend=backend
        )
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal1_radius)
        in_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal2_radius)
        phi = stl.STLOr(stl.STLUntimedEventually(in_goal1),
                        stl.STLUntimedEventually(in_goal2))
        return phi


class SimpleMazeObligationConstraint1(SimpleMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.obs1_corners = (jnp.array([6.0, 2.0]), jnp.array([10.0, 10.0]))

        self.goal1_location = jnp.array([12.0, 4.0])
        self.goal1_radius = 2.0

        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = SimpleMaze(
            terminate_when_unhealthy=False,
            maze_layout_name="open_maze",
            backend=backend
        )
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        at_obs1 = inside_box(obs_var.position, *self.obs1_corners)
        phi_safety = stl.STLUntimedAlways(stl.STLNegation(at_obs1))

        in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal1_radius)
        phi_liveness = stl.STLUntimedEventually(in_goal1)

        phi = stl.STLAnd(phi_liveness, phi_safety)
        return phi

    @property
    def hdcqn_her_hps(self):
        return {
            **self.hdqn_her_hps,
            "cost_scaling": 1.0,
            "safety_minimum": -0.2,
        }


class SimpleMazeObligationConstraint2(SimpleMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.obs1_corners = (jnp.array([6.0, 2.0]), jnp.array([10.0, 10.0]))

        self.goal1_location = jnp.array([12.0, 4.0])
        self.goal1_radius = 2.0
        self.goal2_location = jnp.array([4.0, 12.0])
        self.goal2_radius = 2.0

        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = SimpleMaze(
            terminate_when_unhealthy=False,
            maze_layout_name="open_maze",
            backend=backend
        )
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        at_obs1 = inside_box(obs_var.position, *self.obs1_corners)
        phi_safety = stl.STLUntimedAlways(stl.STLNegation(at_obs1))

        in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal1_radius)
        in_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal2_radius)
        phi_liveness = stl.STLUntimedEventually(
            stl.STLAnd(in_goal1, stl.STLNext(stl.STLUntimedEventually(in_goal2)))
        )

        phi = stl.STLAnd(phi_liveness, phi_safety)
        return phi

class SimpleMazeObligationConstraint3(SimpleMazeTaskBase):
    def __init__(self, backend="mjx"):
        self.obs1_corners = (jnp.array([6.0, 2.0]), jnp.array([10.0, 10.0]))

        self.goal1_location = jnp.array([12.0, 4.0])
        self.goal1_radius = 2.0
        self.goal2_location = jnp.array([4.0, 12.0])
        self.goal2_radius = 2.0
        self.goal3_location = jnp.array([12.0, 12.0])
        self.goal3_radius = 2.0

        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = SimpleMaze(
            terminate_when_unhealthy=False,
            maze_layout_name="open_maze",
            backend=backend
        )
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        at_obs1 = inside_box(obs_var.position, *self.obs1_corners)
        phi_safety = stl.STLUntimedAlways(stl.STLNegation(at_obs1))

        in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal1_radius)
        in_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal2_radius)
        in_goal3 = inside_circle(obs_var.position, self.goal3_location, self.goal3_radius)
        phi_liveness = stl.STLUntimedEventually(
            stl.STLAnd(in_goal1, stl.STLNext(stl.STLUntimedEventually(in_goal2)))
        )

        phi = stl.STLAnd(phi_liveness, phi_safety)
        return phi
