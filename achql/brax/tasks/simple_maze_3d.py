import random

import numpy as np
import jax.numpy as jnp

from achql.brax.envs.simple_maze_3d import SimpleMaze3D
from achql.brax.envs.base import GoalConditionedEnv
from achql.brax.tasks.base import BraxTaskBase
from achql.brax.tasks.templates import sequence, inside_circle, outside_circle, inside_box, true_exp
from achql.brax.tasks.mixins import *
from achql.hierarchy.xyz_point.load import load_hard_coded_xyz_point_options
from achql.hierarchy.option import FixedLengthTerminationPolicy

from achql.stl import Expression, Var
import achql.stl as stl


class SimpleMaze3DTaskBase(BraxTaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = SimpleMaze3D(
            terminate_when_unhealthy=False,
            backend=backend
        )
        return env

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=3)
        self.obs_var = Var("obs", idx=0, dim=self.env.state_dim, position=(0, 3))

    def get_options(self):
        return self.get_hard_coded_options()

    def get_hard_coded_options(self):
        return load_hard_coded_xyz_point_options(termination_policy=FixedLengthTerminationPolicy(5))

    @property
    def crm_hps(self):
        return super().crm_hps | { "episode_length": 200,
                                   "multiplier_num_sgd_steps": 4 }
    @property
    def achql_hps(self):
        return super().achql_hps | { "episode_length": 200,
                                     "multiplier_num_sgd_steps": 4,
                                     "cost_scaling": 5.0 }



class SimpleMaze3DNav(SimpleMaze3DTaskBase):
    def __init__(self, backend="mjx"):
        super().__init__(None, 1000, backend=backend)

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        return true_exp(obs_var)


class SimpleMaze3DCenterConstraint(CenterConstraintMixin, SimpleMaze3DTaskBase):
    def __init__(self, backend="mjx"):
        self.obs1_location = jnp.array([8.0, 8.0, 6.0])
        self.obs_radius = 2.0

        super().__init__(None, 1000, backend=backend)


class SimpleMaze3DUMazeConstraint(UMazeConstraintMixin, SimpleMaze3DTaskBase):
    def __init__(self, backend="mjx"):
        self.obs_corners = (jnp.array([6.0, 2.0, 0.0]), jnp.array([10.0, 10.0, 8.0]))

        super().__init__(None, 1000, backend=backend)


class SimpleMaze3DSingleSubgoal(SingleSubgoalMixin, SimpleMaze3DTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([12.0, 8.0, 10.0])
        self.goal1_radius = 2.0

        super().__init__(None, 1000, backend=backend)


class SimpleMaze3DTwoSubgoals(TwoSubgoalsMixin, SimpleMaze3DTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([12.0, 4.0, 2.0])
        self.goal1_radius = 2.0
        self.goal2_location = jnp.array([4.0, 12.0, 10.0])
        self.goal2_radius = 2.0

        super().__init__(None, 1000, backend=backend)


class SimpleMaze3DTwoSubgoals2(TwoSubgoalsMixin, SimpleMaze3DTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([12.0, 12.0, 10.0])
        self.goal1_radius = 2.0
        self.goal2_location = jnp.array([4.0, 4.0, 2.0])
        self.goal2_radius = 2.0

        super().__init__(None, 1000, backend=backend)


class SimpleMaze3DBranching1(Branching1Mixin, SimpleMaze3DTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([12.0, 4.0, 10.0])
        self.goal1_radius = 2.0
        self.goal2_location = jnp.array([4.0, 12.0, 2.0])
        self.goal2_radius = 2.0

        super().__init__(None, 1000, backend=backend)


class SimpleMaze3DObligationConstraint1(ObligationConstraint1Mixin, SimpleMaze3DTaskBase):
    def __init__(self, backend="mjx"):
        self.obs1_corners = (jnp.array([6.0, 2.0, 0.0]), jnp.array([10.0, 10.0, 8.0]))

        self.goal1_location = jnp.array([12.0, 12.0, 12.0])
        self.goal1_radius = 2.0

        super().__init__(None, 1000, backend=backend)


class SimpleMaze3DObligationConstraint2(ObligationConstraint2Mixin, SimpleMaze3DTaskBase):
    def __init__(self, backend="mjx"):
        self.obs1_location = jnp.array([8.0, 8.0, 6.0])
        self.obs_radius = 2.0

        self.goal1_location = jnp.array([12.0, 12.0, 12.0])
        self.goal1_radius = 2.0

        super().__init__(None, 1000, backend=backend)


class SimpleMaze3DObligationConstraint3(ObligationConstraint3Mixin, SimpleMaze3DTaskBase):
    def __init__(self, backend="mjx"):
        self.obs1_location = jnp.array([8.0, 8.0, 6.0])
        self.obs_radius = 2.0

        self.goal1_location = jnp.array([8.0, 8.0, 10.0])
        self.goal1_radius = 2.0

        super().__init__(None, 1000, backend=backend)


class SimpleMaze3DUntil1(Until1Mixin, SimpleMaze3DTaskBase):
    def __init__(self, backend="mjx"):
        self.obs1_location = jnp.array([8.0, 8.0, 6.0])
        self.obs1_radius = 2.0

        self.goal1_location = jnp.array([12.0, 12.0, 12.0])
        self.goal1_radius = 2.0
        self.goal2_location = jnp.array([4.0, 4.0, 2.0])
        self.goal2_radius = 2.0

        super().__init__(None, 1000, backend=backend)

    def get_hard_coded_options(self):
        return load_hard_coded_xyz_point_options(termination_policy=FixedLengthTerminationPolicy(10))

    @property
    def crm_hps(self):
        return super().crm_hps | { "episode_length": 100 }

    @property
    def achql_hps(self):
        return super().achql_hps | { "episode_length": 100 }


class SimpleMaze3DUntil2(Until1Mixin, SimpleMaze3DTaskBase):
    def __init__(self, backend="mjx"):
        self.obs1_location = jnp.array([8.0, 8.0, 6.0])
        self.obs1_radius = 2.0

        self.goal1_location = jnp.array([12.0, 8.0, 6.0])
        self.goal1_radius = 2.0
        self.goal2_location = jnp.array([4.0, 8.0, 6.0])
        self.goal2_radius = 2.0

        super().__init__(None, 1000, backend=backend)

    def get_hard_coded_options(self):
        return load_hard_coded_xyz_point_options(termination_policy=FixedLengthTerminationPolicy(10))

    @property
    def crm_hps(self):
        return super().crm_hps | { "episode_length": 100 }

    @property
    def achql_hps(self):
        return super().achql_hps | { "episode_length": 100 }


class SimpleMaze3DLoop(LoopMixin, SimpleMaze3DTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([12.0, 12.0, 10.0])
        self.goal1_radius = 2.0
        self.goal2_location = jnp.array([4.0, 4.0, 2.0])
        self.goal2_radius = 2.0

        super().__init__(None, 1000, backend=backend)


class SimpleMaze3DLoopWithObs(LoopWithObsMixin, SimpleMaze3DTaskBase):
    def __init__(self, backend="mjx"):
        self.obs1_location = jnp.array([8.0, 8.0, 6.0])
        self.obs1_radius = 2.0

        self.goal1_location = jnp.array([12.0, 12.0, 10.0])
        self.goal1_radius = 2.0
        self.goal2_location = jnp.array([4.0, 4.0, 2.0])
        self.goal2_radius = 2.0

        super().__init__(None, 1000, backend=backend)


# class SimpleMazeCenterConstraint(SimpleMazeTaskBase):
#     def __init__(self, backend="mjx"):
#         self.obs1_location = jnp.array([8.0, 8.0])
#         self.obs_radius = 2.0

#         super().__init__(None, 1000, backend=backend)

#     def _build_env(self, backend: str) -> GoalConditionedEnv:
#         env = SimpleMaze(
#             terminate_when_unhealthy=False,
#             maze_layout_name="open_maze",
#             backend=backend
#         )
#         return env

#     def _build_hi_spec(self, wp_var: Var) -> Expression:
#         pass

#     def _build_lo_spec(self, obs_var: Var) -> Expression:
#         at_obs1 = inside_circle(obs_var.position, self.obs1_location, self.obs_radius)
#         phi = stl.STLUntimedAlways(stl.STLNegation(at_obs1))
#         return phi


# class SimpleMazeUMazeConstraint(SimpleMazeTaskBase):
#     def __init__(self, backend="mjx"):
#         self.obs_corners = (jnp.array([6.0, 2.0]), jnp.array([10.0, 10.0]))

#         super().__init__(None, 1000, backend=backend)

#     def _build_env(self, backend: str) -> GoalConditionedEnv:
#         env = SimpleMaze(
#             terminate_when_unhealthy=False,
#             maze_layout_name="open_maze",
#             backend=backend
#         )
#         return env

#     def _build_hi_spec(self, wp_var: Var) -> Expression:
#         pass

#     def _build_lo_spec(self, obs_var: Var) -> Expression:
#         at_obs1 = inside_box(obs_var.position, *self.obs_corners)
#         phi = stl.STLUntimedAlways(stl.STLNegation(at_obs1))
#         return phi


# class SimpleMazeSingleSubgoal(SimpleMazeTaskBase):
#     def __init__(self, backend="mjx"):
#         self.goal1_location = jnp.array([12.0, 4.0])
#         self.goal1_radius = 2.0

#         super().__init__(None, 1000, backend=backend)

#     def _build_env(self, backend: str) -> GoalConditionedEnv:
#         env = SimpleMaze(
#             terminate_when_unhealthy=False,
#             maze_layout_name="open_maze",
#             backend=backend
#         )
#         return env

#     def _build_hi_spec(self, wp_var: Var) -> Expression:
#         pass

#     def _build_lo_spec(self, obs_var: Var) -> Expression:
#         in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal1_radius)
#         phi = stl.STLUntimedEventually(in_goal1)
#         return phi

#     @property
#     def hdcqn_her_hps(self):
#         return {
#             "num_timesteps": 10_000_000,
#             "reward_scaling": 1,
#             "cost_scaling": 1,
#             "cost_budget": -1.0,
#             "num_evals": 50,
#             "episode_length": 1000,
#             "normalize_observations": True,
#             "action_repeat": 1,
#             "discounting": 0.99,
#             # "learning_rate": 3e-4,
#             "num_envs": 256,
#             "batch_size": 256,
#             "unroll_length": 62,
#             "multiplier_num_sgd_steps": 1,
#             "max_devices_per_host": 1,
#             "max_replay_size": 10000,
#             # 8192, the default, causes the error "TypeError: broadcast_in_dim shape must have every element be nonnegative, got (-2, 50)."
#             "min_replay_size": 1000,
#             "use_her": True,
#         }


# class SimpleMazeTwoSubgoals(SimpleMazeTaskBase):
#     def __init__(self, backend="mjx"):
#         self.goal1_location = jnp.array([12.0, 4.0])
#         self.goal1_radius = 2.0
#         self.goal2_location = jnp.array([4.0, 12.0])
#         self.goal2_radius = 2.0

#         super().__init__(None, 1000, backend=backend)

#     def _build_env(self, backend: str) -> GoalConditionedEnv:
#         env = SimpleMaze(
#             terminate_when_unhealthy=False,
#             maze_layout_name="open_maze",
#             backend=backend
#         )
#         return env

#     def _build_hi_spec(self, wp_var: Var) -> Expression:
#         pass

#     def _build_lo_spec(self, obs_var: Var) -> Expression:
#         in_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal1_radius)
#         in_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal2_radius)
#         phi = stl.STLUntimedEventually(
#             stl.STLAnd(in_goal1, stl.STLNext(stl.STLUntimedEventually(in_goal2)))
#         )
#         return phi


# class SimpleMazeObligationConstraint1(SimpleMazeTaskBase):
#     def __init__(self, backend="mjx"):
#         self.obs1_corners = (jnp.array([6.0, 2.0]), jnp.array([10.0, 10.0]))

#         self.goal1_location = jnp.array([12.0, 4.0])
#         self.goal1_radius = 2.0

#         super().__init__(None, 1000, backend=backend)

#     def _build_env(self, backend: str) -> GoalConditionedEnv:
#         env = SimpleMaze(
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
#         phi_liveness = stl.STLUntimedEventually(in_goal1)

#         phi = stl.STLAnd(phi_liveness, phi_safety)
#         return phi
