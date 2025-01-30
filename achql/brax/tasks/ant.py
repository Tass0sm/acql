import random

import numpy as np
import jax.numpy as jnp

from gymnasium.spaces.utils import flatdim

from achql.brax.envs.ant import Ant
from achql.brax.envs.base import GoalConditionedEnv
from achql.brax.tasks.base import TaskBase
from achql.brax.tasks.templates import inside_circle

from achql.stl import Expression, Var
import achql.stl.expression_jax2 as stl


class AntTaskBase(TaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=2)

        # In our ant environment, the the x and y position of the ant is
        # included in the observation, so the default dimension of an
        # observation is 29.
        # The y_velocity of the torso is the 16th feature of this
        # observation.
        self.obs_var = Var("obs", idx=0, dim=self.env.observation_size, position=(0, 2), y_velocity=16)

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
            "discounting": 0.95,
            "learning_rate": 3e-4,
            "entropy_cost": 1e-2,
            "num_envs": 4096,
            "batch_size": 2048,
        }


class AntRun(AntTaskBase):
    def __init__(self, backend="mjx"):
        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = Ant(
            exclude_current_positions_from_observation=True,
            backend=backend,
        )
        return env

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=2)
        self.obs_var = Var("obs", idx=0, dim=self.env.observation_size, y_velocity=14)

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        true = stl.STLPredicate(obs_var, lambda s: 999, lower_bound=0.0)
        return true


class AntPositiveY(AntTaskBase):
    """
    Always(y_velocity > 0)
    """

    def __init__(self, backend="mjx"):
        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = Ant(backend=backend)
        # for pos in self.goals:
        #     env.add_goal_mark(pos, color=(0, 1, 1, 0.5))
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        phi = stl.STLUntimedAlways(stl.STLPredicate(obs_var.y_velocity, lambda s: s[0], 0.0))
        return phi


class AntStraightSequence(AntTaskBase):
    """
    F(region1 ^ X(F(region2)))
    """

    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([2.0, 0.0])
        self.goal2_location = jnp.array([6.0, 0.0])
        self.goal_radius = 1.0

        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = Ant(backend=backend)
        # for pos in self.goals:
        #     env.add_goal_mark(pos, color=(0, 1, 1, 0.5))
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


class AntTurnSequence(AntTaskBase):
    """
    F(region1 ^ X(F(region2)))
    """

    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([2.0, 0.0])
        self.goal2_location = jnp.array([2.0, 4.0])
        self.goal_radius = 1.0

        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = Ant(
            terminate_when_unhealthy=True,
            backend=backend
        )
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

    @property
    def ppo_hps(self):
        return {
            "num_timesteps": 10_000_000, # 200_000_000
            "num_evals": 20,
            "reward_scaling": 5,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "unroll_length": 5,
            "num_minibatches": 32,
            "num_updates_per_batch": 4,
            "discounting": 0.95,
            "learning_rate": 3e-4,
            "entropy_cost": 1e-2,
            "num_envs": 4096,
            "batch_size": 2048,
        }
