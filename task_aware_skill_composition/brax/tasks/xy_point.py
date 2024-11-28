import random

import numpy as np
import jax.numpy as jnp

from gymnasium.spaces.utils import flatdim

from task_aware_skill_composition.brax.envs.xy_point import XYPoint
from task_aware_skill_composition.brax.envs.base import GoalConditionedEnv
from task_aware_skill_composition.brax.tasks.base import TaskBase
from task_aware_skill_composition.brax.tasks.templates import sequence, inside_circle, outside_circle

from corallab_stl import Expression, Var
import corallab_stl.expression_jax2 as stl


class XYPointTaskBase(TaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=2)
        self.obs_var = Var("obs", idx=0, dim=self.env.observation_size)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = XYPoint(backend=backend) # , exclude_current_positions_from_observation=True)
        return env

    @property
    def hppo_hps(self):
        return {
            "num_timesteps": 50_000_000,
            "num_evals": 10,
            "reward_scaling": 10,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "unroll_length": 5,
            "num_minibatches": 32,
            "num_updates_per_batch": 4,
            "discounting": 0.97,
            "learning_rate": 3e-4,
            "entropy_cost": 1e-2,
            "num_envs": 4096,
            "batch_size": 2048,
        }

    @property
    def sac_hps(self):
        return {
            # "num_timesteps": 2_000_000,
            # "reward_scaling": 10,
            # "num_evals": 50,
            # "episode_length": 1000,
            # "normalize_observations": True,
            # "action_repeat": 1,
            # "unroll_length": 62, # TODO: Reducing this increases time. What else does it affect?
            # "multiplier_num_sgd_steps": 1,
            # "max_devices_per_host": 1,
            # "max_replay_size": 10000,
            # # 8192, the default, causes the error "TypeError: broadcast_in_dim shape must have every element be nonnegative, got (-2, 50)."
            # "min_replay_size": 1000,
            # "use_her": False,
            # "discounting": 0.97,
            # # "learning_rate": 3e-4,
            # "num_envs": 512,
            # "batch_size": 256,
        }

    @property
    def sac_lagrangian_hps(self):
        return self.sac_hps

    # @property
    # def sac_her_hps(self):
    #     return {
    #         "num_timesteps": 2_000_000,
    #         "reward_scaling": 10,
    #         "num_evals": 50,
    #         "episode_length": 1000,
    #         "normalize_observations": True,
    #         "action_repeat": 1,
    #         "unroll_length": 62, # TODO: Reducing this increases time. What else does it affect?
    #         "multiplier_num_sgd_steps": 1,
    #         "max_devices_per_host": 1,
    #         "max_replay_size": 10000,
    #         # 8192, the default, causes the error "TypeError: broadcast_in_dim shape must have every element be nonnegative, got (-2, 50)."
    #         "min_replay_size": 1000,
    #         "use_her": False,
    #         "discounting": 0.97,
    #         # "learning_rate": 3e-4,
    #         "num_envs": 512,
    #         "batch_size": 256,
    #     }

    @property
    def sac_her_lagrangian_hps(self):
        return self.sac_her_hps

    @property
    def hsac_hps(self):
        return {
            "num_timesteps": 10_000_000,
            "reward_scaling": 10,
            "num_evals": 50,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "unroll_length": 62, # TODO: Reducing this increases time. What else does it affect?
            "multiplier_num_sgd_steps": 1,
            "max_devices_per_host": 1,
            "max_replay_size": 10000,
            # 8192, the default, causes the error "TypeError: broadcast_in_dim shape must have every element be nonnegative, got (-2, 50)."
            "min_replay_size": 1000,
            "use_her": False,
            "discounting": 0.97,
            # "learning_rate": 3e-4,
            "num_envs": 512,
            "batch_size": 256,
        }

    @property
    def hdqn_hps(self):
        return {
            "num_timesteps": 1_000_000,
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
            "min_replay_size": 8192,
        }



class XYPointStraight(XYPointTaskBase):
    def __init__(self, backend="mjx"):
        super().__init__(None, 1000, backend=backend)

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        true = stl.STLPredicate(obs_var, lambda s: 999, lower_bound=0.0)
        return true


class XYPointDiagonal1(XYPointTaskBase):
    def __init__(self, backend="mjx"):
        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = XYPoint(backend=backend,
                      target_velocity_vec=jnp.array([0.0, 0.0, 0.0]),
                      target_disp_vec=jnp.array([1.0, 1.0, 0.0]))
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        true = stl.STLPredicate(obs_var, lambda s: 999, lower_bound=0.0)
        return true


class XYPointDiagonal2(XYPointTaskBase):
    def __init__(self, backend="mjx"):
        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = XYPoint(backend=backend,
                      target_velocity_vec=jnp.array([0.0, 0.0, 0.0]),
                      target_disp_vec=jnp.array([1.0, -1.0, 0.0]))
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        true = stl.STLPredicate(obs_var, lambda s: 999, lower_bound=0.0)
        return true


class XYPointStraightSequence(XYPointTaskBase):
    """
    F(region1 ^ X(F(region2)))
    """

    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([2.0, 0.0])
        self.goal2_location = jnp.array([6.0, 0.0])
        self.goal_radius = 1.0

        super().__init__(None, 1000, backend=backend)

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        at_goal1 = inside_circle(obs_var.position, self.goal1_location, self.goal_radius)
        at_goal2 = inside_circle(obs_var.position, self.goal2_location, self.goal_radius)

        phi = stl.STLUntimedEventually(
            stl.STLAnd(at_goal1, stl.STLNext(stl.STLUntimedEventually(at_goal2)))
        )

        return phi


class XYPointTurnSequence(XYPointTaskBase):
    """
    F(region1 ^ X(F(region2)))
    """

    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([2.0, 0.0])
        self.goal2_location = jnp.array([-2.0, 4.0])
        self.goal_radius = 1.0

        super().__init__(None, 1000, backend=backend)

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=2)
        self.obs_var = Var("obs", idx=0, dim=self.env.observation_size, position=(0, 2))

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = XYPoint(backend=backend, exclude_current_positions_from_observation=False)
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


class XYPointObstacle(XYPointTaskBase):
    """
    G(!region1)
    """

    def __init__(self, backend="mjx"):
        self.obs1_location = jnp.array([2.0, 0.0])
        self.obs_radius = 1.0

        super().__init__(None, 1000, backend=backend)

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=2)
        self.obs_var = Var("obs", idx=0, dim=self.env.observation_size, position=(0, 2))

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = XYPoint(backend=backend, exclude_current_positions_from_observation=False)
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        at_obs1 = inside_circle(obs_var.position, self.obs1_location, self.obs_radius)
        phi = stl.STLUntimedAlways(stl.STLNegation(at_obs1))
        return phi

class XYPointObstacle2(XYPointTaskBase):
    """
    G(!region1)
    """

    def __init__(self, backend="mjx"):
        self.obs1_location = jnp.array([2.0, 2.0])
        self.obs_radius = 1.0

        super().__init__(None, 1000, backend=backend)

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=2)
        self.obs_var = Var("obs", idx=0, dim=self.env.observation_size, position=(0, 2))

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = XYPoint(backend=backend, exclude_current_positions_from_observation=False)
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        at_obs1 = inside_circle(obs_var.position, self.obs1_location, self.obs_radius)
        phi = stl.STLUntimedAlways(stl.STLNegation(at_obs1))
        return phi
    
