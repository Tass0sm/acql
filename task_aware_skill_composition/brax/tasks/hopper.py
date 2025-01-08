import random

import numpy as np
import jax.numpy as jnp

from gymnasium.spaces.utils import flatdim

from task_aware_skill_composition.brax.safety_gymnasium_envs.safety_hopper_velocity_v1 import SafetyHopperVelocityEnv as Hopper
from task_aware_skill_composition.brax.tasks.base import TaskBase
from task_aware_skill_composition.brax.tasks.templates import inside_circle, outside_circle

from corallab_stl import Expression, Var
import corallab_stl.expression_jax2 as stl


class HopperTaskBase(TaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=2)
        self.obs_var = Var("obs", idx=0, dim=self.env.observation_size)

    @property
    def sac_hps(self):
        return {
            "num_timesteps": 2_553_600,
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

    @property
    def ddpg_hps(self):
        return {
            "num_timesteps": 1_500_000,
            "num_evals": 20,
            "reward_scaling": 30,
            "cost_scaling": 30,
            "cost_budget": 30,
            "lambda_update_interval": 24,
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

class HopperStraight(HopperTaskBase):
    def __init__(self, backend="mjx"):
        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str):
        env = Hopper(backend=backend)
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        true = stl.STLPredicate(obs_var, lambda s: 999, lower_bound=0.0)
        return true


class HopperLowVelocity(HopperTaskBase):
    def __init__(self, backend="mjx"):
        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str):
        env = Hopper(backend=backend)
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        true = stl.STLPredicate(obs_var, lambda s: 999, lower_bound=0.0)
        return true
