import random

import numpy as np
import jax.numpy as jnp

from task_aware_skill_composition.brax.envs.manipulation.ur5e_reach import UR5eReach
from task_aware_skill_composition.brax.envs.base import GoalConditionedEnv
from task_aware_skill_composition.brax.tasks.base import TaskBase
from task_aware_skill_composition.brax.tasks.templates import sequence, inside_circle, outside_circle, inside_box, true_exp
from task_aware_skill_composition.hierarchy.ur5e.load import load_hard_coded_ur5e_options

from corallab_stl import Expression, Var
import corallab_stl.expression_jax2 as stl


class UR5eTaskBase(TaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=2)
        self.obs_var = Var("obs", idx=0, dim=self.env.observation_size, position=(0, 2))

    def get_options(self):
        raise NotImplementedError


class UR5eReachTask(UR5eTaskBase):
    def __init__(self, backend="mjx"):
        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = UR5eReach(
            backend=backend
        )
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        return true_exp(obs_var)

    @property
    def sac_her_hps(self):
        return {
            "num_timesteps": 2_000_000,
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
