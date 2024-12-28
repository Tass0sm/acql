import random

import numpy as np
import jax.numpy as jnp

from task_aware_skill_composition.brax.envs.ur5e_envs.ur5e_reach import UR5eReach
from task_aware_skill_composition.brax.envs.base import GoalConditionedEnv
from task_aware_skill_composition.brax.tasks.base import TaskBase
from task_aware_skill_composition.brax.tasks.templates import sequence, inside_circle, outside_circle, inside_box
from task_aware_skill_composition.hierarchy.ur5e.load import load_hard_coded_ur5e_options

from corallab_stl import Expression, Var
import corallab_stl.expression_jax2 as stl


class UR5eReachTaskBase(TaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=2)
        self.obs_var = Var("obs", idx=0, dim=self.env.observation_size, position=(0, 2))

    def get_options(self):
        return self.get_hard_coded_options()

    def get_hard_coded_options(self):
        return load_hard_coded_ur5e_options()


class UR5eReachNav(UR5eReachTaskBase):
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
        true = stl.STLPredicate(obs_var, lambda s: 999, lower_bound=0.0)
        return true
