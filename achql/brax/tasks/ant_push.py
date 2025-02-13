import random

import numpy as np
import jax.numpy as jnp

from gymnasium.spaces.utils import flatdim

from jaxgcrl.envs.ant_push import AntPush
from achql.brax.envs.base import GoalConditionedEnv
from achql.brax.tasks.base import BraxTaskBase

from achql.stl import Expression, Var
import achql.stl.expression_jax2 as stl


class AntPushTaskBase(BraxTaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        assert backend == "mjx", "Ant Push environment is only stable with mjx backend"
        env = AntPush(backend=backend)
        return env

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=2)
        self.obs_var = Var("obs", idx=0, dim=self.env.observation_size, position=(0, 2))


class AntPushDefault(AntPushTaskBase):
    def __init__(self, backend="mjx"):
        super().__init__(None, 1000, backend=backend)

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        true = stl.STLPredicate(obs_var, lambda s: 999, lower_bound=0.0)
        return true
