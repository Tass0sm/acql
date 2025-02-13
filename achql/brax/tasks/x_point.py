import random

import numpy as np
import jax.numpy as jnp

from achql.brax.envs.x_point import XPoint
from achql.brax.envs.base import GoalConditionedEnv
from achql.brax.tasks.base import BraxTaskBase
from achql.brax.tasks.templates import sequence, inside_circle, outside_circle, inside_box, true_exp
from achql.brax.tasks.mixins import *
from achql.hierarchy.x_point.load import load_hard_coded_x_point_options

from achql.stl import Expression, Var
import achql.stl.expression_jax2 as stl


class XPointTaskBase(BraxTaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = XPoint(backend=backend)
        return env

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=1)
        self.obs_var = Var("obs", idx=0, dim=self.env.observation_size, position=(0, 2))

    def get_options(self):
        return load_hard_coded_x_point_options(k=5)

    @property
    def hdcqn_hps(self):
        return {
            **self.hdqn_hps,
            "cost_scaling": 1.0,
            "safety_threshold": 0.0,
            "gamma_init_value": 0.8,
            "gamma_end_value": 0.98,
        }

    @property
    def hdcqn_her_hps(self):
        return {
            **self.hdqn_her_hps,
            "cost_scaling": 1.0,
            "safety_threshold": 0.0,
            "gamma_end_value": 0.93,
        }

class XPointFisacTask(XPointTaskBase):
    def __init__(self, backend="mjx"):
        self.obs1_corners = (jnp.array([2.0, 8.0]), jnp.array([6.0, 10.0]))
        self.obs2_corners = (jnp.array([10.0, 8.0]), jnp.array([14.0, 10.0]))

        super().__init__(None, 1000, backend=backend)

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        in_obs1 = inside_box(obs_var.position, *self.obs1_corners)
        in_obs2 = inside_box(obs_var.position, *self.obs2_corners)
        return stl.STLUntimedAlways(stl.STLNegation(stl.STLOr(in_obs1, in_obs2)))

    # @property
    # def rm_config(self) -> dict:
    #     return {
    #         "final_state": 0,
    #         "terminal_states": [2],
    #         "reward_functions": {
    #             (1, 1): lambda s_t, a_t, s_t1: 0.0,
    #             (1, 0): lambda s_t, a_t, s_t1: 1.0,
    #             (1, 2): lambda s_t, a_t, s_t1: 0.0,
    #             (0, 0): lambda s_t, a_t, s_t1: 1.0,
    #             (0, 2): lambda s_t, a_t, s_t1: 0.0,
    #             (2, 2): lambda s_t, a_t, s_t1: 0.0,
    #         },
    #         "pruned_edges": [(1, 0), (0, 0)]
    #     }

    # @property
    # def lof_task_state_costs(self) -> jnp.ndarray:
    #     return jnp.array([0.0, 1.0, 1.0])
