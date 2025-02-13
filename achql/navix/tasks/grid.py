import random
import functools

import numpy as np
import jax.numpy as jnp
import navix as nx
from navix import observations
from navix.actions import rotate_ccw, rotate_cw, forward

from achql.navix.tasks.base import NavixTaskBase, FlattenObsWrapper
# from achql.brax.tasks.templates import sequence, inside_circle, outside_circle, inside_box, true_exp
from achql.navix.tasks.mixins import *
from achql.navix import observations as custom_observations
from achql.hierarchy.option import Option, FixedLengthTerminationPolicy

from achql.stl import Expression, Var
import achql.stl.expression_jax2 as stl


class GridTaskBase(NavixTaskBase):
    def __init__(self, *args, **kwargs):
        self.obs_type = kwargs.get("obs_type", "unique_id")
        if "obs_type" in kwargs:
            del kwargs["obs_type"]
        super().__init__(*args, **kwargs)

    def _build_env(self, backend: str):
        if self.obs_type == "unique_id":
            obs_fn = observations.unique_id
        elif self.obs_type == "rgb":
            obs_fn = observations.rgb
        else:
            raise NotImplementedError()

        env = nx.make(
            "Navix-Empty-5x5-v0",
            observation_fn=obs_fn, # observations.symbolic_first_person,
            action_set=(rotate_ccw, rotate_cw, forward),
            gamma=0.99,
        )
        # env = FlattenObsWrapper(env)
        return env

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=2)
        self.obs_var = Var("obs", idx=0, dim=self.env.observation_space.shape[0])

    def get_options(self):
        options_l = []

        def hard_policy(
                ctrl,
                observations,
                key_sample
        ):
            return ctrl, {
                'log_prob': 0.0,
                'raw_action': ctrl
            }

        for i in range(self.env.action_space.n):
            options_l.append(
                Option(
                    f"option_{i}", None, None, functools.partial(hard_policy, jnp.array(i, dtype=jnp.int32)),
                    termination_policy=FixedLengthTerminationPolicy(1)
                )
            )

        return options_l


class GridSingleSubgoal(SingleSubgoalMixin, GridTaskBase):
    def __init__(self, backend="mjx", **kwargs):
        self.goal1_location = jnp.array([12.0, 4.0])
        self.goal1_radius = 2.0

        super().__init__(None, 1000, backend=backend, **kwargs)
