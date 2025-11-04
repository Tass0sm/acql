import random
import functools

import numpy as np
import jax.numpy as jnp
import navix as nx
from navix import Environment, observations
from navix.spaces import Discrete
from navix.actions import rotate_ccw, rotate_cw, forward

from acql.navix.tasks.base import NavixTaskBase
from acql.navix.tasks.mixins import *
from acql.navix import observations as custom_observations
from acql.hierarchy.option import Option, FixedLengthTerminationPolicy

from acql.stl import Expression, Var
import acql.stl as stl


def make_tabular(env: Environment):
    unique_id_obs_fn = env.get_unique_id_obs_fn()
    n_unique_states = env.get_n_unique_states()
    unique_id_obs_space = Discrete.create(n_elements=n_unique_states, shape=(1,))

    return env.replace(
        observation_fn=unique_id_obs_fn,
        observation_space=unique_id_obs_space,
    )


class GridTaskBase(NavixTaskBase):
    def __init__(self, *args, **kwargs):
        self.obs_type = kwargs.get("obs_type", "unique_id")
        if "obs_type" in kwargs:
            del kwargs["obs_type"]
        super().__init__(*args, **kwargs)

    def _build_env(self, backend: str):
        env = nx.make(
            self.env_name,
            observation_fn=observations.none,
            action_set=(rotate_ccw, rotate_cw, forward),
            gamma=0.99,
        )
        env = make_tabular(env)
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


class GridWithObstacle(BoxObstacleMixin, GridTaskBase):
    def __init__(self, backend="mjx", **kwargs):
        self.obs_corners = (jnp.array([0.9, 1.9]), jnp.array([2.1, 3.1]))
        self.env_name = "Navix-Empty-5x5-v0"
        super().__init__(None, 1000, backend=backend, **kwargs)


class Grid8x8(NoConstraintMixin, GridTaskBase):
    def __init__(self, backend="mjx", **kwargs):
        self.env_name = "Navix-Empty-8x8-v0"
        super().__init__(None, 1000, backend=backend, **kwargs)


class Grid16x16(NoConstraintMixin, GridTaskBase):
    def __init__(self, backend="mjx", **kwargs):
        self.env_name = "Navix-Empty-16x16-v0"
        super().__init__(None, 1000, backend=backend, **kwargs)

    @property
    def tabular_hql_hps(self):
        return {
            "num_timesteps": 2_000_000,
            "reward_scaling": 1,
            "num_evals": 25,
            "episode_length": 100,
            "discounting": 0.99,
            "learning_rate": 1e-1,
            "num_envs": 256,
            "batch_size": 256,
            "max_devices_per_host": 1,
            "max_replay_size": 10000,
            "min_replay_size": 1000,
        }

class GridRandom5x5(NoConstraintMixin, GridTaskBase):
    def __init__(self, backend="mjx", **kwargs):
        self.env_name = "Navix-Empty-Random-5x5-v0"
        super().__init__(None, 1000, backend=backend, **kwargs)


class GridRandom8x8(NoConstraintMixin, GridTaskBase):
    def __init__(self, backend="mjx", **kwargs):
        self.env_name = "Navix-Empty-Random-8x8-v0"
        super().__init__(None, 1000, backend=backend, **kwargs)


class GridLargeWithObstacle(BoxObstacleMixin, GridTaskBase):
    def __init__(self, backend="mjx", **kwargs):
        self.obs_corners = (jnp.array([0.9, 1.9]), jnp.array([2.1, 3.1]))

        super().__init__(None, 1000, backend=backend, **kwargs)

    def _build_env(self, backend: str):
        if self.obs_type == "unique_id":
            obs_fn = observations.unique_id
        elif self.obs_type == "rgb":
            obs_fn = observations.rgb
        else:
            raise NotImplementedError()

        env = nx.make(
            "Navix-Empty-16x16-v0",
            observation_fn=obs_fn, # observations.symbolic_first_person,
            action_set=(rotate_ccw, rotate_cw, forward),
            gamma=0.99,
        )
        # env = FlattenObsWrapper(env)
        return env


class GridSingleSubgoal(SingleSubgoalMixin, GridTaskBase):
    def __init__(self, backend="mjx", **kwargs):
        self.goal1_location = jnp.array([12.0, 4.0])
        self.goal1_radius = 2.0

        super().__init__(None, 1000, backend=backend, **kwargs)
