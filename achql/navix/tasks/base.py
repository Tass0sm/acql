import numpy as np

import jax.numpy as jnp

from navix.environments.environment import Environment

from achql.tasks.base import TaskBase


def FlattenObsWrapper(env: Environment):
    flatten_obs_fn = lambda x: jnp.ravel(env.observation_fn(x))
    flatten_obs_shape = (int(np.prod(env.observation_space.shape)),)
    return env.replace(
        observation_fn=flatten_obs_fn,
        observation_space=env.observation_space.replace(shape=flatten_obs_shape),
    )


class NavixTaskBase(TaskBase):

    @property
    def tabular_hql_hps(self):
        return {
            "num_timesteps": 500_000,
            "reward_scaling": 1,
            "num_evals": 25,
            "episode_length": 100,
            "normalize_observations": True,
            "discounting": 0.99,
            "learning_rate": 1e-1,
            "num_envs": 256,
            "batch_size": 256,
            "max_devices_per_host": 1,
            "max_replay_size": 10000,
            "min_replay_size": 1000,
        }

    @property
    def tabular_achql_hps(self):
        return {
            **self.tabular_hql_hps,
            "cost_scaling": 1.0,
            "safety_threshold": -0.1,
        }
