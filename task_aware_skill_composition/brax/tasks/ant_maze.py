import random

import numpy as np
import jax.numpy as jnp

from gymnasium.spaces.utils import flatdim

from jaxgcrl.envs.ant_maze import AntMaze
from task_aware_skill_composition.brax.envs.base import GoalConditionedEnv
from task_aware_skill_composition.brax.tasks.base import TaskBase
from task_aware_skill_composition.hierarchy.ant.load import load_ant_options

from corallab_stl import Expression, Var
import corallab_stl.expression_jax2 as stl


class AntMazeTaskBase(TaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=2)
        self.obs_var = Var("obs", idx=0, dim=self.env.observation_size, position=(0, 2))

    def get_options(self):
        return self.get_learned_options()

    def get_learned_options(self):
        adapter = lambda x: x[..., 2:-2]
        return load_ant_options(termination_prob=1.0, adapter=adapter)

    def get_hard_coded_options(self):
        raise NotImplementedError()


class AntMazeNav(AntMazeTaskBase):
    def __init__(self, backend="mjx"):
        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = AntMaze(
            terminate_when_unhealthy=False,
            dense_reward=True,
            backend=backend
        )
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        true = stl.STLPredicate(obs_var, lambda s: 999, lower_bound=0.0)
        return true

    @property
    def ppo_hps(self):
        return {
            "num_timesteps": 50_000_000,
            "num_evals": 20,
            "reward_scaling": 5,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "unroll_length": 128,
            "num_minibatches": 32,
            "num_updates_per_batch": 4,
            "discounting": 0.95,
            "learning_rate": 3e-4,
            "entropy_cost": 1e-2,
            "num_envs": 2048,
            "batch_size": 256,
        }

    @property
    def hdqn_her_hps(self):
        return {
            "num_timesteps": 10_000_000,
            "reward_scaling": 1,
            "num_evals": 50,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "discounting": 0.99,
            # "learning_rate": 3e-4,
            "num_envs": 256,
            "batch_size": 256,
            "unroll_length": 62,
            "multiplier_num_sgd_steps": 1,
            "max_devices_per_host": 1,
            "max_replay_size": 10000,
            # 8192, the default, causes the error "TypeError: broadcast_in_dim shape must have every element be nonnegative, got (-2, 50)."
            "min_replay_size": 1000,
            "use_her": True,
        }

    @property
    def sac_her_hps(self):
        return {
            "num_timesteps": 10_000_000,
            "reward_scaling": 1,
            "num_evals": 50,
            "episode_length": 1000,
            "normalize_observations": True,
            "action_repeat": 1,
            "discounting": 0.99,
            # "learning_rate": 3e-4,
            "num_envs": 256,
            "batch_size": 256,
            # "unroll_length": 62,
            "multiplier_num_sgd_steps": 1,
            "max_devices_per_host": 1,
            "max_replay_size": 10000,
            # 8192, the default, causes the error "TypeError: broadcast_in_dim shape must have every element be nonnegative, got (-2, 50)."
            "min_replay_size": 1000,
            "use_her": True,
        }
