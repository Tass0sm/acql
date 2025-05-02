import random

import numpy as np
import jax.numpy as jnp

from jaxgcrl.envs.wrappers import TrajectoryIdWrapper

from achql.brax.envs.panda import PandaReach, PandaPushEasy
from achql.brax.envs.base import GoalConditionedEnv
from achql.brax.tasks.base import BraxTaskBase
from achql.brax.tasks.mixins import *
from achql.brax.tasks.templates import sequence, inside_circle, outside_circle, inside_box, true_exp

from achql.stl import Expression, Var
import achql.stl as stl


class PandaTaskBase(BraxTaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=2)
        self.obs_var = Var("obs", idx=0, dim=self.env.state_dim, position=(7, 10))

    def get_options(self):
        raise NotImplementedError

    @property
    def sac_her_hps(self):
        return {
            "learning_rate": 1e-4,
            "discounting": 0.97,
            "batch_size": 256,
            "normalize_observations": True,
            "reward_scaling": 10.0,
            # target update rate
            "tau": 0.005,
            "min_replay_size": 1000,
            "max_replay_size": 10000,
            "deterministic_eval": False,
            "train_step_multiplier": 1,
            "unroll_length": 50,
            "h_dim": 256,
            "n_hidden": 2,
            # layer norm
            "use_ln": False,
            # hindsight experience replay
            "use_her": False,
            # --------------------
            # run params
            "total_env_steps": 1_000_000,
            "episode_length": 256,
            "num_envs": 256,
            "num_eval_envs": 256,
            "num_evals": 50,
            "action_repeat": 1,
            "max_devices_per_host": 1,
            # --------------------
            # # "learning_rate": 3e-4,
            # "discounting": 0.97,
            # "batch_size": 256,
            # "normalize_observations": True,
            # "reward_scaling": 10,
            # "num_timesteps": 2_000_000,
            # "num_evals": 50,
            # "episode_length": 256,
            # "action_repeat": 1,
            # "unroll_length": 62, # TODO: Reducing this increases time. What else does it affect?
            # "multiplier_num_sgd_steps": 1,
            # "max_devices_per_host": 1,
            # "max_replay_size": 10000,
            # # 8192, the default, causes the error "TypeError: broadcast_in_dim shape must have every element be nonnegative, got (-2, 50)."
            # "min_replay_size": 1000,
            # "use_her": True,
            # "num_envs": 512,
        }


class PandaReachTaskBase(PandaTaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = PandaReach(backend=backend)
        return env


class PandaReachTrue(PandaReachTaskBase):
    def __init__(self, backend="mjx"):
        super().__init__(None, 1000, backend=backend)

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        return true_exp(obs_var)


class PandaReachSingleSubgoal(SingleSubgoalMixin, PandaReachTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([0, 0.5, 0.3])
        self.goal1_radius = 0.1

        super().__init__(None, 1000, backend=backend)


class PandaPushEasyTaskBase(PandaTaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = PandaPushEasy(
            backend=backend
        )
        return env

class PandaPushEasyTrue(PandaPushEasyTaskBase):
    def __init__(self, backend="mjx"):
        super().__init__(None, 1000, backend=backend)

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        return true_exp(obs_var)


class PandaPushEasySingleSubgoal(SingleSubgoalMixin, PandaPushEasyTaskBase):
    def __init__(self, backend="mjx"):
        self.goal1_location = jnp.array([0.1, 0.6, 0.03])
        self.goal1_radius = 0.1

        super().__init__(None, 1000, backend=backend)
