import random

import numpy as np
import jax.numpy as jnp

from achql.brax.envs.manipulation.arm_eef_binpick_easy import ArmEEFBinpickEasy
from achql.brax.envs.base import GoalConditionedEnv
from achql.brax.tasks.base import BraxTaskBase
from achql.brax.tasks.templates import sequence, inside_circle, outside_circle, inside_box, true_exp
from achql.brax.tasks.mixins import *
from achql.hierarchy.ur5e.load import load_ur5e_options, load_ur5e_eef_options
from achql.hierarchy.option import FixedLengthTerminationPolicy

from achql.stl import Expression, Var
import achql.stl as stl


class ArmEEFTaskBase(BraxTaskBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_vars(self):
        self.wp_var = Var("wp", idx=0, dim=3)
        self.obs_var = Var("obs", idx=0, dim=self.env.observation_size, position=(0, 3))

    def get_options(self):
        return self.get_learned_options()

    def get_learned_options(self):
        # Tentative: option env size may change
        adapter = lambda x: x[..., :12]
        return load_ur5e_options(termination_policy=FixedLengthTerminationPolicy(1),
                                 adapter=adapter)

    def get_hard_coded_options(self):
        raise NotImplementedError()


class ArmEEFBinpickEasyTask(ArmEEFTaskBase):
    def __init__(self, backend="mjx"):
        super().__init__(None, 1000, backend=backend)

    def _build_env(self, backend: str) -> GoalConditionedEnv:
        env = ArmEEFBinpickEasy(backend=backend)
        return env

    def _build_hi_spec(self, wp_var: Var) -> Expression:
        pass

    def _build_lo_spec(self, obs_var: Var) -> Expression:
        return true_exp(obs_var)

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
            "use_ln": True,
            # hindsight experience replay
            "use_her": True,
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

    @property
    def crl_hps(self):
        return {
            "policy_lr": 3e-4,
            "critic_lr": 3e-4,
            "alpha_lr": 3e-4,
            "batch_size": 256,
            # gamma
            "discounting": 0.99,
            # forward CRL logsumexp penalty
            "logsumexp_penalty_coeff": 0.1,
            "train_step_multiplier": 1,
            "disable_entropy_actor": False,
            "max_replay_size": 10000,
            "min_replay_size": 1000,
            "unroll_length": 62,
            "h_dim": 256,
            "n_hidden": 2,
            "skip_connections": 4,
            "use_relu": False,
            # phi(s,a) and psi(g) repr dimension
            "repr_dim": 64,
            # layer norm
            "use_ln": False,
            "contrastive_loss_fn": "fwd_infonce",
            "energy_fn": "norm",
            # --------------------
            # run params
            "total_env_steps": 10_000_000,
            "episode_length": 256,
            "num_envs": 256,
            "num_eval_envs": 256,
            "num_evals": 50,
            "action_repeat": 1,
            "max_devices_per_host": 1,
            # --------------------
        }
