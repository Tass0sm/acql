"""Wrapper for adding specification satisfaction as reward."""

from typing import Callable, Dict, Optional, Tuple

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from flax import struct
import jax
from jax import numpy as jnp

from acql.stl import Expression


class SpecificationRewardWrapper(Wrapper):
    """Adds specification robustness to envs's reward"""

    def __init__(
            self,
            env: Env,
            specification: Expression,
            state_var,
            history_length: int = 1000,
            rho_weight: float = 1.0
    ):
        super().__init__(env)

        self.specification = specification
        self.state_var = state_var
        # self.action_var = action_var
        self.history_length = history_length
        self.rho_weight = rho_weight

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)

        state_history_buffer = jnp.zeros((self.history_length, self.observation_size))
        # action_history_buffer = jnp.zeros((self.history_length, self.action_size))

        state_history_buffer = state_history_buffer.at[0].set(state.obs)

        state.info["state_history_buffer"] = state_history_buffer
        # state.info["action_history_buffer"] = action_history_buffer

        return state

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)

        # compute rho based on current action, new state, and previous step's
        # carry
        # carry = nstate.info['spec_carry']

        state_history_buffer = state.info["state_history_buffer"]
        # action_history_buffer = state.info["action_history_buffer"]

        nstep_idx = nstate.info["steps"].astype(int)
        state_history_buffer = state_history_buffer.at[nstep_idx].set(nstate.obs)

        state.info.update(state_history_buffer=state_history_buffer)
        # state.info.update(action_history_buffer=action_history_buffer)

        tau = self.specification({ self.state_var.idx: state_history_buffer }, end_time=nstep_idx+1)
        rho = tau[0]

        reward_forward, reward_survive, reward_ctrl, reward_contact = [state.metrics.get(s) for s in [
            "reward_forward",
            "reward_survive",
            "reward_ctrl",
            "reward_contact",
        ]]

        new_reward = (self.rho_weight * rho) + \
            reward_forward + reward_survive + reward_ctrl + reward_contact

        # replace reward with rho
        nstate = nstate.replace(reward=new_reward)

        return nstate
