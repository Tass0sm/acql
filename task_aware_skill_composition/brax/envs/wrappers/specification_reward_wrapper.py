"""Wrapper for adding specification satisfaction as reward."""

from typing import Callable, Dict, Optional, Tuple

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from flax import struct
import jax
from jax import numpy as jnp

from corallab_stl import Expression


class SpecificationRewardWrapper(Wrapper):
    """Adds specification robustness to envs's reward"""

    def __init__(self, env: Env, specification: Expression, state_var):
        super().__init__(env)

        self.specification = specification
        self.state_var = state_var
        # self.action_var = action_var

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)

        state_history_buffer = jnp.zeros((4096, self.episode_length, self.observation_size))
        # action_history_buffer = jnp.zeros((4096, self.episode_length, self.action_size))

        state_history_buffer = state_history_buffer.at[:, 0].set(state.obs)

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

        def set_state_at_i(buffer, i, state):
            return buffer.at[i].set(state)

        p_set_state_at_i = jax.vmap(set_state_at_i, in_axes=(0, 0, 0))
        state_history_buffer = p_set_state_at_i(state_history_buffer, nstep_idx, nstate.obs)

        state.info.update(state_history_buffer=state_history_buffer)
        # state.info.update(action_history_buffer=action_history_buffer)

        taus = self.specification({ self.state_var.idx: state_history_buffer }, end_time=nstep_idx+1)
        rhos = taus[:, 0]

        reward_forward, reward_survive, reward_ctrl, reward_contact = [state.metrics.get(s) for s in [
            "reward_forward",
            "reward_survive",
            "reward_ctrl",
            "reward_contact",
        ]]

        new_reward = rhos + reward_survive + reward_ctrl + reward_contact

        # replace reward with rho
        nstate = nstate.replace(reward=new_reward)

        return nstate
