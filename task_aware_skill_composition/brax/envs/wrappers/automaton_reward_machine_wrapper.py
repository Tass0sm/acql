"""Wrapper for turning automaton MDP into reward machine MDP."""

from typing import Callable, Dict, Optional, Tuple

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from flax import struct
import jax
from jax import numpy as jnp

from corallab_stl import Expression


class AutomatonRewardMachineWrapper(Wrapper):
    """Wrapper for an augmented reward signal for an automaton environment"""

    def __init__(
            self,
            env: Env,
            rm_config,
    ):
        super().__init__(env)

        self.rm_config = rm_config

        dummy_func = lambda s_t, a_t, s_t1: 0.0

        self.u_fs = []
        for u in range(self.automaton.n_states):
            v_fs = []
            for v in range(self.automaton.n_states):
                f_uv = self.rm_config.get((u, v), dummy_func)
                v_fs.append(f_uv)

            f_u = lambda v, s_t, a_t, s_t1: jax.lax.switch(v, v_fs, s_t, a_t, s_t1)
            self.u_fs.append(f_u)
        
    def compute_reward(self, u, v, state, action, nstate):
        return jax.lax.switch(u, self.u_fs, v, state, action, nstate)

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        u = state.info["automata_state"]

        nstate = self.env.step(state, action)
        v = nstate.info["automata_state"]
        
        new_reward = self.compute_reward(u, v, state, action, nstate)
        nstate = nstate.replace(reward=new_reward)

        return nstate
