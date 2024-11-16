"""Wrapper for adding specification automata to an environment."""

from typing import Callable, Dict, Optional, Tuple

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from flax import struct, nnx
import jax
from jax import numpy as jnp


class OptionsWrapper(Wrapper):
    """Adds to environment information necessary for options framework"""

    def __init__(
            self,
            env: Env,
            num_options: int,
    ):
        super().__init__(env)

        self.num_options = num_options

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)

        state.info["option"] = jnp.uint32(0)
        state.info["option_beta"] = jnp.uint32(1)

        return state

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)

        # labels = self.automaton.eval_aps(action, nstate.obs)

        # automata_state = state.info["automata_state"]

        # nautomata_state = self.automaton.step(automata_state, labels)

        # made_transition = jnp.where(automata_state != nautomata_state, jnp.uint32(1), jnp.uint32(0))

        # state.info.update(
        #     automata_state=nautomata_state,
        #     made_transition=made_transition
        # )

        # if self.augment_obs:
        #     new_obs = jnp.concatenate((nstate.obs, self.automaton.embed(nautomata_state)), axis=-1)
        #     # new_obs = jnp.concatenate((nstate.obs, self.automaton.one_hot_encode(nautomata_state)), axis=-1)
        #     nstate = nstate.replace(obs=new_obs)

        return nstate
