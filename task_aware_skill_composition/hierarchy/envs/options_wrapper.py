"""Wrapper for creating Semi-MDP with set of options and existing environment."""

from typing import Sequence, Callable, Dict, Optional, Tuple

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from brax.training.types import PRNGKey
from flax import struct, nnx
import jax
from jax import numpy as jnp

from task_aware_skill_composition.hierarchy.option import Option


class OptionsWrapper(Wrapper):
    """Adds to environment information necessary for options framework"""

    def __init__(
            self,
            env: Env,
            options: Sequence[Option],
            discounting: float,
    ):
        super().__init__(env)

        self.options = options
        self.num_options = len(options)
        self.discounting = discounting

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info['low_steps'] = jnp.zeros(rng.shape[:-1])
        return state

    # def step(self, state: State, option: jax.Array, key: PRNGKey) -> State:
    def step(self, state: State, option_and_key: Tuple[jax.Array, PRNGKey]) -> State:
        """Turning the MDP into a Semi-MDP requries passing a PRNG key to the
        step function. To stay compatible with the existing brax training
        wrappers, this key is passed in a tuple with the option (which was
        originally the action).

        """

        o_t, key = option_and_key

        def while_cond(x):
            s_t, _, _, key = x
            beta_t = jax.lax.switch(o_t, [o.termination for o in self.options], s_t, key)
            return beta_t != 1

        def while_body(x):
            s_t, iters, r_t, key = x
            key, inf_key = jax.random.split(key)
            a_t, _ = jax.lax.switch(o_t, [o.inference for o in self.options], s_t, inf_key)
            s_t1 = self.env.step(s_t, a_t)
            r_t1 = r_t + jnp.pow(self.discounting, iters) * s_t1.reward
            return (s_t1, iters+1, r_t1, key)

        # run body at least once
        nstate, steps, reward, nkey = while_body((state, 0, 0, key))

        # then continue running until terminated
        fstate, k, freward, _ = jax.lax.while_loop(while_cond, while_body, (nstate, steps, reward, nkey))

        # update reward of final state with the option history reward
        fstate = fstate.replace(reward=freward)

        # update state info
        fstate.info['low_steps'] = state.info['low_steps'] + k

        return fstate
