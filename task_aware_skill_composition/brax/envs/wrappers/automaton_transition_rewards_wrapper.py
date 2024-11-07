"""Wrapper for adding specification satisfaction as reward."""

from typing import Callable, Dict, Optional, Tuple

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from flax import struct
import jax
from jax import numpy as jnp

from corallab_stl import Expression


class AutomatonTransitionRewardsWrapper(Wrapper):
    """Adds specification robustness to envs's reward"""

    def __init__(
            self,
            env: Env,
    ):
        super().__init__(env)

    def _get_good_edge_satisfaction(self, state: State):
        """Returns the degree to which this state is close to causing a transition in the automaton."""

        num_bits = self.automaton.n_aps

        aut_state = state.info["automata_state"]

        # [num_aps x num_possible_edges]
        binary_matrix = jnp.unpackbits(jnp.expand_dims(jnp.arange(2**num_bits, dtype=jnp.uint8), 0), axis=0)[-num_bits:]

        # [num_aps]
        aut_vector = jnp.expand_dims(self.automaton.quantitative_eval_aps(state.obs), 1)

        # [num_aps x num_possible_edges]
        ap_satisfaction_per_edge = binary_matrix * aut_vector
        edge_satisfaction = jnp.min(jnp.where(binary_matrix == 1, ap_satisfaction_per_edge, 999), axis=0)

        # For now, just consider a good edge as one that is not a self-loop

        # [num_possible_edges]
        good_edges = jnp.where(self.automaton.delta_u[aut_state] != aut_state, 1, 0)

        # single number
        good_edge_satisfaction = jnp.max(jnp.where(good_edges > 0, edge_satisfaction, -999), axis=0)

        return good_edge_satisfaction

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        good_edge_satisfaction = self._get_good_edge_satisfaction(state)

        state.info.update(
            good_edge_satisfaction=good_edge_satisfaction,
            delta_good_edge_satisfaction=0.0
        )

        return state

    def step(self, state: State, action: jax.Array) -> State:
        good_edge_satisfaction = state.info["good_edge_satisfaction"]
        nstate = self.env.step(state, action)
        good_edge_satisfaction_prime = self._get_good_edge_satisfaction(nstate)

        delta_good_edge_satisfaction = good_edge_satisfaction_prime - good_edge_satisfaction

        # from IPython.core.debugger import set_trace; set_trace()

        new_reward = jnp.sign(delta_good_edge_satisfaction) * nstate.reward

        nstate = nstate.replace(
            reward=new_reward
        )

        nstate.info.update(
            good_edge_satisfaction=good_edge_satisfaction_prime,
            delta_good_edge_satisfaction=delta_good_edge_satisfaction,
        )

        return nstate
