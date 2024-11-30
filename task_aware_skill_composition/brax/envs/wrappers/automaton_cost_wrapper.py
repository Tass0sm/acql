"""Wrapper for adding specification satisfaction as reward."""

from typing import Callable, Dict, Optional, Tuple

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from flax import struct
import jax
from jax import numpy as jnp

from corallab_stl import Expression


class AutomatonCostWrapper(Wrapper):
    """Produces a separate cost signal for an automaton environment, making a CMDP"""

    def __init__(
            self,
            env: Env,
    ):
        super().__init__(env)

    def _get_good_edge_satisfaction(self, aut_state: int, state: State):
        """Returns the degree to which this state is close to causing a transition in the automaton."""

        num_bits = self.automaton.n_aps

        # [num_aps x num_possible_edges]
        binary_matrix = jnp.unpackbits(jnp.expand_dims(jnp.arange(2**num_bits, dtype=jnp.uint8), 0), axis=0)[-num_bits:]
        binary_matrix = jnp.flip(binary_matrix, 0)

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

    def _compute_cost(self, state: State, action: jax.Array, nstate: State):
        # aps = self.automaton.quantitative_eval_aps(state.obs)
        # return jax.nn.relu(aps[0])
        return 0.0

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        # good_edge_satisfaction = self._get_good_edge_satisfaction(state.info["automata_state"], state)
        cost = 0.0

        # putting cost into the info dict for now...
        state.info.update(
            cost=cost
            # good_edge_satisfaction=good_edge_satisfaction,
            # delta_good_edge_satisfaction=0.0
        )

        return state

    def step(self, state: State, action: jax.Array) -> State:
        # current_aut_state = state.info["automata_state"]
        # current_good_edge_satisfaction = state.info["good_edge_satisfaction"]

        nstate = self.env.step(state, action)

        # Calculate satisfaction of previous state's good edges for new state,
        # instead of nstate's good edges
        # new_good_edge_satisfaction = self._get_good_edge_satisfaction(current_aut_state, nstate)

        # delta_good_edge_satisfaction = new_good_edge_satisfaction - current_good_edge_satisfaction

        # new_reward = jnp.sign(delta_good_edge_satisfaction) * nstate.reward
        # new_reward = jnp.where(delta_good_edge_satisfaction > 0.0, nstate.reward + 0.1, -0.1)
        # new_reward = jnp.where(nstate.info["made_transition"] > 0.0, new_reward + 100.0, new_reward)

        # nstate = nstate.replace(
        #     reward=new_reward
        # )
        
        # next_aut_state = nstate.info["automata_state"]
        # next_state_good_edge_satisfaction = self._get_good_edge_satisfaction(next_aut_state, nstate)

        # nstate.info.update(
        #     good_edge_satisfaction=next_state_good_edge_satisfaction,
        #     delta_good_edge_satisfaction=delta_good_edge_satisfaction,
        # )

        cost = self._compute_cost(state, action, nstate)

        nstate.info.update(
            cost=cost
        )

        # nstate.info.update(
        #     good_edge_satisfaction=next_state_good_edge_satisfaction,
        #     delta_good_edge_satisfaction=delta_good_edge_satisfaction,
        # )

        return nstate
