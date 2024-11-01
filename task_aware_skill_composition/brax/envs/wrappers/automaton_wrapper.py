"""Wrapper for adding specification automata to an environment."""

from typing import Callable, Dict, Optional, Tuple

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from flax import struct
import jax
from jax import numpy as jnp

import spot

from corallab_stl import Expression
from corallab_stl.automata import get_spot_formula_and_aps, eval_spot_formula


class JaxAutomaton:

    def __init__(self, exp, state_var):
        formula, aps = get_spot_formula_and_aps(exp)

        self.aps = aps
        self.n_aps = len(aps)
        self.automaton = formula.translate('Buchi', 'state-based', 'complete')
        self.n_states = self.automaton.num_states()
        self.n_edges = self.automaton.num_edges()

        delta_u = jnp.ones((2**self.n_aps, self.n_states), dtype=int)
        delta_u = delta_u * jnp.arange(self.n_states)

        bdd_dict = self.automaton.get_dict()
        for edge in self.automaton.edges():
            f = spot.bdd_to_formula(edge.cond, bdd_dict)
            # print(f"{edge.src} goes to {edge.dst} when {f}")
            for i in range(2**self.n_aps):
                if eval_spot_formula(f, i, self.n_aps):
                    delta_u = delta_u.at[i, edge.src].set(edge.dst)

        self.delta_u = delta_u.T

        self.state_var = state_var

    @property
    def init_state(self):
        return self.automaton.get_init_state_number()

    # Labelling function in the reward machine paper is the following
    # L :: S x A x S -> 2^P
    # We instead just use the current action and next state.
    # L :: A x S -> 2^P
    def eval_aps(self, action, nstate):
        bit_vector = jnp.zeros((4096,), dtype=int)

        for i, ap in self.aps.items():
            # If ap_i is true for an env, flip the i-th bit in its
            # bit-vector. vectorize over all envs.
            bit_vector = jnp.bitwise_or(bit_vector, (ap({ self.state_var.idx: nstate }) > 0.0).squeeze() * 2**i)

        return bit_vector

    def step(self, aut_state, ap_bit_vector):

        def step_i(state, bitvec):
            return self.delta_u[state, bitvec]

        return jax.vmap(step_i, in_axes=(0, 0))(aut_state, ap_bit_vector)



class AutomatonWrapper(Wrapper):
    """Adds specification robustness to envs's reward"""

    def __init__(self, env: Env, specification: Expression, state_var):
        super().__init__(env)

        self.automaton = JaxAutomaton(specification, state_var)

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)

        automata_state = jnp.ones((4096,), dtype=int) * self.automaton.init_state
        state.info["automata_state"] = automata_state

        return state

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)

        labels = self.automaton.eval_aps(action, nstate.obs)

        automata_state = state.info["automata_state"]

        nautomata_state = self.automaton.step(automata_state, labels)

        state.info.update(automata_state=nautomata_state)

        return nstate
