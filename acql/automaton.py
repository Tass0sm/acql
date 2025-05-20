"""Wrapper for adding specification automata to an environment."""

import warnings
from functools import reduce
from typing import Callable, Dict, Optional, Tuple
from operator import itemgetter

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from flax import struct, nnx
import jax
from jax import numpy as jnp

import spot
import buddy

from achql.stl import get_spot_formula_and_aps, eval_spot_formula, make_just_liveness_automaton, get_outgoing_conditions


class JaxAutomaton:

    def __init__(self, exp, state_var):
        formula, aps, spot_aps = get_spot_formula_and_aps(exp)

        self.aps = aps
        self.spot_aps = spot_aps
        self.n_aps = len(aps)
        self.formula = formula
        self.automaton = formula.translate('Buchi', 'state-based', 'complete')
        self.n_states = self.automaton.num_states()
        self.n_edges = self.automaton.num_edges()

        delta_u = jnp.ones((2**self.n_aps, self.n_states), dtype=jnp.uint32)
        delta_u = delta_u * jnp.arange(self.n_states, dtype=jnp.uint32)

        self.bdd_dict = self.automaton.get_dict()
        for edge in self.automaton.edges():
            f = spot.bdd_to_formula(edge.cond, self.bdd_dict)
            for i in range(2**self.n_aps):
                if eval_spot_formula(f, i, self.n_aps):
                    delta_u = delta_u.at[i, edge.src].set(edge.dst)

        self.delta_u = delta_u.T

        self.state_var = state_var

        self.embedding = nnx.Embed(
            num_embeddings=self.n_states,
            features=self.n_states,
            rngs=nnx.Rngs(0),
        )

    @property
    def init_state(self):
        return self.automaton.get_init_state_number()

    @property
    def num_states(self):
        return self.automaton.num_states()

    # Labelling function in the reward machine paper is the following
    # L :: S x A x S -> 2^P
    # We instead just use the current action and next state.
    # L :: A x S -> 2^P
    def eval_aps(self, action, nstate, ap_params):
        bit_vector = 0

        for i, ap in self.aps.items():
            # If ap_i is true for an env, flip the i-th bit in its
            # bit-vector. vectorize over all envs.
            bit_vector = jnp.bitwise_or(bit_vector, (ap({
                self.state_var.idx: jax.tree.map(lambda x: jnp.expand_dims(x, 0), nstate),
                # 32 + i is a special input which predicate i will look up for params
                32 + i: ap_params[i],
            }) > 0.0).squeeze() * 2**i)

        return bit_vector

    def quantitative_eval_aps(self, state):
        aps_vector = jnp.zeros((self.n_aps,))

        for i, ap in self.aps.items():
            # If ap_i is true for an env, flip the i-th bit in its
            # bit-vector. vectorize over all envs.
            aps_vector = aps_vector.at[i].set(ap({ self.state_var.idx: jnp.expand_dims(state, 0) }).squeeze())

        return aps_vector

    def step(self, aut_state, ap_bit_vector):
        return self.delta_u[aut_state, ap_bit_vector]

    def one_hot_encode(self, obs: jax.Array) -> jax.Array:
        if self.num_states <= 1:
            return jnp.array([])
        else:
            return jax.nn.one_hot(obs, self.num_states)

    def one_hot_decode(self, obs: jax.Array) -> jax.Array:
        if self.num_states <= 1:
            return jnp.zeros((obs.shape[0],), dtype=jnp.int32)
        else:
            return jnp.argmax(obs, axis=-1)

    def embed(self, state: jax.Array) -> jax.Array:
        return self.embedding(state)
