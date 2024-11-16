"""Wrapper for adding specification automata to an environment."""

from typing import Callable, Dict, Optional, Tuple

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from flax import struct, nnx
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

        delta_u = jnp.ones((2**self.n_aps, self.n_states), dtype=jnp.uint32)
        delta_u = delta_u * jnp.arange(self.n_states, dtype=jnp.uint32)

        bdd_dict = self.automaton.get_dict()
        for edge in self.automaton.edges():
            f = spot.bdd_to_formula(edge.cond, bdd_dict)
            # print(f"{edge.src} goes to {edge.dst} when {f}")
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
    def eval_aps(self, action, nstate):
        bit_vector = 0

        for i, ap in self.aps.items():
            # If ap_i is true for an env, flip the i-th bit in its
            # bit-vector. vectorize over all envs.
            bit_vector = jnp.bitwise_or(bit_vector, (ap({ self.state_var.idx: jnp.expand_dims(nstate, 0) }) > 0.0).squeeze() * 2**i)

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

    def one_hot_encode(self, state: jax.Array) -> jax.Array:
        return jax.nn.one_hot(state, self.num_states)

    def one_hot_decode(self, state: jax.Array) -> jax.Array:
        return jnp.argmax(state, axis=1)

    def embed(self, state: jax.Array) -> jax.Array:
        return self.embedding(state)


class AutomatonWrapper(Wrapper):
    """Tracks automaton state during interaction with environment"""

    def __init__(
            self,
            env: Env,
            specification: Expression,
            state_var,
            augment_obs: bool = True
    ):
        super().__init__(env)

        self.automaton = JaxAutomaton(specification, state_var)
        self.augment_obs = augment_obs

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)

        automata_state = jnp.uint32(self.automaton.init_state)
        state.info["automata_state"] = automata_state
        state.info["made_transition"] = jnp.uint32(0)

        if self.augment_obs:
            new_obs = jnp.concatenate((state.obs, self.automaton.embed(automata_state)), axis=-1)
            # new_obs = jnp.concatenate((state.obs, self.automaton.one_hot_encode(automata_state)), axis=-1)
            state = state.replace(obs=new_obs)

        return state

    def step(self, state: State, action: jax.Array) -> State:

        if self.augment_obs:
            state = state.replace(obs=state.obs[..., :-self.automaton.num_states])

        nstate = self.env.step(state, action)

        labels = self.automaton.eval_aps(action, nstate.obs)

        automata_state = state.info["automata_state"]

        nautomata_state = self.automaton.step(automata_state, labels)

        made_transition = jnp.where(automata_state != nautomata_state, jnp.uint32(1), jnp.uint32(0))

        state.info.update(
            automata_state=nautomata_state,
            made_transition=made_transition
        )

        if self.augment_obs:
            new_obs = jnp.concatenate((nstate.obs, self.automaton.embed(nautomata_state)), axis=-1)
            # new_obs = jnp.concatenate((nstate.obs, self.automaton.one_hot_encode(nautomata_state)), axis=-1)
            nstate = nstate.replace(obs=new_obs)

        return nstate
