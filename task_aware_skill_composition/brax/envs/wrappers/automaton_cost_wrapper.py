"""Wrapper for adding automaton cost function."""

import copy
from typing import Callable, Dict, Optional, Tuple

import spot

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from flax import struct
import jax
from jax import numpy as jnp

import corallab_stl.expression_jax2 as stl
from corallab_stl import Expression
from corallab_stl.automata import get_safety_conditions
from corallab_stl.utils import fold_tree, fold_spot_formula


def get_compiled_conditions(
        safety_conditions,
        state_var,
        aps,
):
    stls = []

    def to_stl_helper(root, children):
        nonlocal aps

        children = list(children)
        k = root.kind()

        if k == spot.op_ff:
            return stl.STLPredicate(state_var, lambda s: -1, lower_bound=0.0)
        elif k == spot.op_tt:
            return stl.STLPredicate(state_var, lambda s: 1, lower_bound=0.0)
            # ap_pred = copy.deepcopy(aps[0])
            # ap_pred.lower_bound = -2.0
            # return ap_pred
            # return aps[0]
        elif k == spot.op_ap:
            ap_id = int(root.ap_name()[3:])
            # return aps[ap_id]
            ap_pred = copy.deepcopy(aps[ap_id])
            ap_pred.lower_bound = -1.0
            return ap_pred
        elif k == spot.op_Not:
            return stl.STLNegation(children[0])
        elif k == spot.op_And:
            return stl.STLAnd(children[0], children[1])
        elif k == spot.op_Or:
            return stl.STLOr(children[0], children[1])
        else:
            raise NotImplementedError(f"Formula {root} with kind = {k}")

    for k, form_k in sorted(safety_conditions.items(), key=lambda x: x[0]):
        stl_k = fold_spot_formula(to_stl_helper, form_k)
        not_stl_k = stl.STLNegation(stl_k)
        stls.append(not_stl_k)

    return stls


class AutomatonCostWrapper(Wrapper):
    """Produces a separate cost signal for an automaton environment, making a CMDP"""

    def __init__(
            self,
            env: Env,
    ):
        super().__init__(env)

        self.state_var = self.automaton.state_var

        self.safety_conditions = get_safety_conditions(self.automaton.automaton)
        self.safety_exps = get_compiled_conditions(self.safety_conditions,
                                                   self.automaton.state_var,
                                                   self.automaton.aps)

    def cost_obs(self, obs):
        return jnp.concatenate((self.goalless_obs(obs),
                                self.automaton_obs(obs)), axis=-1)

    def _compute_cost(self, state: State, action: jax.Array, nstate: State):
        """
        Note: state and action are unused.
        """

        nobs = jnp.expand_dims(self.original_obs(nstate.obs), 0)
        safety_rho = jax.lax.switch(nstate.info["automata_state"],
                                    self.safety_exps,
                                    { self.state_var.idx: nobs }).squeeze()
        # return jax.nn.relu(safety_rho)
        # return jnp.tanh(safety_rho)
        return safety_rho

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)

        cost = 0.0

        # putting cost into the info dict for now...
        state.info.update(
            cost=cost
        )

        return state

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)

        cost = self._compute_cost(state, action, nstate)

        nstate.info.update(
            cost=cost
        )

        return nstate

    @property
    def cost_observation_size(self):
        # return self.env.goalless_observation_size
        return self.env.goalless_observation_size + self.env.automaton.n_states
