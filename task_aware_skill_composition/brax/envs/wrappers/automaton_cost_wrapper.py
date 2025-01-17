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
        margin=1.0,
        using_relu_cost=False,
):
    stls = []

    def to_stl_helper(root, children):
        nonlocal aps

        children = list(children)
        k = root.kind()

        if k == spot.op_ff:
            return stl.STLPredicate(state_var, lambda s, _: -1, lower_bound=0.0)
        elif k == spot.op_tt:
            return stl.STLPredicate(state_var, lambda s, _: 1, lower_bound=0.0)
            # ap_pred = copy.deepcopy(aps[0])
            # ap_pred.lower_bound = -2.0
            # return ap_pred
            # return aps[0]
        elif k == spot.op_ap:
            ap_id = int(root.ap_name()[3:])
            # return aps[ap_id]
            ap_pred = copy.deepcopy(aps[ap_id])
            ap_pred.lower_bound = -margin
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

        if using_relu_cost:
            # For the ablation
            stls.append(stl_k)
        else:
            not_stl_k = stl.STLNegation(stl_k)
            stls.append(not_stl_k)

    return stls


class AutomatonCostWrapper(Wrapper):
    """Produces a separate cost signal for an automaton environment, making a CMDP"""

    def __init__(
            self,
            env: Env,
            margin: float = 1.0,
            relu_cost: bool = False,
    ):
        super().__init__(env)

        self.state_var = self.automaton.state_var

        self.safety_conditions = get_safety_conditions(self.automaton.automaton)
        self.safety_exps = get_compiled_conditions(self.safety_conditions,
                                                   self.automaton.state_var,
                                                   self.automaton.aps,
                                                   margin=margin,
                                                   using_relu_cost=relu_cost)

        # For the number of conditioning inputs to Q functions
        self.unique_safety_conds = []
        self.state_to_unique_safety_cond_idx = {}
        self.n_unique_safety_conds = 0
        for k, cond in self.safety_conditions.items():
            if cond not in self.unique_safety_conds:
                self.state_to_unique_safety_cond_idx[k] = self.n_unique_safety_conds
                self.unique_safety_conds.append(cond)
                self.n_unique_safety_conds += 1
            else:
                unique_safety_cond_idx = self.unique_safety_conds.index(cond)
                self.state_to_unique_safety_cond_idx[k] = unique_safety_cond_idx

        self.state_to_unique_safety_cond_idx_arr = jnp.array([
            self.state_to_unique_safety_cond_idx[q] for q in range(self.automaton.num_states)
        ], dtype=jnp.int32)

        self.relu_cost = relu_cost

    def cost_obs(self, obs):
        return jnp.concatenate((self.goalless_obs(obs),
                                self.automaton_obs(obs)), axis=-1)

    def _compute_cost(self, state: State, action: jax.Array, nstate: State):
        """
        Note: state and action are unused.
        """

        nobs = jnp.expand_dims(self.original_obs(nstate.obs), 0)

        # Unless using the ablation "relu_cost", this should be negative when
        # unsafe, positive when safe.  That way taking the min over the whole
        # trajectory returns the least safe point.
        safety_rho = jax.lax.switch(nstate.info["automata_state"],
                                    self.safety_exps,
                                    { self.state_var.idx: nobs }).squeeze()

        if self.relu_cost:
            # in the ablation, its only positive non-zero when unsafe.
            return jax.nn.relu(safety_rho)
        else:
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
