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

from acql.stl import Expression
from acql.automaton import JaxAutomaton
from acql.stl import get_spot_formula_and_aps, eval_spot_formula, make_just_liveness_automaton, get_outgoing_conditions, get_func_from_spot_formula
from acql.brax.envs.wrappers.automaton_wrapper import AutomatonWrapper


def partition(pred, d):
    fs = {}
    ts = {}
    for q, item in d.items():
        if not pred(q, item):
            fs[q] = (item)
        else:
            ts[q] = (item)
    return fs, ts


class AutomatonMultiGoalConditionedWrapper(Wrapper):
    """Wraps a product MDP that is not goal conditioned and doesn't augment the
    goal, and instead augments the obs with all of the goals in the task and an
    encoding of the automaton state (one-hot).
    """

    def __init__(
            self,
            env: Env,
            use_incoming_conditions_for_final_state: bool = True,
            randomize_goals: bool = True,
            randomize_starts: bool = True,
            # reward_type="positive_sparse",
    ):
        super().__init__(env)

        assert isinstance(env, AutomatonWrapper), "Env must be a product MDP"
        assert env.strip_goal_obs == True, "Env cannot already have goal conditioning"
        assert env.augment_obs == False, "Env cannot already be augmenting the observation"

        # To determine the necessary goal inputs, the atomic propositions used
        # in the task specification are filtered based on weather or not a goal
        # has been associated with them. Non goal APs are collected into a
        # single BDD so that they can be filtered out from liveness conditions.
        no_goal_aps, goal_aps = partition(lambda _, v: "goal" in v.info, self.automaton.aps)
        no_goal_spot_aps, goal_spot_aps = partition(lambda k, _: k in goal_aps, self.automaton.spot_aps)
        no_goal_spot_aps_bdds = [spot.formula_to_bdd(x, self.automaton.bdd_dict, None) for x in no_goal_spot_aps.values()]
        no_goal_spot_aps_union_bdd = reduce(buddy.bdd_or, no_goal_spot_aps_bdds, buddy.bddfalse)
        self.n_goal_aps = len(goal_aps)
        self.goal_ap_indices = jnp.array([k for k in goal_aps])
        self.max_param_dim = max([ap.info["goal"].size for ap in goal_aps.values()])

        # collect a dict mapping ap id to its position in a vector of just goal
        self.ap_id_to_goal_idx_dict = {}
        for i, goal_ap in enumerate(goal_aps):
            self.ap_id_to_goal_idx_dict[goal_ap] = i

        # We construct an automaton encoding the liveness constraint within the
        # input task specification. This uses a modified version of the Alpern
        # and Schneider algorithm, where instead of undefined transitions moving
        # to an accepting trap state, they are made into self-loops.
        self.liveness_automaton = make_just_liveness_automaton(self.automaton.automaton)
        plain_out_conditions = get_outgoing_conditions(self.liveness_automaton)

        # By default, we interpret the final accepting state as a persistence
        # constraint. Instead of assuming that the episode terminates once
        # reaching the final state, we attempt to continue satisfying the
        # condition that brought us to the state for the remainder of the
        # episode. We use the same interpretation for our baselines.
        self.use_incoming_conditions_for_final_state = use_incoming_conditions_for_final_state
        if use_incoming_conditions_for_final_state:
            incoming_conds = []

            for edge in self.liveness_automaton.edges():
                if edge.src != self.accepting_state and edge.dst == self.accepting_state:
                    incoming_conds.append(edge.cond)

            incoming_cond_bdd = reduce(buddy.bdd_or, incoming_conds, buddy.bddfalse)
            plain_out_conditions[self.accepting_state] = incoming_cond_bdd
            
            incoming_cond_formula = spot.bdd_to_formula(incoming_cond_bdd, self.automaton.bdd_dict)
            self._incoming_cond_func = get_func_from_spot_formula(incoming_cond_formula, len(self.automaton.aps))
        else:
            self._incoming_cond_func = None

        # At this point, we filter out the non-goal APs from the liveness
        # constraints for each step in the trajectory. For instance, the
        # liveness constraint "a0 & !a1", where a1 is not associated with a
        # subgoal, would be transformed into "a0".
        self.out_conditions = {}
        self.out_condition_formulas = {}
        self.out_condition_functions = {}
        for q, cond in plain_out_conditions.items():
            self.out_conditions[q] = buddy.bdd_exist(cond, buddy.bdd_support(no_goal_spot_aps_union_bdd))
            self.out_condition_formulas[q] = spot.bdd_to_formula(self.out_conditions[q], self.automaton.bdd_dict)
            self.out_condition_functions[q] = get_func_from_spot_formula(self.out_condition_formulas[q], len(self.automaton.aps))

        self.live_states = jnp.array([k for k in self.out_conditions])

        # In this implementation, the observation is a fixed size
        # array. Therefore, we must scan the liveness constraints to determine
        # the maximum number of goal inputs needed at any step.
        supports = [buddy.bdd_support(x) for x in self.out_conditions.values()]
        max_i, max_e = max(enumerate([buddy.bdd_nodecount(s) for s in supports]), key=itemgetter(1))

        if max_e < 1:
            warnings.warn("Warning: No goals in specification.")

        self.goal_width = max_e
        self.goal_dim = self.env.goal_indices.size

        self.randomize_goals = randomize_goals
        self._randomize_starts = randomize_starts

        if randomize_starts:
            self.possible_starts = jnp.concatenate((self.possible_starts, self.possible_goals), axis=0)
        
        automaton_ap_params = jnp.zeros((self.automaton.n_aps, self.max_param_dim))
        automaton_goal_ap_params = jnp.zeros((self.n_goal_aps, self.max_param_dim))
        for k, ap in self.automaton.aps.items():
            if k in goal_aps:
                dim = ap.info["goal"].size
                automaton_ap_params = automaton_ap_params.at[k, :dim].set(ap.info["goal"])

                g_k = self.ap_id_to_goal_idx_dict[k]
                automaton_goal_ap_params = automaton_goal_ap_params.at[g_k, :dim].set(ap.info["goal"])
            elif ap.default_params is not None:
                dim = ap.default_params.size
                automaton_ap_params = automaton_ap_params.at[k, :dim].set(ap.default_params)

        self.automaton_ap_params = automaton_ap_params
        self.automaton_goal_ap_params = automaton_goal_ap_params
        self.all_goals_dim = automaton_goal_ap_params.flatten().size
        self.original_obs_dim = env.observation_size
        self.no_goal_obs_dim = self.state_dim

    def _get_random_ap_params(self, rng: jax.Array):
        ap_params = jnp.zeros((self.n_goal_aps, self.max_param_dim))
        goal_ap_params = jnp.zeros((self.n_goal_aps, self.max_param_dim))

        for k, ap in self.automaton.aps.items():
            if "goal" in ap.info:
                ap_goal_key, rng = jax.random.split(rng)
                goal = self._random_target(ap_goal_key)
                ap_params = ap_params.at[k].set(goal)

                g_k = self.ap_id_to_goal_idx_dict[k]
                goal_ap_params = goal_ap_params.at[g_k].set(goal)
            elif ap.default_params is not None:
                ap_params = ap_params.at[k].set(ap.default_params)

        return ap_params, goal_ap_params

    # def original_obs(self, obs):
    #     return obs[..., :self.original_obs_dim]

    def _original_obs(self, obs):
        return obs[..., :self.original_obs_dim]

    def split_obs(self, obs):
        state_obs = obs[..., :self.no_goal_obs_dim]
        goals = obs[..., self.no_goal_obs_dim:self.no_goal_obs_dim+self.all_goals_dim]
        aut_state = self.automaton.one_hot_decode(obs[..., self.no_goal_obs_dim+self.all_goals_dim:])
        return jnp.atleast_2d(state_obs), jnp.atleast_2d(goals), jnp.atleast_1d(aut_state)

    def split_aut_obs(self, obs):
        state_and_goals_obs = obs[..., :self.no_goal_obs_dim+self.all_goals_dim]
        aut_state = obs[..., self.no_goal_obs_dim+self.all_goals_dim:]
        return jnp.atleast_2d(state_and_goals_obs), jnp.atleast_2d(aut_state)

    # def automaton_obs(self, obs):
    #     return obs[..., -self.automaton.n_states:]

    # def no_automaton_obs(self, obs):
    #     return obs[..., :-self.automaton.n_states]

    # def _default_goal(self, obs):
    #     original_goal = obs[..., self.env.goal_indices]
    #     return jnp.zeros((self.all_goals_dim,)).at[:self.goal_dim].set(original_goal)

    def ith_goal(self, obs, i):
        all_goals = obs[..., self.no_goal_obs_dim:self.no_goal_obs_dim+self.all_goals_dim]
        return all_goals[..., i*self.goal_dim:(i+1)*self.goal_dim]

    # def _goalless_obs(self, obs):
    #     return obs[..., :self.no_goal_obs_dim]

    def current_goal(self, obs, i):
        all_goals = obs[..., self.no_goal_obs_dim:self.no_goal_obs_dim+self.all_goals_dim]
        return all_goals[..., i*self.goal_dim:(i+1)*self.goal_dim]

    def _modify_goals_for_aut_state(self, obs, aut_state, goal_ap_params):
        return jnp.concatenate((obs,
                                goal_ap_params.flatten(),
                                self.automaton.one_hot_encode(aut_state)), axis=-1)

    # def aut_state_from_obs(self, obs):
    #     return self.automaton.one_hot_decode(obs[..., -self.automaton.n_states:])

    # def get_automaton_qs(self, hdq_networks, params, observation):
    #     breakpoint()
    #     qs = hdq_networks.option_q_network.apply(*params, observation)
    #     min_q = jnp.min(qs, axis=-1)
    #     return min_q

    def reset(self, rng: jax.Array) -> State:
        reset_key, random_goal_key = jax.random.split(rng)

        state = self.env.reset(reset_key)

        if self.randomize_goals:
            ap_params, goal_ap_params = self._get_random_ap_params(random_goal_key)
            state.info["ap_params"] = ap_params
            state.info["goal_ap_params"] = goal_ap_params
        else:
            state.info["ap_params"] = self.automaton_ap_params
            state.info["goal_ap_params"] = self.automaton_goal_ap_params

        automaton_state = state.info["automata_state"]
        new_obs = self._modify_goals_for_aut_state(state.obs, automaton_state, state.info["goal_ap_params"])
        state = state.replace(obs=new_obs)

        # In this environment, these keys are sometimes meaningless because
        # there may be multiple goals, or no goals, or changing goals.
        if "dist" in state.metrics:
            del state.metrics["dist"]
        if "success" in state.metrics:
            del state.metrics["success"]
        if "success_easy" in state.metrics:
            del state.metrics["success_easy"]
        if "success_hard" in state.metrics:
            del state.metrics["success_hard"]

        automaton_success = jnp.array(automaton_state == self.accepting_state, dtype=float)
        state.metrics.update(
            automaton_success=automaton_success,
        )

        return state

    def step(self, state: State, action: jax.Array) -> State:
        state = state.replace(obs=self._original_obs(state.obs))

        nstate = self.env.step(state, action)

        # In this environment, these keys are sometimes meaningless because
        # there may be multiple goals, or no goals, or changing goals.
        if "dist" in nstate.metrics:
            del nstate.metrics["dist"]
        if "success" in state.metrics:
            del nstate.metrics["success"]
        if "success_easy" in state.metrics:
            del nstate.metrics["success_easy"]
        if "success_hard" in state.metrics:
            del nstate.metrics["success_hard"]

        nautomaton_state = nstate.info["automata_state"]

        # For now the only reward comes from completing the task.

        if self.use_incoming_conditions_for_final_state:
            automaton_success = jnp.array(nautomaton_state == self.accepting_state, dtype=float)
            labels = self.automaton.eval_aps(action, nstate.obs, ap_params=nstate.info["ap_params"])
            final_cond_satisfied = jnp.array(self._incoming_cond_func(labels) > 0, dtype=float)
            reward = automaton_success * final_cond_satisfied
        else:
            automaton_success = jnp.array(nautomaton_state == self.accepting_state, dtype=float)
            reward = automaton_success
        
        nstate.metrics.update(
            automaton_success=automaton_success,
        )

        new_obs = self._modify_goals_for_aut_state(nstate.obs, nautomaton_state, state.info["goal_ap_params"])
        nstate = nstate.replace(obs=new_obs, reward=reward)

        return nstate

    # def reached_goal(self, obs: jax.Array, threshold: float = 2.0, ap_params = None):
    #     labels = self.automaton.eval_aps(None, obs, ap_params=ap_params)
    #     pos_non_aut = obs[..., self.pos_indices]
    #     goal_non_aut = obs[..., self.goal_indices]
    #     self.out_condition_functions[q]
    #     reached = jnp.array(non_aut_dist < threshold, dtype=float)
    #     return reached

    @property
    def observation_size(self) -> int:
        return self.no_goal_obs_dim + self.automaton_goal_ap_params.flatten().size + (self.automaton.n_states if self.automaton.n_states > 1 else 0)

    @property
    def qr_nn_input_size(self) -> int:
        return self.no_goal_obs_dim + self.goal_dim * self.goal_width
        # return self.no_goal_obs_dim + self.automaton_ap_params.flatten().size

    @property
    def qc_nn_input_size(self) -> int:
        return self.no_goal_obs_dim
