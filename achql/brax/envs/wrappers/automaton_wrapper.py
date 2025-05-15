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

from achql.stl import Expression
from achql.automaton import JaxAutomaton
from achql.stl import get_spot_formula_and_aps, eval_spot_formula, make_just_liveness_automaton, get_outgoing_conditions, get_func_from_spot_formula


def partition(pred, d):
    fs = {}
    ts = {}
    for q, item in d.items():
        if not pred(q, item):
            fs[q] = (item)
        else:
            ts[q] = (item)
    return fs, ts


class AutomatonWrapper(Wrapper):
    """Make product mdp with automaton from an input specification

    This class can also make a goal conditioned mdp a non-goal-conditioned MDP
    if necessary.

    This wrapper does not change the reward at all.
    """

    def __init__(
            self,
            env: Env,
            specification: Expression,
            state_var,
            augment_obs: bool = True,
            strip_goal_obs: bool = False,
            # treat_aut_state_as_goal: bool = False,
            overwrite_reward: bool = False,
            use_incoming_conditions_for_final_state: bool = False,
    ):
        super().__init__(env)

        self.automaton = JaxAutomaton(specification, state_var)
        self.augment_obs = augment_obs
        self.original_obs_dim = env.observation_size
        self.strip_goal_obs = strip_goal_obs

        # get accepting state
        accepting_states = [self.automaton.automaton.state_is_accepting(i) for i in range(self.automaton.automaton.num_states())]
        assert sum(accepting_states) == 1, "Can't handle more than one accepting state"
        self.accepting_state = accepting_states.index(True)

        # Get Params
        no_goal_aps, goal_aps = partition(lambda _, v: "goal" in v.info, self.automaton.aps)
        self.max_param_dim = max([ap.info["goal"].size for ap in goal_aps.values()], default=0)

        automaton_ap_params = jnp.zeros((self.automaton.n_aps, self.max_param_dim))
        for k, ap in goal_aps.items():
            dim = ap.info["goal"].size
            automaton_ap_params = automaton_ap_params.at[k, :dim].set(ap.info["goal"])
        self.automaton_ap_params = automaton_ap_params

        # Set up masks for obs manipulation...
        mask = jnp.zeros(self.original_obs_dim, dtype=bool)
        mask = mask.at[env.goal_indices].set(True)
        self.non_goal_indices = ~mask

        if self.augment_obs and not self.strip_goal_obs:
            obs_dim = self.original_obs_dim + (self.automaton.n_states if self.automaton.n_states > 1 else 0)
            mask = jnp.zeros(obs_dim, dtype=bool)
            mask = mask.at[env.goal_indices].set(True)
            self.full_non_goal_indices = ~mask
        elif self.augment_obs:
            obs_dim = self.original_obs_dim - env.goal_indices.size + (self.automaton.n_states if self.automaton.n_states > 1 else 0)
            mask = jnp.zeros(obs_dim, dtype=bool)
            mask = mask.at[env.goal_indices].set(True)
            self.full_non_goal_indices = ~mask
        elif self.strip_goal_obs:
            obs_dim = self.original_obs_dim - env.goal_indices.size
            mask = jnp.zeros(obs_dim, dtype=bool)
            mask = mask.at[env.goal_indices].set(True)
            self.full_non_goal_indices = ~mask
        else:
            obs_dim = self.original_obs_dim
            self.full_non_goal_indices = self.non_goal_indices

        # make the normalization mask, where positions where mask == 1 are not normalized
        if self.augment_obs and not self.strip_goal_obs:
            normalization_mask = jnp.zeros(obs_dim, dtype=bool)
            if self.automaton.n_states > 1:
                normalization_mask = normalization_mask.at[:-self.automaton.n_states].set(True)
        elif self.augment_obs:
            normalization_mask = jnp.zeros(obs_dim, dtype=bool)
            if self.automaton.n_states > 1:
                normalization_mask = normalization_mask.at[:-self.automaton.n_states].set(True)
        elif self.strip_goal_obs:
            normalization_mask = jnp.zeros(obs_dim, dtype=bool)
        else:
            normalization_mask = jnp.zeros(obs_dim, dtype=bool)

        # if treat_aut_state_as_goal and self.automaton.n_states > 1:
        #     aut_indices = jnp.arange(obs_dim - self.automaton.n_states, obs_dim)
        #     self.goal_indices = jnp.concatenate((env.goal_indices, aut_indices))
        #     self.full_non_goal_indices = self.full_non_goal_indices.at[aut_indices].set(False)

        # By default, we interpret the final accepting state as a persistence
        # constraint. Instead of assuming that the episode terminates once
        # reaching the final state, we attempt to continue satisfying the
        # condition that brought us to the state for the remainder of the
        # episode. We use the same interpretation for our baselines.

        self.liveness_automaton = make_just_liveness_automaton(self.automaton.automaton)
        plain_out_conditions = get_outgoing_conditions(self.liveness_automaton)

        self._use_incoming_conditions_for_final_state = use_incoming_conditions_for_final_state
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

        self._overwrite_reward = overwrite_reward

        self.normalization_mask = normalization_mask

    def _original_obs(self, obs):
        # takes an obs from this mdp, and returns an obs from the original
        # underlying mdp

        if self.augment_obs and self.automaton.n_states > 1:
            obs = obs[..., :-self.automaton.n_states]

        if self.strip_goal_obs:
            dummy_goal = jnp.zeros_like(self.goal_indices)
            obs = jnp.insert(obs, self.goal_indices, dummy_goal, axis=-1)

        return obs

    # def no_goal_obs(self, obs):
    #     return obs[..., self.full_non_goal_indices]

    # def no_automaton_obs(self, obs):
    #     return obs[..., :-self.automaton.n_states]

    def automaton_obs(self, obs):
        return obs[..., -self.automaton.n_states:]

    # def original_obs(self, obs):
    #     if self.augment_with_subgoals and self.augment_obs:
    #         n_subgoals = 1
    #         subgoal_dim = 2
    #         return obs[..., :-(n_subgoals * subgoal_dim + self.automaton.n_states)]
    #     elif self.augment_obs:
    #         return obs[..., :-self.automaton.n_states]
    #     else:
    #         return obs

    # def split_obs(self, obs):
    #     if self.augment_with_subgoals and self.augment_obs:
    #         n_subgoals = 1
    #         subgoal_dim = 2
    #         return (
    #             obs[..., :-(n_subgoals * subgoal_dim + self.automaton.n_states)],
    #             obs[..., -(n_subgoals * subgoal_dim + self.automaton.n_states):],
    #         )
    #     elif self.augment_obs:
    #         return (
    #             obs[..., :-self.automaton.n_states],
    #             obs[..., -self.automaton.n_states:]
    #         )
    #     else:
    #         return (
    #             obs,
    #             None,
    #         )

    # def get_obs_goal(self, obs, i):
    #     # if i > 0 and not self.augment_with_subgoals:
    #     #     raise NotImplementedError
    #     goal_start_idx = 4 + 2*i
    #     return obs[..., goal_start_idx:goal_start_idx+2]

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)

        automaton_state = jnp.uint32(self.automaton.init_state)
        # automaton_state = jnp.uint32(0)
        state.info["automata_state"] = automaton_state
        state.info["made_transition"] = jnp.uint32(0)
        state.info["ap_params"] = self.automaton_ap_params

        # In this environment, these keys are meaningless because the
        # environment is often not goal-conditioned.
        # del state.metrics["dist"]
        # del state.metrics["success"]
        # del state.metrics["success_easy"]

        automaton_success = jnp.array(automaton_state == self.accepting_state, dtype=float)
        state.metrics.update(
            automaton_success=automaton_success,
        )

        if self.strip_goal_obs:
            new_obs = state.obs[..., self.non_goal_indices]
            state = state.replace(obs=new_obs)

        if self.augment_obs:
            # new_obs = jnp.concatenate((state.obs, self.automaton.embed(automaton_state)), axis=-1)
            new_obs = jnp.concatenate((state.obs, self.automaton.one_hot_encode(automaton_state)), axis=-1)
            state = state.replace(obs=new_obs)

        return state

    def step(self, state: State, action: jax.Array) -> State:

        state = state.replace(obs=self._original_obs(state.obs))
        automaton_state = state.info["automata_state"]

        nstate = self.env.step(state, action)

        # del state.metrics["dist"]
        # del state.metrics["success"]
        # del state.metrics["success_easy"]

        labels = self.automaton.eval_aps(action, nstate.obs, ap_params=state.info["ap_params"])
        nautomaton_state = self.automaton.step(automaton_state, labels)

        made_transition = jnp.where(automaton_state != nautomaton_state, jnp.uint32(1), jnp.uint32(0))

        nstate.info.update(
            automata_state=nautomaton_state,
            made_transition=made_transition
        )

        if self.strip_goal_obs:
            new_obs = nstate.obs[..., self.non_goal_indices]
            nstate = nstate.replace(obs=new_obs)

        if self.augment_obs:
            # new_obs = jnp.concatenate((nstate.obs, self.automaton.embed(nautomaton_state)), axis=-1)
            new_obs = jnp.concatenate((nstate.obs, self.automaton.one_hot_encode(nautomaton_state)), axis=-1)
            nstate = nstate.replace(obs=new_obs)

        # reward = jnp.array(nautomaton_state == self.accepting_state, dtype=float)
        automaton_success = jnp.array(nautomaton_state == self.accepting_state, dtype=float)
        nstate.metrics.update(
            automaton_success=automaton_success,
        )

        if self._overwrite_reward and self._use_incoming_conditions_for_final_state:
            final_cond_satisfied = jnp.array(self._incoming_cond_func(labels) > 0, dtype=float)
            reward = automaton_success * final_cond_satisfied
            nstate = nstate.replace(reward=reward)
        elif self._overwrite_reward:
            nstate = nstate.replace(reward=automaton_success)

        return nstate

    def make_augmented_obs(self, obs, aut_state):
        return jnp.concatenate((obs, self.automaton.one_hot_encode(aut_state)), axis=-1)

    @property
    def observation_size(self) -> int:
        if self.augment_obs and not self.strip_goal_obs:
            return self.env.observation_size + (self.automaton.n_states if self.automaton.n_states > 1 else 0)
        elif self.augment_obs:
            return self.env.observation_size - self.env.goal_indices.size + (self.automaton.n_states if self.automaton.n_states > 1 else 0)
        elif self.strip_goal_obs:
            return self.env.observation_size - self.env.goal_indices.size
        else:
            return self.env.observation_size


def get_automaton_goal_array(shape, out_conditions, bdd_dict, aps, state_and_ap_to_goal_idx_dict, goal_dim):
    goal_array = jnp.zeros(shape)

    # A little hacky
    def register_ap(x, goal_dict_k):
        if x.kind() == spot.op_ap:
            ap_id = int(x.ap_name()[3:])
            goal_dict_k[ap_id] = aps[ap_id].info["goal"]

    for k, cond_bdd in out_conditions.items():
        goal_dict_k = { }
        f = spot.bdd_to_formula(cond_bdd, bdd_dict)
        f.traverse(register_ap, goal_dict_k)

        for i, goal in goal_dict_k.items():
            goal_idx = state_and_ap_to_goal_idx_dict[(k, i)]
            goal_array = goal_array.at[k, goal_idx*goal_dim:(goal_idx+1)*goal_dim].set(goal)

    return goal_array


class AutomatonGoalConditionedWrapper(Wrapper):
    """Tracks automaton state and adds goal condition inputs as necessary

    First it takes an goal-conditioned base MDP and removes the goal
    from the state space, making it non goal conditioned.

    If one provides a specification like "F(g1)" and variable_goals is true,
    then the wrapped MDP essentially becomes the same MDP as before. If not, the
    goal is not randomly sampled and it is always the default goal provided by
    the specification.

    In either case, this class obtains the DBA from the specification and uses
    it to keep track of progress made with respect to the specification.

    The reward of this environment is based on full satisfaction of the
    specification. There are a few options:

    - positive_sparse: 0 everywhere except 1 in automaton accepting states (and
      when condition associated with accepting state if
      "treat_accepting_state_like_goal" is true)

    - time_sparse: -1 everywhere except 0 in automaton accepting states (and
      when condition associated with accepting state if
      "treat_accepting_state_like_goal" is true)

    """

    def __init__(
            self,
            env: Env,
            specification: Expression,
            state_var,
            use_incoming_conditions_for_final_state: bool = True,
            randomize_goals: bool = True,
            treat_accepting_state_like_goal: bool = False,
            reward_type="positive_sparse",
    ):
        super().__init__(env)

        # First, construct the deterministic buchi automaton object based on the
        # input specification. In this environment, the automaton state is
        # always added to the observation. Also, every goal input deemed
        # necessary is added to the observation in this environment.
        self.automaton = JaxAutomaton(specification, state_var)
        self.augment_obs = True
        self.augment_with_subgoals = True

        # get accepting state
        accepting_states = [self.automaton.automaton.state_is_accepting(i) for i in range(self.automaton.automaton.num_states())]
        assert sum(accepting_states) == 1, "Can't handle more than one accepting state"
        self.accepting_state = accepting_states.index(True)

        # To determine the necessary goal inputs, the atomic propositions used
        # in the task specification are filtered based on weather or not a goal
        # has been associated with them. Non goal APs are collected into a
        # single BDD so that they can be filtered out from liveness conditions.
        no_goal_aps, goal_aps = partition(lambda _, v: "goal" in v.info, self.automaton.aps)
        no_goal_spot_aps, goal_spot_aps = partition(lambda k, _: k in goal_aps, self.automaton.spot_aps)
        no_goal_spot_aps_bdds = [spot.formula_to_bdd(x, self.automaton.bdd_dict, None) for x in no_goal_spot_aps.values()]
        no_goal_spot_aps_union_bdd = reduce(buddy.bdd_or, no_goal_spot_aps_bdds, buddy.bddfalse)
        self.goal_ap_indices = jnp.array([k for k in goal_aps])
        self.max_param_dim = max([ap.info["goal"].size for ap in goal_aps.values()])
        
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

        # At this point, we filter out the non-goal APs from the liveness
        # constraints for each step in the trajectory. For instance, the
        # liveness constraint "a0 & !a1", where a1 is not associated with a
        # subgoal, would be transformed into "a0".
        self.out_conditions = {}
        for q, cond in plain_out_conditions.items():
            self.out_conditions[q] = buddy.bdd_exist(cond, buddy.bdd_support(no_goal_spot_aps_union_bdd))

        self.live_states = jnp.array([k for k in self.out_conditions])

        # In this implementation, the observation is a fixed size
        # array. Therefore, we must scan the liveness constraints to determine
        # the maximum number of goal inputs needed at any step.
        supports = [buddy.bdd_support(x) for x in self.out_conditions.values()]
        max_i, max_e = max(enumerate([buddy.bdd_nodecount(s) for s in supports]), key=itemgetter(1))

        # There is always an implicit final goal. If there are no subgoals, set
        # the goal_width and goal_idx dict accordingly.
        self.state_and_ap_to_goal_idx_dict = {}
        self.state_and_goal_idx_to_ap_dict = {}

        # if max_e == 0:
        #     self.goal_width = 1
        # else:

        if max_e < 1:
            warnings.warn("Warning: No goals in specification.")

        self.goal_width = max_e

        for k, cond in self.out_conditions.items():
            goal_idx = 0
            cond_formula = spot.bdd_to_formula(cond, self.automaton.bdd_dict)

            # A little hacky
            def register_ap(x):
                nonlocal goal_idx
                if x.kind() == spot.op_ap:
                    ap_id = int(x.ap_name()[3:])
                    self.state_and_ap_to_goal_idx_dict[(k, ap_id)] = goal_idx
                    self.state_and_goal_idx_to_ap_dict[(k, goal_idx)] = ap_id
                    goal_idx += 1

            cond_formula.traverse(register_ap)

        self.goal_dim = self.env.goal_indices.size
        self.all_goals_dim = self.goal_width * self.goal_dim

        self.randomize_goals = randomize_goals

        automaton_ap_params = jnp.zeros((self.automaton.n_aps, self.max_param_dim))
        for k, ap in self.automaton.aps.items():
            if k in goal_aps:
                dim = ap.info["goal"].size
                automaton_ap_params = automaton_ap_params.at[k, :dim].set(ap.info["goal"])
            elif ap.default_params is not None:
                dim = ap.default_params.size
                automaton_ap_params = automaton_ap_params.at[k, :dim].set(ap.default_params)

        self.automaton_ap_params = automaton_ap_params
        self.automaton_goal_array = get_automaton_goal_array((self.automaton.n_states, self.all_goals_dim),
                                                             self.out_conditions,
                                                             self.automaton.bdd_dict,
                                                             self.automaton.aps,
                                                             self.state_and_ap_to_goal_idx_dict,
                                                             self.goal_dim)
        self.original_obs_dim = env.observation_size
        self.no_goal_obs_dim = self.original_obs_dim - self.goal_dim

    def _get_random_goal_array(self, rng: jax.Array):
        ap_goals = []
        ap_params = jnp.zeros((self.automaton.n_aps, self.max_param_dim))

        for k, ap in self.automaton.aps.items():
            if "goal" in ap.info:
                ap_goal_key, rng = jax.random.split(rng)
                goal = self._random_target(ap_goal_key)
                ap_params = ap_params.at[k].set(goal)
                ap_goals.append(goal)
            elif ap.default_params is not None:
                ap_params = ap_params.at[k].set(ap.default_params)

        ap_goal_array = jnp.stack(ap_goals)
        goal_array = jnp.zeros((self.automaton.n_states, self.all_goals_dim))

        for (k, gi), ap in self.state_and_goal_idx_to_ap_dict.items():
            goal_array = goal_array.at[k, gi*self.goal_dim:(gi+1)*self.goal_dim].set(ap_goal_array[ap])
        
        return ap_params, goal_array
        
    def goalless_obs(self, obs):
        return obs[..., :self.no_goal_obs_dim]

    def original_obs(self, obs):
        return obs[..., :self.original_obs_dim]

    def split_obs(self, obs):
        state_obs = obs[..., :self.no_goal_obs_dim]
        goals = obs[..., self.no_goal_obs_dim:self.no_goal_obs_dim+self.all_goals_dim]
        aut_state = self.automaton.one_hot_decode(obs[..., self.no_goal_obs_dim+self.all_goals_dim:])
        return jnp.atleast_2d(state_obs), jnp.atleast_2d(goals), jnp.atleast_1d(aut_state)

    def split_aut_obs(self, obs):
        state_and_goals_obs = obs[..., :self.no_goal_obs_dim+self.all_goals_dim]
        aut_state = obs[..., self.no_goal_obs_dim+self.all_goals_dim:]
        return state_and_goals_obs, aut_state

    def automaton_obs(self, obs):
        return obs[..., -self.automaton.n_states:]

    def no_automaton_obs(self, obs):
        return obs[..., :-self.automaton.n_states]
    
    def default_goal(self, obs):
        original_goal = obs[..., self.env.goal_indices]
        return jnp.zeros((self.all_goals_dim,)).at[:self.goal_dim].set(original_goal)

    def ith_goal(self, obs, i):
        all_goals = obs[..., self.no_goal_obs_dim:self.no_goal_obs_dim+self.all_goals_dim]
        return all_goals[..., i*self.goal_dim:(i+1)*self.goal_dim]

    def current_goals(self, aut_state, default_goal, goal_array):
        # assumes that state zero is the accepting state
        # return jax.lax.cond(jnp.logical_and(aut_state != 0, jnp.isin(aut_state, self.live_states)),
        #                     lambda: goal_array.at[aut_state].get(),
        #                     lambda: default_goal)
        return jax.lax.cond(jnp.isin(aut_state, self.live_states),
                            lambda: goal_array.at[aut_state].get(),
                            lambda: default_goal)

    def modify_goals_for_aut_state(self, obs, aut_state, goal_array):
        default_goal = self.default_goal(obs)
        return jnp.concatenate((self.goalless_obs(obs),
                                self.current_goals(aut_state, default_goal, goal_array),
                                self.automaton.one_hot_encode(aut_state)), axis=-1)

    def aut_state_from_obs(self, obs):
        return self.automaton.one_hot_decode(obs[..., -self.automaton.n_states:])
    
    # def get_automaton_qs(self, hdq_networks, params, observation):
    #     breakpoint()
    #     qs = hdq_networks.option_q_network.apply(*params, observation)
    #     min_q = jnp.min(qs, axis=-1)
    #     return min_q

    def reset(self, rng: jax.Array) -> State:
        reset_key, random_goal_key = jax.random.split(rng)

        state = self.env.reset(reset_key)

        if self.randomize_goals:
            ap_params, goal_array = self._get_random_goal_array(random_goal_key)
            state.info["ap_params"] = ap_params
            state.info["goal_array"] = goal_array
        else:
            state.info["ap_params"] = self.automaton_ap_params
            state.info["goal_array"] = self.automaton_goal_array

        automaton_state = jnp.uint32(self.automaton.init_state)
        state.info["automata_state"] = automaton_state
        state.info["made_transition"] = jnp.uint32(0)

        new_obs = self.modify_goals_for_aut_state(state.obs, automaton_state, state.info["goal_array"])
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
        state = state.replace(obs=self.original_obs(state.obs))

        nstate = self.env.step(state, action)

        labels = self.automaton.eval_aps(action, nstate.obs, ap_params=state.info["ap_params"])

        automaton_state = state.info["automata_state"]

        nautomaton_state = self.automaton.step(automaton_state, labels)

        made_transition = jnp.where(automaton_state != nautomaton_state, jnp.uint32(1), jnp.uint32(0))

        nstate.info.update(
            automata_state=nautomaton_state,
            made_transition=made_transition
        )

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

        # For now the only reward comes from completing the task.
        reward = jnp.array(nautomaton_state == self.accepting_state, dtype=float)
        automaton_success = reward
        nstate.metrics.update(
            automaton_success=automaton_success,
        )

        new_obs = self.modify_goals_for_aut_state(nstate.obs, automaton_state, state.info["goal_array"])
        nstate = nstate.replace(obs=new_obs, reward=reward)

        return nstate

    @property
    def observation_size(self) -> int:
        # return self.original_obs_dim
        return self.no_goal_obs_dim + min(self.goal_width, 1) * self.goal_dim

    @property
    def goalless_observation_size(self) -> int:
        return self.no_goal_obs_dim

    @property
    def full_observation_size(self) -> int:
        return (self.observation_size - self.goal_dim) + self.all_goals_dim + (self.automaton.n_states if self.automaton.n_states > 1 else 0)
