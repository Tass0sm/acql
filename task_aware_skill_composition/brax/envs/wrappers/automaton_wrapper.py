"""Wrapper for adding specification automata to an environment."""

from typing import Callable, Dict, Optional, Tuple
from operator import itemgetter

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from flax import struct, nnx
import jax
from jax import numpy as jnp

import spot
import buddy

from corallab_stl import Expression
from corallab_stl.automata import get_spot_formula_and_aps, eval_spot_formula, make_just_liveness_automaton, get_outgoing_conditions


class JaxAutomaton:

    def __init__(self, exp, state_var):
        formula, aps = get_spot_formula_and_aps(exp)

        self.aps = aps
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
            augment_obs: bool = True,
            augment_with_subgoals: bool = True,
    ):
        super().__init__(env)

        self.automaton = JaxAutomaton(specification, state_var)
        self.augment_obs = augment_obs
        self.augment_with_subgoals = augment_with_subgoals

    def original_obs(self, obs):
        if self.augment_with_subgoals and self.augment_obs:
            n_subgoals = 1
            subgoal_dim = 2
            return obs[..., :-(n_subgoals * subgoal_dim + self.automaton.n_states)]
        elif self.augment_obs:
            return obs[..., :-self.automaton.n_states]
        else:
            return obs

    def split_obs(self, obs):
        if self.augment_with_subgoals and self.augment_obs:
            n_subgoals = 1
            subgoal_dim = 2
            return (
                obs[..., :-(n_subgoals * subgoal_dim + self.automaton.n_states)],
                obs[..., -(n_subgoals * subgoal_dim + self.automaton.n_states):],
            )
        elif self.augment_obs:
            return (
                obs[..., :-self.automaton.n_states],
                obs[..., -self.automaton.n_states:]
            )
        else:
            return (
                obs,
                None,
            )

    def get_obs_goal(self, obs, i):
        # if i > 0 and not self.augment_with_subgoals:
        #     raise NotImplementedError
        goal_start_idx = 4 + 2*i
        return obs[..., goal_start_idx:goal_start_idx+2]

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)

        automata_state = jnp.uint32(self.automaton.init_state)
        # automata_state = jnp.uint32(0)
        state.info["automata_state"] = automata_state
        state.info["made_transition"] = jnp.uint32(0)

        if self.augment_with_subgoals and self.augment_obs:
            # new_obs = jnp.concatenate((state.obs,
            #                            jnp.array([12.0, 4.0]),
            #                            self.automaton.embed(automata_state)), axis=-1)
            new_obs = jnp.concatenate((state.obs,
                                       jnp.array([12.0, 4.0]),
                                       self.automaton.one_hot_encode(automata_state)), axis=-1)
            state = state.replace(obs=new_obs)
        elif self.augment_obs:
            # new_obs = jnp.concatenate((state.obs, self.automaton.embed(automata_state)), axis=-1)
            new_obs = jnp.concatenate((state.obs, self.automaton.one_hot_encode(automata_state)), axis=-1)
            state = state.replace(obs=new_obs)

        return state

    def step(self, state: State, action: jax.Array) -> State:

        state = state.replace(obs=self.original_obs(state.obs))

        nstate = self.env.step(state, action)

        labels = self.automaton.eval_aps(action, nstate.obs)

        automata_state = state.info["automata_state"]

        nautomata_state = self.automaton.step(automata_state, labels)

        made_transition = jnp.where(automata_state != nautomata_state, jnp.uint32(1), jnp.uint32(0))

        state.info.update(
            automata_state=nautomata_state,
            made_transition=made_transition
        )

        if self.augment_with_subgoals and self.augment_obs:
            # new_obs = jnp.concatenate((nstate.obs,
            #                            jnp.array([12.0, 4.0]),
            #                            self.automaton.embed(nautomata_state)), axis=-1)
            new_obs = jnp.concatenate((nstate.obs,
                                       jnp.array([12.0, 4.0]),
                                       self.automaton.one_hot_encode(nautomata_state)), axis=-1)
            nstate = nstate.replace(obs=new_obs)
        elif self.augment_obs:
            # new_obs = jnp.concatenate((nstate.obs, self.automaton.embed(nautomata_state)), axis=-1)
            new_obs = jnp.concatenate((nstate.obs, self.automaton.one_hot_encode(nautomata_state)), axis=-1)
            nstate = nstate.replace(obs=new_obs)

        return nstate

    @property
    def observation_size(self) -> int:
        if self.augment_with_subgoals and self.augment_obs:
            n_subgoals = 1
            subgoal_dim = 2
            return self.env.observation_size + (n_subgoals * subgoal_dim) + self.automaton.n_states
        elif self.augment_obs:
            return self.env.observation_size + self.automaton.n_states
        else:
            return self.env.observation_size


def get_automaton_goal_array(shape, out_conditions, bdd_dict, aps, goal_dim):
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
            goal_array = goal_array.at[k, i*goal_dim:(i+1)*goal_dim].set(goal)

    return goal_array


class AutomatonGoalConditionedWrapper(Wrapper):
    """Tracks automaton state and adds goal condition inputs as necessary"""

    def __init__(
            self,
            env: Env,
            specification: Expression,
            state_var,
    ):
        super().__init__(env)

        self.automaton = JaxAutomaton(specification, state_var)
        self.augment_obs = True
        self.augment_with_subgoals = True

        self.liveness_automaton = make_just_liveness_automaton(self.automaton.automaton)
        self.out_conditions = get_outgoing_conditions(self.liveness_automaton)

        # Scan the liveness automaton to identify the maximum number of goals
        # needed at any stage in the trajectory.
        supports = [buddy.bdd_support(x) for x in self.out_conditions.values()]
        max_i, max_e = max(enumerate([buddy.bdd_nodecount(s) for s in supports]), key=itemgetter(1))

        # There is always an implicit final goal. If there are no subgoals, set
        # the goal_width and goal_idx dict accordingly.
        if max_e == 0:
            self.goal_width = 1
            self.ap_to_goal_idx_dict = {}
        else:
            self.goal_width = max_e
    
            goal_idx = 0
            max_e_formula = spot.bdd_to_formula(supports[max_i], self.automaton.bdd_dict)
            self.ap_to_goal_idx_dict = {}
    
            # A little hacky
            def register_ap(x):
                nonlocal goal_idx
                if x.kind() == spot.op_ap:
                    ap_id = int(x.ap_name()[3:])
                    self.ap_to_goal_idx_dict[ap_id] = goal_idx
                    goal_idx += 1
    
            max_e_formula.traverse(register_ap)

        self.goal_dim = self.env.goal_indices.size
        self.all_goals_dim = self.goal_width * self.goal_dim

        self.automaton_goal_array = get_automaton_goal_array((self.automaton.n_states, self.all_goals_dim),
                                                             self.out_conditions,
                                                             self.automaton.bdd_dict,
                                                             self.automaton.aps,
                                                             self.goal_dim)
        self.original_obs_dim = env.observation_size
        self.no_goal_obs_dim = self.original_obs_dim - self.goal_dim

    def goalless_obs(self, obs):
        return obs[..., :self.no_goal_obs_dim]

    def original_obs(self, obs):
        return obs[..., :self.original_obs_dim]

    def default_goal(self, obs):
        original_goal = obs[..., self.env.goal_indices]
        return jnp.zeros((self.all_goals_dim,)).at[:self.goal_dim].set(original_goal)

    def ith_goal(self, obs, i):
        all_goals = obs[..., self.no_goal_obs_dim:self.no_goal_obs_dim+self.all_goals_dim]
        return all_goals[..., i*self.goal_dim:(i+1)*self.goal_dim]
    
    def current_goals(self, aut_state, default_goal):
        # assumes that state zero is the accepting state
        return jax.lax.cond(aut_state != 0,
                            lambda: self.automaton_goal_array.at[aut_state].get(),
                            lambda: default_goal)

    def modify_goals_for_aut_state(self, obs, aut_state):
        default_goal = self.default_goal(obs)
        return jnp.concatenate((self.goalless_obs(obs),
                                self.current_goals(aut_state, default_goal),
                                self.automaton.one_hot_encode(aut_state)), axis=-1)

    def aut_state_from_obs(self, obs):
        return self.automaton.one_hot_decode(obs[..., -self.automaton.n_states:])
    
    # def get_automaton_qs(self, hdq_networks, params, observation):
    #     breakpoint()
    #     qs = hdq_networks.option_q_network.apply(*params, observation)
    #     min_q = jnp.min(qs, axis=-1)
    #     return min_q

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)

        automaton_state = jnp.uint32(self.automaton.init_state)
        state.info["automata_state"] = automaton_state
        state.info["made_transition"] = jnp.uint32(0)

        new_obs = self.modify_goals_for_aut_state(state.obs, automaton_state)
        state = state.replace(obs=new_obs)

        return state

    def step(self, state: State, action: jax.Array) -> State:

        state = state.replace(obs=self.original_obs(state.obs))

        nstate = self.env.step(state, action)

        labels = self.automaton.eval_aps(action, nstate.obs)

        automaton_state = state.info["automata_state"]

        nautomaton_state = self.automaton.step(automaton_state, labels)

        made_transition = jnp.where(automaton_state != nautomaton_state, jnp.uint32(1), jnp.uint32(0))

        state.info.update(
            automata_state=nautomaton_state,
            made_transition=made_transition
        )

        new_obs = self.modify_goals_for_aut_state(nstate.obs, automaton_state)
        nstate = nstate.replace(obs=new_obs)

        return nstate

    @property
    def observation_size(self) -> int:
        return self.original_obs_dim

    @property
    def goalless_observation_size(self) -> int:
        return self.no_goal_obs_dim

    @property
    def full_observation_size(self) -> int:
        return (self.observation_size - self.goal_dim) + self.all_goals_dim + self.automaton.n_states
