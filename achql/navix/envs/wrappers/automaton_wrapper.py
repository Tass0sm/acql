# import warnings
# from functools import reduce
# from typing import Callable, Dict, Optional, Tuple
# from operator import itemgetter

from brax.envs.base import Env, State, Wrapper
import jax
from jax import numpy as jnp
from navix.environments import Timestep
from navix.rendering.cache import RenderingCache

import spot
import buddy

from achql.stl import Expression
from achql.automaton import JaxAutomaton
from achql.stl.automata import get_spot_formula_and_aps, eval_spot_formula, make_just_liveness_automaton, get_outgoing_conditions


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
    """Tracks automaton state during interaction with environment"""

    def __init__(
            self,
            env: Env,
            specification: Expression,
            state_var,
            augment_obs: bool = True,
    ):
        super().__init__(env)

        self.automaton = JaxAutomaton(specification, state_var)
        self.augment_obs = augment_obs
        self.original_obs_dim = env.observation_space.shape[0]

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
        self.non_goal_indices = ~mask

        if self.augment_obs:
            mask = jnp.zeros(self.original_obs_dim + self.automaton.n_states, dtype=bool)
            self.full_non_goal_indices = ~mask
        else:
            self.full_non_goal_indices = self.non_goal_indices

        if self.augment_obs:
            normalization_mask = jnp.zeros(self.original_obs_dim + self.automaton.n_states, dtype=bool)
            normalization_mask = normalization_mask.at[:-self.automaton.n_states].set(True)
        else:
            normalization_mask = jnp.ones(self.original_obs_dim, dtype=bool)

        self.normalization_mask = normalization_mask

    def original_obs(self, obs):
        # takes an obs from this mdp, and returns an obs from the original
        # underlying mdp

        if self.augment_obs:
            obs = obs[..., :-self.automaton.n_states]

        return obs

    def no_goal_obs(self, obs):
        return obs[..., self.full_non_goal_indices]

    def no_automaton_obs(self, obs):
        return obs[..., :-self.automaton.n_states]

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


    def reset(self, key: jax.Array, cache: RenderingCache | None = None) -> Timestep:
        timestep = self.env.reset(key, cache=cache)

        automaton_state = jnp.uint32(self.automaton.init_state)
        timestep.info["automata_state"] = automaton_state
        timestep.info["made_transition"] = jnp.uint32(0)
        timestep.info["ap_params"] = self.automaton_ap_params

        # In this environment, these keys are meaningless because the
        # environment is often not goal-conditioned.
        # if "dist" in state.metrics:
        #     del state.metrics["dist"]
        # if "success" in state.metrics:
        #     del state.metrics["success"]
        # if "success_easy" in state.metrics:
        #     del state.metrics["success_easy"]

        # automaton_success = jnp.array(automaton_state == self.accepting_state, dtype=float)
        # state.metrics.update(
        #     automaton_success=automaton_success,
        # )

        if self.augment_obs:
            # new_obs = jnp.concatenate((state.obs, self.automaton.embed(automaton_state)), axis=-1)
            new_obs = jnp.concatenate((timestep.observation, self.automaton.one_hot_encode(automaton_state)), axis=-1)
            timestep = timestep.replace(observation=new_obs)

        return timestep

    def step(self, timestep: Timestep, action: jax.Array) -> Timestep:
        # timestep = timestep.replace(obs=self.original_obs(state.obs))

        ntimestep = self.env.step(timestep, action)

        # if "dist" in state.metrics:
        #     del state.metrics["dist"]
        # if "success" in state.metrics:
        #     del state.metrics["success"]
        # if "success_easy" in state.metrics:
        #     del state.metrics["success_easy"]

        automaton_state = timestep.info["automata_state"]
        ap_params = timestep.info["ap_params"]

        labels = self.automaton.eval_aps(action, ntimestep, ap_params=ap_params)

        nautomaton_state = self.automaton.step(automaton_state, labels)

        made_transition = jnp.where(automaton_state != nautomaton_state, jnp.uint32(1), jnp.uint32(0))

        ntimestep.info.update(
            automata_state=nautomaton_state,
            made_transition=made_transition,
            ap_params=ap_params,
        )

        if self.augment_obs:
            # new_obs = jnp.concatenate((nstate.obs, self.automaton.embed(nautomaton_state)), axis=-1)
            new_obs = jnp.concatenate((ntimestep.observation, self.automaton.one_hot_encode(nautomaton_state)), axis=-1)
            ntimestep = ntimestep.replace(observation=new_obs)

        reward = jnp.array(nautomaton_state == self.accepting_state, dtype=float)
        # automaton_success = reward
        # ntimestep.metrics.update(
        #     automaton_success=automaton_success,
        # )

        ntimestep = ntimestep.replace(reward=reward)

        return ntimestep

    def make_augmented_obs(self, obs, aut_state):
        return jnp.concatenate((obs, self.automaton.one_hot_encode(aut_state)), axis=-1)

    @property
    def observation_size(self) -> int:
        if self.augment_obs:
            return self.env.observation_size + self.automaton.n_states
        else:
            return self.env.observation_size


# def get_automaton_goal_array(shape, out_conditions, bdd_dict, aps, state_and_ap_to_goal_idx_dict, goal_dim):
#     goal_array = jnp.zeros(shape)

#     # A little hacky
#     def register_ap(x, goal_dict_k):
#         if x.kind() == spot.op_ap:
#             ap_id = int(x.ap_name()[3:])
#             goal_dict_k[ap_id] = aps[ap_id].info["goal"]

#     for k, cond_bdd in out_conditions.items():
#         goal_dict_k = { }
#         f = spot.bdd_to_formula(cond_bdd, bdd_dict)
#         f.traverse(register_ap, goal_dict_k)

#         for i, goal in goal_dict_k.items():
#             goal_idx = state_and_ap_to_goal_idx_dict[(k, i)]
#             goal_array = goal_array.at[k, goal_idx*goal_dim:(goal_idx+1)*goal_dim].set(goal)

#     return goal_array


# def partition(pred, d):
#     fs = {}
#     ts = {}
#     for q, item in d.items():
#         if not pred(q, item):
#             fs[q] = (item)
#         else:
#             ts[q] = (item)
#     return fs, ts
