"""Wrapper for turning automaton MDP into reward machine MDP."""

from typing import Callable, Dict, Optional, Tuple

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from flax import struct
import jax
from jax import numpy as jnp

from acql.stl import Expression


def reward_machine_value_iteration(num_states, terminal_states, delta_u, delta_r, n_aps, gamma):
    """
    num_states = | U and F |
    """

    U = [i for i in range(num_states) if i not in terminal_states]

    # initialize the value function
    V = jnp.zeros((num_states,))

    # Q function for state u and symbol sigma
    def get_Q(u, sigma):
        # next aut state
        v = delta_u[u, sigma].item()
        # get the reward for the transition. only works for simple reward
        # machines that ignore args.
        r = delta_r[u](v, None, None, None).item()
        # target is the reward for the transition + discounted value at new state
        return r + gamma * V[v]

    e = 1
    while e > 0:
        e = 0
        for u in U:
            new_V = max([get_Q(u, sigma) for sigma in range(2**n_aps)])
            e = max(e, abs(V[u] - new_V))
            V = V.at[u].set(new_V)

    return V

# 1: Input: U , F , P, δu, δr, γ
# 2: for u ∈ U ∪ F do
# 3: v(u) ← 0 {initializing v-values}
# 4: e ← 1
# 5: while e > 0 do
# 6: e ← 0
# 7: for u ∈ U do
# 8: v′ ← max{δr(u, σ) + γv(δu(u, σ)) | ∀σ ∈ 2P }
# 9: e = max{e, |v(u) − v′|}
# 10: v(u) ← v′
# 11: return v

class AutomatonRewardMachineWrapper(Wrapper):
    """Wrapper for an augmented reward signal for an automaton environment"""

    def __init__(
            self,
            env: Env,
            rm_config,
            reward_shaping: bool = False
    ):
        super().__init__(env)

        self.rm_final_state = rm_config["final_state"]
        self.rm_term_states = jnp.array(rm_config["terminal_states"], dtype=jnp.int32)
        self.rm_edge_config = rm_config["reward_functions"]
        self.rm_pruned_edges = rm_config["pruned_edges"]

        dummy_func = lambda s_t, a_t, s_t1: 0.0

        self.u_fs = []
        for u in range(self.automaton.n_states):
            v_fs = []
            for v in range(self.automaton.n_states):
                f_uv = self.rm_edge_config.get((u, v), dummy_func)
                v_fs.append(f_uv)

            f_u = lambda v, s_t, a_t, s_t1: jax.lax.switch(v, v_fs, s_t, a_t, s_t1)
            self.u_fs.append(f_u)

        self.use_reward_shaping = reward_shaping
        if reward_shaping:
            self.reward_shaping_gamma = 0.9
            V = reward_machine_value_iteration(self.automaton.num_states,
                                               jnp.append(self.rm_term_states, self.rm_final_state),
                                               self.automaton.delta_u,
                                               self.u_fs,
                                               self.automaton.n_aps,
                                               self.reward_shaping_gamma)

            self.potential_function = lambda u: -V[u]
        else:
            self.potential_function = lambda u: 0

    def original_obs(self, obs):
        return obs[..., :-self.automaton.n_states]

    def strip_obs_goal(obs):
        return obs[..., self.strip_goal_indices]
        
    def compute_reward(self, u, v, state, action, nstate):
        if self.use_reward_shaping:
            return jax.lax.switch(u, self.u_fs, v, state, action, nstate) + self.reward_shaping_gamma * self.potential_function(v) - self.potential_function(u)
        else:
            return jax.lax.switch(u, self.u_fs, v, state, action, nstate)

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        u = state.info["automata_state"]

        nstate = self.env.step(state, action)
        v = nstate.info["automata_state"]

        # recompute reward via rm
        new_reward = self.compute_reward(u, v, state, action, nstate)
        nstate = nstate.replace(reward=new_reward)
        
        # if v is in termination states, terminate the episode
        one = jnp.ones_like(nstate.done)
        done = jnp.where(jnp.isin(v, self.rm_term_states), one, nstate.done)
        nstate = nstate.replace(done=done)

        return nstate

    def split_aut_obs(self, obs):
        state_obs = obs[..., :-self.automaton.num_states]
        aut_state = obs[..., -self.automaton.num_states:]
        return state_obs, aut_state
