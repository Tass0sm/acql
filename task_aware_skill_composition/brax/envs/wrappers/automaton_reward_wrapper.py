"""Wrapper for adding specification satisfaction as reward."""

from typing import Callable, Dict, Optional, Tuple

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from flax import struct
import jax
from jax import numpy as jnp

from corallab_stl import Expression


class AutomatonRewardWrapper(Wrapper):
    """Wrapper for an augmented reward signal for an automaton environment"""

    def __init__(
            self,
            env: Env,
            multiplier: float,
    ):
        super().__init__(env)
        self.multiplier = multiplier

    # def _compute_reward(self, state: State, action: jax.Array, nstate: State):
    #     "Temporarily hardcoded identification of good edge"
    #     aps = self.automaton.quantitative_eval_aps(nstate.obs)
    #     return nstate.reward - self.multiplier*jax.nn.relu(aps[0])

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        nstate = self.env.step(state, action)

        obs = nstate.obs

        healthy_reward = nstate.metrics["reward_survive"]
        ctrl_cost = -nstate.metrics["reward_ctrl"]
        contact_cost = -nstate.metrics["reward_contact"]

        # Compute reward according to automaton state.
        def get_subgoal_dist(automaton_state):
            subgoal = jax.lax.cond(automaton_state == 1,
                                   lambda: self.get_obs_goal(obs, 1),
                                   lambda: self.get_obs_goal(obs, 0))
            dist = jnp.linalg.norm(obs[:2] - subgoal)
            return dist

        dist = get_subgoal_dist(nstate.info["automata_state"])
        new_reward = -dist + healthy_reward - ctrl_cost - contact_cost

        nstate = nstate.replace(
            reward=new_reward
        )

        return nstate
