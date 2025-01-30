"""Hindsight Experience Replay - Replay Buffer

See: https://arxiv.org/abs/1707.01495
"""

import functools
import time
from typing import Any, Callable, Optional, Tuple, Union, Generic, NamedTuple, Sequence

from absl import logging
from brax import base
from brax import envs
from brax.io import model
from brax.training import gradients
from brax.training import pmap
from brax.training.replay_buffers_test import jit_wrap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.sac import losses as sac_losses
from brax.training.agents.sac import networks as sac_networks
from brax.training.acme.types import NestedArray
from brax.training.types import Params, Policy
from brax.training.types import Transition
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp

from achql.brax.her.replay_buffers.automaton_her import AutomatonTrajectoryUniformSamplingQueue


class SimpleAutomatonTrajectoryUniformSamplingQueue(AutomatonTrajectoryUniformSamplingQueue):
    """Implements an uniform sampling limited-size replay queue but with trajectories."""

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["use_her", "env"])
    def flatten_crl_fn(use_her, env, transition: Transition, sample_key: PRNGKey) -> Transition:
        # transition is a Transition object containing one env's trajectory. And
        # this function is vmapped over an env-shuffled batch of trajectories.

        if use_her:
            obs = transition.observation
            automaton_state = transition.extras["state_extras"]["automata_state"]
            next_obs = transition.next_observation

            # create a mask where mask[i,j] == 1 if j > i
            seq_len = transition.observation.shape[0]
            arrangement = jnp.arange(seq_len)
            is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)

            # repeat the seeds sequence in the transition object for every
            # offset into the trajectory possible.
            single_trajectories = jnp.concatenate(
                [transition.extras["state_extras"]["seed"][:, jnp.newaxis].T] * seq_len, axis=0
            )

            # create the final step mask, where mask[i, j] == 1 if traj[j] is
            # the final step in the trajectory starting at traj[i].

            # first step: combine the is_future mask with a test for whether
            # step j is part of the same rollout as step i. Also add a term to
            # final_step_mask.shape == (seq_len, seq_len)
            final_step_mask = is_future_mask * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5

            # second step: filter to steps where either:
            # 1. the episode is truncated
            # 2. a transition in automaton takes place
            # truncation_or_transition_mask = jnp.logical_or(
            #     transition.extras["state_extras"]["truncation"],
            #     transition.extras["state_extras"]["made_transition"]
            # )
            truncation_mask = transition.extras["state_extras"]["truncation"]
            final_step_mask = jnp.logical_and(final_step_mask, truncation_mask[None, :])

            # for every row, get the index of the terminal state
            # non_zero_columns = jnp.nonzero(final_step_mask, size=seq_len)[1]

            def get_first_row_nonzero(row):
                return jnp.nonzero(row, size=1, fill_value=0)[0]

            non_zero_columns = jax.vmap(get_first_row_nonzero)(final_step_mask).squeeze()

            # If final state is not present use original goal (i.e. don't change anything)
            new_goals_idx = jnp.where(non_zero_columns == 0, arrangement, non_zero_columns)
            binary_mask = jnp.logical_and(non_zero_columns, non_zero_columns)

            # by default, use all the original goals.  if no transition or
            # truncation is present, these will stay unmodified in the sampled
            # trajectory.

            # if truncation is present, change the subgoal/final goal to the
            # state at termination, depending on the automaton states in the
            # trajectory.

            # for now don't introduce any artificial transitions...

            #### TODO: randomly generate index to select subgoal for sections of the trajectory
            # random_goal_idx_key = sample_key
            # random_goal_idx = 0 jax.random.randint(new_aut_state_key, (), 0, env.goal_width, jnp.int32)
            #####

            final_position = obs[new_goals_idx][:, env.pos_indices]
            final_automaton_state = automaton_state[new_goals_idx]

            def update_goal(o, aut_state, final_pos, final_aut_state):
                o = jax.lax.cond(aut_state == final_aut_state,
                                 lambda: o.at[env.goal_indices].set(final_pos),
                                 lambda: o)
                return o

            def conditional_update_goal(should_update, o, aut_state, final_pos, final_aut_state):
                return jax.lax.cond(should_update,
                                    lambda: update_goal(o, aut_state, final_pos, final_aut_state),
                                    lambda: o)

            new_obs = jax.vmap(conditional_update_goal)(
                binary_mask,
                obs,
                automaton_state,
                final_position,
                final_automaton_state,
            )

            # Recalculate reward
            def get_new_reward(o, aut_state):
                # return jnp.float32(jnp.linalg.norm(o.at[env.goal_indices].get() - o.at[env.pos_indices].get()) < env.goal_dist)
                return jax.lax.cond(aut_state == 0,
                                    lambda: jnp.float32(jnp.linalg.norm(o.at[env.goal_indices].get() - o.at[env.pos_indices].get()) < env.goal_dist),
                                    lambda: jnp.float32(jnp.linalg.norm(o.at[env.goal_indices].get() - o.at[env.pos_indices].get()) < 5*env.goal_dist))

            new_reward = jax.vmap(get_new_reward)(new_obs, automaton_state)

            # Transform next observation
            new_next_obs = jax.vmap(conditional_update_goal)(
                binary_mask,
                next_obs,
                automaton_state,
                final_position,
                final_automaton_state,
            )

            return transition._replace(
                observation=jnp.squeeze(new_obs),
                next_observation=jnp.squeeze(new_next_obs),
                reward=jnp.squeeze(new_reward),
            )

        return transition
