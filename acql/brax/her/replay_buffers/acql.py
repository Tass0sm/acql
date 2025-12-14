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
from brax.v1 import envs as envs_v1
import flax
import jax
import jax.numpy as jnp
import optax

from .jaxgcrl_replay_buffer import QueueBase, Sample

from acql.hierarchy.training.types import HierarchicalTransition


Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
ReplayBufferState = Any

_PMAP_AXIS_NAME = "i"


class TrajectoryACQLSamplingQueue(QueueBase[Sample], Generic[Sample]):
    """Implements an uniform sampling limited-size replay queue but with trajectories."""

    def __init__(
            self,
            max_replay_size: int,
            dummy_data_sample: Sample,
            sample_batch_size: int,
            num_envs: int,
            episode_length: int,
            **kwargs
    ):
        super(TrajectoryACQLSamplingQueue, self).__init__(
            max_replay_size,
            dummy_data_sample,
            sample_batch_size,
            num_envs,
            episode_length
        )

    def sample_internal(self, buffer_state: ReplayBufferState) -> Tuple[ReplayBufferState, Sample]:
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"Data shape expected by the replay buffer ({self._data_shape}) does "
                f"not match the shape of the buffer state ({buffer_state.data.shape})"
            )
        key, sample_key, shuffle_key = jax.random.split(buffer_state.key, 3)
        # NOTE: this is the number of envs to sample but it can be modified if there is OOM
        shape = self.num_envs

        # Sampling envs idxs
        envs_idxs = jax.random.choice(sample_key, jnp.arange(self.num_envs), shape=(shape,), replace=False)

        @functools.partial(jax.jit, static_argnames=("rows", "cols"))
        def create_matrix(rows, cols, min_val, max_val, rng_key):
            rng_key, subkey = jax.random.split(rng_key)
            start_values = jax.random.randint(subkey, shape=(rows,), minval=min_val, maxval=max_val)
            row_indices = jnp.arange(cols)
            matrix = start_values[:, jnp.newaxis] + row_indices
            return matrix

        @jax.jit
        def create_batch(arr_2d, indices):
            return jnp.take(arr_2d, indices, axis=0, mode="wrap")

        create_batch_vmaped = jax.vmap(create_batch, in_axes=(1, 0))

        # create matrix where each row is a sequence of indices into the buffer
        matrix = create_matrix(
            shape,
            self.episode_length,
            buffer_state.sample_position,
            buffer_state.insert_position - self.episode_length,
            sample_key,
        )

        # create batch by taking from env-shuffled buffer the indices in each row of matrix
        batch = create_batch_vmaped(buffer_state.data[:, envs_idxs, :], matrix)
        transitions = self._unflatten_fn(batch)
        return buffer_state.replace(key=key), transitions

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

            # * create the final step mask, where mask[i, j] == 1 if traj[j] is
            # the final step in the trajectory starting at traj[i].

            # ** first step: combine the is_future mask with a test for whether
            # step j is part of the same rollout as step i.
            final_step_mask = is_future_mask * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5

            # ** second step: filter to steps where the episode is truncated
            truncation_mask = transition.extras["state_extras"]["truncation"]
            final_step_mask = jnp.logical_and(final_step_mask, truncation_mask[None, :])

            # for every row, get the index of the terminal state
            def get_first_row_nonzero(row):
                return jnp.nonzero(row, size=1, fill_value=0)[0]
            non_zero_columns = jax.vmap(get_first_row_nonzero)(final_step_mask).squeeze()

            # If final state is not present use original goal (i.e. don't change anything)
            new_goals_idx = jnp.where(non_zero_columns == 0, arrangement, non_zero_columns)
            has_final_step_mask = jnp.logical_and(non_zero_columns, non_zero_columns)

            # for each step, get the final position and final automaton state
            final_position = obs[new_goals_idx][:, env.pos_indices]

            # also, only update when aut_state == final_aut_state
            final_automaton_state = automaton_state[new_goals_idx]
            should_update_mask = jnp.logical_and(automaton_state == final_automaton_state, has_final_step_mask)

            def update_goal(o, final_pos):
                # for now, update the first goal

                # for i in range(env.n_goal_aps):

                i = 0
                o = o.at[env.no_goal_obs_dim+(i*env.goal_dim):env.no_goal_obs_dim+((i+1)*env.goal_dim)].set(final_pos)

                return o

            def conditional_update_goal(should_update, o, final_pos):
                return jax.lax.cond(should_update,
                                    lambda: update_goal(o, final_pos),
                                    lambda: o)

            # update, where necessary, the goal for the transitions.
            # only update where there is a future step and the automaton state matches the final automaton state
            new_obs = jax.vmap(conditional_update_goal)(should_update_mask,
                                                        obs,
                                                        final_position)

            # Transform next observation
            new_next_obs = jax.vmap(conditional_update_goal)(should_update_mask,
                                                             next_obs,
                                                             final_position)

            # Recalculate reward
            def get_new_reward(o):
                aut_state = env.automaton.one_hot_decode(o[-env.automaton.n_states:])
                return jnp.float32(jnp.linalg.norm(o.at[4:6].get() - o.at[0:2].get()) < env.goal_dist)

            new_reward = jax.vmap(get_new_reward)(new_obs)

            return transition._replace(
                observation=jnp.squeeze(new_obs),
                next_observation=jnp.squeeze(new_next_obs),
                reward=jnp.squeeze(new_reward),
            )

        return transition
