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

from achql.hierarchy.training.types import HierarchicalTransition


Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
ReplayBufferState = Any

_PMAP_AXIS_NAME = "i"


class TrajectoryACHQLSamplingQueue(QueueBase[Sample], Generic[Sample]):
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
        super(TrajectoryACHQLSamplingQueue, self).__init__(
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
        if use_her:

            # # remove automaton state
            # if hasattr(env, "automaton") and env.augment_obs:
            #     obs, automaton_obs = env.split_aut_obs(transition.observation)
            #     next_obs, next_automaton_obs = env.split_aut_obs(transition.next_observation)
            # else:
            #     obs = transition.observation
            #     next_obs = transition.next_observation

            obs = transition.observation
            next_obs = transition.next_observation

            # Find truncation indexes if present
            seq_len = transition.observation.shape[0]
            arrangement = jnp.arange(seq_len)
            is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
            single_trajectories = jnp.concatenate(
                [transition.extras["state_extras"]["seed"][:, jnp.newaxis].T] * seq_len, axis=0
            )

            # final_step_mask.shape == (seq_len, seq_len)
            final_step_mask = is_future_mask * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
            final_step_mask = jnp.logical_and(final_step_mask, transition.extras["state_extras"]["truncation"][None, :])
            non_zero_columns = jnp.nonzero(final_step_mask, size=seq_len)[1]

            # If final state is not present use original goal (i.e. don't change anything)
            new_goals_idx = jnp.where(non_zero_columns == 0, arrangement, non_zero_columns)
            binary_mask = jnp.logical_and(non_zero_columns, non_zero_columns)

            # # also don't change anything when not in aut_state 0
            # terminal_aut_state_mask = transition.extras["state_extras"]["automata_state"] == 0
            # binary_mask = jnp.logical_and(binary_mask, terminal_aut_state_mask)

            new_goals = (
                binary_mask[:, None] * obs[new_goals_idx][:, env.pos_indices] # use obs x/y position as new goal
                + jnp.logical_not(binary_mask)[:, None] * obs[new_goals_idx][..., env.goal_indices] # use original goal in obs
            )

            # Transform observation
            new_obs = obs.at[..., env.goal_indices].set(new_goals)

            # Recalculate reward
            new_reward = env.reached_goal(new_obs, threshold = env.goal_dist)

            # Transform next observation
            new_next_obs = next_obs.at[..., env.goal_indices].set(new_goals)

            # # add back automaton state
            # if hasattr(env, "automaton") and env.augment_obs:
            #     new_obs = jnp.concatenate([new_obs, automaton_obs], axis=1)
            #     new_next_obs = jnp.concatenate([new_next_obs, next_automaton_obs], axis=1)

            return transition._replace(
                observation=jnp.squeeze(new_obs),
                next_observation=jnp.squeeze(new_next_obs),
                reward=jnp.squeeze(new_reward),
            )

        return transition
