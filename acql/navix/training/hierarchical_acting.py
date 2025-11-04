# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Navix training hierarchical acting functions."""

import time
from typing import Callable, Sequence, Tuple, Union

from brax import envs
from brax.training.types import Metrics
from brax.training.types import Policy
from brax.training.types import PolicyParams
from brax.training.types import PRNGKey
from brax.training.types import Transition
import jax
import jax.numpy as jnp

from acql.hierarchy.training.types import HierarchicalTransition
from acql.hierarchy.types import HierarchicalPolicy
from acql.hierarchy.state import OptionState
from acql.hierarchy.option import Option

State = Union[envs.State]
Env = Union[envs.Env]


def actor_step(
    env: Env,
    env_state: State,
    option_state: OptionState,
    policy: HierarchicalPolicy,
    options: Sequence[Option],
    key: PRNGKey,
    extra_fields: Sequence[str] = ()
) -> Tuple[State, OptionState, HierarchicalTransition]:
  """Collect data."""

  # env_state = s_t
  # option_state = (o_t-1, b_t)

  # calculate o_t from s_t, b_t, o_t-1
  # calculate a_t from s_t, o_t
  actions, policy_extras = policy(env_state.obs, option_state, key)
  option = policy_extras["logical_option"]

  # calculate s_t+1 from s_t, a_t
  nstate = env.step(env_state, actions)
  state_extras = {x: nstate.info[x] for x in extra_fields}

  # calculate b_t+1 from s_t+1, o_t
  def get_beta(s_t, o_t, beta_key):
    return jax.lax.switch(o_t, [o.termination for o in options], s_t, beta_key)

  beta_keys = jax.random.split(key, option.shape[0])
  termination = jax.vmap(get_beta)(nstate.obs, option, beta_keys)

  # noption_state = (o_t, b_t+1)
  noption_state = OptionState(option, termination)

  return nstate, noption_state, HierarchicalTransition(  # pytype: disable=wrong-arg-types  # jax-ndarray
    prev_option=option_state.option,
    termination=option_state.option_beta,
    observation=env_state.obs,
    option=option,
    action=actions,
    reward=nstate.reward,
    discount=1 - nstate.done,
    next_observation=nstate.obs,
    extras={
      'policy_extras': policy_extras,
      'state_extras': state_extras
    })


def generate_unroll(
    env: Env,
    env_state: State,
    option_state: OptionState,
    policy: Policy,
    options: Sequence[Option],
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = ()
) -> Tuple[State, HierarchicalTransition]:
  """Collect trajectories of given unroll_length."""

  @jax.jit
  def f(carry, unused_t):
    state, option_state, current_key = carry
    current_key, next_key = jax.random.split(current_key)
    nstate, next_option_state, transition = actor_step(
        env, state, option_state, policy, options, current_key, extra_fields=extra_fields
    )
    return (nstate, next_option_state, next_key), transition

  (final_state, final_option_state, _), data = jax.lax.scan(
      f, (env_state, option_state, key), (), length=unroll_length)
  return final_state, final_option_state, data


def semimdp_actor_step(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    extra_fields: Sequence[str] = ()
) -> Tuple[State, Transition]:
  """Collect data."""
  step_key, policy_key = jax.random.split(key)
  actions, policy_extras = policy(env_state.observation, policy_key)

  # split step key so that all vectorized envs get their own key
  step_keys = jax.random.split(step_key, env_state.observation.shape[0])
  ntimestep = env.step(env_state, (actions, step_keys))

  state_extras = {x: ntimestep.info[x] for x in extra_fields}

  return ntimestep, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=env_state.observation,
      action=actions,
      reward=ntimestep.reward,
      discount=1 - ntimestep.is_termination(),
      next_observation=ntimestep.observation,
      extras={
          'policy_extras': policy_extras,
          'state_extras': state_extras
      })


def semimdp_generate_unroll(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = ()
) -> Tuple[State, Transition]:
  """Collect trajectories of given unroll_length."""

  @jax.jit
  def f(carry, unused_t):
    state, current_key = carry
    current_key, next_key = jax.random.split(current_key)
    nstate, transition = semimdp_actor_step(
        env, state, policy, current_key, extra_fields=extra_fields)
    return (nstate, next_key), transition

  (final_state, _), data = jax.lax.scan(
      f, (env_state, key), (), length=unroll_length)
  return final_state, data
