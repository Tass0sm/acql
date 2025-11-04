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

"""Deep Deterministic Policy Gradient losses.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

from typing import Any

from brax.training import types
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp

from acql.brax.agents.ddpg import networks as ddpg_networks

Transition = types.Transition


def make_losses(
    ddpg_network: ddpg_networks.DDPGNetworks,
    reward_scaling: float,
    discounting: float,
):
  """Creates the DDPG losses."""

  policy_network = ddpg_network.policy_network
  q_network = ddpg_network.q_network
  parametric_action_distribution = ddpg_network.parametric_action_distribution

  def critic_loss(
      q_params: Params,
      policy_params: Params,
      normalizer_params: Any,
      target_q_params: Params,
      transitions: Transition,
      key: PRNGKey,
  ) -> jnp.ndarray:
    q_old_action = q_network.apply(
        normalizer_params, q_params, transitions.observation, transitions.action
    )
    next_dist_params = policy_network.apply(
        normalizer_params, policy_params, transitions.next_observation
    )
    next_action = parametric_action_distribution.sample_no_postprocessing(
        next_dist_params, key
    )
    next_log_prob = parametric_action_distribution.log_prob(
        next_dist_params, next_action
    )
    next_action = parametric_action_distribution.postprocess(next_action)
    next_q = q_network.apply(
        normalizer_params,
        target_q_params,
        transitions.next_observation,
        next_action,
    )
    next_v = jnp.min(next_q, axis=-1)
    target_q = jax.lax.stop_gradient(
        transitions.reward * reward_scaling
        + transitions.discount * discounting * next_v
    )
    q_error = q_old_action - jnp.expand_dims(target_q, -1)

    # Better bootstrapping for truncated episodes.
    truncation = transitions.extras['state_extras']['truncation']
    q_error *= jnp.expand_dims(1 - truncation, -1)

    q_loss = 0.5 * jnp.mean(jnp.square(q_error))
    return q_loss

  def actor_loss(
      policy_params: Params,
      normalizer_params: Any,
      q_params: Params,
      transitions: Transition,
      key: PRNGKey,
  ) -> jnp.ndarray:
    dist_params = policy_network.apply(
        normalizer_params, policy_params, transitions.observation
    )
    action = parametric_action_distribution.sample_no_postprocessing(
        dist_params, key
    )
    log_prob = parametric_action_distribution.log_prob(dist_params, action)
    action = parametric_action_distribution.postprocess(action)
    q_action = q_network.apply(
        normalizer_params, q_params, transitions.observation, action
    )
    min_q = jnp.min(q_action, axis=-1)
    actor_loss = -min_q
    return jnp.mean(actor_loss)

  return critic_loss, actor_loss
