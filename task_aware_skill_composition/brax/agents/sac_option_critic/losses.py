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

"""Soft Option Actor-Critic losses.

See: https://arxiv.org/pdf/1812.05905.pdf
"""
from typing import Any

from brax.training import types
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp

from task_aware_skill_composition.brax.agents.sac_option_critic import networks as soac_networks


Transition = types.Transition


def make_losses(
    soac_network: soac_networks.SOACNetworks,
    reward_scaling: float,
    discounting: float,
    action_size: int,
    num_options: int,
):
  """Creates the SOAC losses."""

  target_entropy_ratio = 0.98
  # target_entropy = -0.5 * num_options
  target_entropy = -jnp.log(1.0 / num_options) * target_entropy_ratio

  hi_policy_network = soac_network.hi_policy_network
  option_q_network = soac_network.option_q_network
  parametric_option_distribution = soac_network.parametric_option_distribution

  def alpha_loss(
      log_alpha: jnp.ndarray,
      policy_params: Params,
      normalizer_params: Any,
      transitions: Transition,
      key: PRNGKey
  ) -> jnp.ndarray:
    """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
    dist_params = hi_policy_network.apply(normalizer_params, policy_params,
                                          transitions.observation)

    option_dist = parametric_option_distribution.create_dist(dist_params)

    alpha = jnp.exp(log_alpha)

    # equation 11 in https://arxiv.org/pdf/1910.07207
    option_probs = option_dist.probs
    option_log_probs = option_dist.logits
    temp_obj = jnp.sum(option_probs * (-option_log_probs + target_entropy), axis=-1)
    alpha_loss = alpha * jax.lax.stop_gradient(temp_obj)

    return jnp.mean(alpha_loss)

  def critic_loss(
      q_params: Params,
      policy_params: Params,
      normalizer_params: Any,
      target_q_params: Params,
      alpha: jnp.ndarray,
      transitions: Transition,
      key: PRNGKey
  ) -> jnp.ndarray:

    # E a_t~pi [ Q(s_t, a_t) ]
    qs_old = option_q_network.apply(normalizer_params, q_params, transitions.observation)
    q_old_action = jax.vmap(lambda x, i: x.at[i].get())(qs_old, transitions.option)

    # log pi(o_t+1 | s_t+1) for all options
    next_option_dist_params = hi_policy_network.apply(normalizer_params, policy_params, transitions.next_observation)
    next_option_dist = parametric_option_distribution.create_dist(next_option_dist_params)
    next_option_probs = next_option_dist.probs
    next_option_log_probs = next_option_dist.logits

    # Q1(s_t+1, o_t+1)/Q2(s_t+1, o_t+1) for all options
    next_qs = option_q_network.apply(normalizer_params, target_q_params, transitions.next_observation)

    # Q(s_t+1, o_t+1) - alpha * log pi(o_t+1 | s_t+1) for all options
    next_v_os = jnp.min(next_qs, axis=-1) - alpha * next_option_log_probs

    # V(s_t+1) = E a_t+1 ~ pi [ Q(s_t+1, a_t+1) ]
    next_v = jnp.sum(next_option_probs * next_v_os, axis=-1)

    # E (s_t, a_t, s_t+1) ~ D [r(s_t, a_t) + gamma * V(s_t+1)]
    target_q = jax.lax.stop_gradient(transitions.reward * reward_scaling +
                                     transitions.discount * discounting * next_v)

    # Q(s_t, a_t) - E[r(s_t, a_t) + gamma * V(s_t+1)]
    q_error = q_old_action - jnp.expand_dims(target_q, -1)

    # Better bootstrapping for truncated episodes.
    truncation = transitions.extras['state_extras']['truncation']
    q_error *= jnp.expand_dims(1 - truncation, -1)

    q_loss = 0.5 * jnp.mean(jnp.square(q_error))
    return q_loss

  def actor_loss(policy_params: Params, normalizer_params: Any,
                 q_params: Params, alpha: jnp.ndarray, transitions: Transition,
                 key: PRNGKey) -> jnp.ndarray:

    # p(o_t | s_t) for all options
    option_dist_params = hi_policy_network.apply(normalizer_params, policy_params, transitions.observation)
    option_dist = parametric_option_distribution.create_dist(option_dist_params)

    option_probs = option_dist.probs

    # log p(o_t | s_t) for all options
    option_log_probs = option_dist.logits

    # Q(s_t, o_t) for all options
    qs_action = option_q_network.apply(normalizer_params, q_params, transitions.observation)
    min_qs = qs_action[..., 0] # jnp.min(qs_action, axis=-1)

    # alpha * log_prob(o_t | s_t) - Q(s_t, o_t) for all options
    actor_loss_os = alpha * option_log_probs - jax.lax.stop_gradient(min_qs)

    # alpha * log_prob(o_t | s_t) - Q(s_t, o_t) over D
    actor_loss = jnp.sum(option_probs * actor_loss_os, axis=-1)

    # estimation of E s_t~D [ alpha * log_prob(o_t | s_t) - Q(s_t, o_t) ]
    return jnp.mean(actor_loss)

  return alpha_loss, critic_loss, actor_loss
