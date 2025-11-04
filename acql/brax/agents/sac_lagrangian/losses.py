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

"""SAC-Lagrangian losses.

See: https://arxiv.org/pdf/1812.05905.pdf
"""
from typing import Any

from brax.training import types
from brax.training.types import Params
from brax.training.types import PRNGKey
import jax
import jax.numpy as jnp

from achql.brax.agents.sac_lagrangian import networks as sac_lagrangian_networks


Transition = types.Transition


def make_losses(
    sac_lagrangian_network: sac_lagrangian_networks.SACLagrangianNetworks,
    reward_scaling: float,
    cost_scaling: float,
    cost_budget: float,
    discounting: float,
    action_size: int
):
  """Creates the SAC losses."""

  target_entropy = -0.5 * action_size
  policy_network = sac_lagrangian_network.policy_network
  q_network = sac_lagrangian_network.q_network
  cost_q_network = sac_lagrangian_network.cost_q_network
  parametric_action_distribution = sac_lagrangian_network.parametric_action_distribution

  def alpha_loss(log_alpha: jnp.ndarray, policy_params: Params,
                 normalizer_params: Any, transitions: Transition,
                 key: PRNGKey) -> jnp.ndarray:
    """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
    dist_params = policy_network.apply(normalizer_params, policy_params,
                                       transitions.observation)
    action = parametric_action_distribution.sample_no_postprocessing(
        dist_params, key)
    log_prob = parametric_action_distribution.log_prob(dist_params, action)
    alpha = jnp.exp(log_alpha)
    alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
    return jnp.mean(alpha_loss)

  def lambda_loss(
      lambda_multiplier: jnp.ndarray,
      cost_q_params: Params,
      normalizer_params: Any,
      transitions: Transition,
      key: PRNGKey) -> jnp.ndarray:
    """Eq 7 from https://arxiv.org/pdf/2002.08550"""

    # First version based on: https://github.com/ammarhydr/SAC-Lagrangian/blob/232a0772205007068449740e0a0f1b7ddca3d946/SAC_Agent.py#L278
    # log_lambda = jax.nn.softplus(lambda_multiplier)
    # used cost critic

    # Second version based on omnisafe: https://github.com/PKU-Alignment/omnisafe/blob/main/omnisafe/algorithms/off_policy/sac_lag.py#L97
    # uses mean cost from batch

    violation = jnp.mean(transitions.extras["state_extras"]["cost"] - cost_budget)
    lambda_loss = -lambda_multiplier * jax.lax.stop_gradient(violation)

    return jnp.sum(lambda_loss), {
      # "cost_q": jnp.mean(min_cost_q),
      "mean_violation": violation,
    }

  def critic_loss(q_params: Params, policy_params: Params,
                  normalizer_params: Any, target_q_params: Params,
                  alpha: jnp.ndarray, transitions: Transition,
                  key: PRNGKey) -> jnp.ndarray:
    q_old_action = q_network.apply(normalizer_params, q_params,
                                   transitions.observation, transitions.action)
    next_dist_params = policy_network.apply(normalizer_params, policy_params,
                                            transitions.next_observation)
    next_action = parametric_action_distribution.sample_no_postprocessing(
        next_dist_params, key)
    next_log_prob = parametric_action_distribution.log_prob(
        next_dist_params, next_action)
    next_action = parametric_action_distribution.postprocess(next_action)
    next_q = q_network.apply(normalizer_params, target_q_params,
                             transitions.next_observation, next_action)
    next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob
    target_q = jax.lax.stop_gradient(transitions.reward * reward_scaling +
                                     transitions.discount * discounting *
                                     next_v)
    q_error = q_old_action - jnp.expand_dims(target_q, -1)

    # Better bootstrapping for truncated episodes.
    truncation = transitions.extras['state_extras']['truncation']
    q_error *= jnp.expand_dims(1 - truncation, -1)

    q_loss = 0.5 * jnp.mean(jnp.square(q_error))
    return q_loss

  def cost_critic_loss(cost_q_params: Params, policy_params: Params,
                       normalizer_params: Any, target_cost_q_params: Params,
                       alpha: jnp.ndarray, transitions: Transition,
                       key: PRNGKey) -> jnp.ndarray:
    cost_q_old_action = cost_q_network.apply(normalizer_params, cost_q_params,
                                        transitions.observation, transitions.action)
    next_dist_params = policy_network.apply(normalizer_params, policy_params,
                                            transitions.next_observation)
    next_action = parametric_action_distribution.sample_no_postprocessing(
      next_dist_params, key)
    next_log_prob = parametric_action_distribution.log_prob(
      next_dist_params, next_action)
    next_action = parametric_action_distribution.postprocess(next_action)
    next_cost_q = cost_q_network.apply(normalizer_params, target_cost_q_params,
                                       transitions.next_observation, next_action)
    next_cost_v = jnp.min(next_cost_q, axis=-1) # - alpha * next_log_prob
    target_cost_q = jax.lax.stop_gradient(transitions.extras["state_extras"]["cost"] * cost_scaling +
                                          transitions.discount * discounting *
                                          next_cost_v)
    cost_q_error = cost_q_old_action - jnp.expand_dims(target_cost_q, -1)

    # Better bootstrapping for truncated episodes.
    truncation = transitions.extras['state_extras']['truncation']
    cost_q_error *= jnp.expand_dims(1 - truncation, -1)

    cost_q_loss = 0.5 * jnp.mean(jnp.square(cost_q_error))

    return cost_q_loss, {
      # "min_cost_q_old_action": jnp.min(cost_q_old_action),
      # "mean_cost_q_old_action": jnp.mean(cost_q_old_action),
      # "max_cost_q_old_action": jnp.max(cost_q_old_action),
      # "min_target_cost_q": jnp.min(target_cost_q),
      # "mean_target_cost_q": jnp.mean(target_cost_q),
      # "max_target_cost_q": jnp.max(target_cost_q),
      # "mean_cost_over_batch": jnp.mean(transitions.extras["state_extras"]["cost"])
    }

  def actor_loss(policy_params: Params,
                 normalizer_params: Any,
                 q_params: Params,
                 cost_q_params: Params,
                 alpha: jnp.ndarray,
                 lambda_multiplier: jnp.ndarray,
                 transitions: Transition,
                 key: PRNGKey) -> jnp.ndarray:
    dist_params = policy_network.apply(normalizer_params, policy_params,
                                       transitions.observation)
    action = parametric_action_distribution.sample_no_postprocessing(
        dist_params, key)
    log_prob = parametric_action_distribution.log_prob(dist_params, action)
    action = parametric_action_distribution.postprocess(action)
    q_action = q_network.apply(normalizer_params, q_params,
                               transitions.observation, action)
    min_q = jnp.min(q_action, axis=-1)

    cost_q_action = cost_q_network.apply(normalizer_params, cost_q_params,
                                         transitions.observation, action)
    min_cost_q = jnp.min(cost_q_action, axis=-1)

    # maximize entropy (minimize alpha*log_prob)
    # maximize return (minimize -Q)
    # minimize cost (minimize Qc)
    actor_loss = alpha * log_prob - min_q + (lambda_multiplier * min_cost_q)
    return jnp.mean(actor_loss)

  return alpha_loss, lambda_loss, critic_loss, cost_critic_loss, actor_loss
